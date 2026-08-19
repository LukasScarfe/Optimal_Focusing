"""One-time expensive theory propagation for the l=1 OFVB focusing comparison.

The costly step is the Fresnel propagation of the three pupil masks
(Optimal / PFBZ / FBZ) to an intensity map ``I(tau, rho)``.  It depends *only*
on the dimensionless parameters ``(L, N0)`` and the ``tau`` / ``rho`` grids --
NOT on the aperture radius ``A`` or the ``tau -> z`` scale, which are cheap
display-time factors applied in the notebook.  So changing ``A`` or the axial
mapping never requires re-running this.

Run once from the shell::

    python theory_cache.py            # build the cache if missing / stale
    python theory_cache.py --force    # force a rebuild

or from the notebook::

    from theory_cache import load_or_build
    cache = load_or_build(params=dict(L=1, N0=14))   # builds on first call, loads after

``load_or_build`` returns a dict with ``taus``, ``rho_out``, ``intensity`` (a
``{condition: I(tau, rho)}`` dict), and the scalars ``beta_opt``, ``theta``,
``rho_star``, ``tau_f``.  The on-disk cache is keyed by a signature of the
parameters, so a mismatched cache is transparently rebuilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import jv

# Parameters that define the expensive computation.  Anything NOT here (A, the
# tau->z scale, colour maps, ...) is a cheap display choice and must stay out.
DEFAULTS = dict(L=1, N0=14, NRHO_IN=4000, NRHO_OUT=700, NTAU=320,
                TAU_MAX_FACTOR=2.0, NBETA_GRID=40001)

CONDITIONS = ("Optimal", "PFBZ", "FBZ")


def _find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml at or above {here}")


def default_cache_path(root: Path | None = None) -> Path:
    root = root or _find_repo_root()
    return root / "outputs" / "theory_vs_experiment_l1" / "theory_l1_cache.npz"


def _signature(params: dict) -> str:
    return json.dumps({k: params[k] for k in sorted(params)}, sort_keys=True)


def compute_theory(L, N0, NRHO_IN, NRHO_OUT, NTAU, TAU_MAX_FACTOR, NBETA_GRID,
                   verbose: bool = True) -> dict:
    """Run the full theory pipeline; return dimensionless maps + scalars."""
    TAU_F = 1.0 / (2 * np.pi * N0)

    # --- 1. optimise the Bessel scale beta ---------------------------------
    rho_beta = np.linspace(0.0, 1.0, NBETA_GRID)

    def I0_of_beta(beta):
        J = jv(L, beta * rho_beta)
        return np.trapz(rho_beta * J**2, rho_beta) / (2 * np.pi)

    def R_of_beta(beta):
        J = jv(L, beta * rho_beta)
        phase = np.exp(1j * rho_beta**2 / TAU_F)
        return np.trapz(rho_beta * J**2 * phase, rho_beta) / (2 * np.pi)

    def lambda_plus(beta):
        return (I0_of_beta(beta) + np.abs(R_of_beta(beta))) / (2 * TAU_F**2)

    opt = minimize_scalar(lambda b: -lambda_plus(b), bounds=(1e-4, 20.0),
                          method="bounded", options={"xatol": 1e-6})
    beta_opt = float(opt.x)
    if verbose:
        print(f"  beta_opt = {beta_opt:.9f}  lambda_opt = {lambda_plus(beta_opt):.6f}")

    # --- 2. build the three pupil masks ------------------------------------
    rho_in = np.linspace(0.0, 1.0, NRHO_IN)
    J_beta = jv(L, beta_opt * rho_in)
    I0_beta = np.trapz(rho_in * J_beta**2, rho_in) / (2 * np.pi)
    R_beta = np.trapz(rho_in * J_beta**2 * np.exp(1j * rho_in**2 / TAU_F), rho_in) / (2 * np.pi)
    theta = 0.5 * np.angle(R_beta)
    phase_arg = rho_in**2 / (2 * TAU_F) - theta

    def normalize_pupil(psi_raw):
        norm = np.sqrt(2 * np.pi * np.trapz(rho_in * np.abs(psi_raw)**2, rho_in))
        return psi_raw / norm

    masks = {
        "Optimal": normalize_pupil(J_beta * np.cos(phase_arg)),
        "PFBZ": normalize_pupil((2 * np.sqrt(np.pi))
                                * np.where(J_beta * np.cos(phase_arg) >= 0, 1.0, -1.0)),
        "FBZ": normalize_pupil((J_beta / np.sqrt(I0_beta))
                               * np.where(np.cos(phase_arg) >= 0, 1.0, -1.0)),
    }
    rho_star = beta_opt * TAU_F

    # --- 3. propagate each mask to I(tau, rho) (the expensive part) --------
    TAU_MAX = TAU_MAX_FACTOR * TAU_F
    taus = np.concatenate(([0.0], np.linspace(TAU_F / 250, TAU_MAX, NTAU - 1)))
    rho_out = np.linspace(0.0, 1.0, NRHO_OUT)

    def propagate(psi_radial):
        intensity = np.zeros((len(taus), len(rho_out)), dtype=float)
        intensity[0, :] = np.abs(np.interp(rho_out, rho_in, psi_radial))**2
        rho_prime = rho_in[None, :]
        psi_row = psi_radial[None, :]
        rho_col = rho_out[:, None]
        for j, tau in enumerate(taus[1:], start=1):
            phase_in = np.exp(1j * rho_prime**2 / (2 * tau))
            bessel = jv(L, rho_col * rho_prime / tau)
            integrand = rho_prime * phase_in * bessel * psi_row
            radial_int = np.trapz(integrand, rho_in, axis=1)
            amp = ((-1j)**L) / (1j * tau) * np.exp(1j * rho_out**2 / (2 * tau)) * radial_int
            intensity[j, :] = np.abs(amp)**2
        intensity /= intensity.max()
        return intensity

    intensity = {}
    for name, psi in masks.items():
        if verbose:
            print(f"  propagating {name} ...", flush=True)
        intensity[name] = propagate(psi)

    return dict(
        taus=taus, rho_out=rho_out,
        intensity_Optimal=intensity["Optimal"],
        intensity_PFBZ=intensity["PFBZ"],
        intensity_FBZ=intensity["FBZ"],
        beta_opt=np.float64(beta_opt), theta=np.float64(theta),
        rho_star=np.float64(rho_star), tau_f=np.float64(TAU_F),
    )


def load_or_build(params: dict | None = None, path: Path | str | None = None,
                  force: bool = False, verbose: bool = True) -> dict:
    """Load the cached theory maps, computing + saving them on first use.

    A mismatch between the cached parameter signature and ``params`` triggers a
    rebuild, so bumping N0 / grid sizes stays correct automatically.
    """
    p = {**DEFAULTS, **(params or {})}
    path = Path(path) if path else default_cache_path()
    sig = _signature(p)

    def _unpack(d) -> dict:
        out = {k: d[k] for k in d.files if k != "signature"}
        out["intensity"] = {c: out.pop(f"intensity_{c}") for c in CONDITIONS}
        return out

    if path.is_file() and not force:
        with np.load(path, allow_pickle=False) as d:
            if str(d["signature"]) == sig:
                if verbose:
                    print(f"loaded theory cache {path.name} (params match, no recompute)")
                return _unpack(d)
        if verbose:
            print(f"theory cache {path.name} params differ -- rebuilding")

    if verbose:
        print("computing theory (one-time expensive propagation)...")
    res = compute_theory(**p, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signature=sig, **res)
    if verbose:
        print(f"saved theory cache -> {path}")
    out = dict(res)
    out["intensity"] = {c: out.pop(f"intensity_{c}") for c in CONDITIONS}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if a valid cache exists")
    args = ap.parse_args()
    load_or_build(force=args.force)
