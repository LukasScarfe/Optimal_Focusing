### ALL OF THIS CODE COMES DIRECTLY FROM FOCUSING_DATA.PY BUT IS PUT INTO A NORMAL PYTHON CODE NOW RATHER THAN A MARIMO NOTEBOOK SO THAT IT CAN BE ACCESSED FROM FIGURES.PY ###

import pickle
import tifffile
import time
from matplotlib.colors import LogNorm
from pathlib import Path
import re
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

### BASIC SETUP AND CHOICE OF WHICH DATA TO ANALYZE ###

ROOT = Path("/Users/carriecrane/GitHub/Optimal_Focusing/data/focusing_images/29_n15")
z_vals = np.arange(0,25.5, 0.5)

BEAM_SIZE = 29
N_FOCUS = 15
PIXEL_PITCH_MM = 0.0048


### CELL 3 FROM NOTEBOOK ###

def get_files(l_val):
    folder = ROOT / f"l={l_val}"
    files = sorted(folder.glob("*.tif"), key = lambda f: int(re.findall(r'\d+',f.stem)[-1]))
    return files

for i in range(6):
    files = get_files(i)
    #print(f"l={i}: {len(files)} files - first: {files[0].name}, last:{files[-1].name}")

### CELL 4 FROM NOTEBOOK ###

def find_center_and_profile(img, search_radius=200):
    blurred = gaussian_filter(img, sigma=3)
    ny, nx  = img.shape

    # Step 1: brightest pixel is ON the innermost bright ring
    yb, xb = np.unravel_index(np.argmax(blurred), blurred.shape)

    # Step 2: walk inward from brightest pixel toward image center
    # looking for the dark hole minimum
    img_cx, img_cy = nx // 2, ny // 2
    dx   = img_cx - xb
    dy   = img_cy - yb
    dist = np.sqrt(dx**2 + dy**2)

    steps = np.arange(1, min(search_radius, int(dist)))
    xs    = np.clip((xb + steps * dx / dist).astype(int), 0, nx - 1)
    ys    = np.clip((yb + steps * dy / dist).astype(int), 0, ny - 1)
    line_vals = blurred[ys, xs]

    # Dark hole = minimum along this inward line
    # but only search within a reasonable range (5 to search_radius px)
    min_idx = np.argmin(line_vals[5:]) + 5
    cx = float(xs[min_idx])
    cy = float(ys[min_idx])

    # Step 3: radial profile from dark center
    Y, X    = np.ogrid[:ny, :nx]
    R       = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    r_max   = min(300, R.max())
    profile = np.array([
        img[R == r].mean() if np.any(R == r) else 0.0
        for r in range(r_max)
    ])

    return cx, cy, profile

def get_innermost_ring(profile, img_max, prev_r=None):

    smooth = gaussian_filter(profile.astype(float), sigma=3)

    r_min = 5

    if len(smooth) <= r_min:
        return np.nan, np.nan, np.nan

    r_peak = np.argmax(smooth[r_min:]) + r_min

    peak = smooth[r_peak]

    return float(r_peak), float(peak), np.nan


def process_frame(path, l_val, prev_r=None):
    raw = tifffile.imread(path).astype(float)
    if raw.ndim == 3:
        raw = raw.mean(axis=0)

    bg  = np.median([raw[:20,:20], raw[:20,-20:],
                     raw[-20:,:20], raw[-20:,-20:]])
    img = np.clip(raw - bg, 0, None)
    width = beam_width(img)

    if l_val == 0:
        blurred = gaussian_filter(img, sigma=2)
        yc, xc  = np.unravel_index(np.argmax(blurred), blurred.shape)
        cx, cy  = float(xc), float(yc)
        peak    = float(img[int(cy), int(cx)])
        return dict(name=path.name, image=raw, profile=np.zeros(300),
                    cx=cx, cy=cy, ring_r=np.nan, ring_w=np.nan, peak=peak,width=width)

    cx, cy, profile = find_center_and_profile(img)

    # Try with prev_r first, then fall back to looser global search
    ring_r, peak, ring_w = get_innermost_ring(profile, img.max(), prev_r)

    if np.isnan(ring_r):
        # Retry with lower threshold — transition zone has fainter rings
        ring_r, peak, ring_w = get_innermost_ring(profile, img.max() * 0.3, None)

    return dict(name=path.name, image=raw, profile=profile,
                cx=cx, cy=cy, ring_r=ring_r, ring_w=ring_w, peak=peak, width = width)

def beam_width(img):

    blurred = gaussian_filter(img, sigma=2)

    thresh = 0.05 * blurred.max()

    mask = blurred > thresh

    if mask.sum() < 20:
        return np.nan

    Y, X = np.indices(img.shape)

    I = blurred * mask

    total = np.sum(I)

    cx = np.sum(X * I) / total
    cy = np.sum(Y * I) / total

    r2 = (X - cx)**2 + (Y - cy)**2

    w = np.sqrt(np.sum(I * r2) / total)

    return float(w)

def find_crop_center(img):

    blurred = gaussian_filter(img, sigma=3)

    threshold = np.percentile(
        blurred,
        95
    )

    bright = blurred >= threshold

    Y, X = np.mgrid[
        :img.shape[0],
        :img.shape[1]
    ]

    cx = float(
        np.average(
            X[bright],
            weights=blurred[bright]
        )
    )

    cy = float(
        np.average(
            Y[bright],
            weights=blurred[bright]
        )
    )

    return cx, cy

all_results = {}

for ls_val in range(6):
    files_l   = get_files(ls_val)
    results_l = []
    prev_r = None
    t0        = time.time()

    for i_frame, f in enumerate(files_l):
        r = process_frame(f, ls_val)
        results_l.append(r)
        if not np.isnan(r["ring_r"]):
            prev_r = r["ring_r"]
        elapsed = time.time() - t0
        #print(f"l={ls_val}  frame {i_frame+1}/{len(files_l)}  ({elapsed:.0f}s)  ring_r={r['ring_r']:.1f}px  peak={r['peak']:.1f}")
    all_results[ls_val] = results_l

def plot_ring_radius():
    fig_r, axes_r = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes_r = axes_r.flatten()

    for l_val in range(6):
        res    = all_results[l_val]
        z_l    = z_vals[:len(res)]
        ring_r = np.array([r["ring_r"] for r in res])
        valid  = ~np.isnan(ring_r)

        ax = axes_r[l_val]
        ax.plot(z_l[valid], ring_r[valid], 'o-', ms=3, lw=1.5, color=f'C{l_val}')
        ax.set_title(f'ℓ = {l_val}')
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('inner ring radius (px)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Innermost ring radius vs z - beam = {BEAM_SIZE} mm   n = {N_FOCUS}', fontsize=13)

    plt.tight_layout()
    plt.savefig("/Users/carriecrane/GitHub/Optimal_Focusing/ring_radius_vs_z.png", dpi=130)
    plt.close()
    print("saved ring_radius_vs_z.png")


def plot_beam_width():

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14,8),
        sharex=True
    )

    axes = axes.flatten()

    for l_val in range(6):

        res = all_results[l_val]

        z_l = z_vals[:len(res)]

        width = np.array([
            r["width"] for r in res
        ])

        valid = ~np.isnan(width)

        ax = axes[l_val]

        ax.plot(
            z_l[valid],
            width[valid],
            'o-',
            ms=3,
            lw=1.5,
            color=f'C{l_val}'
        )

        ax.set_title(f'ℓ = {l_val}')
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('beam width (px)')
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Second-moment beam width vs z',
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        "/Users/carriecrane/GitHub/Optimal_Focusing/beam_width_vs_z.png",
        dpi=130
    )

    plt.close()

    print("saved beam_width_vs_z.png")



def plot_focus_position_vs_oam():

    focus_z = []

    for l_val in range(6):

        width = np.array([
            r["width"]
            for r in all_results[l_val]
        ])

        valid = ~np.isnan(width)

        width_valid = width[valid]
        z_valid = z_vals[:len(width)][valid]

        best_idx = np.argmin(width_valid)

        focus_z.append(
            z_valid[best_idx]
        )

    plt.figure(figsize=(6,4))

    plt.plot(
        range(6),
        focus_z,
        'o-',
        linewidth=2
    )

    plt.xlabel("OAM mode ℓ")
    plt.ylabel("Focus position (mm)")
    plt.title("Focus position vs OAM mode")

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_beam_radius_all_modes():

    plt.figure(figsize=(8,6))

    for l_mode in range(6):

        width = np.array([
            r["width"]
            for r in all_results[l_mode]
        ])

        radius = width / 2

        plt.plot(
            z_vals[:len(radius)],
            radius,
            linewidth=2,
            label=f"ℓ={l_mode}"
        )

    plt.xlabel("z (mm)")
    plt.ylabel("Beam radius (pixels)")
    plt.title("Beam radius versus propagation distance")

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()



def make_longitudinal_map(l_mode, half_width=150):

    rows = []

    for result in all_results[l_mode]:

        img = result["image"].astype(float)

        # Background subtraction
        bg = np.median([
            img[:20,:20],
            img[:20,-20:],
            img[-20:,:20],
            img[-20:,-20:]
        ])

        img = np.clip(img - bg, 0, None)
        img = img / (img.max() + 1e-9)

        # Beam center
        cx, cy = find_crop_center(img)

        x0 = int(round(cx))
        y0 = int(round(cy))

        # Crop around beam center
        left  = max(0, x0 - half_width)
        right = min(img.shape[1], x0 + half_width)

        # Average several rows to reduce noise
        row = np.mean(
            img[y0-1:y0+2, :],
            axis=0
        )

        rows.append(row)

    xz_map = np.array(rows)

    # Normalize each z slice independently
    xz_map = xz_map / (
        np.max(xz_map, axis=1, keepdims=True) + 1e-12
    )

    return xz_map


def plot_longitudinal_map(l_mode):

    xz_map = make_longitudinal_map(l_mode)

    vmax = np.percentile(xz_map, 99)

    plt.figure(figsize=(8,6))

    plt.imshow(
        xz_map,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=np.percentile(xz_map,99),
        extent=[
            -xz_map.shape[1]/2,
             xz_map.shape[1]/2,
             z_vals[0],
             z_vals[len(xz_map)-1]
        ]
    )

    plt.xlabel("x (pixels)")
    plt.ylabel("z (mm)")
    plt.title(f"Longitudinal intensity map (ℓ={l_mode})")

    plt.colorbar(label="Normalized intensity")

    plt.tight_layout()
    plt.show()


def plot_all_longitudinal_maps():

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14,8),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for l_mode in range(6):

        xz_map = make_longitudinal_map(l_mode)

        vmax = np.percentile(xz_map, 99)

        axes[l_mode].imshow(
            xz_map,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmax=vmax,
            extent=[
                -xz_map.shape[1]/2,
                 xz_map.shape[1]/2,
                 z_vals[0],
                 z_vals[len(xz_map)-1]
            ]
        )

        axes[l_mode].set_title(f"ℓ={l_mode}")

    plt.tight_layout()
    plt.show()


def make_radial_map(l_mode, max_radius=250):

    radial_profiles = []

    for result in all_results[l_mode]:

        img = result["image"].astype(float)

        # Background subtraction
        bg = np.median([
            img[:20,:20],
            img[:20,-20:],
            img[-20:,:20],
            img[-20:,-20:]
        ])

        img = np.clip(img - bg, 0, None)

        # Beam center
        cx, cy = find_crop_center(img)

        Y, X = np.indices(img.shape)

        R = np.sqrt(
            (X - cx)**2 +
            (Y - cy)**2
        )

        R_int = np.floor(R).astype(int)

        profile = np.zeros(max_radius)

        for r in range(max_radius):

            mask = (R_int == r)

            if np.any(mask):
                profile[r] = np.mean(img[mask])

        radial_profiles.append(profile)

    radial_map = np.array(radial_profiles)

    # Normalize each z slice
    radial_map = radial_map / (
        radial_map.max(axis=1, keepdims=True) + 1e-12
    )

    return radial_map


def plot_radial_map(l_mode):

    radial_map = make_radial_map(l_mode)

    vmax = np.percentile(radial_map, 99)

    plt.figure(figsize=(8,6))

    plt.imshow(
        radial_map,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmax=vmax,
        extent=[
            0,
            radial_map.shape[1],
            z_vals[0],
            z_vals[len(radial_map)-1]
        ]
    )

    plt.xlabel("Radius (pixels)")
    plt.ylabel("z (mm)")
    plt.title(f"Radial intensity map (ℓ={l_mode})")

    plt.colorbar(label="Normalized intensity")

    plt.tight_layout()
    plt.show()


def plot_all_radial_maps():

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14,8),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for l_mode in range(6):

        radial_map = make_radial_map(l_mode)

        vmax = np.percentile(radial_map, 99)

        axes[l_mode].imshow(
            radial_map,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmax=vmax,
            extent=[
                0,
                radial_map.shape[1],
                z_vals[0],
                z_vals[len(radial_map)-1]
            ]
        )

        axes[l_mode].set_title(f"ℓ={l_mode}")

    plt.tight_layout()
    plt.show()


def make_xz_map(l_mode):

    profiles = []

    cx_list = []
    cy_list = []

    for result in all_results[l_mode]:

        img = result["image"].astype(float)

        bg = np.median([
            img[:20, :20],
            img[:20, -20:],
            img[-20:, :20],
            img[-20:, -20:]
        ])

        img = np.clip(img - bg, 0, None)

        cx, cy = find_crop_center(img)

        cx_list.append(cx)
        cy_list.append(cy)

        cx = int(round(cx))
        cy = int(round(cy))

        half_width = 250

        x1 = max(0, cx - half_width)
        x2 = min(img.shape[1], cx + half_width)

        profile = np.mean(
            img[cy-3:cy+4, x1:x2],
            axis=0
        )

        profiles.append(profile)

    #print(f"ℓ = {l_mode}")
    #print("cx range:", min(cx_list), max(cx_list))
    #print("cy range:", min(cy_list), max(cy_list))

    #plt.figure(figsize=(8, 3))
    #plt.plot(cx_list, ".-", label="cx")
    #plt.plot(cy_list, ".-", label="cy")
    #plt.legend()
    #plt.xlabel("z step")
    #plt.ylabel("pixel")
    #plt.title(f"Center Positions, ℓ={l_mode}")
    #plt.show()

    return np.array(profiles)

def plot_xz_map(l_mode):

    xz = make_xz_map(l_mode)

    xz = xz / (xz.max() + 1e-12)

    plt.figure(figsize=(7.2,5))

    im = plt.imshow(
        xz.T,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        extent=[
            z_vals[0],
            z_vals[len(xz)-1],
            -250,
            250
        ]
    )

    plt.xlabel("Propagation distance z (mm)")
    plt.ylabel("x (pixels)")

    plt.title(
        f"Experimental propagation cross-section ($\\ell$={l_mode})"
    )

    plt.colorbar(
        im,
        label="normalized intensity"
    )

    plt.tight_layout()
    plt.show()

def plot_all_xz_maps():

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12,8),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()
        
    vmax_global = max(make_xz_map(l).max() for l in range (6)
)
    for l_mode in range(6):

        xz = make_xz_map(l_mode)
        
        im = axes[l_mode].imshow(
            xz.T / vmax_global,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            extent=[
                z_vals[0],
                z_vals[-1],
                -32,
                32
            ],
            vmin=0,
            vmax=1
        )

        axes[l_mode].set_title(rf'$\ell={l_mode}$')

        axes[l_mode].set_xlabel('Propagation distance z (mm)')
        axes[l_mode].set_ylabel('Transverse position x (pixels)')

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.85,
        pad=0.02
    )
    cbar.set_label('Normalized intensity')

    fig.suptitle(
        'Experimental propagation cross-sections',
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

##### SIMULATION RESULTS ####

def build_optimal_mask(l, n0, a):
    lam = 633e-9
    k = 2 * np.pi / lam
    tau_f = 1.0 / (2 * np.pi * n0)

    rho_beta = np.linspace(0.0, 1.0, 4000)

    def I0_of_beta(beta):
        J = jv(l, beta * rho_beta)
        return np.trapz(rho_beta * J**2, rho_beta) / (2 * np.pi)

    def R_of_beta(beta):
        J = jv(l, beta * rho_beta)
        phase = np.exp(1j * rho_beta**2 / tau_f)
        return np.trapz(rho_beta * J**2 * phase, rho_beta) / (2 * np.pi)

    def lambda_plus(beta):
        I0 = I0_of_beta(beta)
        R = R_of_beta(beta)
        return (I0 + np.abs(R)) / (2 * tau_f**2)

    opt = minimize_scalar(lambda b: -lambda_plus(b), bounds=(1e-4, 20.0),
                           method='bounded', options={'xatol': 1e-6})
    beta_opt = float(opt.x)

    Nrho_in = 1000
    rho_in = np.linspace(0.0, 1.0, Nrho_in)
    J_beta = jv(l, beta_opt * rho_in)

    R_beta = np.trapz(rho_in * J_beta**2 * np.exp(1j * rho_in**2 / tau_f), rho_in) / (2 * np.pi)
    theta = 0.5 * np.angle(R_beta)
    phase_arg = rho_in**2 / (2 * tau_f) - theta

    def normalize_pupil(psi_raw, rho):
        norm = np.sqrt(2 * np.pi * np.trapz(rho * np.abs(psi_raw)**2, rho))
        return psi_raw / norm

    psi_opt_raw = J_beta * np.cos(phase_arg)
    psi_opt = normalize_pupil(psi_opt_raw, rho_in)

    return psi_opt, rho_in, k, tau_f


def build_all_simulations(n0, a=29e-3, Ntau=100, Nrho_out=250, Nx=300,
                           cache_path="sim_data_cache.pkl", force_recompute=False):
    """Build simulated xz propagation maps for l=0..5, matching experimental units (mm).
    Caches results to disk since this is expensive (~minutes)."""

    cache_file = Path(cache_path)
    cache_key = (n0, a, Ntau, Nrho_out, Nx)
	
    print("cache path =", cache_file.resolve())
    print("cache exists =", cache_file.exists())
    if not force_recompute and cache_file.exists():
        with open(cache_file, "rb") as f:
            cached_key, cached_data = pickle.load(f)
        if cached_key == cache_key:
            print(f"Loaded cached sim_data from {cache_file}")
            return cached_data
        else:
            print("Cache exists but parameters changed — recomputing.")

    rho_out = np.linspace(0.0, 1.0, Nrho_out)
    x_out = np.linspace(-1.0, 1.0, Nx)

    sim_data = {}

    for l_mode in range(6):
        print(f"Computing l={l_mode}...")
        psi_opt, rho_in, k, tau_f = build_optimal_mask(l_mode, n0, a)

        tau_max = 2.0 * tau_f
        taus = np.concatenate(([0.0], np.linspace(tau_f / 250, tau_max, Ntau - 1)))

        intensity_tau_rho = propagate_mask(psi_opt, rho_in, rho_out, taus, l_mode)

        sim_x_mm = 1e3 * a * x_out
        sim_z_mm = 1e3 * (k * a **2 * taus)

        sim_xz = np.zeros((len(taus), Nx))
        for j in range(len(taus)):
            sim_xz[j, :] = np.interp(np.abs(x_out), rho_out, intensity_tau_rho[j, :])

        sim_data[l_mode] = (sim_z_mm, sim_x_mm, sim_xz)

    with open(cache_file, "wb") as f:
        pickle.dump((cache_key, sim_data), f)
    print(f"Saved sim_data to {cache_file}")

    return sim_data


def propagate_mask(psi_radial, rho_in, rho_out, taus, l):
    intensity_tau_rho = np.zeros((len(taus), len(rho_out)), dtype=float)

    initial_amp = np.interp(rho_out, rho_in, psi_radial)
    intensity_tau_rho[0, :] = np.abs(initial_amp)**2

    rho_prime = rho_in[None, :]
    psi_row = psi_radial[None, :]
    rho_col = rho_out[:, None]

    for j, tau in enumerate(taus[1:], start=1):
        phase_in = np.exp(1j * rho_prime**2 / (2 * tau))
        bessel = jv(l, rho_col * rho_prime / tau)
        integrand = rho_prime * phase_in * bessel * psi_row
        radial_int = np.trapz(integrand, rho_in, axis=1)
        amp = ((-1j)**l) / (1j * tau) * np.exp(1j * rho_out**2 / (2 * tau)) * radial_int
        intensity_tau_rho[j, :] = np.abs(amp)**2

    intensity_tau_rho /= intensity_tau_rho.max()
    return intensity_tau_rho


def make_centered_xz_map(l_mode):

    profiles = []

    for result in all_results[l_mode]:

        img = result["image"].astype(float)

        bg = np.median([
            img[:20,:20],
            img[:20,-20:],
            img[-20:,:20],
            img[-20:,-20:]
        ])

        img = np.clip(img - bg, 0, None)

        cx, cy = find_crop_center(img)

        cy = int(round(cy))

        half_width = 120

        profile = np.mean(
            img[max(0,cy-3):cy+4,:],
            axis=0
        )

        cx_int = int(round(cx))

        left = max(0, cx_int-half_width)
        right = min(len(profile), cx_int+half_width)

        profile = gaussian_filter(
            profile,
            sigma=2
        )

        profiles.append(profile)

    return np.array(profiles)

def plot_centered_xz_map(l_mode):

    xz = make_centered_xz_map(l_mode)

    xz = xz / (xz.max() + 1e-9)

    nx = xz.shape[1]

    x_pixels = np.arange(nx) - nx/2

    fig, ax = plt.subplots(figsize=(7,5))

    im = ax.imshow(
        xz.T,
        origin='lower',
        aspect='auto',
        extent=[
            z_vals[0],
            z_vals[len(xz)-1],
            x_pixels[0],
            x_pixels[-1]
        ],
        cmap='viridis',
        vmin = 0,
        vmax = 1
    )

    ax.set_xlabel('Propagation distance z (mm)')
    ax.set_ylabel('Transverse position x (pixels)')
    ax.set_title(f'Experimental propagation map (ℓ={l_mode})')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized intensity')

    plt.tight_layout()
    plt.show()

def plot_all_centered_xz_maps():

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14,8)
    )

    axes = axes.flatten()

    for l_mode in range(6):

        xz = make_centered_xz_map(l_mode)

        xz = xz / (xz.max() + 1e-9)

        nx = xz.shape[1]
        x_pixels = np.arange(nx) - nx/2

        im = axes[l_mode].imshow(
            xz.T,
            origin='lower',
            aspect='auto',
            extent=[
                z_vals[0],
                z_vals[len(xz)-1],
                x_pixels[0],
                x_pixels[-1]
            ],
            cmap='viridis',
            norm=LogNorm(vmin=1e-3, vmax=1)
        )

        axes[l_mode].set_title(f'ℓ={l_mode}')
        axes[l_mode].set_xlabel('z (mm)')
        axes[l_mode].set_ylabel('x (pixels)')

    plt.tight_layout()
    plt.show()

PIXEL_PITCH_MM = 0.0048  # <-- set this to your camera's pixel pitch (mm/pixel),
                          #     divided by any magnification factor in your imaging setup

def make_centered_xz_map_mm(l_mode, half_width=120):
    """Like make_centered_xz_map, but actually crops to half_width and
    returns an x-axis in mm instead of raw pixels."""
    profiles = []
    for result in all_results[l_mode]:
        img = result["image"].astype(float)
        bg = np.median([
            img[:20,:20], img[:20,-20:], img[-20:,:20], img[-20:,-20:]
        ])
        img = np.clip(img - bg, 0, None)
        cx, cy = find_crop_center(img)
        cy = int(round(cy))
        cx_int = int(round(cx))

        row = np.mean(img[max(0,cy-3):cy+4, :], axis=0)
        row = gaussian_filter(row, sigma=2)

        # actually crop around cx_int (this was computed but unused before)
        left = max(0, cx_int - half_width)
        right = min(len(row), cx_int + half_width)
        cropped = row[left:right]

        # pad so all profiles have the same width even near image edges
        pad_left = half_width - (cx_int - left)
        pad_right = (2*half_width) - len(cropped) - pad_left
        cropped = np.pad(cropped, (pad_left, pad_right), mode='constant')
        profiles.append(cropped)

    xz = np.array(profiles)
    nx = xz.shape[1]
    x_mm = (np.arange(nx) - nx/2) * PIXEL_PITCH_MM
    return xz, x_mm


def plot_experiment_vs_simulation(l_mode, sim_z, sim_x_mm, sim_xz,
                                   z_focus_exp=None, z_focus_sim=None,
                                   vmin=5e-3, vmax=1.0):
    """Side-by-side comparison on matched mm axes and matched log color scale."""
    xz_exp, x_mm_exp = make_centered_xz_map_mm(l_mode)
    xz_exp = xz_exp / (xz_exp.max() + 1e-9)

    sim_xz_norm = sim_xz / (sim_xz.max() + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        xz_exp.T, origin='lower', aspect='auto', cmap='viridis',
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=[z_vals[0], z_vals[len(xz_exp)-1], x_mm_exp[0], x_mm_exp[-1]]
    )
    axes[0].set_title(f'Experiment (ℓ={l_mode})')
    axes[0].set_xlabel('z (mm)')
    axes[0].set_ylabel('x (mm)')
    if z_focus_exp is not None:
        axes[0].axvline(z_focus_exp, color='w', ls='--', lw=1)

    im1 = axes[1].imshow(
        sim_xz_norm.T, origin='lower', aspect='auto', cmap='viridis',
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=[sim_z[0], sim_z[-1], sim_x_mm[0], sim_x_mm[-1]]
    )
    axes[1].set_title(f'Simulation (ℓ={l_mode})')
    axes[1].set_xlabel('z (mm)')
    axes[1].set_ylabel('x (mm)')
    if z_focus_sim is not None:
        axes[1].axvline(z_focus_sim, color='w', ls='--', lw=1)

    # match x-range to whichever is narrower so the comparison window is identical
    z_lo = max(z_vals[0], sim_z[0])
    z_hi = min(z_vals[len(xz_exp)-1], sim_z[-1])
    axes[0].set_xlim(z_lo, z_hi)
    axes[1].set_xlim(z_lo, z_hi)

    cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label('Normalized intensity (log)')
    plt.show()



def plot_all_xz_maps_clean():

    # ----------------------------------------
    # Build all maps first
    # ----------------------------------------

    maps = []

    for l_mode in range(6):

        xz = make_centered_xz_map(l_mode)

        xz = xz / (xz.max() + 1e-9)

        maps.append(xz)

    # ----------------------------------------
    # Shared color scaling
    # ----------------------------------------

    all_pixels = np.concatenate(
        [m.ravel() for m in maps]
    )

    vmax = 1.0

    # suppress extreme background
    vmin = 5e-3

    # ----------------------------------------
    # Figure layout
    # ----------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14,8),
        constrained_layout=True
    )

    axes = axes.flatten()

    # ----------------------------------------
    # Plot
    # ----------------------------------------

    for l_mode, ax in enumerate(axes):

        xz = maps[l_mode]
        
        img = xz.T

        y_center = img.shape[0]//2
        half = 100
        
        img = img[y_center-half : y_center+half, :]

        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            norm=LogNorm(
                vmin=vmin,
                vmax=vmax
            ),
            extent=[
                z_vals[0],
                z_vals[len(xz)-1],
                -100,100
            ]
        )
        
        ax.set_title(
            rf"$\ell={l_mode}$",
            fontsize=12
        )

        ax.set_xlabel(
            "z (mm)"
        )

        ax.set_ylabel(
            "x (pixels)"
        )

    # ----------------------------------------
    # One shared colorbar
    # ----------------------------------------

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.85,
        pad=0.02
    )

    cbar.set_label(
        "Normalized intensity"
    )

    fig.suptitle(
        "Experimental propagation maps",
        fontsize=16
    )

    plt.show()

def make_rz_map(l_mode):

    profiles = []

    max_len = 0

    for result in all_results[l_mode]:

        img = result["image"].astype(float)

        bg = np.median([
            img[:20,:20],
            img[:20,-20:],
            img[-20:,:20],
            img[-20:,-20:]
        ])

        img = np.clip(img - bg, 0, None)

        cx, cy = find_crop_center(img)

        Y, X = np.indices(img.shape)

        R = np.sqrt(
            (X-cx)**2 +
            (Y-cy)**2
        )

        R = R.astype(int)

        max_r = 350

        radial_profile = np.zeros(max_r)

        for r in range(max_r):

            mask = (R >= r) & (R < r+1)

            if np.any(mask):
                radial_profile[r] = np.mean(
                    img[mask]
                )

        profiles.append(radial_profile)

        max_len = max(
            max_len,
            len(radial_profile)
        )

    padded_profiles = []

    for profile in profiles:

        padded = np.pad(
            profile,
            (0, max_len - len(profile)),
            mode="constant"
        )

        padded_profiles.append(padded)

    return np.array(padded_profiles)

def plot_rz_map(l_mode):

    rz = make_rz_map(l_mode)

    rz = rz / (rz.max() + 1e-9)

    plt.figure(figsize=(7,6))

    vmin = 0
    vmax = np.percentile(rz, 99.5)

    plt.imshow(
        rz + 1e-4,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        norm=LogNorm(vmin=1e-3, vmax =1),
        extent=[
            0,
            rz.shape[1],
            z_vals[0],
            z_vals[len(rz)-1]
        ]
    )

    plt.xlabel("Radius (pixels)")
    plt.ylabel("z (mm)")
    plt.title(f"Experimental r-z propagation map (ℓ={l_mode})")

    plt.colorbar(label="Normalized intensity")

    plt.tight_layout()
    plt.show()

def plot_all_rz_maps():

    for l_mode in range(6):

        plot_rz_map(l_mode)


def plot_all_beam_centers():

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14,8),
        sharex=True
    )

    axes = axes.flatten()

    for l_mode in range(6):

        centers_x = []
        centers_y = []

        for result in all_results[l_mode]:

            img = result["image"].astype(float)

            bg = np.median([
                img[:20,:20],
                img[:20,-20:],
                img[-20:,:20],
                img[-20:,-20:]
            ])

            img = np.clip(img - bg, 0, None)

            cx, cy = find_crop_center(img)

            centers_x.append(cx)
            centers_y.append(cy)

        z_plot = z_vals[:len(centers_x)]

        axes[l_mode].plot(z_plot, centers_x, label='cx')
        axes[l_mode].plot(z_plot, centers_y, label='cy')

        axes[l_mode].set_title(f'ℓ={l_mode}')
        axes[l_mode].grid(alpha=0.3)

    axes[0].legend()

    plt.tight_layout()
    plt.show()


def plot_all_xz_maps_with_simulation(sim_data, vmin=5e-3, vmax=1.0):
    """
    sim_data: dict mapping l_mode -> (sim_z_mm, sim_x_mm, sim_xz)
              sim_z_mm:  1D array, propagation distance in mm
              sim_x_mm:  1D array, transverse position in mm
              sim_xz:    2D array, shape (len(sim_z_mm), len(sim_x_mm))
    """

    # ---- Build experimental maps (mm units, matches make_centered_xz_map_mm) ----
    exp_maps = {}
    for l_mode in range(6):
        xz, x_mm = make_centered_xz_map_mm(l_mode)
        xz = xz / (xz.max() + 1e-9)
        exp_maps[l_mode] = (xz, x_mm)

    fig, axes = plt.subplots(6, 2, figsize=(10, 22), constrained_layout=True)

    for l_mode in range(6):
        xz_exp, x_mm_exp = exp_maps[l_mode]
        sim_z, sim_x_mm, sim_xz = sim_data[l_mode]
        sim_xz_norm = sim_xz / (sim_xz.max() + 1e-9)

        sim_z_plot = 25 * (sim_z - sim_z.min()) / (sim_z.max() - sim_z.min())
        sim_x_plot = 0.55 * sim_x_mm / np.max(np.abs(sim_x_mm))

        ax_exp, ax_sim = axes[l_mode]

        im = ax_exp.imshow(
            xz_exp.T, origin='lower', aspect='auto', cmap='viridis',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extent=[z_vals[0], z_vals[len(xz_exp)-1], x_mm_exp[0], x_mm_exp[-1]]
        )
        ax_exp.set_title(f'Experiment ℓ={l_mode}', fontsize=10)
        ax_exp.set_ylabel('x (mm)')

        ax_sim.imshow(
            sim_xz_norm.T, origin='lower', aspect='auto', cmap='viridis',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extent=[sim_z_plot[0], sim_z_plot[-1], sim_x_plot[0], sim_x_plot[-1]]
        )
        ax_sim.set_title(f'Simulation ℓ={l_mode}', fontsize=10)

        # match z-range so both panels in a row show the same window
        z_lo = max(z_vals[0], sim_z[0])
        z_hi = min(z_vals[len(xz_exp)-1], sim_z[-1])
        ax_exp.set_xlim(z_lo, z_hi)
        ax_sim.set_xlim(z_lo, z_hi)

        if l_mode == 5:
            ax_exp.set_xlabel('z (mm)')
            ax_sim.set_xlabel('z (mm)')

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label('Normalized intensity (log)')

    fig.suptitle('Experiment vs. simulation propagation maps', fontsize=15)
    plt.show()

# CELL 6.17

def plot_all_peak_intensities_normalized(save_dir=None):

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.titlesize": 16
    })

    fig, axes = plt.subplots(
        2, 3,
        figsize=(10, 6),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for l_mode in range(6):

        peak_vals = []

        for result in all_results[l_mode]:

            img = result["image"].astype(float)

            bg = np.median([
                img[:20,:20],
                img[:20,-20:],
                img[-20:,:20],
                img[-20:,-20:]
            ])

            img = np.clip(img - bg, 0, None)

            peak_vals.append(np.max(img))

        peak_vals = np.array(peak_vals)

        peak_vals /= peak_vals.max()

        z_plot = z_vals[:len(peak_vals)]

        axes[l_mode].plot(
            z_plot,
            peak_vals,
            'o-',
            linewidth=2,
            markersize=4
        )

        focus_z = z_plot[np.argmax(peak_vals)]

        axes[l_mode].axvline(
            focus_z,
            linestyle='--',
            alpha=0.6
        )

        axes[l_mode].set_title(f'ℓ = {l_mode}')
        axes[l_mode].grid(alpha=0.3)

        if l_mode >= 3:
            axes[l_mode].set_xlabel('z (mm)')

        if l_mode % 3 == 0:
            axes[l_mode].set_ylabel('Normalized intensity')

    fig.suptitle(
        'Peak Intensity vs Propagation Distance',
        y=1.02
    )

    plt.tight_layout()

    # Save figure
    if save_dir is not None:

        os.makedirs(save_dir, exist_ok=True)

        png_file = os.path.join(
            save_dir,
            "peak_intensity_vs_z.png"
        )

        pdf_file = os.path.join(
            save_dir,
            "peak_intensity_vs_z.pdf"
        )

        plt.savefig(
            png_file,
            dpi=300,
            bbox_inches='tight'
        )

        plt.savefig(
            pdf_file,
            bbox_inches='tight'
        )

        print(f"Saved:")
        print(png_file)
        print(pdf_file)

    plt.show()

def plot_peak_intensity():
    fig_i, axes_i = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes_i = axes_i.flatten()

    for l_val in range(6):
        res   = all_results[l_val]
        z_l   = z_vals[:len(res)]
        peak  = np.array([r["peak"] for r in res])
        valid = ~np.isnan(peak)

        ax = axes_i[l_val]
        ax.plot(z_l[valid], peak[valid], 'o-', ms=3, lw=1.5, color=f'C{l_val}')
        ax.set_title(f'ℓ = {l_val}')
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('peak intensity (counts)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Innermost ring peak intensity vs z position', fontsize=13)
    plt.tight_layout()
    plt.savefig("/Users/carriecrane/GitHub/Optimal_Focusing/analysis/peak_intensity_vs_z.png", dpi=130)
    plt.close()
    print("saved peak_intensity_vs_z.png")

def plot_focusing_images():
    ## CELL 8 ##
    from matplotlib.colors import LogNorm
    from pathlib import Path

    # --------------------------------------------------
    # Create output folders
    # --------------------------------------------------
    SAVE_ROOT = Path(
        "/Users/carriecrane/GitHub/Optimal_Focusing/analysis/29_n15"
    )

    LINEAR_DIR = SAVE_ROOT / "beam_grids_linear"
    LOG_DIR    = SAVE_ROOT / "beam_grids_log"

    LINEAR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------
    # Beam center finder
    # --------------------------------------------------
    def find_crop_center(img):

        blurred = gaussian_filter(img, sigma=3)

        threshold = np.percentile(blurred, 95)

        bright = blurred >= threshold

        Y, X = np.mgrid[:img.shape[0], :img.shape[1]]

        cx = float(
            np.average(X[bright], weights=blurred[bright])
        )

        cy = float(
            np.average(Y[bright], weights=blurred[bright])
        )

        return cx, cy


    # --------------------------------------------------
    # Fourier denoiser
    # --------------------------------------------------
    def fourier_clean(img, r_cut=100):

        from numpy.fft import fft2, ifft2
        from numpy.fft import fftshift, ifftshift

        ny, nx = img.shape

        Y, X = np.ogrid[:ny, :nx]

        R_f = np.sqrt(
            (X - nx//2)**2 +
            (Y - ny//2)**2
        )

        lpf = np.exp(-(R_f/r_cut)**4)

        F = fftshift(fft2(img))

        F *= lpf

        out = np.real(
            ifft2(ifftshift(F))
        )

        return np.clip(out, 0, None)


    # --------------------------------------------------
    # Main plotting routine
    # --------------------------------------------------
    def plot_beam_grid(
        l_val,
        step=5,
        use_log=False
    ):

        res = all_results[l_val]

        indices = list(
            range(0, len(res), step)
        )

        ncols = len(indices)

        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=(ncols*3.2, 4.0)
        )

        # ----------------------------------------------
        # Anchor center
        # ----------------------------------------------
        raw_bg = res[0]["image"].astype(float)

        bg_val = np.median([
            raw_bg[:20,:20],
            raw_bg[:20,-20:],
            raw_bg[-20:,:20],
            raw_bg[-20:,-20:]
        ])

        img_bg = np.clip(
            raw_bg - bg_val,
            0,
            None
        )

        blurred0 = gaussian_filter(
            img_bg,
            sigma=3
        )

        thresh0 = np.percentile(
            blurred0,
            95
        )

        bright0 = blurred0 >= thresh0

        Y0, X0 = np.mgrid[
            :blurred0.shape[0],
            :blurred0.shape[1]
        ]

        anchor_cx = float(
            np.average(
                X0[bright0],
                weights=blurred0[bright0]
            )
        )

        anchor_cy = float(
            np.average(
                Y0[bright0],
                weights=blurred0[bright0]
            )
        )

        half = 350

        global_max = max(
            r["image"].astype(float).max()
            for r in res
        )

        vmax = global_max * 0.95

        # ----------------------------------------------
        # Loop over z positions
        # ----------------------------------------------
        for ax, idx in zip(axes, indices):

            img = res[idx]["image"].astype(float)

            bg = np.median([
                img[:20,:20],
                img[:20,-20:],
                img[-20:,:20],
                img[-20:,-20:]
            ])

            img = np.clip(
                img - bg,
                0,
                None
            )

            img = fourier_clean(
                img,
                r_cut=100
            )

            blurred = gaussian_filter(
                img,
                sigma=3
            )

            threshold = np.percentile(
                blurred,
                95
            )

            bright = blurred >= threshold

            Y, X = np.mgrid[
                :img.shape[0],
                :img.shape[1]
            ]

            dist = np.sqrt(
                (X-anchor_cx)**2 +
                (Y-anchor_cy)**2
            )

            bright &= (dist < 150)

            if bright.sum() > 10:

                cx = float(
                    np.average(
                        X[bright],
                        weights=blurred[bright]
                    )
                )

                cy = float(
                    np.average(
                        Y[bright],
                        weights=blurred[bright]
                    )
                )

            else:

                cx, cy = anchor_cx, anchor_cy

            ny, nx = img.shape

            y1 = int(
                np.clip(
                    cy-half,
                    0,
                    ny-2*half
                )
            )

            x1 = int(
                np.clip(
                    cx-half,
                    0,
                    nx-2*half
                )
            )

            crop = img[
                y1:y1+2*half,
                x1:x1+2*half
            ]

            # suppress center artifact
            crop_cx = cx - x1
            crop_cy = cy - y1

            Yc, Xc = np.ogrid[
                :crop.shape[0],
                :crop.shape[1]
            ]

            Rc = np.sqrt(
                (Xc-crop_cx)**2 +
                (Yc-crop_cy)**2
            )

            soft_mask = 1 / (
                1 +
                np.exp(-(Rc-20)/5)
            )

            crop *= soft_mask

            if use_log:

                ax.imshow(
                    crop + 1,
                    cmap="viridis",
                    origin="upper",
                    norm=LogNorm()
                )

            else:

                ax.imshow(
                    crop,
                    cmap="viridis",
                    origin="upper",
                    vmin=0,
                    vmax=vmax
                )

            ax.set_title(
                f"z={z_vals[idx]:.1f} mm",
                fontsize=8
            )

            ax.axis("off")

        scale_name = "LOG" if use_log else "LINEAR"

        plt.suptitle(
            f"ℓ={l_val}   beam={BEAM_SIZE}   n={N_FOCUS}   ({scale_name})",
            fontsize=14,
            y=0.98
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.92]
        )

        if use_log:

            outfile = LOG_DIR / f"beam_grid_l{l_val}.png"

        else:

            outfile = LINEAR_DIR / f"beam_grid_l{l_val}.png"

        plt.savefig(
            outfile,
            dpi=200,
            bbox_inches="tight"
        )
        
        plt.show()

        return fig 


        print(f"saved {outfile}")


    # --------------------------------------------------
    # Generate everything
    # --------------------------------------------------
    for l_val in range(6):

        plot_beam_grid(
            l_val,
            step=5,
            use_log=False
        )

        plot_beam_grid(
            l_val,
            step=5,
            use_log=True
        )


def load_dataset():

    all_results = {}

    for l_val in range(6):

        files_l = get_files(l_val)

        results_l = []

        prev_r = None

        for f in files_l:

            r = process_frame(f, l_val, prev_r)

            results_l.append(r)

            if not np.isnan(r["ring_r"]):
                prev_r = r["ring_r"]

        all_results[l_val] = results_l

    return all_results




