"""Straight-line centroid model shared by the OAM-focusing waterfall notebooks.

Motivation
----------
The waterfalls used to be built by locating the beam centroid *independently in
every frame* (``find_centroid``) and then (a) choosing the slice row and (b)
rolling the transverse cut so that per-frame centroid sits in the middle.  A
per-frame centroid carries shot noise, background structure and an integer
rounding, so the stacked waterfall picks up a frame-to-frame wobble -- most
visible as the left/right jitter of the beam on the shared-``z`` waterfalls.

Fix
---
Replace the per-frame centroid by a straight line in the scan coordinate ``z``,

        centroid(z) = slope * z + offset ,

fit once per *run*.  All sections (segments) of a run share ONE slope -- the
geometric beam walk-off per mm is a property of the alignment, not of where a
given segment happens to start -- while each section keeps its own offset, which
absorbs the transverse shift when the stage is repositioned between segments.
Both frame-centroid axes (row = y, column = x) are fitted, and each frame is
weighted by its brightness so the bright, well-defined in-focus frames dominate
the line and the dim tails (where the centroid is unreliable) barely count.

A single full-span scan is just the degenerate one-section run: it gets its own
slope and offset, still a straight line rather than a wobbling per-frame track.
"""
from __future__ import annotations

import numpy as np


def fit_shared_slope(z_by_key, val_by_key, w_by_key):
    """Weighted least squares of ``val = slope*z + offset[key]``.

    One ``slope`` is shared across every key; each key keeps its own ``offset``.
    The three arguments are dicts keyed by section id, each mapping to a 1-D
    array (per-frame scan coordinate, value and brightness weight of that
    section).  Returns ``(slope, offsets)`` with ``offsets`` a ``key -> offset``
    dict.
    """
    keys = list(z_by_key)
    z = np.concatenate([np.asarray(z_by_key[k], float) for k in keys])
    v = np.concatenate([np.asarray(val_by_key[k], float) for k in keys])
    w = np.concatenate([np.asarray(w_by_key[k], float) for k in keys])

    # design matrix:  [ z | one-hot(section) ]  ->  params [slope, off_0, off_1, ...]
    X = np.zeros((z.size, 1 + len(keys)))
    X[:, 0] = z
    row = 0
    for j, k in enumerate(keys):
        m = np.asarray(z_by_key[k]).size
        X[row:row + m, 1 + j] = 1.0
        row += m

    sw = np.sqrt(np.clip(w, 0.0, None))
    if not np.any(sw > 0):                 # a run with no usable brightness: fall back
        sw = np.ones_like(sw)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)

    slope = float(coef[0])
    offsets = {k: float(coef[1 + j]) for j, k in enumerate(keys)}
    return slope, offsets


def _fit_line_weighted(z, v, w):
    """Weighted least squares of a single section's line ``v = slope*z + offset``."""
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    sw = np.sqrt(np.clip(np.asarray(w, float), 0.0, None))
    if not np.any(sw > 0):
        sw = np.ones_like(sw)
    X = np.stack([z, np.ones_like(z)], axis=1)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_centroid_lines(sections, shared_slope=True):
    """Fit straight centroid lines for every section of one run.

    ``sections`` maps a section id to a mapping with keys ``z``, ``cy``, ``cx``,
    ``w`` (1-D arrays of equal length: scan coordinate, raw row centroid, raw
    column centroid, per-frame brightness weight).

    ``shared_slope=True`` fits ONE slope shared across every section (each keeps
    its own offset) -- the original run-wide model.  ``shared_slope=False`` fits
    every section INDEPENDENTLY, giving each its own slope *and* offset; this is
    the per-section model to use when the beam walk-off differs section to
    section within a run.

    Returns ``(lines, slopes)`` where ``lines[key] = (cy_fit, cx_fit)`` are the
    fitted centroid arrays to use in place of the raw per-frame centroids.  For
    the shared model ``slopes = (slope_y, slope_x)`` are scalars; for the
    per-section model they are ``(slope_y_by_key, slope_x_by_key)`` dicts.
    """
    z = {k: np.asarray(s["z"], float) for k, s in sections.items()}
    cy = {k: np.asarray(s["cy"], float) for k, s in sections.items()}
    cx = {k: np.asarray(s["cx"], float) for k, s in sections.items()}
    w = {k: np.asarray(s["w"], float) for k, s in sections.items()}

    if shared_slope:
        slope_y, off_y = fit_shared_slope(z, cy, w)
        slope_x, off_x = fit_shared_slope(z, cx, w)
        lines = {k: (slope_y * z[k] + off_y[k], slope_x * z[k] + off_x[k]) for k in sections}
        return lines, (slope_y, slope_x)

    lines, slope_y, slope_x = {}, {}, {}
    for k in sections:
        sy, oy = _fit_line_weighted(z[k], cy[k], w[k])
        sx, ox = _fit_line_weighted(z[k], cx[k], w[k])
        lines[k] = (sy * z[k] + oy, sx * z[k] + ox)
        slope_y[k], slope_x[k] = sy, sx
    return lines, (slope_y, slope_x)


def continuous_knots(sections, order):
    """Handoff points (one per adjacent pair) for a continuous piecewise fit.

    Sections are taken in ``order`` (already sorted along ``z``); the handoff
    between consecutive sections is the MIDPOINT of their overlap -- i.e. the
    same place the ``hardcut`` stitch blend switches from one segment to the
    next -- so the tie is enforced exactly at the visible seam.  For a pair that
    happens not to overlap it degenerates to the midpoint of the gap.  ``z``
    here must be the GLOBAL scan coordinate shared by every section.
    """
    zmax = [np.asarray(sections[k]["z"], float).max() for k in order]
    zmin = [np.asarray(sections[k]["z"], float).min() for k in order]
    return [0.5 * (zmax[i] + zmin[i + 1]) for i in range(len(order) - 1)]


def fit_centroid_lines_continuous(sections, order=None, knots=None):
    """Fit ONE continuous piecewise-linear centroid across a run's sections.

    :func:`fit_centroid_lines` gives every section its own slope AND offset, so
    two neighbours generally DISAGREE at the point where the stitched waterfall
    hands off from one to the next -- the beam, centred on each section's own
    line, jumps sideways at that seam.  This model instead ties the sections
    into a single continuous curve of ``z`` made of one straight piece per
    section, with a slope change allowed only at each handoff ``t_k``::

        centroid(z) = c0 + c1*z + sum_k d_k * (z - t_k)_+ .

    The hinge basis ``(z - t_k)_+`` bakes continuity in, so neighbouring pieces
    are EQUAL at their shared handoff by construction -- no jump -- while each
    section still takes its own local slope (the real per-mm beam walk-off), and
    the brightness weights keep the dim tails from pulling the line.

    ``sections`` maps a section id to a mapping with keys ``z`` (the GLOBAL scan
    coordinate -- every section must already be on one shared axis so the
    overlaps line up), ``cy``, ``cx`` and ``w``.  ``order`` overrides the
    along-``z`` section order (default: sorted by each section's minimum ``z``);
    ``knots`` overrides the handoff locations (default: :func:`continuous_knots`).

    Returns ``(lines, info)`` with ``lines[key] = (cy_fit, cx_fit)`` the fitted
    centroid arrays to use in place of the raw per-frame centroids, and
    ``info = (coef_y, coef_x, knots, order)``.
    """
    keys = list(sections) if order is None else list(order)
    keys = sorted(keys, key=lambda k: np.asarray(sections[k]["z"], float).min())
    if knots is None:
        knots = continuous_knots(sections, keys)
    knots = [float(t) for t in knots]

    def basis(z):
        z = np.asarray(z, float)
        cols = [np.ones_like(z), z] + [np.clip(z - t, 0.0, None) for t in knots]
        return np.stack(cols, axis=1)

    z = np.concatenate([np.asarray(sections[k]["z"], float) for k in keys])
    cy = np.concatenate([np.asarray(sections[k]["cy"], float) for k in keys])
    cx = np.concatenate([np.asarray(sections[k]["cx"], float) for k in keys])
    w = np.concatenate([np.asarray(sections[k]["w"], float) for k in keys])
    sw = np.sqrt(np.clip(w, 0.0, None))
    if not np.any(sw > 0):
        sw = np.ones_like(sw)

    B = basis(z)
    coef_y, *_ = np.linalg.lstsq(B * sw[:, None], cy * sw, rcond=None)
    coef_x, *_ = np.linalg.lstsq(B * sw[:, None], cx * sw, rcond=None)

    lines = {k: (basis(sections[k]["z"]) @ coef_y, basis(sections[k]["z"]) @ coef_x)
             for k in sections}
    return lines, (coef_y, coef_x, knots, keys)


def _resolve_order_knots(sections, order, knots):
    keys = list(sections) if order is None else list(order)
    keys = sorted(keys, key=lambda k: np.asarray(sections[k]["z"], float).min())
    if knots is None:
        knots = continuous_knots(sections, keys)
    return keys, [float(t) for t in knots]


def seam_jumps(lines, sections, order=None, knots=None):
    """Per-seam centroid mismatch of a set of straight-line fits.

    For each adjacent pair (in chain ``order``) this evaluates both sections'
    lines at the shared seam (``knots``, the overlap midpoints by default) and
    returns how far they disagree -- the sideways jump the stitched beam takes
    at that handoff.  ``lines[key] = (cy_fit, cx_fit)`` as returned by
    :func:`fit_centroid_lines`; ``sections[key]`` carries the matching ``z``.

    Returns ``{right_label: (dy, dx)}`` where ``(dy, dx) = left_line - right_line``
    at the seam.  These are exactly the offsets that, ADDED to the downstream
    (``right``) section, would make the two lines meet -- i.e. the sensible
    default for the manual offsets of :func:`apply_seam_offsets` (start here and
    tune by hand).  The cumulative shift of upstream sections cancels, so this
    default does not depend on any offsets already applied further up the chain.
    """
    keys, knots = _resolve_order_knots(sections, order, knots)
    out = {}
    for i in range(len(keys) - 1):
        left, right, seam = keys[i], keys[i + 1], knots[i]
        zl, zr = sections[left]["z"], sections[right]["z"]
        ly = np.interp(seam, zl, lines[left][0]); ry = np.interp(seam, zr, lines[right][0])
        lx = np.interp(seam, zl, lines[left][1]); rx = np.interp(seam, zr, lines[right][1])
        out[right] = (float(ly - ry), float(lx - rx))
    return out


def apply_seam_offsets(lines, seam_offsets=None, order=None, sections=None):
    """Shift each section's fitted centroid line bodily UP/DOWN at the stitch seams.

    This does NOT refit anything and NEVER changes a slope -- it just moves whole
    sections vertically so their straight lines line up across the seams.  The
    offset is a MANUAL, per-stitching-point knob (like the hand-tuned per-segment
    ``STARTS_MM`` shift along ``z``, but transverse): each seam offset is applied
    CUMULATIVELY to every section downstream of it, so tuning one seam slides that
    section and all later ones together and leaves the earlier ones untouched.

    ``lines[key] = (cy_fit, cx_fit)`` from :func:`fit_centroid_lines`.
    ``seam_offsets`` maps the DOWNSTREAM section label of a seam to its
    ``(dy, dx)`` shift (missing / ``None`` -> no shift); :func:`seam_jumps` gives
    the jump-closing defaults.  ``order`` sets the chain order (default: sections
    sorted along ``z``); pass ``sections`` when ``lines`` alone can't be ordered
    (it can here, since the fitted arrays follow each section's ``z``).

    Returns a new ``{key: (cy_fit, cx_fit)}`` with the shifts baked in.
    """
    keys = list(lines) if order is None else list(order)
    if sections is not None:
        keys = sorted(keys, key=lambda k: np.asarray(sections[k]["z"], float).min())
    seam_offsets = seam_offsets or {}
    out, cy_shift, cx_shift = {}, 0.0, 0.0
    for i, k in enumerate(keys):
        if i > 0:
            dy, dx = seam_offsets.get(k) or (0.0, 0.0)
            cy_shift += float(dy); cx_shift += float(dx)
        cy, cx = lines[k]
        out[k] = (np.asarray(cy, float) + cy_shift, np.asarray(cx, float) + cx_shift)
    return out


def fit_sections_for_plot(sections, labels=None):
    """Build :func:`plot_centroid_fit` section dicts with a per-section line fit.

    ``sections`` maps a label to a dict of RAW centroids (keys ``z``, ``cy``,
    ``cx`` and optional ``w``).  Each section gets its OWN straight-line fit
    (independent slope + offset) overlaid on the raw per-frame centroid, so the
    diagnostic shows what the per-section straight-line model would give
    regardless of which centroid mode actually built the waterfalls.  Returns a
    list of dicts ready for :func:`plot_centroid_fit`.
    """
    lines, _ = fit_centroid_lines(sections, shared_slope=False)
    labels = list(sections) if labels is None else labels
    out = []
    for lab in labels:
        s = sections[lab]
        cy_fit, cx_fit = lines[lab]
        out.append(dict(
            z=np.asarray(s["z"], float),
            cy_raw=np.asarray(s["cy"], float), cx_raw=np.asarray(s["cx"], float),
            cy_fit=cy_fit, cx_fit=cx_fit,
            w=(None if s.get("w") is None else np.asarray(s["w"], float)),
            label=lab))
    return out


def fit_sections_for_plot_continuous(sections, labels=None, order=None, knots=None):
    """Like :func:`fit_sections_for_plot`, but overlay the CONTINUOUS piecewise-
    linear fit (:func:`fit_centroid_lines_continuous`) instead of the independent
    per-section lines -- so the diagnostic shows the neighbouring lines meeting
    at each handoff (no jump).  ``z`` must be the GLOBAL scan coordinate.
    """
    lines, _ = fit_centroid_lines_continuous(sections, order=order, knots=knots)
    labels = list(sections) if labels is None else labels
    out = []
    for lab in labels:
        s = sections[lab]
        cy_fit, cx_fit = lines[lab]
        out.append(dict(
            z=np.asarray(s["z"], float),
            cy_raw=np.asarray(s["cy"], float), cx_raw=np.asarray(s["cx"], float),
            cy_fit=cy_fit, cx_fit=cx_fit,
            w=(None if s.get("w") is None else np.asarray(s["w"], float)),
            label=lab))
    return out


def _weighted_rms(resid, w):
    """Brightness-weighted RMS of a residual (falls back to plain RMS)."""
    resid = np.asarray(resid, float)
    w = np.clip(np.asarray(w, float), 0.0, None)
    if w.sum() <= 0:
        return float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
    return float(np.sqrt((w * resid ** 2).sum() / w.sum()))


def plot_centroid_fit(groups, *, path=None, title=None, xlabel="global z (mm)",
                      figsize=None, dpi=150, show=True):
    """Overlay the fitted straight-line centroid on the raw per-frame centroid.

    This is the diagnostic that shows *how the line lines up with the centroid
    determined on a frame-by-frame basis* -- the raw per-frame centroids are
    drawn as scatter points (point area proportional to the brightness weight
    each frame carried in the fit) and the fitted straight line as a solid curve
    of the same colour.

    Parameters
    ----------
    groups : ordered mapping ``group_label -> sections``
        One grid row per group.  ``sections`` is a list of dicts, each with
        1-D arrays ``z``, ``cy_raw``, ``cx_raw``, ``cy_fit``, ``cx_fit`` and,
        optionally, ``w`` (brightness weight -> point size) and ``label``
        (legend entry).  Left column is the row centroid ``y``, right column the
        column centroid ``x``.
    path : path-like or None
        If given, the figure is saved there (parent dirs created).
    title, xlabel : str
        Figure suptitle and shared x-axis label.
    show : bool
        Call ``plt.show()`` (notebook display) before returning.

    Returns the created ``Figure``.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    groups = dict(groups)
    if not groups:
        raise ValueError("plot_centroid_fit: no groups to plot")

    # Common point-size scale from the brightness weights of every section shown.
    all_w = np.concatenate([
        np.asarray(sec["w"], float)
        for secs in groups.values() for sec in secs if sec.get("w") is not None
    ]) if any(sec.get("w") is not None for secs in groups.values() for sec in secs) else None
    if all_w is not None and all_w.size and np.ptp(all_w) > 0:
        w_lo, w_span = float(all_w.min()), float(np.ptp(all_w))
        def _size(w):
            return 8.0 + 52.0 * (np.clip(np.asarray(w, float), w_lo, None) - w_lo) / w_span
    else:
        def _size(w):
            return np.full(np.asarray(w).shape, 22.0) if w is not None else 22.0

    nrows = len(groups)
    if figsize is None:
        figsize = (11.0, 2.7 * nrows + 0.8)
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, squeeze=False)

    for r, (gkey, secs) in enumerate(groups.items()):
        ax_y, ax_x = axes[r, 0], axes[r, 1]
        colours = plt.get_cmap("tab10")(np.arange(len(secs)) % 10)
        res_y, res_x, wres = [], [], []
        for sec, colour in zip(secs, colours):
            z = np.asarray(sec["z"], float)
            w = sec.get("w")
            sz = _size(w) if w is not None else 22.0
            lbl = sec.get("label")
            for ax, raw_key, fit_key, res in (
                (ax_y, "cy_raw", "cy_fit", res_y), (ax_x, "cx_raw", "cx_fit", res_x)):
                raw = np.asarray(sec[raw_key], float)
                fit = np.asarray(sec[fit_key], float)
                ax.scatter(z, raw, s=sz, color=colour, alpha=0.35,
                           edgecolors="none", zorder=2)
                order = np.argsort(z)
                ax.plot(z[order], fit[order], "-", color=colour, lw=1.8,
                        label=lbl, zorder=3)
                res.append(raw - fit)
            if w is not None:
                wres.append(np.asarray(w, float))

        w_cat = np.concatenate(wres) if wres else None
        for ax, res, name in ((ax_y, res_y, "row $y$"), (ax_x, res_x, "col $x$")):
            resid = np.concatenate(res)
            rms = _weighted_rms(resid, w_cat if w_cat is not None else np.ones_like(resid))
            ax.set_ylabel(f"centroid {name} (px)")
            ax.grid(alpha=0.3)
            ax.text(0.98, 0.03, f"wRMS resid = {rms:.2f} px", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
            if r == 0:
                ax.set_title({"row $y$": "row centroid $y$",
                              "col $x$": "column centroid $x$"}[name], fontsize=10)
        ax_y.annotate(str(gkey), xy=(0, 0.5), xytext=(-ax_y.yaxis.labelpad - 6, 0),
                      xycoords=ax_y.yaxis.label, textcoords="offset points",
                      ha="right", va="center", rotation=90, fontweight="bold")
        if any(sec.get("label") for sec in secs):
            ax_y.legend(fontsize=7, ncol=max(1, len(secs) // 3))

    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig
