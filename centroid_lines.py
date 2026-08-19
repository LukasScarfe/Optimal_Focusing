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


def fit_centroid_lines(sections):
    """Fit straight centroid lines for every section of one run (shared slope).

    ``sections`` maps a section id to a mapping with keys ``z``, ``cy``, ``cx``,
    ``w`` (1-D arrays of equal length: scan coordinate, raw row centroid, raw
    column centroid, per-frame brightness weight).

    Returns ``(lines, slopes)`` where ``lines[key] = (cy_fit, cx_fit)`` are the
    fitted centroid arrays to use in place of the raw per-frame centroids, and
    ``slopes = (slope_y, slope_x)`` are the run-wide shared slopes (px per
    z-unit).
    """
    z = {k: np.asarray(s["z"], float) for k, s in sections.items()}
    cy = {k: np.asarray(s["cy"], float) for k, s in sections.items()}
    cx = {k: np.asarray(s["cx"], float) for k, s in sections.items()}
    w = {k: np.asarray(s["w"], float) for k, s in sections.items()}

    slope_y, off_y = fit_shared_slope(z, cy, w)
    slope_x, off_x = fit_shared_slope(z, cx, w)

    lines = {k: (slope_y * z[k] + off_y[k], slope_x * z[k] + off_x[k]) for k in sections}
    return lines, (slope_y, slope_x)
