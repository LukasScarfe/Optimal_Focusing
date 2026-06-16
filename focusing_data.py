import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np 
    import tifffile
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.ndimage import gaussian_filter
    from scipy.signal import find_peaks
    from pathlib import Path
    import io, re, time

    return (
        Path,
        curve_fit,
        find_peaks,
        gaussian_filter,
        mo,
        np,
        plt,
        re,
        tifffile,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal Focusing of OAM Beams
    """)
    return


@app.cell
def _(Path, mo, np):
    ROOT = Path("/Users/carriecrane/GitHub/Optimal_Focusing/data/focusing_images/29_n14_pt2")
    z_vals = np.arange(0,25.5, 0.5)

    BEAM_SIZE = 29
    N_FOCUS = 14

    blur = mo.ui.slider(0 , 3 , step = 0.5, value = 1.0, label = "Pre blur σ (px)")
    mo.vstack([mo.callout(mo.md(f"**Current dataset:** beam size = {BEAM_SIZE}, n = {N_FOCUS}"), kind = "info"),blur])
    return BEAM_SIZE, N_FOCUS, ROOT, z_vals


@app.cell
def _(ROOT, re):
    ## CELL 3 ##
    def get_files(l_val):
        folder = ROOT / f"l={l_val}"
        files = sorted(folder.glob("*.tif"), key = lambda f: int(re.findall(r'\d+',f.stem)[-1]))
        return files

    for i in range(6):
        files = get_files(i)
        print(f"l={i}: {len(files)} files - first: {files[0].name}, last:{files[-1].name}")
    return (get_files,)


@app.cell
def _(curve_fit, find_peaks, gaussian_filter, np, tifffile):
    ## CELL 4 ##
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
        """
        Find innermost ring peak. If prev_r is given, search near it first
        for continuity across frames.
        """
        smooth = gaussian_filter(profile.astype(float), sigma=3)

        # If we have a previous radius, search in a window around it first
        if prev_r is not None and not np.isnan(prev_r):
            window = 30  # px either side
            lo = max(20, int(prev_r - window))
            hi = min(len(smooth), int(prev_r + window))
            local_peaks, _ = find_peaks(
                smooth[lo:hi],
                height=img_max * 0.03,
                distance=5
            )
            if len(local_peaks) > 0:
                # Pick peak closest to prev_r
                local_peaks += lo
                closest = local_peaks[np.argmin(np.abs(local_peaks - prev_r))]
                r_peak = float(closest)
                # Fit Gaussian around it
                half_win = 25
                r_lo = max(0, int(r_peak) - half_win)
                r_hi = min(len(profile), int(r_peak) + half_win)
                r_w  = np.arange(r_lo, r_hi, dtype=float)
                I_w  = profile[r_lo:r_hi].astype(float)
                def gauss1d(r, amp, r0, sigma, bg):
                    return bg + amp * np.exp(-((r - r0)**2) / (2 * sigma**2))
                try:
                    popt, _ = curve_fit(gauss1d, r_w, I_w,
                                        p0=[I_w.max(), r_peak, 8, I_w.min()],
                                        maxfev=5000)
                    amp, r0, sigma, bg = popt
                    return abs(r0), amp + bg, abs(sigma)
                except RuntimeError:
                    return r_peak, float(profile[int(r_peak)]), np.nan

        # No previous radius — global search, skip first 20px
        peaks, _ = find_peaks(
            smooth[20:],
            height=img_max * 0.05,
            distance=10
        )
        peaks += 20

        if len(peaks) == 0:
            return np.nan, np.nan, np.nan

        r_peak = float(peaks[0])
        half_win = 25
        r_lo = max(0, int(r_peak) - half_win)
        r_hi = min(len(profile), int(r_peak) + half_win)
        r_w  = np.arange(r_lo, r_hi, dtype=float)
        I_w  = profile[r_lo:r_hi].astype(float)

        def gauss1d(r, amp, r0, sigma, bg):
            return bg + amp * np.exp(-((r - r0)**2) / (2 * sigma**2))
        try:
            popt, _ = curve_fit(gauss1d, r_w, I_w,
                                p0=[I_w.max(), r_peak, 8, I_w.min()],
                                maxfev=5000)
            amp, r0, sigma, bg = popt
            return abs(r0), amp + bg, abs(sigma)
        except RuntimeError:
            return r_peak, float(profile[int(r_peak)]), np.nan


    def process_frame(path, l_val, prev_r=None):
        raw = tifffile.imread(path).astype(float)
        if raw.ndim == 3:
            raw = raw.mean(axis=0)

        bg  = np.median([raw[:20,:20], raw[:20,-20:],
                         raw[-20:,:20], raw[-20:,-20:]])
        img = np.clip(raw - bg, 0, None)

        if l_val == 0:
            blurred = gaussian_filter(img, sigma=2)
            yc, xc  = np.unravel_index(np.argmax(blurred), blurred.shape)
            cx, cy  = float(xc), float(yc)
            peak    = float(img[int(cy), int(cx)])
            return dict(name=path.name, image=raw, profile=np.zeros(300),
                        cx=cx, cy=cy, ring_r=np.nan, ring_w=np.nan, peak=peak)

        cx, cy, profile = find_center_and_profile(img)

        # Try with prev_r first, then fall back to looser global search
        ring_r, peak, ring_w = get_innermost_ring(profile, img.max(), prev_r)

        if np.isnan(ring_r):
            # Retry with lower threshold — transition zone has fainter rings
            ring_r, peak, ring_w = get_innermost_ring(profile, img.max() * 0.3, None)

        return dict(name=path.name, image=raw, profile=profile,
                    cx=cx, cy=cy, ring_r=ring_r, ring_w=ring_w, peak=peak)

    return (process_frame,)


@app.cell
def _(get_files, mo, np, process_frame, time):
    ## CELL 5 ##
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
            print(f"l={ls_val}  frame {i_frame+1}/{len(files_l)}  ({elapsed:.0f}s)  ring_r={r['ring_r']:.1f}px  peak={r['peak']:.1f}")
        all_results[ls_val] = results_l

    mo.callout(mo.md("All frames processed."), kind="success")
    return (all_results,)


@app.cell
def _(all_results, np, plt, z_vals):
    ## CELL 6 ##
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

    plot_ring_radius()
    return


@app.cell
def _(all_results, np, plt, z_vals):
    ## CELL 7 ##
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
        plt.savefig("/Users/carriecrane/GitHub/Optimal_Focusing/peak_intensity_vs_z.png", dpi=130)
        plt.close()
        print("saved peak_intensity_vs_z.png")

    plot_peak_intensity()
    return


@app.cell
def _(BEAM_SIZE, N_FOCUS, all_results, gaussian_filter, mo, np, plt, z_vals):
    ## CELL 8 ##
    def find_crop_center(img):
        """Find beam center by taking weighted centroid of top 5% brightest pixels."""
        blurred   = gaussian_filter(img, sigma=3)
        threshold = np.percentile(blurred, 95)
        bright    = blurred >= threshold
        Y, X      = np.mgrid[:img.shape[0], :img.shape[1]]
        cx = float(np.average(X[bright], weights=blurred[bright]))
        cy = float(np.average(Y[bright], weights=blurred[bright]))
        return cx, cy

    def fourier_clean(img, r_cut=100):
        """Suppress high spatial frequency noise via low-pass Fourier filter."""
        from numpy.fft import fft2, ifft2, fftshift, ifftshift
        ny, nx = img.shape
        Y, X   = np.ogrid[:ny, :nx]
        R_f    = np.sqrt((X - nx//2)**2 + (Y - ny//2)**2)
        lpf    = np.exp(-(R_f / r_cut)**4)
        F_filt = fftshift(fft2(img)) * lpf
        out    = np.real(ifft2(ifftshift(F_filt)))
        return np.clip(out, 0, None)

    def plot_beam_grid(l_val, step=5, bg_scale=0.3):
        res      = all_results[l_val]
        indices  = list(range(0, len(res), step))
        ncols    = len(indices)
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3.5))

        # Load background frame (first frame of this l value)
        raw_bg  = res[0]["image"].astype(float)
        bg_val  = np.median([raw_bg[:20,:20], raw_bg[:20,-20:],
                             raw_bg[-20:,:20], raw_bg[-20:,-20:]])
        img_bg  = np.clip(raw_bg - bg_val, 0, None)
        img_bg_norm = img_bg / (img_bg.max() + 1e-6)

        # Compute anchor center from first frame
        blurred0   = gaussian_filter(img_bg, sigma=3)
        thresh0    = np.percentile(blurred0, 95)
        bright0    = blurred0 >= thresh0
        Y0, X0     = np.mgrid[:blurred0.shape[0], :blurred0.shape[1]]
        anchor_cx  = float(np.average(X0[bright0], weights=blurred0[bright0]))
        anchor_cy  = float(np.average(Y0[bright0], weights=blurred0[bright0]))

        half = 350

        # ---- CHANGE 1: compute global vmax across all frames before the loop ----
        global_max = max(r["image"].astype(float).max() for r in res)
        vmax = global_max * 0.95
        # ------------------------------------------------------------------------

        for ax, idx in zip(axes, indices):
            r   = res[idx]
            img = r["image"].astype(float)
            bg  = np.median([img[:20,:20], img[:20,-20:],
                             img[-20:,:20], img[-20:,-20:]])
            img = np.clip(img - bg, 0, None)
            img = fourier_clean(img, r_cut=100)

            blurred   = gaussian_filter(img, sigma=3)
            threshold = np.percentile(blurred, 95)
            bright    = blurred >= threshold
            Y, X      = np.mgrid[:img.shape[0], :img.shape[1]]
            dist_from_anchor = np.sqrt((X - anchor_cx)**2 + (Y - anchor_cy)**2)
            bright = bright & (dist_from_anchor < 150)

            if bright.sum() > 10:
                cx = float(np.average(X[bright], weights=blurred[bright]))
                cy = float(np.average(Y[bright], weights=blurred[bright]))
            else:
                cx, cy = anchor_cx, anchor_cy

            ny, nx = img.shape
            y1 = int(np.clip(cy - half, 0, ny - 2*half))
            x1 = int(np.clip(cx - half, 0, nx - 2*half))
            crop = img[y1:y1+2*half, x1:x1+2*half]

            crop_cx = cx - x1
            crop_cy = cy - y1
            Y_c, X_c = np.ogrid[:crop.shape[0], :crop.shape[1]]
            R_c = np.sqrt((X_c - crop_cx)**2 + (Y_c - crop_cy)**2)
            mask_r   = 20
            softness = 5
            soft_mask = 1 / (1 + np.exp(-(R_c - mask_r) / softness))
            crop = crop * soft_mask

            # ---- CHANGE 2: use global vmax instead of per-frame ----
            ax.imshow(crop, cmap='inferno', origin='upper', vmax=vmax, vmin=0)
            # --------------------------------------------------------

            ax.set_title(f'z={z_vals[idx]:.1f}mm', fontsize=8)
            ax.axis('off')

        plt.suptitle(f'ℓ={l_val}  beam={BEAM_SIZE} n={N_FOCUS} — focusing sequence', y=1.02)
        plt.tight_layout()
        plt.savefig(f"/Users/carriecrane/GitHub/Optimal_Focusing/analysis/29_n14_pt2/beam_grid_l{l_val}.png", dpi=130)
        plt.close()
        print(f"saved beam_grid_l{l_val}.png")

    for l_val3 in range(6):
        plot_beam_grid(l_val3, step=5, bg_scale=0.3)

    mo.callout(mo.md("Beam grids saved."), kind="success")
    return


if __name__ == "__main__":
    app.run()
