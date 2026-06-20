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

    return Path, gaussian_filter, mo, np, plt, re, tifffile, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal Focusing of OAM Beams
    """)
    return


@app.cell
def _(Path, mo, np):
    ROOT = Path("/Users/carriecrane/GitHub/Optimal_Focusing/data/focusing_images/29_n15")
    z_vals = np.arange(0,25.5, 0.5)

    BEAM_SIZE = 29
    N_FOCUS = 15

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
def _(gaussian_filter, np, tifffile):
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

    plot_beam_width()
    return


@app.cell
def _(all_results, np, z_vals):
    # CELL 6.6

    for l_val in range(6):

        res = all_results[l_val]

        width = np.array([
            r["width"] for r in res
        ])

        valid = ~np.isnan(width)

        if valid.sum() == 0:
            continue

        width_valid = width[valid]
        z_valid = z_vals[:len(width)][valid]

        best_idx = np.argmin(width_valid)

        print(
            f"ℓ={l_val}: "
            f"minimum width = {width_valid[best_idx]:.2f} px "
            f"at z = {z_valid[best_idx]:.2f} mm"
        )
    return


@app.cell
def _(all_results, np, plt, width_arr, z_vals):
    # CELL 6.65

    plt.figure(figsize=(8,6))

    for l_mode3 in range(6):

        results_l3 = all_results[l_mode3]

        width_arr3 = np.array([
            r["width"] for r in results_l3
        ])

        valid_mask2 = ~np.isnan(width_arr3)

        z_plot2 = z_vals[:len(width_arr)][valid_mask2]
        w_plot2 = width_arr[valid_mask2]

        plt.plot(
            z_plot2,
            +w_plot2,
            lw=2,
            label=f"ℓ={l_mode3}"
        )

        plt.plot(
            z_plot2,
            -w_plot2,
            lw=2
        )

    plt.axhline(0, color='k', lw=0.5)

    plt.xlabel("Propagation distance z (mm)")
    plt.ylabel("Beam radius (px)")
    plt.title("Beam Envelope (Unnormalized)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(all_results, np, plt, z_vals):
    # CELL 6.7

    plt.figure(figsize=(7,5))

    for l_mode in range(6):

        results_l2 = all_results[l_mode]

        width_arr = np.array([
            r["width"] for r in results_l2
        ])

        valid_mask = ~np.isnan(width_arr)

        z_plot = z_vals[:len(width_arr)][valid_mask]
        w_plot = width_arr[valid_mask]

        w_plot = w_plot / w_plot[0]

        plt.plot(
            z_plot,
            +w_plot,
            lw=2,
            label=f"ℓ={l_mode}"
        )

        plt.plot(
            z_plot,
            -w_plot,
            lw=2
        )

    plt.axhline(0, color='k', lw=0.5)

    plt.xlabel("Propagation Distance z (mm)")
    plt.ylabel("Normalized Beam Radius")
    plt.title("Beam Convergence")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return (width_arr,)


@app.cell
def _(all_results, np, plt, z_vals):
    # CELL 6.8

    focus_positions = []

    for l_mode2 in range(6):

        width_arr2 = np.array([
            r["width"]
            for r in all_results[l_mode2]
        ])

        idx_min = np.nanargmin(width_arr2)

        focus_positions.append(
            z_vals[idx_min]
        )

    plt.figure(figsize=(5,4))

    plt.plot(
        range(6),
        focus_positions,
        'o-',
        lw=2
    )

    plt.xlabel("OAM charge ℓ")
    plt.ylabel("Focus position (mm)")
    plt.title("Focus position vs OAM")

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(all_results, find_crop_center, np, plt, z_vals):
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

            # Beam center
            cx, cy = find_crop_center(img)

            x0 = int(round(cx))
            y0 = int(round(cy))

            # Crop around beam center
            left  = max(0, x0 - half_width)
            right = min(img.shape[1], x0 + half_width)

            # Average several rows to reduce noise
            row = np.mean(
                img[max(0, y0-3):min(img.shape[0], y0+4),
                    left:right],
                axis=0
            )

            rows.append(row)

        xz_map = np.array(rows)

        # Normalize each z slice independently
        xz_map = xz_map / (
            np.max(xz_map, axis=1, keepdims=True) + 1e-12
        )

        return xz_map

    xz_map = make_longitudinal_map(0)

    def plot_longitudinal_map(l_mode):

        xz_map = make_longitudinal_map(l_mode)

        vmax = np.percentile(xz_map, 99)

        plt.figure(figsize=(8,6))

        plt.imshow(
            xz_map,
            aspect='auto',
            origin='lower',
            cmap='inferno',
            vmax=vmax,
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

    plot_longitudinal_map(0)
    plot_longitudinal_map(1)
    plot_longitudinal_map(2)

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
                cmap='inferno',
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

    return


@app.cell
def _(all_results, find_crop_center, np):
    # CELL 6.11

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

    return (make_radial_map,)


@app.cell
def _(make_radial_map, np, plt, z_vals):
    # CELL 6.12

    def plot_radial_map(l_mode):

        radial_map = make_radial_map(l_mode)

        vmax = np.percentile(radial_map, 99)

        plt.figure(figsize=(8,6))

        plt.imshow(
            radial_map,
            aspect='auto',
            origin='lower',
            cmap='inferno',
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

    return (plot_radial_map,)


@app.cell
def _(plot_radial_map):
    plot_radial_map(0)
    plot_radial_map(1)
    plot_radial_map(2)
    return


@app.cell
def _(make_radial_map, np, plt, z_vals):
    # CELL 6.13

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
                cmap='inferno',
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

    plot_all_radial_maps()
    return


@app.cell
def _(all_results, find_crop_center, np, plt, z_vals):
    # CELL 6.15

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

    plot_all_beam_centers()
    return


@app.cell
def _(all_results, z_vals):
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

    plot_all_peak_intensities_normalized(
        save_dir="/Users/carriecrane/GitHub/Optimal_Focusing/analysis/figures"
    )
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
                        cmap="inferno",
                        origin="upper",
                        norm=LogNorm()
                    )

                else:

                    ax.imshow(
                        crop,
                        cmap="inferno",
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

            plt.close()

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
        return mo.callout(
            mo.md("Linear and log beam grids saved."),
            kind="success"
        )


    plot_focusing_images()
    return


if __name__ == "__main__":
    app.run()
