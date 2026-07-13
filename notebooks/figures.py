import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import focusing_library as fl

    return fl, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## This notebook pull from focusing_data.py to create necessary figures for analysis
    """)
    return


@app.cell
def _(fl):
    fl.plot_beam_width()
    return


@app.cell
def _(fl):
    fl.plot_peak_intensity()
    return


@app.cell
def _(fl):
    fl.plot_all_radial_maps()
    return


@app.cell
def _(fl):
    fl.plot_all_xz_maps()
    return


@app.cell
def _(fl):
    fl.make_prl_style_figure(3)
    return


@app.cell
def _(fl):
    sim_data = fl.build_all_simulations(n0 = 15, Ntau = 80, Nrho_out = 300, Nx = 900, force_recompute = False)
    fl.plot_all_xz_maps_with_simulation(sim_data)
    return (sim_data,)


@app.cell
def _(np, plt, sim_data):

    for l in [0,1,2,5]:

        sim_z, sim_x, sim_xz = sim_data[l]

        focus_idx = np.argmax(np.sum(sim_xz, axis=1))

        plt.figure()
        plt.plot(sim_x, sim_xz[focus_idx])
        plt.title(f"l={l} profile at focus")
        plt.xlabel("x")
        plt.ylabel("Intensity")
        plt.show()
    return (sim_xz,)


@app.cell
def _(fl):
    fl.plot_all_centered_xz_maps()
    return


@app.cell
def _(fl):
    fl.plot_all_xz_maps_clean()
    return


@app.cell
def _(fl):
    fl.plot_all_peak_intensities_normalized()
    return


@app.cell
def _(all_results, fl, np, plt):
    l_mode = 5
    xz = fl.make_centered_xz_map(l_mode)

    focus_idx2 = np.argmax(np.sum(xz, axis=1))

    exp_profile = xz[focus_idx2]
    exp_profile /= exp_profile.max()

    result = all_results[l_mode][focus_idx2]

    img = result["image"].astype(float)

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="inferno")
    plt.colorbar()
    plt.show()
    return exp_profile, l_mode


@app.cell
def _(np, sim_xz):

    focus_idx_sim = np.argmax(np.sum(sim_xz, axis=1))

    sim_profile = sim_xz[focus_idx_sim]
    sim_profile = sim_profile / sim_profile.max()
    return (sim_profile,)


@app.cell
def _(exp_profile, l_mode, plt, sim_profile):


    plt.figure(figsize=(8,4))

    plt.plot(exp_profile, label="Experiment")
    plt.plot(sim_profile, label="Simulation")

    plt.legend()
    plt.title(f"l={l_mode}")
    plt.xlabel("Pixel index")
    plt.ylabel("Normalized intensity")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(fl):
    fl.plot_focusing_images()
    return


if __name__ == "__main__":
    app.run()
