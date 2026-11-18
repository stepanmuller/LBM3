from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add matching values to these three lists to include K17.
K17_lim005_velocity = [6.18917e-05, 0.002]
K17_lim005_pressure_loss = [2.9255, 2.6963]
K17_lim005_density_error = [9.9813e-6, 0.0094011]

# Add your values below. Each dataset's three lists must have matching lengths.
# Leave all three lists empty to omit that dataset from both plots.
K17_lim001_velocity = [6.18917e-05, 0.002]
K17_lim001_pressure_loss = [2.9339, 2.7089]
K17_lim001_density_error = [1.0012e-5, 0.009445]

K17_lim001_2xres_velocity = [0.002]
K17_lim001_2xres_pressure_loss = [2.7692]
K17_lim001_2xres_density_error = [0.0096497]

K17_lim001_4xres_velocity = [0.002]
K17_lim001_4xres_pressure_loss = [2.8372]
K17_lim001_4xres_density_error = [0.0098843]

folder = Path(__file__).resolve().parent
K15 = np.loadtxt(folder / "LBM_velocity_results.txt", delimiter=";", skiprows=1, ndmin=2)
K15 = K15[np.argsort(K15[:, 0])].T
datasets = [
    ("K15", K15, "black", "+"),
    ("K17, lim 0.05", [K17_lim005_velocity, K17_lim005_pressure_loss, K17_lim005_density_error], "teal", "X"),
    # Edit each tuple's label, colour, and marker here.
    ("K17, lim 0.01", [K17_lim001_velocity, K17_lim001_pressure_loss, K17_lim001_density_error], "teal", "x"),
    ("K17, lim 0.01, 2x res", [K17_lim001_2xres_velocity, K17_lim001_2xres_pressure_loss, K17_lim001_2xres_density_error], "orange", "x"),
    ("K17, lim 0.01, 4x res", [K17_lim001_4xres_velocity, K17_lim001_4xres_pressure_loss, K17_lim001_4xres_density_error], "purple", "x"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})

plots = [
    (1, r"$\mathrm{Area\ averaged\ pressure\ loss}\ [\mathrm{Pa}]$", "pressure_loss.png"),
    (2, r"$\mathrm{LBM\ weakly\ compressible}$" + "\n"
     + r"$\mathrm{density\ error}\ [1]$", "density_error.png"),
]
for column, ylabel, filename in plots:
    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    for label, data, color, marker in datasets:
        if len(data[0]):
            ax.plot(
                data[0], data[column], linestyle="none", marker=marker,
                color=color, markersize=6, markeredgewidth=1.3, label=label,
            )
    ax.set_xlabel(r"$\mathrm{LBM\ inlet\ velocity}\ [1]$")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.minorticks_on()
    ax.grid(True, which="major", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.05, y=0.08)
    ax.legend()
    ax.set_ylim(0.0, 0.0015)
    ax.set_xlim(0.0, 6e-4)
    fig.savefig(folder / filename, facecolor="white", bbox_inches="tight")

plt.show()
plt.close("all")
