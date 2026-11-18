from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 12,
})

cases = [
    ("_",      "Fine reference",   "black",   "-"),
]

for derivative in (False, True):
    fig, ax = plt.subplots(figsize=(7, 4))

    for filename, label, color, linestyle in cases:
        zCoord, pressure = np.loadtxt(
            RESULTS / f"centerlinePressure.txt",
            delimiter=";",
            skiprows=1,
            unpack=True,
        )

        values = -np.gradient(pressure, zCoord, edge_order=1) if derivative else pressure

        ax.plot(
            zCoord, values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
        )

    ax.set_xlabel(r"$z$ [mm]")
    ax.set_ylabel(
        r"$-\mathrm{d}p/\mathrm{d}z$ [Pa/mm]"
        if derivative else r"$p$ [Pa]"
    )
    #ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    ax.margins(x=0)
    ax.set_xlim(7, 10)
    ax.set_ylim(0.0000, 0.0005)
    ax.set_title("Pressure vs Z coordinate close to the outlet")

    fig.tight_layout()
    filename = "pressureDerivativeSingle.png" if derivative else "pressureSingle.png"
    fig.savefig(RESULTS / filename, dpi=600)
    plt.close(fig)
