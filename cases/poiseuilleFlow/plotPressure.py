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
    ("FINE",      "Fine reference",   "black",   "-"),
    ("COARSE",    "Coarse reference", "0.55",    "-"),
    ("QUADRATIC", "Quadratic",        "purple", "--"),
    ("LINEAR",    "Linear",           "teal",   "--"),
    ("ITERATIVE", "Iterative",        "#e67e22", ":"),
]

for derivative in (False, True):
    fig, ax = plt.subplots(figsize=(7, 4))

    for filename, label, color, linestyle in cases:
        zCoord, pressure = np.loadtxt(
            RESULTS / f"{filename}_centerlinePressure.txt",
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
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    ax.margins(x=0)

    fig.tight_layout()
    filename = "pressureDerivative.png" if derivative else "pressure.png"
    fig.savefig(RESULTS / filename, dpi=600)
    plt.close(fig)
    
fine_zCoord, fine_pressure = np.loadtxt(
    RESULTS / "FINE_centerlinePressure.txt",
    delimiter=";",
    skiprows=1,
    unpack=True,
)

fig, ax = plt.subplots(figsize=(7, 4))

for filename, label, color, linestyle in cases:
    zCoord, pressure = np.loadtxt(
        RESULTS / f"{filename}_centerlinePressure.txt",
        delimiter=";",
        skiprows=1,
        unpack=True,
    )

    # Match the fine reference to this case's sampling zCoords.
    error = pressure - np.interp(zCoord, fine_zCoord, fine_pressure)

    ax.plot(
        zCoord, error,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1.5,
    )

ax.set_xlabel(r"$z$ [mm]")
ax.set_ylabel(r"$p - p_{fine}$ [Pa]")
ax.legend(loc="upper right")
ax.grid(alpha=0.2)
ax.margins(x=0)
ax.set_ylim( -0.0002, 0.0010 )

fig.tight_layout()
fig.savefig(RESULTS / "pressureError.png", dpi=600)
plt.close(fig)
