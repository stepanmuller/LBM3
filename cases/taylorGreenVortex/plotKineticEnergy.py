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
        time, energy = np.loadtxt(
            RESULTS / f"{filename}_kineticEnergy.txt",
            delimiter=";",
            skiprows=1,
            unpack=True,
        )

        values = -np.gradient(energy, time, edge_order=1) if derivative else energy

        ax.plot(
            time, values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
        )

    ax.set_xlabel(r"$t^*$")
    ax.set_ylabel(
        r"$-\mathrm{d}E_k^*/\mathrm{d}t^*$"
        if derivative else r"$E_k^*$"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    ax.margins(x=0)

    fig.tight_layout()
    filename = "kineticEnergyDerivative.png" if derivative else "kineticEnergy.png"
    fig.savefig(RESULTS / filename, dpi=600)
    plt.close(fig)
    
fine_time, fine_energy = np.loadtxt(
    RESULTS / "FINE_kineticEnergy.txt",
    delimiter=";",
    skiprows=1,
    unpack=True,
)

fig, ax = plt.subplots(figsize=(7, 4))

for filename, label, color, linestyle in cases:
    time, energy = np.loadtxt(
        RESULTS / f"{filename}_kineticEnergy.txt",
        delimiter=";",
        skiprows=1,
        unpack=True,
    )

    # Match the fine reference to this case's sampling times.
    error = energy - np.interp(time, fine_time, fine_energy)

    ax.plot(
        time, error,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1.5,
    )

ax.set_xlabel(r"$t^*$")
ax.set_ylabel(r"$E_k^* - E_{k,\mathrm{fine}}^*$")
ax.legend(loc="lower left")
ax.grid(alpha=0.2)
ax.margins(x=0)
ax.set_ylim( -0.0035, 0.0005 )

fig.tight_layout()
fig.savefig(RESULTS / "kineticEnergyError.png", dpi=600)
plt.close(fig)
