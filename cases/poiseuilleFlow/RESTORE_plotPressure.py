from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Physical parameters
Re = 10.0
D = 20.0          # Pipe diameter [mm]
nu = 1.5e-5       # Kinematic viscosity [m²/s]
rho = 1.225       # Density [kg/m³]
z_in = -10.0      # Inlet coordinate [mm]
z_out = 10.0      # Outlet coordinate [mm]
p_out = 0.0       # Outlet gauge pressure [Pa]

# Your datasets are currently in results/RESTORE.
RESULTS = Path(__file__).resolve().parent / "results" / "RESTORE"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 11,
})

cases = [
    ("RESTORE_OLD", "Second order raw moment reconstruction", "teal", "-"),
    ("RESTORE_NEW", "Third order central moment reconstruction", "purple", "-"),
]

# Analytical solution: convert diameter to metres, then Pa/m to Pa/mm.
D_m = D * 1e-3
pressure_gradient = 32.0 * rho * nu**2 * Re / D_m**3 * 1e-3

z_analytic = np.linspace(z_in, z_out, 500)
p_analytic = p_out + pressure_gradient * (z_out - z_analytic)

for derivative in (False, True):
    fig, ax = plt.subplots(figsize=(7, 4))

    for filename, label, color, linestyle in cases:
        zCoord, pressure = np.loadtxt(
            RESULTS / f"{filename}_centerlinePressure.txt",
            delimiter=";",
            skiprows=1,
            unpack=True,
        )

        values = (
            -np.gradient(pressure, zCoord, edge_order=1)
            if derivative else pressure
        )

        ax.plot(
            zCoord, values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
        )

    ax.plot(
        z_analytic,
        np.full_like(z_analytic, pressure_gradient)
        if derivative else p_analytic,
        label="Analytic solution",
        color="black",
        linestyle="dotted",
        linewidth=2.5,
    )

    ax.set_xlim(z_in, z_out)
    ax.set_xlabel(r"$z$ [mm]")
    ax.set_ylabel(
        r"$-\mathrm{d}p/\mathrm{d}z$ [Pa/mm]"
        if derivative else r"$p$ [Pa]"
    )
    ax.set_title(
        "Pressure derivative vs Z coordinate"
        if derivative else "Pressure vs Z coordinate"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    ax.margins(x=0)

    fig.tight_layout()
    filename = (
        "RESTORE_pressureDerivative.png"
        if derivative else "RESTORE_pressure.png"
    )
    fig.savefig(RESULTS / filename, dpi=300)
    plt.close(fig)
