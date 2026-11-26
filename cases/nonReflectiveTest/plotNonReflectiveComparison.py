from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Corresponding entries describe the same case.
fileList = [
    "dirichlet.txt",
    "impedance2e-5.txt",
    "impedance2e-6.txt",
    "impedance2e-7.txt",
    "characteristic.txt",
]

labelList = [
    "Strict Dirichlet baseline",
    r"Impedance, $\delta\rho_{\mathrm{tol}}=2 \cdot 10^{-5}$",
    r"Impedance, $\delta\rho_{\mathrm{tol}}=2 \cdot 10^{-6}$",
    r"Impedance, $\delta\rho_{\mathrm{tol}}=2 \cdot 10^{-7}$",
    "Characteristic",  
]

colorList = ["black", "red", "orange", "purple", "teal"]
lineStyleList = ["solid", "dashed", "dashed", "dashed", "dashdot"]

# Detail-plot Y limits: (minimum, maximum).
detailYLimits = {
    1: (0.74, 0.76),   # Inlet velocity [m/s], using your positive inflow convention
    2: (-0.3, 0.3),    # Inlet pressure [Pa]
    3: (0.0, 0.00008),    # Relative L2 error [-]
}

excludeFromDetailList = [
    True,  # Strict Dirichlet baseline
    True,  # Impedance 1e-5
    False,  # Impedance 1e-6
    True,  # Impedance 1e-7
    False,  # Characteristic
]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

FIGSIZE = (8, 5)
LINEWIDTH = 2.5
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 11,
})

# Columns: iteration, mean inlet velocity, mean inlet pressure, relative L2 error.
# Load each file once.
data = [
    np.loadtxt(BASE_DIR / filename, delimiter=";", skiprows=1, ndmin=2)
    for filename in fileList
]

# reverse sign of the inlet velocity to show -u_n which is positive
for arr in data:
    arr[:, 1] *= -1

plots = [
    (
        1,
        r"$-u_n$ [m/s]",
        "Area-averaged inlet velocity",
        "NRBC_inletVelocity.png",
    ),
    (
        2,
        r"$p$ [Pa]",
        "Area-averaged inlet pressure",
        "NRBC_inletPressure.png",
    ),
    (
        3,
        r"$E_{\mathrm{inlet}}$ [-]",
        "Relative L2 inlet velocity error",
        "NRBC_inletProfileError.png",
    ),
]

for column, ylabel, title, filename in plots:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for values, label, color, linestyle in zip(
        data, labelList, colorList, lineStyleList
    ):
        ax.plot(
            values[:, 0], values[:, column],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=LINEWIDTH,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.margins(x=0)
    ax.legend(loc="best")
    # Export the full plot.
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=DPI)

    # Remove excluded curves after saving the full plot.
    for line, exclude in zip(list(ax.lines), excludeFromDetailList):
        if exclude:
            line.remove()

    # Rebuild the legend using the remaining curves.
    ax.legend(loc="best")

    # Export the detail plot.
    ax.set_ylim(*detailYLimits[column])
    ax.set_title(title + " (detail)")
    fig.tight_layout()

    detailFilename = f"{Path(filename).stem}_detail.png"
    fig.savefig(OUTPUT_DIR / detailFilename, dpi=DPI)

    plt.close(fig)
