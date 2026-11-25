from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Corresponding entries describe the same case.
fileList = [
    "dirichlet.txt",
    "characteristic.txt",
]

labelList = [
    "Strict Dirichlet baseline",
    "Characteristic",]
    #r"Impedance, $\delta\rho_{\mathrm{tol}}=10^{-4}$",
    #r"Impedance, $\delta\rho_{\mathrm{tol}}=10^{-5}$",    
#]

colorList = ["black", "purple"]
lineStyleList = ["-", "--"]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

FIGSIZE = (7, 4)
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
        "inletVelocity.png",
    ),
    (
        2,
        r"$p$ [Pa]",
        "Area-averaged inlet pressure",
        "inletPressure.png",
    ),
    (
        3,
        r"$E_{\mathrm{inlet}}$ [-]",
        "Relative L2 inlet velocity error",
        "inletProfileError.png",
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
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=DPI)
    plt.close(fig)
