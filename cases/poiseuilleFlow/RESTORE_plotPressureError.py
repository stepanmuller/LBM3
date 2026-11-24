from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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
    ("RESTORE_NEW", "Third order central moment reconstruction", "purple", "dashed"),
]


def load_pressure(filename):
    z, pressure = np.loadtxt(
        RESULTS / f"{filename}_centerlinePressure.txt",
        delimiter=";",
        skiprows=1,
        unpack=True,
    )
    order = np.argsort(z)
    return z[order], pressure[order]


# Define one reference line for both datasets:
# p_ref(z_first) = p_NEW(z_first), p_ref(z_last) = 0.
z_new, p_new = load_pressure("RESTORE_NEW")
z_first = z_new[0]
z_last = z_new[-1]
p_first = p_new[0]


def reference_pressure(z):
    return p_first * (z_last - z) / (z_last - z_first)


fig, ax = plt.subplots(figsize=(7, 4))

for filename, label, color, linestyle in cases:
    z, pressure = load_pressure(filename)
    error = pressure - reference_pressure(z)

    ax.plot(
        z,
        error,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=2.5,
    )

ax.set_xlim(5.0, 10.0)
ax.set_xlabel(r"$z$ [mm]")
ax.set_ylabel(r"$p-p_{\mathrm{ref}}$ [Pa]")
ax.set_title("Pressure deviation from linear reference near outlet")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(loc="upper left")
ax.grid(alpha=0.2)
ax.margins(x=0)

fig.tight_layout()
output = RESULTS / "RESTORE_pressureError.png"
fig.savefig(output, dpi=300)
plt.close(fig)

print(f"Saved: {output}")
