from pathlib import Path

import matplotlib.pyplot as plt

markerSize = 36  # Marker area in points squared, shared by all datasets.

# Repeat these five append lines for each dataset.
# Use flat lists with matching lengths; empty lists are skipped.
xLists = []
yLists = []
labels = []
markerTypes = []
markerColors = []

# COARSE RES 0.175
xLists.append([0.05305, 2*0.05305, 4*0.05305, 8*0.05305, 16*0.05305, 32*0.05305])  # Fill with v_inlet_phys values [m/s].
yLists.append([2.7089, 6.0042, 14.387, 35.823, 88.937, 216.61])  # Fill with corresponding pressure losses [Pa].
labels.append("Res 0.175mm")
markerTypes.append("x")
markerColors.append("teal")

# MEDIUM RES 0.175/2
xLists.append([0.05305, 2*0.05305, 4*0.05305, 8*0.05305, 16*0.05305, 32*0.05305])  # Fill with v_inlet_phys values [m/s].
yLists.append([2.7692, 5.9884, 13.697, 32.046, 75.363, 179.13])  # Fill with corresponding pressure losses [Pa].
labels.append("Res 0.0875mm")
markerTypes.append("x")
markerColors.append("orange")

# FINE RES 0.175/4
xLists.append([0.05305])  # Fill with v_inlet_phys values [m/s].
yLists.append([2.8372])  # Fill with corresponding pressure losses [Pa].
labels.append("Res 0.04375mm")
markerTypes.append("x")
markerColors.append("purple")

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

if not (
    len(xLists) == len(yLists) == len(labels)
    == len(markerTypes) == len(markerColors)
):
    raise ValueError("Add one entry per dataset to each of the five lists.")

fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
for x, y, label, markerType, markerColor in zip(
    xLists, yLists, labels, markerTypes, markerColors
):
    if len(x) != len(y):
        raise ValueError(f"{label}: x and y must have matching lengths.")
    if not len(x):
        continue

    ax.scatter(
        x, y, s=markerSize, marker=markerType, color=markerColor,
        linewidths=1.3, label=label,
    )

ax.set_xlabel(r"$v_{\mathrm{inlet\_phys}}\ [\mathrm{m/s}]$")
ax.set_ylabel(r"$\mathrm{Area\ averaged\ pressure\ loss}\ [\mathrm{Pa}]$")
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax.minorticks_on()
ax.grid(True, which="major", color="0.83", linewidth=0.4)
ax.grid(True, which="minor", color="0.92", linewidth=0.25)
ax.set_axisbelow(True)
ax.margins(x=0.05, y=0.08)
if ax.collections:
    ax.legend()
ax.set_title("Pressure loss vs physical inlet velocity for N32")

# Axis limits are automatic. Uncomment and edit to set them manually.
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 10.0)

output = Path(__file__).resolve().parent / "pressure_loss_vs_velocity.png"
fig.savefig(output, facecolor="white", bbox_inches="tight")
plt.show()
plt.close(fig)
