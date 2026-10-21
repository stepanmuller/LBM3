import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

# ==========================================
# 1. DATA DEFINITION
# ==========================================
filename = "resultsAll"
supTitle = r"\textbf{Pressure loss vs N}"
xLabel = r"N [1]"
yLabel = r"Pressure loss [Pa]"

xMin = 0
xMax = 35
yMin = 0.0
yMax = 4.0

# Grouping for easy iteration
xList = []
yList = []
yLegendList = []
yColorList = []
yStyleList = []

# coarse
xList.append([32])
yList.append([3.0398])
yLegendList.append("res 0.0875 mm")
yColorList.append("gray")
yStyleList.append("X")

# medium
xList.append([2, 4, 8, 16, 32])
yList.append([0.29952, 0.47228, 0.81849, 1.5134, 2.9134])
yLegendList.append("res 0.04375 mm")
yColorList.append("black")
yStyleList.append("X")

# medium
xList.append([2, 4, 8, 16])
yList.append([0.028828, 0.028883, 0.028995, 0.028828])
yLegendList.append("res 0.04375 mm, pressure loss across last 1/3 length (free pipe)")
yColorList.append("black")
yStyleList.append("o")

# linear line
x_medium = np.array([2, 4, 8, 16, 32])
y_medium = np.array([0.29952, 0.47228, 0.81849, 1.5134, 2.9134])
y_free = np.array([0.028828, 0.028883, 0.028995, 0.028828])
# Fixed intercept and least-squares slope
b = 3 * np.mean(y_free)
m = np.dot(x_medium, y_medium - b) / np.dot(x_medium, x_medium)
# Extend the line from zero to the largest x across all datasets
x_line = np.linspace(0, max(max(xs) for xs in xList), 200)
y_line = m * x_line + b
xList.append(x_line.tolist())
yList.append(y_line.tolist())
yLegendList.append("res 0.04375 mm, fit from free pipe reference")
yColorList.append("purple")
yStyleList.append("-")

# ==========================================
# 2. PLOT STYLING (LaTeX Enabled)
# ==========================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
})


# ==========================================
# 3. PLOT GENERATION
# ==========================================
# Single subplot, maintaining the 16:9 aspect ratio
fig, ax = plt.subplots(1, 1, figsize=(12, 9))

# Loop through all datasets and plot them
for i in range(len(yList)):
    ax.plot(xList[i], yList[i], yStyleList[i], color=yColorList[i], label=yLegendList[i], linewidth=2.0, markersize=8, alpha=0.8)

# Apply limits
ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)

# Set labels and title
ax.set_xlabel(xLabel)
ax.set_ylabel(yLabel)
ax.set_title(supTitle)

# Visual grid and locators
# 1. Y-axis Locators: Increase major bins slightly and enable minor ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 
ax.yaxis.set_minor_locator(AutoMinorLocator(5)) # Divides each major y-interval into 5 minor intervals
# 2. Denser, highly visible grid
# Major grid lines (solid, darker)
ax.grid(which='major', color='black', linestyle='-', linewidth=0.8, alpha=0.6)
# Minor grid lines (dashed, lighter, fills in the log scale and new y-axis minor ticks)
ax.grid(which='minor', color='dimgray', linestyle='--', linewidth=0.5, alpha=0.4)

# Add a legend, mimicking the style of the old square bbox
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, borderpad=0.5)

# Clean layout
plt.tight_layout()

# Save the plot
plt.savefig(filename, dpi=300)
plt.close()

print(f"Successfully plotted and saved to: {filename}")
