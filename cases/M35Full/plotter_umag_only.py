import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma

# 1. Read binary data
file_path = "/dev/shm/sim_data.bin"
with open(file_path, "rb") as f:
    plotNumber = np.fromfile(f, dtype=np.int32, count=1)[0]
    # Header: [plotNumber, nVertical, nHorizontal, n_vars]
    dims = np.fromfile(f, dtype=np.int32, count=3)
    nVertical, nHorizontal, n_vars = dims
    data = np.fromfile(f, dtype=np.float32).reshape((nVertical, nHorizontal, n_vars))

# 2. Extract variables (Matches C++ fwrite order)
p = data[:, :, 0]
uHorizontal = data[:, :, 1]
uVertical = data[:, :, 2]
uNormal = data[:, :, 3]
mask = data[:, :, 4] 
gridID = data[:, :, 5]

uPlanar = np.sqrt(uHorizontal**2 + uVertical**2)
uMagTotal = np.sqrt(uHorizontal**2 + uVertical**2 + uNormal**2)

fig = plt.figure(figsize=(16, 9), constrained_layout=True)
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0])

is_solid = mask > 0.5 

def setup_plot(ax, data_array, label):
	# Get only the fluid data (ignore solid mask for scale calculation)
    fluid_data = data_array[mask <= 0.5]
    
    # Calculate 5th and 95th percentiles (removes 2% smallest and 2% largest)
    vmin = np.nanpercentile(fluid_data, 1)
    vmax = np.nanpercentile(fluid_data, 99)
    
    # If the range is zero (constant field), default to data min/max
    if (vmin == vmax) or label == "Grid ID [1]":
        vmin, vmax = np.min(fluid_data), np.max(fluid_data)
        
    masked_data = ma.array(data_array, mask=is_solid)
    img = ax.imshow(masked_data, origin="lower", cmap="viridis", aspect="equal", interpolation="nearest")
    # REMOVE THE FRAME (SPINES)
    for spine in ax.spines.values():
        spine.set_visible(False)
    img.cmap.set_bad(color="black")
    if vmin is not None: img.set_clim(vmin, vmax)
    
    cbar = fig.colorbar(
		img,
		ax=ax,
		orientation="horizontal",
		location="bottom",
		shrink=0.50,   # Colorbar uses 80% of the plot width
		pad=0.08,
		aspect=40,
	)
    cbar.set_label(label)
    return img

# Plot 1: Velocity magnitude
setup_plot(ax0, uMagTotal, "Velocity Magnitude [m/s]")

# 5. Save
os.makedirs("results", exist_ok=True)
#plt.savefig(f"results/{plotNumber}.png", dpi=min(1000, max([300, nVertical/2, nHorizontal/2])), bbox_inches="tight")
plt.savefig(f"results/{plotNumber}.png", dpi=200, bbox_inches="tight")
plt.close()
