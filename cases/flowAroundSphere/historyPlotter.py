import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def set_smart_ylim(ax, data):
    """Sets y-limits based on the second half of data with a 5% margin."""
    if len(data) < 2:
        return
    mid = len(data) // 2
    recent_data = data[mid:]
    y_min, y_max = np.min(recent_data), np.max(recent_data)
    padding = (y_max - y_min) * 0.05
    if padding == 0: padding = 1e-7
    ax.set_ylim(y_min - padding, y_max + padding)

def add_average_diagnostics(ax, iterations, data, unit_str="", fmt=".3f"):
    """Calculates last 20% average, draws the line, and returns the formatted string."""
    count = len(data)
    window = max(1, count // 5)
    avg_val = np.mean(data[-window:])
    
    ax.hlines(y=avg_val, xmin=iterations[-window], xmax=iterations[-1], 
              colors='black', linestyles='--', linewidth=2.0, alpha=0.9, zorder=3)
    
    return f"{avg_val:{fmt}}{unit_str}"

def plot_history(file_number):
    try:
        with open("/dev/shm/historyData.bin", "rb") as f:
            count = np.fromfile(f, dtype=np.int32, count=1)[0]
            # Assuming the binary file now only contains the Drag Coefficient data
            drag_coeff = np.fromfile(f, dtype=np.float32, count=count)
    except Exception:
        return

    # Single subplot, maintaining the 16:9 aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    iterations = np.arange(count)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)

    # --- Drag Coefficient Plot ---
    ax.plot(iterations, drag_coeff, color='black', linewidth=1.2, alpha=0.7)
    ax.set_ylabel(r"$C_D$ [1]")
    ax.set_xlabel(r"Iteration")
    ax.set_title(r"\textbf{Drag Coefficient History}")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    
    set_smart_ylim(ax, drag_coeff)
    
    label_cd = add_average_diagnostics(ax, iterations, drag_coeff)
    ax.text(0.98, 0.95, f'\\textbf{{Avg $C_D$: {label_cd}}}', 
            transform=ax.transAxes, ha='right', va='top', bbox=bbox_props)

    plt.tight_layout()
    os.makedirs("results/History", exist_ok=True)
    filename = str(file_number) + "History.png"
    plt.savefig("results/History/" + filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Get the argument from the command line, defaulting to 0 if none is provided
    file_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    plot_history(file_num)
