import sys
from pathlib import Path

sys.path.insert(0, "../../include/plotter")
from plotGridsFull import read_lbm_data, INPUT_FILE
import matplotlib.pyplot as plt

iteration, ny, nz, data = read_lbm_data(INPUT_FILE)

# Select the middle row of the ZY section.
pressure = data[ny // 2, :, 0]

plt.plot(pressure)
plt.xlabel("Cell index along Z")
plt.ylabel("Static pressure [Pa]")
plt.grid(True)
plt.tight_layout()

Path("results").mkdir(exist_ok=True)
plt.savefig(f"results/centerlinePressure_{iteration}.png", dpi=150)
plt.close()
