import sys
from pathlib import Path
import numpy as np

from plotGridsFullCustom import read_lbm_data, INPUT_FILE

_, ny, nz, data = read_lbm_data(INPUT_FILE)

z_first, spacing = map(float, sys.argv[1:3])
z = z_first + np.arange(nz) * spacing

# Average the two central rows if ny is even.
pressure = data[(ny - 1) // 2 : ny // 2 + 1, :, 0].mean(axis=0)

Path("results").mkdir(exist_ok=True)

np.savetxt(
    "results/centerlinePressure.txt",
    np.column_stack((z, pressure)),
    delimiter=";",
    header="Z coordinate [mm];pressure [Pa]",
    comments="",
    fmt="%.9g",
)
