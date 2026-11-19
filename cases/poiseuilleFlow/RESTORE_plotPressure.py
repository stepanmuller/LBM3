from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

plt.rcParams.update({
	"font.family": "serif",
	"font.serif": ["cmr10"],
	"mathtext.fontset": "cm",
	"axes.formatter.use_mathtext": True,
	"axes.unicode_minus": False,
	"font.size": 12,
})

cases = [
	("RESTORE_OLD", "Second order raw moment reconstruction", "teal",   "-"),
	("RESTORE_NEW", "Third order central moment reconstruction", "purple", "--"),
]

for derivative in (False, True):
	fig, ax = plt.subplots(figsize=(7, 4))

	for filename, label, color, linestyle in cases:
		zCoord, pressure = np.loadtxt(
			RESULTS / f"{filename}_centerlinePressure.txt",
			delimiter=";",
			skiprows=1,
			unpack=True,
		)

		values = -np.gradient(pressure, zCoord, edge_order=1) if derivative else pressure

		ax.plot(
			zCoord, values,
			label=label,
			color=color,
			linestyle=linestyle,
			linewidth=1.5,
		)
	
	ax.set_xlim(-10, 10)
	#ax.set_ylim(-0.00001, 0.00003)
	ax.set_xlabel(r"$z$ [mm]")
	ax.set_ylabel(
		r"$-\mathrm{d}p/\mathrm{d}z$ [Pa/mm]"
		if derivative else r"$p$ [Pa]"
	)
	ax.set_title(
		r"Pressure derivative vs Z coordinate"
		if derivative else r"Pressure vs Z coordinate"
	)
	ax.legend(loc="upper right")
	ax.grid(alpha=0.2)
	ax.margins(x=0)

	fig.tight_layout()
	filename = "RESTORE_pressureDerivative.png" if derivative else "RESTORE_pressure.png"
	fig.savefig(RESULTS / filename, dpi=600)
	plt.close(fig)
