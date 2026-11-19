from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Physical parameters
Re = 100.0        # Based on mean velocity and pipe diameter
D = 20.0         # Pipe diameter [mm]
nu = 1.5e-5      # Kinematic viscosity [m²/s]
rho = 1.225      # Density [kg/m³]
z_in = -10.0     # Inlet coordinate [mm]
z_out = 10.0     # Outlet coordinate [mm]
p_out = 0.0      # Outlet gauge pressure [Pa]

RESULTS = Path(__file__).resolve().parent / "results" / "INTERFACE"

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

cases = [
	("FINE",      "Fine reference",   "0.30",    "-"),
	("COARSE",    "Coarse reference", "0.55",    "-"),
	("QUADRATIC", "Quadratic",        "purple",  "--"),
	("LINEAR",    "Linear",           "teal",    "--"),
	("ITERATIVE", "Iterative",        "#e67e22", ":"),
]

# Load each dataset once.
data = {}
for filename, _, _, _ in cases:
	z, pressure = np.loadtxt(
		RESULTS / f"{filename}_centerlinePressure.txt",
		delimiter=";",
		skiprows=1,
		unpack=True,
	)
	order = np.argsort(z)
	data[filename] = (z[order], pressure[order])

# Fully developed circular-pipe Poiseuille solution.
# Convert diameter to metres, then pressure gradient to Pa/mm.
D_m = D * 1e-3
pressure_gradient = 32.0 * rho * nu**2 * Re / D_m**3 * 1e-3


def analytic_pressure(z):
	return p_out + pressure_gradient * (z_out - z)


def finish_plot(fig, ax, ylabel, title, filename, position):
	ax.set_xlim(z_in, z_out)
	ax.set_xlabel(r"$z$ [mm]")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(alpha=0.2)
	ax.margins(x=0)
	ax.legend(loc=position)
	fig.tight_layout()
	fig.savefig(RESULTS / filename, dpi=DPI)
	plt.close(fig)

z_analytic = np.linspace(z_in, z_out, 500)

for derivative in (False, True):
	fig, ax = plt.subplots(figsize=FIGSIZE)

	for filename, label, color, linestyle in cases:
		z, pressure = data[filename]
		values = (
			-np.gradient(pressure, z, edge_order=1)
			if derivative else pressure
		)

		ax.plot(
			z, values,
			label=label,
			color=color,
			linestyle=linestyle,
			linewidth=LINEWIDTH,
		)

	ax.plot(
		z_analytic,
		np.full_like(z_analytic, pressure_gradient)
		if derivative else analytic_pressure(z_analytic),
		label="Analytic solution",
		color="black",
		linestyle="dotted",
		linewidth=LINEWIDTH,
	)

	finish_plot(
		fig, ax,
		ylabel=(
			r"$-\mathrm{d}p/\mathrm{d}z$ [Pa/mm]"
			if derivative else r"$p$ [Pa]"
		),
		title=(
			"Pressure derivative vs Z coordinate"
			if derivative else "Pressure vs Z coordinate"
		),
		filename=(
			"INTERFACE_pressureDerivative.png"
			if derivative else "INTERFACE_pressure.png"
		),
		position=(
			"upper center" if derivative else "upper right"
		),
	)

# Pressure error relative to the analytic solution.
fig, ax = plt.subplots(figsize=FIGSIZE)

for filename, label, color, linestyle in cases:
	z, pressure = data[filename]
	error = pressure - analytic_pressure(z)

	ax.plot(
		z, error,
		label=label,
		color=color,
		linestyle=linestyle,
		linewidth=LINEWIDTH,
	)

finish_plot(
	fig, ax,
	ylabel=r"$p - p_{\mathrm{analytic}}$ [Pa]",
	title="Pressure error relative to analytic solution",
	filename="INTERFACE_pressureError.png",
	position="upper right"
)
