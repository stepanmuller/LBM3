"""Tracker plots. Dependencies: numpy, matplotlib. 

Binary format (native byte order): int32 count, int32 object ID,
then six float32 histories, each containing count samples.
Rows are column-major plot order: left top/middle/bottom, right top/middle/bottom.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def _plot_tracker(kind, labels, data_file, output_directory, average_percent):
	if not 0 < average_percent <= 100:
		raise ValueError("average_percent must be in (0, 100]")
	with open(data_file, "rb") as stream:
		header = np.fromfile(stream, dtype=np.int32, count=2)
		if header.size != 2:
			raise ValueError("Incomplete tracker header")
		count, object_id = map(int, header)
		if count < 1 or object_id < 0:
			raise ValueError("Invalid sample count or object ID")
		values = np.fromfile(stream, dtype=np.float32, count=6 * count)
		if values.size != 6 * count or stream.read(1):
			raise ValueError("Tracker payload size does not match header")
	values = values.reshape(6, count)
	iterations = np.arange(1, count + 1)
	window = max(1, int(count * average_percent / 100))
	output_directory = Path(output_directory)
	output_directory.mkdir(parents=True, exist_ok=True)
	slug = {"Flow through Open Boundary": "boundary", "Force exerted on fluid by Wall": "wall", "Force exerted on fluid by Rotor": "rotor"}[kind]
	output_path = output_directory / f"{slug}_{object_id}.png"

	style = {
		"text.usetex": False,
		"font.family": "serif",
		"font.serif": ["cmr10", "DejaVu Serif"],
		"mathtext.fontset": "cm",
		"axes.formatter.use_mathtext": True,
		"font.size": 16,
		"axes.labelsize": 16,
		"axes.titlesize": 18,
		"xtick.labelsize": 16,
		"ytick.labelsize": 16,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"axes.unicode_minus": False,
		"savefig.bbox": None,
	}
	with plt.rc_context(style):
		fig, axes = plt.subplots(3, 2, figsize=(19.2, 10.8), dpi=200,
								 sharex=True, layout="constrained")
		try:
			fig.suptitle(f"{kind} ID {object_id}", fontsize=24)
			for component, (title, ylabel, unit) in enumerate(labels):
				row, column = component % 3, component // 3
				ax = axes[row, column]
				data = values[component].astype(np.float64)
				data[~np.isfinite(data)] = np.nan
				ax.plot(iterations, data, color="0.30", linewidth=1.1)
				ax.set_title(title, loc="left")
				ax.set_ylabel(ylabel)
				ax.set_xlim(0, count)
				ax.margins(y=0.12)
				# Scale y-axis using the last two averaging intervals.
				recent = data[-min(count, 2 * window):]
				recent = recent[np.isfinite(recent)]
				if recent.size:
					ymin, ymax = recent.min(), recent.max()
					padding = 0.05 * (ymax - ymin)
					# Ensure valid limits for constant or zero data.
					if padding == 0:
						padding = max(abs(ymin) * 0.05, 1e-7)
					ax.set_ylim(ymin - padding, ymax + padding)
				ax.grid(True, color="0.88", linewidth=0.6)
				ax.set_axisbelow(True)
				ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
				ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
				ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
				ax.tick_params(axis="x", labelbottom=(row == 2))
				if row == 2:
					ax.set_xlabel("Iterations finished [1]")
				tail = data[-window:]
				finite = tail[np.isfinite(tail)]
				if finite.size:
					average = finite.mean()
					ax.hlines(average, iterations[-window], iterations[-1],
							  colors="black", linestyles="--", linewidth=1.5)
					if window == 1:
						ax.plot(iterations[-1], average, "o", color="black", markersize=3)
					value_label = f"{average:.5g} {unit}"
				else:
					value_label = "unavailable"
				qualifier = " (finite samples)" if finite.size != window else ""
				ax.text(0.98, 0.96,
						f"Last {average_percent:g}% mean{qualifier}: {value_label}",
						transform=ax.transAxes, ha="right", va="top", fontsize=16,
						bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.92, pad=4))
			fig.savefig(output_path, dpi=200, bbox_inches=None, facecolor="white")
		finally:
			plt.close(fig)
	print(f"Exported to {output_path}", flush=True)
	return output_path


def plotTrackerWall(data_file="/dev/shm/trackerData.bin",
					output_directory="results/tracker", average_percent=20.0):
	return _plot_tracker("Force exerted on fluid by Wall", [
		(r"Force $F_x$ in global frame", r"$F_x$ [N]", "N"),
		(r"Force $F_y$ in global frame", r"$F_y$ [N]", "N"),
		(r"Force $F_z$ in global frame", r"$F_z$ [N]", "N"),
		(r"Torque $T_x$ in global frame", r"$T_x$ [N m]", "Nm"),
		(r"Torque $T_y$ in global frame", r"$T_y$ [N m]", "Nm"),
		(r"Torque $T_z$ in global frame", r"$T_z$ [N m]", "Nm"),
	], data_file, output_directory, average_percent)


def plotTrackerRotor(data_file="/dev/shm/trackerData.bin",
					 output_directory="results/tracker", average_percent=20.0):
	return _plot_tracker("Force exerted on fluid by Rotor", [
		(r"Force $F_x$ exerted on fluid in rotor frame", r"$F_x$ [N]", "N"),
		(r"Force $F_y$ exerted on fluid in rotor frame", r"$F_y$ [N]", "N"),
		(r"Force $F_z$ exerted on fluid in rotor frame", r"$F_z$ [N]", "N"),
		(r"Torque $T_x$ exerted on fluid in rotor frame", r"$T_x$ [N m]", "Nm"),
		(r"Torque $T_y$ exerted on fluid in rotor frame", r"$T_y$ [N m]", "Nm"),
		(r"Torque $T_z$ exerted on fluid in rotor frame", r"$T_z$ [N m]", "Nm"),
	], data_file, output_directory, average_percent)


def plotTrackerBoundary(data_file="/dev/shm/trackerData.bin",
                        output_directory="results/tracker",
                        average_percent=20.0):
    return _plot_tracker("Flow through Open Boundary", [
        (
            r"Area averaged normal velocity $\quad \frac{1}{A}\int_A u_n\,dA$",
            r"$u_n$ [m/s]", "m/s"
        ),
        (
            r"Mass flow out $\quad \int_A \rho u_n\,dA$",
            r"$\dot{m}$ [kg/s]", "kg/s"
        ),
        (
            r"Normal momentum thrust out $\quad \int_A \rho u_n |u_n|\,dA$",
            r"$T_n$ [N]", "N"
        ),
        (
            r"Area averaged pressure $\quad \frac{1}{A}\int_A p\,dA$",
            r"$p$ [Pa]", "Pa"
        ),
        (
            r"Pressure power out $\quad \int_A p u_n\,dA$",
            r"$P_p$ [W]", "W"
        ),
        (
            r"Normal kinetic power out $\quad \int_A \frac{1}{2}\rho u_n^3\,dA$",
            r"$P_{k,n}$ [W]", "W"
        ),
    ], data_file, output_directory, average_percent)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("kind", choices=["wall", "rotor", "boundary"])
	parser.add_argument("data_file", nargs="?", default="/dev/shm/trackerData.bin")
	parser.add_argument("--output-directory", default="results/tracker")
	parser.add_argument("--average-percent", type=float, default=20.0)
	args = parser.parse_args()
	try:
		{"wall": plotTrackerWall, "rotor": plotTrackerRotor,
		 "boundary": plotTrackerBoundary}[args.kind](
			args.data_file, args.output_directory, args.average_percent)
	except Exception as error:
		print(f"Tracker export failed: {error}", file=sys.stderr)
		sys.exit(1)
