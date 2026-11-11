from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_FILE = Path("/dev/shm/sim_data.bin")
OUTPUT_DIRECTORY = Path("results")

# Physical print size. Insert the image at this width in the thesis.
FIGURE_WIDTH_MM = 160.0
DPI = 600
PANEL_COLUMNS = 2             # Original four-panel row; use 2 for larger panels.
TICK_FONT_SIZE = 9.0          # Points at the final 160 mm width.
LABEL_FONT_SIZE = 9.0

MARKER_GRAY = 0.65
PERCENTILE_RANGE = (1.0, 99.0)
STREAMLINE_DENSITY = 0.8
STREAMLINE_ARROW_SIZE = 0.6
STREAMLINE_WIDTH_PT = 0.5
SHOW_AXIS_TICKS = False

# Computer Modern is bundled with Matplotlib; no external LaTeX run is needed.
plt.rcParams.update({
	"font.family": "serif",
	"font.serif": ["cmr10"],
	"font.size": LABEL_FONT_SIZE,
	"mathtext.fontset": "cm",
	"axes.formatter.use_mathtext": True,
	"axes.unicode_minus": False,
	"text.usetex": False,
	"savefig.facecolor": "white",
})

def read_lbm_data(file_path: Path):
	"""Read the binary file written by the LBM simulation."""
	with file_path.open("rb") as file:
		plot_number_data = np.fromfile(file, dtype=np.int32, count=1)
		dimensions = np.fromfile(file, dtype=np.int32, count=3)

		if plot_number_data.size != 1 or dimensions.size != 3:
			raise ValueError(f"Incomplete binary header in {file_path}")

		plot_number = int(plot_number_data[0])
		n_vertical, n_horizontal, n_variables = map(int, dimensions)

		if n_vertical <= 0 or n_horizontal <= 0 or n_variables < 6:
			raise ValueError(
				"Invalid dimensions in binary header: "
				f"nVertical={n_vertical}, nHorizontal={n_horizontal}, "
				f"n_vars={n_variables}; at least six variables are required."
			)

		expected_values = n_vertical * n_horizontal * n_variables
		flat_data = np.fromfile(file, dtype=np.float32, count=expected_values)

	if flat_data.size != expected_values:
		raise ValueError(
			f"Incomplete data in {file_path}: expected {expected_values} "
			f"float32 values, found {flat_data.size}."
		)

	data = flat_data.reshape((n_vertical, n_horizontal, n_variables))
	return plot_number, n_vertical, n_horizontal, data


def color_limits(data_array, is_solid, use_full_range=False):
	"""Calculate finite color limits using only fluid cells."""
	valid = data_array[(~is_solid) & np.isfinite(data_array)]

	if valid.size == 0:
		if np.all(is_solid):
			return 0.0, 1.0  # The marker overlay covers the entire field.
		raise ValueError("The selected field contains no finite fluid-cell values.")

	data_min = float(np.min(valid))
	data_max = float(np.max(valid))

	if use_full_range:
		vmin, vmax = data_min, data_max
	else:
		vmin, vmax = map(float, np.percentile(valid, PERCENTILE_RANGE))

		# Fall back to the full range if percentile clipping collapses the scale.
		if np.isclose(vmin, vmax):
			vmin, vmax = data_min, data_max

	# Matplotlib needs a nonzero range even when the entire field is constant.
	if np.isclose(vmin, vmax):
		padding = max(abs(vmin), 1.0) * 0.5
		vmin -= padding
		vmax += padding

	return vmin, vmax


def add_panel(
	fig,
	ax,
	colorbar_ax,
	data_array,
	is_solid,
	label,
	n_vertical,
	n_horizontal,
	marker_rgba,
	use_full_range=False,
	discrete_resolution=False,
	):
	"""Draw one field without changing the explicitly sized data axes."""
	colorbar_options = {}
	if discrete_resolution:
		# Bin in log2 space so every factor-of-two refinement has equal spacing.
		valid = (~is_solid) & np.isfinite(data_array)
		levels = np.unique(data_array[valid])
		if np.any(levels <= 0):
			raise ValueError("Grid resolutions must be positive for a logarithmic scale.")
		if levels.size == 0 and not np.all(is_solid):
			raise ValueError("The resolution field contains no finite fluid-cell values.")

		log_levels = np.log2(levels.astype(np.float64))
		if levels.size > 1:
			boundaries = np.concatenate((
				[log_levels[0] - (log_levels[1] - log_levels[0]) / 2],
				(log_levels[:-1] + log_levels[1:]) / 2,
				[log_levels[-1] + (log_levels[-1] - log_levels[-2]) / 2],
			))
		else:
			center = log_levels[0] if levels.size else 0.0
			boundaries = np.array([center - 0.5, center + 0.5])

		colormap = plt.get_cmap("viridis_r", max(1, levels.size)).copy()
		logged_data = np.zeros(data_array.shape, dtype=np.float64)
		np.log2(data_array.astype(np.float64), out=logged_data, where=valid)
		masked_data = ma.array(logged_data, mask=~valid)
		image_options = {"norm": BoundaryNorm(boundaries, colormap.N)}
		colorbar_options = {
			"boundaries": boundaries,
			"ticks": log_levels,
			"spacing": "proportional",
			"drawedges": True,
		}
	else:
		vmin, vmax = color_limits(
			data_array,
			is_solid,
			use_full_range=use_full_range,
		)
		masked_data = ma.array(data_array, mask=is_solid)
		colormap = plt.get_cmap("viridis").copy()
		image_options = {"vmin": vmin, "vmax": vmax}

	colormap.set_bad("black")

	image = ax.imshow(
		masked_data,
		origin="lower",
		cmap=colormap,
		**image_options,
		interpolation="nearest",
		resample=False,
		aspect="equal",
	)

	# Positive markers are opaque gray-to-black; zero remains transparent.
	# Draw above streamlines so they cannot brighten marked cells.
	ax.imshow(
		marker_rgba,
		origin="lower",
		interpolation="nearest",
		resample=False,
		aspect="equal",
		zorder=3,
	)

	# These limits put the outer cell edges exactly on the axes boundaries.
	ax.set_xlim(-0.5, n_horizontal - 0.5)
	ax.set_ylim(-0.5, n_vertical - 0.5)

	tick_font_size = TICK_FONT_SIZE
	label_font_size = LABEL_FONT_SIZE
	tick_length = 2.5
	tick_padding = 2.0
	tick_width = 0.6

	if SHOW_AXIS_TICKS:
		# Keep the original cell-index axes, with fewer labels at print size.
		ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
		ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
		ax.tick_params(
			axis="both",
			labelsize=tick_font_size,
			length=tick_length,
			width=tick_width,
			pad=tick_padding,
		)
	else:
		ax.set_xticks([])
		ax.set_yticks([])

	for spine in ax.spines.values():
		spine.set_visible(False)

	colorbar = fig.colorbar(
		image,
		cax=colorbar_ax,
		orientation="horizontal",
		**colorbar_options,
	)
	if discrete_resolution:
		# Use original float32 values for concise, round-trip-accurate labels.
		colorbar.set_ticklabels([str(value) for value in levels])
		colorbar.minorticks_off()
	else:
		colorbar.locator = MaxNLocator(nbins=2)
		colorbar.formatter = ScalarFormatter(useOffset=False, useMathText=True)
		colorbar.formatter.set_powerlimits((-2, 3))
		colorbar.update_ticks()
		# Put scientific powers above the bar, clear of its large label below.
		colorbar.formatter.set_locs(colorbar.get_ticks())
		colorbar.ax.xaxis.get_offset_text().set_visible(False)
		colorbar.ax.text(
			1.0, 1.2, colorbar.formatter.get_offset(),
			transform=colorbar.ax.transAxes,
			ha="right", va="bottom", fontsize=tick_font_size,
		)

	colorbar.ax.tick_params(
		labelsize=tick_font_size,
		length=tick_length,
		width=tick_width,
		pad=tick_padding,
	)

	colorbar.ax.xaxis.get_offset_text().set_fontsize(tick_font_size)
	colorbar.outline.set_linewidth(tick_width)

	colorbar.set_label(
		label,
		fontsize=label_font_size,
		labelpad=1.0,
	)

	return image

def create_print_figure(n_vertical, n_horizontal, number_of_panels):
	"""Size the figure in millimetres and preserve square simulation cells."""
	if PANEL_COLUMNS not in (1, 2, 4):
		raise ValueError("PANEL_COLUMNS must be 1, 2, or 4.")
	columns = min(PANEL_COLUMNS, number_of_panels)
	rows = (number_of_panels + columns - 1) // columns

	left_mm, right_mm = 1.0, 1.0
	panel_gap_mm = 2.0
	top_mm = 1.0
	label_space_mm = 9.0
	colorbar_height_mm = 3.0
	colorbar_gap_mm = 5.0
	row_gap_mm = 2.0

	panel_width_mm = (
		FIGURE_WIDTH_MM - left_mm - right_mm - (columns - 1) * panel_gap_mm
	) / columns
	if panel_width_mm <= 0:
		raise ValueError("Figure width is too small for the selected layout.")
	panel_height_mm = panel_width_mm * n_vertical / n_horizontal
	row_height_mm = (label_space_mm + colorbar_height_mm
					 + colorbar_gap_mm + panel_height_mm)
	figure_height_mm = top_mm + rows * row_height_mm + (rows - 1) * row_gap_mm

	fig = plt.figure(
		figsize=(FIGURE_WIDTH_MM / 25.4, figure_height_mm / 25.4),
		dpi=DPI,
		facecolor="white",
	)
	axes, colorbar_axes = [], []
	for panel_index in range(number_of_panels):
		row, column = divmod(panel_index, columns)
		left = left_mm + column * (panel_width_mm + panel_gap_mm)
		row_bottom = (rows - 1 - row) * (row_height_mm + row_gap_mm)
		colorbar_bottom = row_bottom + label_space_mm
		plot_bottom = colorbar_bottom + colorbar_height_mm + colorbar_gap_mm
		axes.append(fig.add_axes([
			left / FIGURE_WIDTH_MM,
			plot_bottom / figure_height_mm,
			panel_width_mm / FIGURE_WIDTH_MM,
			panel_height_mm / figure_height_mm,
		]))
		colorbar_width_mm = 0.9 * panel_width_mm
		colorbar_left_mm = left + (panel_width_mm - colorbar_width_mm) / 2

		colorbar_axes.append(fig.add_axes([
			colorbar_left_mm / FIGURE_WIDTH_MM,
			colorbar_bottom / figure_height_mm,
			colorbar_width_mm / FIGURE_WIDTH_MM,
			colorbar_height_mm / figure_height_mm,
		]))

	geometry = {
		"plot_width_px": panel_width_mm / 25.4 * DPI,
		"plot_height_px": panel_height_mm / 25.4 * DPI,
	}
	return fig, axes, colorbar_axes, geometry


def verify_axes_pixel_sizes(fig, axes, expected_width_px, expected_height_px):
	"""Verify the physical panel dimensions and equal cell aspect ratio."""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()

	for panel_number, ax in enumerate(axes, start=1):
		bounds = ax.get_window_extent(renderer=renderer)
		actual_width = float(bounds.width)
		actual_height = float(bounds.height)

		if not (
			np.isclose(actual_width, expected_width_px, atol=0.01)
			and np.isclose(actual_height, expected_height_px, atol=0.01)
		):
			raise RuntimeError(
				f"Panel {panel_number} is {actual_width:.3f} x "
				f"{actual_height:.3f} pixels; expected "
				f"{expected_width_px} x {expected_height_px}."
			)


def main():
	plot_number, n_vertical, n_horizontal, data = read_lbm_data(INPUT_FILE)

	# Variable order must match the C++ fwrite order.
	pressure = data[:, :, 0]
	velocity_horizontal = data[:, :, 1]
	velocity_vertical = data[:, :, 2]
	velocity_normal = data[:, :, 3]
	mask = data[:, :, 4]
	grid_resolution = data[:, :, 5]

	velocity_planar = np.hypot(velocity_horizontal, velocity_vertical)

	marker = np.clip(mask, 0.0, 1.0)
	is_solid = marker > 0.0
	marker_rgba = np.empty((*marker.shape, 4), dtype=np.float32)
	marker_rgba[:, :, :3] = (MARKER_GRAY * (1.0 - marker))[:, :, None]
	marker_rgba[:, :, 3] = is_solid

	fig, axes, colorbar_axes, geometry = create_print_figure(
		n_vertical,
		n_horizontal,
		number_of_panels=4,
	)

	add_panel(
		fig,
		axes[0],
		colorbar_axes[0],
		velocity_planar,
		is_solid,
		"Planar velocity [m/s]",
		n_vertical,
		n_horizontal,
		marker_rgba=marker_rgba,
	)
	add_panel(
		fig,
		axes[1],
		colorbar_axes[1],
		velocity_normal,
		is_solid,
		"Normal velocity [m/s]",
		n_vertical,
		n_horizontal,
		marker_rgba=marker_rgba,
	)
	add_panel(
		fig,
		axes[2],
		colorbar_axes[2],
		pressure,
		is_solid,
		"Static pressure [Pa]",
		n_vertical,
		n_horizontal,
		marker_rgba=marker_rgba,
	)
	add_panel(
		fig,
		axes[3],
		colorbar_axes[3],
		grid_resolution,
		is_solid,
		"Grid resolution [mm]",
		n_vertical,
		n_horizontal,
		marker_rgba=marker_rgba,
		discrete_resolution=True,
	)

	horizontal_coordinates = np.arange(n_horizontal)
	vertical_coordinates = np.arange(n_vertical)
	horizontal_stream_velocity = np.where(
		is_solid,
		np.nan,
		velocity_horizontal,
	)
	vertical_stream_velocity = np.where(
		is_solid,
		np.nan,
		velocity_vertical,
	)

	axes[0].streamplot(
		horizontal_coordinates,
		vertical_coordinates,
		horizontal_stream_velocity,
		vertical_stream_velocity,
		color="white",
		linewidth=STREAMLINE_WIDTH_PT,
		density=STREAMLINE_DENSITY,
		arrowsize=STREAMLINE_ARROW_SIZE,
		zorder=2,
	)

	# streamplot can autoscale; restore the exact half-cell outer boundaries.
	axes[0].set_xlim(-0.5, n_horizontal - 0.5)
	axes[0].set_ylim(-0.5, n_vertical - 0.5)

	verify_axes_pixel_sizes(
		fig,
		axes,
		geometry["plot_width_px"],
		geometry["plot_height_px"],
	)

	OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
	output_path = OUTPUT_DIRECTORY / f"{plot_number}.png"

	# Keep the full canvas: tight cropping would change the 160 mm print width.
	fig.savefig(
		output_path,
		dpi=DPI,
		bbox_inches=None,
		pad_inches=0,
		facecolor="white",
	)
	plt.close(fig)

	print(f"Exported to {output_path} ({FIGURE_WIDTH_MM:g} mm wide, {DPI} dpi)")


if __name__ == "__main__":
	main()
