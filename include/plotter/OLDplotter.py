from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_FILE = Path("/dev/shm/sim_data.bin")
OUTPUT_DIRECTORY = Path("results")

# Set this to 1 for exactly one image pixel per LBM cell. Values such as 2 or 4
# preserve exact integer alignment and make streamlines easier to see.
PIXELS_PER_CELL = 1

SOLID_MASK_THRESHOLD = 0.5
PERCENTILE_RANGE = (1.0, 99.0) #(1.0, 99.0)
STREAMLINE_DENSITY = 1.2
STREAMLINE_ARROW_SIZE = 0.4
STREAMLINE_WIDTH_PX = 1.0
SHOW_AXIS_TICKS = True

# DPI only converts the explicitly specified pixel dimensions to figure inches.
# It no longer controls the number of pixels used for each simulation cell.
DPI = 100
TARGET_SCREEN_WIDTH_PX = 1920
TARGET_SCREEN_HEIGHT_PX = 1080
# Prevent UI elements from becoming smaller for low-resolution simulations.
MIN_UI_SCALE = 1.0

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
	use_full_range=False,
	ui_scale=1.0,
	):
	"""Draw one field without changing the explicitly sized data axes."""
	vmin, vmax = color_limits(
		data_array,
		is_solid,
		use_full_range=use_full_range,
	)

	masked_data = ma.array(data_array, mask=is_solid)
	colormap = plt.get_cmap("viridis").copy()
	colormap.set_bad("black")

	image = ax.imshow(
		masked_data,
		origin="lower",
		cmap=colormap,
		vmin=vmin,
		vmax=vmax,
		interpolation="nearest",
		resample=False,
		aspect="equal",
	)

	# These limits put the outer cell edges exactly on the axes boundaries.
	ax.set_xlim(-0.5, n_horizontal - 0.5)
	ax.set_ylim(-0.5, n_vertical - 0.5)

	tick_font_size = 7.0 * ui_scale
	label_font_size = 8.0 * ui_scale
	tick_length = 2.0 * ui_scale
	tick_padding = 2.0 * ui_scale
	tick_width = 0.8 * ui_scale

	if SHOW_AXIS_TICKS:
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
		labelpad=2.0 * ui_scale,
	)

	return image


def create_exact_pixel_figure(n_vertical, n_horizontal, number_of_panels):
	"""Create exact-size data axes and fullscreen-scaled UI elements."""
	if PIXELS_PER_CELL < 1 or not isinstance(PIXELS_PER_CELL, int):
		raise ValueError("PIXELS_PER_CELL must be a positive integer.")

	plot_width_px = n_horizontal * PIXELS_PER_CELL
	plot_height_px = n_vertical * PIXELS_PER_CELL

	# These are the desired decoration sizes after the image has been
	# scaled to TARGET_SCREEN_WIDTH_PX x TARGET_SCREEN_HEIGHT_PX.
	base_left_px = 60
	base_right_px = 20
	base_panel_gap_px = 80
	base_bottom_px = 60
	base_colorbar_height_px = 15
	base_colorbar_gap_px = 15
	base_top_px = 15

	base_horizontal_ui_px = (
		base_left_px
		+ base_right_px
		+ (number_of_panels - 1) * base_panel_gap_px
	)

	base_vertical_ui_px = (
		base_bottom_px
		+ base_colorbar_height_px
		+ base_colorbar_gap_px
		+ base_top_px
	)

	if TARGET_SCREEN_WIDTH_PX <= base_horizontal_ui_px:
		raise ValueError("TARGET_SCREEN_WIDTH_PX is too small for the layout.")

	if TARGET_SCREEN_HEIGHT_PX <= base_vertical_ui_px:
		raise ValueError("TARGET_SCREEN_HEIGHT_PX is too small for the layout.")

	total_data_width_px = number_of_panels * plot_width_px

	# This is approximately the reciprocal of the scale factor that a
	# fullscreen image viewer will apply to the completed PNG.
	width_ui_scale = total_data_width_px / (
		TARGET_SCREEN_WIDTH_PX - base_horizontal_ui_px
	)

	height_ui_scale = plot_height_px / (
		TARGET_SCREEN_HEIGHT_PX - base_vertical_ui_px
	)

	ui_scale = max(
		MIN_UI_SCALE,
		width_ui_scale,
		height_ui_scale,
	)

	# Keep axes boundaries on integer output pixels.
	def scaled_pixels(base_size):
		return max(1, int(round(base_size * ui_scale)))

	left_px = scaled_pixels(base_left_px)
	right_px = scaled_pixels(base_right_px)
	panel_gap_px = scaled_pixels(base_panel_gap_px)
	bottom_px = scaled_pixels(base_bottom_px)
	colorbar_height_px = scaled_pixels(base_colorbar_height_px)
	colorbar_gap_px = scaled_pixels(base_colorbar_gap_px)
	top_px = scaled_pixels(base_top_px)

	plot_bottom_px = (
		bottom_px
		+ colorbar_height_px
		+ colorbar_gap_px
	)

	figure_width_px = (
		left_px
		+ number_of_panels * plot_width_px
		+ (number_of_panels - 1) * panel_gap_px
		+ right_px
	)

	figure_height_px = (
		plot_bottom_px
		+ plot_height_px
		+ top_px
	)

	fig = plt.figure(
		figsize=(figure_width_px / DPI, figure_height_px / DPI),
		dpi=DPI,
		facecolor="white",
	)

	axes = []
	colorbar_axes = []

	for panel_index in range(number_of_panels):
		left_edge_px = (
			left_px
			+ panel_index * (plot_width_px + panel_gap_px)
		)

		ax = fig.add_axes([
			left_edge_px / figure_width_px,
			plot_bottom_px / figure_height_px,
			plot_width_px / figure_width_px,
			plot_height_px / figure_height_px,
		])

		colorbar_ax = fig.add_axes([
			left_edge_px / figure_width_px,
			bottom_px / figure_height_px,
			plot_width_px / figure_width_px,
			colorbar_height_px / figure_height_px,
		])

		axes.append(ax)
		colorbar_axes.append(colorbar_ax)

	geometry = {
		"figure_width_px": figure_width_px,
		"figure_height_px": figure_height_px,
		"plot_width_px": plot_width_px,
		"plot_height_px": plot_height_px,
		"ui_scale": ui_scale,
	}

	return fig, axes, colorbar_axes, geometry


def verify_axes_pixel_sizes(fig, axes, expected_width_px, expected_height_px):
	"""Fail early if layout changes have broken the cell-to-pixel mapping."""
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
	grid_id = data[:, :, 5]

	velocity_planar = np.hypot(velocity_horizontal, velocity_vertical)

	is_solid = mask > SOLID_MASK_THRESHOLD

	fig, axes, colorbar_axes, geometry = create_exact_pixel_figure(
		n_vertical,
		n_horizontal,
		number_of_panels=4,
	)
	ui_scale = geometry["ui_scale"]

	add_panel(
		fig,
		axes[0],
		colorbar_axes[0],
		velocity_planar,
		is_solid,
		"Planar velocity [m/s]",
		n_vertical,
		n_horizontal,
		ui_scale=ui_scale,
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
		ui_scale=ui_scale,
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
		ui_scale=ui_scale,
	)
	add_panel(
		fig,
		axes[3],
		colorbar_axes[3],
		grid_id,
		is_solid,
		"Grid ID [1]",
		n_vertical,
		n_horizontal,
		use_full_range=True,
		ui_scale=ui_scale,
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

	# Matplotlib line widths are in points, so convert the requested pixel width.
	streamline_width_points = STREAMLINE_WIDTH_PX * 72.0 / DPI

	axes[0].streamplot(
		horizontal_coordinates,
		vertical_coordinates,
		horizontal_stream_velocity,
		vertical_stream_velocity,
		color="white",
		linewidth=streamline_width_points,
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

	# Do not use constrained_layout, tight_layout, or bbox_inches="tight" here:
	# they would change the axes geometry and break exact pixel alignment.
	fig.savefig(
		output_path,
		dpi=DPI,
		bbox_inches=None,
		pad_inches=0,
		facecolor="white",
	)
	plt.close(fig)

	print(f"Exported to {output_path}")
	"""
	print(
		f"Each {n_horizontal} x {n_vertical} panel uses "
		f"{geometry['plot_width_px']} x {geometry['plot_height_px']} pixels "
		f"({PIXELS_PER_CELL} pixel(s) per cell in each direction)."
	)
	print(
		f"Full PNG size: {geometry['figure_width_px']} x "
		f"{geometry['figure_height_px']} pixels."
	)
	"""


if __name__ == "__main__":
	main()
