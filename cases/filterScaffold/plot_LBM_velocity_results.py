#!/usr/bin/env python3
"""Export pressure-loss and density-error plots against LBM inlet velocity.

Dependencies: numpy, matplotlib (pip install numpy matplotlib).
Keep filter_results.txt beside this script, then run:
    python3 plot_filter_results.py
Or specify a data file and output directory:
    python3 plot_filter_results.py data.txt --output-dir figures --no-show

The semicolon-delimited input has one header row and three numeric columns:
LBM inlet velocity [1], area averaged pressure loss [Pa], and LBM weakly
compressible density error [1]. Both responses use the 'Last 30% mean'
annotations in results/<LBM inlet velocity>/custom.png, retaining the displayed
precision. The density error comes from the subplot at x = 4.2 mm.

Exports pressure_loss.png and density_error.png at 300 dpi beside the input
file, unless --output-dir is supplied. Both use black x markers. Computer
Modern fonts and Matplotlib mathtext provide LaTeX-style formatting without
requiring a separate LaTeX installation.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data", nargs="?", type=Path,
        default=Path(__file__).resolve().with_name("LBM_velocity_results.txt"),
        help="Three-column, semicolon-delimited text file (default: beside script).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory for both PNG images (default: beside input file).",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save without opening plot windows.",
    )
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.loadtxt(args.data, delimiter=";", skiprows=1, ndmin=2)
    if data.shape[0] == 0 or data.shape[1] != 3 or not np.isfinite(data).all():
        parser.error("Input must contain three finite numeric columns and at least one row.")
    data = data[np.argsort(data[:, 0])]
    output_dir = args.output_dir or args.data.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.data.resolve() in {
        (output_dir / name).resolve()
        for name in ("pressure_loss.png", "density_error.png")
    }:
        parser.error("Output images must differ from the input data file.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "axes.formatter.use_mathtext": True,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
    })

    plots = [
        (1, r"$\mathrm{Area\ averaged\ pressure\ loss}\ [\mathrm{Pa}]$",
         "pressure_loss.png"),
        (2, r"$\mathrm{LBM\ weakly\ compressible}$" + "\n"
         + r"$\mathrm{density\ error}\ [1]$", "density_error.png"),
    ]
    figures = []
    for column, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
        figures.append(fig)
        ax.plot(
            data[:, 0], data[:, column], linestyle="none", marker="x",
            color="black", markersize=6, markeredgewidth=1.3,
        )
        ax.set_xlabel(r"$\mathrm{LBM\ inlet\ velocity}\ [1]$")
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.minorticks_on()
        ax.grid(True, which="major", color="0.88", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.05, y=0.08)

        output = output_dir / filename
        fig.savefig(output, facecolor="white", bbox_inches="tight")
        print(f"Plotted {len(data)} points. Saved: {output.resolve()}")
    if not args.no_show:
        plt.show()
    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
