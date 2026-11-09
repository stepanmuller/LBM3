#!/usr/bin/env python3
"""Run the compiled intake solver sequentially and archive each case."""

from pathlib import Path
import shutil
import subprocess
import time


CASE_DIR = Path("/home/stepan/Documents/GitHub/LBM3/cases/BruteforceIntakeOnly")

# (Source STL filename, destination folder name inside results/)
CASES = [
    ("1.STL", "P1"),
    ("2.STL", "P2"),
    ("3.STL", "P3"),
    ("4.STL", "P4"),
    ("5.STL", "P5"),
    ("6.STL", "P6"),
    ("7.STL", "P7"),
    ("8.STL", "P8"),
]

# Match the compiled solver: final plots are 100000.png ... 100031.png.
LAST_ITERATION = 100000
PLOTS_PER_ITERATION = 32


def main():
    executable = CASE_DIR / "BruteforceIntakeMain"
    results = CASE_DIR / "results"
    tracker = results / "tracker"

    # Check everything before starting an expensive batch. Never overwrite archives.
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if tracker.exists():
        raise FileExistsError(f"Archive or move the existing folder first: {tracker}")
    destinations = [name for _, name in CASES]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Each case must have a different destination folder name.")
    for stl, name in CASES:
        if not (CASE_DIR / stl).is_file():
            raise FileNotFoundError(CASE_DIR / stl)
        if name in ("", ".", "..", "tracker") or Path(name).name != name:
            raise ValueError(f"Use a simple destination folder name: {name!r}")
        if (results / name).exists():
            raise FileExistsError(results / name)

    results.mkdir(exist_ok=True)
    for stl, name in CASES:
        print(f"\nRunning {name}: {stl}", flush=True)
        shutil.copy2(CASE_DIR / stl, CASE_DIR / "IntakeSTL.STL")
        started = time.time_ns()
        subprocess.run([str(executable)], cwd=CASE_DIR, check=True)

        pictures = [
            results / f"{LAST_ITERATION + offset}.png"
            for offset in range(PLOTS_PER_ITERATION)
        ]
        # Old pictures can remain in results/. Do not archive them after a failed plot.
        for picture in pictures:
            if not picture.is_file() or picture.stat().st_mtime_ns < started:
                raise RuntimeError(f"Missing or stale final picture: {picture}")
        if not tracker.is_dir() or not any(tracker.glob("*.png")):
            raise RuntimeError(f"No tracker pictures were produced: {tracker}")

        for picture in pictures:
            shutil.copy2(picture, tracker / picture.name)
        tracker.rename(results / name)
        print(f"Saved {results / name}", flush=True)


if __name__ == "__main__":
    main()
