#!/usr/bin/env python3
"""Interactive, two-level Boolean LBM experiment. Run with Python 3 + Tkinter.

Edit the settings below, then: python3 lbm_validity_visualizer.py
No NumPy, Matplotlib, CUDA, repository files, or pip packages are needed.

Paint DEEP refinement and walls on coarse cells. On initialization/reset, coarse
extension populations are invalid; all other fluid populations start valid.
Four fine children replace eight in this 2-D slice.
The rest population is a dot; the other eight populations are outward arrows.

The transport model is the K-independent slice of LBM3's Esotwist addresses:
I+ advances in the entire sorted cell list (including the final global wrap);
J+ cycles through the retained cells in the same I column. Diagonal addresses
use I+(J+(cell)), not an assumed Cartesian torus. A non-geometric transport is
invalid even if the wrapped source contains valid data. Explicit periodic BC
copies are separate, using the original level-dependent donor spacing.

IBB uses the repository's enabled-link rules reduced to 2-D and the validity of
its local pre/post-collision inputs (all coefficients nonzero, link length 0.5).
MBBC is the requested conservative Boolean dependency, without history/forcing.
Walls have no outgoing fluid populations. Fine wall status inherits its parent.
The grid keeps one coarse wall halo and retained-parent fine children, then a
fine wall halo, as in deleteExcessCells. Deep coarse cells are removed.

This is a dependency experiment, not a numerical LBM solver. Interface steps
use the editable proposed mask below, not the old Geier implementation.
Build smart masks restarts validity, runs one coarse and two fine updates,
then records actual invalid populations on both sets of communication targets.
It stops before F->C without performing any interface transfers. These masks
stay fixed during stepping and Reset validity; editing/loading geometry restores
the proposed rule-based masks. Back can undo the mask capture itself.
A bad assumed-known population OR a bad interpolation donor poisons the whole target.
Periodic targets skip interpolation and receive explicit refreshes. An all-red
interface failure remains visible until another operation actually repairs it.
"""

# -------------------------- USER SETTINGS ----------------------------------
WIDTH = 10
HEIGHT = 6
MBBC = "MBBC"
PERIODIC = "PERIODIC"
LEFT = MBBC
RIGHT = MBBC
TOP = PERIODIC
BOTTOM = PERIODIC

APPLY_IBB = True
INITIAL_DEEP = set()       # Optional coarse (i, j) coordinates, e.g. {(5, 3)}
INITIAL_WALLS = set()      # Independent of refinement; a wall may be refined.
# ---------------------------------------------------------------------------

import copy
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# K=0 directions in the same relative order as D3Q27Directions.h.
DIRS = ((0, 0), (1, 0), (-1, 0), (0, -1), (0, 1),
        (-1, -1), (1, 1), (-1, 1), (1, -1))
NAMES = ("Rest", "E", "W", "S", "N", "SW", "NE", "NW", "SE")
D3_IDS = (0, 1, 2, 5, 6, 11, 12, 15, 16)
INV = (0, 2, 1, 4, 3, 6, 5, 8, 7)
CHILDREN = ((0, 0), (1, 0), (0, 1), (1, 1))
ALL = (1 << 9) - 1


def add(p, d):
    return p[0] + d[0], p[1] + d[1]


def inside(p, width, height):
    return 0 <= p[0] < width and 0 <= p[1] < height


def spread(cells, domain):
    return {add(p, d) for p in cells for d in DIRS} & domain


@dataclass
class Config:
    width: int = WIDTH
    height: int = HEIGHT
    left: str = LEFT
    right: str = RIGHT
    top: str = TOP
    bottom: str = BOTTOM
    ibb: bool = APPLY_IBB

    def check(self):
        if not (isinstance(self.width, int) and isinstance(self.height, int)
                and self.width >= 3 and self.height >= 3):
            raise ValueError("WIDTH and HEIGHT must be integers of at least 3.")
        if any(v not in (MBBC, PERIODIC) for v in
               (self.left, self.right, self.top, self.bottom)):
            raise ValueError("Each boundary must be MBBC or PERIODIC.")


class Grid:
    def __init__(self, config, level, cells, walls):
        self.config, self.level = config, level
        self.width = config.width * (1 << level)
        self.height = config.height * (1 << level)
        self.cells = sorted(cells, key=lambda p: (p[1], p[0]))
        self.index = {p: n for n, p in enumerate(self.cells)}
        self.walls = set(walls) & set(cells)
        self.fluid = set(cells) - self.walls
        self.ip, self.jp, self.jm = {}, {}, {}
        columns = defaultdict(list)
        for n, p in enumerate(self.cells):
            self.ip[p] = self.cells[(n + 1) % len(self.cells)]
            columns[p[0]].append(p)
        self.im = {v: k for k, v in self.ip.items()}
        for column in columns.values():
            for n, p in enumerate(column):
                self.jp[p] = column[(n + 1) % len(column)]
                self.jm[p] = column[(n - 1) % len(column)]

        # Match post-collision writes to next-pre-collision reads. The flipper
        # changes the population slot, but cancels out of this owner mapping.
        self.source = {}
        for q, d in enumerate(DIRS):
            writer = {self.slot(p, d, post=True): p for p in self.cells}
            for p in self.cells:
                self.source[p, q] = writer[self.slot(p, d, post=False)]
        self.links = {p: self.wall_links(p) for p in self.fluid}
        self.periodic = {p: self.periodic_donor(p) for p in self.fluid
                         if self.boundary_type(p) == PERIODIC}
        self.reset()

    def slot(self, p, d, post):
        x, y = d if post else (-d[0], -d[1])
        if y > 0:
            p = self.jp[p]
        if x > 0:
            p = self.ip[p]
        return p

    def expected_source(self, p, q):
        dx, dy = DIRS[q]
        return p[0] - dx, p[1] - dy

    def outgoing_neighbour(self, p, q):
        # buildLinkExistenceMarkerArray uses J after I for its diagonal checks.
        dx, dy = DIRS[q]
        if dx:
            p = (self.ip if dx > 0 else self.im)[p]
        if dy:
            p = (self.jp if dy > 0 else self.jm)[p]
        return p

    def boundary_sides(self, p):
        result = []
        if p[0] == 0:
            result.append(self.config.left)
        if p[0] == self.width - 1:
            result.append(self.config.right)
        if p[1] == 0:
            result.append(self.config.bottom)
        if p[1] == self.height - 1:
            result.append(self.config.top)
        return result

    def boundary_type(self, p):
        if p in self.walls:
            return "Wall"
        sides = self.boundary_sides(p)
        return MBBC if MBBC in sides else PERIODIC if sides else None

    def crossing(self, p):
        return [q for q in range(9)
                if not inside(self.expected_source(p, q), self.width, self.height)]

    def periodic_donor(self, p):
        x, y = p
        shift = 2 * (1 << self.level)
        if x == 0:
            x = self.width - shift
        elif x == self.width - 1:
            x = shift - 1
        if y == 0:
            y = self.height - shift
        elif y == self.height - 1:
            y = shift - 1
        return x, y

    def wall_links(self, p):
        if not self.config.ibb:
            return set()
        geometric, missing_inside = {}, []
        for q in range(1, 9):
            expected = add(p, DIRS[q])
            geometric[q] = self.outgoing_neighbour(p, q) == expected
            if not geometric[q] and inside(expected, self.width, self.height):
                missing_inside.append(DIRS[q])
        if not self.boundary_sides(p) and missing_inside:
            return set()
        result = set()
        for q in range(1, 9):
            if not geometric[q] or self.outgoing_neighbour(p, q) not in self.walls:
                continue
            d = DIRS[q]
            if self.boundary_sides(p) and any(
                    any(d[a] * m[a] > 0 for a in (0, 1)) for m in missing_inside):
                continue
            result.add(q)
        return result

    def reset(self):
        self.f = {p: [True] * 9 for p in self.fluid}
        self.why = {p: ["Initial valid population."] * 9 for p in self.fluid}
        self.pre = {}
        self.reflections = {}
        self.phase = "Initial / pre-collision"

    def collision(self):
        self.pre = copy.deepcopy(self.f)
        for p in self.fluid:
            bad = [NAMES[q] for q, v in enumerate(self.pre[p]) if not v]
            good = not bad
            self.f[p] = [good] * 9
            reason = ("Valid collision: all 9 inputs valid." if good else
                      "Collision used invalid inputs: " + ", ".join(bad))
            self.why[p] = [reason] * 9
        self.reflections = {}
        for p, links in self.links.items():
            for q in links:
                # IBB also reads rho from all post-collision populations.
                valid = self.pre[p][q] and all(self.f[p])
                self.reflections[p, INV[q]] = valid
        self.phase = "Post-collision; IBB prepared for streaming"

    def streaming(self):
        post = copy.deepcopy(self.f)
        for p in self.fluid:
            for q in range(9):
                if (p, q) in self.reflections:
                    v = self.reflections[p, q]
                    reason = "Local IBB reflection; " + ("valid inputs." if v else "invalid local inputs.")
                else:
                    s = self.source[p, q]
                    expected = self.expected_source(p, q)
                    if s != expected:
                        v = False
                        reason = f"Non-geometric stream: actual {s}, expected {expected}."
                    elif s not in self.fluid:
                        v = False
                        reason = f"Wall source {s}; no enabled IBB link for this population."
                    else:
                        v = post[s][q]
                        reason = f"Streamed from {s}, {NAMES[q]}: " + ("valid." if v else "invalid collision output.")
                self.f[p][q], self.why[p][q] = v, reason
        self.phase = "Pre-collision populations after streaming"

    def mbbc(self):
        for p in self.fluid:
            if self.boundary_type(p) != MBBC:
                continue
            incoming = self.crossing(p)
            bad = [NAMES[q] for q in range(9) if q not in incoming and not self.f[p][q]]
            for q in incoming:
                self.f[p][q] = not bad
                self.why[p][q] = ("MBBC: all non-crossing inputs valid." if not bad else
                                  "MBBC used invalid non-crossing inputs: " + ", ".join(bad))
        self.phase = "After MBBC"

    def copy_periodic(self, affected=None):
        before = copy.deepcopy(self.f)
        count = 0
        for p, s in self.periodic.items():
            if affected is not None and s not in affected:
                continue
            count += 1
            if s not in self.fluid:
                self.f[p] = [False] * 9
                self.why[p] = [f"Periodic donor {s} is missing or wall; LBM3 would reject this layout."] * 9
            else:
                self.f[p] = before[s].copy()
                self.why[p] = [f"Periodic copy from {s}: " + ("valid." if v else "donor invalid.")
                               for v in before[s]]
        self.phase = "After periodic BC" if affected is None else "After interface periodic refresh"
        return count


# ------------------- EDIT THE EXPERIMENTAL MASK RULE HERE -------------------
def source_outermost(grid):
    """Geometric rule plus periodic-source correction; does not consult truth."""
    raw = {p: any(inside(add(p, d), grid.width, grid.height)
                  and add(p, d) not in grid.index for d in DIRS[1:])
           for p in grid.cells}
    corrected = raw.copy()
    for p, donor in grid.periodic.items():
        corrected[p] = raw.get(donor, True) or donor in grid.walls
    return corrected


def interface_mask(grid, p, outermost):
    """Current proposed mask. Bit 1 requests interface reconstruction.

    Intentionally experimental near walls: baseline geometry is not silently
    replaced by the true validity. Inspect mask/actual mismatches to develop
    the wall correction. Periodic zero means 'handled by periodic refresh'.
    """
    mask = 0
    for q in range(9):
        s = grid.source[p, q]
        if s != grid.expected_source(p, q) or outermost.get(s, True):
            mask |= 1 << q
    bc = grid.boundary_type(p)
    if bc == MBBC:
        incoming = sum(1 << q for q in grid.crossing(p))
        if mask & (ALL ^ incoming) == 0:
            mask = 0
    if bc == PERIODIC:
        mask = 0
    return mask


class Experiment:
    def __init__(self, config=None, deep=(), walls=()):
        self.config = config or Config()
        self.config.check()
        self.domain = set((i, j) for j in range(self.config.height)
                          for i in range(self.config.width))
        self.deep_paint = set(deep) & self.domain
        self.wall_paint = set(walls) & self.domain
        self.rebuild()

    def rebuild(self):
        kept = spread(self.domain - self.wall_paint, self.domain)
        self.deep = self.deep_paint & kept
        self.extension = spread(self.deep, self.domain) & kept - self.deep
        self.interface = spread(self.extension, self.domain) & kept - self.deep - self.extension
        refined = self.deep | self.extension | self.interface
        coarse_cells = kept - self.deep
        fine_full = {(2*p[0]+b[0], 2*p[1]+b[1]) for p in refined for b in CHILDREN}
        fine_walls = {p for p in fine_full if (p[0]//2, p[1]//2) in self.wall_paint}
        fine_overlap = {p for p in fine_full if (p[0]//2, p[1]//2) in coarse_cells}
        fine_keep = spread((fine_full - fine_walls) | fine_overlap, fine_full)
        self.grids = [Grid(self.config, 0, coarse_cells, self.wall_paint),
                      Grid(self.config, 1, fine_keep, fine_walls)]
        self.coarse_targets = self.interface & self.grids[0].fluid
        self.fine_targets = {p for p in self.grids[1].fluid
                             if (p[0]//2, p[1]//2) in self.coarse_targets}
        self.masks = []
        for grid, targets in zip(self.grids, (self.coarse_targets, self.fine_targets)):
            outer = source_outermost(grid)
            self.masks.append({p: interface_mask(grid, p, outer) for p in targets})
        self.mask_mode = "Rules"
        self.steps = []
        for level, label in ((0, "Coarse"), (1, "Fine 1/2"), (1, "Fine 2/2")):
            for action, title in (("collision", "collision + IBB" if self.config.ibb else "collision"),
                                  ("streaming", "streaming"), ("mbbc", "MBBC"),
                                  ("periodic", "periodic BC")):
                self.steps.append((f"{label} {title}", level, action))
        self.steps += [("Interface F→C", 0, "fc"),
                       ("Coarse periodic refresh", 0, "refresh"),
                       ("Interface C→F", 1, "cf"),
                       ("Fine periodic refresh", 1, "refresh")]
        self.reset()

    def reset(self):
        for grid in self.grids:
            grid.reset()
        coarse = self.grids[0]
        for p in self.extension & coarse.fluid:
            coarse.f[p] = [False]*9
            coarse.why[p] = ["Coarse extension populations start invalid."]*9
        self.cursor = 0
        self.history = []
        self.failures = {}
        self.message = ("Coarse extension populations start invalid; other fluid populations start valid. "
                        "Paint a layout, step, or build smart masks.")

    def build_smart_masks(self):
        """Observe a fresh cycle and freeze both masks immediately before F->C."""
        self.reset()
        while self.steps[self.cursor][2] != "fc":
            self.step()
        # Capture both grids now, before either transfer or periodic refresh.
        # Include periodic targets: the observed mask is exact even where the
        # existing transfer procedure defers reconstruction to periodic copies.
        self.history.append(self.snapshot())
        self.masks = [{p: sum(1 << q for q, valid in enumerate(grid.f[p]) if not valid)
                       for p in targets}
                      for grid, targets in zip(self.grids, (self.coarse_targets, self.fine_targets))]
        self.mask_mode = "Smart (before F→C)"
        counts = [sum(bin(mask).count("1") for mask in masks.values()) for masks in self.masks]
        self.message = (f"Smart masks built: {counts[0]} coarse and {counts[1]} fine populations requested. "
                        "Stopped before F→C; no interface transfers have run.")

    def next_title(self):
        return self.steps[self.cursor % len(self.steps)][0]

    def snapshot(self):
        return ([copy.deepcopy((g.f, g.why, g.pre, g.reflections, g.phase)) for g in self.grids],
                self.cursor, self.failures.copy(), self.message,
                [m.copy() for m in self.masks], self.mask_mode)

    def back(self):
        if not self.history:
            return
        (grids, self.cursor, self.failures, self.message,
         self.masks, self.mask_mode) = self.history.pop()
        for g, state in zip(self.grids, grids):
            g.f, g.why, g.pre, g.reflections, g.phase = state

    def step(self):
        self.history.append(self.snapshot())
        title, level, action = self.steps[self.cursor % len(self.steps)]
        grid = self.grids[level]
        if action in ("collision", "streaming", "mbbc"):
            getattr(grid, action)()
            self.message = title + " finished."
        elif action == "periodic":
            n = grid.copy_periodic()
            self.message = f"{title}: copied {n} cells."
        elif action == "refresh":
            affected = self.coarse_targets if level == 0 else self.fine_targets
            n = grid.copy_periodic(affected)
            self.message = f"{title}: copied {n} cells whose donors belong to this interface."
        else:
            self.transfer(action)
        for key in list(self.failures):
            failed_level, p = key
            if all(self.grids[failed_level].f[p]):
                del self.failures[key]
        self.cursor += 1
        if self.cursor % len(self.steps) == 12:
            self.message += "  Synchronized: the optional interface transfer is next."
        if self.cursor % len(self.steps) == 0:
            self.message += "  Cycle complete; the next cycle keeps the resulting validity."

    def coarse_stencil(self, parent, child):
        coarse = self.grids[0]
        result = [parent]
        descriptions = []
        for axis in (0, 1):
            d = [0, 0]
            d[axis] = 1 if child[axis] % 2 else -1
            p, opposite = add(parent, d), add(parent, (-d[0], -d[1]))
            if p in coarse.fluid:
                result.append(p)
            elif opposite in coarse.fluid:
                result.append(opposite)
                descriptions.append(f"opposite fallback {opposite}")
            else:
                descriptions.append(f"constant fallback, axis {'IJ'[axis]}")
        return list(dict.fromkeys(result)), descriptions

    def transfer(self, action):
        level = 0 if action == "fc" else 1
        target_grid, donor_grid = self.grids[level], self.grids[1-level]
        targets = self.coarse_targets if level == 0 else self.fine_targets
        before = copy.deepcopy(donor_grid.f)
        failed = repaired = skipped = unnecessary = 0
        for p in sorted(targets):
            self.failures.pop((level, p), None)
            if target_grid.boundary_type(p) == PERIODIC:
                skipped += 1
                continue
            mask = self.masks[level][p]
            errors = [f"assumed-known {NAMES[q]} was invalid" for q in range(9)
                      if not mask & (1 << q) and not target_grid.f[p][q]]
            unnecessary += sum(bool(mask & (1 << q)) and target_grid.f[p][q] for q in range(9))
            if action == "fc":
                donors = [(2*p[0]+b[0], 2*p[1]+b[1]) for b in CHILDREN]
                fallbacks = []
            else:
                parent = p[0]//2, p[1]//2
                donors, fallbacks = self.coarse_stencil(parent, p)
            for q in range(9):
                if not mask & (1 << q):
                    continue
                for donor in donors:
                    if donor not in before or not before[donor][q]:
                        errors.append(f"donor {donor} / {NAMES[q]} invalid or absent")
            if errors:
                failed += 1
                self.failures[level, p] = errors
                target_grid.f[p] = [False] * 9
                target_grid.why[p] = ["Interface failed: " + "; ".join(errors[:4])] * 9
            elif mask:
                repaired += 1
                for q in range(9):
                    if mask & (1 << q):
                        target_grid.f[p][q] = True
                        target_grid.why[p][q] = ("Interface: valid known populations and donors "
                                                + str(donors) + ("; " + ", ".join(fallbacks) if fallbacks else ""))
        target_grid.phase = "After experimental " + ("F→C" if level == 0 else "C→F")
        self.message = (f"{target_grid.phase}: {repaired} repaired, {failed} failed (whole cell red), "
                        f"{skipped} periodic targets deferred. {unnecessary} valid populations requested.")


# ------------------------------ GUI ----------------------------------------
COLORS = dict(coarse="#2563c9", fine="#07836c", invalid="#dc303e",
              deep="#e5ddf7", extension="#fce6af", interface="#cce9f3",
              wall="#667085", removed="#f3f4f6", normal="#ffffff")


class App:
    def __init__(self, root, config=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.exp = Experiment(config, INITIAL_DEEP, INITIAL_WALLS)
        self.selected = None
        self.q = 1
        self.drag = None
        root.title("LBM · coarse / fine validity")
        root.geometry("1380x900")
        root.minsize(980, 700)
        root.configure(bg="#f5f7fa")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fa")
        style.configure("TLabel", background="#f5f7fa", foreground="#182434", font=("DejaVu Sans", 10))
        style.configure("TButton", font=("DejaVu Sans", 10), padding=(9, 6))
        style.configure("Go.TButton", foreground="white", background="#245bc2", font=("DejaVu Sans", 11, "bold"))
        header = ttk.Frame(root, padding=(18, 13))
        header.pack(fill="x")
        ttk.Label(header, text="Coarse / fine validity", font=("DejaVu Sans", 18, "bold")).pack(side="left")
        self.progress = ttk.Label(header, text="")
        self.progress.pack(side="right")
        controls = ttk.Frame(root, padding=(18, 0, 18, 9))
        controls.pack(fill="x")
        self.tool = tk.StringVar(value="refine")
        for text, value in (("Refine", "refine"), ("Wall", "wall"), ("Erase", "erase"), ("Inspect", "inspect")):
            ttk.Radiobutton(controls, text=text, value=value, variable=self.tool).pack(side="left", padx=(0, 10))
        ttk.Button(controls, text="Build smart masks", command=self.build_smart_masks).pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="Clear layout", command=self.clear).pack(side="right")
        ttk.Button(controls, text="Load", command=self.load).pack(side="right", padx=5)
        ttk.Button(controls, text="Save", command=self.save).pack(side="right")
        body = ttk.Frame(root, padding=(18, 0, 18, 0))
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg="white", highlightthickness=1, highlightbackground="#dbe1e8")
        self.canvas.pack(side="left", fill="both", expand=True)
        sidebar = ttk.Frame(body, width=310)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)
        panel_canvas = tk.Canvas(sidebar, bg="#f5f7fa", highlightthickness=0)
        panel_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=panel_canvas.yview)
        panel_scroll.pack(side="right", fill="y")
        panel_canvas.pack(side="left", fill="both", expand=True)
        panel_canvas.configure(yscrollcommand=panel_scroll.set)
        panel = ttk.Frame(panel_canvas, padding=(12, 0, 0, 0))
        panel_window = panel_canvas.create_window(0, 0, anchor="nw", window=panel)
        panel.bind("<Configure>", lambda e: panel_canvas.configure(scrollregion=panel_canvas.bbox("all")))
        panel_canvas.bind("<Configure>", lambda e: panel_canvas.itemconfigure(panel_window, width=e.width))
        self.panel_canvas = panel_canvas
        self.stats = ttk.Label(panel, text="", justify="left")
        self.stats.pack(anchor="w", pady=(0, 10))
        for key, label in (("deep", "Deep refinement · coarse removed"),
                           ("extension", "Extension · coarse retained"),
                           ("interface", "Communication layer"), ("wall", "Wall"),
                           ("coarse", "Valid coarse population"), ("fine", "Valid fine population"),
                           ("invalid", "Invalid population · either grid")):
            row = ttk.Frame(panel)
            row.pack(fill="x", pady=2)
            tk.Label(row, bg=COLORS[key], width=2).pack(side="left", padx=(0, 7))
            ttk.Label(row, text=label, font=("DejaVu Sans", 9)).pack(side="left")
        self.show_coarse = tk.BooleanVar(value=True)
        self.show_fine = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=False)
        for text, var in (("Show coarse", self.show_coarse), ("Show fine", self.show_fine),
                          ("Ring populations requested by mask", self.show_mask)):
            ttk.Checkbutton(panel, text=text, variable=var, command=self.redraw).pack(anchor="w", pady=2)
        ttk.Separator(panel).pack(fill="x", pady=10)
        self.info = ttk.Label(panel, text="Inspect a cell to see its populations.", wraplength=280, justify="left")
        self.info.pack(fill="x")
        self.rows = []
        for q in range(9):
            button = tk.Button(panel, text="", anchor="w", relief="flat", bg="#f5f7fa",
                               font=("DejaVu Sans Mono", 9), command=lambda q=q: self.pick_direction(q))
            button.pack(fill="x", pady=0)
            self.rows.append(button)
        self.reason = ttk.Label(panel, text="", wraplength=280, justify="left", font=("DejaVu Sans", 9))
        self.reason.pack(fill="x", pady=7)
        footer = ttk.Frame(root, padding=(18, 10))
        footer.pack(fill="x")
        self.next_button = ttk.Button(footer, text="", command=self.step, style="Go.TButton")
        self.next_button.pack(side="left")
        self.back_button = ttk.Button(footer, text="Back", command=self.back)
        self.back_button.pack(side="left", padx=6)
        ttk.Button(footer, text="Reset validity", command=self.reset).pack(side="left")
        ttk.Label(footer, text="Space: next · Backspace: back · R: reset").pack(side="right")
        self.status = ttk.Label(root, text="", wraplength=1300, padding=(18, 0, 18, 5))
        self.status.pack(fill="x")
        ttk.Label(root, text="Click / drag to paint. Right-click: wall. Inspect: click a cell, then a direction. "
                  "Rest = dot; arrows point along population velocity. Editing resets validity.",
                  font=("DejaVu Sans", 9), padding=(18, 0, 18, 11)).pack(fill="x")
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<Button-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "drag", None))
        self.canvas.bind("<Button-3>", lambda e: self.press(e, forced="wall"))
        root.bind("<space>", lambda e: self.step())
        root.bind("<BackSpace>", lambda e: self.back())
        root.bind("r", lambda e: self.reset())
        root.bind("<MouseWheel>", lambda e: panel_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        root.bind("<Button-4>", lambda e: panel_canvas.yview_scroll(-1, "units"))
        root.bind("<Button-5>", lambda e: panel_canvas.yview_scroll(1, "units"))
        root.after_idle(self.redraw)

    def transform(self):
        w, h = max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height())
        c = self.exp.config
        self.scale = min((w - 85) / c.width, (h - 85) / c.height)
        self.left = (w - c.width * self.scale) / 2
        self.bottom = (h + c.height * self.scale) / 2

    def xy(self, x, y):
        return self.left + x*self.scale, self.bottom - y*self.scale

    def center(self, level, p):
        factor = 1 << level
        return self.xy((p[0]+0.5)/factor, (p[1]+0.5)/factor)

    def coarse_at(self, event):
        p = (math.floor((event.x-self.left)/self.scale),
             math.floor((self.bottom-event.y)/self.scale))
        return p if p in self.exp.domain else None

    def press(self, event, forced=None):
        tool = forced or self.tool.get()
        p = self.coarse_at(event)
        if p is None:
            return
        if tool == "inspect":
            choices = [(math.dist((event.x, event.y), self.center(level, cell)), level, cell)
                       for level, grid in enumerate(self.exp.grids)
                       if (self.show_coarse.get() if level == 0 else self.show_fine.get())
                       for cell in grid.cells]
            if choices:
                _, level, cell = min(choices)
                self.selected = level, cell
                self.redraw()
                self.panel_canvas.yview_moveto(0.32)
            return
        target = self.exp.deep_paint if tool == "refine" else self.exp.wall_paint
        self.drag = (tool, p not in target, set())
        self.paint(p)

    def motion(self, event):
        if self.drag:
            p = self.coarse_at(event)
            if p is not None:
                self.paint(p)

    def paint(self, p):
        tool, on, visited = self.drag
        if p in visited:
            return
        visited.add(p)
        if tool == "erase":
            self.exp.deep_paint.discard(p)
            self.exp.wall_paint.discard(p)
        else:
            target = self.exp.deep_paint if tool == "refine" else self.exp.wall_paint
            (target.add if on else target.discard)(p)
        self.exp.rebuild()
        self.redraw()

    def pick_direction(self, q):
        self.q = q
        self.redraw()

    def step(self):
        self.exp.step()
        self.redraw()

    def back(self):
        self.exp.back()
        self.redraw()

    def reset(self):
        self.exp.reset()
        self.redraw()

    def build_smart_masks(self):
        self.exp.build_smart_masks()
        self.show_mask.set(True)
        self.redraw()

    def clear(self):
        self.exp.deep_paint.clear()
        self.exp.wall_paint.clear()
        self.exp.rebuild()
        self.selected = None
        self.redraw()

    def save(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Layout", "*.json")])
        if path:
            data = dict(config=vars(self.exp.config), deep=sorted(self.exp.deep_paint),
                        walls=sorted(self.exp.wall_paint))
            try:
                Path(path).write_text(json.dumps(data, indent=2))
            except OSError as error:
                from tkinter import messagebox
                messagebox.showerror("Cannot save layout", str(error))
                return
            self.exp.message = "Layout saved. Validity states are reset when a layout is loaded."
            self.redraw()

    def load(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(filetypes=[("Layout", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            self.exp = Experiment(Config(**data["config"]), map(tuple, data["deep"]), map(tuple, data["walls"]))
            self.selected = None
            self.redraw()
        except (ValueError, TypeError, KeyError, OSError) as error:
            messagebox.showerror("Cannot load layout", str(error))

    def redraw(self):
        self.transform()
        cv, e, s = self.canvas, self.exp, self.scale
        cv.delete("all")
        for p in sorted(e.domain):
            region = ("deep" if p in e.deep else "extension" if p in e.extension
                      else "interface" if p in e.interface else "normal")
            x0, y0 = self.xy(p[0], p[1]+1)
            x1, y1 = self.xy(p[0]+1, p[1])
            cv.create_rectangle(x0, y0, x1, y1, fill=COLORS[region], outline="#c8d0dc", width=1)
            if p in e.wall_paint:
                cv.create_rectangle(x0+3, y0+3, x1-3, y1-3, fill=COLORS["wall"], outline="")
                if p in e.deep_paint:
                    cv.create_text(x0+8, y0+9, text="D", fill="white", anchor="w", font=("DejaVu Sans", 8))
            elif p in e.deep:
                x, y = self.center(0, p)
                cv.create_line(x-3, y-3, x+3, y+3, fill="#b4a6ce")
                cv.create_line(x-3, y+3, x+3, y-3, fill="#b4a6ce")
        # Fine grid outlines are light; cell centres and all arrows remain apart.
        if self.show_fine.get():
            for p in e.grids[1].cells:
                x, y = self.center(1, p)
                r = s/4
                cv.create_rectangle(x-r, y-r, x+r, y+r, outline="#c5cfd5", dash=(2, 4))
        for level, grid in enumerate(e.grids):
            if not (self.show_coarse.get() if level == 0 else self.show_fine.get()):
                continue
            length = s*(0.185 if level == 0 else 0.13)
            for p in grid.fluid:
                x, y = self.center(level, p)
                mask = e.masks[level].get(p, 0)
                for q, (dx, dy) in enumerate(DIRS):
                    color = COLORS["invalid"] if not grid.f[p][q] else COLORS["coarse" if level == 0 else "fine"]
                    if q == 0:
                        r = 3.4 if level == 0 else 2.4
                        cv.create_oval(x-r, y-r, x+r, y+r, fill=color, outline="white", width=0.5)
                        ex, ey = x, y
                    else:
                        norm = math.hypot(dx, dy)
                        ex, ey = x+length*dx/norm, y-length*dy/norm
                        start = min(5 if level == 0 else 3.6, length*0.4)
                        head = min(1.0, length/9)
                        head_shape = (4.5, 5.5, 2.2) if level == 0 else (3.2, 4, 1.5)
                        cv.create_line(x+start*dx/norm, y-start*dy/norm, ex, ey,
                                       fill=color, width=1.7 if level == 0 else 1.1,
                                       arrow="last", arrowshape=tuple(a*head for a in head_shape))
                    if self.show_mask.get() and mask & (1 << q):
                        cv.create_oval(ex-3, ey-3, ex+3, ey+3, outline="#9e681c", width=1)
                if (level, p) in e.failures:
                    r = length+3
                    cv.create_rectangle(x-r, y-r, x+r, y+r, outline=COLORS["invalid"], dash=(3, 2), width=2)
        for i in range(e.config.width):
            x, y = self.xy(i+0.5, 0)
            cv.create_text(x, y+15, text=str(i), fill="#64748b", font=("DejaVu Sans", 9))
        for j in range(e.config.height):
            x, y = self.xy(0, j+0.5)
            cv.create_text(x-15, y, text=str(j), fill="#64748b", font=("DejaVu Sans", 9))
        w, h = e.config.width, e.config.height
        for x, y, text, anchor in ((w/2, h, "TOP · "+e.config.top, "s"),
                                   (w/2, 0, "BOTTOM · "+e.config.bottom, "n")):
            px, py = self.xy(x, y)
            cv.create_text(px, py+(-15 if y else 29), text=text, anchor=anchor, fill="#43536a", font=("DejaVu Sans", 9))
        self.inspect()
        stats = []
        missing = 0
        for label, grid in zip(("Coarse", "Fine"), e.grids):
            invalid = sum(not v for values in grid.f.values() for v in values)
            stats.append(f"{label}: {len(grid.fluid)} fluid cells\n{invalid} / {9*len(grid.fluid)} populations invalid")
            missing += sum(s not in grid.fluid for s in grid.periodic.values())
        self.stats.configure(text="\n\n".join(stats)+f"\n\nLeft: {e.config.left}  Right: {e.config.right}"
                             f"\nMasks: {e.mask_mode}")
        self.progress.configure(text=f"Cycle {e.cursor//len(e.steps)+1}  ·  stage {e.cursor%len(e.steps)}/{len(e.steps)}")
        self.next_button.configure(text="Next step: " + e.next_title())
        self.back_button.configure(state="normal" if e.history else "disabled")
        message = e.message
        if missing:
            message += f"  Layout issue: {missing} periodic donors are absent/wall; these copies turn red."
        self.status.configure(text=message)

    def inspect(self):
        e, cv = self.exp, self.canvas
        if self.selected is None or self.selected[1] not in e.grids[self.selected[0]].index:
            self.info.configure(text="Inspect a cell to see its populations.\nMask = interface reconstruction request.")
            self.reason.configure(text="")
            for row in self.rows:
                row.configure(text="", state="disabled")
            return
        level, p = self.selected
        grid = e.grids[level]
        x, y = self.center(level, p)
        r = self.scale*(0.22 if level == 0 else 0.14)
        cv.create_rectangle(x-r, y-r, x+r, y+r, outline="#17263d", width=2)
        bc = grid.boundary_type(p) or "fluid"
        target = p in e.masks[level]
        self.info.configure(text=f"{'Coarse' if level == 0 else 'Fine'} {p} · {bc}\n"
                                 f"{'Communication target' if target else 'Not a communication target'}\n{grid.phase}")
        if p not in grid.fluid:
            for row in self.rows:
                row.configure(text="", state="disabled")
            self.reason.configure(text="Wall cells do not collide or emit fluid populations.")
            return
        mask = e.masks[level].get(p, 0)
        for q, row in enumerate(self.rows):
            valid = grid.f[p][q]
            request = bool(mask & (1 << q))
            suffix = ("repair" if request else "keep") if target else "—"
            if bc == PERIODIC and target:
                suffix = "periodic"
            row.configure(text=f"{NAMES[q]:4} [{D3_IDS[q]:2}]  {'valid  ' if valid else 'INVALID'}  {suffix}",
                          fg=COLORS["coarse" if level == 0 else "fine"] if valid else COLORS["invalid"],
                          bg="#e3eaf4" if q == self.q else "#f5f7fa", state="normal")
        q = self.q
        source = grid.source[p, q]
        expected = grid.expected_source(p, q)
        text = grid.why[p][q]
        text += f"\n\nRaw stream source: {source}\nGeometric source: {expected}"
        if bc == PERIODIC:
            text += f"\nPeriodic donor: {grid.periodic[p]}"
        if INV[q] in grid.links[p]:
            text += "\nThis incoming direction has an enabled IBB link."
        if target and bc != PERIODIC:
            if not mask & (1 << q) and not grid.f[p][q]:
                text += "\nMask currently assumes this invalid population is known."
            elif mask & (1 << q) and grid.f[p][q]:
                text += "\nMask requests a currently valid population."
        self.reason.configure(text=text)
        if source != p:
            sx, sy = self.center(level, source)
            cv.create_line(sx, sy, x, y, dash=(4, 3), width=1.5, arrow="last",
                           fill="#7c5bb1" if source == expected else COLORS["invalid"])


def main():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print("A graphical desktop and Tkinter are required. On Debian/Ubuntu: sudo apt install python3-tk", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
