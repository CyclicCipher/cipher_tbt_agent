"""sheet.py — neocortex as a 2D cortical sheet.

The neocortex is a single flat sheet of cortical columns. This module
models it as a 2D plane of named regions connected by arbitrary typed
projections. No modality code lives here; the sheet doesn't know about
eyes, audio, or text. Sensors attach externally via brain.observe().

  Region     — named rectangular patch of the sheet containing a grid of
               MacroColumns.  Corresponds to a cortical area (V1, IT, PFC…).

  Connection — typed directed projection between two Regions.
               kind: 'feedforward' | 'feedback' | 'lateral' | 'modulatory'

  BrainSheet — directed graph of Regions.  Runs the processing cycle:
               begin → observe → forward → lateral → feedback → commit.

YAML config format
------------------
regions:
  V1:
    x: 0    y: 0    w: 60   h: 60
    rows: 6  cols: 6  n_mini: 1
  IT:
    x: 10   y: 80   w: 40   h: 40
    rows: 3  cols: 1  n_mini: 10

connections:
  - src: V1   dst: IT   kind: feedforward   strength: 1.0   projection: topographic
  - src: IT   dst: V1   kind: feedback      strength: 0.3   projection: topographic
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from column import MacroColumn
from reference_frames import AllocentricFrame


# ── Declarative specs (pure data, no column objects) ─────────────────────────

@dataclass
class RegionSpec:
    name:   str
    x:      float           # position on the 2D sheet (top-left corner, any unit)
    y:      float
    w:      float           # size on the sheet
    h:      float
    rows:   int             # column grid dimensions
    cols:   int
    n_mini: int  = 1        # minicolumns per macrocolumn
    meta:   dict = field(default_factory=dict)   # pass-through extras (e.g. encoder)


@dataclass
class ConnectionSpec:
    src:        str           # source region name
    dst:        str           # destination region name
    kind:       str           # 'feedforward' | 'feedback' | 'lateral' | 'modulatory'
    strength:   float = 1.0
    projection: str   = 'topographic'   # 'topographic' | 'all_to_all' | 'one_to_one'


# ── Region ────────────────────────────────────────────────────────────────────

class Region:
    """One cortical area: a grid of MacroColumns with shared I/O state.

    Each column has:
      _outputs[i]  — the SDR it is currently outputting (injected or computed)
      _pos[i]      — its (gy, gx) position key in the region grid

    Columns output their injected SDR when acting as sensors, and output a
    one-hot of their winner index after commit() when acting as higher areas.
    """

    def __init__(self, spec: RegionSpec) -> None:
        self.spec     = spec
        self.name     = spec.name
        self.columns:  list[MacroColumn]       = []
        self._outputs: list[np.ndarray | None] = []
        self._pos:     list[tuple[int, int]]   = []

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Instantiate MacroColumns. Called once before training begins."""
        self.columns = [
            MacroColumn(
                frame=AllocentricFrame(position=(float(gx), float(gy))),
                n_mini=self.spec.n_mini,
            )
            for gy in range(self.spec.rows)
            for gx in range(self.spec.cols)
        ]
        n = len(self.columns)
        self._outputs = [None] * n
        self._pos     = [
            (gy, gx)
            for gy in range(self.spec.rows)
            for gx in range(self.spec.cols)
        ]

    def begin(self) -> None:
        """Reset evidence accumulators and clear output buffers."""
        for col in self.columns:
            col.begin_image()
        self._outputs = [None] * len(self.columns)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def inject(self, col_idx: int, sdr: np.ndarray) -> None:
        """Inject one sensor SDR into column col_idx (also sets its output)."""
        col = self.columns[col_idx]
        loc = self._pos[col_idx]
        col.observe_multi([(sdr, loc)])
        self._outputs[col_idx] = sdr

    def deliver(self, col_idx: int, obs: list[tuple[np.ndarray, tuple]]) -> None:
        """Deliver a list of (sdr, location_key) pairs to one column."""
        if obs:
            self.columns[col_idx].observe_multi(obs)

    def commit(self, labels: dict[int, int] | None = None,
               write: bool = True) -> list[int]:
        """WTA commit for all columns.  labels[col_idx] = class_label for supervised."""
        winners = []
        for i, col in enumerate(self.columns):
            lbl = labels.get(i) if labels else None
            w   = (col.commit_supervised(lbl, write=write)
                   if lbl is not None else col.commit(write=write))
            winners.append(w)
            # Update output to one-hot winner for downstream regions
            vec = np.zeros(col.N_MINI, dtype=np.int8)
            vec[w] = 1
            self._outputs[i] = vec
        return winners

    # ── accessors ─────────────────────────────────────────────────────────────

    def n_cols(self) -> int:                   return len(self.columns)
    def output(self, idx: int):                return self._outputs[idx]
    def pos(self, idx: int) -> tuple:          return self._pos[idx]
    def winner(self, idx: int) -> int:         return self.columns[idx].tentative_winner()


# ── BrainSheet ────────────────────────────────────────────────────────────────

class BrainSheet:
    """Directed graph of cortical regions on a 2D sheet.

    Typical per-image cycle
    -----------------------
    brain.begin()
    brain.observe('V1', [(sdr, col_idx), ...])   # inject encoder output
    brain.forward()                               # feedforward sweep
    brain.lateral()                               # lateral voting
    brain.feedback()                              # top-down predictions
    brain.commit('IT', labels={0: 3, 1: 3, 2: 3})  # supervised
    brain.commit('V1')                              # unsupervised
    """

    def __init__(self) -> None:
        self.regions:     dict[str, Region]     = {}
        self._order:      list[str]             = []      # insertion order
        self.connections: list[ConnectionSpec]  = []
        self._rfs:        list[list[list[int]]] = []      # precomputed RF per connection
        self._built = False

    # ── graph construction ────────────────────────────────────────────────────

    def add_region(self, spec: RegionSpec) -> 'BrainSheet':
        if spec.name in self.regions:
            raise ValueError(f"Region '{spec.name}' already exists")
        self.regions[spec.name] = Region(spec)
        self._order.append(spec.name)
        return self

    def add_connection(self, spec: ConnectionSpec) -> 'BrainSheet':
        if spec.src not in self.regions:
            raise ValueError(f"Unknown source region '{spec.src}'")
        if spec.dst not in self.regions:
            raise ValueError(f"Unknown destination region '{spec.dst}'")
        self.connections.append(spec)
        return self

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self) -> 'BrainSheet':
        """Instantiate columns and precompute receptive fields."""
        for name in self._order:
            self.regions[name].build()
        self._rfs = []
        for conn in self.connections:
            src, dst = self.regions[conn.src], self.regions[conn.dst]
            self._rfs.append(_project(
                src.spec.rows, src.spec.cols,
                dst.spec.rows, dst.spec.cols,
                conn.projection,
            ))
        self._built = True
        return self

    # ── processing cycle ──────────────────────────────────────────────────────

    def begin(self) -> None:
        """Reset all columns for a new image / timestep."""
        for r in self.regions.values():
            r.begin()

    def observe(self, region_name: str,
                inputs: list[tuple[np.ndarray, int]]) -> None:
        """Inject sensor data into a region.

        inputs: list of (sdr, col_idx) pairs.  col_idx is the flat column
        index (row-major: col_idx = gy * cols + gx).
        """
        region = self.regions[region_name]
        for sdr, col_idx in inputs:
            region.inject(col_idx, sdr)

    def forward(self) -> None:
        """Propagate all feedforward connections.

        Each destination column receives (sdr, location_key) observations
        from its source columns.  The location key is the SOURCE column's
        grid position — object-relative coordinates are preserved.
        """
        for conn, rf in zip(self.connections, self._rfs):
            if conn.kind != 'feedforward':
                continue
            src = self.regions[conn.src]
            dst = self.regions[conn.dst]
            for dst_idx, src_indices in enumerate(rf):
                obs = [
                    (src.output(si), src.pos(si))
                    for si in src_indices
                    if src.output(si) is not None
                ]
                dst.deliver(dst_idx, obs)

    def lateral(self) -> None:
        """Apply lateral connections (winner-index voting from src → dst columns)."""
        for conn, rf in zip(self.connections, self._rfs):
            if conn.kind != 'lateral':
                continue
            src = self.regions[conn.src]
            dst = self.regions[conn.dst]
            for dst_idx, src_indices in enumerate(rf):
                winners = [src.winner(si) for si in src_indices]
                if winners:
                    dst.columns[dst_idx].apply_lateral_input(winners, conn.strength)

    def feedback(self) -> None:
        """Apply feedback connections (winner-index bonus from src to dst columns).

        The source column's current tentative winner index is used as a
        top-down prediction: the corresponding dst minicolumn gets a bonus.
        This mirrors the IT→V1 feedback in cortex.py.
        """
        for conn, rf in zip(self.connections, self._rfs):
            if conn.kind != 'feedback':
                continue
            src = self.regions[conn.src]
            dst = self.regions[conn.dst]
            for dst_idx, src_indices in enumerate(rf):
                for si in src_indices:
                    dst.columns[dst_idx].receive_feedback_by_index(
                        src.winner(si), conn.strength)

    def commit(self, region_name: str,
               labels: dict[int, int] | None = None,
               write: bool = True) -> list[int]:
        """Commit a region; return winner indices.

        labels: {col_idx: class_label} for supervised regions.
        """
        return self.regions[region_name].commit(labels=labels, write=write)

    # ── config I/O ────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str, build: bool = True) -> 'BrainSheet':
        """Load a BrainSheet from a YAML config file.

        build=False returns a sheet with specs populated but no columns
        instantiated — useful for visualization without training.
        """
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)

        brain = cls()
        for name, rc in cfg.get('regions', {}).items():
            brain.add_region(RegionSpec(
                name=name,
                x=float(rc.get('x', 0)),   y=float(rc.get('y', 0)),
                w=float(rc.get('w', 50)),   h=float(rc.get('h', 50)),
                rows=int(rc['rows']),       cols=int(rc['cols']),
                n_mini=int(rc.get('n_mini', 1)),
                meta={k: v for k, v in rc.items()
                      if k not in ('x', 'y', 'w', 'h', 'rows', 'cols', 'n_mini')},
            ))
        for c in cfg.get('connections', []):
            brain.add_connection(ConnectionSpec(
                src=c['src'],   dst=c['dst'],
                kind=c['kind'],
                strength=float(c.get('strength', 1.0)),
                projection=str(c.get('projection', 'topographic')),
            ))
        return brain.build() if build else brain

    def to_yaml(self, path: str) -> None:
        """Save region and connection specs to a YAML file."""
        import yaml
        cfg: dict[str, Any] = {'regions': {}, 'connections': []}
        for name, region in self.regions.items():
            s = region.spec
            cfg['regions'][name] = {
                'x': s.x, 'y': s.y, 'w': s.w, 'h': s.h,
                'rows': s.rows, 'cols': s.cols, 'n_mini': s.n_mini,
                **s.meta,
            }
        for conn in self.connections:
            cfg['connections'].append({
                'src': conn.src, 'dst': conn.dst, 'kind': conn.kind,
                'strength': conn.strength, 'projection': conn.projection,
            })
        with open(path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ── RF projection helper ──────────────────────────────────────────────────────

def _project(src_rows: int, src_cols: int,
             dst_rows: int, dst_cols: int,
             projection: str) -> list[list[int]]:
    """For each dst column, return the list of src column indices in its RF.

    projection options
    ------------------
    topographic  — standard RF centred on the corresponding position in src
    all_to_all   — every dst column receives from every src column
    one_to_one   — dst column i receives from src column i (equal counts required)
    """
    n_src = src_rows * src_cols
    n_dst = dst_rows * dst_cols

    if projection == 'all_to_all':
        return [list(range(n_src)) for _ in range(n_dst)]

    if projection == 'one_to_one':
        if n_src != n_dst:
            raise ValueError(
                f"one_to_one requires equal column counts "
                f"(src={n_src}, dst={n_dst})")
        return [[i] for i in range(n_dst)]

    # topographic: RF half-extents proportional to grid size ratio
    rf_half_w = src_cols / max(dst_cols, 1)
    rf_half_h = src_rows / max(dst_rows, 1)
    result: list[list[int]] = []
    for dy in range(dst_rows):
        for dx in range(dst_cols):
            cx = (dx * (src_cols - 1) / (dst_cols - 1)
                  if dst_cols > 1 else (src_cols - 1) / 2.0)
            cy = (dy * (src_rows - 1) / (dst_rows - 1)
                  if dst_rows > 1 else (src_rows - 1) / 2.0)
            indices = [
                vy * src_cols + vx
                for vy in range(src_rows)
                for vx in range(src_cols)
                if abs(vx - cx) <= rf_half_w and abs(vy - cy) <= rf_half_h
            ]
            result.append(indices)
    return result
