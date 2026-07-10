"""The sensor / retina — the bridge from raw ARC frames to the column's input.

The retina does ONLY retina-level work: adapt to the background, deliver the frame's non-background cells as raw
FEATURE-AT-LOCATIONS `[(x, y, colour), ...]`, and compute a bottom-up SALIENCE peak (center-surround pop-out). It does
NOT segment the scene into proto-objects — object membership, permanence, and identity are the COLUMN's doing (common-fate
attention + L2/3 recognition + the hippocampus's path-integrated bump), never a hand-coded tracker (the old proto-object /
config-state harness is gone). `read(frame, action)` passes the efference copy (the last action) through to the column's
one perception step and returns `(state, change)`; `salient_target()` is the cued-discovery peak; `objects()` are the
salience peaks a coordinate (click) action can address. Pure stdlib + the retina primitives."""

from __future__ import annotations

from .retina import background, salient_cells, salient_targets, view_sdr   # retina primitives (adaptation, change, pop-out, content)


class Sensor:
    """Frame -> column input. With `integrate=True` (the live path) the column OWNS perception: the retina hands it the
    frame's raw coloured cells and it returns `(content, here)` — the recognised content id + the path-integrated
    allocentric position. Without a column (standalone) it returns the whole-frame content signature. `salient_target()`
    / `objects()` are the retina's bottom-up salience peaks (the self is excluded via the column's tracked body)."""

    def __init__(self, local: bool = False, window: int = 7, integrate: bool = False, pos_bin: int = 4):
        self._prev = None
        self.local = local
        self.window = window
        self.integrate = integrate
        self.pos_bin = pos_bin
        self.column = None                                   # the column that OWNS perception + path integration (wired by arc_sdk)

    def reset(self):
        self._prev = None
        # The column's L5 efference PERSISTS across levels; its location belief is reset separately (`column.track_reset`).

    def _coloured(self, frame):
        """The frame's raw non-background feature-at-locations `[(x, y, colour), ...]` — the retina's whole output (no
        segmentation, no grouping): the column groups the self by common fate and recognises it."""
        bg = background(frame)
        return [(x, y, v) for y, row in enumerate(frame) for x, v in enumerate(row) if v != bg]

    def read(self, frame, action=None):
        """One perception step. Integrate + column: the column GROUPS the self (common fate) + recognises it + path-
        integrates the pose; returns `((content, here), change)`. Otherwise: the whole-frame content signature."""
        change = salient_cells(self._prev, frame) if self._prev is not None else set()
        coloured = self._coloured(frame)
        if self.column is not None and self.integrate:
            content, _pose = self.column.perceive(action, coloured)   # the ONE perception step (recognise -> bump belief)
            here = self.column.here_position()
            state = (content, here)
        else:
            state = view_sdr(coloured)                       # standalone / no column: the whole-frame content SDR
        self._prev = frame
        return state, change

    def _self_cells(self):
        """The column's currently-tracked self body (to exclude from salience), or ()."""
        return getattr(self.column, "_self_cells", None) or () if self.column is not None else ()

    def salient_target(self):
        """Feature-contrast salience (DISCOVERY.md D1): the most distinct location that is NOT the tracked self — the
        bottom-up 'distinct from surround' cue that lets a static marker become a candidate goal. `(x, y)` or None."""
        if self._prev is None:
            return None
        peaks = salient_targets(self._prev, exclude=self._self_cells(), k=1)
        return peaks[0] if peaks else None

    def objects(self):
        """The retina's salience PEAKS as click-addressable targets `{index: ((x, y), 1)}` (top-K, non-max suppressed) —
        the coordinate-action slots. Not a proto-object tracker: just the winner-take-all peaks of the pop-out map."""
        if self._prev is None:
            return {}
        peaks = salient_targets(self._prev, exclude=self._self_cells(), k=12)
        return {i: ((float(x), float(y)), 1) for i, (x, y) in enumerate(peaks)}
