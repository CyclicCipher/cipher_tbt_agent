"""Object affordances from Core-Knowledge perception (GROUNDING_PLAN §3, G-A) — GENERAL, not per-mechanic.

Two estimates from the frame STREAM, used by the GSG to propose the goal-hypothesis "place a MOVABLE object on a
SALIENT MARKER":
  * MOVABLE colours — colours seen to MOVE (a cell's colour changes => that colour is controllable / dynamic). This is
    objectness + agency from motion, the same signal the sensor's path-integration self-gate uses.
  * SALIENT MARKERS — RARE, non-background cells whose colour is NOT movable (static distinct targets: a goal cell, a
    pad). Accumulated and REMEMBERED (a marker covered by a movable object — a pad under a block — stays mapped, the
    same memory the L7-A map gives).

Detected from MOTION + SALIENCE only, NEVER from a colour id — "block=6 / pad=7" in code would be the bitter-lesson bug
([[feedback_bitter_lesson]], [[feedback_subgoal_types_from_dynamics]]). The schema "movable belongs on marker" is a
HYPOTHESIS the GSG tests (G-D); a game where it's wrong simply never confirms it. Pure stdlib.
"""

from __future__ import annotations

from collections import Counter


def _background(frame) -> int:
    """The most common colour = background (the canvas objects sit on)."""
    c = Counter(v for row in frame for v in row)
    return c.most_common(1)[0][0] if c else 0


class Affordances:
    """Track, across the frame stream, which colours MOVE (movable) and which rare static cells are MARKERS. `update`
    each step on the primary grid; `movable_colors()` / `markers()` read the accumulated estimate."""

    def __init__(self, rare_max: int = 6):
        self.rare_max = rare_max                          # a colour with <= this many cells is RARE (salient)
        self._prev = None
        self._moved: set = set()                          # colours seen to MOVE (controllable / dynamic)
        self._markers: dict = {}                          # (x, y) -> colour, cumulative (rare static non-bg; remembered)

    def update(self, frame) -> None:
        bg = _background(frame)
        if self._prev is not None:                        # MOTION: a colour that ARRIVES at a new cell MOVED (translation)
            for row_p, row_f in zip(self._prev, frame):
                for vp, vf in zip(row_p, row_f):
                    if vf != vp and vf != bg:
                        self._moved.add(vf)               # only the ARRIVING colour -- a colour that merely LEAVES may
                        #                                   just be COVERED (a static pad under a block), not moving
        counts = Counter(v for row in frame for v in row)   # SALIENCE: rare non-bg cells -> candidate markers (remembered)
        for y, row in enumerate(frame):
            for x, v in enumerate(row):
                if v != bg and counts[v] <= self.rare_max:
                    self._markers.setdefault((x, y), v)
        self._prev = [row[:] for row in frame]

    def movable_colors(self) -> set:
        """The colours seen to move -- the controllable / pushable objects (objectness + agency from motion)."""
        return set(self._moved)

    def markers(self) -> dict:
        """The remembered SALIENT MARKER cells `{(x, y): colour}` -- rare, non-background, NOT movable (static targets:
        a goal cell, a pad). A marker covered by a movable object stays mapped (the L7-A memory)."""
        return {pos: col for pos, col in self._markers.items() if col not in self._moved}
