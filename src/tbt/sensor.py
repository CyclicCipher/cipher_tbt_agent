"""The sensor / retina -- the bridge from raw ARC frames to the column's input.

Per frame it: (a) segments + tracks objects (`perceive.ObjectField` -- permanence, contact); (b) encodes the scene as
a TRANSLATION-INVARIANT state so the loop's states RECUR (`config_state` -- each object's size + pose RELATIVE to the
largest object, rounded; the same arrangement anywhere on the board is one state); and (c) exposes the CHANGE stream
(`salient_cells` -- the cells that differ between frames, the magno / dorsal input the dynamics column will consume).
Object poses feed the L5 reseat (finish); the change stream feeds the dynamics column; the loop runs on the state.

This is the Monty sensor module: raw input -> features at poses. RF-patch local sensing (the archived `Retina`) is the
future refinement for the STATIC layout; object-level segmentation is the perception the loop needs now. Pure stdlib.
"""

from __future__ import annotations

from .perceive import ObjectField, canonicalize


def config_state(objects, contents=None):
    """The scene as a hashable, TRANSLATION-INVARIANT state: each object's `(size, integer pose relative to the LARGEST
    object[, content])`, sorted. A thin wrapper over `perceive.canonicalize` (the ONE encoding the operator/L5 reuses):
    each object becomes `(pose, feature)` with `feature = (size,)` or `(size, content)`, so the operator can apply a
    position-invariant displacement and re-encode to the SAME form. The same RELATIVE arrangement anywhere on the board
    yields the SAME state; a removed object is simply absent (so a 'required-absent' goal needs no special case)."""
    elements = []
    for oid, (pose, size) in objects.items():
        feat = (size,) if contents is None else (size, contents.get(oid))
        elements.append((pose, feat))
    return canonicalize(elements)


def salient_cells(prev, cur):
    """The cells that CHANGED between two frames -- the dynamic residual (magno / dorsal input) the dynamics column
    consumes, and the bottom-up salience that bootstraps the saccade policy."""
    return {(x, y) for y in range(len(cur)) for x in range(len(cur[0])) if prev[y][x] != cur[y][x]}


class Sensor:
    """Frame -> column input. `read(frame)` segments + tracks objects and returns `(state, change)`: the
    translation-invariant scene state for the loop, and the change stream for the dynamics column. Holds the tracker
    (ids stable across frames) and the previous frame (for the change). `predict(oid, pose)->pose` (the column's L5)
    may be passed to disambiguate objects in contact; omitted, the tracker assumes no contact."""

    def __init__(self):
        self.field = ObjectField()
        self._prev = None

    def reset(self):
        self.field.reset()
        self._prev = None

    def read(self, frame, predict=None):
        objects = self.field.perceive(frame, predict)
        change = salient_cells(self._prev, frame) if self._prev is not None else set()
        self._prev = frame
        return config_state(objects, self.field.contents), change

    def objects(self):
        """The current tracked objects `{id: (pose, size)}` -- poses feed the L5 reseat finish."""
        return dict(self.field._last)
