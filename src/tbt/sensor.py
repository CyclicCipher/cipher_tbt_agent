"""The sensor / retina -- the bridge from raw ARC frames to the column's input.

Two perception modes, ONE tracker underneath:
  * GLOBAL (`config_state`) -- segment + track objects (`perceive.ObjectField`: permanence, contact) and encode the
    scene as a TRANSLATION-INVARIANT (size, rel-pose, content) state. Faithful for STRUCTURED / small scenes, and the
    home of the L5 operator's generalization. But on real 64x64 ARC frames it NEVER recurs (measured: global states
    are unique every frame -> the loop is starved), because the full per-pixel content churns with any animation.
  * LOCAL / EGOCENTRIC (`local=True`) -- the thousand-brains way: sense a small WINDOW around a FOVEA placed on the
    dynamic residual (the controllable change; `retina.salient_cells`/`dominant_region`). Local views RECUR (measured
    on real cn04/ls20: recurrence 0.00->0.6 vs the global encoding), so the column can finally learn. The fovea
    persists (a no-change step keeps it; cold start = the largest object), the state is the raw WxW egocentric patch.

The object tracker runs in BOTH modes (so `objects()` still feeds the click-slots and the L5 poses); only the STATE
the loop runs on differs. The change stream is returned in both. Pure stdlib.
"""

from __future__ import annotations

from .perceive import ObjectField, canonicalize
from .retina import dominant_region, salient_cells          # the canonical residual + foveation (reused, not redefined)


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


class Sensor:
    """Frame -> column input. `read(frame)` tracks objects and returns `(state, change)`. `local=True` makes the state
    an EGOCENTRIC window (size `window`) around the dynamic-residual fovea -- the representation that RECURS on real
    64x64 frames; `local=False` (default) is the global translation-invariant `config_state`. `objects()` (the tracker)
    works in both modes. `predict(oid, pose)->pose` (the column's L5) may be passed to disambiguate objects in contact."""

    def __init__(self, local: bool = False, window: int = 7):
        self.field = ObjectField()
        self._prev = None
        self.local = local
        self.window = window
        self._fovea = None                                   # the persistent attention locus (egocentric mode)

    def reset(self):
        self.field.reset()
        self._prev = None
        self._fovea = None

    def read(self, frame, predict=None):
        objects = self.field.perceive(frame, predict)
        change = salient_cells(self._prev, frame) if self._prev is not None else set()
        if self.local:
            state = self._local_state(frame, change, objects)
        else:
            state = config_state(objects, self.field.contents)
        self._prev = frame
        return state, change

    def objects(self):
        """The current tracked objects `{id: (pose, size)}` -- poses feed the click-slots and the L5 reseat."""
        return dict(self.field._last)

    # ----- egocentric local sensing (the recurrence fix) ------------------------------------------------
    def _local_state(self, frame, change, objects):
        """Place the FOVEA on the dynamic residual (the controllable change), persisting it across a no-change step,
        and return the raw WxW egocentric patch -- a LOCAL view that recurs as the agent acts. Cold start (no change
        yet) foveates the largest object, else the frame centre."""
        if change:
            _comp, c = dominant_region(change)
            if c is not None:
                self._fovea = c
        if self._fovea is None:
            if objects:
                (px, py), _size = max(objects.values(), key=lambda v: (v[1], v[0]))
                self._fovea = (px, py)
            else:
                self._fovea = (len(frame[0]) / 2.0, len(frame) / 2.0)
        return self._patch(frame, self._fovea)

    def _patch(self, frame, fovea):
        """The raw `window x window` patch of `frame` centred on `fovea` (out-of-bounds = -1), as a hashable tuple."""
        h = self.window // 2
        cx, cy = int(round(fovea[0])), int(round(fovea[1]))
        H, W = len(frame), len(frame[0])
        return tuple(tuple(frame[y][x] if 0 <= x < W and 0 <= y < H else -1
                           for x in range(cx - h, cx + h + 1)) for y in range(cy - h, cy + h + 1))
