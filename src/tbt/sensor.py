"""The sensor / retina -- the bridge from raw ARC frames to the column's input.

Perception modes, ONE tracker underneath:
  * GLOBAL (`config_state`) -- segment + track objects and encode a TRANSLATION-INVARIANT (size, rel-pose, content)
    state. Faithful for STRUCTURED / small scenes (and the home of the L5 operator's generalization), but on real 64x64
    ARC frames it NEVER recurs (the full per-pixel content churns with any animation -> the loop is starved).
  * LOCAL / EGOCENTRIC (`local=True`) -- the thousand-brains way: sense a WINDOW around a FOVEA placed on the dynamic
    residual (the controllable change), and ENCODE it to an L4 FEATURE id (`self.encode` = the column's `L4.encode`,
    the content vocabulary grown online) -- so the local state is a FEATURE-at-location, not a raw pixel tuple. Local
    views RECUR (measured on real cn04/ls20: ~0 -> ~0.6-0.9), so the column can learn. But a local view has NO
    allocentric position, so distinct board locations ALIAS and the agent cannot tell explored from unexplored.
  * PATH-INTEGRATED (`integrate=True`, on top of local) -- track the controllable object's ALLOCENTRIC position by
    EFFERENCE (the learned per-action displacement predicts where it goes) + CORRECTION (snap to the residual sighting
    NEAREST that prediction, so the ACTION-CONSISTENT object is followed and autonomous animation ignored). The coarse
    position augments the state -> aliased views separate, novelty drives systematic COVERAGE, navigation works. A
    SELF-GATE keeps the position out of the state until the object proves controllable (a consistent, non-trivial
    learned displacement), so a STATE-CHANGE game (no controllable position) keeps its recurring local view.

The object tracker runs in all modes (so `objects()` still feeds the click-slots). `read(frame, action)` takes the
efference copy (the last action) for path integration. Pure stdlib.
"""

from __future__ import annotations

from .perceive import ObjectField, background, canonicalize, components
from .retina import dominant_region, salient_cells          # the canonical dynamic residual + foveation (reused)


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
    """Frame -> column input. `read(frame, action)` tracks objects and returns `(state, change)`. With `local=True` the
    state is an EGOCENTRIC window (size `window`) around the dynamic-residual fovea; with `integrate=True` it is
    `(window, coarse path-integrated position)` once the object proves controllable. `local=False` (default) is the
    global translation-invariant `config_state`. `objects()` works in all modes."""

    def __init__(self, local: bool = False, window: int = 7, integrate: bool = False, pos_bin: int = 4, encode=None):
        self.field = ObjectField()
        self._prev = None
        self.local = local
        self.window = window
        self.integrate = integrate
        self.pos_bin = pos_bin
        self._fovea = None                                   # the attention locus / path-integrated position
        self._delta: dict = {}                               # action -> learned position displacement (the efference)
        # FEATURE-AT-LOCATION (the L4 seam): the egocentric patch is encoded to an L4 FEATURE id (the column's content
        # vocabulary, grown online), so the local state is (feature, location) -- L4's job, not a raw pixel tuple.
        # Defaults to identity (the raw patch) until the column wires its `L4.encode` in (so tests run column-free).
        self.encode = encode if encode is not None else (lambda patch: patch)

    def reset(self):
        self.field.reset()
        self._prev = None
        self._fovea = None
        # self._delta PERSISTS across levels -- the per-action displacement is the same game mechanic everywhere.

    def read(self, frame, action=None, predict=None):
        objects = self.field.perceive(frame, predict)
        change = salient_cells(self._prev, frame) if self._prev is not None else set()
        if self.local:
            self._update_fovea(frame, change, objects, action)
            feat = self.encode(self._patch(frame, self._fovea))             # L4 feature-at-location (identity until wired)
            state = (feat, self._coarse_pos()) if self.integrate else feat
        else:
            state = config_state(objects, self.field.contents)
        self._prev = frame
        return state, change

    def objects(self):
        """The current tracked objects `{id: (pose, size)}` -- poses feed the click-slots and the L5 reseat."""
        return dict(self.field._last)

    # ----- egocentric local sensing + path integration (the recurrence + navigation fix) ----------------
    def _update_fovea(self, frame, change, objects, action):
        """Place the FOVEA on the controllable object, LEARN the per-action displacement from its move (the efference),
        and snap (CORRECT). Cold start (no motion yet) foveates the largest object / the frame centre."""
        if change:
            c = self._locate(frame, change, action)
            if c is not None:
                if action is not None and self._fovea is not None:   # learn the per-action displacement (EWMA -> the efference)
                    actual = (c[0] - self._fovea[0], c[1] - self._fovea[1])
                    old = self._delta.get(action)
                    self._delta[action] = actual if old is None else (0.6 * old[0] + 0.4 * actual[0], 0.6 * old[1] + 0.4 * actual[1])
                self._fovea = c                                      # correct (snap to the sighting)
        if self._fovea is None:
            self._fovea = self._largest_centroid(objects) or (len(frame[0]) / 2.0, len(frame) / 2.0)

    def _locate(self, frame, change, action):
        """Where the controllable object went. Once its per-action displacement is known, the ARRIVED (non-background)
        residual region nearest the EFFERENCE prediction (fovea + displacement) -- clean tracking that ignores
        autonomous animation. Until then (cold start / no efference), the largest connected residual (robust, the
        7b behaviour validated live)."""
        d = self._delta.get(action)
        if d is not None and self._fovea is not None and (d[0] * d[0] + d[1] * d[1]) > 0.25:
            bg = background(frame)
            appeared = {(x, y) for (x, y) in change if frame[y][x] != bg}     # where the object IS now, not vacated
            comps = components(appeared)
            if comps:
                pred = (self._fovea[0] + d[0], self._fovea[1] + d[1])
                return min(comps, key=lambda cc: (cc[1][0] - pred[0]) ** 2 + (cc[1][1] - pred[1]) ** 2)[1]
        _comp, c = dominant_region(change)
        return c

    def _largest_centroid(self, objects):
        if not objects:
            return None
        (px, py), _size = max(objects.values(), key=lambda v: (v[1], v[0]))
        return (px, py)

    def _controllable(self) -> bool:
        """The fovea's object responds to actions (a consistent, non-trivial learned displacement) -> its allocentric
        POSITION is informative. Threshold 0.5 catches a 1-cell-per-action mover (|delta|~1) while still rejecting
        noise (random animation averages to ~0), so a state-change game keeps its recurring local view."""
        return any(d[0] * d[0] + d[1] * d[1] > 0.5 for d in self._delta.values())

    def _coarse_pos(self):
        """The coarse allocentric position, gated by controllability (else a constant -- keeps the state type stable)."""
        if self._fovea is None or not self._controllable():
            return (0, 0)
        return (int(round(self._fovea[0])) // self.pos_bin, int(round(self._fovea[1])) // self.pos_bin)

    def _patch(self, frame, fovea):
        """The raw `window x window` patch of `frame` centred on `fovea` (out-of-bounds = -1), as a hashable tuple."""
        h = self.window // 2
        cx, cy = int(round(fovea[0])), int(round(fovea[1]))
        H, W = len(frame), len(frame[0])
        return tuple(tuple(frame[y][x] if 0 <= x < W and 0 <= y < H else -1
                           for x in range(cx - h, cx + h + 1)) for y in range(cy - h, cy + h + 1))
