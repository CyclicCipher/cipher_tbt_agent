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
  * PATH-INTEGRATED (`integrate=True`, on top of local) -- the controllable object's ALLOCENTRIC position, but the
    sensor no longer computes it: it DETECTS the residual candidates and hands them to the COLUMN (`column.track`),
    which owns the efference (L5's per-action displacement), the correction (snap to the sighting nearest the
    prediction -> action-consistent tracking, animation ignored), the controllability GATE, and the coarse state node
    (`column.track_state`). The P1 unification (GROUNDING_PLAN): ONE path integrator, in the column where TBT seats it.

The stateless proto-object proposer runs in all modes (so `objects()` still feeds the click-slots, per frame).
`read(frame, action)` passes the efference copy (the last action) THROUGH to the column's path integration. Pure stdlib.
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
        self._fovea = None                                   # the S-frame attention locus (where the patch is extracted)
        self.column = None                                   # P1: the column that OWNS path integration (its `track_*`); wired by arc_sdk, like `encode`
        # FEATURE-AT-LOCATION (the L4 seam): the egocentric patch is encoded to an L4 FEATURE id (the column's content
        # vocabulary, grown online), so the local state is (feature, location) -- L4's job, not a raw pixel tuple.
        # Defaults to identity (the raw patch) until the column wires its `L4.encode` in (so tests run column-free).
        self.encode = encode if encode is not None else (lambda patch: patch)

    def reset(self):
        self.field.reset()
        self._prev = None
        self._fovea = None
        # The column's L5 displacement (the efference) PERSISTS across levels (the same mechanic everywhere); its
        # location belief is reset separately by the loop (`column.track_reset`). The sensor keeps no position state.

    def read(self, frame, action=None):
        objects = self.field.perceive(frame)                               # stateless proto-object proposal this frame
        change = salient_cells(self._prev, frame) if self._prev is not None else set()
        if self.local:
            appeared, dominant, cold = self._residual_candidates(frame, change, objects)
            if self.column is not None:                                    # the COLUMN owns the fovea (efference disambiguation) -- both modes
                self._fovea = self.column.track(action, appeared, dominant, cold)
                feat = self.encode(self._patch(frame, self._fovea))
                if self.integrate:                                          # integrate ADDS the location node: the POSE (x,y,heading)
                    node = (self.column.pose_state(self.pos_bin)             # when the dynamics are heading-dependent (non-abelian),
                            if self.column.L5.heading_dependent()           # else the abelian POSITION (== track_state, no regression).
                            else self.column.track_state(self.pos_bin))     # L6_NONABELIAN S1e gate (scaffolding -> dissolve at step 5)
                    state = (feat, node)
                else:
                    state = feat
            else:                                                          # standalone sensor (no column): dominant-residual fovea, no path integration
                self._fovea = dominant if dominant is not None else (self._fovea or cold)
                feat = self.encode(self._patch(frame, self._fovea))
                state = (feat, (0, 0)) if self.integrate else feat         # integrate w/o a column -> the gate-off constant position
        else:
            state = config_state(objects, self.field.contents)
        self._prev = frame
        return state, change

    def objects(self):
        """The current tracked objects `{id: (pose, size)}` -- poses feed the click-slots and the L5 reseat."""
        return dict(self.field._last)

    # ----- egocentric S-frame perception: detect the residual candidates the COLUMN path-integrates ------------
    def _residual_candidates(self, frame, change, objects):
        """The S-frame perception the column's tracker consumes -- (appeared, dominant, cold):
          appeared = centroids of the ARRIVED (non-background) change components -- the candidate controllable sightings
                     the column DISAMBIGUATES by the efference (rejecting autonomous animation);
          dominant = the largest connected change centroid (the cold-motion / fallback sighting), or None;
          cold     = the largest object's centroid / the frame centre -- the cold-start locus before anything moves.
        The sensor only DETECTS; the COLUMN owns the displacement + position (no `_delta`/position here -- the P1 unification)."""
        bg = background(frame)
        appeared = [c for _comp, c in components({(x, y) for (x, y) in change if frame[y][x] != bg})]
        _comp, dominant = dominant_region(change)
        cold = self._largest_centroid(objects) or (len(frame[0]) / 2.0, len(frame) / 2.0)
        return appeared, dominant, cold

    def _largest_centroid(self, objects):
        if not objects:
            return None
        (px, py), _size = max(objects.values(), key=lambda v: (v[1], v[0]))
        return (px, py)

    def _patch(self, frame, fovea):
        """The raw `window x window` patch of `frame` centred on `fovea` (out-of-bounds = -1), as a hashable tuple."""
        h = self.window // 2
        cx, cy = int(round(fovea[0])), int(round(fovea[1]))
        H, W = len(frame), len(frame[0])
        return tuple(tuple(frame[y][x] if 0 <= x < W and 0 <= y < H else -1
                           for x in range(cx - h, cx + h + 1)) for y in range(cy - h, cy + h + 1))
