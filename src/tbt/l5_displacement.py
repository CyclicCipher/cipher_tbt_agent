"""Layer 5 — the per-action OPERATOR layer: the column's dynamics + motor-output seat.

L5 is the cortex's main OUTPUT layer and its displacement-cell layer (see reference_layer5_role). The chosen change is
ONE object with four uses: the position-invariant GENERALIZING base operator (predicts an action's effect at an
UNVISITED state), the MOTOR command (the enacted action), the EFFERENCE COPY (the predicted effect -> the predictive
state), and the feed-forward DRIVER of the higher-order thalamus (the inter-column message).

The operator is KIND-GENERAL: an action does not just MOVE things, so L5 learns a position-invariant DELTA in whatever
feature dimension the action changes (see reference_l5_operator_kinds), keyed on the stable SHAPE identity (`size`):
  * `disp[(shape, action)]`  — the modal POSE delta (translation / movement); the ventral "where".
  * `recolor[(shape, action)]` — the CONTENT transition map {old_content -> new_content} (in-place change / a colour
    flip); the dorsal "what changed". It generalizes over POSITION (the same change wherever the shape is).
`predict` applies BOTH then re-encodes -- so it models movement, recolouring, or both, at an unvisited state. The
discrete EDGES are the per-(state, action) EXCEPTIONS (a wall/door; a CONTEXT-DEPENDENT change) that OVERRIDE the base
operator; the column's conditional-dynamics faculty generalizes a PRECONDITION (the rest of "conditioned on context").
This unifies the dorsal/ventral specialisations into one operator (the dimension that changes is emergent); genuine
separate reference frames + cross-frame voting are a later step. Rotation (theta) is a deferred extension (one more
delta dimension, fed by the recogniser's inferred angle).

The MATRIX associative memory (`learn`/`apply`) is the offline / archived form (it crosstalks over correlated SR
codes; reserved for orthonormal codes).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np                                              # the pose operators act on small point clouds
import torch
import torch.nn as nn

from .perceive import canonicalize


# ---- pose = a GROUP ELEMENT acting on displacements (the displacement-cell geometry) -------------------------
# A "pose" re-expresses one set of displacement vectors as another under a group element. The SPATIAL instance
# (this visual column) is SO(2) + a translation, so pose = (theta, t). For an ABSTRACT column the group is
# LEARNED from the action-orbit structure (see the memory reference_tbt_frames_and_hippocampus): these functions
# are the SO(2) plug-in, deliberately shaped as "apply a group element to displacements" so the abstract case
# slots in without a rewrite. They are the ONE home of this geometry -- recognition (L2/3) imports them here, it
# does not keep its own copy.

def rot(theta: float) -> np.ndarray:
    """A group element in matrix form. Spatial instance: the 2-D rotation R(theta) in SO(2)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def local_disps(locs, i, radius):
    """Node `i`'s local patch = the displacement vectors to cells within `radius` (group-EQUIVARIANT -- they
    transform with the object). This is the sensed 'feature pose': the local neighbourhood geometry a pose acts on."""
    p = locs[i]
    return np.array([locs[j] - p for j in range(len(locs)) if j != i and np.linalg.norm(locs[j] - p) <= radius])


def _sets_match(A, B, tol):
    """Do point sets A, B coincide (each a in A matched to a distinct b in B within tol)?"""
    B = list(B)
    for a in A:
        for k, b in enumerate(B):
            if np.linalg.norm(a - b) <= tol:
                B.pop(k)
                break
        else:
            return False
    return True


def pose_between(model_disps, sensed_disps, tol=0.05):
    """SOLVE the group element(s) g with g.model_disps ~= sensed_disps (as sets) -- 'which transform re-expresses
    these displacements as those'. Spatial instance: g = a rotation, returned as theta in [0, 2pi); continuous, and
    a symmetric patch yields several (Monty's multiple pose hypotheses). Reads the pose off the local geometry
    rather than searching angles. (For an abstract column this becomes a solve over the learned group.)"""
    if len(model_disps) != len(sensed_disps) or len(model_disps) == 0:
        return []
    out: list[float] = []
    v0 = model_disps[0]
    for w in sensed_disps:                                   # pair v0 with each equal-length sensed vector -> a candidate
        if abs(np.linalg.norm(w) - np.linalg.norm(v0)) > tol:
            continue
        theta = float(np.arctan2(w[1], w[0]) - np.arctan2(v0[1], v0[0]))
        if _sets_match([rot(theta) @ v for v in model_disps], sensed_disps, tol) and \
                all(abs((theta - o + np.pi) % (2 * np.pi) - np.pi) > 1e-2 for o in out):
            out.append(theta % (2 * np.pi))
    return out


def apply_pose(cloud, theta, t):
    """Apply a pose (group element, translation) to a point cloud: R(theta).cloud + t. The universal, continuous
    operator -- correct by construction (there is no per-orientation entry to learn wrong)."""
    R, t = rot(theta), np.asarray(t, float)
    return [R @ np.asarray(loc, float) + t for loc in cloud]


align_rotations = pose_between                                  # the spatial-instance name (kept for callers)


class L5_Displacement(nn.Module):
    FIELD_OUT = -1                                                   # out-of-frame token for the per-location field operator

    def __init__(self, field_radius: int = 2):
        super().__init__()
        self.ops: dict = {}                                          # (domain, relation) -> matrix operator (offline/archived)
        self.edges: dict = {}                                        # state -> {action -> next state}: observed transitions / exceptions
        self.disp: dict = {}                                         # (shape, action) -> modal pose delta (dx, dy): the movement operator
        self.move_delta: dict = {}                                   # action -> continuous pixel translation (dx, dy): the metric efference (the tracker path-integrates by it)
        self.recolor: dict = {}                                      # (shape, action) -> {old_content -> new_content}: the in-place-change operator
        self._votes: dict = {}                                       # (shape, action) -> Counter of pose deltas (-> disp = mode)
        # The per-LOCATION operator (FM1) -- the SAME position-invariant, context-keyed principle as disp/recolor, at
        # the FINEST grain: a location's next FEATURE from its local feature-NEIGHBOURHOOD + action. The generalizing
        # base for STRUCTURED DYNAMICS (a propagating frontier) that the whole-object operator cannot see; it reads
        # L4's feature-at-location field (NOT raw pixels) and is the TEM objective ("predict the next observation")
        # at cell grain. disp/recolor stay the coarser whole-object form; the discrete edges stay the exceptions.
        self.field_radius = field_radius
        self.field_rule: dict = defaultdict(Counter)                 # (local feature-neighbourhood, action) -> Counter(center next feature)

    # ---- the online operator: edges (exceptions) + the position-invariant delta (generalization) -------------
    def observe(self, s, a, s2) -> None:
        """Learn one per-action transition. A real change (s2 != s) records its edge AND votes the operator (pose delta
        + content transition); a blocked move (s2 == s) over a config-state records a self-edge -- the EXCEPTION that
        overrides the base operator -- but does NOT vote it down (the operator is the rule; the block is the exception)."""
        if s2 != s or self._is_config(s):                           # config: record blocked self-edges (exceptions); opaque: keep no-self-edge
            self.edges.setdefault(s, {})[a] = s2
        if s2 != s:
            self._learn_op(s, a, s2)

    def predict(self, s, a):
        """The operator / efference copy: where action `a` takes state `s`. Observed edge (incl. a blocked self-edge)
        first -- the state-dependent exception; else the position-invariant operator GENERALIZES (move + recolour) to
        this unvisited (s, a); else stay (no model yet)."""
        edge = self.edges.get(s, {}).get(a)
        if edge is not None:
            return edge
        gen = self._generalize(s, a)
        return gen if gen is not None else s

    def successors(self, s):
        """{action -> next state} learned from `s` — the operator's outgoing edges."""
        return self.edges.get(s, {})

    # ---- the CONTINUOUS per-action displacement (the SPATIAL column's group action) --------------------------
    # L5's operator "in whatever dimension the action changes" at the METRIC grain: the pixel translation an action
    # produces. The spatial column's group is SO(2)+translation; with rotation deferred this is just the translation
    # `move_delta[action]`, learned online by EWMA from the sensed displacement (the efference copy). This is the ONE
    # home of the per-action displacement -- the sensor no longer keeps its own copy (the P1 unification): L6 (the
    # column tracker) path-integrates the location belief by this delta, and reads `controllable` to gate the position.
    def observe_move(self, action, delta, rate: float = 0.4) -> None:
        """Learn the per-action pixel translation from one sensed move `delta=(dx,dy)` -- an EWMA (the efference the
        tracker path-integrates by). First sighting sets it; later ones average (robust to a noisy sighting)."""
        old = self.move_delta.get(action)
        self.move_delta[action] = tuple(delta) if old is None else (
            (1.0 - rate) * old[0] + rate * delta[0], (1.0 - rate) * old[1] + rate * delta[1])

    def move(self, action):
        """The learned per-action translation (dx, dy) -- the efference the location belief dead-reckons by; (0, 0)
        for an action whose effect is not yet learned (dead-reckon says 'stay', the sighting then corrects)."""
        return self.move_delta.get(action, (0.0, 0.0))

    def controllable(self, thresh: float = 0.5) -> bool:
        """Does SOME action produce a non-trivial translation? -> the tracked object responds to actions, so its
        allocentric POSITION is informative (gate ON). A state-change scene (in-place animation) averages to ~0 ->
        gate OFF, keeping the recurring local view. The `_controllable` gate, now sourced from L5's displacement."""
        return any(d[0] * d[0] + d[1] * d[1] > thresh for d in self.move_delta.values())

    # ---- motor output + thalamus driver (the other two uses of the one operator) -------------------------
    def motor(self, a):
        """The MOTOR command: the enacted action. L5 is the cortex's output layer -- the chosen action is its output
        (the name->GameAction mapping is the motor ORGAN, in arc_sdk). Identity over discrete actions, by design."""
        return a

    def driver(self, s, a):
        """The feed-forward DRIVER message (what a higher-order thalamus would relay to another column): for each shape
        in `s`, the effect action `a` has -- a ('move', delta) and/or a ('recolor', new_content) -- the inter-column
        'this changed by that'."""
        if not self._is_config(s):
            return ()
        msg = {}
        for elem in s:
            shape, content = self._key(elem), tuple(elem[2:])
            effects = []
            d = self.disp.get((shape, a))
            if d and d != (0, 0):
                effects.append(("move", d))
            nc = self.recolor.get((shape, a), {}).get(content)
            if nc is not None and nc != content:
                effects.append(("recolor", nc))
            if effects:
                msg[shape] = tuple(effects)
        return tuple(sorted(msg.items()))

    # ---- the config-state structure the operator reads (CMP: features at poses) ---------------------
    @staticmethod
    def _is_config(s) -> bool:
        """True if `s` is a config-state (a tuple of `(size, pose, *content)` elements) rather than an opaque symbol."""
        if not (isinstance(s, tuple) and s):
            return False
        e = s[0]
        return isinstance(e, tuple) and len(e) >= 2 and isinstance(e[1], tuple) and len(e[1]) == 2

    @staticmethod
    def _key(elem):
        """A config element's SHAPE identity (`size` only) -- the operator is keyed on it, so the same shape shares one
        position-invariant operator and BOTH its pose (movement) and its content (recolouring) can be factored out."""
        return (elem[0],)

    def _learn_op(self, s, a, s2) -> None:
        """Vote the per-(shape, action) operator from a real transition: align each element of `s` to the same-shape
        element of `s2` nearest in pose, then learn its POSE delta (mode -> disp) and its CONTENT transition (-> recolor)."""
        if not (self._is_config(s) and self._is_config(s2)):
            return
        by_shape = defaultdict(list)
        for e in s2:
            by_shape[self._key(e)].append(e)
        for e in s:
            shape, pose, content = self._key(e), e[1], tuple(e[2:])
            cands = by_shape.get(shape)
            if not cands:
                continue
            tgt = min(cands, key=lambda c: abs(c[1][0] - pose[0]) + abs(c[1][1] - pose[1]))
            delta = (tgt[1][0] - pose[0], tgt[1][1] - pose[1])
            self._votes.setdefault((shape, a), Counter())[delta] += 1
            self.disp[(shape, a)] = self._votes[(shape, a)].most_common(1)[0][0]
            tgt_content = tuple(tgt[2:])
            if tgt_content != content:                              # an in-place (or accompanying) content change
                self.recolor.setdefault((shape, a), {})[content] = tgt_content

    def _generalize(self, s, a):
        """Predict an UNVISITED (s, a) by applying the position-invariant operator -- the pose delta AND the content
        transition -- to each element, then re-encoding to the SAME translation-invariant form. None if `s` is opaque
        or nothing changes (no generalization)."""
        if not self._is_config(s):
            return None
        elements, changed = [], False
        for elem in s:
            shape, pose, content = self._key(elem), elem[1], tuple(elem[2:])
            d = self.disp.get((shape, a))
            if d and d != (0, 0):
                pose = (pose[0] + d[0], pose[1] + d[1])
                changed = True
            nc = self.recolor.get((shape, a), {}).get(content)
            if nc is not None and nc != content:
                content = nc
                changed = True
            elements.append((pose, (elem[0],) + content))
        return canonicalize(elements) if changed else None

    # ---- the per-LOCATION operator: the generative forward model (FM1) ------------------------------------
    # L5's operator at the FINEST grain. `field` is L4's feature-at-location map (feature ids over the L6 grid); the
    # rule `(local feature-neighbourhood, action) -> next centre feature` is learned online and applied per location.
    # Position-invariant (the rule is keyed on the LOCAL pattern, not the absolute place), so it GENERALISES to
    # board positions never visited whose local pattern was seen -- which is why it captures a propagating frontier
    # that no tabular state recurs on. The column feeds the field (it reads L4); L5 owns the operator.
    def _field_patch(self, field, x, y, H, W):
        r = self.field_radius
        return tuple(field[y + dy][x + dx] if 0 <= x + dx < W and 0 <= y + dy < H else self.FIELD_OUT
                     for dy in range(-r, r + 1) for dx in range(-r, r + 1))

    @staticmethod
    def _field_bg(field):
        return Counter(v for row in field for v in row).most_common(1)[0][0]

    def _field_all_bg(self, field, x, y, H, W, bg) -> bool:
        r = self.field_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and field[ny][nx] != bg:
                    return False
        return True

    def observe_field(self, field, action, next_field) -> None:
        """Learn one feature-field transition: for every location vote `(its neighbourhood, action) -> its NEXT
        feature`. All locations are learned (a no-change location teaches the identity branch), so prediction defaults
        correctly. Deterministic dynamics -> a single-entry Counter (exact); the Counter holds not-yet-resolved context."""
        H, W = len(field), len(field[0])
        for y in range(H):
            for x in range(W):
                self.field_rule[(self._field_patch(field, x, y, H, W), action)][next_field[y][x]] += 1

    def field_step(self, field, action):
        """ONE pass over the active region -> (predicted next field, confidence). The planner's hot path needs BOTH
        the next field (pragmatic value) and the trust (epistemic value) per action, so they share a single scan
        (avoids double-iterating the active region -- the FM4 performance fix). `confidence` = fraction of active
        locations whose rule is UNAMBIGUOUS; an unseen context keeps its current feature (the no-change default).
        Skips ALL-background neighbourhoods (they cannot change) -> the scan is bounded to the active region."""
        H, W = len(field), len(field[0])
        bg = self._field_bg(field)
        out = [row[:] for row in field]
        seen = sure = 0
        for y in range(H):
            for x in range(W):
                if field[y][x] == bg and self._field_all_bg(field, x, y, H, W, bg):
                    continue
                c = self.field_rule.get((self._field_patch(field, x, y, H, W), action))
                if c:
                    seen += 1
                    if len(c) == 1:
                        sure += 1
                        out[y][x] = next(iter(c))                # the single outcome (skip most_common)
                    else:
                        out[y][x] = c.most_common(1)[0][0]
        return out, (sure / seen if seen else 0.0)

    def predict_field(self, field, action):
        """The predicted next feature-field -- the efference copy at field grain (thin wrapper over `field_step`)."""
        return self.field_step(field, action)[0]

    def field_confidence(self, field, action) -> float:
        """Fraction of active locations whose rule is UNAMBIGUOUS -- a trust / learning-progress signal (over `field_step`)."""
        return self.field_step(field, action)[1]

    # ---- pose operators (the displacement-cell geometry, the layer's API) --------------------------------
    # The continuous-pose half of L5: a pose is a GROUP ELEMENT acting on displacements (spatial instance: SO(2)
    # + translation; abstract columns: a learned group -- reference_tbt_frames_and_hippocampus). Recognition
    # (L2/3) reads its (object, pose) hypotheses through these; the module-level functions are the shared home.
    local_disps = staticmethod(local_disps)                  # a patch's neighbour-displacement vectors (the feature pose)
    pose_between = staticmethod(pose_between)                # SOLVE the group element(s) aligning model -> sensed
    apply_pose = staticmethod(apply_pose)                   # APPLY a (group element, translation) to a point cloud

    # ---- the matrix associative-memory operator (offline / archived) -------------------------------------
    def learn(self, key, place: torch.Tensor, edges) -> None:
        M = torch.zeros(place.shape[1], place.shape[1], device=place.device)
        for s, t in edges:
            M = M + torch.outer(place[t], place[s])
        self.ops[key] = M

    def apply(self, key, v: torch.Tensor) -> torch.Tensor:
        return self.ops[key] @ v
