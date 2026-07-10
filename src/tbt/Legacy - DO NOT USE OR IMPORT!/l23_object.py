"""Layer 2/3 — the object / identity layer: graph-memory + evidence-based recognition + lateral voting.

L2/3 forms the STABLE object IDENTITY — the code that stays fixed as the sensor moves, POOLING over the L4
feature-at-location sequence, settled by recurrent self-bias + LATERAL (inter-column) voting (Hawkins 2019;
reference_tbt_layers_4_23). The object MODEL is a GRAPH distributed across the layers — L6 locations, L4
features, L5 displacements — and L2/3 is where that graph + the identity live (Monty's Graph Memory). This is
the home of the deleted `recognize.py`: the parallel object library is seated here, where objects belong, not
in a side faculty.

  * the LIBRARY (graph-memory): `objects`, learned online + label-free (`learn` / `add_if_novel`).
  * RECOGNITION = INCREMENTAL EVIDENCE accumulation over (object, pose) hypotheses, PERSISTENT across the
    sensorimotor sequence (never recomputed from scratch): `start` / `sense` / `best`. Pose is SOLVED (via L5's
    pose operators) not recalled, so a known object is recognised at an orientation never seen; `identify` /
    `recognize` are the one-shot convenience (sense the whole shape at once). The persistence is the fix to the
    per-step from-scratch recognition cost.
  * VOTING (CMP): columns sensing DIFFERENT parts of one object independently solve the SAME world pose; pool by
    world pose and the truth wins even when each column alone is ambiguous (`vote`). This works in the shared
    object metric frame (Monty's assumption); voting across DIFFERENT learned SR navigational frames needs
    cross-frame registration and is the deferred hard case (TARGET_ARCHITECTURE; reference_tbt_frames_and_hippocampus).

"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from .encoders import SDR, ScalarEncoder
from .l5_displacement import apply_pose, local_disps, rot

# ---- FULLY SDR-NATIVE pose: N discrete ORIENTATION BINS + VIRTUAL ROTATION (SDR_MIGRATION.md M4) --------------
# Grounded in Lewis et al. 2022 (grid-cell object recognition: a CIRCULAR BUFFER of N grid-cell orientation modules ->
# N discrete orientations; recognise a rotated object by "mental rotation" -- replay across all N orientations, pick the
# lowest-ambiguity one) and Monty (a discrete rotation-hypothesis set, evidence accumulated). Pose is QUANTISED to 2*pi/N,
# not an analytic angle: this RETIRES `align_rotations`/`pose_between` (the atan2 solve). N is divisible by 4 so the
# 90/180/270-degree cases are exact bins.
N_BINS = 24
_BIN_ANG = np.array([r * 2 * np.pi / N_BINS for r in range(N_BINS)])   # each bin's orientation
_BIN_ROT = [rot(a) for a in _BIN_ANG]                                  # precomputed rotation operators R_r (the virtual-rotation buffer)
_BIN_TOL = 0.35                                                        # set-match tolerance (>= the residual a random angle leaves after snapping to the nearest bin)

# the local patch's rotation-INVARIANT descriptor as an OVERLAP-BEARING SDR (retires the exact-match `invariant_sig` tuple)
_PATCH_DIST_ENC = ScalarEncoder(0.0, 8.0, n=48, w=7, clip=True)       # a neighbour distance -> an overlapping bump


def _desc_sdr(disps) -> SDR:
    """The rotation-INVARIANT local descriptor as an SDR (the 'what'): the union of the neighbour DISTANCES' overlapping
    bumps (distances are rotation/translation invariant). Similar patches -> overlapping SDRs, so a near-miss is SIMILAR,
    not orthogonal (the exact-match `invariant_sig` key retired). A point (no neighbours) -> the empty SDR."""
    active = set()
    for v in disps:
        active |= _PATCH_DIST_ENC.encode((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5).active
    return SDR(_PATCH_DIST_ENC.n, active)


def _bin_residual(model_disps, sensed_disps, r: int) -> float:
    """The alignment RESIDUAL of orientation bin `r`: rotate `model_disps` by `R_r` and greedily match each to its
    nearest sensed disp — the mean matched distance (the bin's "ambiguity"; Lewis's inverse firing rate). 0 = a perfect
    orientation match. Length-mismatched patches -> infinite (they cannot be the same local structure)."""
    if len(model_disps) != len(sensed_disps):
        return float("inf")
    rotated = [_BIN_ROT[r] @ v for v in model_disps]
    pool = [np.asarray(b, float) for b in sensed_disps]
    total = 0.0
    for a in rotated:
        if not pool:
            return float("inf")
        k = int(np.argmin([(a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for b in pool]))
        total += float(((a[0] - pool[k][0]) ** 2 + (a[1] - pool[k][1]) ** 2) ** 0.5)
        pool.pop(k)
    return total / len(rotated)


def _bins_matching(model_disps, sensed_disps, tol: float = _BIN_TOL, eps: float = 0.15):
    """VIRTUAL ROTATION (retires the atan2 `align_rotations`): the BEST orientation bin(s) — those whose rotation `R_r`
    aligns `model_disps` onto `sensed_disps` with the LOWEST residual (Lewis's "lowest-ambiguity / highest-firing-rate"
    orientation), swept over the N precomputed rotations ("mental rotation") instead of solving the angle. Returns the
    argmin bin PLUS any genuine ties within `eps` (a symmetric object's stabiliser orbit — several exact orientations),
    but NOT the loose halo of near-bins (which spread evidence + translation). A POINT (no local structure) is
    orientation-free -> bin 0. Returns the list of best-bin indices (empty if none is within `tol`)."""
    if len(model_disps) != len(sensed_disps):
        return []
    if len(model_disps) == 0:                                # a point: orientation-free -> canonical bin 0 (as the old θ=0)
        return [0]
    res = [_bin_residual(model_disps, sensed_disps, r) for r in range(N_BINS)]
    m = min(res)
    if m > tol:
        return []
    return [r for r in range(N_BINS) if res[r] <= m + eps]


def _bin_matches(model_disps, sensed_disps, r: int, tol: float = _BIN_TOL) -> bool:
    """Does the SINGLE orientation bin `r` still align `model_disps` onto `sensed_disps`? (the UPDATE check — verify the
    hypothesis's committed orientation holds, without re-sweeping every bin)."""
    if len(model_disps) != len(sensed_disps):
        return False
    if len(model_disps) == 0:
        return r == 0
    return _bin_residual(model_disps, sensed_disps, r) <= tol


def _desc_match(a: SDR, b: SDR) -> bool:
    """Do two local-descriptor SDRs denote the SAME local structure? Overlap-θ (retires the exact-match `invariant_sig ==`):
    the descriptor is rotation-INVARIANT (distances), so a true match at ANY orientation is (near-)IDENTICAL -> full
    overlap; a near-miss is SIMILAR (high overlap), not orthogonal. A point matches only a point (both empty)."""
    if a.w == 0 or b.w == 0:
        return a.w == b.w
    return a.overlap(b) >= max(a.w, b.w) - 2                  # near-identical distance sets (small slack for float bucket edges)


def _union(sdrs) -> SDR:
    """The union (bitwise OR) of a list of SDRs — the object's IDENTITY SDR = the pool of its node descriptors (the
    associative-recall target; a sensed node overlapping it means the object HAS such a node)."""
    n = sdrs[0].n if sdrs else _PATCH_DIST_ENC.n
    active = set()
    for s in sdrs:
        active |= s.active
    return SDR(n, active)


def _local_disps_all(locs, radius):
    """Every node's local displacement-vectors at once, in O(n * neighbours) via a SPATIAL HASH on the (integer)
    cell positions -- NOT O(n^2) scanning every pair, which is the cost that makes recognising a big object (e.g. a
    144-cell ARC block) ~O(cells^2). `radius` is small (~1.5), so each node checks only a fixed cell neighbourhood.
    Equivalent to per-node `local_disps` for integer-celled clouds (ARC cells; stored objects are integer)."""
    import math
    grid: dict = {}
    for i, p in enumerate(locs):
        grid.setdefault((round(float(p[0])), round(float(p[1]))), []).append(i)
    r = int(math.ceil(radius))
    r2 = radius * radius
    out = []
    for i, p in enumerate(locs):
        px, py = round(float(p[0])), round(float(p[1]))
        nb = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for j in grid.get((px + dx, py + dy), ()):
                    if j != i and (locs[j][0] - p[0]) ** 2 + (locs[j][1] - p[1]) ** 2 <= r2:
                        nb.append(locs[j] - p)
        out.append(np.array(nb) if nb else np.empty((0, 2)))
    return out


class ObjectGraph:
    """One object in the graph-memory, stored ONCE in its own reference frame: node LOCATIONS (L6) + each node's
    local DISPLACEMENTS (L5) + its rotation-invariant descriptor SDR (L4) + the object IDENTITY SDR. `cells_at` = L5's
    pose operator — the object reconstituted at a bin orientation (quantised by 2π/N; SDR_MIGRATION.md M4)."""

    def __init__(self, name, cloud, radius):
        self.name = name
        self.radius = radius
        self.locs = [np.asarray(c, float) for c in cloud]
        self.locs_arr = np.asarray(self.locs, float) if self.locs else np.empty((0, 2))   # (N,2) for a VECTORISED nearest (no per-node np.linalg.norm)
        self.disps = _local_disps_all(self.locs, radius)                                  # L5: equivariant geometry (O(n))
        self.desc = [_desc_sdr(d) for d in self.disps]                                    # L4: per-node rotation-invariant descriptor SDR (overlap-bearing)
        self.identity = _union(self.desc)                                                # the object's stable IDENTITY SDR (associative recall)

    def nearest(self, loc):
        d2 = ((self.locs_arr - loc) ** 2).sum(axis=1)                                     # squared distances, one vectorised op — sqrt only the winner
        i = int(np.argmin(d2))
        return i, float(d2[i] ** 0.5)

    def cells_at(self, theta, t):
        return apply_pose(self.locs, theta, t)


class _Hyp:
    __slots__ = ("obj", "theta", "t", "loc", "ev")

    def __init__(self, obj, theta, t, loc, ev):
        self.obj, self.theta, self.t, self.loc, self.ev = obj, theta, t, loc, ev


def _canonical(cloud):
    """Store an object translation-normalised (bbox-min at the origin). Rotation is handled by recognition, so any
    single observed orientation is a fine canonical frame; pose is reported relative to it."""
    locs = [np.asarray(c, float) for c in cloud]
    mn = np.min(locs, axis=0)
    return [tuple(np.round(p - mn, 6)) for p in locs]


class L23_Object(nn.Module):
    """The object/identity layer — the graph-memory + evidence recognition loop (no VSA params). `radius` = the
    local-patch radius, `keep` = prune hypotheses more than `keep` evidence below the top."""

    def __init__(self, radius: float = 1.5, keep: float = 3.0):
        super().__init__()
        self.radius = radius
        self.keep = keep
        self.objects: list[ObjectGraph] = []                        # the graph-memory (was recognize.Recognizer.models)
        self.start()

    # ---- the object library (graph-memory; learned online, label-free) -----------------------------------
    def learn(self, cloud, name=None):
        """Add an object to the graph-memory. `name` given → store under it (returns the ObjectGraph); else learn
        ONLINE + label-free (recognise-or-add; returns (name, is_new))."""
        if name is None:
            return self.add_if_novel(cloud)
        m = ObjectGraph(name, cloud, self.radius)
        self.objects.append(m)
        return m

    def add_if_novel(self, cloud):
        """Is this shape a rotation of a known object, or new? Recognise it against the library; a strong match IS
        that object (name, new=False); else store it (new=True). The object set is learned by watching."""
        name = self.identify(cloud)
        if name is not None:
            return name, False
        return self._learn_canonical(cloud).name, True

    def _learn_canonical(self, cloud) -> ObjectGraph:
        new = ObjectGraph(f"obj{len(self.objects)}", _canonical(cloud), self.radius)
        self.objects.append(new)
        return new

    # ---- recognition: persistent incremental evidence over (object, pose) hypotheses ---------------------
    def start(self):
        """Begin (or restart) a recognition session — the persistent hypothesis set the sensorimotor loop fills."""
        self.hyps: list[_Hyp] = []
        self.prev = None

    def sense(self, loc, disps, only=None):
        """One sensation (location + local displacements) — accumulate evidence over the (object, ORIENTATION-BIN)
        hypotheses, FULLY SDR-native (SDR_MIGRATION.md M4). INIT: ASSOCIATIVE RECALL shortlists objects whose IDENTITY
        SDR overlaps the sensed descriptor (skip the rest, ~O(1) in library size); for each node with a matching
        descriptor SDR, VIRTUAL ROTATION seeds one hypothesis per orientation bin `r` whose `R_r` aligns the model patch
        onto the sensed one (no atan2 solve — mental rotation over the N bins). UPDATE: path-integrate each hypothesis by
        the bin-rotated displacement, re-verify the descriptor overlap + the committed bin — match adds evidence, a
        mismatch subtracts it; prune. Persistent across calls. `only` restricts the seed to a given object list (TRACKING
        one identity)."""
        loc = np.asarray(loc, float)
        sensed = _desc_sdr(disps)
        if not self.hyps:                                   # INIT: virtual-rotation seed (no atan2), associative-recall shortlisted
            for O in (self.objects if only is None else only):
                if sensed.w and O.identity.overlap(sensed) < sensed.w - 2:   # ASSOCIATIVE RECALL: this object has no such node → skip
                    continue
                for i in range(len(O.locs)):
                    if not _desc_match(sensed, O.desc[i]):
                        continue
                    for r in _bins_matching(O.disps[i], disps):
                        self.hyps.append(_Hyp(O, float(_BIN_ANG[r]), loc - _BIN_ROT[r] @ O.locs[i], O.locs[i], 1.0))
        else:                                               # UPDATE: displacement under the hypothesised bin rotation
            d = loc - self.prev
            for h in self.hyps:
                r = int(round(h.theta / (2 * np.pi / N_BINS))) % N_BINS   # the hypothesis's committed orientation bin
                pred = h.loc + _BIN_ROT[(-r) % N_BINS] @ d                 # rot(-θ)·d = R_{-r}·d (path-integrate into the object frame)
                i, dist = h.obj.nearest(pred)
                if dist < 0.4 and _desc_match(sensed, h.obj.desc[i]) and _bin_matches(h.obj.disps[i], disps, r):
                    h.ev += 1.0                             # location + local structure + orientation all agree
                    h.loc = h.obj.locs[i]
                else:
                    h.ev -= 1.0                             # morphology / orientation mismatch
            if self.hyps:
                top = max(h.ev for h in self.hyps)
                self.hyps = [h for h in self.hyps if h.ev > top - self.keep]
        self.prev = loc

    def sense_absent(self, loc, tol: float = 0.4):
        """The ABSENT half of a hypothesis-test: observe that `loc` is EMPTY (nothing sensed there) and PENALISE
        each hypothesis that PREDICTED a cell there (its pose-placed cloud has a point within `tol`) -- the
        prediction is falsified. The complement of `sense` (present); together they let a sample at a graph-mismatch
        point discriminate EITHER way (present supports the predictor, absent eliminates it). Prune as usual."""
        loc = np.asarray(loc, float)
        for h in self.hyps:
            if any(np.linalg.norm(loc - np.asarray(p, float)) < tol for p in h.obj.cells_at(h.theta, h.t)):
                h.ev -= 1.0
        if self.hyps:
            top = max(h.ev for h in self.hyps)
            self.hyps = [h for h in self.hyps if h.ev > top - self.keep]

    def best(self):
        """The winning (object name, theta, t, evidence) of the current session, or None.

        SYMMETRY (the stabiliser): a symmetric object produces several tied (object, pose) hypotheses — its symmetry
        group (e.g. a square's C4) maps the object onto itself, so those poses are genuinely equivalent and their
        evidence stays tied forever. `max` alone breaks that tie by list order, which flips frame-to-frame -> a
        jittering pose. Instead the pose is QUOTIENTED by the stabiliser: among the tied top hypotheses (the orbit) we
        report the CANONICAL representative (a deterministic key), so a rotationally-symmetric mover tracks with a
        STABLE pose and a clean translation — no spurious rotation, no centroid tracker (rule 5). Genuine two-object
        ambiguity (tied across DIFFERENT objects) is resolved by `disambiguation_goal`, not here."""
        if not self.hyps:
            return None
        top_ev = max(h.ev for h in self.hyps)
        orbit = [h for h in self.hyps if top_ev - h.ev < 1e-6]         # the tied top hypotheses = the object's stabiliser orbit
        h = min(orbit, key=lambda h: (h.obj.name, round(float(h.theta) % (2 * np.pi), 6),
                                      round(float(h.t[0]), 3), round(float(h.t[1]), 3)))   # canonical rep (pose mod stabiliser)
        return h.obj.name, h.theta, h.t, h.ev

    def disambiguation_goal(self, margin: float = 1.5, narrowed: int = 4):
        """The hypothesis-TESTING goal (Monty's GRAPH-MISMATCH): when the field has NARROWED to a few competing
        (object, pose) hypotheses, return the WORLD location where the top-2 most DISAGREE -- a point present in one
        model (at its hypothesised pose) but FAR from every point of the other. Sensing there maximally
        discriminates them (present -> supports that hypothesis; absent -> the other). None when there is nothing to
        resolve: < 2 hypotheses; one already leads by > `margin`; or the field is NOT YET narrowed (> `narrowed`
        competitors -- the top-2 graph-mismatch is premature, let passive sensing narrow first; this is why Monty
        fires the test on TRIGGERS, not every step). DOMAIN-GENERAL: it reads only the column's own hypotheses."""
        if not (2 <= len(self.hyps) <= narrowed):              # nothing to resolve, or not yet narrowed to a few
            return None
        h1, h2 = sorted(self.hyps, key=lambda h: h.ev, reverse=True)[:2]
        if h1.ev - h2.ev > margin:                             # a clear leader -> nothing to resolve
            return None
        c1 = [np.asarray(p, float) for p in h1.obj.cells_at(h1.theta, h1.t)]
        c2 = [np.asarray(p, float) for p in h2.obj.cells_at(h2.theta, h2.t)]
        if not c1 or not c2:
            return None
        best, best_d = None, -1.0                              # the point (in either model) most distant from the other
        for cloud, other in ((c1, c2), (c2, c1)):
            for p in cloud:
                d = min(float(np.linalg.norm(p - q)) for q in other)
                if d > best_d:
                    best, best_d = p, d
        return tuple(round(float(x), 3) for x in best) if best is not None and best_d > 1e-6 else None

    # ---- one-shot convenience (sense a whole shape; built on the session) --------------------------------
    def _sense_shape(self, cloud, max_sense: int = 16, only=None) -> int:
        """Start a session and sense up to `max_sense` points SPREAD across `cloud` (every k/max_sense-th) -- so
        recognition is O(max_sense * cells), not O(cells^2) (the cost that killed it on a 144-cell ARC block). A
        handful of points discriminate; full-frame ARC never needs every cell, and the GSG is the principled
        'which points' on top of this. `only` restricts the seed to a given object list (TRACKING one identity).
        Returns the number of points actually sensed."""
        locs = [np.asarray(c, float) for c in cloud]
        k = len(locs)
        idx = range(k) if k <= max_sense else (j * k // max_sense for j in range(max_sense))
        self.start()
        m = 0
        for i in idx:
            self.sense(locs[i], local_disps(locs, i, self.radius), only=only)
            m += 1
        return m

    def identify_model(self, cloud, max_sense: int = 16):
        """Recognise a complete shape in one shot — the winning ObjectGraph, or None. Senses a SUBSAMPLE (≤
        `max_sense` points) for affordability; confidence = evidence reaching ~one match per SENSED point."""
        if not self.objects:
            return None
        m = self._sense_shape(cloud, max_sense)
        if not self.hyps:
            return None
        h = max(self.hyps, key=lambda h: h.ev)
        return h.obj if h.ev >= max(1.0, m - 1.0) else None      # confidence scales with size (a 1-point object: 1 match suffices)

    def identify(self, cloud):
        """Recognise a complete shape against the library WITHOUT adding a new one — the name, or None."""
        m = self.identify_model(cloud)
        return m.name if m is not None else None

    def recognize(self, cloud, max_sense: int = 16, learn: bool = True):
        """Identify a shape's object + continuous pose — (name, theta, t, ev). The pose-invariant recognition perception
        uses to TRACK an object across frames despite rotation/translation. The object IDENTITY is PERSISTENT: it is
        learned once and then re-recognised, not re-guessed every frame — so `learn=False` senses the pose against the
        established library WITHOUT spawning a spurious duplicate (the caller keeps the identity stable across frames, which
        is what stops the per-frame flip; §2 L2/3, TBP)."""
        m = self._sense_shape(cloud, max_sense)
        best_ev = max((h.ev for h in self.hyps), default=0.0)
        if learn and (not self.objects or best_ev < 1.0):        # clearly novel -> learn; a weak-but-nonzero match is a known object seen poorly
            self._learn_canonical(cloud)
            self._sense_shape(cloud, max_sense)
        return self.best()

    # ---- lateral voting (CMP) ----------------------------------------------------------------------------
    def vote(self, others=()):
        """Pool THIS column's (object, pose) hypotheses with its neighbours' by WORLD pose and return the consensus
        (name, theta, t, ev) — Monty's structure-preserving voting in the shared object metric frame."""
        return vote([self, *others])


_VOTE_POS_TOL = 2.0        # the world-position pooling resolution (cells) — coarse enough for the QUANTISED-pose t scatter


def vote(columns):
    """Pose-aware lateral VOTING across L2/3 columns (the CMP channel): pool every column's (object, ORIENTATION-BIN,
    world-position) hypotheses and sum the evidence. Columns sensing different parts independently infer the SAME
    (object, orientation, position), so agreement IS the consensus; the truth accumulates support and wins even when
    each column alone is ambiguous. With the FULLY SDR-native quantised pose (M4), a single glance localises the origin
    only COARSELY (the translation `t` inherits the bin-quantisation scatter, ~1 cell — it is node-dependent under a
    non-exact bin), so the world position is pooled at `_VOTE_POS_TOL` resolution, NOT the exact `t` (which would split
    the vote across the scatter). Returns (name, theta, t, ev) or None. NB assumes a SHARED metric frame across voters
    (Monty's assumption); voting across DIFFERENT learned SR-frames needs learned cross-frame registration — deferred."""
    pooled: dict = {}                                          # key -> [summed ev, representative (theta, t)]
    for col in columns:
        for h in col.hyps:
            gx = round(float(h.t[0]) / _VOTE_POS_TOL) * _VOTE_POS_TOL
            gy = round(float(h.t[1]) / _VOTE_POS_TOL) * _VOTE_POS_TOL
            key = (h.obj.name, round(h.theta, 3), gx, gy)      # (object, orientation bin, coarse world position)
            slot = pooled.setdefault(key, [0.0, (h.theta, h.t)])
            slot[0] += h.ev
    if not pooled:
        return None
    (name, th, _gx, _gy), (ev, (theta, t)) = max(pooled.items(), key=lambda kv: kv[1][0])
    return name, theta, t, ev
