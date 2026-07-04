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

The legacy VSA content store S (`pool` / `revise`) is kept as the WITHIN-object feature-at-location
superposition (read back by L4.readout); its global-blob use retires with the offline VSA path (Phase 5).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .l4_feature_location import invariant_sig
from .l5_displacement import align_rotations, apply_pose, local_disps, rot


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
    out = []
    for i, p in enumerate(locs):
        px, py = round(float(p[0])), round(float(p[1]))
        nb = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for j in grid.get((px + dx, py + dy), ()):
                    if j != i and np.linalg.norm(locs[j] - p) <= radius:
                        nb.append(locs[j] - p)
        out.append(np.array(nb) if nb else np.empty((0, 2)))
    return out


class ObjectGraph:
    """One object in the graph-memory, stored ONCE in its own reference frame: node LOCATIONS (L6) + each node's
    local DISPLACEMENTS (L5) + its rotation-invariant FEATURE signature (L4). `cells_at` = L5's pose operator —
    the object reconstituted at any continuous pose (correct by construction, no per-orientation entry)."""

    def __init__(self, name, cloud, radius):
        self.name = name
        self.radius = radius
        self.locs = [np.asarray(c, float) for c in cloud]
        self.disps = _local_disps_all(self.locs, radius)                                  # L5: equivariant geometry (O(n))
        self.sig = [invariant_sig(d) for d in self.disps]                                 # L4: invariant 'what'

    def nearest(self, loc):
        d = [np.linalg.norm(loc - n) for n in self.locs]
        i = int(np.argmin(d))
        return i, d[i]

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
    """The object/identity layer. Recognition needs no VSA params (it is the graph + evidence loop), so `feat_dim`
    / `d_mem` default for standalone use; the column passes its own. `radius` = the local-patch radius, `keep` =
    prune hypotheses more than `keep` evidence below the top."""

    def __init__(self, feat_dim: int = 256, d_mem: int = 512, radius: float = 1.5, keep: float = 3.0):
        super().__init__()
        self.register_buffer("S", torch.zeros(feat_dim, d_mem))     # legacy within-object content superposition
        self.radius = radius
        self.keep = keep
        self.objects: list[ObjectGraph] = []                        # the graph-memory (was recognize.Recognizer.models)
        self.start()

    # ---- the legacy VSA content store (within-object feature-at-location; read back by L4.readout) --------
    def pool(self, binding: torch.Tensor) -> None:
        self.S = self.S + binding

    def revise(self, place: torch.Tensor, target_value: torch.Tensor) -> None:
        """Delta-rule overwrite: drive the stored value at `place` to target_value, leaving the rest intact."""
        self.S = self.S + torch.outer(target_value - self.S @ place, place)

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

    def sense(self, loc, disps):
        """One sensation (location + local displacements) — accumulate evidence over the (object, pose) hypotheses.
        INIT (first sensation): for each model node whose LOCAL structure matches, SOLVE the pose(s) (L5) aligning
        it onto the sensed patch; seed evidence. UPDATE: project each hypothesis by the (rotated) displacement,
        find the nearest node, compare — match adds evidence, a morphology mismatch subtracts it; prune. Persistent
        across calls (never recomputed)."""
        loc = np.asarray(loc, float)
        sig = invariant_sig(disps)
        if not self.hyps:                                   # INIT: solve pose from the first sensation
            for O in self.objects:
                for i in range(len(O.locs)):
                    if O.sig[i] != sig:
                        continue
                    for theta in align_rotations(O.disps[i], disps):
                        self.hyps.append(_Hyp(O, theta, loc - rot(theta) @ O.locs[i], O.locs[i], 1.0))
        else:                                               # UPDATE: displacement under the hypothesised rotation
            d = loc - self.prev
            for h in self.hyps:
                pred = h.loc + rot(-h.theta) @ d
                i, dist = h.obj.nearest(pred)
                cands = align_rotations(h.obj.disps[i], disps) if (dist < 0.4 and h.obj.sig[i] == sig) else []
                if any(abs((th - h.theta + np.pi) % (2 * np.pi) - np.pi) < 0.05 for th in cands):
                    h.ev += 1.0                             # location + local structure + rotation all agree
                    h.loc = h.obj.locs[i]
                else:
                    h.ev -= 1.0                             # morphology mismatch
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
    def _sense_shape(self, cloud, max_sense: int = 16) -> int:
        """Start a session and sense up to `max_sense` points SPREAD across `cloud` (every k/max_sense-th) -- so
        recognition is O(max_sense * cells), not O(cells^2) (the cost that killed it on a 144-cell ARC block). A
        handful of points discriminate; full-frame ARC never needs every cell, and the GSG is the principled
        'which points' on top of this. Returns the number of points actually sensed."""
        locs = [np.asarray(c, float) for c in cloud]
        k = len(locs)
        idx = range(k) if k <= max_sense else (j * k // max_sense for j in range(max_sense))
        self.start()
        m = 0
        for i in idx:
            self.sense(locs[i], local_disps(locs, i, self.radius))
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


def vote(columns):
    """Pose-aware lateral VOTING across L2/3 columns (the CMP channel): pool every column's (object, pose)
    hypotheses by their WORLD pose and sum the evidence. An object's world pose (theta, t) is SHARED — columns
    sensing different parts independently solve the SAME (object, theta, t), so agreement on the world pose IS the
    consensus; the truth accumulates support and wins even when each column alone is ambiguous (a single glance).
    Returns (name, theta, t, ev) or None. NB assumes a SHARED metric frame across voters (Monty's assumption);
    voting across DIFFERENT learned SR-frames needs learned cross-frame registration — deferred."""
    pooled: dict = {}
    for col in columns:
        for h in col.hyps:
            key = (h.obj.name, round(h.theta, 3), round(float(h.t[0]), 2), round(float(h.t[1]), 2))
            pooled[key] = pooled.get(key, 0.0) + h.ev
    if not pooled:
        return None
    (name, th, tx, ty), ev = max(pooled.items(), key=lambda kv: kv[1])
    return name, th, (tx, ty), ev
