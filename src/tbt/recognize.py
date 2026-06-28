"""Pose-invariant object recognition — the column's evidence-based "what + pose" faculty (TBT/Monty).

The column's SR-frame (L6) recognises locations in a FIXED navigable space, but — as Lewis et al. 2019 state
outright — "our model does not yet have a representation of orientation … it will only recognize an object if the
object is at its learned orientation." This faculty is the fix, faithful to the Thousand Brains Project's Monty
(arXiv:2412.18354): an object is stored ONCE in its own reference frame, and its pose is SOLVED for, not recalled —
so a known object is recognised at an orientation never seen, in CONTINUOUS space. It is a COLUMN FACULTY (the
"what + pose" channel), complementary to L6 (the "where"); `CorticalColumn` composes it.

DOMAIN-GENERAL (the Danganronpa litmus): it consumes `(location, local-shape descriptor)` sensations and object
models — no grid, no colours, no game. The caller's sensor turns a frame into objects + sensations.

Mechanism. A hypothesis is `(object, pose=(theta, t))` carrying a continuous EVIDENCE scalar.
  * INIT (first sensation): for each model node whose LOCAL structure matches the sensed patch, SOLVE the rotation(s)
    aligning the node's neighbour-displacements onto the sensed ones (continuous; a symmetric patch yields several —
    Monty's "two rotation hypotheses"); `t` fixes the translation. Seed evidence by the match.
  * UPDATE (move by displacement d): rotate d into the object frame (R(-theta)·d), predict the next object-frame
    location, find the nearest model node, compare its local structure. Match adds evidence; a morphology mismatch
    subtracts it. Displacement-under-the-hypothesised-rotation IS the consistency test.
  * recognised when the top hypothesis dominates.

The rotation is one universal OPERATOR (`ObjectModel.cells_at` = R(theta)·model + t), correct by construction — there
is no per-orientation entry to learn wrong (the bug class a memorised table allowed simply cannot exist here).
Learning is online + label-free: a freshly-segmented shape recognised as a rotation of a known object IS that object;
otherwise it is new (`add_if_novel`) — the object set is discovered by watching, never injected ([[feedback_bitter_lesson]]).
"""

from __future__ import annotations

import numpy as np

TOL = 1e-3


def rot(theta: float) -> np.ndarray:
    """The 2-D rotation operator R(theta) — continuous; the universal group action on the location frame."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def local_disps(locs, i, radius):
    """A node's local patch: displacement vectors to cells within `radius` (rotation-EQUIVARIANT — rotates with the
    object). This is the sensed 'feature pose' for a polyomino: the local neighbourhood geometry."""
    p = locs[i]
    return np.array([locs[j] - p for j in range(len(locs)) if j != i and np.linalg.norm(locs[j] - p) <= radius])


def invariant_sig(disps):
    """A rotation-INVARIANT signature of a patch (sorted neighbour distances) — to match WHICH node, cheaply."""
    return tuple(sorted(round(float(np.linalg.norm(v)), 3) for v in disps))


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


def align_rotations(model_disps, sensed_disps, tol=0.05):
    """SOLVE the rotation(s) theta with R(theta)·model_disps ~= sensed_disps (as sets). Continuous; returns ALL valid
    thetas (a symmetric patch yields several). This is single-sensation pose inference: read the rotation off the
    local geometry rather than searching angles."""
    if len(model_disps) != len(sensed_disps) or len(model_disps) == 0:
        return []
    out: list[float] = []
    v0 = model_disps[0]
    for w in sensed_disps:                                   # pair v0 with each equal-length sensed vector → a candidate
        if abs(np.linalg.norm(w) - np.linalg.norm(v0)) > tol:
            continue
        theta = float(np.arctan2(w[1], w[0]) - np.arctan2(v0[1], v0[0]))
        if _sets_match([rot(theta) @ v for v in model_disps], sensed_disps, tol) and \
                all(abs((theta - o + np.pi) % (2 * np.pi) - np.pi) > 1e-2 for o in out):
            out.append(theta % (2 * np.pi))
    return out


class ObjectModel:
    """One object, stored ONCE in its own reference frame: node locations + each node's local patch + its invariant
    signature. `cells_at(theta, t)` is the rotation OPERATOR — the object reconstituted at any continuous pose."""

    def __init__(self, name, cloud, radius):
        self.name = name
        self.radius = radius
        self.locs = [np.asarray(c, float) for c in cloud]
        self.disps = [local_disps(self.locs, i, radius) for i in range(len(self.locs))]
        self.sig = [invariant_sig(d) for d in self.disps]

    def nearest(self, loc):
        d = [np.linalg.norm(loc - n) for n in self.locs]
        i = int(np.argmin(d))
        return i, d[i]

    def cells_at(self, theta, t):
        """R(theta)·model + t — the object at pose (theta, t). The universal, continuous rotation operator."""
        R, t = rot(theta), np.asarray(t, float)
        return [R @ loc + t for loc in self.locs]


class _Hyp:
    __slots__ = ("obj", "theta", "t", "loc", "ev")

    def __init__(self, obj, theta, t, loc, ev):
        self.obj, self.theta, self.t, self.loc, self.ev = obj, theta, t, loc, ev


class Recognizer:
    """An evidence-based recogniser over a library of `ObjectModel`s (one cortical column / sensor). Feed it a
    sensorimotor sequence of `observe(location, local_disps)`; read `best()` for the winning (object, pose)."""

    def __init__(self, radius: float = 1.5, keep: float = 3.0):
        self.radius = radius
        self.keep = keep                                    # prune hypotheses more than `keep` evidence below the top
        self.models: list[ObjectModel] = []
        self.reset()

    # ---- the object library (learned online, label-free) ------------------------------------------------
    def add(self, name, cloud) -> ObjectModel:
        m = ObjectModel(name, cloud, self.radius)
        self.models.append(m)
        return m

    def add_if_novel(self, cloud):
        """Online discovery: is this shape a rotation of a known object, or new? Recognise it against the library;
        if it matches strongly it IS that object (return its name, new=False), else store it (new=True). So the
        object set is learned by watching — never injected."""
        name = self.identify(cloud)
        if name is not None:
            return name, False
        new = self.add(f"obj{len(self.models)}", _canonical(cloud))
        return new.name, True

    def identify(self, cloud):
        """Recognise a complete shape in one shot (sense all its points) — the name, or None if unrecognised."""
        m = self.identify_model(cloud)
        return m.name if m is not None else None

    def identify_model(self, cloud):
        """Recognise a complete shape in one shot — the winning ObjectModel, or None. Confidence = evidence reaching
        ~one match per point (a strong, full-object recognition)."""
        if not self.models:
            return None
        locs = [np.asarray(c, float) for c in cloud]
        self.reset()
        for i in range(len(locs)):
            self.observe(locs[i], local_disps(locs, i, self.radius))
        if not self.hyps:
            return None
        h = max(self.hyps, key=lambda h: h.ev)
        return h.obj if h.ev >= max(2.0, len(locs) - 1.0) else None

    def recognize(self, cloud):
        """Identify a shape's object + continuous pose, learning it online if novel (`add_if_novel`). Returns
        (name, theta, t, ev) — the full pose-invariant recognition perception uses to TRACK an object across frames
        despite rotation/translation (object permanence), where a translation-invariant shape key would not."""
        if self.identify_model(cloud) is None:
            self.add(f"obj{len(self.models)}", _canonical(cloud))
        locs = [np.asarray(c, float) for c in cloud]
        self.reset()
        for i in range(len(locs)):
            self.observe(locs[i], local_disps(locs, i, self.radius))
        return self.best()

    # ---- the evidence loop ------------------------------------------------------------------------------
    def reset(self):
        self.hyps: list[_Hyp] = []
        self.prev = None

    def observe(self, loc, disps):
        loc = np.asarray(loc, float)
        sig = invariant_sig(disps)
        if not self.hyps:                                   # INIT: solve pose from the first sensation
            for O in self.models:
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

    def best(self):
        if not self.hyps:
            return None
        h = max(self.hyps, key=lambda h: h.ev)
        return h.obj.name, h.theta, h.t, h.ev


def vote(recognizers):
    """Pose-aware lateral VOTING across columns (Monty's consensus, the CMP channel): pool every column's
    (object, pose) hypotheses by their WORLD pose and sum the evidence. The key insight is that an object's world
    pose (theta, t) is SHARED — columns sensing different parts of the same object independently solve the SAME
    (object, theta, t), so agreement on the world pose IS the consensus signal. The true pose accumulates support
    across columns and wins even when each column alone is ambiguous (a single glance). Returns (name, theta, t, ev)
    or None. NB: this assumes a SHARED metric frame across the voters (Monty's assumption); voting across columns with
    DIFFERENT learned SR-frames needs learned cross-frame registration — deferred (see TARGET_ARCHITECTURE)."""
    pooled: dict = {}
    for rec in recognizers:
        for h in rec.hyps:
            key = (h.obj.name, round(h.theta, 3), round(float(h.t[0]), 2), round(float(h.t[1]), 2))
            pooled[key] = pooled.get(key, 0.0) + h.ev
    if not pooled:
        return None
    (name, th, tx, ty), ev = max(pooled.items(), key=lambda kv: kv[1])
    return name, th, (tx, ty), ev


def _canonical(cloud):
    """Store an object translation-normalised (bbox-min at the origin). Rotation is handled by recognition, so any
    single observed orientation is a fine canonical frame; pose is reported relative to it."""
    locs = [np.asarray(c, float) for c in cloud]
    mn = np.min(locs, axis=0)
    return [tuple(np.round(p - mn, 6)) for p in locs]
