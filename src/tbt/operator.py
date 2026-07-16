"""operator.py — the TRANSFORM primitive (ARCHITECTURE.md §8) on a CONTINUOUS location state, in ANY dimension.

The SECOND of the column's two primitives, distinct from ASSOCIATE (`htm.HTMLayer`) and composing with it: the operator moves
the LOCATION (L6a), then the HTMLayer reads the feature at that new location (L4). That composition IS the TBT forward model
— "predict the next feature from the next MOVEMENT, not from the previous feature" (`htm.py`).

WHY THIS IS NOT AN HTMLayer. An HTMLayer MEMORISES (context → next) per instance, with NO weight-sharing across positions, so
a transition learned at one place does not exist at another (`project_place_invariance_needs_factored_state`). Path
integration needs the OPPOSITE: one action's effect, learned once, applied EVERYWHERE. That is a group action, not an
association — hence a second primitive. Do NOT try to unify them.

WHY THE STATE IS CONTINUOUS (cut-over 2026-07-14; full record: `notes/rotation_invariance_plan.md`).
This module previously represented the location as a discrete SDR and each action as an exact PERMUTATION of it (a per-module
cyclic shift). That is elegant but it is a **lattice-restricted special case**: a permutation can only express group elements
that map the code's lattice onto itself (crystallographic restriction), so translation was exact only on an axis-aligned grid
and rotation only at multiples of 360/N. Measured: quantised phases DRIFT linearly (failing by step 2, a coherent bias that
multi-scale error correction cannot bound), and continuous modular phases need N > 2π·r orientations — modules growing
linearly with object radius, quadratically in SO(3). So the location STATE is continuous and the SDR is a per-fixation
READ-OUT (its quantisation is then bounded and NEVER accumulates, because nothing is applied repeatedly).

WHY ORIENTATION IS A MATRIX, NOT AN ANGLE (cut-over 2026-07-15 — the SAME lesson, one level up).
A pose is `(position, R)`: an n-vector and an n×n ROTATION MATRIX. Orientation was a scalar `heading°`, which is an SO(2)-only
encoding — SO(3) is 3-DOF and non-abelian, so no scalar can name it. Both primary sources say matrix outright:
`reference_tbt_pose_invariant_recognition` — Monty's pose is "location + orientation (**three orthonormal vectors**),
CONTINUOUS (rotation matrices/quaternions, never discretized)"; `reference_operator_as_group_representation` (Gao 2021) —
motion is "a learned **group-representation matrix** acting on the location CODE". So 2-D was the special case all along, and
DEGREES are now a read-out (`to_angle`/`from_angle`) exactly as the grid SDR is a read-out of the continuous pose. One code
path serves n=2 (ARC frames) and n=3 (3-D environments); `wrap` is gone, because angle wrap-around was an artifact of the
scalar encoding and matrices compose without it.

WHAT IS LEARNED vs GIVEN. The operator LEARNS what each ACTION does — a body-frame displacement + a body-frame rotation,
averaged over observations, position- AND orientation-invariant BY CONSTRUCTION (the same body-frame delta everywhere).
Nothing about the action is coded: the agent cannot know what "ACTION1" means (`feedback_bitter_lesson`). A rotation applied
as a HYPOTHESIS (the pose solve) is not an action to discover but plain geometry — supplied, as the reference frame's
structure always was (ARCHITECTURE §10 P3).

NON-ABELIAN motion falls out for free: a body-frame displacement is mapped to the world through the CURRENT orientation, so
FORWARD's effect depends on which way the body faces and FORWARD;TURN ≠ TURN;FORWARD — the semidirect product Rⁿ⋊SO(n). In
3-D the ROTATIONS themselves stop commuting too (yaw∘pitch ≠ pitch∘yaw), which SE(2) cannot exhibit; `R' = R·ΔR` gets it with
no new code. Pure translation is the orientation-invariant special case, not a separate mechanism. Pure stdlib.
"""

from __future__ import annotations

import math
from typing import Hashable

_TOL = 1e-9


# ── vectors ────────────────────────────────────────────────────────────────────────────────────────────────────────
def sub(a, b) -> tuple:
    return tuple(x - y for x, y in zip(a, b))


def add(a, b) -> tuple:
    return tuple(x + y for x, y in zip(a, b))


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v) -> float:
    return math.sqrt(dot(v, v))


def scale(v, k: float) -> tuple:
    return tuple(x * k for x in v)


def dist(a, b) -> float:
    return norm(sub(a, b))


# ── rotations: an n×n matrix as a tuple of ROW tuples ──────────────────────────────────────────────────────────────
def eye(n: int) -> tuple:
    """The identity rotation in n dimensions."""
    return tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))


def rotate(R, v) -> tuple:
    """Apply a rotation to a vector — the group action. Exact at any angle, in any dimension."""
    return tuple(dot(row, v) for row in R)


def compose(A, B) -> tuple:
    """A∘B — apply B first, then A. Rotation composition is matrix multiplication (and in 3-D it does NOT commute)."""
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))) for i in range(len(A)))


def invert(R) -> tuple:
    """The inverse of a rotation — its transpose (a rotation is orthogonal, so this is exact, never a matrix solve)."""
    return tuple(tuple(R[j][i] for j in range(len(R))) for i in range(len(R[0])))


def from_angle(deg: float) -> tuple:
    """READ-IN: the 2-D rotation by `deg` degrees. Degrees are a 2-D convenience at the periphery; the STATE is the matrix."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return ((c, -s), (s, c))


def to_angle(R) -> float:
    """READ-OUT: a 2-D rotation as degrees in [0, 360). Defined only for n=2 — SO(3) has no scalar angle, which is exactly
    why the state is a matrix."""
    if len(R) != 2:
        raise ValueError(f"to_angle is a 2-D read-out; this rotation is {len(R)}-D (SO(3) has no single angle).")
    return math.degrees(math.atan2(R[1][0], R[0][0])) % 360.0


def _det(M) -> float:
    """Determinant by Laplace expansion. O(n!) — fine, and only ever called at n ≤ 3 (the dimensions of real space)."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return sum((-1.0) ** j * M[0][j] * _det([row[:j] + row[j + 1:] for row in M[1:]]) for j in range(n))


def cross(vectors) -> tuple:
    """The GENERALIZED cross product of n−1 vectors in n-space: the vector orthogonal to all of them whose sign makes the
    frame RIGHT-handed. Defined by `(v₁×…×v_{n−1})·w = det[v₁; …; v_{n−1}; w]`, so component i is that determinant with
    `w = eᵢ`. At n=2 this is the perpendicular; at n=3 the familiar cross product."""
    n = len(vectors) + 1
    out = []
    for i in range(n):
        e = [1.0 if j == i else 0.0 for j in range(n)]
        out.append(_det([list(v) for v in vectors] + [e]))
    return tuple(out)


def gram_schmidt(vectors):
    """Orthonormalize `vectors`, or None if any of them is DEPENDENT on the earlier ones. `None` is the load-bearing answer:
    dependent directions mean the sample does not pin a rotation down (coincident points in 2-D, collinear in 3-D)."""
    basis = []
    for v in vectors:
        u = tuple(v)
        for b in basis:                                   # remove what the earlier axes already explain
            u = sub(u, scale(b, dot(u, b)))
        m = norm(u)
        if m < _TOL:
            return None                                   # no independent direction left
        basis.append(scale(u, 1.0 / m))
    return basis


def frame_from(vectors):
    """Build a right-handed ORTHONORMAL frame (rows) from n−1 vectors spanning n-space: Gram-Schmidt them, then complete the
    last axis by the generalized cross product. None if they are degenerate."""
    basis = gram_schmidt(vectors)
    if basis is None:
        return None
    basis.append(cross(basis))                            # the last axis, signed for a right-handed frame
    return tuple(basis)


def solve_rotation(sensed_dirs, model_dirs):
    """SOLVE the rotation that carries `model_dirs` onto `sensed_dirs` — Monty's "align the sensed frame to the stored one",
    with the frames built from DISPLACEMENT geometry rather than surface normals (our features carry no local frame). The
    TRIAD method: orthonormalize each side into a right-handed frame (U sensed, V model), then R = Uᵀ·V, since R maps each
    model axis vᵢ onto the corresponding sensed axis uᵢ. Closed-form and exact at any angle; n−1 independent displacements
    suffice (2-D: 1; 3-D: 2). Returns None if either side is degenerate. R is ALWAYS a proper rotation (det +1), so a
    MIRRORED configuration is not silently accepted — no hypothesis explains it, and the evidence refutes it."""
    U, V = frame_from(sensed_dirs), frame_from(model_dirs)
    if U is None or V is None:
        return None
    return compose(invert(U), V)


def orthonormalize(M):
    """Project a matrix back onto SO(n) by Gram-Schmidt on its rows — used to keep an AVERAGE of rotations a rotation. This is
    the CHORDAL projection, biased toward the first axis: exact when the averaged observations agree (our case, and a no-op on
    a matrix that is already a rotation), good for tight clusters. The Karcher/Fréchet mean (or an SVD polar projection) is the
    refinement if sensor noise ever makes the bias matter — noted, not hidden."""
    return frame_from(tuple(M)[:len(M) - 1])


class MotionOperator:
    """The TRANSFORM primitive: learn what each ACTION does to the continuous pose, then apply it.

    A pose is `(position, R)` — an n-vector and an n×n rotation matrix (`reference_tbt_pose_invariant_recognition`: Monty's
    orientation is three orthonormal vectors). For each action the operator learns the mean **body-frame** displacement and
    the mean **body-frame rotation**, from observed `(before, action, after)` transitions. Body-frame is what makes it
    general: the same learned delta applies at EVERY position and EVERY orientation, and mapping it to the world through the
    current orientation is exactly the Rⁿ⋊SO(n) semidirect product — so orientation-dependent motion is non-commutative by
    construction, with no keying, no ring, and no discretisation. An unlearned action is the identity (predict staying put —
    the correct prior, and a large prediction error until it is learned)."""

    def __init__(self) -> None:
        self._acc: dict = {}                      # action -> [count, Σ body displacement, Σ body rotation]

    def learn(self, action: Hashable, before, after) -> None:
        """Observe one transition; `before`/`after` are poses `(position, R)`. The world displacement is un-rotated by the
        orientation it was made AT, giving a body-frame delta invariant to where the body was and which way it faced;
        likewise the rotation is expressed in the body's own frame (`ΔR = Rᵀ·R'`)."""
        (bp, bR), (ap, aR) = before, after
        d_body = rotate(invert(bR), sub(ap, bp))
        dR = compose(invert(bR), aR)
        acc = self._acc.get(action)
        if acc is None:
            n = len(bp)
            acc = self._acc[action] = [0, tuple(0.0 for _ in range(n)), tuple(tuple(0.0 for _ in range(n)) for _ in range(n))]
        acc[0] += 1
        acc[1] = add(acc[1], d_body)
        acc[2] = tuple(tuple(a + b for a, b in zip(ra, rb)) for ra, rb in zip(acc[2], dR))

    def move_of(self, action: Hashable):
        """The learned `(body displacement, body rotation)` for an action; None if unlearned."""
        acc = self._acc.get(action)
        if not acc or acc[0] == 0:
            return None
        n = float(acc[0])
        return scale(acc[1], 1.0 / n), orthonormalize(tuple(tuple(x / n for x in row) for row in acc[2]))

    def known(self, action: Hashable) -> bool:
        return action in self._acc

    def apply(self, pose, action: Hashable):
        """Dead-reckon: apply the learned action to a pose. The body-frame displacement is mapped to the world through the
        CURRENT orientation (hence non-commutative), then the body turns in its own frame. Exact — no rounding, so nothing
        accumulates."""
        m = self.move_of(action)
        if m is None:
            return pose
        d_body, dR = m
        p, R = pose
        return add(p, rotate(R, d_body)), compose(R, dR)
