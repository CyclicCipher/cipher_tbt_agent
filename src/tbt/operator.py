"""The OPERATOR — a composable action (L6_NONABELIAN Stage 0).

The GENERAL form of the thing L5 applies to the L6 location code. Where `l5_displacement.move_delta` is an additive
DISPLACEMENT — a vector you ADD, which HARD-WIRES commutativity (`a + b = b + a`) — an `Operator` is a linear MAP you
COMPOSE by matrix product, and matrix products do NOT commute in general (`A∘B ≠ B∘A`). The abelian TRANSLATION is the
special case (a homogeneous translation matrix, or the grid's block-diagonal phase rotation); a ROTATION is a
non-commuting witness. This OPENS THE DOOR to non-abelian structure (rotations / orderings / constrained dynamics) that a
commuting-phase code cannot hold — WITHOUT changing abelian behaviour (Stage 0 is the "negative first step").

A faithful GROUP REPRESENTATION means COMPOSITION FIDELITY: `op(a)∘op(b) == op(a∘b)`, i.e. `M(a)·M(b) = M(a∘b)`.
Translations satisfy this (and commute); the interface does NOT assume they commute. Stage 1 makes the per-action operator
LEARNED — a matrix constrained to a proper representation (Gao/TEM) — so a non-commuting action becomes representable;
the additive `move` is then the abelian special case, not the substrate. Pure numpy (small matrices; matches `l6_sr`)."""

from __future__ import annotations

import numpy as np


class Operator:
    """A linear map on a state code, COMPOSED by matrix product. `apply(z) = M·z`; `a.then(b)` = do `a` THEN `b`."""

    def __init__(self, M):
        self.M = np.asarray(M, dtype=float)

    @property
    def dim(self) -> int:
        return self.M.shape[0]

    def apply(self, z):
        """Act on a state code `z` (a `dim`-vector, or `dim×n` batch): `M·z`."""
        return self.M @ np.asarray(z, dtype=float)

    def then(self, other: "Operator") -> "Operator":
        """Compose: apply SELF first, then OTHER. The composed matrix is `other.M @ self.M` — NON-commutative in general."""
        return Operator(other.M @ self.M)

    def inverse(self) -> "Operator":
        return Operator(np.linalg.inv(self.M))

    def commutes_with(self, other: "Operator", tol: float = 1e-9) -> bool:
        """Does composition order matter? `M·N == N·M`? Abelian ⇒ True for all pairs; non-abelian ⇒ False for some."""
        return np.allclose(self.M @ other.M, other.M @ self.M, atol=tol)

    def __eq__(self, other) -> bool:
        return isinstance(other, Operator) and self.M.shape == other.M.shape and np.allclose(self.M, other.M)

    def __repr__(self) -> str:
        return f"Operator(dim={self.dim})"

    # ----- abelian instances (Stage 0): translation is the special case that reproduces `pos += delta` ---------
    @staticmethod
    def identity(dim: int) -> "Operator":
        return Operator(np.eye(dim))

    @staticmethod
    def translation(delta) -> "Operator":
        """The abelian TRANSLATION operator in HOMOGENEOUS coords: on `[x…, 1]` it ADDS `delta`. A d-vector → a
        (d+1)×(d+1) matrix. `translation(a).then(translation(b)) == translation(a+b)` (composes additively, commutes) —
        so `move_delta` IS this operator, viewed additively."""
        delta = np.asarray(delta, dtype=float)
        d = delta.shape[0]
        M = np.eye(d + 1)
        M[:d, d] = delta
        return Operator(M)

    @staticmethod
    def rotation(theta: float) -> "Operator":
        """A 2-D ROTATION about the origin, in homogeneous coords (3×3) so it composes with `translation`. The
        NON-COMMUTING witness: `rotation ∘ translation ≠ translation ∘ rotation` — the interface is not baked-in abelian."""
        c, s = float(np.cos(theta)), float(np.sin(theta))
        return Operator([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # ----- LEARN an operator from transitions (Stage 1) — the constraint is what buys composition fidelity ----
    @staticmethod
    def fit(before, after, orthogonal: bool = True) -> "Operator":
        """LEARN the operator mapping each `before` state code to its `after` (rows = samples, cols = dim) — L6_NONABELIAN
        Stage 1. `orthogonal=True` imposes the GAO REPRESENTATION CONSTRAINT (orthogonal Procrustes: `M = U Vᵀ` from
        `afterᵀ·before = U S Vᵀ`) → the learned operator is a proper rotation (spectral radius 1), so its COMPOSITION /
        POWERS stay faithful and EXTRAPOLATE. `False` = unconstrained least-squares (`M = after⁺·before`): fits ONE step
        but its powers DRIFT (spectral radius ≠ 1). The abelian composition-fidelity gate (`test_operator.py`) shows the
        constraint is REQUIRED — validated empirically before trusting non-abelian. (Batch; an ONLINE constrained update is
        a later Stage-1 refinement.)"""
        before = np.atleast_2d(np.asarray(before, dtype=float))
        after = np.atleast_2d(np.asarray(after, dtype=float))
        if orthogonal:
            U, _, Vt = np.linalg.svd(after.T @ before)
            return Operator(U @ Vt)
        return Operator(after.T @ np.linalg.pinv(before.T))

    def spectral_radius(self) -> float:
        """max |eigenvalue| — 1.0 for a faithful (unitary) representation; ≠ 1 means COMPOSITIONS/POWERS drift."""
        return float(np.max(np.abs(np.linalg.eigvals(self.M))))


class OnlineOperator:
    """LEARN an operator ONLINE from a STREAM of (before, after) transitions — the AGENT form of `Operator.fit`
    (L6_NONABELIAN Stage 1). Maintains a running cross-covariance `C ≈ Σ after ⊗ before` (a cheap rank-1 update per
    transition); the operator is read as its orthogonal PROCRUSTES `M = U Vᵀ` (SVD only on read — throttle it).

    KEY PROPERTY: this DECOUPLES accumulation (unconstrained → EXPRESSIVE) from the representation CONSTRAINT (a PROJECTION
    at read), so the constraint⊥expressivity tension does NOT bite — the constraint never fights the fit, and the operator
    stays a proper rotation (spectral radius 1) throughout. The real online challenge is COVERAGE, not the constraint: the
    operator is well-estimated only over the region the stream samples, so it needs BROAD/relatively-uniform exploration
    (a running SUM, `decay=1.0`, for a stationary operator; a gentle `decay<1` for a drifting one). A LOCAL random walk's
    peaked occupancy under-covers → poor extrapolation. (Empirically validated in `test_operator.py`.)"""

    def __init__(self, dim: int, decay: float = 1.0):
        self.dim = dim
        self.decay = decay                                            # 1.0 = a pure running SUM (stationary op); <1 = EWMA (drift)
        self.C = np.zeros((dim, dim))
        self._M = np.eye(dim)
        self._dirty = False

    def observe(self, before, after) -> None:
        """Accumulate one transition into the cross-covariance (cheap; the SVD is deferred to `operator()`)."""
        outer = np.outer(np.asarray(after, dtype=float), np.asarray(before, dtype=float))
        self.C = self.C + outer if self.decay >= 1.0 else self.decay * self.C + (1.0 - self.decay) * outer
        self._dirty = True

    def operator(self) -> Operator:
        """Read the current operator = orthogonal Procrustes of the accumulated cross-covariance (SVD; cached until the
        next `observe`). The agent throttles this (read every N steps), like the eigenpurpose SVD."""
        if self._dirty:
            U, _, Vt = np.linalg.svd(self.C)
            self._M = U @ Vt
            self._dirty = False
        return Operator(self._M)


def homog(pos):
    """Lift a position to homogeneous coords `[x…, 1]` — the state that `translation`/`rotation` operators act on."""
    return np.concatenate([np.asarray(pos, dtype=float), [1.0]])


def dehomog(z):
    """Project a homogeneous state `[x…, w]` back to a position `[x…]/w`."""
    z = np.asarray(z, dtype=float)
    return z[:-1] / z[-1]
