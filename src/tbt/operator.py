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


def homog(pos):
    """Lift a position to homogeneous coords `[x…, 1]` — the state that `translation`/`rotation` operators act on."""
    return np.concatenate([np.asarray(pos, dtype=float), [1.0]])


def dehomog(z):
    """Project a homogeneous state `[x…, w]` back to a position `[x…]/w`."""
    z = np.asarray(z, dtype=float)
    return z[:-1] / z[-1]
