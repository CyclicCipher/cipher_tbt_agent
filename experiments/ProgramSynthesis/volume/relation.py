"""Phase-2 prototype — a RELATION / edge as a conditionable manifold in the joint (input × output)
space (docs/phase2/VOLUME_CONCEPTS.md §10C, case 1). The dog's kind of rule: how things relate,
learned as a manifold, applied by prediction — no symbols.

A relation is STORED as the *low-variance directions* of the standardized joint data: a near-constant
linear combination of (input, output) IS a constraint. It is USED as an operation by CONDITIONING —
given the input, solve the constraints for the output (least squares). This makes "rule = region in
product space (storage), operation when conditioned (use)" concrete (the §0 concept-vs-operation
reconciliation).

MDL-flavoured model selection picks the codimension (how many constraints): standardize so "constraint"
means linear dependence (scale-free), then keep the low-variance directions separated from the free
ones by a clear spectral gap. An independent relation yields zero constraints (it declines to predict);
a functional one yields exactly its codimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_RES = 64           # resolution → noise floor for the spectral gap
_GAP = 10.0         # a "constraint" direction must carry <~1/_GAP the variance of the free ones


@dataclass(frozen=True)
class RelationConcept:
    in_dims: int
    out_dims: int
    W: np.ndarray   # (K, in+out) constraint normals, in ORIGINAL coordinates
    c: np.ndarray   # (K,) offsets — the manifold is { z : W z ≈ c },  z = concat(x, y)

    @property
    def codim(self) -> int:
        return int(self.W.shape[0])

    def predict(self, x):
        """Condition the manifold on input x → least-squares output y solving W·[x; y] = c.
        Returns None when no constraint was learned (the relation carries no signal x→y)."""
        if self.codim == 0:
            return None
        x = np.asarray(x, dtype=float)
        Wy = self.W[:, self.in_dims:]                       # (K, out)
        rhs = self.c - self.W[:, : self.in_dims] @ x        # (K,)
        y, *_ = np.linalg.lstsq(Wy, rhs, rcond=None)
        return y


def fit_relation(inputs, outputs) -> RelationConcept:
    X = np.asarray(inputs, dtype=float)
    Y = np.asarray(outputs, dtype=float)
    if X.ndim == 1:                                         # (n,) scalars → (n, 1)
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    Z = np.hstack([X, Y])                                   # (n, d) joint points
    n, d = Z.shape
    in_dims = X.shape[1]

    mu = Z.mean(axis=0)
    std = Z.std(axis=0) + 1e-9
    Zs = (Z - mu) / std                                     # standardize → constraint = correlation
    cov = (Zs.T @ Zs) / max(n - 1, 1)
    vals, vecs = np.linalg.eigh(cov)                        # ascending eigenvalues, columns = vectors

    floor = (float(vals.sum()) / d) / (_RES ** 2)
    v = np.maximum(vals, floor)
    ratios = v[1:] / v[:-1]
    if len(ratios) and float(ratios.max()) >= _GAP:
        k = int(np.argmax(ratios))                          # gap after index k → 0..k are constraints
        cols = list(range(k + 1))
    else:
        cols = []                                           # no clear gap → no constraint (free)

    if cols:
        Ws = vecs[:, cols].T                                # (K, d) standardized normals
        W = Ws / std[None, :]                               # back to original coordinates
        c = W @ mu
    else:
        W = np.zeros((0, d))
        c = np.zeros((0,))
    return RelationConcept(in_dims, Y.shape[1], W, c)
