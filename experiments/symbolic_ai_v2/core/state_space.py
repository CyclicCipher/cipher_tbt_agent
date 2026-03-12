"""state_space.py — Spectral state space for sequence prediction.

Phase B of the Hankel/spectral redesign (ROADMAP_REDESIGN.md §III.3).

A StateSpace is the result of joint Tucker/SVD decomposition of the Hankel
tensor family.  It is the object set of the learned category in Mat(ℝ≥0):

    H ≈ U · diag(S) · V^T

    U[i, :] · S  = forward state vector for context row_keys[i]
    V[j, :]      = backward state vector for atom col_keys[j]

Prediction at a seen context is exact reconstruction: (U[i]*S) @ V^T.
Prediction at an unseen context uses Kan extension: degrade to shorter
suffix contexts until a match is found, falling back to marginal.

Public API
----------
StateSpace  — dataclass produced by HankelEstimator.build_state_space()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StateSpace:
    """Spectral state space derived from joint SVD of Hankel matrices.

    Attributes
    ----------
    rank       : dimension of the state space S (number of singular values kept)
    U          : forward context embedding, shape (n_contexts, rank)
    V          : atom embedding, shape (n_atoms, rank)
    S          : singular values, shape (rank,)
    row_index  : {k-gram tuple: row index in U}
    col_index  : {atom string: column index in V}
    col_keys   : ordered atom list (inverse of col_index)
    k_max      : maximum context width used during build
    raw_dist   : {k-gram tuple: {atom: probability}} — MLE distributions for
                 all seen contexts.  Used by predict_dist for exact MLE
                 prediction (strictly better than SVD reconstruction for seen
                 contexts, where rank truncation introduces unnecessary noise).
                 Populated by HankelEstimator.build_state_space(); defaults to
                 empty dict so StateSpace can still be constructed manually in
                 tests without supplying raw counts.
    """

    rank:      int
    U:         np.ndarray
    V:         np.ndarray
    S:         np.ndarray
    row_index: dict[tuple, int]
    col_index: dict[str, int]
    col_keys:  list[str]
    k_max:     int
    raw_dist:  dict = field(default_factory=dict)

    # ── Encoding ───────────────────────────────────────────────────────────

    def encode(self, left_context: tuple[str, ...]) -> np.ndarray:
        """Map a left context to a point in state space (forward encoding).

        Returns U[idx] * S — the scaled forward state vector for this context.
        This is a linear projection of the Hankel row into the state space.

        For unseen contexts returns the zero vector.
        """
        idx = self.row_index.get(left_context)
        if idx is None:
            return np.zeros(self.rank)
        return self.U[idx] * self.S

    # ── Prediction at seen contexts ────────────────────────────────────────

    def predict_dist(
        self, left_context: tuple[str, ...]
    ) -> dict[str, float]:
        """P(next atom | left_context).

        For seen contexts returns the exact MLE distribution from raw_dist
        (if populated by build_state_space).  This is strictly more accurate
        than SVD reconstruction because rank truncation introduces noise at
        seen contexts.

        Falls back to SVD reconstruction if raw_dist is empty (e.g. when
        StateSpace is constructed manually in tests without raw counts).

        Returns an empty dict if the context is not in row_index.
        """
        # Fast path: exact MLE from stored raw counts
        if self.raw_dist:
            d = self.raw_dist.get(left_context)
            if d is not None:
                return d
            # Context not seen at all — caller should try shorter suffix
            if left_context in self.row_index:
                pass  # fall through to SVD (shouldn't happen if raw_dist filled)
            else:
                return {}

        idx = self.row_index.get(left_context)
        if idx is None:
            return {}
        # SVD reconstruction fallback (used only when raw_dist is empty)
        # (rank,) @ (rank, n_atoms) → (n_atoms,)
        scores = (self.U[idx] * self.S) @ self.V.T
        scores = np.maximum(scores, 0.0)
        total = scores.sum()
        if total <= 0.0:
            return {}
        probs = scores / total
        return {
            self.col_keys[j]: float(probs[j])
            for j in range(len(self.col_keys))
            if probs[j] > 0.0
        }

    def predict_log_prob(
        self,
        left_context: tuple[str, ...],
        next_atom: str,
    ) -> float:
        """Direct log2 P(next_atom | left_context) — no dict construction.

        For the perplexity use case this is far faster than predict_dist()
        followed by .get(): we only compute the scalar probability for
        next_atom rather than building a full 192-entry distribution dict.

        Returns -math.inf when:
          - next_atom is not in the vocabulary, or
          - left_context is not seen and no fallback succeeds.
        """
        next_idx = self.col_index.get(next_atom)
        if next_idx is None:
            return float('-inf')

        # Fast path: raw MLE lookup
        if self.raw_dist:
            d = self.raw_dist.get(left_context)
            if d is not None:
                p = d.get(next_atom, 0.0)
                return math.log2(p) if p > 0.0 else float('-inf')
            return float('-inf')  # context unseen

        # SVD fallback (raw_dist not available)
        idx = self.row_index.get(left_context)
        if idx is None:
            return float('-inf')
        scores = (self.U[idx] * self.S) @ self.V.T
        scores = np.maximum(scores, 0.0)
        total = float(scores.sum())
        if total <= 0.0:
            return float('-inf')
        p = float(scores[next_idx]) / total
        return math.log2(p) if p > 0.0 else float('-inf')

    # ── Prediction at unseen contexts (Kan extension) ──────────────────────

    def predict_unseen(
        self,
        left_context: tuple[str, ...],
        fallback_atoms: list[str],
    ) -> dict[str, float]:
        """Predict for unseen contexts using Yoneda-correct state-space Kan extension.

        The left Kan extension along i: C_train ↪ C_all is approximated by:

          1. Find the nearest seen k-gram context (b₁, …, bₖ) of the same
             length by sum of squared atom-embedding distances:
                 d = Σᵢ ‖V[aᵢ] − V[bᵢ]‖²

          2. Compute the embedding delta:  Δ = Σᵢ (V[aᵢ] − V[bᵢ])

          3. Predicted state:  pred_state = U[nearest]·S + Δ

          4. Return P(next) ∝ max(0, pred_state @ V^T)

        This is domain-agnostic: no integer parsing, no token names, no guards.
        Objects are defined entirely by their co-occurrence geometry (Yoneda).
        When digit embeddings form an arithmetic progression in V-space — as
        they do for arithmetic corpora — this formula recovers the correct sum
        without ever calling int().

        Falls back to shorter-suffix exact lookup, then marginal, then uniform.
        """
        k = len(left_context)

        # ── Step 1: Yoneda nearest-neighbour Kan extension ─────────────────
        if k > 0 and self.V.shape[0] > 0:
            # Collect query atom embedding vectors; abort if any atom is unknown.
            query_vecs: list[np.ndarray] = []
            for atom in left_context:
                aidx = self.col_index.get(atom)
                if aidx is None:
                    query_vecs = []
                    break
                query_vecs.append(self.V[aidx])

            if query_vecs:
                query_V = np.stack(query_vecs, axis=0)  # (k, rank)

                best_dist    = float('inf')
                best_row_idx = -1
                best_ctx_V: np.ndarray = np.empty(0)

                for ctx, ridx in self.row_index.items():
                    if len(ctx) != k:
                        continue
                    # Embed this training context.
                    ctx_vecs: list[np.ndarray] = []
                    for atom in ctx:
                        cidx = self.col_index.get(atom)
                        if cidx is None:
                            break
                        ctx_vecs.append(self.V[cidx])
                    else:
                        if len(ctx_vecs) == k:
                            ctx_V = np.stack(ctx_vecs, axis=0)  # (k, rank)
                            dist  = float(np.sum((query_V - ctx_V) ** 2))
                            if dist < best_dist:
                                best_dist    = dist
                                best_row_idx = ridx
                                best_ctx_V   = ctx_V

                if best_row_idx >= 0:
                    # Embedding delta: Σᵢ (V[aᵢ] − V[bᵢ])
                    delta      = np.sum(query_V - best_ctx_V, axis=0)  # (rank,)
                    pred_state = self.U[best_row_idx] * self.S + delta  # (rank,)
                    scores     = pred_state @ self.V.T                   # (n_atoms,)
                    scores     = np.maximum(scores, 0.0)
                    total      = scores.sum()
                    if total > 0.0:
                        probs = scores / total
                        return {
                            self.col_keys[j]: float(probs[j])
                            for j in range(len(self.col_keys))
                            if probs[j] > 0.0
                        }

        # ── Step 2: Shorter-suffix exact lookup ────────────────────────────
        for length in range(len(left_context) - 1, 0, -1):
            ctx = left_context[-length:]
            dist_result = self.predict_dist(ctx)
            if dist_result:
                return dist_result

        # ── Step 3: Marginal — mean over all row vectors ───────────────────
        if self.U.shape[0] > 0:
            mean_state = self.U.mean(axis=0) * self.S   # (rank,)
            scores     = mean_state @ self.V.T           # (n_atoms,)
            scores     = np.maximum(scores, 0.0)
            total      = scores.sum()
            if total > 0.0:
                probs = scores / total
                return {
                    self.col_keys[j]: float(probs[j])
                    for j in range(len(self.col_keys))
                    if probs[j] > 0.0
                }

        # ── Step 4: Uniform over fallback_atoms ───────────────────────────
        if fallback_atoms:
            p = 1.0 / len(fallback_atoms)
            return {a: p for a in fallback_atoms}
        return {}
