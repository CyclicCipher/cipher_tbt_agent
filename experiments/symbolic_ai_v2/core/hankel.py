"""hankel.py — Hankel tensor estimator for spectral sequence learning.

Phase A of the Hankel/spectral redesign (ROADMAP_REDESIGN.md §III.2).

The Hankel matrix H_k[u_1...u_k, v] = P(u_1·...·u_k·v) is the Hom-set
of the learned category in Mat(ℝ≥0).  Its SVD is the morphism factorisation.
This is the Myhill-Nerode theorem computed in Mat(ℝ≥0) rather than Set —
spectral learning of weighted finite automata (Hsu-Kakade-Zhang 2009).

Building the family {H_1, ..., H_k_max} and decomposing them jointly finds
the minimal shared state space S across all scales simultaneously, which is
the object set of the learned category.

Public API
----------
HankelEstimator  — accumulates co-occurrence counts, builds StateSpace
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .topology import Topology
    from .state_space import StateSpace

K_MAX_DEFAULT = 4

# Threshold above which we switch from exact numpy SVD to randomized SVD.
# Exact SVD on an m×n matrix costs O(m·n²); randomized costs O(m·r·log(r))
# where r is the target rank.  For m > _RANDOMIZED_SVD_THRESHOLD the
# speedup is decisive (e.g. 400 K×192 exact ≈ 150 s, randomized ≈ 2 s).
_RANDOMIZED_SVD_THRESHOLD = 150_000  # rows


class HankelEstimator:
    """Accumulates Hankel co-occurrence counts and builds a shared state space.

    H_k[left_context, right_atom] = count of (left_context · right_atom)
    in the observed corpus, where left_context is a k-gram tuple.

    After calling observe() on the corpus, build_state_space() performs a
    joint SVD of all H_k matrices stacked vertically, yielding a StateSpace
    whose U matrix maps contexts to state vectors and whose V matrix maps
    atoms to state vectors.  The singular values S give the canonical state
    space dimension (numerical rank at 1e-3 · σ_max).
    """

    def __init__(self, k_max: int = K_MAX_DEFAULT) -> None:
        if k_max < 1:
            raise ValueError(f"k_max must be ≥ 1, got {k_max}")
        self.k_max = k_max
        # _counts[k-1][left_ctx_tuple][right_atom] = integer count
        self._counts: list[dict[tuple, dict[str, int]]] = [
            {} for _ in range(k_max)
        ]

    # ── Observation ────────────────────────────────────────────────────────

    def observe(self, seq: list[str], topo: Optional['Topology'] = None) -> None:
        """Accumulate Hankel counts from one sequence.

        For each position i and context width k = 1 .. k_max:
            left_context  = seq[i-k : i]      (k atoms before position i)
            right_context = seq[i]             (atom at position i)
            H_k[left_context][right_context] += 1

        The topo parameter is accepted for API consistency but is not used
        in the count accumulation — Hankel estimation is topology-agnostic.
        """
        n = len(seq)
        for i in range(n):
            for k in range(1, min(self.k_max, i) + 1):
                left_ctx: tuple = tuple(seq[i - k:i])
                right: str = seq[i]
                d = self._counts[k - 1]
                if left_ctx not in d:
                    d[left_ctx] = {}
                inner = d[left_ctx]
                inner[right] = inner.get(right, 0) + 1

    # ── Matrix access ──────────────────────────────────────────────────────

    def matrix(self, k: int) -> tuple[list[tuple], list[str], np.ndarray]:
        """Return (row_keys, col_keys, H_k_matrix) for context width k.

        row_keys : sorted list of k-gram tuples (left contexts)
        col_keys : sorted list of atom strings (right contexts)
        H_k_matrix : float64 array of shape (len(row_keys), len(col_keys))
                     where H[i, j] = count(row_keys[i] → col_keys[j])
        """
        if k < 1 or k > self.k_max:
            raise ValueError(f"k must be in [1, {self.k_max}], got {k}")

        counts = self._counts[k - 1]
        row_keys: list[tuple] = sorted(counts.keys())

        col_set: set[str] = set()
        for inner in counts.values():
            col_set.update(inner.keys())
        col_keys: list[str] = sorted(col_set)

        if not row_keys or not col_keys:
            return row_keys, col_keys, np.zeros((len(row_keys), len(col_keys)))

        row_idx = {r: i for i, r in enumerate(row_keys)}
        col_idx = {c: j for j, c in enumerate(col_keys)}

        mat = np.zeros((len(row_keys), len(col_keys)), dtype=np.float64)
        for left_ctx, inner in counts.items():
            i = row_idx[left_ctx]
            for right, count in inner.items():
                mat[i, col_idx[right]] = float(count)

        return row_keys, col_keys, mat

    # ── State space construction ───────────────────────────────────────────

    def build_state_space(
        self,
        k_max: int = 4,
        rank: Optional[int] = None,
    ) -> 'StateSpace':
        """Compute joint SVD of H_1 ... H_k_max to find a shared state space.

        The H_k matrices are stacked vertically (all sharing the same column
        space = atom vocabulary) and decomposed via thin SVD:

            [H_1]
            [H_2]  ≈  U · diag(S) · V^T
            [...]

        U has shape (n_all_contexts, r); V has shape (n_atoms, r); S has shape (r,).
        The same V is reused for prediction at all context widths.

        rank=None : use numerical rank at 1e-3 · σ_max (zero free parameters).
        rank=r    : truncate to exactly r singular components.

        For matrices with more than _RANDOMIZED_SVD_THRESHOLD rows, uses
        scipy's randomized SVD to stay within reasonable time bounds (e.g.
        k_max=6 on a 3 M-char corpus gives ~500 K rows; exact SVD would take
        minutes, randomized takes seconds).

        Also populates StateSpace.raw_dist — exact MLE conditional distributions
        for every seen context.  predict_dist() uses these instead of SVD
        reconstruction, which gives strictly better perplexity on seen contexts
        (rank truncation introduces noise that raw MLE avoids).

        Returns a StateSpace with shared state vectors and transition tensors.
        """
        from .state_space import StateSpace

        k_max = min(k_max, self.k_max)

        # Collect all right-context atoms across all k
        col_set: set[str] = set()
        for k in range(1, k_max + 1):
            for inner in self._counts[k - 1].values():
                col_set.update(inner.keys())
        col_keys: list[str] = sorted(col_set)
        col_index: dict[str, int] = {c: j for j, c in enumerate(col_keys)}
        n_cols = len(col_keys)

        if n_cols == 0:
            raise ValueError(
                "No observations — call observe() before build_state_space()"
            )

        # Build stacked Hankel matrix row by row.
        # k-gram tuples have different lengths so row_index keys are unambiguous.
        # Also build raw_dist: normalised row = MLE conditional distribution.
        row_index: dict[tuple, int] = {}
        raw_dist:  dict[tuple, dict[str, float]] = {}
        rows:      list[np.ndarray] = []

        for k in range(1, k_max + 1):
            for ctx in sorted(self._counts[k - 1].keys()):
                if ctx in row_index:
                    continue  # shouldn't happen; k-tuples have different lengths
                inner = self._counts[k - 1][ctx]
                row = np.zeros(n_cols, dtype=np.float64)
                for right, count in inner.items():
                    row[col_index[right]] = float(count)
                row_index[ctx] = len(rows)
                rows.append(row)
                # MLE distribution for this context
                total_count = sum(inner.values())
                if total_count > 0:
                    raw_dist[ctx] = {
                        atom: cnt / total_count
                        for atom, cnt in inner.items()
                    }

        if not rows:
            raise ValueError("Empty Hankel matrix — insufficient observations")

        H = np.vstack(rows)  # shape (n_contexts, n_atoms)
        n_rows = H.shape[0]

        # Choose SVD algorithm based on matrix size.
        # Exact SVD: O(m·n²) — fine for small matrices.
        # Randomized SVD: O(m·r·log(r)) — necessary for large matrices.
        if n_rows > _RANDOMIZED_SVD_THRESHOLD:
            # Randomized SVD for large tall-thin matrices.
            # Cap n_components well below n_cols: the auto-rank for typical
            # corpora is ~40-50; we use 100 for headroom without the O(m·r²)
            # cost of nearly-full-rank decompositions.  n_iter=2 gives good
            # approximation for the top singular vectors when there is a clear
            # spectral gap (which there always is for structured text corpora).
            r_target = rank if rank is not None else min(n_cols, 100)
            r_target = min(r_target, n_cols - 1, n_rows - 1)
            try:
                from sklearn.utils.extmath import randomized_svd
                U_full, S_full, Vt_full = randomized_svd(
                    H,
                    n_components=r_target,
                    n_iter=2,
                    random_state=42,
                )
            except ImportError:
                # scipy fallback: build sparse matrix and use ARPACK svds.
                # Much more efficient than dense randomized when nnz << m*n.
                from scipy.sparse.linalg import svds
                import scipy.sparse as sp
                H_sparse = sp.csr_matrix(H)
                r_cap = min(r_target, min(H.shape) - 1)
                U_full, S_full, Vt_full = svds(H_sparse, k=r_cap)
                # svds returns singular values in ascending order; reverse.
                idx = np.argsort(S_full)[::-1]
                U_full, S_full, Vt_full = (
                    U_full[:, idx], S_full[idx], Vt_full[idx, :]
                )
        else:
            U_full, S_full, Vt_full = np.linalg.svd(H, full_matrices=False)

        # Rank selection
        if rank is None:
            threshold = 1e-3 * S_full[0] if len(S_full) > 0 else 0.0
            r = max(1, int(np.sum(S_full > threshold)))
        else:
            r = max(1, min(rank, len(S_full)))

        U = U_full[:, :r]            # (n_contexts, r)
        S = S_full[:r]               # (r,)
        V = Vt_full[:r, :].T         # (n_atoms, r)

        return StateSpace(
            rank=r,
            U=U,
            V=V,
            S=S,
            row_index=row_index,
            col_index=col_index,
            col_keys=col_keys,
            k_max=k_max,
            raw_dist=raw_dist,
        )
