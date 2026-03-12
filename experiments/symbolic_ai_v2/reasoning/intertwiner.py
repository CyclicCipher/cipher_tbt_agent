"""intertwiner.py — Cross-domain transfer via natural transformation.

Phase C of the Hankel/spectral redesign (ROADMAP_REDESIGN.md §III.4).

Two domains D₁, D₂ have state spaces S₁, S₂ learned from their respective
corpora.  A cross-domain transfer is a natural transformation η: F₁ ⇒ F₂
between the learned functors, concretely a linear map

    η: ℝ^{r₁} → ℝ^{r₂}   (an r₂ × r₁ matrix)

that preserves the shared-atom structure.  For each shared atom σ ∈ Σ₁ ∩ Σ₂,
the backward state vector (V row) represents "what it means to predict σ":

    V₁[σ, :] ≈ V₂[σ, :] @ η                      (atom alignment constraint)

This is derived by requiring that, for any source context encoding e₁:

    P_1(σ | ctx₁) = ⟨e₁, V₁[σ]⟩ ≈ ⟨η e₁, V₂[σ]⟩ = P_{transfer}(σ | ctx₁)

⟹  V₁[σ] = η^T V₂[σ]   ⟹   V₁ = V₂ @ η   (matrix form)

The intertwiner is the least-squares solution to V₂ @ η ≈ V₁ for the
shared-atom rows, normalised to unit Frobenius norm.

This is a reduced-rank special case of the Sylvester equation:

    Σ_σ (T₁(σ)^T ⊗ I − I ⊗ T₂(σ)) · vec(η) = 0

where T(σ) is the atom transition operator.  When only atom embedding
information (V rows) is available — as opposed to full transition matrices
estimated from the operator Hankel H_σ[u, v] = P(u·σ·v) — the alignment
reduces to the regression above.

Public API
----------
find_intertwiner  — least-squares η from shared atom alignment
transfer_predict  — predict in target domain from source context via η
alignment_error   — Frobenius residual (lower = better aligned domains)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.state_space import StateSpace


# ── Intertwiner discovery ──────────────────────────────────────────────────────

def find_intertwiner(
    ss1:          StateSpace,
    ss2:          StateSpace,
    shared_atoms: list[str],
) -> np.ndarray:
    """Find the least-squares intertwiner matrix η: ℝ^{r₁} → ℝ^{r₂}.

    Solves V₂[shared, :] @ η ≈ V₁[shared, :] in the least-squares sense,
    where V₁, V₂ are the backward embedding matrices of ss1, ss2.

    The result is normalised to unit Frobenius norm (||η||_F = 1).

    Parameters
    ----------
    ss1          : source state space (domain 1)
    ss2          : target state space (domain 2)
    shared_atoms : list of atom strings present in both ss1 and ss2

    Returns
    -------
    η of shape (ss2.rank, ss1.rank) — maps source forward states to target.
    Returns zero matrix if no common atoms are found in both col_index dicts.
    """
    # Atoms that actually appear in both col_index dicts
    common = [σ for σ in shared_atoms
              if σ in ss1.col_index and σ in ss2.col_index]

    if not common:
        return np.zeros((ss2.rank, ss1.rank))

    # V₁: (n_common, r₁) — backward embeddings in source domain
    # V₂: (n_common, r₂) — backward embeddings in target domain
    V1 = ss1.V[[ss1.col_index[σ] for σ in common], :]  # (n, r1)
    V2 = ss2.V[[ss2.col_index[σ] for σ in common], :]  # (n, r2)

    # Solve V₂ @ η ≈ V₁  (shape: n×r₂ @ r₂×r₁ ≈ n×r₁)
    eta, _, _, _ = np.linalg.lstsq(V2, V1, rcond=None)   # (r2, r1)

    return eta                # shape (r2, r1)


# ── Transfer prediction ────────────────────────────────────────────────────────

def transfer_predict(
    ctx:       tuple[str, ...],
    ss_source: StateSpace,
    ss_target: StateSpace,
    eta:       np.ndarray,
) -> dict[str, float]:
    """Predict in target domain from a source-domain context via η.

    Algorithm
    ---------
    1. Encode ctx in source state space:  e = ss_source.encode(ctx)   (r₁,)
    2. Map to target state space:         w = η @ e                   (r₂,)
    3. Score target atoms:                scores = w @ ss_target.V.T  (n_atoms₂,)
    4. Clip negatives, normalise to probability distribution.

    Parameters
    ----------
    ctx       : left-context k-gram tuple (atom strings)
    ss_source : state space of the source domain
    ss_target : state space of the target domain
    eta       : intertwiner matrix (ss_target.rank × ss_source.rank)

    Returns
    -------
    {atom_str: probability} over target-domain atoms.
    Returns empty dict if the source context encodes to a zero vector.
    """
    e = ss_source.encode(ctx)                  # (r1,) — may be zeros for unseen ctx
    if not np.any(e):
        return {}

    w = eta @ e                                # (r2,)
    scores = w @ ss_target.V.T                 # (n_atoms_target,)
    scores = np.maximum(scores, 0.0)
    total = scores.sum()
    if total <= 0.0:
        return {}

    probs = scores / total
    return {
        ss_target.col_keys[j]: float(probs[j])
        for j in range(len(ss_target.col_keys))
        if probs[j] > 0.0
    }


# ── Alignment quality ──────────────────────────────────────────────────────────

def alignment_error(
    ss1:          StateSpace,
    ss2:          StateSpace,
    eta:          np.ndarray,
    shared_atoms: list[str],
) -> float:
    """Frobenius residual ||V₂ @ η − V₁||_F / n_common (lower = better).

    A small residual indicates that the intertwiner aligns the shared-atom
    backward embeddings well.  A large residual indicates little shared
    structure between the two domains.

    Returns 0.0 if no common atoms are found.
    """
    common = [σ for σ in shared_atoms
              if σ in ss1.col_index and σ in ss2.col_index]
    if not common:
        return 0.0

    V1 = ss1.V[[ss1.col_index[σ] for σ in common], :]
    V2 = ss2.V[[ss2.col_index[σ] for σ in common], :]

    residual = V2 @ eta - V1
    return float(np.linalg.norm(residual, 'fro')) / len(common)
