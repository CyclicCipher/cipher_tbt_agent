"""
Phase 8: parametric lens backward pass — compositional credit assignment.

Architecture §Phase 8:
    The parameter space P for an atomic morphism f: A → B is:
        P = ℝ × ℝ^d
    where ℝ is the log-confidence scalar and ℝ^d is the concept centroid.

    The backward pass updates BOTH:
    1. confidence: incremented (+lr) on correct predictions, decremented (-lr)
       on errors (sign-gradient / perceptron step).
    2. centroid: gradient step toward the observed token's one-hot embedding:
       centroid += lr * (observed_one_hot - centroid)
       This reduces JSD(centroid, observed_token_embedding) incrementally,
       keeping the concept's distributional representation current with
       observed data.

The centroid update is applied to the TARGET concept of each morphism whose
source concept is active at the last prefix position.

`apply_gradients` mutates the MorphismGraph IN-PLACE and returns it.
This is a deliberate performance optimisation: creating a full graph copy on
every gradient step incurs an O(n_morphisms) rebuild cost with no functional
benefit.  Confidence scalars and centroid vectors are mutable dataclass fields,
so direct assignment is valid.  Evidence counts and identity morphisms are
preserved (never written by the lens).

See CTKG_ARCHITECTURE.md §Phase 8 for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import DistributionalConcept


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LensGradient:
    """Gradient signal for a single morphism.

    Attributes
    ----------
    morph_id:
        The morphism whose confidence and centroid should be updated.
    delta_confidence:
        Signed delta for log-confidence: +1 = correct, -1 = wrong.
    target_obj_id:
        Object ID of the target concept (for centroid update).
    observed_atom_idx:
        Index of the observed token in the concept's atom vocabulary.
        -1 if the token is not in the vocabulary.
    """

    morph_id: int
    delta_confidence: float
    target_obj_id: int = -1
    observed_atom_idx: int = -1
    weight: float = 1.0
    """Frequency multiplier — set to bigram count for deduplicated lens pass."""

    def __repr__(self) -> str:
        sign = "+" if self.delta_confidence >= 0 else ""
        return (
            f"LensGradient(morph={self.morph_id}, "
            f"d={sign}{self.delta_confidence:.3f}, "
            f"tgt_obj={self.target_obj_id})"
        )


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def compute_gradients(
    mg: MorphismGraph,
    prefix: list[str],
    observed_token: str,
    r: int = 1,
) -> list[LensGradient]:
    """Compute per-morphism gradients given a prefix and the observed next token.

    Parameters
    ----------
    mg:
        The MorphismGraph containing morphism confidence weights.
    prefix:
        The token sequence observed so far.
    observed_token:
        The actual next token that was observed.
    r:
        Context radius (kept for API consistency; not used in sign-gradient).

    Returns
    -------
    One LensGradient per active non-identity morphism.
    """
    if not prefix:
        return []

    last_tok = prefix[-1]
    gradients: list[LensGradient] = []

    for obj in mg.objects():
        intent_w = obj.concept.intent_weights.get(last_tok, 0.0)
        if intent_w <= 0.0:
            continue

        for m in mg.out_morphisms(obj.obj_id, include_identity=False):
            tgt_obj = mg.object_by_id(m.target)
            if tgt_obj is None:
                continue

            # Confidence gradient: +1 if target predicts observed, -1 otherwise
            tgt_intent_w = tgt_obj.concept.intent_weights.get(observed_token, 0.0)
            delta = +1.0 if tgt_intent_w > 0.0 else -1.0

            # Centroid gradient: index of observed atom in target concept's vocab
            # Use a lazily-initialised cached mapping (keys never change, only values)
            concept = tgt_obj.concept
            if not hasattr(concept, '_atom_to_idx'):
                concept._atom_to_idx = {
                    a: j for j, a in enumerate(sorted(concept.intent_weights.keys()))
                }
            atom_idx = concept._atom_to_idx.get(observed_token, -1)

            gradients.append(LensGradient(
                morph_id=m.morph_id,
                delta_confidence=delta,
                target_obj_id=m.target,
                observed_atom_idx=atom_idx,
            ))

    return gradients


# ---------------------------------------------------------------------------
# Gradient application
# ---------------------------------------------------------------------------

def apply_gradients(
    mg: MorphismGraph,
    gradients: list[LensGradient],
    learning_rate: float = 0.05,
) -> MorphismGraph:
    """Apply gradient updates IN-PLACE and return mg.

    Updates BOTH confidence weights (Phase 8 scalar) AND concept centroids
    (Phase 8 ℝ^d component).

    Mutates morphism.confidence and obj.concept.centroid_vector directly to
    avoid the O(n_morphisms) graph rebuild cost incurred on every gradient step.
    The dataclass fields are not frozen, so direct assignment is valid.

    Parameters
    ----------
    mg:
        The current MorphismGraph (mutated in-place).
    gradients:
        List of LensGradient objects (from compute_gradients).
    learning_rate:
        Step size.  Default 0.05.

    Returns
    -------
    The same MorphismGraph, mutated.
    """
    # Accumulate confidence deltas per morphism id
    conf_deltas: dict[int, float] = {}
    # Accumulate centroid nudges per object id: obj_id → {atom_idx: total_lr}
    centroid_nudges: dict[int, dict[int, float]] = {}

    for g in gradients:
        w = g.weight  # frequency multiplier (default 1.0)
        # Confidence
        conf_deltas[g.morph_id] = (
            conf_deltas.get(g.morph_id, 0.0)
            + g.delta_confidence * learning_rate * w
        )
        # Centroid (only when observed atom is known)
        if g.target_obj_id >= 0 and g.observed_atom_idx >= 0:
            if g.target_obj_id not in centroid_nudges:
                centroid_nudges[g.target_obj_id] = {}
            idx = g.observed_atom_idx
            centroid_nudges[g.target_obj_id][idx] = (
                centroid_nudges[g.target_obj_id].get(idx, 0.0) + learning_rate * w
            )

    # In-place confidence updates — skip identity morphisms (not trained)
    for morph_id, delta in conf_deltas.items():
        morph = mg._morphisms.get(morph_id)
        if morph is not None and not morph.is_identity:
            morph.confidence += delta

    # In-place centroid updates
    for obj_id, nudge in centroid_nudges.items():
        obj = mg._objects.get(obj_id)
        if obj is not None:
            _nudge_centroid_inplace(obj.concept, nudge)

    return mg


# ---------------------------------------------------------------------------
# Centroid nudge helper
# ---------------------------------------------------------------------------

def _nudge_centroid_inplace(
    concept: DistributionalConcept,
    nudge: dict[int, float],
) -> None:
    """Update concept centroid and intent_weights in-place (no copy/replace).

    Identical arithmetic to _nudge_centroid but mutates the concept directly.
    Valid because DistributionalConcept is a non-frozen dataclass.
    """
    if not hasattr(concept, 'centroid_vector') or concept.centroid_vector is None:
        return
    cv = concept.centroid_vector
    if cv.shape[0] == 0:
        return

    total_lr = sum(nudge.values())
    cv *= (1.0 - total_lr)          # in-place shrink
    for j, lr in nudge.items():
        if j < len(cv):
            cv[j] += lr             # in-place push toward 1

    s = cv.sum()
    if s > 1e-12:
        cv /= s                     # in-place renormalise

    # Rebuild intent_weights from updated centroid (in-place dict update)
    atoms_sorted = sorted(concept.intent_weights.keys())
    for i, atom in enumerate(atoms_sorted):
        if i < len(cv):
            concept.intent_weights[atom] = float(cv[i])
        else:
            concept.intent_weights[atom] = 0.0


def _nudge_centroid(
    concept: DistributionalConcept,
    nudge: dict[int, float],
) -> DistributionalConcept:
    """Return a new DistributionalConcept with centroid moved toward observed atoms.

    Architecture §Phase 8:
        centroid += lr * (observed_one_hot - centroid)
        ≡ centroid[j] += lr * (1 - centroid[j])   for observed index j
          centroid[k] -= lr * centroid[k]           for all other indices k
    The nudge dict encodes Σ(lr) for each observed index.
    """
    if not hasattr(concept, 'centroid_vector') or concept.centroid_vector is None:
        return concept

    old_cv = concept.centroid_vector
    if old_cv.shape[0] == 0:
        return concept

    new_cv = old_cv.copy()
    total_lr = sum(nudge.values())

    # Shrink all dimensions toward 0 by total_lr
    new_cv = new_cv * (1.0 - total_lr)
    # Push observed dimensions toward 1
    for j, lr in nudge.items():
        if j < len(new_cv):
            new_cv[j] += lr

    # Renormalize to probability distribution
    s = new_cv.sum()
    if s > 1e-12:
        new_cv /= s

    # Rebuild intent_weights from updated centroid
    atoms_sorted = sorted(concept.intent_weights.keys())
    new_intent = {}
    for i, atom in enumerate(atoms_sorted):
        if i < len(new_cv):
            new_intent[atom] = float(new_cv[i])
        else:
            new_intent[atom] = 0.0

    import dataclasses
    return dataclasses.replace(
        concept,
        centroid_vector=new_cv,
        intent_weights=new_intent,
    )
