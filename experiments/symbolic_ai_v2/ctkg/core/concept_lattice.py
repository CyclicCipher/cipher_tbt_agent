"""
FCA: formal contexts -> distributional concept lattice.

Core data structures for the distributional concept lattice produced by Phase 2
(fca_discover.py).

A DistributionalConcept captures a cluster of context rows (extent) and a
distribution over atoms (intent) discovered from the Hankel matrix H.

A ConceptLattice is the ordered set of DistributionalConcepts for one radius
level, with soft subtype edges derived from distribution inclusion.

Design notes:
  - centroid_vector  : mean of member atom one-hot vectors weighted by the concept
                       distribution.  Shape (vocab_size,).  Equivalent to the
                       concept's aggregate probability distribution over atoms.
  - extent_weights   : graded membership P(context ∈ extent) for each context.
  - intent_weights   : graded membership P(atom ∈ intent), i.e. the concept's
                       marginal distribution over atoms.
  - subtype(A, B)    : soft inclusion — E_{x ~ dist_A}[dist_B(x)] > threshold,
                       i.e. the expected atom mass of A's distribution under B.
  - The ordering (subtype edges) is computed once after FCA and cached.

See CTKG_ARCHITECTURE.md §Phase 2 for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import AtomValue, ContextKey


ConceptId = int


@dataclass
class DistributionalConcept:
    """One node in the concept lattice at a given radius level.

    Parameters
    ----------
    concept_id:
        Unique integer identifier within its ConceptLattice.
    centroid_vector:
        Shape (vocab_size,).  Normalised probability distribution over atoms —
        the aggregate intent of this concept.  Stored as a 1-D numpy array.
    extent_weights:
        Dict mapping context_hash → graded membership P(context ∈ extent).
        Sum need not equal 1 (it is the total support of this concept).
    intent_weights:
        Dict mapping atom_value → graded membership P(atom ∈ intent).
        Should equal centroid_vector indexed by atom label.
    support:
        Total evidence mass: sum of extent_weights values.
    member_contexts:
        The context hashes that were merged into this concept (for diagnostics).
    """

    concept_id: ConceptId
    centroid_vector: np.ndarray           # shape (vocab_size,)
    extent_weights: dict[ContextKey, float]
    intent_weights: dict[AtomValue, float]
    support: float
    member_contexts: list[ContextKey] = field(default_factory=list)

    # Filled in after lattice ordering is computed:
    subtypes: list[ConceptId] = field(default_factory=list)    # more specific
    supertypes: list[ConceptId] = field(default_factory=list)  # more general

    def top_atoms(self, k: int = 10) -> list[tuple[AtomValue, float]]:
        """Return the top-k atoms by intent weight, sorted descending."""
        return sorted(self.intent_weights.items(), key=lambda x: -x[1])[:k]

    def concentration_on(self, target_atoms: set[AtomValue]) -> float:
        """Fraction of intent weight concentrated on target_atoms."""
        return sum(self.intent_weights.get(a, 0.0) for a in target_atoms)

    def __repr__(self) -> str:
        top = self.top_atoms(5)
        top_str = ', '.join(f'{a}:{p:.2f}' for a, p in top)
        return (
            f"Concept(id={self.concept_id}, support={self.support:.1f}, "
            f"contexts={len(self.extent_weights)}, top=[{top_str}])"
        )


class ConceptLattice:
    """Ordered set of DistributionalConcepts for one radius level.

    Parameters
    ----------
    radius:
        The neighbourhood radius at which H was computed.
    concepts:
        List of DistributionalConcept objects, ordered by decreasing support.
    atoms:
        The full atom vocabulary (column labels of H), in order.
    subtype_threshold:
        E_{x~A}[B(x)] threshold for declaring A ⊑ B (default 0.6).
    """

    def __init__(
        self,
        radius: int,
        concepts: list[DistributionalConcept],
        atoms: list[AtomValue],
        subtype_threshold: float = 0.6,
    ) -> None:
        self.radius = radius
        self.concepts = concepts
        self.atoms = atoms
        self.subtype_threshold = subtype_threshold
        self._id_to_concept: dict[ConceptId, DistributionalConcept] = {
            c.concept_id: c for c in concepts
        }
        # Ordering is computed lazily on first access
        self._ordering_computed = False

    # ------------------------------------------------------------------
    # Lattice ordering
    # ------------------------------------------------------------------

    def compute_ordering(self) -> None:
        """Populate subtype/supertype edges for all concept pairs.

        A ⊑ B iff E_{x ~ dist_A}[dist_B(x)] > subtype_threshold.
        Concretely: dot(centroid_A, centroid_B) > threshold  (both are
        distributions, so this is the expected value of B's probability
        under A's distribution).
        """
        n = len(self.concepts)
        for i, ca in enumerate(self.concepts):
            for j, cb in enumerate(self.concepts):
                if i == j:
                    continue
                # E[B(x)] under dist_A = sum_atom P_A(atom) * P_B(atom)
                inclusion = float(np.dot(ca.centroid_vector, cb.centroid_vector))
                if inclusion > self.subtype_threshold:
                    if cb.concept_id not in ca.supertypes:
                        ca.supertypes.append(cb.concept_id)
                    if ca.concept_id not in cb.subtypes:
                        cb.subtypes.append(ca.concept_id)
        self._ordering_computed = True

    def subtype(self, a_id: ConceptId, b_id: ConceptId) -> bool:
        """Return True if concept a is a soft subtype of concept b."""
        if not self._ordering_computed:
            self.compute_ordering()
        ca = self._id_to_concept.get(a_id)
        return ca is not None and b_id in ca.supertypes

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def by_id(self, concept_id: ConceptId) -> Optional[DistributionalConcept]:
        return self._id_to_concept.get(concept_id)

    def top_concepts(self, k: int = 20) -> list[DistributionalConcept]:
        """Return top-k concepts by support (already sorted in __init__)."""
        return self.concepts[:k]

    def find_by_atom_concentration(
        self, target_atoms: set[AtomValue], threshold: float = 0.5
    ) -> list[DistributionalConcept]:
        """Return concepts whose intent is concentrated on target_atoms.

        Concentration is the fraction of intent weight on target_atoms.
        """
        return [
            c for c in self.concepts
            if c.concentration_on(target_atoms) >= threshold
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, top_k: int = 10) -> str:
        lines = [
            f"ConceptLattice(radius={self.radius}, "
            f"n_concepts={len(self.concepts)}, "
            f"vocab={len(self.atoms)})"
        ]
        for c in self.top_concepts(top_k):
            top = ', '.join(f'{a}:{p:.2f}' for a, p in c.top_atoms(5))
            lines.append(
                f"  [{c.concept_id:3d}] support={c.support:6.1f}  "
                f"contexts={len(c.extent_weights):3d}  top=[{top}]"
            )
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            f"ConceptLattice(radius={self.radius}, "
            f"n_concepts={len(self.concepts)})"
        )
