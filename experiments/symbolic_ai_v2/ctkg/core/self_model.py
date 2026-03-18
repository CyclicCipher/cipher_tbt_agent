"""Self-model: meta-FCA over morphism property matrix.

The self-model provides reflective awareness of the agent's own knowledge
structure.  Given a MorphismGraph, it constructs a binary property matrix
(morphisms × properties) and runs Formal Concept Analysis to discover
meta-concepts — clusters of morphisms that share the same property profile.

Four built-in properties are computed from morphism statistics:
  HIGH_CONFIDENCE  — confidence > high_threshold  (well-supported rule)
  CONTESTED        — confidence < low_threshold   (doubt or conflicting evidence)
  DEEPLY_COMPOSED  — len(body) > 2               (appears in deep composition chain)
  SINGLETON        — no other non-identity morphism shares the same (src, tgt)

The MetaConcept objects let the AgentLoop reason about its own uncertainty:
  - A HIGH_CONFIDENCE chain can be executed with low caution.
  - A CONTESTED morphism triggers a reflective_correction penalty (-0.1).
  - A chain containing no DEEPLY_COMPOSED morphisms is a flat (non-hierarchical) rule.

This is the categorical counterpart of the "weight of evidence" meta-level
described in CTKG_ARCHITECTURE.md §SelfModel.  The FCA is implemented here
as a simple greedy grouping (full Galois lattice construction is overkill for
N < 1000 morphisms).

See ROADMAP.md Stage 5, Step 5.2 for design decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetaConcept:
    """A set of morphisms sharing a common property profile.

    Attributes
    ----------
    name:
        One of: HIGH_CONFIDENCE, CONTESTED, DEEPLY_EVIDENCED, SINGLETON.
    morph_ids:
        IDs of the morphisms that belong to this meta-concept.
    """

    name: str
    morph_ids: list[int]

    def __repr__(self) -> str:
        return f"MetaConcept({self.name!r}, n={len(self.morph_ids)})"


# ---------------------------------------------------------------------------
# SelfModel
# ---------------------------------------------------------------------------

class SelfModel:
    """Meta-FCA over a MorphismGraph's property matrix.

    Parameters
    ----------
    confidence_high:
        Minimum confidence for HIGH_CONFIDENCE membership.
    confidence_low:
        Maximum confidence for CONTESTED membership.
    min_body_depth:
        Minimum body length for DEEPLY_COMPOSED membership.
        A morphism is DEEPLY_COMPOSED iff len(body) > min_body_depth.
        Default 2 (body = [src, tgt] is a direct edge; anything longer is
        a composition).
    """

    def __init__(
        self,
        confidence_high: float = 0.5,
        confidence_low: float = -0.5,
        min_body_depth: int = 2,
    ) -> None:
        self._conf_high = confidence_high
        self._conf_low = confidence_low
        self._min_body_depth = min_body_depth
        self._meta_concepts: list[MetaConcept] = []
        self._updated: bool = False

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, mg: MorphismGraph) -> list[MetaConcept]:
        """Run meta-FCA on mg and cache the result.

        Computes binary property membership for every non-identity morphism
        and groups morphisms by property.  Identity morphisms are excluded
        (they have confidence 0.0 by definition and would dominate every
        group).

        Parameters
        ----------
        mg:
            The current MorphismGraph (after lens update or morphism discovery).

        Returns
        -------
        List of MetaConcept objects (one per property that has >= 1 member).
        """
        morphisms = mg.morphisms(include_identity=False)
        if not morphisms:
            self._meta_concepts = []
            self._updated = True
            return []

        # Build property -> [morph_id] groups
        groups: dict[str, list[int]] = {
            "HIGH_CONFIDENCE": [],
            "CONTESTED":       [],
            "DEEPLY_COMPOSED": [],
            "SINGLETON":       [],
        }

        # Singleton detection: count morphisms per (source, target) pair
        pair_count: dict[tuple[int, int], int] = {}
        for m in morphisms:
            key = (m.source, m.target)
            pair_count[key] = pair_count.get(key, 0) + 1

        for m in morphisms:
            if m.confidence > self._conf_high:
                groups["HIGH_CONFIDENCE"].append(m.morph_id)
            if m.confidence < self._conf_low:
                groups["CONTESTED"].append(m.morph_id)
            # DEEPLY_COMPOSED: body has more than two objects (src + tgt),
            # meaning this morphism passes through at least one intermediate node.
            if len(m.body) > self._min_body_depth:
                groups["DEEPLY_COMPOSED"].append(m.morph_id)
            if pair_count[(m.source, m.target)] == 1:
                groups["SINGLETON"].append(m.morph_id)

        self._meta_concepts = [
            MetaConcept(name=name, morph_ids=ids)
            for name, ids in groups.items()
            if ids
        ]
        self._updated = True
        return list(self._meta_concepts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def meta_concepts(self) -> list[MetaConcept]:
        """Return the last computed meta-concepts.

        Returns [] until update() has been called at least once.
        """
        return list(self._meta_concepts)

    def concept_for(self, name: str) -> Optional[MetaConcept]:
        """Return the MetaConcept with the given name, or None."""
        for mc in self._meta_concepts:
            if mc.name == name:
                return mc
        return None

    def morph_properties(self, morph_id: int) -> list[str]:
        """Return all property names that morph_id belongs to."""
        return [
            mc.name for mc in self._meta_concepts
            if morph_id in mc.morph_ids
        ]

    # ------------------------------------------------------------------
    # Reflective correction
    # ------------------------------------------------------------------

    def reflective_correction(self, mg: MorphismGraph, morph_id: int) -> float:
        """Return a confidence delta for morph_id based on meta-concepts.

        Used by the AgentLoop to adjust the effective confidence of a
        morphism before using it in active inference:

          +0.1  if morph_id is in HIGH_CONFIDENCE group
          -0.1  if morph_id is in CONTESTED group
           0.0  otherwise (including if update() has never been called)

        Parameters
        ----------
        mg:
            Not used directly, but kept for API symmetry.
        morph_id:
            The morphism to query.

        Returns
        -------
        float -- confidence delta to add.
        """
        delta = 0.0
        props = self.morph_properties(morph_id)
        if "HIGH_CONFIDENCE" in props:
            delta += 0.1
        if "CONTESTED" in props:
            delta -= 0.1
        return delta

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._updated:
            return "SelfModel(not yet updated)"
        names = [mc.name for mc in self._meta_concepts]
        return f"SelfModel(meta_concepts={names})"
