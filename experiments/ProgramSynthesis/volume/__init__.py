"""Phase-2 volume-concept prototypes (see docs/phase2/VOLUME_CONCEPTS.md)."""

from .algebra import entails, meet, volume
from .box import BoxConcept, fit_box_concept
from .halfspace import HalfspaceConcept, fit_halfspace_concept
from .relation import RelationConcept, fit_relation
from .union import UnionConcept, fit_union_concept

__all__ = [
    "BoxConcept", "fit_box_concept",
    "HalfspaceConcept", "fit_halfspace_concept",
    "UnionConcept", "fit_union_concept",
    "RelationConcept", "fit_relation",
    "meet", "entails", "volume",
]
