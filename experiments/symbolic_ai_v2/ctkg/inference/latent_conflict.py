"""
Latent hypothesis generation from symmetry conflicts between theories.

When two theories have conflicting symmetry groups over a shared observable,
this module generates the minimal latent structure that resolves the conflict.

Iron Law: no dispatch on domain concept names. All hypotheses are structural
(morph_type, body shape, ObjectId). No physics terminology in the code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph, MorphId, ObjectId
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId, SymmetryConflict

_LATENT_CONCEPT_TYPE = "LATENT_CONCEPT"


@dataclass
class ConflictLatent:
    """A latent hypothesis that resolves a symmetry conflict.

    Attributes
    ----------
    hypothesis_morphism : MorphId of the LATENT_CONCEPT morphism in the graph.
    resolving_transforms: the transforms in conflict.b_only (theory_b's symmetries).
    justification       : structural description (no physics names).
    concept_id          : ObjectId of the new "preferred frame" concept node.
    """
    hypothesis_morphism: MorphId
    resolving_transforms: list
    justification: str
    concept_id: ObjectId


def hypothesise_from_symmetry_conflict(
    conflict: SymmetryConflict,
    theory_a_id: TheoryId,
    theory_b_id: TheoryId,
    mg: MorphismGraph,
    tm: TheoryManager,
    label: str = "__latent_conflict__",
) -> Optional[ConflictLatent]:
    """Generate minimal latent hypothesis resolving a symmetry conflict.

    When theory_a has transforms T in a_only (A invariant, B not invariant),
    generates a LATENT_CONCEPT morphism asserting existence of a preferred
    reference point P such that theory_b's invariants hold in P's frame.

    The latent is: "The transforms in b_only (B's symmetries) are the true
    symmetries; A's a_only transforms hold only approximately / in a preferred
    frame."

    Parameters
    ----------
    conflict     : SymmetryConflict from compare_symmetry_groups.
    theory_a_id  : TheoryId of theory A.
    theory_b_id  : TheoryId of theory B.
    mg           : MorphismGraph to add morphisms to.
    tm           : TheoryManager.
    label        : prefix for new object labels.

    Returns
    -------
    ConflictLatent if conflict.a_only is non-empty, else None.

    Iron Law
    --------
    No physics names used. All objects are created with opaque labels derived
    from the label parameter + structural counts. The concept_id is just an
    ObjectId integer — its label string is metadata only.
    """
    if not conflict.a_only:
        return None

    # Create a new "preferred frame" concept node (opaque ObjectId, no semantics)
    pref_frame = mg.get_or_create_object(f"{label}_pref_frame")

    # Create anchor nodes for the two theories
    theory_a_anchor = mg.get_or_create_object(f"{label}_theory_a")
    theory_b_anchor = mg.get_or_create_object(f"{label}_theory_b")

    justification = (
        f"theory_a has {len(conflict.a_only)} transforms not in theory_b; "
        f"a preferred reference point resolves the conflict"
    )

    # Create the LATENT_CONCEPT morphism: theory_a_anchor → theory_b_anchor
    hyp_morph = mg.add_morphism(
        theory_a_anchor.obj_id, theory_b_anchor.obj_id,
        morph_type=_LATENT_CONCEPT_TYPE,
        payload={
            "preferred_frame": pref_frame.obj_id,
            "a_only_count": len(conflict.a_only),
            "b_only_count": len(conflict.b_only),
            "justification": justification,
        },
    )

    return ConflictLatent(
        hypothesis_morphism=hyp_morph.morph_id,
        resolving_transforms=conflict.b_only,
        justification=justification,
        concept_id=pref_frame.obj_id,
    )
