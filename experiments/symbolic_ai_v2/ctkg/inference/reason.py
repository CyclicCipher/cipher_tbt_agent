"""
Causal inference and do-calculus intervention on the MorphismGraph.

`intervene(mg, do_obj_id)` implements the Pearl do-calculus "graph surgery"
operation: all non-identity morphisms whose *target* is `do_obj_id` are
removed, cutting the causal influence of that object's parents.  The result
is a new MorphismGraph in which `do_obj_id` is an autonomous node (its value
is fixed externally, no longer determined by its predecessors).

This is the categorical counterpart of `intervene()` from `experiments/ctkg/graph.py`
(KnowledgeGraph.intervene), but operating on the runtime MorphismGraph produced
by discovery rather than on the declarative CTKG schema.

`causal_effect(predictor, prefix, target_token)` is a thin convenience wrapper:
it returns P(target_token | prefix) under the supplied predictor.  To compute
interventional probabilities, build a Predictor with the mutilated graph from
`intervene` and call `causal_effect` with it.

Reference: Jacobs, Kissinger & Zanasi (2019) — "Causal inference by string
diagram surgery."  MSCS.

See CTKG_ARCHITECTURE.md §Reason for the full specification.
"""

from __future__ import annotations

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor


# ---------------------------------------------------------------------------
# Intervention (graph surgery)
# ---------------------------------------------------------------------------

def intervene(
    mg: MorphismGraph,
    do_obj_id: int,
) -> MorphismGraph:
    """Return a mutilated MorphismGraph for `do(do_obj_id)`.

    All non-identity morphisms whose *target* is `do_obj_id` are removed.
    Identity morphisms are always preserved.  All other morphisms (including
    those whose source is `do_obj_id`) are preserved unchanged.

    The original graph is not modified.  Objects are copied in order so that
    the new object IDs match the old ones.

    Parameters
    ----------
    mg:
        The MorphismGraph to mutilate.
    do_obj_id:
        The object ID whose incoming morphisms should be cut.

    Returns
    -------
    A new MorphismGraph with the same objects and the surviving morphisms.
    """
    new_mg = MorphismGraph()
    obj_map: dict[int, int] = {}

    for obj in mg.objects():
        new_obj = new_mg.add_object(obj.concept, label=obj.label)
        obj_map[obj.obj_id] = new_obj.obj_id

    for m in mg.morphisms(include_identity=False):
        # Cut all incoming morphisms to the intervened-on object
        if m.target == do_obj_id:
            continue

        new_src = obj_map.get(m.source)
        new_tgt = obj_map.get(m.target)
        if new_src is None or new_tgt is None:
            continue

        new_body = [obj_map.get(oid, oid) for oid in m.body]
        new_mg.add_morphism(
            source_id=new_src,
            target_id=new_tgt,
            body=new_body,
            evidence=m.evidence_count,
            morph_type=m.morph_type,
            confidence=m.confidence,
        )

    return new_mg


# ---------------------------------------------------------------------------
# Causal effect (convenience wrapper)
# ---------------------------------------------------------------------------

def causal_effect(
    predictor: Predictor,
    prefix: list[str],
    target_token: str,
) -> float:
    """Return P(target_token | prefix) under the given predictor.

    To evaluate an interventional distribution do(X=x):
    1. Build a mutilated graph: `mg_do = intervene(mg, do_obj_id)`.
    2. Build a Predictor with `mg_do`.
    3. Call `causal_effect(predictor_do, prefix, target_token)`.

    Parameters
    ----------
    predictor:
        A fitted Predictor (possibly built from a mutilated graph).
    prefix:
        Token sequence so far.
    target_token:
        The token whose probability to return.

    Returns
    -------
    P(target_token | prefix) ∈ [0, 1].
    """
    dist = predictor.predict_next(prefix)
    return dist.get(target_token, 0.0)
