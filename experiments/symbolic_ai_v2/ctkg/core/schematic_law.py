"""
Parametric morphisms and law discovery — Phase 2 of the Einstein Roadmap.

A SchematicLaw is an ExprLaw with two disjoint sets of variable names:
  - params:    variable positions that are FIXED within a single family of
               examples but VARY across families.  These represent the free
               parameters of the law (e.g. k in F = k·x).
  - variables: variable positions that VARY within each family.  These are
               the true inputs to the law (e.g. x in F = k·x).

A GroundLaw is the degenerate case where all positions are constant across
all examples (the expression has no free variables at all).

Discovery algorithm
-------------------
Given `families: list[list[tuple[Expr, Expr]]]` (each family is a list of
(input_expr, output_expr) pairs sharing the same parameter values):

1. Anti-unify all input expressions → input_lgg  (positions that differ
   across ANY example become var() nodes; positions constant everywhere
   remain concrete atoms)
2. Anti-unify all output expressions → output_lgg
3. Classify each var in input_lgg:
     - If the var's value VARIES WITHIN any single family → "variable"
     - Otherwise (constant within every family, but differs across) → "parameter"
4. If input_lgg contains no vars → GroundLaw
5. Otherwise → SchematicLaw(params=..., variables=...)

Iron Law compliance
-------------------
Every operator is identified by NodeId (Expr.head is a NodeId, not a string).
No string comparisons on operator content occur inside this module.
The cage tests verify this across 10 independent anonymous symbol tables.

Bitter Lesson compliance
------------------------
The parameter/variable distinction is derived purely from the structural
variation pattern in the supplied examples.  No hardcoded parameter names,
no special-cased law templates (Hooke, Ohm, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    MorphId,
)
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import (
    Expr,
    anti_unify_list,
    match,
    substitute,
    variables as _expr_variables,
)
from experiments.symbolic_ai_v2.ctkg.core.expr_law import (
    ExprLaw,
    add_expr_law,
    query_expr_laws,
    _ensure_object,
    _find_object,
    _EXPR_LAW_TYPE,
)

_SCHEMATIC_LAW_TYPE = "SCHEMATIC_LAW"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class SchematicLaw:
    """An algebraic law schema with named free parameters.

    Attributes
    ----------
    pattern    : Expr template for the input.  Contains var() nodes for both
                 parameter positions (names in `params`) and variable positions
                 (names in `variables`).
    conclusion : Expr template for the output.  Contains var() nodes that
                 correspond to the same variable/parameter names as `pattern`.
    params     : frozenset of var names that are PARAMETERS — fixed within a
                 family, vary across families.
    variables  : frozenset of var names that are VARIABLES — vary within each
                 family (the true inputs to the law).
    evidence   : total number of training examples that generated this law.
    """
    pattern:    Expr
    conclusion: Expr
    params:     frozenset
    variables:  frozenset
    evidence:   int = 0


@dataclass
class GroundLaw:
    """A fully-concrete law with no free variables or parameters.

    Both pattern and conclusion are ground Expr trees (contain no var() nodes).
    Returned by discover_parametric_law when all examples are identical.
    """
    pattern:    Expr
    conclusion: Expr
    evidence:   int = 0


# ---------------------------------------------------------------------------
# Core discovery
# ---------------------------------------------------------------------------

def discover_parametric_law(
    families: list[list[tuple[Expr, Expr]]],
) -> Union[SchematicLaw, GroundLaw]:
    """Discover a SchematicLaw or GroundLaw from grouped example families.

    Parameters
    ----------
    families : list of families, where each family is a list of
               (input_expr, output_expr) pairs that share the same parameter
               values.  All examples within a family differ only in their
               variable positions.  Families differ in their parameter values.

    Returns
    -------
    SchematicLaw  — if at least one var() position exists in the LGG.
                    params contains positions constant-within / varying-across
                    families; variables contains positions varying-within.
    GroundLaw     — if all examples are structurally identical (LGG is ground).

    Raises
    ------
    ValueError  — if families is empty or contains no examples.
    """
    all_examples = [ex for fam in families for ex in fam]
    if not all_examples:
        raise ValueError("discover_parametric_law: no examples supplied")

    all_inputs  = [inp for inp, _out in all_examples]
    all_outputs = [out for _inp, out in all_examples]
    family_sizes = [len(fam) for fam in families]

    # Step 1 — anti-unify inputs
    input_lgg, input_substs = anti_unify_list(all_inputs)

    # Step 2 — anti-unify outputs
    output_lgg, _output_substs = anti_unify_list(all_outputs)

    # Step 3 — classify positions
    all_var_names = _expr_variables(input_lgg)
    if not all_var_names:
        # LGG is fully ground: all examples identical (or single example)
        return GroundLaw(
            pattern=input_lgg,
            conclusion=output_lgg,
            evidence=len(all_examples),
        )

    params, variables = _classify_positions(input_substs, family_sizes, all_var_names)

    return SchematicLaw(
        pattern=input_lgg,
        conclusion=output_lgg,
        params=params,
        variables=variables,
        evidence=len(all_examples),
    )


def _classify_positions(
    input_substs: list[dict[str, Expr]],
    family_sizes:  list[int],
    all_var_names: set[str],
) -> tuple[frozenset, frozenset]:
    """Partition variable names into (params, variables).

    A var name V is classified as a "variable" if its bound value differs
    across examples in at least one family.  It is classified as a "parameter"
    if its value is identical within every family (it may still differ across
    families — that is what makes it a parameter, not a constant, since it
    appeared as a var() in the LGG rather than a concrete atom).

    Returns
    -------
    (params, variables) — two disjoint frozensets of var name strings.
    """
    true_variables: set[str] = set()

    offset = 0
    for fsize in family_sizes:
        family_substs = input_substs[offset:offset + fsize]
        offset += fsize
        for v in all_var_names:
            # Collect all values this var takes within this family
            vals_in_family = {
                s[v] for s in family_substs if v in s
            }
            if len(vals_in_family) > 1:
                # v has more than one value inside this family → it's a variable
                true_variables.add(v)

    params = frozenset(all_var_names - true_variables)
    return params, frozenset(true_variables)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def instantiate(
    law: SchematicLaw,
    param_bindings: dict[str, Expr],
) -> ExprLaw:
    """Substitute parameter values into a SchematicLaw to produce a concrete ExprLaw.

    Parameters
    ----------
    law           : the schematic law to instantiate.
    param_bindings: dict mapping parameter var names → their concrete Expr values.
                    Only names in law.params are consumed; extra keys are ignored.
                    Missing parameter names leave the var() node unresolved.

    Returns
    -------
    ExprLaw whose pattern and conclusion have all parameter var() nodes
    replaced by their concrete values.  Variable var() nodes remain.

    The returned ExprLaw has morph_id = -1 (sentinel — it is not stored in any
    MorphismGraph; use add_expr_law to store if needed).
    """
    relevant = {k: v for k, v in param_bindings.items() if k in law.params}
    pat = substitute(law.pattern, relevant)
    con = substitute(law.conclusion, relevant)
    return ExprLaw(pattern=pat, conclusion=con, morph_id=-1)


# ---------------------------------------------------------------------------
# Prediction from a schema
# ---------------------------------------------------------------------------

def predict_from_schema(
    law:          SchematicLaw,
    known_params: dict[str, Expr],
    query:        Expr,
) -> Optional[tuple[dict[str, Expr], Expr]]:
    """Match query against the instantiated law and return bindings + conclusion.

    Algorithm:
      1. Instantiate law with known_params (substitute parameter values).
      2. Match query against instantiated pattern.
      3. If match succeeds, substitute bindings into instantiated conclusion.

    Parameters
    ----------
    law          : SchematicLaw to apply.
    known_params : dict mapping param names → concrete Expr values.
    query        : ground Expr to match against the instantiated pattern.

    Returns
    -------
    (variable_bindings, instantiated_conclusion) on success, or None on failure.
    variable_bindings maps each variable name in law.variables to its value.
    instantiated_conclusion has all var() nodes resolved that appear in bindings.
    """
    concrete_law = instantiate(law, known_params)
    bindings = match(concrete_law.pattern, query)
    if bindings is None:
        return None
    result = substitute(concrete_law.conclusion, bindings)
    return bindings, result


# ---------------------------------------------------------------------------
# Graph storage
# ---------------------------------------------------------------------------

def add_schematic_law(
    mg:        MorphismGraph,
    law_label: str,
    law:       Union[SchematicLaw, GroundLaw],
) -> MorphId:
    """Store a SchematicLaw or GroundLaw as a SCHEMATIC_LAW self-loop in mg.

    Deduplicates: if an identical law is already stored under law_label,
    returns the existing morphism id.

    Returns the MorphId of the stored morphism.
    """
    anchor_id = _ensure_object(mg, law_label)

    # Dedup: compare pattern + conclusion + type
    for m in mg.source_morphisms(anchor_id, morph_type=_SCHEMATIC_LAW_TYPE):
        stored = m.payload
        if (type(stored) is type(law)
                and stored.pattern == law.pattern
                and stored.conclusion == law.conclusion):
            return m.morph_id

    m = mg.add_morphism(
        anchor_id,
        anchor_id,
        morph_type=_SCHEMATIC_LAW_TYPE,
        evidence=law.evidence,
        payload=law,
    )
    return m.morph_id


def query_schematic_laws(
    mg:        MorphismGraph,
    law_label: str,
) -> list[Union[SchematicLaw, GroundLaw]]:
    """Return all SchematicLaw/GroundLaw objects stored under law_label."""
    anchor_id = _find_object(mg, law_label)
    if anchor_id is None:
        return []
    return [
        m.payload
        for m in mg.source_morphisms(anchor_id, morph_type=_SCHEMATIC_LAW_TYPE)
    ]
