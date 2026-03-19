"""
Stream-to-Theory Pipeline — Phase 14 of the Einstein Roadmap.

learn_from_stream
-----------------
Given a PhysicsStream (multiple observation_sets), runs discover_law on each
observation set, creates a theory in the TheoryManager, stores each law as a
FITTED_LAW morphism, and extracts shared constants (parameters with consensus
values across multiple laws).

Iron Law compliance
-------------------
No dispatch on stream names or domain names. All logic is structural: we
iterate observation sets by index, not by name. The theory name is stored
as metadata only (not used for dispatch).

Bitter Lesson compliance
------------------------
No physics knowledge is hardcoded. The same pipeline processes Newtonian
mechanics, EM waves, or any other stream with the same code path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph, MorphId
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw, add_fitted_law
from experiments.symbolic_ai_v2.ctkg.core.prim_ops import PrimSpec, get_prim_specs, make_prim_ctx
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.einstein.physics_streams import PhysicsStream
from experiments.symbolic_ai_v2.ctkg.inference.compose_search import discover_law
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId


@dataclass
class LearnedTheory:
    """Result of learning from a PhysicsStream.

    Attributes
    ----------
    theory_id        : TheoryId registered in the TheoryManager.
    morph_ids        : list of MorphIds added (one per observation_set).
    fitted_laws      : the FittedLaw objects discovered (in same order as morph_ids).
    shared_constants : dict of parameter name → value for params with consensus
                       across multiple laws.
    """
    theory_id:        TheoryId
    morph_ids:        list[MorphId] = field(default_factory=list)
    fitted_laws:      list[FittedLaw] = field(default_factory=list)
    shared_constants: dict[str, float] = field(default_factory=dict)


def learn_from_stream(
    stream: PhysicsStream,
    mg: MorphismGraph,
    tm: TheoryManager,
    prim_ctx: Optional[EvalContext] = None,
    prim_specs: Optional[list[PrimSpec]] = None,
    theory_name: str = "",
    label_prefix: str = "",
    max_depth: int = 4,
    beam_width: int = 60,
    extra_atom_values: Optional[list[float]] = None,
    consensus_tol: float = 0.02,
) -> LearnedTheory:
    """Learn laws from each observation set in a PhysicsStream.

    Algorithm
    ---------
    1. Register a new theory in the TheoryManager.
    2. For each observation_set in stream.observation_sets:
       a. Run discover_law to find the best-fit expression.
       b. Store as a FITTED_LAW morphism in the graph.
       c. Assign the morphism to the theory.
    3. Extract shared constants: parameters where ≥80% of laws that use the
       parameter agree on its value within consensus_tol (relative tolerance).
    4. Return LearnedTheory.

    Parameters
    ----------
    stream           : PhysicsStream with one or more observation_sets.
    mg               : MorphismGraph to add morphisms to.
    tm               : TheoryManager to register the theory with.
    prim_ctx         : EvalContext for primitive ops. Defaults to make_prim_ctx().
    prim_specs       : list of PrimSpec. Defaults to get_prim_specs().
    theory_name      : name for the registered theory (metadata only).
    label_prefix     : prefix for morphism labels (default: stream.name).
    max_depth        : max expression tree depth for discover_law.
    beam_width       : beam width for discover_law.
    extra_atom_values: additional fixed constants to add to the terminal set.
    consensus_tol    : relative tolerance for shared-constant consensus.
                       If std/mean < consensus_tol and ≥1 law has the param,
                       promote to shared_constants.

    Returns
    -------
    LearnedTheory with theory_id, morph_ids, fitted_laws, shared_constants.
    """
    if prim_ctx is None:
        prim_ctx = make_prim_ctx()
    if prim_specs is None:
        prim_specs = get_prim_specs()

    # Step 1: register theory
    name = theory_name or stream.name
    theory_id = tm.register_theory(name)
    prefix = label_prefix or stream.name

    morph_ids: list[MorphId] = []
    fitted_laws: list[FittedLaw] = []

    # Step 2: learn from each observation set
    for i, obs_set in enumerate(stream.observation_sets):
        if not obs_set:
            continue
        law = discover_law(
            obs_set,
            prim_ctx=prim_ctx,
            prim_specs=prim_specs,
            max_depth=max_depth,
            beam_width=beam_width,
            extra_atom_values=extra_atom_values,
        )
        mid = add_fitted_law(mg, f"{prefix}_{i}", law)
        tm.assign_morphism(mid, theory_id)
        morph_ids.append(mid)
        fitted_laws.append(law)

    # Step 3: extract shared constants
    # Collect param values across all laws
    param_values: dict[str, list[float]] = {}
    for law in fitted_laws:
        for pname, pval in law.params.items():
            if math.isfinite(pval):
                param_values.setdefault(pname, []).append(pval)

    shared_constants: dict[str, float] = {}
    for pname, vals in param_values.items():
        if len(vals) < 1:
            continue
        if len(vals) == 1:
            # Only one law has this param — still store it
            shared_constants[f"{stream.name}_{pname}"] = vals[0]
            continue
        mean_val = sum(vals) / len(vals)
        if abs(mean_val) < 1e-12:
            continue
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in vals) / len(vals))
        rel_std = std_val / abs(mean_val)
        if rel_std < consensus_tol:
            shared_constants[f"{stream.name}_{pname}"] = mean_val

    return LearnedTheory(
        theory_id=theory_id,
        morph_ids=morph_ids,
        fitted_laws=fitted_laws,
        shared_constants=shared_constants,
    )
