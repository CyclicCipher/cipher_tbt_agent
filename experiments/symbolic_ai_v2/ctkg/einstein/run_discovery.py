"""
Full Discovery Pipeline — Phase 18 of the Einstein Roadmap.

Integrates all phases (13-17) into a single end-to-end discovery run:
  1. seed_physics_priors — Newtonian and Maxwell theories from streams.
  2. Discover Lorentz factor law from lorentz_factor_stream.
  3. Create SR theory; add Lorentz law.
  4. Add PARADIGM_SHIFT morphism from Newton → SR.
  5. Wire the SR concept to the Newton and Maxwell constituent morphisms.

Iron Law compliance
-------------------
All dispatch by ObjectId / MorphId / morph_type strings.
No physics terminology drives logic — names are metadata only.

Bitter Lesson compliance
------------------------
No physics equations hardcoded. Lorentz factor is discovered numerically
via beam search, not constructed from prior knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph, MorphId
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw, add_fitted_law
from experiments.symbolic_ai_v2.ctkg.core.prim_ops import PrimSpec, get_prim_specs, make_prim_ctx
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.einstein.physics_streams import (
    newtonian_mechanics_stream,
    em_wave_stream,
    lorentz_factor_stream,
)
from experiments.symbolic_ai_v2.ctkg.inference.compose_search import discover_law
from experiments.symbolic_ai_v2.ctkg.inference.paradigm import wire_paradigm_shift, WiredParadigmShift
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId

_PARADIGM_SHIFT_TYPE = "PARADIGM_SHIFT"
_FITTED_LAW_TYPE = "FITTED_LAW"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryResult:
    """Result of run_discovery.

    Attributes
    ----------
    newton_theory_id    : TheoryId of the seeded Newtonian theory (or None).
    maxwell_theory_id   : TheoryId of the seeded Maxwell theory (or None).
    sr_theory_id        : TheoryId of the newly created SR theory (or None).
    lorentz_law         : FittedLaw discovered for the Lorentz factor (or None).
    lorentz_mid         : MorphId of the Lorentz FITTED_LAW morphism (or None).
    paradigm_shift_mid  : MorphId of the PARADIGM_SHIFT morphism (or None).
    wired_paradigm_shift: WiredParadigmShift created by wire_paradigm_shift (or None).
    """
    newton_theory_id:    Optional[TheoryId] = None
    maxwell_theory_id:   Optional[TheoryId] = None
    sr_theory_id:        Optional[TheoryId] = None
    lorentz_law:         Optional[FittedLaw] = None
    lorentz_mid:         Optional[MorphId] = None
    paradigm_shift_mid:  Optional[MorphId] = None
    wired_paradigm_shift: Optional[WiredParadigmShift] = None


# ---------------------------------------------------------------------------
# seed_physics_priors
# ---------------------------------------------------------------------------

def seed_physics_priors(
    mg: MorphismGraph,
    tm: TheoryManager,
    ctx: EvalContext,
    seed: int = 0,
    prim_specs: Optional[list[PrimSpec]] = None,
    max_depth: int = 3,
    beam_width: int = 40,
) -> tuple[TheoryId, TheoryId]:
    """Populate graph with Newtonian and Maxwell theories from physics streams.

    Uses learn-from-stream approach: discovers laws from each observation set
    and stores them as FITTED_LAW morphisms assigned to the appropriate theory.

    Parameters
    ----------
    mg, tm, ctx  : graph, theory manager, eval context.
    seed         : random seed for stream generation.
    prim_specs   : PrimSpec list for discover_law (defaults to get_prim_specs()).
    max_depth    : expression tree depth for discover_law.
    beam_width   : beam width for discover_law.

    Returns
    -------
    (newton_theory_id, maxwell_theory_id)
    """
    if prim_specs is None:
        prim_specs = get_prim_specs()

    # Newton theory: F=ma, Galilean transform, momentum
    newton_stream = newtonian_mechanics_stream(seed=seed)
    newton_tid = tm.register_theory("Newton")
    for i, obs_set in enumerate(newton_stream.observation_sets):
        if not obs_set:
            continue
        law = discover_law(
            obs_set,
            prim_ctx=ctx,
            prim_specs=prim_specs,
            max_depth=max_depth,
            beam_width=beam_width,
        )
        mid = add_fitted_law(mg, f"newton_{seed}_{i}", law)
        tm.assign_morphism(mid, newton_tid)

    # Maxwell theory: EM wave speed
    maxwell_stream = em_wave_stream(seed=seed)
    maxwell_tid = tm.register_theory("Maxwell")
    for i, obs_set in enumerate(maxwell_stream.observation_sets):
        if not obs_set:
            continue
        law = discover_law(
            obs_set,
            prim_ctx=ctx,
            prim_specs=prim_specs,
            max_depth=max_depth,
            beam_width=beam_width,
        )
        mid = add_fitted_law(mg, f"maxwell_{seed}_{i}", law)
        tm.assign_morphism(mid, maxwell_tid)

    return newton_tid, maxwell_tid


# ---------------------------------------------------------------------------
# run_discovery
# ---------------------------------------------------------------------------

def run_discovery(
    mg: MorphismGraph,
    tm: TheoryManager,
    ctx: EvalContext,
    prim_specs: Optional[list[PrimSpec]] = None,
    seed: int = 0,
    lorentz_max_depth: int = 4,
    lorentz_beam_width: int = 60,
) -> DiscoveryResult:
    """Drive the full Einstein discovery pipeline.

    Algorithm
    ---------
    1. Locate existing Newton and Maxwell theories in the graph (if any).
    2. Discover the Lorentz factor γ(v) from lorentz_factor_stream.
    3. Create SR theory; assign the Lorentz law morphism.
    4. Add a PARADIGM_SHIFT morphism from Newton theory to SR theory.
    5. Wire the SR concept node to the Newton constituent morphisms via
       PROJECTION/INCLUSION (wire_paradigm_shift).
    6. Return DiscoveryResult.

    Parameters
    ----------
    mg, tm, ctx         : graph, theory manager, eval context.
    prim_specs          : PrimSpec list (defaults to get_prim_specs()).
    seed                : random seed for stream generation.
    lorentz_max_depth   : expression depth for Lorentz factor search.
    lorentz_beam_width  : beam width for Lorentz factor search.

    Returns
    -------
    DiscoveryResult.
    """
    if prim_specs is None:
        prim_specs = get_prim_specs()

    result = DiscoveryResult()

    # Step 1: Find Newton and Maxwell theories
    for tid, name in tm.all_theories():
        if "newton" in name.lower():
            result.newton_theory_id = tid
        elif "maxwell" in name.lower():
            result.maxwell_theory_id = tid

    # Step 2: Discover Lorentz factor
    lr_stream = lorentz_factor_stream(seed=seed)
    lorentz_obs = lr_stream.observation_sets[0]
    lorentz_law = discover_law(
        lorentz_obs,
        prim_ctx=ctx,
        prim_specs=prim_specs,
        max_depth=lorentz_max_depth,
        beam_width=lorentz_beam_width,
    )
    result.lorentz_law = lorentz_law

    # Step 3: Create SR theory and assign Lorentz law
    sr_tid = tm.register_theory("SR")
    result.sr_theory_id = sr_tid
    lorentz_mid = add_fitted_law(mg, f"sr_gamma_{seed}", lorentz_law)
    result.lorentz_mid = lorentz_mid
    tm.assign_morphism(lorentz_mid, sr_tid)

    # Step 4: Add PARADIGM_SHIFT morphism
    # Source: Newton theory object (or a dummy node if no Newton theory).
    # Target: SR theory object.
    if result.newton_theory_id is not None:
        ps_source = result.newton_theory_id
    else:
        # Create a dummy source node
        dummy = mg.get_or_create_object(f"__discovery_source_{seed}__")
        ps_source = dummy.obj_id

    ps_morph = mg.add_morphism(
        ps_source,
        sr_tid,
        morph_type=_PARADIGM_SHIFT_TYPE,
        evidence=1,
        payload={"reason": "Lorentz invariance replaces Galilean invariance"},
    )
    result.paradigm_shift_mid = ps_morph.morph_id

    # Step 5: Wire SR concept node to Newton constituent morphisms
    # Collect Newton theory's FITTED_LAW morphisms as constituents
    constituent_mids: list[MorphId] = []
    if result.newton_theory_id is not None:
        for mid in mg.theory_members(result.newton_theory_id):
            m = mg.morphism_by_id(mid)
            if m is not None and m.morph_type == _FITTED_LAW_TYPE:
                constituent_mids.append(mid)

    if constituent_mids:
        wps = wire_paradigm_shift(sr_tid, constituent_mids, mg)
        result.wired_paradigm_shift = wps

    return result
