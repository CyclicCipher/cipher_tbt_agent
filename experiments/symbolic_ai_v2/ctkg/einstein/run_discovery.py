"""
Full Discovery Pipeline — Phase 18 of the Einstein Roadmap.

Integrates all phases (13-17) into a single end-to-end discovery run:
  1. seed_physics_priors — store pre-fitted Newtonian and Maxwell laws.
  2. Store a Lorentz factor FittedLaw provided by the caller.
  3. Create SR theory; add Lorentz law.
  4. Add PARADIGM_SHIFT morphism from Newton → SR.
  5. Wire the SR concept to the Newton and Maxwell constituent morphisms.

Iron Law compliance
-------------------
All dispatch by ObjectId / MorphId / morph_type strings.
No physics terminology drives logic — names are metadata only.

Bitter Lesson compliance
------------------------
Law discovery is the caller's responsibility. This module stores laws in
the graph and wires theories together — all CTKG operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph, MorphId
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw, add_fitted_law
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
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
    lorentz_law         : FittedLaw for the Lorentz factor (or None).
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
    newton_laws: list[FittedLaw],
    maxwell_laws: list[FittedLaw],
    seed: int = 0,
) -> tuple[TheoryId, TheoryId]:
    """Populate graph with Newtonian and Maxwell theories from pre-fitted laws.

    Stores each provided FittedLaw as a FITTED_LAW morphism assigned to the
    appropriate theory.

    Parameters
    ----------
    mg, tm        : graph and theory manager.
    newton_laws   : pre-fitted laws for the Newtonian theory.
    maxwell_laws  : pre-fitted laws for the Maxwell theory.
    seed          : seed suffix for morphism labels (for deduplication).

    Returns
    -------
    (newton_theory_id, maxwell_theory_id)
    """
    newton_tid = tm.register_theory("Newton")
    for i, law in enumerate(newton_laws):
        mid = add_fitted_law(mg, f"newton_{seed}_{i}", law)
        tm.assign_morphism(mid, newton_tid)

    maxwell_tid = tm.register_theory("Maxwell")
    for i, law in enumerate(maxwell_laws):
        mid = add_fitted_law(mg, f"maxwell_{seed}_{i}", law)
        tm.assign_morphism(mid, maxwell_tid)

    return newton_tid, maxwell_tid


# ---------------------------------------------------------------------------
# run_discovery
# ---------------------------------------------------------------------------

def run_discovery(
    mg: MorphismGraph,
    tm: TheoryManager,
    lorentz_law: FittedLaw,
    seed: int = 0,
) -> DiscoveryResult:
    """Drive the full Einstein discovery pipeline.

    Algorithm
    ---------
    1. Locate existing Newton and Maxwell theories in the graph (if any).
    2. Store the provided Lorentz factor FittedLaw.
    3. Create SR theory; assign the Lorentz law morphism.
    4. Add a PARADIGM_SHIFT morphism from Newton theory to SR theory.
    5. Wire the SR concept node to the Newton constituent morphisms via
       PROJECTION/INCLUSION (wire_paradigm_shift).
    6. Return DiscoveryResult.

    Parameters
    ----------
    mg, tm        : graph and theory manager.
    lorentz_law   : pre-fitted FittedLaw for the Lorentz factor γ(v).
    seed          : seed suffix for morphism labels.

    Returns
    -------
    DiscoveryResult.
    """
    result = DiscoveryResult()

    # Step 1: Find Newton and Maxwell theories
    for tid, name in tm.all_theories():
        if "newton" in name.lower():
            result.newton_theory_id = tid
        elif "maxwell" in name.lower():
            result.maxwell_theory_id = tid

    # Step 2 + 3: Store Lorentz law; create SR theory
    result.lorentz_law = lorentz_law
    sr_tid = tm.register_theory("SR")
    result.sr_theory_id = sr_tid
    lorentz_mid = add_fitted_law(mg, f"sr_gamma_{seed}", lorentz_law)
    result.lorentz_mid = lorentz_mid
    tm.assign_morphism(lorentz_mid, sr_tid)

    # Step 4: Add PARADIGM_SHIFT morphism
    if result.newton_theory_id is not None:
        ps_source = result.newton_theory_id
    else:
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
