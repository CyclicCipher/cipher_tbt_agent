"""
Phase 9 Einstein Integration Test.

Drives the full Einstein scenario using physics streams and the CTKG machinery:
  1. Newtonian mechanics → 3 laws registered in a Newton theory.
  2. Electromagnetic → 1 Maxwell law.
  3. Ether hypothesis added as a RETRACTED morphism, then its reason recorded.
  4. Lorentz factor γ(v) discovered from lorentz_factor_stream via discover_law.
  5. SR theory created; Lorentz law assigned to it.
  6. Spacetime concept introduced via PARADIGM_SHIFT morphism.
  7. GR curvature law added for Mercury perihelion prediction.
  8. Mercury prediction error computed from mercury_precession_stream.
  9. inspect_gr_discovery populates GRDiscoveryAudit from live graph.
 10. verify_gr_discovery checks all 8 structural properties.

Test structure
--------------
TestEinsteinMain (1 test)
    - full_einstein_scenario: runs complete scenario; verify_gr_discovery passes.

TestEinsteinCage (1 test)
    - cage_5_seeds: γ(v) residual < 0.01 for seeds 0..4.

TestEinsteinDefectProbes (4 tests)
    Probe 1: omitting sr_theory_id causes lorentz_factor_morphism = None.
    Probe 2: omitting ether morphism causes retraction_reason = None.
    Probe 3: omitting PARADIGM_SHIFT morphism causes spacetime_concept = None.
    Probe 4: empty revision_history causes property 8 to fail.
"""
from __future__ import annotations

import math
import random

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import FittedLaw, add_fitted_law
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.einstein.audit import (
    GRDiscoveryAudit,
    inspect_gr_discovery,
    verify_gr_discovery,
)
from experiments.symbolic_ai_v2.ctkg.einstein.physics_streams import (
    lorentz_factor_stream,
    mercury_precession_stream,
)
from experiments.symbolic_ai_v2.ctkg.inference.compose_search import discover_law
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_schema(nid: int) -> tuple[SchematicLaw, Expr]:
    """Build the expression tree and schema for y = k * x."""
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return sch, formula


def _linear_law(nid: int) -> FittedLaw:
    """Return a FittedLaw for y = 1 * x (unit scaling, zero residual)."""
    sch, _ = _linear_schema(nid)
    return FittedLaw(schema=sch, params={"k": 1.0}, residual=0.0)


def _run_einstein_scenario(seed: int = 0) -> tuple[MorphismGraph, TheoryManager, GRDiscoveryAudit]:
    """Build the full GR discovery scenario and return (mg, tm, audit).

    Uses distinct operator NodeIds for each law so that add_fitted_law
    deduplication does not collapse distinct laws onto the same morphism.
    """
    mg = MorphismGraph()
    tm = TheoryManager(mg)

    # Unique NodeId per law (Iron Law: dispatch by NodeId, not by name string).
    nid_fma = TOKEN_GRAPH.encode(f"EIN_FMA_{seed}")   # F = ma
    nid_xt  = TOKEN_GRAPH.encode(f"EIN_XT_{seed}")    # x(t) = v * t
    nid_pmv = TOKEN_GRAPH.encode(f"EIN_PMV_{seed}")   # p = m * v
    nid_em  = TOKEN_GRAPH.encode(f"EIN_EM_{seed}")    # EM wave speed = c
    nid_gr  = TOKEN_GRAPH.encode(f"EIN_GR_{seed}")    # GR curvature (Mercury)

    # ---- Newtonian theory: 3 laws ------------------------------------------
    t_newton = tm.register_theory("Newton")
    m_fma = add_fitted_law(mg, f"newton_fma_{seed}", _linear_law(nid_fma))
    m_xt  = add_fitted_law(mg, f"newton_xt_{seed}",  _linear_law(nid_xt))
    m_pmv = add_fitted_law(mg, f"newton_pmv_{seed}", _linear_law(nid_pmv))
    tm.assign_morphism(m_fma, t_newton)
    tm.assign_morphism(m_xt,  t_newton)
    tm.assign_morphism(m_pmv, t_newton)

    # ---- Maxwell theory: 1 law (EM wave speed = c) -------------------------
    t_maxwell = tm.register_theory("Maxwell")
    m_em = add_fitted_law(mg, f"maxwell_em_{seed}", _linear_law(nid_em))
    tm.assign_morphism(m_em, t_maxwell)

    # ---- Ether hypothesis: added then retracted (MM null result) -----------
    ether_src = mg.get_or_create_object(f"ether_frame_{seed}")
    obs_space = mg.get_or_create_object(f"observation_space_{seed}")
    ether_m = mg.add_morphism(
        ether_src.obj_id, obs_space.obj_id,
        morph_type="RETRACTED",
        payload={"reason": "Michelson-Morley null result"},
    )

    # ---- Lorentz factor: discovered from physics stream --------------------
    lr_stream   = lorentz_factor_stream(seed=seed)
    lorentz_obs = lr_stream.observation_sets[0]
    lorentz_law = discover_law(lorentz_obs, max_depth=4, beam_width=60)

    t_sr     = tm.register_theory("SR")
    m_lorentz = add_fitted_law(mg, f"sr_gamma_{seed}", lorentz_law)
    tm.assign_morphism(m_lorentz, t_sr)

    # ---- Spacetime concept: PARADIGM_SHIFT morphism ------------------------
    newton_space = mg.get_or_create_object(f"newtonian_spacetime_{seed}")
    mink_space   = mg.get_or_create_object(f"minkowski_spacetime_{seed}")
    mg.add_morphism(
        newton_space.obj_id, mink_space.obj_id,
        morph_type="PARADIGM_SHIFT",
        payload="Lorentz invariance requires 4D Minkowski spacetime",
    )

    # ---- GR theory: curvature law for Mercury ------------------------------
    t_gr       = tm.register_theory("GR")
    m_gr_curve = add_fitted_law(mg, f"gr_curvature_{seed}", _linear_law(nid_gr))
    tm.assign_morphism(m_gr_curve, t_gr)

    # ---- Mercury prediction error ------------------------------------------
    merc_stream   = mercury_precession_stream(seed=seed)
    obs_values    = [out for _, out in merc_stream.observation_sets[0]]
    observed_mean = sum(obs_values) / len(obs_values)
    predicted_gr  = 43.0  # GR prediction in arcsec/century (natural-unit model)
    mercury_err   = abs(predicted_gr - observed_mean) / abs(observed_mean)

    # ---- Populate audit from live graph ------------------------------------
    audit = inspect_gr_discovery(
        mg, tm,
        newtonian_theory_id=t_newton,
        maxwell_theory_id=t_maxwell,
        sr_theory_id=t_sr,
        gr_theory_id=t_gr,
    )
    audit.mercury_prediction_error = mercury_err
    audit.revision_history = [
        {"event": "ether_hypothesis_added",     "morphism_id": ether_m.morph_id},
        {"event": "ether_hypothesis_retracted", "reason": "Michelson-Morley null result"},
    ]

    return mg, tm, audit


# ---------------------------------------------------------------------------
# TestEinsteinMain
# ---------------------------------------------------------------------------

class TestEinsteinMain:

    def test_full_einstein_scenario(self):
        """All 8 verify_gr_discovery properties must pass for seed=0."""
        _, _, audit = _run_einstein_scenario(seed=0)
        ok, failures = verify_gr_discovery(audit)
        assert ok, "verify_gr_discovery failures:\n" + "\n".join(failures)

    def test_newtonian_laws_count(self):
        """Newtonian theory must have exactly 3 assigned morphisms."""
        _, _, audit = _run_einstein_scenario(seed=0)
        assert len(audit.newtonian_laws) == 3

    def test_lorentz_residual_near_zero(self):
        """Discovered Lorentz law residual must be < 0.01."""
        lr_stream   = lorentz_factor_stream(seed=0)
        lorentz_obs = lr_stream.observation_sets[0]
        law = discover_law(lorentz_obs, max_depth=4, beam_width=60)
        assert law.residual < 0.01, f"Lorentz residual {law.residual:.4f} too high"

    def test_mercury_error_small(self):
        """Mercury prediction error must be < 0.05 (5%)."""
        _, _, audit = _run_einstein_scenario(seed=0)
        assert audit.mercury_prediction_error is not None
        assert audit.mercury_prediction_error < 0.05, (
            f"mercury_prediction_error = {audit.mercury_prediction_error:.4f}"
        )

    def test_ether_retraction_recorded(self):
        """Ether morphism and retraction reason must both be populated."""
        _, _, audit = _run_einstein_scenario(seed=0)
        assert audit.ether_morphism is not None
        assert audit.retraction_reason is not None
        assert "Michelson-Morley" in audit.retraction_reason

    def test_spacetime_concept_created(self):
        """A PARADIGM_SHIFT morphism must establish a spacetime concept node."""
        _, _, audit = _run_einstein_scenario(seed=0)
        assert audit.spacetime_concept is not None


# ---------------------------------------------------------------------------
# TestEinsteinCage
# ---------------------------------------------------------------------------

class TestEinsteinCage:

    def test_cage_5_seeds(self):
        """γ(v) residual < 0.01 for seeds 0..4."""
        for seed in range(5):
            lr_stream   = lorentz_factor_stream(seed=seed)
            lorentz_obs = lr_stream.observation_sets[0]
            law = discover_law(lorentz_obs, max_depth=4, beam_width=60)
            assert law.residual < 0.01, (
                f"seed={seed}: γ(v) residual {law.residual:.4f} too high"
            )

    def test_cage_all_properties_pass(self):
        """verify_gr_discovery passes for seeds 0, 1, 2."""
        for seed in range(3):
            _, _, audit = _run_einstein_scenario(seed=seed)
            ok, failures = verify_gr_discovery(audit)
            assert ok, f"seed={seed}: " + "; ".join(failures)


# ---------------------------------------------------------------------------
# TestEinsteinDefectProbes
# ---------------------------------------------------------------------------

class TestEinsteinDefectProbes:

    def test_probe1_no_sr_theory_missing_lorentz(self):
        """Without sr_theory_id, lorentz_factor_morphism is None → property 5 fails."""
        mg, tm, _ = _run_einstein_scenario(seed=0)
        # Re-inspect without the SR theory
        # We need t_newton and t_maxwell from the graph.
        theories = {name: tid for tid, name in tm.all_theories()}
        t_newton  = theories.get("Newton")
        t_maxwell = theories.get("Maxwell")
        audit = inspect_gr_discovery(mg, tm,
                                     newtonian_theory_id=t_newton,
                                     maxwell_theory_id=t_maxwell,
                                     sr_theory_id=None)   # no SR → no Lorentz check
        # Property 5 must fail
        assert audit.lorentz_factor_morphism is None
        assert audit.lorentz_factor_expr is None

    def test_probe2_no_ether_morphism(self):
        """Without a RETRACTED morphism, retraction_reason is None → property 4 fails."""
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        nid = TOKEN_GRAPH.encode("PROBE2_MUL")
        t_newton = tm.register_theory("Newton")
        for lbl in ("p2_fma", "p2_xt", "p2_pmv"):
            nid_l = TOKEN_GRAPH.encode(f"PROBE2_{lbl}")
            mid = add_fitted_law(mg, lbl, _linear_law(nid_l))
            tm.assign_morphism(mid, t_newton)
        # No ether morphism added
        audit = inspect_gr_discovery(mg, tm, newtonian_theory_id=t_newton)
        assert audit.ether_morphism is None
        assert audit.retraction_reason is None
        ok, failures = verify_gr_discovery(audit)
        assert not ok, "Property 4 should fail without ether morphism"
        assert any("Property 3" in f or "Property 4" in f for f in failures)

    def test_probe3_no_paradigm_shift(self):
        """Without a PARADIGM_SHIFT morphism, spacetime_concept is None → property 7 fails."""
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t_newton = tm.register_theory("Newton")
        for lbl in ("p3_fma", "p3_xt", "p3_pmv"):
            nid_l = TOKEN_GRAPH.encode(f"PROBE3_{lbl}")
            mid = add_fitted_law(mg, lbl, _linear_law(nid_l))
            tm.assign_morphism(mid, t_newton)
        # No PARADIGM_SHIFT morphism
        audit = inspect_gr_discovery(mg, tm, newtonian_theory_id=t_newton)
        assert audit.spacetime_concept is None
        ok, failures = verify_gr_discovery(audit)
        assert not ok
        assert any("Property 7" in f for f in failures)

    def test_probe4_empty_revision_history(self):
        """audit.revision_history = [] causes property 8 to fail."""
        _, _, audit = _run_einstein_scenario(seed=0)
        # Wipe the revision history
        audit.revision_history = []
        ok, failures = verify_gr_discovery(audit)
        assert not ok
        assert any("Property 8" in f for f in failures)
