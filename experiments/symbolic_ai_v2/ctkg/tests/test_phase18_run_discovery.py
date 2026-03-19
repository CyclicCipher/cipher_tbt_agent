"""
Phase 18 Integration Test — Full Discovery Pipeline.

Tests run_discovery and seed_physics_priors from einstein/run_discovery.py:
  - seed_physics_priors populates graph with Newton and EM theories.
  - run_discovery processes anomaly streams and proposes paradigm shifts.
  - Structural invariants hold: Newton theory unchanged, SR theory created.
"""
from __future__ import annotations

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.prim_ops import make_prim_ctx, get_prim_specs
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager
from experiments.symbolic_ai_v2.ctkg.einstein.run_discovery import (
    DiscoveryResult,
    seed_physics_priors,
    run_discovery,
)


def _make_components():
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    ctx = make_prim_ctx()
    return mg, tm, ctx


# ---------------------------------------------------------------------------
# TestSeedPhysicsPriors
# ---------------------------------------------------------------------------

class TestSeedPhysicsPriors:

    def test_creates_newton_and_maxwell_theories(self):
        """seed_physics_priors creates Newton and Maxwell theory compartments."""
        mg, tm, ctx = _make_components()
        seed_physics_priors(mg, tm, ctx, seed=0)

        theory_names = {name for _, name in tm.all_theories()}
        assert any("newton" in n.lower() or "Newton" in n for n in theory_names), (
            f"Expected Newton theory, got: {theory_names}"
        )
        assert any("maxwell" in n.lower() or "Maxwell" in n for n in theory_names), (
            f"Expected Maxwell theory, got: {theory_names}"
        )

    def test_newton_theory_has_morphisms(self):
        """Newton theory has at least 1 FITTED_LAW morphism."""
        mg, tm, ctx = _make_components()
        seed_physics_priors(mg, tm, ctx, seed=0)

        theories = {name: tid for tid, name in tm.all_theories()}
        newton_tid = next((tid for name, tid in {n: t for t, n in tm.all_theories()}.items()
                           if "newton" in name.lower() or "Newton" in name), None)
        assert newton_tid is not None
        members = mg.theory_members(newton_tid)
        fitted = [mid for mid in members
                  if mg.morphism_by_id(mid) is not None
                  and mg.morphism_by_id(mid).morph_type == "FITTED_LAW"]
        assert len(fitted) >= 1, f"Expected ≥1 FITTED_LAW in Newton, got {len(fitted)}"

    def test_seed_reproducible(self):
        """Same seed produces same number of theories."""
        mg1, tm1, ctx1 = _make_components()
        seed_physics_priors(mg1, tm1, ctx1, seed=42)
        count1 = len(tm1.all_theories())

        mg2, tm2, ctx2 = _make_components()
        seed_physics_priors(mg2, tm2, ctx2, seed=42)
        count2 = len(tm2.all_theories())

        assert count1 == count2


# ---------------------------------------------------------------------------
# TestRunDiscovery
# ---------------------------------------------------------------------------

class TestRunDiscovery:

    def test_returns_discovery_result(self):
        """run_discovery returns a DiscoveryResult."""
        mg, tm, ctx = _make_components()
        specs = get_prim_specs()
        seed_physics_priors(mg, tm, ctx, seed=0)

        result = run_discovery(mg, tm, ctx, prim_specs=specs, seed=0)
        assert isinstance(result, DiscoveryResult)

    def test_lorentz_law_discovered(self):
        """run_discovery discovers a Lorentz factor law with low residual."""
        mg, tm, ctx = _make_components()
        specs = get_prim_specs()
        seed_physics_priors(mg, tm, ctx, seed=0)

        result = run_discovery(mg, tm, ctx, prim_specs=specs, seed=0)
        assert result.lorentz_law is not None, "Expected Lorentz law to be discovered"
        assert result.lorentz_law.residual < 0.05, (
            f"Lorentz residual {result.lorentz_law.residual:.4f} too high"
        )

    def test_sr_theory_created(self):
        """run_discovery creates an SR theory with the Lorentz law assigned."""
        mg, tm, ctx = _make_components()
        specs = get_prim_specs()
        seed_physics_priors(mg, tm, ctx, seed=0)

        result = run_discovery(mg, tm, ctx, prim_specs=specs, seed=0)
        assert result.sr_theory_id is not None, "Expected SR theory to be created"

        # SR theory should have at least one FITTED_LAW morphism
        members = mg.theory_members(result.sr_theory_id)
        fitted = [mid for mid in members
                  if mg.morphism_by_id(mid) is not None
                  and mg.morphism_by_id(mid).morph_type == "FITTED_LAW"]
        assert len(fitted) >= 1, (
            f"Expected ≥1 FITTED_LAW in SR theory, got {len(fitted)}"
        )

    def test_newton_theory_unchanged(self):
        """Newton theory morphism count unchanged after run_discovery."""
        mg, tm, ctx = _make_components()
        specs = get_prim_specs()
        seed_physics_priors(mg, tm, ctx, seed=0)

        # Capture Newton theory membership before
        theories_before = {name: tid for tid, name in tm.all_theories()}
        newton_tid = next((tid for tid, name in tm.all_theories()
                           if "newton" in name.lower() or "Newton" in name), None)
        assert newton_tid is not None
        members_before = set(mg.theory_members(newton_tid))

        run_discovery(mg, tm, ctx, prim_specs=specs, seed=0)

        members_after = set(mg.theory_members(newton_tid))
        assert members_before == members_after, (
            "Newton theory morphisms changed — paradigm shift contaminated old theory"
        )

    def test_paradigm_shift_morphism_exists(self):
        """run_discovery adds a PARADIGM_SHIFT morphism to the graph."""
        mg, tm, ctx = _make_components()
        specs = get_prim_specs()
        seed_physics_priors(mg, tm, ctx, seed=0)

        result = run_discovery(mg, tm, ctx, prim_specs=specs, seed=0)
        assert result.paradigm_shift_mid is not None, (
            "Expected a PARADIGM_SHIFT morphism to be created"
        )
        ps_morph = mg.morphism_by_id(result.paradigm_shift_mid)
        assert ps_morph is not None
        assert ps_morph.morph_type == "PARADIGM_SHIFT"


# ---------------------------------------------------------------------------
# TestPhase18Cage
# ---------------------------------------------------------------------------

class TestPhase18Cage:

    def test_cage_3_seeds(self):
        """run_discovery succeeds for seeds 0, 1, 2 — Lorentz law found each time."""
        specs = get_prim_specs()
        for seed in range(3):
            mg, tm, ctx = _make_components()
            seed_physics_priors(mg, tm, ctx, seed=seed)
            result = run_discovery(mg, tm, ctx, prim_specs=specs, seed=seed)
            assert result.lorentz_law is not None, (
                f"seed={seed}: Expected Lorentz law to be discovered"
            )
            assert result.lorentz_law.residual < 0.05, (
                f"seed={seed}: Lorentz residual {result.lorentz_law.residual:.4f} too high"
            )


# ---------------------------------------------------------------------------
# TestPhase18DefectProbes
# ---------------------------------------------------------------------------

class TestPhase18DefectProbes:

    def test_probe1_no_priors_still_returns_result(self):
        """run_discovery works even without seed_physics_priors (empty graph)."""
        mg, tm, ctx = _make_components()
        specs = get_prim_specs()
        result = run_discovery(mg, tm, ctx, prim_specs=specs, seed=0)
        assert isinstance(result, DiscoveryResult)

    def test_probe2_discovery_result_fields_populated(self):
        """DiscoveryResult has all expected fields."""
        mg, tm, ctx = _make_components()
        specs = get_prim_specs()
        seed_physics_priors(mg, tm, ctx, seed=0)
        result = run_discovery(mg, tm, ctx, prim_specs=specs, seed=0)

        assert hasattr(result, "lorentz_law")
        assert hasattr(result, "sr_theory_id")
        assert hasattr(result, "newton_theory_id")
        assert hasattr(result, "paradigm_shift_mid")
        assert hasattr(result, "wired_paradigm_shift")
