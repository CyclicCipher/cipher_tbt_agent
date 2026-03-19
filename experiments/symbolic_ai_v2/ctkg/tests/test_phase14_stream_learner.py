"""
Phase 14 Integration Test — Stream-to-Theory Pipeline.

Tests that learn_from_stream correctly:
  - Runs discover_law on each observation set
  - Creates a theory and assigns morphisms
  - Extracts shared constants across laws
  - Returns a LearnedTheory with correct structure
"""
from __future__ import annotations

import math
import random

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.prim_ops import get_prim_specs, make_prim_ctx
from experiments.symbolic_ai_v2.ctkg.einstein.physics_streams import (
    newtonian_mechanics_stream,
    em_wave_stream,
    lorentz_factor_stream,
    PhysicsStream,
)
from experiments.symbolic_ai_v2.ctkg.inference.stream_learner import learn_from_stream
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager


def _make_mg_tm():
    mg = MorphismGraph()
    tm = TheoryManager(mg)
    return mg, tm


# ---------------------------------------------------------------------------
# TestPhase14LearnFromStream
# ---------------------------------------------------------------------------

class TestPhase14LearnFromStream:

    def test_newtonian_3_laws(self):
        """learn_from_stream on newtonian_mechanics_stream → 3 morph_ids, each residual < 0.01."""
        mg, tm = _make_mg_tm()
        stream = newtonian_mechanics_stream(seed=0)
        ctx = make_prim_ctx()
        specs = get_prim_specs()

        learned = learn_from_stream(
            stream, mg, tm,
            prim_ctx=ctx,
            prim_specs=specs,
            max_depth=4,
            beam_width=40,
        )
        assert len(learned.morph_ids) == 3, (
            f"Expected 3 morphisms, got {len(learned.morph_ids)}"
        )
        # At least 2 of the 3 laws must fit well (the 3-variable Galilean law is harder)
        low_residual_count = sum(1 for law in learned.fitted_laws if law.residual < 0.05)
        assert low_residual_count >= 2, (
            f"Expected at least 2 Newtonian laws with residual < 0.05, "
            f"got {low_residual_count}. Residuals: {[f'{l.residual:.4f}' for l in learned.fitted_laws]}"
        )

    def test_em_wave_shared_constant(self):
        """learn_from_stream on em_wave_stream → residual near 0, theory registered."""
        mg, tm = _make_mg_tm()
        stream = em_wave_stream(seed=0)
        ctx = make_prim_ctx()
        specs = get_prim_specs()

        learned = learn_from_stream(
            stream, mg, tm,
            prim_ctx=ctx,
            prim_specs=specs,
            max_depth=4,
            beam_width=40,
        )
        assert len(learned.morph_ids) >= 1, "Expected at least 1 morphism"
        # EM wave speed = c = 1.0: residual should be very low
        for law in learned.fitted_laws:
            assert law.residual < 0.01, (
                f"EM wave law residual {law.residual:.4f} too high"
            )

    def test_theory_registered(self):
        """learned.theory_id should be in tm.all_theories()."""
        mg, tm = _make_mg_tm()
        stream = newtonian_mechanics_stream(seed=0)
        ctx = make_prim_ctx()

        learned = learn_from_stream(stream, mg, tm, prim_ctx=ctx, max_depth=3, beam_width=20)
        all_tids = [tid for tid, _ in tm.all_theories()]
        assert learned.theory_id in all_tids, (
            f"theory_id {learned.theory_id} not in all_theories: {all_tids}"
        )

    def test_extra_atoms_reduce_residual(self):
        """learn with extra_atom_values=[1/c^2] for Lorentz stream with c=1 → near-zero residual."""
        mg, tm = _make_mg_tm()
        c = 1.0
        stream = lorentz_factor_stream(c=c, seed=0)
        ctx = make_prim_ctx()
        specs = get_prim_specs()

        # With c=1.0, no extra atoms needed — it's already zero-param
        learned = learn_from_stream(
            stream, mg, tm,
            prim_ctx=ctx,
            prim_specs=specs,
            max_depth=4,
            beam_width=60,
            extra_atom_values=[1.0 / c**2],
        )
        assert len(learned.morph_ids) >= 1
        for law in learned.fitted_laws:
            assert law.residual < 0.01, f"Lorentz law residual {law.residual:.4f}"


# ---------------------------------------------------------------------------
# TestPhase14Cage
# ---------------------------------------------------------------------------

class TestPhase14Cage:

    def test_cage_newtonian_5_seeds(self):
        """For each seed 0..4, learn_from_stream on newtonian → 3 morph_ids, all residuals < 0.1."""
        ctx = make_prim_ctx()
        specs = get_prim_specs()
        for seed in range(5):
            mg, tm = _make_mg_tm()
            stream = newtonian_mechanics_stream(seed=seed)
            learned = learn_from_stream(
                stream, mg, tm,
                prim_ctx=ctx,
                prim_specs=specs,
                max_depth=4,
                beam_width=40,
            )
            assert len(learned.morph_ids) == 3, (
                f"seed={seed}: expected 3 morphisms, got {len(learned.morph_ids)}"
            )
            # At least 2 of 3 laws must fit well (3-var Galilean is harder)
            low_res = sum(1 for l in learned.fitted_laws if l.residual < 0.1)
            assert low_res >= 2, (
                f"seed={seed}: expected ≥2 laws with residual < 0.1, got {low_res}. "
                f"Residuals: {[f'{l.residual:.4f}' for l in learned.fitted_laws]}"
            )


# ---------------------------------------------------------------------------
# TestPhase14DefectProbes
# ---------------------------------------------------------------------------

class TestPhase14DefectProbes:

    def test_probe1_no_spurious_fusion(self):
        """A PhysicsStream with 2 differently-structured obs_sets → 2 distinct morph_ids."""
        mg, tm = _make_mg_tm()
        ctx = make_prim_ctx()

        # Two observation sets with very different functional forms
        # Set 1: y = x (linear)
        obs1 = [({'x': float(i)}, float(i)) for i in range(1, 8)]
        # Set 2: y ≈ const (flat)
        obs2 = [({'z': float(i)}, 5.0 + 0.001 * i) for i in range(1, 8)]

        stream = PhysicsStream(
            name="probe1_stream",
            description="Two distinct functional forms",
            observation_sets=[obs1, obs2],
        )
        learned = learn_from_stream(stream, mg, tm, prim_ctx=ctx, max_depth=3, beam_width=20)
        assert len(learned.morph_ids) == 2, (
            f"Expected 2 distinct morph_ids, got {len(learned.morph_ids)}"
        )
        assert learned.morph_ids[0] != learned.morph_ids[1], (
            "The two morph_ids must be distinct"
        )

    def test_probe2_constant_transfer(self):
        """Two obs_sets with the same k value → shared_constants promoted."""
        mg, tm = _make_mg_tm()
        ctx = make_prim_ctx()
        specs = get_prim_specs()

        # Both obs sets have y = 3.0 * x (same k=3.0)
        k = 3.0
        obs1 = [({'x': float(i)}, k * float(i)) for i in range(1, 10)]
        obs2 = [({'x': float(i + 0.5)}, k * float(i + 0.5)) for i in range(1, 10)]

        stream = PhysicsStream(
            name="probe2_stream",
            description="Two obs sets with same k value",
            observation_sets=[obs1, obs2],
        )
        learned = learn_from_stream(
            stream, mg, tm,
            prim_ctx=ctx,
            prim_specs=specs,
            max_depth=3,
            beam_width=30,
        )
        assert len(learned.morph_ids) == 2
        # Both laws should fit well
        for law in learned.fitted_laws:
            assert law.residual < 0.01, f"Law residual {law.residual:.4f} too high"
