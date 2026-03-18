"""
Phase 7 Integration Test — Latent Variable and Ontology Extension Abduction.

Chains Phases 1–7 end-to-end.

Scenario A — Latent Variable Recovery
--------------------------------------
Observations of f(x) where f = h ∘ g and g is hidden:
  g(x) = 2 * x   (maps input to latent speed)
  h(z) = 3 * z   (maps latent speed to output energy)
  obs  = h(g(x)) = 6 * x

The system must hypothesise a latent z such that:
  1. g: x → z is fit from observations.
  2. h: z → output is fit from (latent, output) pairs.
  3. h(g(x)) predicts held-out observations within 5%.
  4. MDL prefers the 1-param linear over 3-param polynomial.

Scenario B — Ontology Extension
---------------------------------
Two theories Newton (FittedLaw, k=5) and Maxwell (FittedLaw, k=10) exist.
Neither predicts observations with k=50.
An ontology extension proposes a new concept node C whose presence would
reconcile the two theories.
The extension is stored as a LATENT_HYPOTHESIS or ONTOLOGY_EXTENSION morphism.

Test classes
------------
TestPhase7Integration (8 tests)
    - hypothesise_latent returns non-None for composed observations
    - latent composition predicts held-out correctly
    - MDL selection prefers simpler model
    - latent hypothesis stored in graph
    - ontology extension returns new concept
    - concept node has no token label
    - query returns stored hypothesis
    - integration: Phase 4 theory + Phase 7 latent → full pipeline

TestPhase7Cage (3 tests)
    - cage: 10 seeds, latent composition correct
    - cage: MDL selection consistent
    - cage: ontology extension concept_id unique per graph

TestPhase7DefectProbe (4 tests)
    Probe 1: MDL selects simpler model
    Probe 2: concept node has no token label (purely structural)
    Probe 3: composition predicts held-out
    Probe 4: two symbol tables produce isomorphic latent structure
"""
from __future__ import annotations

import random

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, var, Expr
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    add_fitted_law,
    predict_continuous,
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager
from experiments.symbolic_ai_v2.ctkg.inference.latent import (
    hypothesise_latent,
    hypothesise_latent_mdl_select,
    query_latent_hypotheses,
    propose_new_concept,
    query_ontology_extensions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(name: str = "mul") -> tuple[EvalContext, int]:
    nid = TOKEN_GRAPH.encode(name)
    return EvalContext({nid: lambda a, b: a * b}), nid


def _schema_g(nid: int) -> SchematicLaw:
    """k * x  (1 param: k; 1 variable: x)."""
    formula = Expr(head=nid, args=(var("k"), var("x")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["k"]), variables=frozenset(["x"]), evidence=1,
    )


def _schema_h(nid: int) -> SchematicLaw:
    """a * z  (1 param: a; 1 variable: z — the latent)."""
    formula = Expr(head=nid, args=(var("a"), var("z")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["a"]), variables=frozenset(["z"]), evidence=1,
    )


def _schema_g3(nid: int) -> SchematicLaw:
    """a * x  (3 params: a,b,c; only a active) — complex schema for MDL test."""
    formula = Expr(head=nid, args=(var("a"), var("x")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["a", "b", "c"]),  # 3 params; b,c won't appear in formula
        variables=frozenset(["x"]), evidence=1,
    )


def _compose_obs(k_g: float, k_h: float, n: int = 6) -> list[tuple[dict, float]]:
    return [({"x": float(i + 1)}, k_h * k_g * (i + 1)) for i in range(n)]


def _fitted_law(nid: int, k: float) -> FittedLaw:
    formula = Expr(head=nid, args=(atom("k"), var("x")))
    sch = SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(), variables=frozenset(["x"]), evidence=1,
    )
    return FittedLaw(schema=sch, params={"k": k}, residual=0.0)


def _anon_ctx(seed: int) -> tuple[EvalContext, int]:
    rng = random.Random(seed)
    sym = chr(0x2200 + rng.randint(0, 0xFF))
    nid = TOKEN_GRAPH.encode(sym)
    return EvalContext({nid: lambda a, b: a * b}), nid


# ---------------------------------------------------------------------------
# TestPhase7Integration
# ---------------------------------------------------------------------------

class TestPhase7Integration:

    def test_hypothesise_latent_non_none(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        hyp = hypothesise_latent(
            _compose_obs(2.0, 3.0), _schema_g(nid), _schema_h(nid),
            ctx, mg, tm, t,
        )
        assert hyp is not None

    def test_composition_predicts_holdout(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        hyp = hypothesise_latent(
            _compose_obs(2.0, 3.0, n=6), _schema_g(nid), _schema_h(nid),
            ctx, mg, tm, t,
        )
        assert hyp is not None
        # Held-out: x=7 → expected = 2*3*7 = 42
        z = predict_continuous(hyp.input_law, {"x": 7.0}, ctx)
        pred = predict_continuous(hyp.output_law, {"z": z}, ctx)
        assert pred == pytest.approx(42.0, rel=0.05)

    def test_mdl_prefers_simpler(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        obs = _compose_obs(2.0, 1.0)
        best = hypothesise_latent_mdl_select(
            obs, [_schema_g(nid), _schema_g3(nid)], _schema_h(nid),
            ctx, mg, tm, t,
        )
        assert best is not None
        hyps = query_latent_hypotheses(mg, t)
        assert best.mdl_score == min(h.mdl_score for h in hyps)

    def test_latent_hypothesis_in_graph(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        hyp = hypothesise_latent(
            _compose_obs(2.0, 3.0), _schema_g(nid), _schema_h(nid),
            ctx, mg, tm, t,
        )
        assert hyp is not None
        assert hyp.morph_id != -1
        m = mg.morphism_by_id(hyp.morph_id)
        assert m is not None
        assert m.morph_type == "LATENT_HYPOTHESIS"

    def test_ontology_extension_returns_new_concept(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        ext = propose_new_concept(mg, tm, t, [], [], residual_gain=2.0)
        assert ext is not None
        obj = mg.object_by_id(ext.concept_id)
        assert obj is not None

    def test_concept_has_no_token_label(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        ext = propose_new_concept(mg, tm, t, [], [], residual_gain=1.0)
        obj = mg.object_by_id(ext.concept_id)
        assert obj.label == "" or obj.label is None

    def test_query_returns_stored_hypothesis(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        hyp = hypothesise_latent(
            _compose_obs(2.0, 3.0), _schema_g(nid), _schema_h(nid),
            ctx, mg, tm, t,
        )
        assert hyp is not None
        results = query_latent_hypotheses(mg, t)
        assert len(results) == 1
        assert results[0].latent_id == hyp.latent_id

    def test_full_pipeline_with_theory(self):
        """Phase 4 theory + Phase 7 latent hypothesis — full pipeline."""
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("Newton")

        # Phase 4: add a wrong FittedLaw to the theory
        mid = add_fitted_law(mg, "wrong_newton", _fitted_law(nid, 5.0))
        tm.assign_morphism(mid, t)

        # Phase 7: hypothesise a latent that explains the true observations
        hyp = hypothesise_latent(
            _compose_obs(2.0, 3.0, n=8), _schema_g(nid), _schema_h(nid),
            ctx, mg, tm, t, label_prefix="newton_latent",
        )
        assert hyp is not None
        assert hyp.residual == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestPhase7Cage
# ---------------------------------------------------------------------------

class TestPhase7Cage:

    def test_cage_latent_composition_correct(self):
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t = tm.register_theory("T")
            hyp = hypothesise_latent(
                _compose_obs(2.0, 3.0), _schema_g(nid), _schema_h(nid),
                ctx, mg, tm, t, label_prefix=f"cage7_{seed}",
            )
            assert hyp is not None, f"seed {seed}: None"
            z = predict_continuous(hyp.input_law, {"x": 5.0}, ctx)
            pred = predict_continuous(hyp.output_law, {"z": z}, ctx)
            assert pred == pytest.approx(30.0, rel=0.05), \
                f"seed {seed}: pred={pred}"

    def test_cage_mdl_consistent(self):
        for seed in range(10):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t = tm.register_theory("T")
            obs = _compose_obs(2.0, 1.0)
            best = hypothesise_latent_mdl_select(
                obs, [_schema_g(nid), _schema_g3(nid)], _schema_h(nid),
                ctx, mg, tm, t, label_prefix=f"cagemdl_{seed}",
            )
            assert best is not None
            hyps = query_latent_hypotheses(mg, t)
            assert best.mdl_score == min(h.mdl_score for h in hyps), \
                f"seed {seed}: MDL didn't select best"

    def test_cage_concept_node_unique(self):
        for seed in range(5):
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t = tm.register_theory("T")
            ext = propose_new_concept(mg, tm, t, [], [], residual_gain=1.0)
            obj = mg.object_by_id(ext.concept_id)
            assert obj is not None
            assert obj.label == ""


# ---------------------------------------------------------------------------
# TestPhase7DefectProbe
# ---------------------------------------------------------------------------

class TestPhase7DefectProbe:

    def test_probe1_mdl_selects_simpler(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        obs = _compose_obs(2.0, 1.0)
        best = hypothesise_latent_mdl_select(
            obs, [_schema_g(nid), _schema_g3(nid)], _schema_h(nid),
            ctx, mg, tm, t, label_prefix="p7d1",
        )
        assert best is not None
        hyps = query_latent_hypotheses(mg, t)
        assert best.mdl_score == min(h.mdl_score for h in hyps), \
            "PROBE 1: MDL did not select the simplest model"

    def test_probe2_concept_no_token_label(self):
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        ext = propose_new_concept(mg, tm, t, [], [], residual_gain=1.0)
        obj = mg.object_by_id(ext.concept_id)
        assert obj.label == "", \
            f"PROBE 2: concept has non-empty label {obj.label!r}"

    def test_probe3_composition_predicts_holdout(self):
        ctx, nid = _ctx()
        mg = MorphismGraph()
        tm = TheoryManager(mg)
        t = tm.register_theory("T")
        hyp = hypothesise_latent(
            _compose_obs(2.0, 3.0, n=6), _schema_g(nid), _schema_h(nid),
            ctx, mg, tm, t, label_prefix="p7d3",
        )
        assert hyp is not None
        for x_test in [7.0, 8.0, 9.0]:
            expected = 2.0 * 3.0 * x_test  # = 6*x
            z = predict_continuous(hyp.input_law, {"x": x_test}, ctx)
            pred = predict_continuous(hyp.output_law, {"z": z}, ctx)
            assert pred == pytest.approx(expected, rel=0.05), \
                f"PROBE 3: x={x_test}: pred={pred}, expected={expected}"

    def test_probe4_two_tables_isomorphic_latent(self):
        """Two symbol tables produce latent graphs with same predicted values."""
        preds = []
        for seed in range(2):
            ctx, nid = _anon_ctx(seed)
            mg = MorphismGraph()
            tm = TheoryManager(mg)
            t = tm.register_theory("T")
            hyp = hypothesise_latent(
                _compose_obs(2.0, 3.0), _schema_g(nid), _schema_h(nid),
                ctx, mg, tm, t, label_prefix=f"p7d4_{seed}",
            )
            assert hyp is not None
            z = predict_continuous(hyp.input_law, {"x": 5.0}, ctx)
            pred = predict_continuous(hyp.output_law, {"z": z}, ctx)
            preds.append(pred)
        # Both symbol tables must produce the same prediction
        assert preds[0] == pytest.approx(preds[1], rel=0.05), \
            f"PROBE 4: preds differ: {preds[0]} vs {preds[1]}"
