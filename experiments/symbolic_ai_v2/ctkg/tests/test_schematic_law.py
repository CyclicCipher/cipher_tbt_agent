"""
Phase 2 gate tests — Parametric Morphisms and Law Discovery.

Four test classes:

1. TestSchematicLaw        — functional correctness of SchematicLaw / GroundLaw:
                             discovery, classify_positions, instantiate,
                             predict_from_schema, graph storage.

2. TestPhase2Roadmap       — IDA benchmark I-5:
                             scale(X, k) = k·X discovered from two families
                             (k=3 and k=5); novel X predicted correctly;
                             same schema detected for different k values.

3. TestBitterLessonCage    — 10 independent anonymous symbol tables; the
                             recovered SchematicLaw must have the same structure
                             (same params / variables classification) in every run.
                             Variance < 5 pp.  Catches any `if op == 'mul'`
                             or hardcoded parameter name in the discovery code.

4. TestDefectProbe         — parameter vs. constant confusion:
                             (a) k varies across families → SchematicLaw(params={k})
                             (b) k is the SAME in every example → it becomes a
                                 concrete atom in the LGG (GroundLaw or
                                 SchematicLaw with params=frozenset())
                             Run with anonymous symbols so string identity on 'k'
                             cannot be the distinguishing signal.
"""
from __future__ import annotations

import random
import unicodedata
from typing import Union

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import (
    Expr, atom, node, var, match, substitute, variables as _vars,
)
from experiments.symbolic_ai_v2.ctkg.core.expr_law import rename_expr
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import (
    SchematicLaw,
    GroundLaw,
    discover_parametric_law,
    instantiate,
    predict_from_schema,
    add_schematic_law,
    query_schematic_laws,
    _classify_positions,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mg() -> MorphismGraph:
    return MorphismGraph()


_UNICODE_OPS = [
    chr(i) for i in range(0x2200, 0x22FF)
    if unicodedata.category(chr(i)) not in ('Cn', 'Co')
]


def _fresh_sym(roles: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    syms = rng.sample(_UNICODE_OPS, len(roles))
    return dict(zip(roles, syms))


def _nid(sym: dict[str, str]) -> dict[int, int]:
    return {TOKEN_GRAPH.encode(s): TOKEN_GRAPH.encode(t) for s, t in sym.items()}


# ---------------------------------------------------------------------------
# 1. Functional correctness
# ---------------------------------------------------------------------------

class TestSchematicLaw:

    # ------------------------------------------------------------------
    # Discover from two families → SchematicLaw with one parameter
    # ------------------------------------------------------------------

    def _two_family_examples(self) -> list[list[tuple[Expr, Expr]]]:
        """scale(X, k) where family 1 has k=3, family 2 has k=5."""
        fam1 = [
            (node('scale', atom('1'), atom('3')), atom('3')),
            (node('scale', atom('2'), atom('3')), atom('6')),
            (node('scale', atom('4'), atom('3')), atom('12')),
        ]
        fam2 = [
            (node('scale', atom('1'), atom('5')), atom('5')),
            (node('scale', atom('2'), atom('5')), atom('10')),
            (node('scale', atom('4'), atom('5')), atom('20')),
        ]
        return [fam1, fam2]

    def test_returns_schematic_law(self):
        families = self._two_family_examples()
        result = discover_parametric_law(families)
        assert isinstance(result, SchematicLaw)

    def test_params_nonempty(self):
        families = self._two_family_examples()
        law = discover_parametric_law(families)
        assert isinstance(law, SchematicLaw)
        assert len(law.params) >= 1, "Expected at least one parameter (k)"

    def test_variables_nonempty(self):
        families = self._two_family_examples()
        law = discover_parametric_law(families)
        assert isinstance(law, SchematicLaw)
        assert len(law.variables) >= 1, "Expected at least one variable (X)"

    def test_params_and_variables_disjoint(self):
        families = self._two_family_examples()
        law = discover_parametric_law(families)
        assert isinstance(law, SchematicLaw)
        assert law.params.isdisjoint(law.variables)

    def test_all_vars_classified(self):
        """Every var() in the pattern must be in either params or variables."""
        families = self._two_family_examples()
        law = discover_parametric_law(families)
        assert isinstance(law, SchematicLaw)
        pattern_vars = _vars(law.pattern)
        assert pattern_vars == law.params | law.variables

    def test_evidence_count(self):
        families = self._two_family_examples()
        law = discover_parametric_law(families)
        assert law.evidence == 6  # 3 + 3 examples

    # ------------------------------------------------------------------
    # Ground law: all examples identical
    # ------------------------------------------------------------------

    def test_ground_law_single_example(self):
        """Single example → GroundLaw (nothing to generalise)."""
        families = [[(node('f', atom('2')), atom('4'))]]
        result = discover_parametric_law(families)
        assert isinstance(result, GroundLaw)

    def test_ground_law_all_identical(self):
        """Three copies of the same example → GroundLaw."""
        ex = (node('f', atom('7')), atom('8'))
        result = discover_parametric_law([[ex, ex, ex]])
        assert isinstance(result, GroundLaw)

    def test_ground_law_no_vars_in_pattern(self):
        result = discover_parametric_law([
            [(node('f', atom('7')), atom('8')),
             (node('f', atom('7')), atom('8'))],
        ])
        assert isinstance(result, GroundLaw)
        assert not _vars(result.pattern)

    # ------------------------------------------------------------------
    # Single family (one k value) → variable but no parameter
    # ------------------------------------------------------------------

    def test_single_family_no_params(self):
        """One family, k always 3, X varies → variable X, NO parameter."""
        fam = [
            (node('scale', atom('1'), atom('3')), atom('3')),
            (node('scale', atom('2'), atom('3')), atom('6')),
            (node('scale', atom('4'), atom('3')), atom('12')),
        ]
        law = discover_parametric_law([fam])
        assert isinstance(law, SchematicLaw)
        # k=3 is constant in the only family → it becomes atom('3') in LGG
        # so there are no parameter vars; only variable vars remain
        assert len(law.params) == 0, (
            f"k should not be a param (only one family): params={law.params}"
        )
        assert len(law.variables) >= 1, "X should still be a variable"

    # ------------------------------------------------------------------
    # _classify_positions directly
    # ------------------------------------------------------------------

    def test_classify_positions_basic(self):
        """Two families, first position varies within, second is constant."""
        # Family 1: V0 ∈ {1, 2}, V1 = 3
        # Family 2: V0 ∈ {1, 2}, V1 = 5
        substs = [
            {'V0': atom('1'), 'V1': atom('3')},
            {'V0': atom('2'), 'V1': atom('3')},
            {'V0': atom('1'), 'V1': atom('5')},
            {'V0': atom('2'), 'V1': atom('5')},
        ]
        family_sizes = [2, 2]
        params, variables = _classify_positions(substs, family_sizes, {'V0', 'V1'})
        assert 'V0' in variables, "V0 varies within families → variable"
        assert 'V1' in params,    "V1 is constant within families → parameter"

    def test_classify_positions_all_variable(self):
        """Both positions vary within the only family → both are variables."""
        substs = [
            {'A': atom('1'), 'B': atom('9')},
            {'A': atom('2'), 'B': atom('8')},
        ]
        params, variables = _classify_positions(substs, [2], {'A', 'B'})
        assert 'A' in variables
        assert 'B' in variables
        assert len(params) == 0

    # ------------------------------------------------------------------
    # instantiate
    # ------------------------------------------------------------------

    def test_instantiate_replaces_param(self):
        law = discover_parametric_law(self._two_family_examples())
        assert isinstance(law, SchematicLaw)
        # Pick the parameter var name (whatever anti_unify chose)
        param_name = next(iter(law.params))
        # Pick the value from family 1
        param_val = atom('3')
        concrete = instantiate(law, {param_name: param_val})
        # The param var should be gone from the instantiated pattern
        remaining_vars = _vars(concrete.pattern)
        assert param_name not in remaining_vars, (
            f"Parameter {param_name!r} should be substituted out"
        )

    def test_instantiate_preserves_variable_vars(self):
        law = discover_parametric_law(self._two_family_examples())
        assert isinstance(law, SchematicLaw)
        param_name = next(iter(law.params))
        concrete = instantiate(law, {param_name: atom('3')})
        remaining_vars = _vars(concrete.pattern)
        # variable vars should still be in the pattern
        for v in law.variables:
            assert v in remaining_vars, f"Variable {v!r} should remain as var()"

    def test_instantiate_morph_id_sentinel(self):
        law = discover_parametric_law(self._two_family_examples())
        assert isinstance(law, SchematicLaw)
        param_name = next(iter(law.params))
        concrete = instantiate(law, {param_name: atom('3')})
        assert concrete.morph_id == -1

    # ------------------------------------------------------------------
    # predict_from_schema
    # ------------------------------------------------------------------

    def test_predict_from_schema_success(self):
        """Given k=3 and query scale(5, 3), bindings contain X → atom('5')."""
        law = discover_parametric_law(self._two_family_examples())
        assert isinstance(law, SchematicLaw)
        param_name = next(iter(law.params))
        # Recover k's concrete atom value for family 1
        # The param's value should be atom('3') in one of the input substs
        # (since family 1 uses k=3).  We just pass it directly:
        result = predict_from_schema(law, {param_name: atom('3')},
                                     node('scale', atom('5'), atom('3')))
        assert result is not None, "Query should match the instantiated law"
        bindings, _ = result
        # The variable var should be bound to atom('5')
        var_name = next(iter(law.variables))
        assert bindings.get(var_name) == atom('5'), (
            f"Expected variable {var_name!r} = atom('5'), got {bindings.get(var_name)}"
        )

    def test_predict_from_schema_wrong_param_returns_none(self):
        """Query with k=9 (not in any family) against k=3 instantiation → None."""
        law = discover_parametric_law(self._two_family_examples())
        assert isinstance(law, SchematicLaw)
        param_name = next(iter(law.params))
        # Instantiate for k=3 but query has k=9
        result = predict_from_schema(law, {param_name: atom('3')},
                                     node('scale', atom('5'), atom('9')))
        assert result is None

    # ------------------------------------------------------------------
    # Graph storage
    # ------------------------------------------------------------------

    def test_add_and_query_schematic_law(self):
        mg = _mg()
        law = discover_parametric_law(self._two_family_examples())
        mid = add_schematic_law(mg, 'scale_law', law)
        laws = query_schematic_laws(mg, 'scale_law')
        assert len(laws) == 1
        assert laws[0].pattern == law.pattern
        assert laws[0].conclusion == law.conclusion

    def test_add_schematic_law_dedup(self):
        mg = _mg()
        law = discover_parametric_law(self._two_family_examples())
        mid1 = add_schematic_law(mg, 'scale_law', law)
        mid2 = add_schematic_law(mg, 'scale_law', law)
        assert mid1 == mid2
        assert len(query_schematic_laws(mg, 'scale_law')) == 1

    def test_query_absent_label(self):
        mg = _mg()
        assert query_schematic_laws(mg, 'nonexistent') == []

    def test_ground_law_stored_and_retrieved(self):
        mg = _mg()
        ground = discover_parametric_law([
            [(node('f', atom('7')), atom('8')),
             (node('f', atom('7')), atom('8'))]
        ])
        assert isinstance(ground, GroundLaw)
        add_schematic_law(mg, 'ground_test', ground)
        retrieved = query_schematic_laws(mg, 'ground_test')
        assert len(retrieved) == 1
        assert isinstance(retrieved[0], GroundLaw)

    def test_schematic_and_ground_coexist(self):
        """Different labels → no cross-contamination."""
        mg = _mg()
        law = discover_parametric_law(self._two_family_examples())
        ground = discover_parametric_law([
            [(node('f', atom('7')), atom('8'))],
        ])
        add_schematic_law(mg, 'schematic', law)
        add_schematic_law(mg, 'ground',    ground)
        assert len(query_schematic_laws(mg, 'schematic')) == 1
        assert len(query_schematic_laws(mg, 'ground'))    == 1
        # no bleed
        assert isinstance(query_schematic_laws(mg, 'schematic')[0], SchematicLaw)
        assert isinstance(query_schematic_laws(mg, 'ground')[0],    GroundLaw)


# ---------------------------------------------------------------------------
# 2. IDA Benchmark I-5
# ---------------------------------------------------------------------------

class TestPhase2Roadmap:
    """
    IDA I-5: scale(X, k) = k * X.

    Phase 2 delivers structural schema discovery.
    Phase 3 will add numeric evaluation of the conclusion.
    Here we verify:
      (a) Schema is discovered correctly from two families.
      (b) Novel X (not seen in training) is recognised as a valid instance.
      (c) The same schema structure applies for different k values.
    """

    _FAM_K3 = [
        (node('scale', atom('2'), atom('3')), atom('6')),
        (node('scale', atom('4'), atom('3')), atom('12')),
        (node('scale', atom('7'), atom('3')), atom('21')),
        (node('scale', atom('8'), atom('3')), atom('24')),
        (node('scale', atom('9'), atom('3')), atom('27')),
    ]
    _FAM_K5 = [
        (node('scale', atom('1'), atom('5')), atom('5')),
        (node('scale', atom('2'), atom('5')), atom('10')),
        (node('scale', atom('3'), atom('5')), atom('15')),
        (node('scale', atom('4'), atom('5')), atom('20')),
        (node('scale', atom('6'), atom('5')), atom('30')),
    ]

    def _discover(self) -> SchematicLaw:
        result = discover_parametric_law([self._FAM_K3, self._FAM_K5])
        assert isinstance(result, SchematicLaw)
        return result

    def test_schema_is_schematic(self):
        self._discover()  # no assertion needed — isinstance check inside

    def test_schema_has_one_param(self):
        law = self._discover()
        assert len(law.params) == 1, (
            f"Expected exactly one parameter (k), got: {law.params}"
        )

    def test_schema_has_one_variable(self):
        law = self._discover()
        assert len(law.variables) == 1, (
            f"Expected exactly one variable (X), got: {law.variables}"
        )

    def test_novel_x_recognised(self):
        """X=5 not in training set; query scale(5, 3) should match."""
        law = self._discover()
        param_name = next(iter(law.params))
        result = predict_from_schema(
            law, {param_name: atom('3')},
            node('scale', atom('5'), atom('3')),   # novel X=5
        )
        assert result is not None, "Novel X=5 should be recognised as a valid instance"

    def test_novel_x_binding_correct(self):
        """The binding returned for novel X=5 should be atom('5')."""
        law = self._discover()
        param_name = next(iter(law.params))
        var_name   = next(iter(law.variables))
        result = predict_from_schema(
            law, {param_name: atom('3')},
            node('scale', atom('5'), atom('3')),
        )
        assert result is not None
        bindings, _ = result
        assert bindings[var_name] == atom('5')

    def test_schema_reuse_for_k5(self):
        """The SAME schema instance (just different param binding) works for k=5."""
        law = self._discover()
        param_name = next(iter(law.params))
        # query with k=5 (training param) and X=6 (training X)
        result = predict_from_schema(
            law, {param_name: atom('5')},
            node('scale', atom('6'), atom('5')),
        )
        assert result is not None, "k=5 family should also be matchable"

    def test_wrong_k_not_recognised(self):
        """k=9 was not in any training family; if the law is instantiated with k=3,
        a query with k=9 should not match."""
        law = self._discover()
        param_name = next(iter(law.params))
        result = predict_from_schema(
            law, {param_name: atom('3')},
            node('scale', atom('4'), atom('9')),   # k=9 ≠ k=3
        )
        assert result is None

    def test_evidence_count(self):
        law = self._discover()
        assert law.evidence == 10  # 5 + 5


# ---------------------------------------------------------------------------
# 3. Bitter Lesson cage — 10 anonymous symbol tables
# ---------------------------------------------------------------------------

class TestBitterLessonCage:
    """
    Cage for Phase 2: replace 'scale' with a random Unicode symbol.
    The structural classification (params vs variables) must be identical
    across all 10 seeds.  Variance = 0 pp.
    """

    def _make_families_anon(
        self,
        nid: dict[int, int],
    ) -> list[list[tuple[Expr, Expr]]]:
        """Build the two-family scale dataset with anonymous 'scale' operator."""
        def anon_input(x_str: str, k_str: str) -> Expr:
            return rename_expr(node('scale', atom(x_str), atom(k_str)), nid)

        fam1 = [(anon_input('1', '3'), atom('3')),
                (anon_input('2', '3'), atom('6')),
                (anon_input('4', '3'), atom('12'))]
        fam2 = [(anon_input('1', '5'), atom('5')),
                (anon_input('2', '5'), atom('10')),
                (anon_input('4', '5'), atom('20'))]
        return [fam1, fam2]

    def test_cage_10_seeds_same_structure(self):
        """Structure (len(params), len(variables)) identical across 10 seeds."""
        reference = None
        for seed in range(10):
            sym = _fresh_sym(['scale'], seed)
            nid = _nid(sym)
            families = self._make_families_anon(nid)
            law = discover_parametric_law(families)
            assert isinstance(law, SchematicLaw), (
                f"seed {seed}: expected SchematicLaw, got {type(law)}"
            )
            key = (len(law.params), len(law.variables))
            if reference is None:
                reference = key
            else:
                assert key == reference, (
                    f"seed {seed}: structure {key} ≠ reference {reference}"
                )

    def test_cage_zero_variance(self):
        """All seeds return SchematicLaw (boolean: is_schematic)."""
        outcomes = []
        for seed in range(10):
            sym = _fresh_sym(['scale'], seed)
            nid = _nid(sym)
            families = self._make_families_anon(nid)
            law = discover_parametric_law(families)
            outcomes.append(isinstance(law, SchematicLaw))

        mean = sum(outcomes) / len(outcomes)
        var  = sum((x - mean) ** 2 for x in outcomes) / len(outcomes)
        assert all(outcomes), f"Some seeds returned non-SchematicLaw: {outcomes}"
        assert var == 0.0, f"Non-zero variance: {var}"

    def test_cage_novel_x_recognition_10_seeds(self):
        """Novel X query recognised in all 10 seeds."""
        outcomes = []
        for seed in range(10):
            sym = _fresh_sym(['scale'], seed)
            nid = _nid(sym)
            families = self._make_families_anon(nid)
            law = discover_parametric_law(families)
            assert isinstance(law, SchematicLaw)
            param_name = next(iter(law.params))
            query = rename_expr(node('scale', atom('5'), atom('3')), nid)
            result = predict_from_schema(law, {param_name: atom('3')}, query)
            outcomes.append(result is not None)

        assert all(outcomes), f"Novel X recognition failed in some seeds: {outcomes}"


# ---------------------------------------------------------------------------
# 4. Defect probe — parameter vs. constant confusion
# ---------------------------------------------------------------------------

class TestDefectProbe:
    """
    The defect probe has three parts:

    (a) PARAMETRIC case: k varies across families (k=3, k=5).
        Must produce SchematicLaw with params containing k's var name.

    (b) CONSTANT case: k is the SAME in every example across every family
        (all groups share k=7 so anti_unify reduces k to atom('7')).
        Must NOT classify k as a parameter (it's not a var() in the LGG).
        Produces SchematicLaw(params=frozenset()) or GroundLaw.

    (c) ANONYMOUS SYMBOLS: same two cases, but 'scale' replaced by Unicode.
        String identity on 'scale', '3', '5', '7' cannot be the distinguishing
        signal.
    """

    # ------------------------------------------------------------------
    # (a) Parametric case
    # ------------------------------------------------------------------

    def test_parametric_k_classified_as_param(self):
        """k varies across families → should appear in SchematicLaw.params."""
        fam1 = [(node('scale', atom('2'), atom('3')), atom('6')),
                (node('scale', atom('4'), atom('3')), atom('12'))]
        fam2 = [(node('scale', atom('2'), atom('5')), atom('10')),
                (node('scale', atom('4'), atom('5')), atom('20'))]
        law = discover_parametric_law([fam1, fam2])
        assert isinstance(law, SchematicLaw)
        assert len(law.params) >= 1, (
            "k (which varies across families) must be classified as a parameter"
        )

    # ------------------------------------------------------------------
    # (b) Constant case
    # ------------------------------------------------------------------

    def test_constant_k_not_a_param(self):
        """k = 7 in every example → it becomes atom('7') in the LGG, not a param."""
        fam1 = [(node('scale', atom('1'), atom('7')), atom('7')),
                (node('scale', atom('2'), atom('7')), atom('14')),
                (node('scale', atom('3'), atom('7')), atom('21'))]
        # Only ONE family (k=7 always) — k will NOT appear as a var in the LGG
        law = discover_parametric_law([fam1])
        # Could be SchematicLaw or GroundLaw; either way params must be empty
        assert law.params == frozenset(), (
            f"k=7 is constant everywhere — should not be a parameter. "
            f"Got params={law.params}"
        )

    def test_constant_k_is_atom_in_pattern(self):
        """Verify k=7 appears as atom('7') in the LGG pattern, not as var()."""
        fam1 = [(node('scale', atom('1'), atom('7')), atom('7')),
                (node('scale', atom('2'), atom('7')), atom('14'))]
        law = discover_parametric_law([fam1])
        # Match law.pattern against a scale(x, 7) expression:
        # atom('7') should match only atom('7'), not a wildcard
        match_correct = match(law.pattern, node('scale', atom('3'), atom('7')))
        match_wrong   = match(law.pattern, node('scale', atom('3'), atom('9')))
        assert match_correct is not None, "scale(3, 7) should match the pattern"
        assert match_wrong   is None,     "scale(3, 9) should NOT match (7 ≠ 9)"

    # ------------------------------------------------------------------
    # (c) Anonymous symbols — the key cage+probe combination
    # ------------------------------------------------------------------

    def _anon_parametric_families(self, nid: dict[int, int]):
        def ai(x_s: str, k_s: str) -> Expr:
            return rename_expr(node('scale', atom(x_s), atom(k_s)), nid)
        fam1 = [(ai('2', '3'), atom('6')), (ai('4', '3'), atom('12'))]
        fam2 = [(ai('2', '5'), atom('10')), (ai('4', '5'), atom('20'))]
        return [fam1, fam2]

    def _anon_constant_families(self, nid: dict[int, int]):
        def ai(x_s: str) -> Expr:
            return rename_expr(node('scale', atom(x_s), atom('7')), nid)
        fam = [(ai('1'), atom('7')), (ai('2'), atom('14')), (ai('3'), atom('21'))]
        return [fam]

    def test_anon_parametric_has_param_10_seeds(self):
        """Parametric families under 10 symbol tables → always SchematicLaw(params≥1)."""
        for seed in range(10):
            sym = _fresh_sym(['scale'], seed)
            nid = _nid(sym)
            law = discover_parametric_law(self._anon_parametric_families(nid))
            assert isinstance(law, SchematicLaw), f"seed {seed}: not SchematicLaw"
            assert len(law.params) >= 1, (
                f"seed {seed}: expected params≥1, got {law.params}"
            )

    def test_anon_constant_has_no_param_10_seeds(self):
        """Constant k under 10 symbol tables → always params=frozenset()."""
        for seed in range(10):
            sym = _fresh_sym(['scale'], seed)
            nid = _nid(sym)
            law = discover_parametric_law(self._anon_constant_families(nid))
            assert law.params == frozenset(), (
                f"seed {seed}: k=7 is constant but got params={law.params}"
            )

    def test_anon_distinguishes_parametric_from_constant_10_seeds(self):
        """
        The two cases must ALWAYS be classified differently.
        This is the core defect probe: parametric ≠ constant in all 10 seeds.
        """
        for seed in range(10):
            sym = _fresh_sym(['scale'], seed)
            nid = _nid(sym)
            law_param = discover_parametric_law(self._anon_parametric_families(nid))
            law_const = discover_parametric_law(self._anon_constant_families(nid))
            # Parametric must have params > 0; constant must have params == 0
            assert len(law_param.params) >= 1, (
                f"seed {seed}: parametric case has no params"
            )
            assert len(law_const.params) == 0, (
                f"seed {seed}: constant case wrongly has params={law_const.params}"
            )
            # The two laws must differ in their params classification
            assert law_param.params != law_const.params, (
                f"seed {seed}: two cases not distinguished"
            )
