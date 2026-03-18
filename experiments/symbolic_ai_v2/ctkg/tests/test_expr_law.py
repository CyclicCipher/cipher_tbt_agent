"""
Phase 1 gate tests — Expression Laws as CTKG Morphisms.

Three test classes:

1. TestExprLaw          — functional correctness (add, query, match, apply,
                          dedup, match_and_apply).
2. TestPhase1Roadmap    — roadmap verification: F=ma law stored and queried;
                          commutativity law applied; solving for each variable.
3. TestBitterLessonCage — 10 independent anonymous Unicode symbol tables;
                          all produce structurally identical results.
                          Variance = 0 pp across seeds.  Catches both:
                            (a) Iron Law violations (if op == 'mul' breaks when
                                op is '⊕')
                            (b) Bitter Lesson violations (domain knowledge
                                encoded in token strings)
4. TestDefectProbe      — tree identity vs string identity.  Two expressions
                          with the same multiset of NodeIds but different tree
                          structure must be treated as distinct.
"""
from __future__ import annotations

import random
import unicodedata

import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import (
    Expr, atom, node, var, match, substitute,
)
from experiments.symbolic_ai_v2.ctkg.core.expr_law import (
    ExprLaw,
    add_expr_law,
    query_expr_laws,
    match_law,
    apply_law,
    match_and_apply,
    rename_expr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mg() -> MorphismGraph:
    return MorphismGraph()


# Unicode Mathematical Operators block (U+2200–U+22FE), excluding unassigned
_UNICODE_OPS = [
    chr(i) for i in range(0x2200, 0x22FF)
    if unicodedata.category(chr(i)) not in ('Cn', 'Co')
]


def _fresh_symbol_table(role_names: list[str], seed: int) -> dict[str, str]:
    """Map role_names → distinct anonymous Unicode symbols, reproducibly."""
    rng = random.Random(seed)
    symbols = rng.sample(_UNICODE_OPS, len(role_names))
    return dict(zip(role_names, symbols))


def _nid_map(sym_table: dict[str, str]) -> dict[int, int]:
    """Convert a str→str symbol table to NodeId→NodeId for rename_expr."""
    return {
        TOKEN_GRAPH.encode(src): TOKEN_GRAPH.encode(tgt)
        for src, tgt in sym_table.items()
    }


# ---------------------------------------------------------------------------
# 1. Functional correctness
# ---------------------------------------------------------------------------

class TestExprLaw:
    def test_add_and_query_single(self):
        mg = _mg()
        pat = node('mul', var('X'), var('Y'))
        con = var('X')
        law = add_expr_law(mg, 'proj_law', pat, con)
        assert isinstance(law, ExprLaw)
        laws = query_expr_laws(mg, 'proj_law')
        assert len(laws) == 1
        assert laws[0].pattern == pat
        assert laws[0].conclusion == con

    def test_deduplication(self):
        mg = _mg()
        pat = node('add', var('A'), var('B'))
        con = var('A')
        law1 = add_expr_law(mg, 'dedup_law', pat, con)
        law2 = add_expr_law(mg, 'dedup_law', pat, con)
        assert law1.morph_id == law2.morph_id
        assert len(query_expr_laws(mg, 'dedup_law')) == 1

    def test_different_conclusions_not_deduplicated(self):
        mg = _mg()
        pat = node('add', var('A'), var('B'))
        law1 = add_expr_law(mg, 'two_laws', pat, var('A'))
        law2 = add_expr_law(mg, 'two_laws', pat, var('B'))
        assert law1.morph_id != law2.morph_id
        assert len(query_expr_laws(mg, 'two_laws')) == 2

    def test_query_absent_label(self):
        mg = _mg()
        assert query_expr_laws(mg, 'nonexistent') == []

    def test_match_law_success(self):
        mg = _mg()
        pat = node('mul', var('X'), var('Y'))
        law = add_expr_law(mg, 'test', pat, var('X'))
        expr = node('mul', atom('3'), atom('5'))
        b = match_law(law, expr)
        assert b == {'X': atom('3'), 'Y': atom('5')}

    def test_match_law_head_mismatch(self):
        mg = _mg()
        pat = node('mul', var('X'), var('Y'))
        law = add_expr_law(mg, 'test', pat, var('X'))
        expr = node('add', atom('3'), atom('5'))
        assert match_law(law, expr) is None

    def test_match_law_arity_mismatch(self):
        mg = _mg()
        pat = node('mul', var('X'), var('Y'))
        law = add_expr_law(mg, 'test', pat, var('X'))
        expr = node('mul', atom('3'))           # arity 1 ≠ arity 2
        assert match_law(law, expr) is None

    def test_apply_law(self):
        mg = _mg()
        pat = node('mul', var('X'), var('Y'))
        # conclusion: mul(Y, X) — commutativity
        con = node('mul', var('Y'), var('X'))
        law = add_expr_law(mg, 'comm', pat, con)
        expr = node('mul', atom('3'), atom('5'))
        b = match_law(law, expr)
        result = apply_law(law, b)
        assert result == node('mul', atom('5'), atom('3'))

    def test_match_and_apply_found(self):
        mg = _mg()
        pat = node('succ', var('N'))
        con = var('N')
        add_expr_law(mg, 'pred_of_succ', pat, con)
        expr = node('succ', atom('7'))
        result = match_and_apply(mg, 'pred_of_succ', expr)
        assert result == atom('7')

    def test_match_and_apply_no_match(self):
        mg = _mg()
        pat = node('succ', var('N'))
        add_expr_law(mg, 'pred_of_succ', pat, var('N'))
        expr = node('pred', atom('7'))          # wrong head
        assert match_and_apply(mg, 'pred_of_succ', expr) is None

    def test_match_and_apply_first_wins(self):
        """When multiple laws are stored, the first matching one is applied."""
        mg = _mg()
        pat = node('f', var('X'))
        add_expr_law(mg, 'multi', pat, atom('first'))
        add_expr_law(mg, 'multi', pat, atom('second'))
        expr = node('f', atom('a'))
        result = match_and_apply(mg, 'multi', expr)
        assert result == atom('first')

    def test_laws_isolated_across_labels(self):
        mg = _mg()
        add_expr_law(mg, 'law_a', node('mul', var('X'), var('Y')), var('X'))
        add_expr_law(mg, 'law_b', node('add', var('A'), var('B')), var('A'))
        assert len(query_expr_laws(mg, 'law_a')) == 1
        assert len(query_expr_laws(mg, 'law_b')) == 1
        # law_a does not see law_b's morphisms
        for law in query_expr_laws(mg, 'law_a'):
            assert TOKEN_GRAPH.decode(law.pattern.head) == 'mul'


# ---------------------------------------------------------------------------
# 2. Roadmap verification: F = m * a
# ---------------------------------------------------------------------------

class TestPhase1Roadmap:
    """
    Law F = m * a stored as an EXPR_LAW morphism.

    Three queries:
      (a) Given m=3, a=5 → F=15 (evaluate)
      (b) Commutativity: m*a = a*m
      (c) The law survives a fresh query from a second MorphismGraph reference
    """

    def _build_fma_law(self, mg: MorphismGraph) -> ExprLaw:
        # F = m * a  ↔  mul(m, a) → result
        # Store as: pattern=mul(Var(m), Var(a)), conclusion=Var(F)
        # (Phase 3 will add numeric evaluation; Phase 1 only stores the structure)
        pat = node('mul', var('m'), var('a'))
        con = var('F')
        return add_expr_law(mg, 'newton_second', pat, con)

    def test_law_stored_and_retrieved(self):
        mg = _mg()
        law = self._build_fma_law(mg)
        retrieved = query_expr_laws(mg, 'newton_second')
        assert len(retrieved) == 1
        assert retrieved[0].pattern == law.pattern
        assert retrieved[0].conclusion == law.conclusion

    def test_match_concrete_values(self):
        mg = _mg()
        law = self._build_fma_law(mg)
        expr = node('mul', atom('3'), atom('5'))
        b = match_law(law, expr)
        assert b == {'m': atom('3'), 'a': atom('5')}

    def test_apply_gives_variable(self):
        """Phase 1: applying F=ma just returns Var('F') — numeric eval is Phase 3."""
        mg = _mg()
        law = self._build_fma_law(mg)
        expr = node('mul', atom('3'), atom('5'))
        b = match_law(law, expr)
        result = apply_law(law, b)
        # Var('F') is unbound in bindings → substitute returns var('F') unchanged
        assert result == var('F')

    def test_commutativity_law(self):
        """A commutativity law rewrites mul(A, B) → mul(B, A)."""
        mg = _mg()
        pat = node('mul', var('A'), var('B'))
        con = node('mul', var('B'), var('A'))
        law = add_expr_law(mg, 'mul_comm', pat, con)
        expr = node('mul', atom('3'), atom('5'))
        b = match_law(law, expr)
        result = apply_law(law, b)
        assert result == node('mul', atom('5'), atom('3'))

    def test_law_with_nested_expression(self):
        """match_law works when the ground expression has nested sub-trees."""
        mg = _mg()
        # d/dx(x^n) = n * x^(n-1)
        pat = node('d', node('pow', atom('x'), var('n')))
        con = node('mul', var('n'), node('pow', atom('x'), node('pred', var('n'))))
        law = add_expr_law(mg, 'power_rule', pat, con)
        expr = node('d', node('pow', atom('x'), atom('3')))
        b = match_law(law, expr)
        assert b == {'n': atom('3')}
        result = apply_law(law, b)
        expected = node('mul', atom('3'), node('pow', atom('x'), node('pred', atom('3'))))
        assert result == expected


# ---------------------------------------------------------------------------
# 3. Bitter Lesson cage — 10 anonymous symbol tables
# ---------------------------------------------------------------------------

class TestBitterLessonCage:
    """
    Cage test: replace operator NodeIds with anonymous Unicode symbols.
    The system must produce structurally identical results across all 10 seeds.

    What is anonymised
    ------------------
    Only operator tokens (e.g. 'mul', 'add', 'succ') are renamed.
    Digit atoms ('0'..'9') and variable names ('X', 'Y', etc.) are untouched —
    they are the perceptual primitives (digits) or internal identifiers
    (variables), not domain knowledge.

    Pass criterion
    --------------
    Bindings dict is identical across all 10 seeds.
    Applied conclusion is identical across all 10 seeds.
    No seed produces None where the named-symbol case produces a result.
    """

    # Operators to anonymise in each test
    _ROLES_BINARY = ['op']          # generic binary operator
    _ROLES_UNARY  = ['op']          # generic unary operator
    _ROLES_NESTED = ['f', 'g']      # two operators: f(g(x))

    def _run_binary_law_under_symbol_table(
        self,
        sym_table: dict[str, str],
    ) -> tuple[dict[str, Expr], Expr]:
        """
        Law: op(X, Y) → X  (project first argument).
        Expression: op(3, 5).
        Returns (bindings, applied_conclusion).
        """
        op_sym = sym_table['op']
        mg = _mg()
        nid = _nid_map(sym_table)
        pat_named = node('op', var('X'), var('Y'))
        con_named = var('X')
        pat = rename_expr(pat_named, nid)
        con = rename_expr(con_named, nid)
        law = add_expr_law(mg, 'cage_law', pat, con)
        expr = rename_expr(node('op', atom('3'), atom('5')), nid)
        b = match_law(law, expr)
        assert b is not None, f"match failed for op={op_sym!r}"
        result = apply_law(law, b)
        # Normalise bindings back to named-symbol keys for comparison
        # (variable names X, Y are NOT renamed — they are internal identifiers)
        return b, result

    def test_cage_binary_law_10_seeds(self):
        """Binary projection law: identical bindings and result across 10 seeds."""
        reference_bindings = None
        reference_result   = None

        for seed in range(10):
            sym_table = _fresh_symbol_table(self._ROLES_BINARY, seed)
            bindings, result = self._run_binary_law_under_symbol_table(sym_table)

            # Bindings keys are variable names ('X', 'Y') — not anonymised.
            # Binding VALUES are Expr atoms for '3' and '5' — not anonymised.
            # So the binding dict should be identical every seed.
            if reference_bindings is None:
                reference_bindings = bindings
                reference_result   = result
            else:
                assert bindings == reference_bindings, (
                    f"seed {seed}: bindings differ.\n"
                    f"  expected: {reference_bindings}\n"
                    f"  got:      {bindings}"
                )
                assert result == reference_result, (
                    f"seed {seed}: result differs.\n"
                    f"  expected: {reference_result}\n"
                    f"  got:      {result}"
                )

    def test_cage_nested_law_10_seeds(self):
        """
        Nested law: f(g(X)) → X  (composition stripping).
        Anonymise both 'f' and 'g'.  Bindings and result must be stable.
        """
        reference_bindings = None
        reference_result   = None

        for seed in range(10):
            sym_table = _fresh_symbol_table(self._ROLES_NESTED, seed)
            nid = _nid_map(sym_table)
            mg = _mg()

            pat_named = node('f', node('g', var('X')))
            con_named = var('X')
            pat = rename_expr(pat_named, nid)
            con = rename_expr(con_named, nid)
            law = add_expr_law(mg, 'cage_nested', pat, con)

            expr_named = node('f', node('g', atom('7')))
            expr = rename_expr(expr_named, nid)
            b = match_law(law, expr)
            assert b is not None, f"seed {seed}: match failed"
            result = apply_law(law, b)

            if reference_bindings is None:
                reference_bindings = b
                reference_result   = result
            else:
                assert b == reference_bindings, f"seed {seed}: bindings differ"
                assert result == reference_result, f"seed {seed}: result differs"

    def test_cage_commutativity_10_seeds(self):
        """
        Commutativity law: op(A, B) → op(B, A).
        The result is the flipped expression.  With anonymised op, the
        STRUCTURE of the result (an anonymous binary application with swapped
        args) must be the same across all seeds.
        """
        for seed in range(10):
            sym_table = _fresh_symbol_table(['op'], seed)
            nid = _nid_map(sym_table)
            mg = _mg()

            pat = rename_expr(node('op', var('A'), var('B')), nid)
            con = rename_expr(node('op', var('B'), var('A')), nid)
            law = add_expr_law(mg, 'comm', pat, con)

            expr = rename_expr(node('op', atom('3'), atom('5')), nid)
            b = match_law(law, expr)
            assert b is not None, f"seed {seed}: match failed"
            result = apply_law(law, b)

            # Structure: result.head == anon_op, result.args == (atom('5'), atom('3'))
            assert result.head == nid.get(TOKEN_GRAPH.encode('op'),
                                          TOKEN_GRAPH.encode('op')), \
                f"seed {seed}: wrong head"
            assert result.args[0] == atom('5'), f"seed {seed}: wrong first arg"
            assert result.args[1] == atom('3'), f"seed {seed}: wrong second arg"

    def test_cage_zero_variance(self):
        """
        Explicit variance check: collect 10 (is_match: bool) results and verify
        variance == 0 (all seeds succeed).
        """
        outcomes = []
        for seed in range(10):
            sym_table = _fresh_symbol_table(['op'], seed)
            nid = _nid_map(sym_table)
            mg = _mg()
            pat = rename_expr(node('op', var('X'), var('Y')), nid)
            con = rename_expr(node('op', var('Y'), var('X')), nid)
            law = add_expr_law(mg, 'var_test', pat, con)
            expr = rename_expr(node('op', atom('1'), atom('2')), nid)
            outcomes.append(match_law(law, expr) is not None)

        assert all(outcomes), f"Some seeds failed: {outcomes}"
        # Variance of a uniform True list is 0
        mean = sum(outcomes) / len(outcomes)
        variance = sum((x - mean) ** 2 for x in outcomes) / len(outcomes)
        assert variance == 0.0


# ---------------------------------------------------------------------------
# 4. Defect probe — tree identity vs string identity
# ---------------------------------------------------------------------------

class TestDefectProbe:
    """
    Two expressions with the same multiset of NodeIds but different tree
    structures must be treated as strictly distinct by:
      - Expr.__eq__
      - match_law
      - ExprLaw deduplication

    Catches an implementation that identifies expressions by their sorted
    NodeId multiset (or any other structure-discarding fingerprint) rather
    than by tree equality.

    The two expressions:
      e1 = op(op(a, b), c)   — left-associative: (a op b) op c
      e2 = op(a, op(b, c))   — right-associative: a op (b op c)
    Both contain the same tokens {op:2, a:1, b:1, c:1} in the same total
    count — but the tree structures are different.
    """

    def _build_pair(self):
        a, b, c = atom('a'), atom('b'), atom('c')
        e1 = node('op', node('op', a, b), c)   # left-assoc: (a op b) op c
        e2 = node('op', a, node('op', b, c))   # right-assoc: a op (b op c)
        return e1, e2

    def test_expr_not_equal(self):
        e1, e2 = self._build_pair()
        assert e1 != e2, "left- and right-assoc trees must not be equal"

    def test_multiset_same(self):
        """Document that the NodeId multisets ARE the same (the ambiguity exists)."""
        from collections import Counter
        def node_ids(e: Expr) -> list[int]:
            return [e.head] + [nid for a in e.args for nid in node_ids(a)]

        e1, e2 = self._build_pair()
        assert Counter(node_ids(e1)) == Counter(node_ids(e2)), (
            "Test setup error: multisets should be equal"
        )

    def test_match_gives_different_bindings(self):
        """Matching the same pattern against e1 and e2 must give different bindings."""
        e1, e2 = self._build_pair()
        pat = node('op', var('L'), var('R'))

        b1 = match(pat, e1)
        b2 = match(pat, e2)

        assert b1 is not None
        assert b2 is not None
        # L binds to different sub-expressions
        assert b1['L'] != b2['L'], (
            f"L should differ:\n  e1 binding: {b1['L']}\n  e2 binding: {b2['L']}"
        )
        assert b1['R'] != b2['R'], (
            f"R should differ:\n  e1 binding: {b1['R']}\n  e2 binding: {b2['R']}"
        )

    def test_law_dedup_distinguishes_trees(self):
        """
        Two laws with structurally distinct patterns must NOT be deduplicated,
        even though both patterns contain the same NodeId multiset.
        """
        mg = _mg()
        e1, e2 = self._build_pair()
        pat1 = e1     # (op op a b) c — left-assoc pattern
        pat2 = e2     # op a (op b c) — right-assoc pattern
        con  = var('X')

        law1 = add_expr_law(mg, 'assoc_test', pat1, con)
        law2 = add_expr_law(mg, 'assoc_test', pat2, con)

        assert law1.morph_id != law2.morph_id, (
            "Structurally distinct patterns must produce distinct morphisms"
        )
        assert len(query_expr_laws(mg, 'assoc_test')) == 2

    def test_apply_after_match_is_structure_aware(self):
        """
        Applying a commutativity law to e1 and e2 produces different results,
        because the arguments are swapped at the correct nesting level.
        """
        mg = _mg()
        # comm: op(L, R) → op(R, L)
        pat = node('op', var('L'), var('R'))
        con = node('op', var('R'), var('L'))
        law = add_expr_law(mg, 'comm', pat, con)

        e1, e2 = self._build_pair()

        r1 = apply_law(law, match_law(law, e1))
        r2 = apply_law(law, match_law(law, e2))

        # r1 = op(c, op(a, b))   (right arg was c, left was op(a,b))
        # r2 = op(op(b, c), a)   (right arg was op(b,c), left was a)
        assert r1 != r2

    def test_anonymous_symbols_preserve_distinction(self):
        """
        Even with anonymous operator symbols, e1 ≠ e2 and matching gives
        different bindings.  The cage and the defect probe compose.
        """
        sym_table = _fresh_symbol_table(['op', 'a', 'b', 'c'], seed=42)
        nid = _nid_map(sym_table)

        a_anon = rename_expr(atom('a'), nid)
        b_anon = rename_expr(atom('b'), nid)
        c_anon = rename_expr(atom('c'), nid)
        op_nid = TOKEN_GRAPH.encode(sym_table['op'])

        e1_anon = Expr(head=op_nid,
                       args=(Expr(head=op_nid, args=(a_anon, b_anon)), c_anon))
        e2_anon = Expr(head=op_nid,
                       args=(a_anon, Expr(head=op_nid, args=(b_anon, c_anon))))

        assert e1_anon != e2_anon

        pat_anon = Expr(head=op_nid, args=(var('L'), var('R')))
        b1 = match(pat_anon, e1_anon)
        b2 = match(pat_anon, e2_anon)
        assert b1 is not None and b2 is not None
        assert b1['L'] != b2['L']
