"""
Next-token prediction: process rule → type assignment → fixpoint → morphism
marginalization → JSD Kan extension → marginal.

Pipeline (priority order):

    Level 1 — Process rule:
        If the current phase is OUTPUT and the operator has a ProcessRule,
        apply the carry-propagation table to deterministically predict the
        next digit.  Also predicts <eos> once all output digits have been
        generated.

    Level 2 — Fixpoint iteration + Markov morphism marginalization:
        Assigns soft type distributions to each prefix token, then iterates
        the presheaf update T^{n+1}(i) = restrict(T^n, context_under_T^n at i)
        until convergence (max_i ||T^{n+1}(i) − T^n(i)||₁ < ε) or N_max
        iterations.  Sheaf obstruction detection: if the L1 norm sequence
        cycles, the position is flagged as ambiguous.

        From the converged T_last, marginalises over morphisms:
            P(next_atom) = Σ_c P(c|last_pos)
                         * Σ_{f: c→d} evidence(f)*exp(conf(f))
                         * intent_weights(d, next_atom)

    Level 3 — JSD Kan extension:
        Builds query centroid = Σ_c T_last[c] * centroid_c, then weights
        each concept by exp(-JSD(query, centroid_c) / τ) and returns the
        support-weighted mixture of intent distributions.

    Level 4 — Marginal (uniform):
        Uniform distribution over all atoms in the vocabulary.

See CTKG_ARCHITECTURE.md §Prediction for the full specification.
"""

from __future__ import annotations

import math
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import (
    ConceptLattice,
    ConceptId,
)
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    ProcessRule,
    ChainRule,
    FreeCategoryGraph,
    BinaryFoldRule,
    build_fc_lookup,
    build_adj_lookup,
    build_unary_chain_maps,
    build_unary_carry_maps,
    complete_succ_map,
    discover_binary_fold_rules,
    unary_chain_predict,
    fold_rules_as_rewrite_rules,
    build_binary_functional_maps,
)
from experiments.symbolic_ai_v2.ctkg.learning.relation_store import (
    RelationStore,
    RelationRule,
    discover_relation_rules,
    predict_from_relation_rules,
    _merge_digit_runs as _rs_merge_digit_runs,
)
from dataclasses import dataclass, field
from experiments.symbolic_ai_v2.ctkg.core.kan_extension import KanExtension
from experiments.symbolic_ai_v2.ctkg.core.working_memory import (
    parse_prefix,
    parse_chain_prefix,
    WorkingMemory,
)
from experiments.symbolic_ai_v2.ctkg.core.spine import Spine


# Fixpoint iteration parameters (architecture §Prediction step 3)
_FP_MAX_ITER: int = 20
_FP_EPS: float = 1e-4
_FP_CYCLE_K: int = 5   # number of snapshots to keep for cycle detection


def _build_multidigit_arith_rules(
    bfm: dict[str, dict[tuple[str, str], str]],
) -> list:
    """Generate add/sub ground rules for two-digit × single-digit operands.

    Covers cases like add('36','3')→'39' that arise when a multi-digit
    intermediate (e.g. mul(6,6)='36') is fed into a second arithmetic op.

    No int() calls: uses only string slicing and bfm lookups.
    Only generates rules whose result is ≤2 digits (enough for linear_eval).
    """
    from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom as _atom, node as _node
    from experiments.symbolic_ai_v2.ctkg.learning.rule_discover import RewriteRule

    add_map = bfm.get('add', {})
    sub_map = bfm.get('sub', {})
    rules: list[RewriteRule] = []

    # Collect two-digit values that appear as bfm results
    two_digit_vals: set[str] = set()
    for op_map in bfm.values():
        for result in op_map.values():
            if len(result) == 2:
                two_digit_vals.add(result)

    # All single-digit keys
    single_digits: set[str] = set()
    for (a, b) in add_map:
        single_digits.add(a)
        single_digits.add(b)

    for R in two_digit_vals:
        tens, ones = R[0], R[1]
        for b in single_digits:
            # --- add(R, b) ---
            sum_ones = add_map.get((ones, b))
            if sum_ones is not None:
                if len(sum_ones) == 1:
                    # No carry
                    result = tens + sum_ones
                    rules.append(RewriteRule(
                        lhs=_node('add', _atom(R), _atom(b)),
                        rhs=_atom(result),
                        algebra_name='add', evidence=1,
                    ))
                elif len(sum_ones) == 2:
                    # sum_ones is '1X' — carry of 1 into tens
                    carry_digit, new_ones = sum_ones[0], sum_ones[1]
                    new_tens = add_map.get((tens, carry_digit))
                    if new_tens is not None and len(new_tens) == 1:
                        result = new_tens + new_ones
                        rules.append(RewriteRule(
                            lhs=_node('add', _atom(R), _atom(b)),
                            rhs=_atom(result),
                            algebra_name='add', evidence=1,
                        ))

            # --- sub(R, b) ---
            # ones - b: if ones >= b (no borrow), use sub_map directly
            diff_ones = sub_map.get((ones, b))
            if diff_ones is not None and len(diff_ones) == 1:
                result = tens + diff_ones
                rules.append(RewriteRule(
                    lhs=_node('sub', _atom(R), _atom(b)),
                    rhs=_atom(result),
                    algebra_name='sub', evidence=1,
                ))
            # borrow case: tens > 0, borrow 1 from tens
            # new_ones = (10 + ones_val) - b  — but we must do this without int()
            # Use: borrow_ones = add_map[('10'[1], ones)] minus b via sub_map
            # Skip for now — borrow requires knowing '10' representation

    return rules


class Predictor:
    """Full CTKG next-token predictor.

    Parameters
    ----------
    hankel:
        Trained HankelCount (Phase 1 output).  Used only for vocabulary.
    lattice:
        ConceptLattice (Phase 2 output).
    morphism_graph:
        MorphismGraph (Phase 3 output).
    process_rules:
        List of ProcessRule objects (Phase 5, from process_discover).
        The operator set is derived from these rules — no hardcoded list.
    k_neighbours:
        Kept for API compatibility; not used in the JSD pipeline.
    r:
        Context radius (kept for API compatibility; not used in JSD pipeline).
    tau:
        JSD softmax temperature for Kan extension.  Default 0.1.
    """

    def __init__(
        self,
        hankel: HankelCount,
        lattice: ConceptLattice,
        morphism_graph: MorphismGraph,
        process_rules: list[ProcessRule],
        k_neighbours: int = 5,
        r: int = 1,
        tau: float = 0.1,
        chain_rules: Optional[list[ChainRule]] = None,
        fc: Optional[FreeCategoryGraph] = None,
    ) -> None:
        self._hankel = hankel
        self._lattice = lattice
        self._morphism_graph = morphism_graph
        self._r = r

        # Operator → ProcessRule lookup (fold-type)
        self._rules: dict[str, ProcessRule] = {
            rule.op_atom: rule for rule in process_rules
        }

        # Operator → ChainRule lookup (trace-format: step/ans)
        self._chain_rules: dict[str, ChainRule] = {
            rule.op_atom: rule for rule in (chain_rules or [])
        }

        # Discovered operator sets
        self._op_atoms: frozenset[str] = frozenset(self._rules.keys())
        self._chain_op_atoms: frozenset[str] = frozenset(self._chain_rules.keys())

        # Level 0.5: FC edge + adjunction-based lookup (CT_REFERENCE §4,17)
        # Build from the free category if provided; extend op_atoms to include
        # all operators discovered in the FC (so parse_prefix identifies them).
        if fc is not None:
            self._fc_lookup: dict[tuple, tuple] = build_fc_lookup(fc)
            self._adj_lookup: dict[tuple, tuple] = build_adj_lookup(fc)
            self._unary_chain_maps: dict[str, dict[str, str]] = build_unary_chain_maps(fc)
            self._unary_carry_maps: dict[str, tuple] = build_unary_carry_maps(fc)
            # Level 0.7: Composition engine — discover NNO fold rules once,
            # apply them dynamically at inference time (CT_REFERENCE §19).
            self._fold_rules: dict[str, BinaryFoldRule] = discover_binary_fold_rules(fc)
            # Extract succ tools for the composition engine.
            # Pick the NNO candidate with the longest discovered chain — that
            # is the true successor/predecessor (not sq, sqrt, etc. which have
            # short chains of 2-3 elements in the training data).
            _succ_carry = self._unary_carry_maps
            _nno = (
                max(fc.nno_candidates, key=lambda n: len(n.successor_map))
                if fc.nno_candidates else None
            )
            if _nno is not None and _nno.op in _succ_carry:
                _raw_succ = _nno.successor_map
                _carry_el = _succ_carry[_nno.op][0]
                _carry_out = _succ_carry[_nno.op][1]
                _zero = _nno.zero_candidate
                # Complete the succ map: infer missing edges (train/test split gaps)
                self._compose_succ_map: dict[str, str] = complete_succ_map(
                    _raw_succ, _zero, _carry_el
                )
                self._compose_carry_el: str = _carry_el
                self._compose_carry_out: tuple = _carry_out
                self._compose_zero: str = _zero
            else:
                self._compose_succ_map = {}
                self._compose_carry_el = ""
                self._compose_carry_out = ()
                self._compose_zero = ""
            # Memoization cache shared across all _compose calls within a session
            self._compose_cache: dict[tuple, tuple] = {}
            fc_ops = frozenset(edge.op for edge in fc.edges)
            self._op_atoms = self._op_atoms | fc_ops
            # Adjunction-mediated inverse solve: for ops that are the right-side
            # of an n-ary adjunction (e.g. sub = right-side of add⊣sub), derive
            # the answer by enumerating candidates via the left (forward) op.
            # Maps right_op → (left_op, preserved_position).
            self._adj_solve_map: dict[str, tuple] = {}
            for adj in fc.adjunctions:
                if adj.preserved_position is not None:
                    rk = adj.right_op
                    if rk not in self._adj_solve_map:
                        self._adj_solve_map[rk] = (adj.left_op, adj.preserved_position)
            # Level 1c: surface-form-agnostic rewrite rules via anti-unification.
            # Builds a corpus of (op + inputs eq output) sequences from chain_rule
            # tables, discovers arities (seeded with the NNO alphabet so digit-name
            # tokens like 'zero'..'nine' are correctly treated as arity-0 atoms),
            # then discovers RewriteRules via anti-unification.
            # Phase V: multi-digit merging for chain_table ans_part; sq normalization;
            # pred/succ functional alignment; ground pred/succ rules for verification.
            # No .isdigit() calls anywhere in this path — Iron Rule compliant.
            nno_atoms: frozenset[str] = (
                frozenset(self._compose_succ_map.keys())
                | frozenset(self._compose_succ_map.values())
            )
            _eq_corpus: list[list[str]] = []
            for cr in (chain_rules or []):
                for inp_toks, out_toks in (cr.eq_table or {}).items():
                    _eq_corpus.append(
                        [cr.op_atom] + list(inp_toks) + ['eq'] + list(out_toks)
                    )
                for inp_toks, out_toks in (cr.chain_table or {}).items():
                    out_list = list(out_toks)
                    if 'ans' in out_list:
                        ans_idx = out_list.index('ans')
                        ans_part = out_list[ans_idx + 1:]
                        if ans_part:
                            # Phase V: merge consecutive NNO-alphabet tokens in
                            # ans_part only (e.g. ['1','2'] → ['12'] for coefficient
                            # 12 in derivative outputs).  eq_table entries use
                            # separate single-digit args and must NOT be merged.
                            merged_ans = _merge_digit_runs(ans_part, nno_atoms)
                            _eq_corpus.append(
                                [cr.op_atom] + list(inp_toks) + ['eq'] + merged_ans
                            )
            if _eq_corpus:
                from experiments.symbolic_ai_v2.ctkg.core.expr_parser import (
                    discover_arities, TERMINATORS,
                )
                from experiments.symbolic_ai_v2.ctkg.learning.rule_discover import discover_rules
                from experiments.symbolic_ai_v2.ctkg.core.term_algebra import (
                    atom as _atom, node as _node, var as _var,
                )
                from experiments.symbolic_ai_v2.ctkg.core.rewrite import RewriteRule

                # Seed extra atoms: NNO alphabet + any compound tokens from merging
                extra_seeds: dict = {t: 0 for t in nno_atoms}
                for seq in _eq_corpus:
                    for tok in seq:
                        if tok not in TERMINATORS and len(tok) > 1:
                            if all(c in nno_atoms for c in tok):
                                extra_seeds[tok] = 0

                self._arities: dict = discover_arities(_eq_corpus, extra_seeds=extra_seeds)
                self._atoms: frozenset[str] = frozenset(
                    t for t, a in self._arities.items() if a == 0
                )

                # Phase V: build normalization rules and functional maps.
                # sq(V)→pow(V,2): structural identity applied to both input/output.
                # atom('x')→pow(x,1): output-only normalization for uniform grouping.
                # Inverse: pow(x,1)→x for post-processing after cata_reduce.
                # These rules are not value-specific — Iron Rule compliant.
                sq_norm = RewriteRule(
                    lhs=_node('sq', _var('V0')),
                    rhs=_node('pow', _var('V0'), _atom('2')),
                    algebra_name='sq_norm', evidence=1,
                )
                _one_tok = self._compose_succ_map.get(self._compose_zero, '1')
                # mul(V0, x) → mul(V0, pow(x,1)): normalize bare-x polynomial
                # output to pow(x,1) so anti-unification aligns n=1 with n>1.
                # More specific than x→pow(x,1): does not replace x inside pow(x,N).
                x_pow1_norm = RewriteRule(
                    lhs=_node('mul', _var('_V_coeff'), _atom('x')),
                    rhs=_node('mul', _var('_V_coeff'), _node('pow', _atom('x'), _atom(_one_tok))),
                    algebra_name='mul_x_pow1_norm', evidence=1,
                )
                # Inverse: pow(x, one) → x  (denormalize before unparsing)
                pow_x1_inv = RewriteRule(
                    lhs=_node('pow', _atom('x'), _atom(_one_tok)),
                    rhs=_atom('x'),
                    algebra_name='pow_x1_inv', evidence=1,
                )

                # Build ground pred/succ rules for verification and inference.
                ground_nno: list = []
                for d_from, d_to in self._compose_succ_map.items():
                    ground_nno.append(RewriteRule(
                        lhs=_node('succ', _atom(d_from)),
                        rhs=_atom(d_to),
                        algebra_name='succ', evidence=1,
                    ))
                    ground_nno.append(RewriteRule(
                        lhs=_node('pred', _atom(d_to)),
                        rhs=_atom(d_from),
                        algebra_name='pred', evidence=1,
                    ))

                # Functional maps for _align_rhs_variables: detect W = pred(V) etc.
                _pred_map = {v: k for k, v in self._compose_succ_map.items()}
                _succ_map = dict(self._compose_succ_map)
                functional_maps = {
                    'pred': _pred_map,
                    'succ': _succ_map,
                }

                # Discover rules with normalization + functional alignment.
                # Use sq_norm on both sides; x_pow1_norm on outputs only; ground NNO
                # for verification; functional maps for pred/succ alignment.
                norm_rules = [sq_norm]
                if 'sq' in self._arities and 'pow' in self._arities:
                    pass  # sq_norm is safe when both operators are known
                else:
                    norm_rules = []   # don't normalize if operators not in arities

                output_norm_rules = []
                if 'x' in self._arities and 'pow' in self._arities:
                    output_norm_rules = [x_pow1_norm]

                # Phase VIII: binary functional maps for W = op(V0,V1) alignment
                # (e.g. coefficient = mul(coeff_in, exponent) in power rule).
                _binary_fmaps = build_binary_functional_maps(
                    fc,
                    self._compose_succ_map,
                    self._compose_carry_el,
                    self._compose_carry_out,
                    self._compose_zero,
                ) if self._compose_succ_map else {}

                # Extend _binary_fmaps with multi-digit add/sub entries so that
                # the two-level binary functional match in _align_rhs_variables
                # can discover composition rules like eval(A,x,B,at,C)→add(mul(A,C),B)
                # even when mul(A,C) is two-digit (e.g. mul('6','6')='36').
                # Uses only string operations — no int() calls.
                if _binary_fmaps:
                    _add_map = _binary_fmaps.get('add', {})
                    _sub_map = _binary_fmaps.get('sub', {})
                    _two_digit_vals: set[str] = set()
                    for _op_m in _binary_fmaps.values():
                        for _res in _op_m.values():
                            if len(_res) == 2:
                                _two_digit_vals.add(_res)
                    _single_digits: set[str] = set()
                    for (_a, _b) in _add_map:
                        _single_digits.add(_a)
                        _single_digits.add(_b)
                    _extended_add: dict[tuple, str] = {}
                    _extended_sub: dict[tuple, str] = {}
                    for _R in _two_digit_vals:
                        _tens, _ones = _R[0], _R[1]
                        for _b in _single_digits:
                            # add(R, b)
                            _s1 = _add_map.get((_ones, _b))
                            if _s1 is not None:
                                if len(_s1) == 1:
                                    _extended_add[(_R, _b)] = _tens + _s1
                                elif len(_s1) == 2:
                                    _carry, _new_ones = _s1[0], _s1[1]
                                    _nt = _add_map.get((_tens, _carry))
                                    if _nt is not None and len(_nt) == 1:
                                        _extended_add[(_R, _b)] = _nt + _new_ones
                            # sub(R, b)
                            _d1 = _sub_map.get((_ones, _b))
                            if _d1 is not None and len(_d1) == 1:
                                _extended_sub[(_R, _b)] = _tens + _d1
                    if _extended_add:
                        _binary_fmaps = dict(_binary_fmaps)
                        _binary_fmaps['add'] = {**_binary_fmaps.get('add', {}), **_extended_add}
                    if _extended_sub:
                        if 'add' not in _binary_fmaps:
                            _binary_fmaps = dict(_binary_fmaps)
                        _binary_fmaps['sub'] = {**_binary_fmaps.get('sub', {}), **_extended_sub}

                # Complete BFM for all single-digit × single-digit pairs using
                # the NNO compose engine.  The FC-based _binary_fmaps only covers
                # (op, a, b) pairs SEEN in training; OOD pairs (e.g. mul(6,5) when
                # that example is in the test split) cause RelationRule lookups to
                # fail.  Using _compose fills the gaps without int() calls.
                if self._compose_succ_map and _binary_fmaps:
                    _all_digits_list: list[str] = []
                    _dc = self._compose_zero
                    _dc_seen: set[str] = {_dc}
                    _all_digits_list.append(_dc)
                    while True:
                        _dn = self._compose_succ_map.get(_dc)
                        if _dn is None or _dn in _dc_seen:
                            break
                        _all_digits_list.append(_dn)
                        _dc_seen.add(_dn)
                        _dc = _dn
                    for _arith_op in ('mul', 'add', 'sub'):
                        if _arith_op not in _binary_fmaps:
                            continue
                        _op_map = dict(_binary_fmaps[_arith_op])
                        _updated = False
                        for _da in _all_digits_list:
                            for _db in _all_digits_list:
                                if (_da, _db) not in _op_map:
                                    _cr = _compose(
                                        _arith_op, ((_da,), (_db,)),
                                        self._fc_lookup, self._fold_rules,
                                        self._compose_succ_map,
                                        self._compose_carry_el,
                                        self._compose_carry_out,
                                        self._compose_zero,
                                        self._compose_cache,
                                    )
                                    if _cr is not None:
                                        _op_map[(_da, _db)] = ''.join(_cr)
                                        _updated = True
                        if _updated:
                            _binary_fmaps = dict(_binary_fmaps)
                            _binary_fmaps[_arith_op] = _op_map

                # Ground binary functional rules for verification: mul(3,4)→12 etc.
                # These let cata_reduce fully evaluate sub-expressions produced by
                # structural rules (e.g. mul(mul(c,n), pow(x,pred(n))) needs mul(c,n)
                # to reduce to its numeric result for consistency checking).
                # Only include pure digit×digit arithmetic ops (add, mul, sub) —
                # NOT pow/succ/pred/sq/sqrt since those appear in structural positions
                # and would interfere with rule verification.
                _ARITH_OPS = frozenset({'add', 'mul', 'sub'})
                ground_bfm: list[RewriteRule] = []
                for _op_name, _op_map in (_binary_fmaps or {}).items():
                    if _op_name not in _ARITH_OPS:
                        continue
                    for (_d1, _d2), _res in _op_map.items():
                        ground_bfm.append(RewriteRule(
                            lhs=_node(_op_name, _atom(_d1), _atom(_d2)),
                            rhs=_atom(_res),
                            algebra_name=_op_name,
                            evidence=1,
                        ))

                self._rewrite_rules: list = discover_rules(
                    _eq_corpus, self._arities,
                    norm_rules=norm_rules,
                    output_norm_rules=output_norm_rules,
                    functional_maps=functional_maps,
                    aux_rules=ground_nno + ground_bfm,
                    binary_functional_maps=_binary_fmaps or None,
                )
                # Add ground NNO rules + inference-time normalization.
                # sq_norm fires bottom-up on inputs so that sq(x) → pow(x,2)
                # before the structural derivative rule is tried.
                self._rewrite_rules.extend(ground_nno)
                if norm_rules:
                    self._rewrite_rules.extend(norm_rules)  # sq(V)→pow(V,2) etc.

                # Post-rules: denormalize outputs back to corpus surface form.
                # pow(x,1)→x (inverse of x_pow1_norm) and pow(x,2)→sq(x) (inverse of sq_norm).
                sq_inv = RewriteRule(
                    lhs=_node('pow', _var('V0'), _atom('2')),
                    rhs=_node('sq', _var('V0')),
                    algebra_name='sq_inv', evidence=1,
                ) if norm_rules else None
                self._post_rules: list = []
                if output_norm_rules:
                    self._post_rules.append(pow_x1_inv)
                if sq_inv is not None:
                    self._post_rules.append(sq_inv)

                # Phase VIII: NNO ground arithmetic rules so that sub-expressions
                # like mul('3','2') reduce to '6' within structural rules.
                # Generated from BinaryFoldRule NNO induction; no int() calls.
                self._nno_rules: list = fold_rules_as_rewrite_rules(
                    fc,
                    self._compose_succ_map,
                    self._compose_carry_el,
                    self._compose_carry_out,
                    self._compose_zero,
                ) if self._compose_succ_map else []
                # Extend with multi-digit add/sub rules (e.g. add('36','3')→'39')
                # so structural composition rules like add(mul(A,C), B) can be
                # fully evaluated when the mul result is two digits.
                if _binary_fmaps:
                    self._nno_rules.extend(
                        _build_multidigit_arith_rules(_binary_fmaps)
                    )

                # ---- RelationStore: per-role rule discovery ----
                # Hypergraph approach: each sequence is a named-role tuple.
                # Discover a separate RewriteRule for each output role (step, ans).
                # This handles trace-format ops (step/ans structure) that
                # _cata_predict can't handle (it produces only the final answer).
                # No discover_arities() needed for ops with input separators.
                _rs = RelationStore()
                _rs_seqs: list[list[str]] = []
                for cr in (chain_rules or []):
                    for inp_toks, out_toks in (cr.chain_table or {}).items():
                        _rs_seqs.append([cr.op_atom] + list(inp_toks) + list(out_toks))
                _rs.update_batch(_rs_seqs)

                # Build step_corpus only for ops with input separators (clean schema)
                _step_corpus: list[list[str]] = _rs.eq_corpus_for_role(
                    'step',
                    ops=_rs.ops_with_input_seps(),
                    merge_digits=True,
                    nno_atoms=nno_atoms,
                )

                # Discover step rules via same anti-unification pipeline as ans rules.
                # Step values may be multi-digit (e.g. '10' for 2*5).  After merging,
                # these compound tokens are NOT in self._arities (which was built from
                # ans_corpus only).  Extend arities with all compound step-output tokens
                # before calling discover_rules so the parser can handle them.
                if _step_corpus:
                    from experiments.symbolic_ai_v2.ctkg.core.expr_parser import TERMINATORS as _TERM
                    _step_arities = dict(self._arities)
                    for _seq in _step_corpus:
                        for _tok in _seq:
                            if _tok not in _TERM and len(_tok) > 1:
                                if all(c in nno_atoms for c in _tok):
                                    _step_arities.setdefault(_tok, 0)
                    self._step_rules: list = discover_rules(
                        _step_corpus, _step_arities,
                        norm_rules=norm_rules,
                        output_norm_rules=None,  # step outputs are plain digits
                        functional_maps=functional_maps,
                        aux_rules=ground_nno + ground_bfm,
                        binary_functional_maps=_binary_fmaps or None,
                    )
                    self._step_rules.extend(ground_nno)
                else:
                    self._step_rules: list = []

                # Record which ops have a step rule (for trace prediction routing)
                self._trace_ops: frozenset[str] = _rs.ops_with_input_seps()

                # ---- Arity-free RelationRule discovery (hypergraph approach) ----
                # For each op with clean input separators, discover rules that map
                # named input roles → output roles using the BFM directly.
                # No discover_arities(), no prefix S-expression, no parse tree.
                self._rs = _rs
                self._binary_fmaps: dict = _binary_fmaps or {}
                self._relation_rules: dict[str, list[RelationRule]] = {}
                for _op in _rs.ops_with_input_seps():
                    _op_rels = _rs.get_relations(_op)
                    _op_rr = discover_relation_rules(_op_rels, _binary_fmaps or {})
                    if _op_rr:
                        self._relation_rules[_op] = _op_rr

            else:
                self._arities = {}
                self._atoms = nno_atoms
                self._rewrite_rules = []
                self._post_rules: list = []
                self._nno_rules: list = []
                self._step_rules: list = []
                self._trace_ops: frozenset[str] = frozenset()
                self._rs = None
                self._binary_fmaps: dict = {}
                self._relation_rules: dict[str, list[RelationRule]] = {}
        else:
            self._fc_lookup = {}
            self._adj_lookup = {}
            self._unary_chain_maps = {}
            self._unary_carry_maps = {}
            self._fold_rules = {}
            self._compose_succ_map = {}
            self._compose_carry_el = ""
            self._compose_carry_out = ()
            self._compose_zero = ""
            self._compose_cache = {}
            self._adj_solve_map = {}
            self._arities = {}
            self._atoms: frozenset[str] = frozenset()
            self._rewrite_rules: list = []
            self._post_rules: list = []
            self._nno_rules: list = []
            self._step_rules: list = []
            self._trace_ops: frozenset[str] = frozenset()
            self._rs = None
            self._binary_fmaps: dict = {}
            self._relation_rules: dict[str, list[RelationRule]] = {}

        # Pre-build MorphismGraph transition matrix (concept_id → concept_id → weight)
        self._trans = _build_transition(morphism_graph)

        # JSD Kan extension (no pre-fitting required)
        self._kan = KanExtension(lattice, tau=tau)

        # Vocabulary for marginal fallback (union of hankel vocab and lattice atoms)
        self._vocab: list[str] = list(hankel.vocabulary())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_next(self, prefix: list[str]) -> dict[str, float]:
        """Return P(next_token | prefix).

        Parameters
        ----------
        prefix:
            All tokens observed / generated so far.

        Returns
        -------
        Mapping from atom string to probability.  Process rule levels return
        a point mass {atom: 1.0}; distribution levels return soft distributions.
        """
        # Level 1b: Chain rule (deterministic, step/ans and eq-mixed formats)
        #
        # MUST fire before Level 1a (fold rules) because chain-op sequences
        # (e.g. ['d', 'add', 'mul', '3', 'sq', 'x', ...]) contain fold-op
        # tokens ('add', 'mul', …) as ARGUMENTS, not as the top-level operator.
        # If Level 1a ran first, parse_prefix would misidentify the inner 'add'
        # as the operator and produce the wrong prediction.
        #
        # Fires in both INPUT and OUTPUT phases:
        #   INPUT  — no output delimiter seen yet; predicts first output token.
        #            Only fires when input_tokens exactly match a chain_table key.
        #   OUTPUT — past first delimiter; predicts next token in the chain.
        #
        # Also handles eq-format sequences whose input contains non-digit tokens
        # (e.g. 'eval', 'd', 'int') — rejected by fold discovery but captured
        # here by extending parse_chain_prefix to recognise 'eq' as a delimiter.
        if self._chain_rules:
            chain_state = parse_chain_prefix(prefix, self._chain_op_atoms)
            if (
                chain_state is not None
                and chain_state.phase in ("INPUT", "OUTPUT")
                and chain_state.op in self._chain_rules
            ):
                chain_rule = self._chain_rules[chain_state.op]
                # Use eq_table when the prefix contains 'eq' (eq-format sequences
                # like plain derivatives/integrals); chain_table for step/ans format.
                use_eq = "eq" in prefix
                result = _chain_predict(chain_rule, chain_state.input_tokens, chain_state.output_tokens, use_eq_table=use_eq)
                if result is not None:
                    return result

                # Level 1c-relational: arity-free hypergraph rule prediction.
                # Uses RelationRules discovered from named-role tuples — no arities,
                # no prefix S-expressions, no parse tree.  Fires for any op with a
                # clean input separator schema (the rel predictor handles k=0..end).
                # Guard: skip when 'eq' is already in the prefix — those sequences
                # use eq-format output, not step/ans trace format.  The relational
                # rules were discovered from chain_table (trace-format) sequences only.
                if (self._relation_rules
                        and chain_state.op in self._relation_rules
                        and self._rs is not None
                        and "eq" not in prefix):
                    _rel_input = [chain_state.op] + list(chain_state.input_tokens)
                    _rel_output = predict_from_relation_rules(
                        _rel_input, self._rs, self._relation_rules, self._binary_fmaps
                    )
                    if _rel_output is not None:
                        _rk = len(chain_state.output_tokens)
                        if _rk < len(_rel_output):
                            return {_rel_output[_rk]: 1.0}
                        if _rk == len(_rel_output):
                            return {'<eos>': 1.0}

                # Level 1c-trace: per-role rule application for trace-format ops.
                # Arity-based fallback for when relational rules haven't been discovered.
                # Only fires after a 'step'/'ans' delimiter is already in output_tokens.
                _out_has_trace = (
                    'step' in chain_state.output_tokens
                    or 'ans' in chain_state.output_tokens
                )
                if self._step_rules and chain_state.op in self._trace_ops and _out_has_trace:
                    _all_rules = self._rewrite_rules + self._nno_rules
                    trace_result = _trace_cata_predict(
                        chain_state.op, chain_state.input_tokens,
                        chain_state.output_tokens,
                        self._step_rules, _all_rules,
                        self._arities, self._post_rules,
                        self._atoms,
                    )
                    if trace_result is not None:
                        return trace_result

                # Level 1c: cata_reduce (surface-form-agnostic rule application).
                # Phase VIII: structural rules + NNO ground arithmetic rules.
                # Structural rules (e.g. d(mul(V0,pow(x,V1)))→mul(mul(V0,V1),pow(x,pred(V1))))
                # fire at the root; NNO rules (add('3','2')→'5' etc.) fire at leaves.
                # cata_reduce bottom-up traversal handles this naturally.
                # No .isdigit() calls — Iron Rule compliant.
                _all_rules = self._rewrite_rules + self._nno_rules
                if _all_rules:
                    cata_result = _cata_predict(
                        chain_state.op, chain_state.input_tokens,
                        chain_state.output_tokens,
                        _all_rules, self._arities,
                        post_rules=self._post_rules,
                        nno_atoms=self._atoms,
                    )
                    if cata_result is not None:
                        return cata_result

        # Pass discovered op_atoms — not a hardcoded list
        state = parse_prefix(prefix, op_atoms=self._op_atoms)

        # Levels 0.5–0.7 fire BEFORE Level 0 (n-gram) for eq-format sequences.
        # Structural composition and adjunction give exact (point-mass) answers;
        # the n-gram is a soft-distribution fallback that fires when all
        # structural levels miss.  Keeping structural first prevents a weak
        # n-gram distribution from masking a correct compositional derivation.

        # Level 0.5: FC edge + adjunction-based exact lookup (CT_REFERENCE §4,17).
        # For sequences containing 'eq', extracts op=prefix[0] and
        # input_tuple=prefix[1:eq_idx], then looks up the expected output in:
        #   1. fc_lookup  — direct edge seen during training
        #   2. adj_lookup — adjoint edge seen in training, adjunction infers the answer
        #                   (e.g. pred(4)→3 in training ⟹ succ(3)=4 even if succ(3) is OOD)
        # Falls through if neither lookup has an entry for this (op, input_tuple).
        if (self._fc_lookup or self._adj_lookup) and prefix and 'eq' in prefix:
            fc_result = _fc_and_adj_predict(prefix, self._fc_lookup, self._adj_lookup)
            if fc_result is not None:
                return fc_result

        # Level 0.6: NNO chain prediction with carry/borrow propagation
        # (CT_REFERENCE §19). Fires when Level 0.5 misses — i.e. both the
        # direct and adjoint edges are in the test split.  Uses the partial
        # single-digit step map (from FC edges) to compute multi-digit
        # successor/predecessor without any int() calls.
        if (self._unary_chain_maps or self._unary_carry_maps) and prefix and 'eq' in prefix:
            nno_result = _nno_chain_predict(
                prefix, self._unary_chain_maps, self._unary_carry_maps
            )
            if nno_result is not None:
                return nno_result

        # Level 0.7: Composition engine (CT_REFERENCE §19).
        # Applies discovered BinaryFoldRule rules recursively: given a query
        # (op, inputs), decomposes via the NNO fold rule until a base case or
        # FC-lookup hit is found.  This is categorical composition — no tables,
        # no flood-fill, just the rule applied on demand.
        # Also handles adjoint operators (e.g. sub) by enumerating candidates
        # via the forward (left) op: sub(c,b) = a where add(a,b) = c.
        if (self._fold_rules or self._adj_solve_map) and self._compose_succ_map and prefix and 'eq' in prefix:
            compose_result = _compose_predict(
                prefix,
                self._fc_lookup,
                self._fold_rules,
                self._compose_succ_map,
                self._compose_carry_el,
                self._compose_carry_out,
                self._compose_zero,
                self._compose_cache,
                self._adj_solve_map,
            )
            if compose_result is not None:
                return compose_result

        # Level 0: Direct left-context n-gram lookup.
        # For the prediction position n = len(prefix), the left-context key
        # r{R}|{-R,tok}|...|{-1,tok} is looked up in the left-only Hankel index.
        # This fires for any seen prefix and gives exact counts — same mechanism
        # for natural language and arithmetic.  Falls through on a cache miss
        # (novel left context or shorter-than-r prefix).
        if prefix:
            lkey = HankelCount._left_key(prefix, len(prefix), self._r)
            ngram_dist = self._hankel.get_left_distribution(lkey)
            if ngram_dist:
                return ngram_dist

        # Level 2: Fixpoint iteration + Markov morphism marginalization
        type_dist: dict[ConceptId, float] = {}
        if prefix:
            type_dist = _fixpoint_iteration(
                prefix, self._lattice, self._trans,
                max_iter=_FP_MAX_ITER, eps=_FP_EPS,
                hankel=self._hankel, r=self._r,
            )
            if type_dist:
                morph_dist = _morphism_marginalize(type_dist, self._morphism_graph)
                if morph_dist:
                    return morph_dist

        # Level 3: JSD Kan extension
        if prefix:
            if not type_dist:
                type_dist = _assign_soft(prefix[-1], self._lattice)
            kan_dist = self._kan.predict(type_dist)
            if kan_dist:
                return kan_dist

        # Level 4: Marginal (uniform)
        vocab = self._vocab
        if vocab:
            return {a: 1.0 / len(vocab) for a in vocab}
        return {}

    def generate(
        self,
        prefix: list[str],
        eos: str = "<eos>",
        max_steps: int = 20,
    ) -> list[str]:
        """Greedily generate tokens after the given prefix.

        Maintains a ``WorkingMemory`` (Store comonad) throughout generation
        so that the compression-tree spine tracks each newly emitted token.
        The spine is advanced on every step and exhausted frames are popped
        (compression trigger), providing D3 integration as specified in
        CTKG_ARCHITECTURE.md §Phase 5.
        """
        generated: list[str] = []
        current = list(prefix)

        # Initialise WorkingMemory with one empty TypeDist per prefix position
        wm = WorkingMemory(
            prefix=current,
            type_dists=[{} for _ in current],
            focus=max(0, len(current) - 1),
            op_atoms=self._op_atoms,
            spine=Spine(),
        )

        for _ in range(max_steps):
            dist = self.predict_next(current)
            if not dist:
                break
            next_tok = max(dist, key=lambda x: dist[x])
            generated.append(next_tok)
            current.append(next_tok)

            # Advance WorkingMemory: append token, update spine
            wm = wm.advance(next_tok)

            if next_tok == eos:
                break

        return generated


# ---------------------------------------------------------------------------
# Internal helpers: FC + adjunction lookup (Level 0.5)
# ---------------------------------------------------------------------------

def _fc_and_adj_predict(
    prefix: list[str],
    fc_lookup: dict[tuple, tuple],
    adj_lookup: dict[tuple, tuple],
) -> Optional[dict[str, float]]:
    """Level 0.5: FC direct + adjunction-based exact lookup.

    Parses the prefix as op = prefix[0], input_tuple = prefix[1:eq_idx],
    output_so_far = prefix[eq_idx+1:].  Looks up the expected output in
    fc_lookup first, then adj_lookup (CT_REFERENCE §4: adjunction counit).

    Returns {next_token: 1.0} or {'<eos>': 1.0} if a match is found,
    None otherwise.
    """
    if not prefix or 'eq' not in prefix:
        return None
    try:
        eq_idx = prefix.index('eq')
    except ValueError:
        return None
    if eq_idx == 0:
        return None

    op = prefix[0]
    input_tuple = tuple(prefix[1:eq_idx])
    output_so_far = prefix[eq_idx + 1:]

    expected = fc_lookup.get((op, input_tuple))
    if expected is None:
        expected = adj_lookup.get((op, input_tuple))
    if expected is None:
        return None

    k = len(output_so_far)
    if k < len(expected):
        return {expected[k]: 1.0}
    if k == len(expected):
        return {'<eos>': 1.0}
    return None


# ---------------------------------------------------------------------------
# Internal helpers: NNO chain prediction (Level 0.6)
# ---------------------------------------------------------------------------

def _nno_chain_predict(
    prefix: list[str],
    unary_chain_maps: dict[str, dict[str, str]],
    unary_carry_maps: dict[str, tuple],
) -> Optional[dict[str, float]]:
    """Level 0.6: NNO chain prediction (CT_REFERENCE §19).

    Fires when both the direct FC edge and the adjoint edge are absent
    (i.e. both succ(n) and pred(n+1) are in the test split).  Uses the
    partial single-digit step map discovered from training data to compute
    multi-digit successor/predecessor via carry/borrow propagation without
    any int() calls.

    unary_chain_predict(step_map, carry_element, carry_out, digits, inverse=False)
    expects the FORWARD step map (succ: {'0':'1',...,'8':'9'}) in both modes.
    When inverse=True it internally inverts step_map to get the pred map.
    """
    if not prefix or 'eq' not in prefix:
        return None
    try:
        eq_idx = prefix.index('eq')
    except ValueError:
        return None
    if eq_idx == 0:
        return None

    op = prefix[0]
    input_digits = list(prefix[1:eq_idx])
    output_so_far = list(prefix[eq_idx + 1:])

    if not input_digits:
        return None

    # Case 1: op has its own step_map AND carry_map (e.g. succ).
    step_map = unary_chain_maps.get(op)
    carry_info = unary_carry_maps.get(op)
    if step_map is not None and carry_info is not None:
        carry_element, carry_out = carry_info
        full_output = unary_chain_predict(
            step_map, carry_element, carry_out, input_digits, inverse=False
        )
        if full_output is not None:
            k = len(output_so_far)
            if k < len(full_output):
                return {full_output[k]: 1.0}
            if k == len(full_output):
                return {'<eos>': 1.0}
        return None

    # Case 2: op lacks a carry_map but may be the adjoint of a chain op
    # (e.g. pred is the adjoint of succ).  Find a forward op that has a
    # carry_map and whose single-digit step map is the inverse of op's map.
    for fwd_op, fwd_step in unary_chain_maps.items():
        fwd_carry = unary_carry_maps.get(fwd_op)
        if fwd_carry is None:
            continue
        # Verify that op's edges are consistent with being the adjoint:
        # pred's single-digit edges should be the inverse of succ's.
        op_step = unary_chain_maps.get(op)
        if op_step is None:
            continue
        inv_fwd = {v: k for k, v in fwd_step.items()}
        if op_step != inv_fwd:
            # Not a perfect match (may be partial due to 80/20 split);
            # accept if at least one entry matches.
            if not any(op_step.get(k) == v for k, v in inv_fwd.items()):
                continue
        # Use the FORWARD step_map with inverse=True.
        carry_element, carry_out = fwd_carry
        full_output = unary_chain_predict(
            fwd_step, carry_element, carry_out, input_digits, inverse=True
        )
        if full_output is not None:
            k = len(output_so_far)
            if k < len(full_output):
                return {full_output[k]: 1.0}
            if k == len(full_output):
                return {'<eos>': 1.0}
        return None

    return None


# ---------------------------------------------------------------------------
# Internal helpers: soft type assignment
# ---------------------------------------------------------------------------

def _assign_soft(token: str, lattice: ConceptLattice) -> dict[ConceptId, float]:
    """Soft type distribution for a single token.

    Primary: uses concept intent_weights (P(atom | concept)).
    Fallback: centroid_vector lookup if no concept has the token in intent.
    """
    weights: dict[int, float] = {}
    for c in lattice.concepts:
        w = c.intent_weights.get(token, 0.0)
        if w > 0.0:
            weights[c.concept_id] = w

    if not weights:
        atom_idx = {a: i for i, a in enumerate(lattice.atoms)}
        tok_idx = atom_idx.get(token)
        if tok_idx is not None:
            for c in lattice.concepts:
                if tok_idx < len(c.centroid_vector):
                    w = float(c.centroid_vector[tok_idx])
                    if w > 0.0:
                        weights[c.concept_id] = w

    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {c_id: w / total for c_id, w in weights.items()}


# ---------------------------------------------------------------------------
# Internal helpers: fixpoint iteration (architecture §Prediction step 3)
# ---------------------------------------------------------------------------

def _build_transition(mg: MorphismGraph) -> dict[int, dict[int, float]]:
    """Row-normalised transition matrix from MorphismGraph."""
    raw: dict[int, dict[int, float]] = {}
    for m in mg.morphisms(include_identity=False):
        src_obj = mg.object_by_id(m.source)
        tgt_obj = mg.object_by_id(m.target)
        if src_obj is None or tgt_obj is None:
            continue
        src_id = src_obj.concept.concept_id
        tgt_id = tgt_obj.concept.concept_id
        w = float(m.evidence_count) * math.exp(min(m.confidence, 500.0))
        if src_id not in raw:
            raw[src_id] = {}
        raw[src_id][tgt_id] = raw[src_id].get(tgt_id, 0.0) + w

    trans: dict[int, dict[int, float]] = {}
    for src_id, row in raw.items():
        total = sum(row.values())
        if total > 0.0:
            trans[src_id] = {tgt_id: w / total for tgt_id, w in row.items()}
    return trans


def _l1_dist(
    a: dict[ConceptId, float],
    b: dict[ConceptId, float],
) -> float:
    """L1 distance between two sparse probability distributions."""
    all_keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in all_keys)


def _best_atom_seq(
    prefix: list[str],
    T: list[dict[ConceptId, float]],
    lattice: ConceptLattice,
) -> list[str]:
    """For each prefix position compute the most-likely atom given T[i].

    Used by the presheaf restrict step to build a typed neighbourhood key
    for the HankelCount query.  Falls back to the raw prefix token when the
    type distribution is empty or no atom has positive weight.

    Parameters
    ----------
    prefix:
        The raw token sequence.
    T:
        Current type distributions: T[i] = dict{concept_id: probability}.
    lattice:
        ConceptLattice (provides intent_weights per concept).

    Returns
    -------
    List of the most-likely atom at each position (same length as prefix).
    """
    result: list[str] = []
    for i, tok in enumerate(prefix):
        td = T[i] if i < len(T) else {}
        if not td:
            result.append(tok)
            continue
        atom_scores: dict[str, float] = {}
        for c in lattice.concepts:
            p_c = td.get(c.concept_id, 0.0)
            if p_c <= 0.0:
                continue
            for atom, w in c.intent_weights.items():
                atom_scores[atom] = atom_scores.get(atom, 0.0) + p_c * w
        if atom_scores:
            result.append(max(atom_scores, key=lambda a: atom_scores[a]))
        else:
            result.append(tok)
    return result


def _fixpoint_iteration(
    prefix: list[str],
    lattice: ConceptLattice,
    trans: dict[int, dict[int, float]],
    max_iter: int = _FP_MAX_ITER,
    eps: float = _FP_EPS,
    hankel: Optional["HankelCount"] = None,
    r: int = 1,
) -> dict[ConceptId, float]:
    """Iterate the presheaf update to convergence; return T_last[last_position].

    Architecture §Prediction step 3:
        T⁰(i)      = initial soft type assignment from intent_weights

        Presheaf restrict (when hankel is provided):
          T^{n+1}(i) = normalise(Σ_{atom} P(atom | N_r(T^n, i)) * _assign_soft(atom))
          where N_r(T^n, i) is the typed neighbourhood key at position i built
          from the most-likely atom at each neighbour position (from T^n).
          This is the sheaf restriction morphism ρ^{i}_{V} applied to T^n.

        Fallback (no hankel): Bayesian product with transition belief:
          T^{n+1}(i) = normalise(T^n(i) * (trans @ T^n(i-1)))

        Halt when max_i ||T^{n+1}(i) − T^n(i)||₁ < ε

    Sheaf obstruction detection: cache the last K snapshots (as hashes);
    if the current snapshot matches any cached snapshot, declare a cycle.
    The iteration still returns its best estimate but the cycle is flagged
    (currently logged; in future surfaced as a metadata field).

    Returns
    -------
    Type distribution at the last position (used by morphism marginalization).
    Empty dict if nothing can be assigned.
    """
    if not prefix:
        return {}

    n = len(prefix)

    # T[i] = type distribution at position i
    T: list[dict[ConceptId, float]] = [_assign_soft(tok, lattice) for tok in prefix]

    if not any(T):
        return {}

    # Snapshot cache for cycle detection (L1-hash of full T vector)
    snapshots: list[tuple[dict[ConceptId, float], ...]] = []

    for iteration in range(max_iter):
        T_new: list[dict[ConceptId, float]] = []
        max_delta = 0.0

        # Presheaf restrict: build typed neighbourhood keys once per iteration
        best_atoms: list[str] = []
        if hankel is not None:
            best_atoms = _best_atom_seq(prefix, T, lattice)

        for i in range(n):
            # -------------------------------------------------------
            # Presheaf restrict (architecture §Prediction step 3)
            # -------------------------------------------------------
            if hankel is not None and best_atoms:
                # Use left-only key so prediction (right context unknown) matches
                # training contexts.  Bidirectional keys never match at prediction
                # time because the right tokens are not yet emitted.
                key = HankelCount._left_key(best_atoms, i, r)
                atom_dist = hankel.get_left_distribution(key)
                if atom_dist:
                    # Convert atom distribution → type distribution
                    restricted: dict[int, float] = {}
                    for atom, prob in atom_dist.items():
                        for c_id, c_prob in _assign_soft(atom, lattice).items():
                            restricted[c_id] = restricted.get(c_id, 0.0) + prob * c_prob
                    total = sum(restricted.values())
                    if total > 0.0:
                        new_ti = {c_id: w / total for c_id, w in restricted.items()}
                        delta = _l1_dist(T[i], new_ti)
                        if delta > max_delta:
                            max_delta = delta
                        T_new.append(new_ti)
                        continue
            # -------------------------------------------------------
            # Fallback: Bayesian product with transition belief
            # -------------------------------------------------------
            prev = T[i - 1] if i > 0 else {}

            # Transition step: T^n(i-1) → T^n(i) via morphism transitions
            if prev:
                transitioned: dict[int, float] = {}
                for c_src, w_src in prev.items():
                    if c_src not in trans:
                        continue
                    for c_tgt, t_prob in trans[c_src].items():
                        transitioned[c_tgt] = (
                            transitioned.get(c_tgt, 0.0) + w_src * t_prob
                        )
            else:
                transitioned = {}

            # Observation step: combine transition belief with token evidence
            obs = T[i]   # current type assignment for this token
            if transitioned and obs:
                combined: dict[int, float] = {}
                for c_id in set(transitioned) | set(obs):
                    v = transitioned.get(c_id, 0.0) * obs.get(c_id, 0.0)
                    if v > 0.0:
                        combined[c_id] = v
                if not combined:
                    # Observation entirely inconsistent: fall back to obs
                    combined = dict(obs)
            elif transitioned:
                combined = dict(transitioned)
            else:
                combined = dict(obs)

            # Normalise
            total = sum(combined.values())
            if total > 0.0:
                new_ti = {c_id: w / total for c_id, w in combined.items()}
            else:
                new_ti = dict(T[i])

            # Track max L1 delta
            delta = _l1_dist(T[i], new_ti)
            if delta > max_delta:
                max_delta = delta

            T_new.append(new_ti)

        T = T_new

        # Convergence check (architecture: halt when max_i ||…||₁ < ε)
        if max_delta < eps:
            break

        # Cycle detection (sheaf obstruction): compare against cached snapshots
        snapshot = tuple(T)
        for prev_snap in snapshots:
            if all(
                _l1_dist(prev_snap[i], T[i]) < eps
                for i in range(n)
            ):
                # Cycling detected — ambiguous sheaf section
                # Return best estimate (last T) and stop
                break
        else:
            # No cycle found; push snapshot (keep last K)
            snapshots.append(snapshot)
            if len(snapshots) > _FP_CYCLE_K:
                snapshots.pop(0)
            continue
        break  # cycle detected

    return T[-1] if T else {}


# ---------------------------------------------------------------------------
# Internal helpers: morphism marginalization
# ---------------------------------------------------------------------------

def _morphism_marginalize(
    type_dist: dict[ConceptId, float],
    mg: MorphismGraph,
) -> dict[str, float]:
    """Markov category morphism marginalisation.

    P(next_atom) = Σ_c P(c) * Σ_{f:c→d} evidence(f)*exp(conf(f)) * intent(d, atom)
    """
    result: dict[str, float] = {}

    for obj in mg.objects():
        c_id = obj.concept.concept_id
        p_c = type_dist.get(c_id, 0.0)
        if p_c <= 0.0:
            continue

        for m in mg.out_morphisms(obj.obj_id, include_identity=False):
            tgt_obj = mg.object_by_id(m.target)
            if tgt_obj is None:
                continue
            w = p_c * float(m.evidence_count) * math.exp(min(m.confidence, 500.0))
            for atom, aw in tgt_obj.concept.intent_weights.items():
                result[atom] = result.get(atom, 0.0) + w * aw

    if not result:
        return {}
    total = sum(result.values())
    if total <= 0.0:
        return {}
    return {a: v / total for a, v in result.items()}


def _chain_predict(
    rule: ChainRule,
    input_tokens: list[str],
    output_tokens_so_far: list[str],
    use_eq_table: bool = False,
) -> Optional[dict[str, float]]:
    """Return {next_token: 1.0} or {'<eos>': 1.0} from chain rule, or None.

    Looks up the input_tokens tuple in the appropriate table:
    - use_eq_table=True  → eq_table  (eq-format: d/dx, int, eval, …)
    - use_eq_table=False → chain_table (step/ans trace format)
    Falls through (returns None) if the input tuple is not in the table.
    """
    key = tuple(input_tokens)
    table = rule.eq_table if use_eq_table else rule.chain_table
    full_output = table.get(key)
    if full_output is None:
        return None
    k = len(output_tokens_so_far)
    if k < len(full_output):
        return {full_output[k]: 1.0}
    if k == len(full_output):
        return {"<eos>": 1.0}
    return None


# ---------------------------------------------------------------------------
# Composition engine (Level 0.7) — CT_REFERENCE §19
# ---------------------------------------------------------------------------

def _compose_predict(
    prefix: list[str],
    fc_lookup: dict[tuple, tuple],
    fold_rules: dict[str, "BinaryFoldRule"],
    succ_map: dict[str, str],
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
    cache: dict[tuple, tuple],
    adj_solve_map: Optional[dict[str, tuple]] = None,
) -> Optional[dict[str, float]]:
    """Level 0.7 entry point: parse prefix, call _compose, return point mass.

    First tries direct fold-rule composition (_compose).  If that fails (no
    rule, or op is an adjoint op like sub), falls back to adjunction-mediated
    search (_compose_adjoint_search): for sub(c, b) = a, enumerates a from
    zero via succ_map and applies the forward op (add) until add(a, b) = c.
    """
    if not prefix or 'eq' not in prefix:
        return None
    eq_idx = prefix.index('eq')
    if eq_idx == 0:
        return None
    op = prefix[0]
    input_tuple = tuple(prefix[1:eq_idx])
    output_so_far = prefix[eq_idx + 1:]

    if not input_tuple:
        return None

    # --- Try fold-rule composition first ---
    # Only fire when input_tuple has exactly 2 tokens (both args single-digit).
    # For multi-digit inputs (e.g. sub(11, 4) → ('1','1','4')), the flat
    # token tuple can't be split into args without arity information, so we
    # skip the fold rule and let the adjunction search handle it instead.
    result = None
    if len(input_tuple) == 2:
        args: tuple = tuple((t,) for t in input_tuple)
        result = _compose(op, args, fc_lookup, fold_rules,
                          succ_map, carry_el, carry_out, zero_digit, cache)

    # --- Adjunction-mediated search (FALLBACK for inverse ops: sub, div, …) ---
    # Fires only when _compose returned None: either no fold rule, or the
    # input had multi-digit tokens we couldn't split.
    # Enumerates candidates via the left (forward) op without any int() calls.
    if result is None and adj_solve_map and op in adj_solve_map:
        result = _compose_adjoint_search(
            op, input_tuple, adj_solve_map[op],
            fc_lookup, fold_rules, succ_map, carry_el, carry_out, zero_digit, cache,
        )

    if result is None:
        return None

    k = len(output_so_far)
    if k < len(result):
        return {result[k]: 1.0}
    if k == len(result):
        return {'<eos>': 1.0}
    return None


def _compose_adjoint_search(
    op: str,
    input_flat: tuple,
    adj_info: tuple,
    fc_lookup: dict[tuple, tuple],
    fold_rules: dict[str, "BinaryFoldRule"],
    succ_map: dict[str, str],
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
    cache: dict[tuple, tuple],
    b_width: int = 1,
) -> Optional[tuple]:
    """Solve op(input_flat) via adjunction + composition.

    For an adjunction left_op ⊣ op (preserved_position p):
        op(c_flat + b_flat) = a  iff  left_op(a_flat + b_flat) = c_flat

    where b_flat is the preserved argument (b_width tokens at the end of
    input_flat) and c_flat is the target output (all preceding tokens).

    Enumerates candidate a_flat from zero_digit via succ_map, computes
    left_op(a, b) using the composition engine, and returns the first a
    where the result equals c_flat.  Handles multi-digit c_flat correctly
    (e.g. sub(11, 4) = 7 iff add(7, 4) = 11).

    Returns the answer as a flat tuple (e.g. ('7',)), or None if not found.
    """
    fwd_op, _preserved_pos = adj_info

    # b = last b_width tokens (preserved arg of the forward op)
    # c = all preceding tokens (target output of the forward op)
    if len(input_flat) < b_width + 1:
        return None
    b_tuple: tuple = input_flat[-b_width:]  # e.g. ('4',) for sub(11, 4)
    c_tuple: tuple = input_flat[:-b_width]  # e.g. ('1', '1') for sub(11, 4)

    # Enumerate a from zero_digit upward, compute fwd_op(a, b), find match
    a_tuple: tuple = (zero_digit,)
    seen: set = set()
    while a_tuple not in seen:
        seen.add(a_tuple)
        # Compute fwd_op(a, b) via composition
        fwd_result = _compose(
            fwd_op, (a_tuple, b_tuple), fc_lookup, fold_rules,
            succ_map, carry_el, carry_out, zero_digit, cache,
        )
        if fwd_result == c_tuple:
            return a_tuple
        # Advance a by one step (succ)
        next_a = unary_chain_predict(succ_map, carry_el, carry_out, list(a_tuple), inverse=False)
        if next_a is None:
            break
        next_a_tuple = tuple(next_a)
        # Stop if we've wrapped around (succ carries into multi-digit)
        if len(next_a_tuple) > len(c_tuple):
            break
        a_tuple = next_a_tuple

    return None


def _compose(
    op: str,
    args: tuple,           # tuple of tuples, each arg is a tuple[str,...]
    fc_lookup: dict[tuple, tuple],
    fold_rules: dict[str, "BinaryFoldRule"],
    succ_map: dict[str, str],
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
    cache: dict[tuple, tuple],
    depth: int = 0,
) -> Optional[tuple[str, ...]]:
    """Recursively apply the NNO fold rule for op(args).

    Base cases (termination):
      1. Direct FC lookup hit.
      2. Induction variable == zero_digit → return base result.

    Inductive step:
      pred(ind_arg), recurse, apply step function.

    Multi-digit induction variable:
      Uses unary_chain_predict (pred on multi-digit tuple) to step down.
      Terminates when the induction variable reaches (zero_digit,).
    """
    if depth > 200:   # Safety valve; normal arithmetic terminates well within this
        return None

    # Cache key includes argument boundaries to avoid ambiguity.
    # e.g. add(('3','5'),('5',)) ≠ add(('3',),('5','5')) — flat key ('3','5','5') is
    # the same for both, but (op,)+args is unambiguous.
    cache_key = (op,) + args

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # FC direct lookup uses flat key (only valid when all args are single-token)
    all_single = all(len(a) == 1 for a in args)
    if all_single:
        flat_inputs = tuple(t for arg in args for t in arg)
        fc_key = (op, flat_inputs)
        direct = fc_lookup.get(fc_key)
        if direct is not None:
            cache[cache_key] = direct
            return direct

    rule = fold_rules.get(op)
    if rule is None:
        return None   # No rule known; can't compose

    ind = rule.induction_arg
    if ind >= len(args):
        return None

    ind_arg: tuple[str, ...] = args[ind]
    other_arg: tuple[str, ...] = args[1 - ind] if len(args) == 2 else args[0]

    # --- Base case ---
    if ind_arg == (zero_digit,):
        if rule.base_fixed is None:
            # Identity: result = other_arg (e.g. add(0, m) = m)
            result = other_arg
        else:
            result = (rule.base_fixed,)
        cache[cache_key] = result
        return result

    # --- Inductive step: pred(ind_arg) ---
    pred_list = unary_chain_predict(
        succ_map, carry_el, carry_out, list(ind_arg), inverse=True
    )
    if pred_list is None:
        return None
    pred_ind = tuple(pred_list)

    # Recurse with pred(ind_arg)
    sub_args = list(args)
    sub_args[ind] = pred_ind
    sub_result = _compose(op, tuple(sub_args), fc_lookup, fold_rules,
                          succ_map, carry_el, carry_out, zero_digit, cache, depth + 1)
    if sub_result is None:
        return None

    # --- Apply step ---
    if rule.step_op is None:
        # Step is succ (forward) or pred (inverse), depending on step_inverse
        inverse_step = getattr(rule, 'step_inverse', False)
        stepped = unary_chain_predict(
            succ_map, carry_el, carry_out, list(sub_result), inverse=inverse_step
        )
        result = tuple(stepped) if stepped is not None else None
    else:
        # Step is a binary op: step_op(sub_result, other_arg)
        step_args = (sub_result, other_arg)
        result = _compose(rule.step_op, step_args, fc_lookup, fold_rules,
                          succ_map, carry_el, carry_out, zero_digit, cache, depth + 1)

    if result is not None:
        cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Phase V helpers: compound-token merging/splitting for multi-digit outputs
# ---------------------------------------------------------------------------


def _merge_digit_runs(toks: list[str], nno_atoms: frozenset) -> list[str]:
    """Merge consecutive single-char NNO tokens into one compound atom.

    Example: ['1', '2'] → ['12'] (coefficient 12 in derivative output).
    Only merges runs of single-character tokens that are all in nno_atoms.
    Multi-character tokens and non-NNO tokens are left unchanged.
    """
    result: list[str] = []
    run: list[str] = []
    for tok in toks:
        if len(tok) == 1 and tok in nno_atoms:
            run.append(tok)
        else:
            if run:
                result.append(''.join(run))
                run = []
            result.append(tok)
    if run:
        result.append(''.join(run))
    return result


def _split_compound(tok: str, nno_atoms: frozenset) -> list[str]:
    """Split a compound NNO token back to individual single-char tokens.

    Example: '12' → ['1', '2'] (when nno_atoms contains '1' and '2').
    Returns [tok] unchanged if tok is already single-char or not all-NNO.
    """
    if len(tok) <= 1:
        return [tok]
    if all(c in nno_atoms for c in tok):
        return list(tok)
    return [tok]


# ---------------------------------------------------------------------------
# Level 1c-trace: per-role prediction for trace-format ops (RelationStore)
# ---------------------------------------------------------------------------


def _trace_cata_predict(
    op: str,
    input_tokens: list[str],
    output_tokens_so_far: list[str],
    step_rules: list,
    ans_rules: list,
    arities: dict,
    post_rules: Optional[list],
    nno_atoms: frozenset,
) -> Optional[dict]:
    """Level 1c-trace: predict the next token in a trace-format (step/ans) sequence.

    Trace format: [step, step_val..., ans, ans_val..., <eos>]

    This function computes the full expected trace by:
      1. Applying step_rules to the input tree → step_result tokens
      2. Applying ans_rules to the input tree → ans_result tokens
      3. Building expected_trace = ['step'] + step_result + ['ans'] + ans_result
      4. Returning a point mass for position k = len(output_tokens_so_far)

    Handles the output_tokens_so_far correctly: it includes the delimiter tokens
    ('step', 'ans') as part of the sequence, so k indexes into the full trace.

    Returns None if either rule fails to fire or the partial output contradicts
    the expected trace.
    """
    from experiments.symbolic_ai_v2.ctkg.core.expr_parser import parse, unparse
    from experiments.symbolic_ai_v2.ctkg.core.rewrite import cata_reduce

    seq = [op] + list(input_tokens)
    inp_tree = parse(seq, arities)
    if inp_tree is None:
        return None

    _max_steps = max(10000, 200 * max(len(step_rules), len(ans_rules), 1))

    # Compute step result
    step_tree = cata_reduce(inp_tree, step_rules, max_steps=_max_steps)
    if step_tree == inp_tree:
        return None  # step rule did not fire

    if post_rules:
        step_tree = cata_reduce(step_tree, post_rules, max_steps=_max_steps)

    step_toks = unparse(step_tree)
    if nno_atoms:
        step_toks = _expand_compound(step_toks, nno_atoms)

    # Compute ans result
    ans_tree = cata_reduce(inp_tree, ans_rules, max_steps=_max_steps)
    if ans_tree == inp_tree:
        return None  # ans rule did not fire

    if post_rules:
        ans_tree = cata_reduce(ans_tree, post_rules, max_steps=_max_steps)

    ans_toks = unparse(ans_tree)
    if nno_atoms:
        ans_toks = _expand_compound(ans_toks, nno_atoms)

    # Build the full expected trace
    expected = ['step'] + list(step_toks) + ['ans'] + list(ans_toks)

    k = len(output_tokens_so_far)

    # Verify consistency with already-generated tokens
    if k > 0 and list(expected[:k]) != list(output_tokens_so_far):
        return None  # contradiction

    if k < len(expected):
        return {expected[k]: 1.0}
    if k == len(expected):
        return {'<eos>': 1.0}
    return None


def _expand_compound(tokens: list[str], nno_atoms: frozenset[str]) -> list[str]:
    """Expand compound NNO tokens back to individual chars.

    E.g. ['12', 'x'] → ['1', '2', 'x'] when '1' and '2' are in nno_atoms.
    Tokens whose chars are NOT all in nno_atoms are left unchanged.
    """
    result: list[str] = []
    for tok in tokens:
        if len(tok) > 1 and all(c in nno_atoms for c in tok):
            result.extend(list(tok))
        else:
            result.append(tok)
    return result


# ---------------------------------------------------------------------------
# Level 1c: cata_reduce prediction (CT_REFERENCE §19 — Iron Rule compliant)
# ---------------------------------------------------------------------------


def _cata_predict(
    op: str,
    input_tokens: list[str],
    output_tokens_so_far: list[str],
    rewrite_rules: list,
    arities: dict,
    post_rules: Optional[list] = None,
    nno_atoms: frozenset = frozenset(),
) -> Optional[dict]:
    """Level 1c: apply discovered RewriteRules to predict the next output token.

    Parses [op] + input_tokens as a prefix expression, applies cata_reduce,
    unparses the result, then returns a point mass for the k-th output token
    (where k = len(output_tokens_so_far)).

    post_rules: applied after the main cata_reduce (e.g. pow(x,1)→x inverse norm).
    nno_atoms: used to expand compound tokens in out_toks back to single chars.

    Returns None when:
    - The input cannot be parsed (unknown arity or multi-token atom sequences).
    - No rule fires (output tree == input tree).
    - The output tree's unparsed tokens conflict with already-generated tokens.
    """
    from experiments.symbolic_ai_v2.ctkg.core.expr_parser import parse, unparse
    from experiments.symbolic_ai_v2.ctkg.core.rewrite import cata_reduce

    seq = [op] + list(input_tokens)
    inp_tree = parse(seq, arities)
    if inp_tree is None:
        return None

    # Scale max_steps with rule set size: each node tries all rules once per pass,
    # plus one restart depth.  200 steps/rule gives enough headroom for depth-5 trees.
    _max_steps = max(10000, 200 * len(rewrite_rules))
    out_tree = cata_reduce(inp_tree, rewrite_rules, max_steps=_max_steps)
    if out_tree == inp_tree:
        return None  # no rule fired

    # Apply post-normalization (e.g. pow(x,1)→x) to match corpus surface form
    if post_rules:
        out_tree = cata_reduce(out_tree, post_rules, max_steps=_max_steps)

    out_toks = unparse(out_tree)

    # Expand any compound NNO tokens back to individual chars to match corpus
    if nno_atoms:
        expanded: list[str] = []
        for t in out_toks:
            expanded.extend(_split_compound(t, nno_atoms))
        out_toks = expanded

    k = len(output_tokens_so_far)
    if k > 0 and list(out_toks[:k]) != list(output_tokens_so_far):
        return None  # generated tokens contradict this rule
    if k < len(out_toks):
        return {out_toks[k]: 1.0}
    if k == len(out_toks):
        return {'<eos>': 1.0}
    return None

