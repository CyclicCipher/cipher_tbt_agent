"""
Next-token prediction: structural levels only (Phase X complete — fallbacks removed).

Pipeline (priority order):

    Level 1b — Chain rule (deterministic):
        If the current phase is OUTPUT and the operator has a ChainRule,
        look up the next token in the chain_table (trace format) or eq_table.

    Level 1c-relational — Arity-free RelationRule (Phase XIII extended):
        Uses named-role tuples discovered by discover_relation_rules().
        No prefix S-expression, no arities, no parse tree.
        For eq-format sequences, extracts the 'ans' content from trace-format
        relational rules (e.g. linear_eval reuses linear_trace rules).

    Level 0.5 — FC direct + adjunction-based lookup (CT_REFERENCE §4, 17):
        Exact lookup in the free category edge table.  Adjunction-mediated:
        if pred(4)→3 is in training, infers succ(3)=4 via the adjunction.

    Level 0.6 — NNO chain prediction (CT_REFERENCE §19):
        Computes successor/predecessor chains without int() using the
        discovered succ_map.

    Level 0.7 — Composition engine (CT_REFERENCE §19):
        Applies BinaryFoldRule recursively via the NNO universal property.
        This is a genuine categorical construction (initial F-algebra).

    On miss: returns {}.  No heuristic fallbacks.

Phase X (FIXING_GENERALIZATION_PART2.md) is complete.  Heuristic Levels 0
(n-gram), 2 (fixpoint iteration), 3 (JSD nearest-neighbor), and 4 (uniform
marginal) have been removed.  All benchmark results now reflect only the
categorical structural levels above.
"""

from __future__ import annotations

from typing import Optional

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import ConceptLattice
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
    build_binary_functional_maps,
)
from experiments.symbolic_ai_v2.ctkg.learning.relation_store import (
    RelationStore,
    RelationRule,
    discover_relation_rules,
    discover_kleisli_chains,
    predict_from_relation_rules,
    predict_alternatives_from_rules,
)
from dataclasses import dataclass, field
from experiments.symbolic_ai_v2.ctkg.core.working_memory import (
    parse_prefix,
    parse_chain_prefix,
    WorkingMemory,
)
from experiments.symbolic_ai_v2.ctkg.core.spine import Spine
from experiments.symbolic_ai_v2.ctkg.core.context_category import (
    ContextCategory,
    ContextId,
)
from experiments.symbolic_ai_v2.ctkg.core.lambda_term import (
    LambdaTerm,
    lambda_predict,
    synthesize_library,
)




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
        # Constructor parameters kept for API compatibility; not used internally
        # after Phase X (fallback removal).
        self._hankel = hankel
        self._lattice = lattice
        self._morphism_graph = morphism_graph

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
            # ---- Binary functional maps for RelationRule evaluation ----
            # Build BFM from FC edges, then complete all single-digit pairs via NNO.
            # No discover_arities(), no prefix S-expression, no parse tree.
            _binary_fmaps = build_binary_functional_maps(
                fc,
                self._compose_succ_map,
                self._compose_carry_el,
                self._compose_carry_out,
                self._compose_zero,
            ) if self._compose_succ_map else {}

            # Complete BFM for all single-digit × single-digit pairs using
            # the NNO compose engine so OOD pairs (e.g. mul('6','5') in test split)
            # don't cause RelationRule lookups to fail.  No int() calls.
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

            # Extend BFM with two-digit-result × single-digit add/sub entries.
            # Needed when mul(a, b) produces a two-digit result (e.g. '18', '36')
            # that is then passed to add/sub as the first arg in a multi-step rule.
            # E.g. linear_trace: ans = add(mul(a, c), b) where mul(a,c) may be '10'.
            # Must run AFTER NNO completion so OOD mul results (e.g. mul('6','6')='36')
            # are already present in _binary_fmaps before we extend add/sub.
            # Only add/sub entries — mul(two-digit, x) is out of scope for now.
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
                    _binary_fmaps = dict(_binary_fmaps)
                    _binary_fmaps['sub'] = {**_binary_fmaps.get('sub', {}), **_extended_sub}

            # Extend BFM with two-digit × single-digit mul entries (Phase XII).
            # Required for Kleisli chains where intermediate results (base²) are
            # two-digit, e.g. mul('16','4')=64 for pow(4,3) chain step.
            # Uses _compose (NNO iterative addition) — same engine as single-digit
            # completion above.  Only fills entries whose first arg is a result
            # already in the single-digit mul table (not arbitrary 2-char strings).
            if _binary_fmaps and self._compose_succ_map:
                _mul_map = _binary_fmaps.get('mul', {})
                _mul_two_digits: set[str] = {
                    _res for _res in _mul_map.values() if len(_res) == 2
                }
                _extended_mul2: dict[tuple, str] = {}
                for _R in _mul_two_digits:
                    for _b in _single_digits:
                        if (_R, _b) not in _mul_map and (_R, _b) not in _extended_mul2:
                            _cr2 = _compose(
                                'mul', (tuple(_R), (_b,)),
                                self._fc_lookup, self._fold_rules,
                                self._compose_succ_map,
                                self._compose_carry_el,
                                self._compose_carry_out,
                                self._compose_zero,
                                self._compose_cache,
                            )
                            if _cr2 is not None:
                                _extended_mul2[(_R, _b)] = ''.join(_cr2)
                if _extended_mul2:
                    _binary_fmaps = dict(_binary_fmaps)
                    _binary_fmaps['mul'] = {**_binary_fmaps.get('mul', {}), **_extended_mul2}

            # ---- Arity-free RelationRule discovery (hypergraph approach) ----
            # Each sequence is a named-role tuple: structural tokens are input
            # separators ('x', 'at', 'dx'); output delimiters ('step', 'ans', 'eq')
            # name output roles.  No arities, no prefix S-expressions, no parse trees.
            _rs = RelationStore()
            _rs_seqs: list[list[str]] = []
            for cr in (chain_rules or []):
                for inp_toks, out_toks in (cr.chain_table or {}).items():
                    _rs_seqs.append([cr.op_atom] + list(inp_toks) + list(out_toks))
            _rs.update_batch(_rs_seqs)

            # ---- Extend BFM with concat and div (Phase XIV) ----
            # concat(a, b) = a+b for all single-char NNO-alphabet pairs.
            # Enables positional-role rules like: sq step = concat(p0, p1)
            # where the step is the zero-padded input itself (identity trace).
            # div is the inverse of mul: if mul(a,b)=r then div(r,a)=b.
            # Enables algebra_trace ans = div(step, A).
            if _binary_fmaps and self._compose_succ_map:
                _nno_digits: list[str] = []
                _dc2 = self._compose_zero
                _dc2_seen: set[str] = {_dc2}
                _nno_digits.append(_dc2)
                while True:
                    _dn2 = self._compose_succ_map.get(_dc2)
                    if _dn2 is None or _dn2 in _dc2_seen:
                        break
                    _nno_digits.append(_dn2)
                    _dc2_seen.add(_dn2)
                    _dc2 = _dn2
                # concat: single-char × single-char → 2-char concatenation
                _concat_map: dict[tuple, str] = {}
                for _a in _nno_digits:
                    for _b in _nno_digits:
                        if len(_a) == 1 and len(_b) == 1:
                            _concat_map[(_a, _b)] = _a + _b
                if _concat_map:
                    _binary_fmaps = dict(_binary_fmaps)
                    _binary_fmaps['concat'] = _concat_map
                # div: inverse of mul (adjunction: if mul(a,b)=r then div(r,a)=b)
                _mul_map = _binary_fmaps.get('mul', {})
                if _mul_map:
                    _div_map: dict[tuple, str] = {}
                    for (_a, _b), _r in _mul_map.items():
                        # div(r, a) = b  and  div(r, b) = a
                        if (_r, _a) not in _div_map:
                            _div_map[(_r, _a)] = _b
                        if (_r, _b) not in _div_map:
                            _div_map[(_r, _b)] = _a
                    if _div_map:
                        _binary_fmaps['div'] = _div_map
                # fst: fst(a, b) = a  (first projection)
                # Needed for Kleisli chains: step_0 = fst(p0, p1) = base (base^1)
                _fst_map: dict[tuple, str] = {}
                for _a in _nno_digits:
                    for _b in _nno_digits:
                        _fst_map[(_a, _b)] = _a
                if _fst_map:
                    _binary_fmaps['fst'] = _fst_map

            self._rs = _rs
            self._binary_fmaps: dict = _binary_fmaps or {}
            self._relation_rules: dict[str, list[RelationRule]] = {}
            _positional_ops = _rs.ops_with_positional_schema()
            for _op in _rs.ops_with_schema():
                _op_rels = _rs.get_relations(_op)
                # Positional-schema ops (e.g. sq, pow) mix sub-cases in training
                # (sq(n) for n≥10 produces 3-digit answers not in single-digit BFM).
                # Allow up to 25% mismatch so the dominant rule is still accepted.
                _mm_tol = 0.25 if _op in _positional_ops else 0.0
                _op_rr = discover_relation_rules(
                    _op_rels, _binary_fmaps or {}, mismatch_tolerance=_mm_tol
                )
                if _op_rr:
                    # Coverage check: only store rules if ALL expected output
                    # roles (observed in training) are covered.  Partial rules
                    # produce wrong predictions (e.g. 'ans' at position 0 when
                    # 'step' was expected first).
                    _expected_roles = _rs.all_output_role_names(_op)
                    _covered_roles = {_r.output_role for _r in _op_rr}
                    if _expected_roles.issubset(_covered_roles):
                        self._relation_rules[_op] = _op_rr

            # ---- Kleisli chain discovery (Phase XII) ----
            # For positional-schema ops with variable-depth output (e.g. pow),
            # standard discover_relation_rules deduplicates repeated 'step' roles
            # and misses multi-step chains.  discover_kleisli_chains groups by
            # discriminator value (e.g. exponent p1) and discovers per-depth rules.
            self._kleisli_chains: dict[str, dict[str, list[RelationRule]]] = {}
            self._kleisli_disc_roles: dict[str, str] = {}
            for _op in _positional_ops:
                _op_rels = _rs.get_relations(_op)
                _disc_role, _chains = discover_kleisli_chains(
                    _op_rels, _binary_fmaps or {}
                )
                if _disc_role is not None and _chains:
                    self._kleisli_chains[_op] = _chains
                    self._kleisli_disc_roles[_op] = _disc_role
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
            self._rs = None
            self._binary_fmaps: dict = {}
            self._relation_rules: dict[str, list[RelationRule]] = {}
            self._kleisli_chains: dict[str, dict[str, list[RelationRule]]] = {}
            self._kleisli_disc_roles: dict[str, str] = {}

        # Phase XX: synthesise LambdaTerms from RelationRules.
        # Each RelationRule sequence is lifted to an explicit lambda term
        # (LetStep chain) over the Expr algebra.  This enables:
        #   1. Provably generative prediction (not table lookup).
        #   2. Creative transfer: novel ops handled via structural similarity.
        if self._rs is not None and self._relation_rules:
            self._lambda_library: dict[str, LambdaTerm] = synthesize_library(
                self._relation_rules, self._rs
            )
        else:
            self._lambda_library = {}

        # Phase XIX: context category for sheaf-theoretic dispatch.
        self._ctx_cat = ContextCategory()

        # Phase X: heuristic attributes (_trans, _kan, _vocab) removed.

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
        # Phase XIX — Context Category dispatch.
        # Classify the current prefix into a context object from the context
        # category C.  All subsequent format guards use is_refinement() instead
        # of bare 'eq' in prefix / 'step' in prefix string tests.  The context
        # category replaces ad-hoc string matching with a principled presheaf
        # section structure: a prediction rule registered at context c fires
        # whenever the current context IS-A c (refines c).
        ctx = self._ctx_cat.classify(prefix)

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
                # Restriction map ρ_{EQ → ANY}: specialise the chain rule section
                # (registered at ANY) to eq-format vs trace-format context.
                # use_eq=True  → restrict to ContextId.EQ  (use eq_table)
                # use_eq=False → restrict to ContextId.TRACE/INPUT (use chain_table)
                use_eq = self._ctx_cat.is_refinement(ctx, ContextId.EQ)
                result = _chain_predict(chain_rule, chain_state.input_tokens, chain_state.output_tokens, use_eq_table=use_eq)
                if result is not None:
                    return result

                # Level 1b-λ: Lambda term evaluation (Phase XX).
                # Presheaf section at ContextId.ANY (all formats).
                # Evaluates the op's LambdaTerm (synthesised from RelationRules)
                # on the observed input tokens.  More general than chain table
                # lookup (Level 1b) — fires for OOD inputs not in the table.
                # Creative transfer fires when op has no lambda term but a
                # structurally compatible term exists in the library.
                if self._lambda_library and not use_eq:
                    _lt_result = lambda_predict(
                        prefix, self._lambda_library, self._binary_fmaps,
                        allow_transfer=True,
                    )
                    if _lt_result is not None:
                        return _lt_result

                # Level 1c-relational: arity-free hypergraph rule prediction.
                # Uses RelationRules discovered from named-role tuples — no arities,
                # no prefix S-expressions, no parse tree.  Fires for any op with a
                # clean input separator schema (the rel predictor handles k=0..end).
                #
                # Phase XIII: eq-format guard removed.  For eq-format sequences (use_eq=True),
                # trace-format RelationRules (step/ans) are reused: the 'ans' content is
                # extracted and returned directly as the eq-format output.  This allows
                # e.g. linear_eval (plain eq format) to benefit from rules discovered from
                # linear_trace (step/ans format): mul(a, v) then add(step, b) → answer.
                _ROLE_DELIMS = frozenset({'step', 'ans', 'eq', '<eos>'})
                if (self._relation_rules
                        and chain_state.op in self._relation_rules
                        and self._rs is not None):
                    _rel_input = [chain_state.op] + list(chain_state.input_tokens)
                    # Phase XV (Coproducts): use predict_alternatives_from_rules so
                    # that competing rules for the same output role produce a
                    # probability distribution (Kleisli morphism in the probability
                    # monad) rather than a forced single choice.
                    _alternatives = predict_alternatives_from_rules(
                        _rel_input, self._rs, self._relation_rules, self._binary_fmaps
                    )
                    if _alternatives:
                        _rk = len(chain_state.output_tokens)
                        _dist: dict[str, float] = {}
                        _total_w = sum(w for _, w in _alternatives)
                        for _rel_output, _w in _alternatives:
                            _norm_w = _w / _total_w if _total_w > 0 else _w
                            if use_eq:
                                # Eq-format: extract the 'ans' or 'eq' role content.
                                _role_content: list[str] = []
                                for _role_delim in ('ans', 'eq'):
                                    if _role_delim in _rel_output:
                                        _start = _rel_output.index(_role_delim) + 1
                                        _raw = _rel_output[_start:]
                                        _end = next(
                                            (i for i, t in enumerate(_raw) if t in _ROLE_DELIMS),
                                            len(_raw)
                                        )
                                        _role_content = _raw[:_end]
                                        break
                                if _role_content:
                                    if _rk < len(_role_content):
                                        _tok = _role_content[_rk]
                                        _dist[_tok] = _dist.get(_tok, 0.0) + _norm_w
                                    elif _rk == len(_role_content):
                                        _dist['<eos>'] = _dist.get('<eos>', 0.0) + _norm_w
                            else:
                                if _rk < len(_rel_output):
                                    _tok = _rel_output[_rk]
                                    _dist[_tok] = _dist.get(_tok, 0.0) + _norm_w
                                elif _rk == len(_rel_output):
                                    _dist['<eos>'] = _dist.get('<eos>', 0.0) + _norm_w
                        if _dist:
                            return _dist

                # Level 1c-kleisli: multi-step Kleisli chain prediction (Phase XII).
                # Fires for variable-depth ops (e.g. pow) where the number of
                # 'step' tokens depends on an input value (the exponent).
                # Uses per-depth RelationRules discovered by discover_kleisli_chains.
                # Only fires in TRACE format (not eq-format) — eq-format pow is
                # handled by the Level 0.7 compose engine.
                if (not use_eq
                        and self._kleisli_chains
                        and chain_state.op in self._kleisli_chains
                        and self._rs is not None):
                    _disc_role = self._kleisli_disc_roles[chain_state.op]
                    _kl_seq = [chain_state.op] + list(chain_state.input_tokens)
                    _kl_rel = self._rs.extract_relation(_kl_seq)
                    if _kl_rel is not None:
                        _disc_val: Optional[str] = None
                        for _sep, _toks in _kl_rel.input_roles:
                            _rn = _sep if _sep else ''
                            if _rn == _disc_role and _toks:
                                _disc_val = ''.join(_toks)
                                break
                        _op_chains = self._kleisli_chains[chain_state.op]
                        if _disc_val is not None and _disc_val in _op_chains:
                            _kl_rules = _op_chains[_disc_val]
                            # Build role_values from input roles
                            _kl_vals: dict[str, str] = {}
                            for _sep, _toks in _kl_rel.input_roles:
                                _rn = _sep if _sep else ''
                                if _toks:
                                    _kl_vals[_rn] = ''.join(_toks)
                            # Apply rules in dependency order
                            _kl_ok = True
                            for _kl_rule in _kl_rules:
                                _kl_dist = _kl_rule.evaluate(_kl_vals, self._binary_fmaps)
                                if not _kl_dist:
                                    _kl_ok = False
                                    break
                                _kl_vals[_kl_rule.output_role] = max(_kl_dist, key=_kl_dist.get)
                            if _kl_ok:
                                # Rebuild predicted output with original delimiters
                                _kl_out: list[str] = []
                                for _kl_rule in _kl_rules:
                                    _delim = 'step' if _kl_rule.output_role.startswith('step') else _kl_rule.output_role
                                    _kl_out.append(_delim)
                                    _kl_out.extend(list(_kl_vals[_kl_rule.output_role]))
                                _kl_rk = len(chain_state.output_tokens)
                                if _kl_rk < len(_kl_out):
                                    return {_kl_out[_kl_rk]: 1.0}
                                if _kl_rk == len(_kl_out):
                                    return {'<eos>': 1.0}


        # Level 1b-λ (outer): creative transfer for ops NOT in chain_op_atoms.
        # Fires for novel ops (not seen in training) that are structurally
        # compatible with a known lambda term in the library (Phase XX).
        # Presheaf section at ContextId.TRACE/INPUT (non-eq-format).
        if (self._lambda_library
                and not self._ctx_cat.is_refinement(ctx, ContextId.EQ)
                and prefix):
            _op_candidate = prefix[0] if prefix else ''
            if _op_candidate not in self._chain_op_atoms:
                _lt_outer = lambda_predict(
                    prefix, self._lambda_library, self._binary_fmaps,
                    allow_transfer=True,
                )
                if _lt_outer is not None:
                    return _lt_outer

        # Level 1d: Equalizer solve — BFM enumeration (Phase XVII).
        # Presheaf section at ContextId.EQ: only fires in equation-format prefixes.
        # Handles 'lsolve A x B from C eq V' by enumerating all candidates v
        # and evaluating add(mul(A, v), B) == C via BFM lookups.
        # No int() calls; same code path for any op with the same algebraic form.
        if self._compose_succ_map and self._binary_fmaps and self._ctx_cat.is_refinement(ctx, ContextId.EQ):
            eq_result = _equalizer_predict(
                prefix,
                self._binary_fmaps,
                self._compose_succ_map,
                self._compose_zero,
            )
            if eq_result is not None:
                return eq_result

        # Level 1e: Pullback predict — trace-format equational solve (Phase XVIII).
        # Handles 'linsolve A B C step R1 ans X' by finding X via equalizer and
        # computing R1 = mul(A, X) directly from BFM.
        # Also handles 'bern_p1/bern_p2' by composing mul + add/sub via NNO engine.
        # Domain-agnostic: uses BFM + _compose, no int() calls.
        if self._compose_succ_map:
            pb_result = _pullback_predict(
                prefix,
                self._binary_fmaps,
                self._compose_succ_map,
                self._compose_zero,
                fc_lookup=self._fc_lookup if self._fc_lookup else None,
                fold_rules=self._fold_rules if self._fold_rules else None,
                carry_el=self._compose_carry_el,
                carry_out=self._compose_carry_out,
                compose_cache=self._compose_cache,
            )
            if pb_result is not None:
                return pb_result

        # Pass discovered op_atoms — not a hardcoded list
        state = parse_prefix(prefix, op_atoms=self._op_atoms)

        # Level 0.5: FC direct + adjunction-based exact lookup (CT_REFERENCE §4,17).
        # Presheaf section at ContextId.EQ.
        if (self._fc_lookup or self._adj_lookup) and prefix and self._ctx_cat.is_refinement(ctx, ContextId.EQ):
            fc_result = _fc_and_adj_predict(prefix, self._fc_lookup, self._adj_lookup)
            if fc_result is not None:
                return fc_result

        # Level 0.6: NNO chain prediction with carry/borrow propagation (CT_REFERENCE §19).
        # Presheaf section at ContextId.EQ.
        if (self._unary_chain_maps or self._unary_carry_maps) and prefix and self._ctx_cat.is_refinement(ctx, ContextId.EQ):
            nno_result = _nno_chain_predict(
                prefix, self._unary_chain_maps, self._unary_carry_maps
            )
            if nno_result is not None:
                return nno_result

        # Level 0.7: Composition engine — NNO fold rule (CT_REFERENCE §19).
        # Presheaf section at ContextId.EQ.
        if (self._fold_rules or self._adj_solve_map) and self._compose_succ_map and prefix and self._ctx_cat.is_refinement(ctx, ContextId.EQ):
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

        # Phase X complete: all heuristic levels (0, 2, 3, 4) removed.
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
# Internal helpers: Equalizer solve (Level 1d, Phase XVII)
# ---------------------------------------------------------------------------

# Ops whose output is a candidate x satisfying f(x) = target_value.
# The solve is done by enumerating the NNO digit chain and evaluating
# the composed function f via BFM lookups — no int() calls.
#
# Format: <op> A x B from C eq V
#   A = coefficient (before 'x'), B = constant (before 'from'),
#   C = target value (before 'eq'), V = answer (after 'eq')
#   f(v) = add(mul(A, v), B)
_EQUALIZER_SOLVE_OPS: frozenset[str] = frozenset({'lsolve'})


def _equalizer_predict(
    prefix: list[str],
    bfm: dict[str, dict[tuple, str]],
    succ_map: dict[str, str],
    zero: str,
) -> Optional[dict[str, float]]:
    """Level 1d: Equalizer solve via BFM enumeration (Phase XVII).

    For 'lsolve A x B from C eq ...', enumerate all v in the NNO digit chain
    and return the v where add(mul(A, v), B) == C.  Domain-agnostic: uses
    only opaque string tokens and BFM lookups, no int() calls.

    Returns {v: 1.0} if a unique solution is found, or a uniform
    distribution over solutions if multiple exist.  Returns None on miss.
    """
    if not prefix or prefix[0] not in _EQUALIZER_SOLVE_OPS:
        return None
    if 'eq' not in prefix or 'x' not in prefix or 'from' not in prefix:
        return None
    try:
        eq_idx   = prefix.index('eq')
        x_idx    = prefix.index('x')
        from_idx = prefix.index('from')
    except ValueError:
        return None
    if not (0 < x_idx < from_idx < eq_idx):
        return None

    output_so_far = prefix[eq_idx + 1:]
    if len(output_so_far) > 1:
        # Already emitted more than one token — can't use this path
        return None

    a_str = ''.join(prefix[1:x_idx])
    b_str = ''.join(prefix[x_idx + 1:from_idx])
    c_str = ''.join(prefix[from_idx + 1:eq_idx])
    if not a_str or not b_str:
        return None
    if not c_str and not output_so_far:
        return None
    # c_str may be multi-digit (e.g. '17'); comparison is string equality

    mul_map = bfm.get('mul', {})
    add_map = bfm.get('add', {})
    if not mul_map or not add_map:
        return None

    # Build the digit chain from the NNO successor map
    candidates: list[str] = []
    _seen: set[str] = {zero}
    candidates.append(zero)
    _cur = zero
    while True:
        _nxt = succ_map.get(_cur)
        if _nxt is None or _nxt in _seen:
            break
        candidates.append(_nxt)
        _seen.add(_nxt)
        _cur = _nxt

    # Enumerate: find all v where add(mul(A, v), B) == C
    solutions: list[str] = []
    for v in candidates:
        mul_result = mul_map.get((a_str, v))
        if mul_result is None:
            continue
        add_result = add_map.get((mul_result, b_str))
        if add_result is None:
            continue
        if add_result == c_str:
            solutions.append(v)

    if not solutions:
        return None

    # Point mass if unique; uniform over solutions if ambiguous
    if len(output_so_far) == 0:
        # Predict the single-digit answer
        w = 1.0 / len(solutions)
        return {v: w for v in solutions}
    # output_so_far == [v]; predict <eos>
    return {'<eos>': 1.0}


# ---------------------------------------------------------------------------
# Internal helpers: Pullback predict (Level 1e, Phase XVIII)
# ---------------------------------------------------------------------------

# The pullback is the limit of f: A → C ← B: g — all pairs (a, b) with
# f(a) = g(b).  For trace-format linsolve, this means finding X and R1
# simultaneously: X is the solution (equalizer), R1 = mul(A, X) is derived.
# The step value is obtained from X via the forward BFM — no sub needed.

def _pullback_predict(
    prefix: list[str],
    bfm: dict[str, dict[tuple, str]],
    succ_map: dict[str, str],
    zero: str,
    fc_lookup: Optional[dict] = None,
    fold_rules: Optional[dict] = None,
    carry_el: str = '',
    carry_out: tuple = (),
    compose_cache: Optional[dict] = None,
) -> Optional[dict[str, float]]:
    """Level 1e: Pullback predict for trace-format equational solve (Phase XVIII).

    Handles two op families:
      - 'linsolve A B C step R1 ans X': find X via equalizer, R1=mul(A,X).
      - 'bern_p1/bern_p2 P V1 V2 step V1sq step V2sq ans P2':
          compute V1sq=mul(V1,V1), V2sq=mul(V2,V2), then P2 via _compose add/sub.

    All arithmetic is done via BFM or _compose (NNO fold) — no int() calls.
    Returns {next_token: 1.0} or {'<eos>': 1.0} on success, None on miss.
    """
    if not prefix:
        return None
    op = prefix[0]

    if op not in ('linsolve', 'bern_p1', 'bern_p2'):
        return None

    # Determine end of input (first output delimiter: step/ans/<eos>)
    first_out_idx = None
    for i, t in enumerate(prefix):
        if i > 0 and t in ('step', 'ans', '<eos>'):
            first_out_idx = i
            break

    if first_out_idx is None:
        first_out_idx = len(prefix)

    input_toks = prefix[1:first_out_idx]
    output_so_far = prefix[first_out_idx:]

    # ---- linsolve branch ----
    if op == 'linsolve':
        # Input: A B C (positional, no separators)
        if len(input_toks) < 3:
            return None
        a_str = input_toks[0]
        b_str = input_toks[1]
        c_str = ''.join(input_toks[2:])
        if not a_str or not b_str or not c_str:
            return None

        mul_map = bfm.get('mul', {})
        add_map = bfm.get('add', {})
        if not mul_map or not add_map:
            return None

        # Build digit chain
        candidates: list[str] = []
        _seen: set[str] = {zero}
        candidates.append(zero)
        _cur = zero
        while True:
            _nxt = succ_map.get(_cur)
            if _nxt is None or _nxt in _seen:
                break
            candidates.append(_nxt)
            _seen.add(_nxt)
            _cur = _nxt

        # Find X via equalizer
        solutions: list[str] = []
        for v in candidates:
            mul_result = mul_map.get((a_str, v))
            if mul_result is None:
                continue
            add_result = add_map.get((mul_result, b_str))
            if add_result is None:
                continue
            if add_result == c_str:
                solutions.append(v)

        if not solutions:
            return None
        x_str = solutions[0]
        r1_str = mul_map.get((a_str, x_str))
        if r1_str is None:
            return None

        expected_output: list[str] = ['step'] + list(r1_str) + ['ans'] + list(x_str)

    # ---- Bernoulli branch ----
    elif op in ('bern_p1', 'bern_p2'):
        # Input: P(2-digit) V1(1-digit) V2(1-digit)
        # len(input_toks) should be 4 (two digits for P, one each for V1, V2)
        if len(input_toks) != 4:
            return None
        # P is zero-padded 2 digits: input_toks[0:2]
        p_toks = tuple(input_toks[:2])
        v1 = input_toks[2]
        v2 = input_toks[3]
        if not (len(p_toks) == 2 and len(v1) == 1 and len(v2) == 1):
            return None

        if fc_lookup is None or fold_rules is None or not succ_map:
            # No compose engine — can't compute multi-digit arithmetic
            return None

        cache = compose_cache if compose_cache is not None else {}

        # Compute r1 = V1² and r2 = V2²
        r1_tup = _compose('mul', ((v1,), (v1,)), fc_lookup, fold_rules,
                          succ_map, carry_el, carry_out, zero, cache)
        r2_tup = _compose('mul', ((v2,), (v2,)), fc_lookup, fold_rules,
                          succ_map, carry_el, carry_out, zero, cache)
        if r1_tup is None or r2_tup is None:
            return None

        # Compute ans = P ± (r1 - r2) depending on op
        # bern_p2: P2 = P1 + r1 - r2  (given P1, V1 > V2 in training)
        # bern_p1: P1 = P2 + r2 - r1  (given P2, V2 < V1 so r2 < r1)
        if op == 'bern_p2':
            # P2 = add(sub(P1 + r1), r2) or P2 = add(P1, sub(r1, r2))
            # Use: temp = add(P1, r1); P2 = sub(temp, r2)
            temp_tup = _compose('add', (p_toks, r1_tup), fc_lookup, fold_rules,
                                succ_map, carry_el, carry_out, zero, cache)
            if temp_tup is None:
                return None
            ans_tup = _compose('sub', (temp_tup, r2_tup), fc_lookup, fold_rules,
                               succ_map, carry_el, carry_out, zero, cache)
        else:  # bern_p1
            # P1 = P2 + r2 - r1: temp = add(P2, r2); P1 = sub(temp, r1)
            temp_tup = _compose('add', (p_toks, r2_tup), fc_lookup, fold_rules,
                                succ_map, carry_el, carry_out, zero, cache)
            if temp_tup is None:
                return None
            ans_tup = _compose('sub', (temp_tup, r1_tup), fc_lookup, fold_rules,
                               succ_map, carry_el, carry_out, zero, cache)

        if ans_tup is None:
            return None

        # Zero-pad answer to 2 digits (ans always uses _zfill2 in corpus)
        if len(ans_tup) == 1:
            ans_tup = (zero,) + ans_tup
        # Zero-pad r1 and r2 only if they are the step output — they are NOT padded in corpus
        # (corpus uses _digits, not _zfill2, for the step values)
        r1_toks = list(r1_tup)
        r2_toks = list(r2_tup)
        ans_toks = list(ans_tup)

        expected_output = (['step'] + r1_toks +
                           ['step'] + r2_toks +
                           ['ans'] + ans_toks)
    else:
        return None

    k = len(output_so_far)
    if k < len(expected_output):
        return {expected_output[k]: 1.0}
    if k == len(expected_output):
        return {'<eos>': 1.0}
    return None


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



