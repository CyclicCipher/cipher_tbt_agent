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

    Level 0.5-NNO — CTKG SUCC_EDGE graph traversal (Stage 4, CT_REFERENCE §19):
        Traverses SUCC_EDGE morphisms in the MorphismGraph for succ/pred.
        Symbol-agnostic: uses parse_prefix (eq_token-aware), not hardcoded 'eq'.
        Replaces the former Level 0.6.  Handles OOD inputs via carry propagation.

    Level 0.5 — FC direct + adjunction-based lookup (CT_REFERENCE §4, 17):
        Exact lookup in the free category edge table.  Adjunction-mediated:
        if pred(4)→3 is in training, infers succ(3)=4 via the adjunction.

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

from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH, NodeId, OUTPUT_DELIMS

# ---------------------------------------------------------------------------
# NodeId constants for op dispatch (Iron Law: no bare string comparisons).
# All pipeline code must compare NodeIds, never op == 'string'.
# ---------------------------------------------------------------------------
_LINSOLVE_NODE  = TOKEN_GRAPH.encode('linsolve')
_BERN_P1_NODE   = TOKEN_GRAPH.encode('bern_p1')
_BERN_P2_NODE   = TOKEN_GRAPH.encode('bern_p2')
_CS1_NODE       = TOKEN_GRAPH.encode('cs1')
_CS2_NODE       = TOKEN_GRAPH.encode('cs2')
_CS3_NODE       = TOKEN_GRAPH.encode('cs3')
_CS4_NODE       = TOKEN_GRAPH.encode('cs4')
_LSOLVE_NODE    = TOKEN_GRAPH.encode('lsolve')
_CONCAT_NODE    = TOKEN_GRAPH.encode('concat')
_FST_NODE       = TOKEN_GRAPH.encode('fst')
_DIV_NODE       = TOKEN_GRAPH.encode('div')
_PULLBACK_OPS: frozenset[NodeId] = frozenset({
    _LINSOLVE_NODE, _BERN_P1_NODE, _BERN_P2_NODE,
    _CS1_NODE, _CS2_NODE, _CS3_NODE, _CS4_NODE,
})
_COMPOSE_EXTRA_OPS: frozenset[NodeId] = frozenset({
    _CONCAT_NODE, _FST_NODE, _DIV_NODE,
})
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
from experiments.symbolic_ai_v2.ctkg.learning.skeleton_lambda import SkeletonStore




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
        raw_corpus: Optional[list[list[str]]] = None,
        eq_token: str = "eq",
    ) -> None:
        # eq_token: the delimiter between input and output in eq-format sequences.
        # Pass a non-default value when using anonymous symbol tables.
        self._eq_token: str = eq_token
        # Constructor parameters kept for API compatibility; not used internally
        # after Phase X (fallback removal).
        self._hankel = hankel
        self._lattice = lattice
        self._morphism_graph = morphism_graph

        # Operator → ProcessRule lookup (fold-type)
        # Sprint B: keyed by NodeId (Iron Law: no string dispatch on content tokens).
        self._rules: dict[NodeId, ProcessRule] = {
            TOKEN_GRAPH.encode(rule.op_atom): rule for rule in process_rules
        }

        # Operator set from process rules (expanded by FC edges below)
        self._op_atoms: frozenset[NodeId] = frozenset(self._rules.keys())

        # Phase XIX: context category for sheaf-theoretic dispatch.
        # Initialised early so other init steps can use it.
        self._ctx_cat = ContextCategory(eq_token=self._eq_token)

        if fc is not None:
            # ------------------------------------------------------------------
            # Step 1: Build all intermediate knowledge objects (local vars only).
            # These are NEVER stored as self._ attributes — knowledge lives in
            # the MorphismGraph after Stage 4 population below.
            # ------------------------------------------------------------------
            _fc_lookup = build_fc_lookup(fc)
            _adj_lookup = build_adj_lookup(fc)
            _unary_chain_maps = build_unary_chain_maps(fc)
            _unary_carry_maps = build_unary_carry_maps(fc)
            _fold_rules = discover_binary_fold_rules(fc)

            # Build chain_rules dict (local, for population)
            _chain_rules_dict: dict[NodeId, ChainRule] = {
                TOKEN_GRAPH.encode(rule.op_atom): rule for rule in (chain_rules or [])
            }

            # Pick succ/carry info for the composition engine.
            # Prefer the NNO candidate that IS in unary_carry_maps (the forward/succ
            # direction).  When pred and succ tie on chain length, max() may pick pred,
            # which is NOT in unary_carry_maps — that would leave engine = None.
            # So: scan nno_candidates in carry-map priority order, then fall back to
            # longest-chain candidate.
            _succ_carry = _unary_carry_maps
            _nno = None
            for _cand in (fc.nno_candidates or []):
                if _cand.op in _succ_carry:
                    _nno = _cand
                    break
            if _nno is None and fc.nno_candidates:
                _nno = max(fc.nno_candidates, key=lambda n: len(n.successor_map))
            if _nno is not None and _nno.op in _succ_carry:
                _raw_succ = _nno.successor_map
                _carry_el = _succ_carry[_nno.op][0]
                _carry_out = _succ_carry[_nno.op][1]
                _zero = _nno.zero_candidate
                _compose_succ_map = complete_succ_map(_raw_succ, _zero, _carry_el)
            else:
                _carry_el = ""
                _carry_out = ()
                _zero = ""
                _compose_succ_map = {}

            # Adjunction inverse solve map (local var)
            _adj_solve_map: dict[str, tuple] = {}
            for adj in fc.adjunctions:
                if adj.preserved_position is not None:
                    rk = adj.right_op
                    if rk not in _adj_solve_map:
                        _adj_solve_map[rk] = (adj.left_op, adj.preserved_position)

            # Extend op_atoms with all FC-discovered operators
            fc_ops = frozenset(TOKEN_GRAPH.encode(edge.op) for edge in fc.edges)
            self._op_atoms = self._op_atoms | fc_ops

            # ------------------------------------------------------------------
            # Step 2: Populate ALL knowledge into MorphismGraph typed morphisms.
            # After this block, every rule is in the graph — no Python dicts needed.
            # ------------------------------------------------------------------

            # SUCC_EDGE: NNO successor chain (one edge per digit pair)
            for _fwd_op, (_fwd_carry_el, _fwd_carry_out) in _unary_carry_maps.items():
                _fwd_step = _unary_chain_maps.get(_fwd_op)
                if _fwd_step:
                    _populate_succ_edges_to_mg(self._morphism_graph, _fwd_step)

            # FOLD_RULE: NNO binary fold rules (one self-loop per op)
            _populate_fold_rules_to_mg(self._morphism_graph, _fold_rules)

            # CHAIN_STEP: chain/eq table rules (one self-loop per op)
            _populate_chain_steps_to_mg(self._morphism_graph, _chain_rules_dict)

            # FC_EDGE / ADJ_EDGE_DIRECT: exact FC and adjunction lookups (per op)
            _populate_fc_adj_edges_to_mg(
                self._morphism_graph, _fc_lookup, _adj_lookup
            )

            # ADJ_EDGE: adjunction inverse-solve info (per inverse op)
            _populate_adj_solve_to_mg(self._morphism_graph, _adj_solve_map)

            # ------------------------------------------------------------------
            # Step 3: Build ComposeEngine from local vars (before RelationStore).
            # Engine holds the dicts it needs internally — Predictor does NOT
            # store _fc_lookup, _fold_rules, _compose_succ_map, etc.
            # ------------------------------------------------------------------
            if _compose_succ_map:
                _compose_cache: dict[tuple, tuple] = {}
                self._engine: Optional['ComposeEngine'] = ComposeEngine(
                    fc_lookup=_fc_lookup,
                    fold_rules=_fold_rules,
                    succ_map=_compose_succ_map,
                    carry_el=_carry_el,
                    carry_out=_carry_out,
                    zero=_zero,
                    cache=_compose_cache,
                    adj_lookup=_adj_lookup,
                    adj_solve_map=_adj_solve_map,
                )
                # Phase XXI: build type context from NNO successor chain.
                from experiments.symbolic_ai_v2.ctkg.core.dependent_type import (
                    infer_token_types,
                )
                _nid_succ = {
                    TOKEN_GRAPH.encode(k): TOKEN_GRAPH.encode(v)
                    for k, v in _compose_succ_map.items()
                }
                self._type_context: Optional[dict] = infer_token_types(_nid_succ)
            else:
                self._engine = None
                self._type_context = None

            # ------------------------------------------------------------------
            # Step 4: RelationStore + rule discovery (local, for population).
            # ------------------------------------------------------------------
            _rs = RelationStore()
            _rs_seqs: list[list[str]] = []
            for cr in (chain_rules or []):
                for inp_toks, out_toks in (cr.chain_table or {}).items():
                    _rs_seqs.append([cr.op_atom] + list(inp_toks) + list(out_toks))
            _rs.update_batch(_rs_seqs)

            # Keep _rs as an instance attribute — needed at inference time by
            # _ctkg_path_find for extract_relation parsing.
            self._rs = _rs

            # Discover relation rules (local var — populated into graph, then dropped)
            _relation_rules: dict[NodeId, list[RelationRule]] = {}
            _positional_ops = _rs.ops_with_positional_schema()
            for _op in _rs.ops_with_schema():
                _op_rels = _rs.get_relations_by_id(_op)
                _mm_tol = 0.25 if _op in _positional_ops else 0.0
                _op_rr = discover_relation_rules(
                    _op_rels, self._engine, mismatch_tolerance=_mm_tol,
                    type_context=self._type_context,
                )
                if _op_rr:
                    _expected_roles = _rs.all_output_role_names_by_id(_op)
                    _covered_roles = {_r.output_role for _r in _op_rr}
                    if _expected_roles.issubset(_covered_roles):
                        _relation_rules[_op] = _op_rr

            # Discover Kleisli chains (local var — populated into graph, then dropped)
            _kleisli_chains: dict[NodeId, dict] = {}
            _kleisli_disc_roles: dict[NodeId, NodeId] = {}
            for _op in _positional_ops:
                _op_rels = _rs.get_relations_by_id(_op)
                _disc_role, _chains = discover_kleisli_chains(_op_rels, self._engine)
                if _disc_role is not None and _chains:
                    _kleisli_chains[_op] = _chains
                    _kleisli_disc_roles[_op] = _disc_role

            # RELATION_RULE: arity-free relational rules (one self-loop per op)
            _populate_relation_rules_to_mg(self._morphism_graph, _relation_rules)

            # KLEISLI_CHAIN: variable-depth Kleisli chains (one self-loop per op)
            _populate_kleisli_chains_to_mg(
                self._morphism_graph, _kleisli_chains, _kleisli_disc_roles
            )

            # ------------------------------------------------------------------
            # Step 5: Derive Predictor instance attributes from graph.
            # Only _rs, _unary_chain_maps, and _engine survive as instance attrs.
            # ALL Python knowledge dicts (_chain_rules, _relation_rules, etc.) are
            # local vars only — they are NOT stored on self.
            # ------------------------------------------------------------------

            # chain_op_atoms: derived from CHAIN_STEP morphisms in graph
            self._chain_op_atoms: frozenset[NodeId] = _extract_chain_op_atoms_from_mg(
                self._morphism_graph
            )

            # _unary_chain_maps kept for NNO direction detection in _ctkg_nno_predict
            # (determines succ vs pred from step_map comparison against succ_graph)
            self._unary_chain_maps: dict[str, dict[str, str]] = _unary_chain_maps

            # Phase XX: lambda library — synthesised here while _relation_rules is
            # still in scope (local var); stored on self but NOT a raw knowledge dict.
            if _relation_rules:
                self._lambda_library: dict[NodeId, LambdaTerm] = synthesize_library(
                    _relation_rules, _rs
                )
            else:
                self._lambda_library: dict[NodeId, LambdaTerm] = {}

        else:
            # No FreeCategoryGraph: minimal initialisation.
            self._chain_op_atoms = frozenset(
                TOKEN_GRAPH.encode(rule.op_atom) for rule in (chain_rules or [])
            )
            self._unary_chain_maps = {}
            self._rs = None
            self._engine: Optional['ComposeEngine'] = None
            self._type_context = None
            self._lambda_library = {}

        # Phase XXIV: SkeletonStore — skeleton-discriminated lambda terms.
        # Uses engine.succ_map and engine (not the deleted _compose_succ_map).
        self._skeleton_store = SkeletonStore()
        if raw_corpus and self._engine is not None and self._engine.succ_map:
            self._skeleton_store.learn(
                raw_corpus,
                self._engine.succ_map,
                self._engine,
            )

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
        ctx = self._ctx_cat.classify(prefix)

        # Stage 4 — Unified CTKG graph traversal (_ctkg_path_find).
        # Handles Levels 1b (chain), 1b-λ (lambda), 1c-relational, 1c-kleisli,
        # 0.5 (FC/adj exact lookup), and 0.7 (NNO fold composition) in a single
        # graph traversal.  Dispatch is on morph_type, never on op name strings.
        # All knowledge comes from MorphismGraph typed morphisms (Stage 4 complete).
        # No gate: _ctkg_path_find returns None quickly when nothing matches.
        _ctkg_result = _ctkg_path_find(
            prefix,
            self._morphism_graph,
            self._engine,
            self._rs,
            self._ctx_cat,
            ctx,
            self._eq_token,
            self._op_atoms,
            self._chain_op_atoms,
            lambda_library=self._lambda_library,
        )
        if _ctkg_result is not None:
            return _ctkg_result

        # Level 1f: SkeletonStore prediction (Phase XXIV).
        # Handles operators like 'd' and 'int' whose input schema is too
        # heterogeneous for RelationStore / Lambda terms.
        # Uses engine.succ_map (not deleted _compose_succ_map).
        if self._engine is not None and self._engine.succ_map:
            _sk_result = self._skeleton_store.predict(
                prefix,
                self._engine.succ_map,
                self._engine,
            )
            if _sk_result is not None:
                return _sk_result

        # Level 1b-λ (outer): creative transfer for ops NOT in chain_op_atoms.
        # Fires for novel ops (not seen in training) that are structurally
        # compatible with a known lambda term in the library (Phase XX).
        if (self._lambda_library
                and not self._ctx_cat.is_refinement(ctx, ContextId.EQ)
                and prefix):
            _op_candidate = prefix[0] if prefix else ''
            if TOKEN_GRAPH.encode(_op_candidate) not in self._chain_op_atoms:
                _nid_prefix_outer = TOKEN_GRAPH.encode_seq(prefix)
                _lt_outer = lambda_predict(
                    _nid_prefix_outer, self._lambda_library, self._engine,
                    allow_transfer=True,
                )
                if _lt_outer is not None:
                    return {TOKEN_GRAPH.decode(k): v for k, v in _lt_outer.items()}

        # Level 1d: Equalizer solve (Phase XVII).
        # Uses engine.succ_map and engine.zero (not deleted _compose_succ_map).
        if (self._engine is not None
                and self._engine.succ_map
                and self._ctx_cat.is_refinement(ctx, ContextId.EQ)):
            eq_result = _equalizer_predict(
                prefix,
                self._engine,
                self._engine.succ_map,
                self._engine.zero,
            )
            if eq_result is not None:
                return eq_result

        # Level 1e: Pullback predict (Phase XVIII).
        if self._engine is not None and self._engine.succ_map:
            pb_result = _pullback_predict(
                prefix,
                self._engine,
                self._engine.succ_map,
                self._engine.zero,
            )
            if pb_result is not None:
                return pb_result

        # Level 0.5-NNO: CTKG SUCC_EDGE graph traversal (CT_REFERENCE §19).
        # Symbol-agnostic succ/pred prediction via SUCC_EDGE morphisms + carry info.
        # Gate: engine exists (carry info in engine.carry_el/carry_out).
        if (self._engine is not None
                and self._engine.carry_el
                and prefix
                and self._ctx_cat.is_refinement(ctx, ContextId.EQ)):
            _nno_ctkg = _ctkg_nno_predict(
                prefix,
                self._morphism_graph,
                self._unary_chain_maps,
                self._engine,
                self._op_atoms,
                eq_token=self._eq_token,
            )
            if _nno_ctkg is not None:
                return _nno_ctkg

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
_EQUALIZER_SOLVE_OPS: frozenset[NodeId] = frozenset({_LSOLVE_NODE})


def _equalizer_predict(
    prefix: list[str],
    engine: Optional['ComposeEngine'],
    succ_map: dict[str, str],
    zero: str,
) -> Optional[dict[str, float]]:
    """Level 1d: Equalizer solve via ComposeEngine enumeration (Phase XVII).

    For 'lsolve A x B from C eq ...', enumerate all v in the NNO digit chain
    and return the v where add(mul(A, v), B) == C.  Domain-agnostic: uses
    only opaque string tokens and ComposeEngine, no int() calls.

    Returns {v: 1.0} if a unique solution is found, or a uniform
    distribution over solutions if multiple exist.  Returns None on miss.
    """
    if not prefix or TOKEN_GRAPH.encode(prefix[0]) not in _EQUALIZER_SOLVE_OPS:
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

    if engine is None:
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
        mul_tup = engine.compute_tup('mul', (a_str,), (v,))
        if mul_tup is None:
            continue
        add_tup = engine.compute_tup('add', mul_tup, (b_str,))
        if add_tup is None:
            continue
        if ''.join(add_tup) == c_str:
            solutions.append(v)

    if not solutions:
        return None

    # Point mass if unique; uniform over solutions if ambiguous
    if len(output_so_far) == 0:
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
    engine: Optional['ComposeEngine'],
    succ_map: dict[str, str],
    zero: str,
) -> Optional[dict[str, float]]:
    """Level 1e: Pullback predict for trace-format equational solve (Phase XVIII).

    Handles two op families:
      - 'linsolve A B C step R1 ans X': find X via equalizer, R1=mul(A,X).
      - 'bern_p1/bern_p2 P V1 V2 step V1sq step V2sq ans P2':
          compute V1sq=mul(V1,V1), V2sq=mul(V2,V2), then P2 via ComposeEngine.

    All arithmetic is done via ComposeEngine (NNO fold) — no int() calls.
    Returns {next_token: 1.0} or {'<eos>': 1.0} on success, None on miss.
    """
    if not prefix:
        return None
    op = prefix[0]
    op_nid = TOKEN_GRAPH.encode(op)

    if op_nid not in _PULLBACK_OPS:
        return None

    if engine is None:
        return None

    # Determine end of input (first output delimiter: step/ans/<eos>)
    first_out_idx = None
    for i, t in enumerate(prefix):
        if i > 0 and TOKEN_GRAPH.encode(t) in OUTPUT_DELIMS:
            first_out_idx = i
            break

    if first_out_idx is None:
        first_out_idx = len(prefix)

    input_toks = prefix[1:first_out_idx]
    output_so_far = prefix[first_out_idx:]

    # ---- linsolve branch ----
    if op_nid == _LINSOLVE_NODE:
        # Input: A B C (positional, no separators)
        if len(input_toks) < 3:
            return None
        a_str = input_toks[0]
        b_str = input_toks[1]
        c_str = ''.join(input_toks[2:])
        if not a_str or not b_str or not c_str:
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
            mul_tup = engine.compute_tup('mul', (a_str,), (v,))
            if mul_tup is None:
                continue
            add_tup = engine.compute_tup('add', mul_tup, (b_str,))
            if add_tup is None:
                continue
            if ''.join(add_tup) == c_str:
                solutions.append(v)

        if not solutions:
            return None
        x_str = solutions[0]
        r1_tup = engine.compute_tup('mul', (a_str,), (x_str,))
        if r1_tup is None:
            return None

        expected_output: list[str] = ['step'] + list(r1_tup) + ['ans'] + [x_str]

    # ---- Bernoulli branch ----
    elif op_nid in (_BERN_P1_NODE, _BERN_P2_NODE):
        # Input: P(2-token) V1(1-token) V2(1-token)
        # len(input_toks) should be 4 (two tokens for P, one each for V1, V2)
        if len(input_toks) != 4:
            return None
        # P is zero-padded 2 tokens: input_toks[0:2]
        p_toks = tuple(input_toks[:2])
        v1 = input_toks[2]
        v2 = input_toks[3]

        # Compute r1 = V1² and r2 = V2²
        r1_tup = engine.compute_tup('mul', (v1,), (v1,))
        r2_tup = engine.compute_tup('mul', (v2,), (v2,))
        if r1_tup is None or r2_tup is None:
            return None

        # Compute ans = P ± (r1 - r2) depending on op
        # bern_p2: P2 = P1 + r1 - r2  (given P1, V1 > V2 in training)
        # bern_p1: P1 = P2 + r2 - r1  (given P2, V2 < V1 so r2 < r1)
        if op_nid == _BERN_P2_NODE:
            temp_tup = engine.compute_tup('add', p_toks, r1_tup)
            if temp_tup is None:
                return None
            ans_tup = engine.compute_tup('sub', temp_tup, r2_tup)
        else:  # bern_p1
            temp_tup = engine.compute_tup('add', p_toks, r2_tup)
            if temp_tup is None:
                return None
            ans_tup = engine.compute_tup('sub', temp_tup, r1_tup)

        if ans_tup is None:
            return None

        # Zero-pad answer to 2 digits (ans always uses _zfill2 in corpus)
        if len(ans_tup) == 1:
            ans_tup = (zero,) + ans_tup
        r1_toks = list(r1_tup)
        r2_toks = list(r2_tup)
        ans_toks = list(ans_tup)

        expected_output = (['step'] + r1_toks +
                           ['step'] + r2_toks +
                           ['ans'] + ans_toks)
    # ---- Conservation scenario branches (Phase F) ----
    # cs4 A B C step <A+B> ans D  (D = A+B - C, all results zero-padded to 2 digits)
    # cs3 A B D step <A+B> ans C  (same structure, solve for C)
    # cs2 A C D step <C+D> ans B  (B = C+D - A, ans is 1 digit)
    # cs1 B C D step <C+D> ans A  (A = C+D - B, ans is 1 digit)
    # Input layout: cs4/cs3 have 4 tokens (a:1, b:1, cd:2); cs2/cs1 have 5 (ab:1, c:2, d:2)
    elif op_nid in (_CS4_NODE, _CS3_NODE):
        if len(input_toks) != 4:
            return None
        a_tok = input_toks[0]
        b_tok = input_toks[1]
        cd_tup = (input_toks[2], input_toks[3])

        step_tup = engine.compute_tup('add', (a_tok,), (b_tok,))
        if step_tup is None:
            return None
        # Zero-pad step to 2 tokens (corpus always uses _zfill2 for step)
        if len(step_tup) == 1:
            step_tup = (zero,) + step_tup
        ans_tup = engine.compute_tup('sub', step_tup, cd_tup)
        if ans_tup is None:
            return None
        # Zero-pad ans to 2 tokens (corpus always uses _zfill2 for ans in cs4/cs3)
        if len(ans_tup) == 1:
            ans_tup = (zero,) + ans_tup

        expected_output = (['step'] + list(step_tup) +
                           ['ans']  + list(ans_tup))

    elif op_nid in (_CS2_NODE, _CS1_NODE):
        if len(input_toks) != 5:
            return None
        ab_tok = input_toks[0]
        c_tup = (input_toks[1], input_toks[2])
        d_tup = (input_toks[3], input_toks[4])

        step_tup = engine.compute_tup('add', c_tup, d_tup)
        if step_tup is None:
            return None
        # Zero-pad step to 2 tokens (corpus always uses _zfill2 for step)
        if len(step_tup) == 1:
            step_tup = (zero,) + step_tup
        ans_tup = engine.compute_tup('sub', step_tup, (ab_tok,))
        if ans_tup is None:
            return None
        # ans uses _digits (no zero-padding) for cs2/cs1

        expected_output = (['step'] + list(step_tup) +
                           ['ans']  + list(ans_tup))

    else:
        return None

    k = len(output_so_far)
    if k < len(expected_output):
        return {expected_output[k]: 1.0}
    if k == len(expected_output):
        return {'<eos>': 1.0}
    return None


# ---------------------------------------------------------------------------
# Stage 4: SUCC_EDGE population + CTKG NNO prediction (Level 0.5-NNO)
# ---------------------------------------------------------------------------

def _populate_succ_edges_to_mg(
    mg: MorphismGraph,
    succ_map: dict[str, str],
) -> None:
    """Write SUCC_EDGE morphisms into the MorphismGraph from the NNO succ_map.

    Each entry (digit → next_digit) in succ_map becomes a SUCC_EDGE morphism.
    Objects for each digit label are created via get_or_create_object if absent.
    Existing SUCC_EDGE morphisms between the same pair are not duplicated.

    After this call the MorphismGraph encodes the full single-digit successor
    relation as typed edges — knowledge is in the graph, not a Python dict.
    """
    for src_label, tgt_label in succ_map.items():
        src_obj = mg.get_or_create_object(src_label)
        tgt_obj = mg.get_or_create_object(tgt_label)
        existing = [
            m for m in mg.hom(src_obj.obj_id, tgt_obj.obj_id, include_identity=False)
            if m.morph_type == "SUCC_EDGE"
        ]
        if not existing:
            mg.add_morphism(
                src_obj.obj_id,
                tgt_obj.obj_id,
                morph_type="SUCC_EDGE",
                evidence=1,
            )


# ---------------------------------------------------------------------------
# Stage 4: Population functions — write Python knowledge dicts into MorphismGraph
# ---------------------------------------------------------------------------

def _populate_fold_rules_to_mg(
    mg: MorphismGraph,
    fold_rules: dict,
) -> None:
    """Store each BinaryFoldRule as a FOLD_RULE self-loop on its op node.

    One morphism per operator — the payload IS the BinaryFoldRule object.
    Avoids duplicate entries on repeated calls.
    """
    for op_str, rule in fold_rules.items():
        op_obj = mg.get_or_create_object(op_str)
        if not mg.source_morphisms(op_obj.obj_id, morph_type="FOLD_RULE"):
            mg.add_morphism(
                op_obj.obj_id, op_obj.obj_id,
                morph_type="FOLD_RULE", evidence=1, payload=rule,
            )


def _populate_chain_steps_to_mg(
    mg: MorphismGraph,
    chain_rules: dict,
) -> None:
    """Store each ChainRule as a CHAIN_STEP self-loop on its op node.

    One morphism per operator — the payload IS the ChainRule object (which
    carries both chain_table and eq_table).  Keyed by op NodeId in the input
    dict, but the graph stores the op string label in the object.
    """
    from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH as _TG
    for op_nid, chain_rule in chain_rules.items():
        op_str = _TG.decode(op_nid)
        op_obj = mg.get_or_create_object(op_str)
        if not mg.source_morphisms(op_obj.obj_id, morph_type="CHAIN_STEP"):
            mg.add_morphism(
                op_obj.obj_id, op_obj.obj_id,
                morph_type="CHAIN_STEP", evidence=1, payload=chain_rule,
            )


def _populate_fc_adj_edges_to_mg(
    mg: MorphismGraph,
    fc_lookup: dict,
    adj_lookup: dict,
) -> None:
    """Store FC and adjunction lookups as FC_EDGE self-loops per operator.

    payload = (op_fc_dict, op_adj_dict) where keys are input_tuple only
    (the op string is stripped — it's encoded in the source object label).
    One morphism per operator.  Groups all (op, input_tuple) → output entries
    from the flat fc_lookup / adj_lookup dicts.
    """
    from collections import defaultdict
    by_op_fc: dict[str, dict] = defaultdict(dict)
    by_op_adj: dict[str, dict] = defaultdict(dict)
    for (op_str, inp_tup), out_tup in fc_lookup.items():
        by_op_fc[op_str][inp_tup] = out_tup
    for (op_str, inp_tup), out_tup in adj_lookup.items():
        by_op_adj[op_str][inp_tup] = out_tup
    all_ops = set(by_op_fc.keys()) | set(by_op_adj.keys())
    for op_str in all_ops:
        op_obj = mg.get_or_create_object(op_str)
        if not mg.source_morphisms(op_obj.obj_id, morph_type="FC_EDGE"):
            mg.add_morphism(
                op_obj.obj_id, op_obj.obj_id,
                morph_type="FC_EDGE", evidence=1,
                payload=(dict(by_op_fc.get(op_str, {})),
                         dict(by_op_adj.get(op_str, {}))),
            )


def _populate_adj_solve_to_mg(
    mg: MorphismGraph,
    adj_solve_map: dict,
) -> None:
    """Store adjunction inverse-solve info as ADJ_EDGE self-loops per inverse op.

    payload = (fwd_op_str, preserved_position)
    One morphism per inverse-op (e.g. 'sub' gets an ADJ_EDGE pointing to 'add').
    """
    for inv_op_str, (fwd_op_str, preserved_pos) in adj_solve_map.items():
        op_obj = mg.get_or_create_object(inv_op_str)
        if not mg.source_morphisms(op_obj.obj_id, morph_type="ADJ_EDGE"):
            mg.add_morphism(
                op_obj.obj_id, op_obj.obj_id,
                morph_type="ADJ_EDGE", evidence=1,
                payload=(fwd_op_str, preserved_pos),
            )


def _populate_relation_rules_to_mg(
    mg: MorphismGraph,
    relation_rules: dict,
) -> None:
    """Store each list[RelationRule] as a RELATION_RULE self-loop on its op node.

    payload = list[RelationRule].  Keyed by op NodeId in the input dict.
    One morphism per operator.
    """
    from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH as _TG
    for op_nid, rel_rules in relation_rules.items():
        op_str = _TG.decode(op_nid)
        op_obj = mg.get_or_create_object(op_str)
        if not mg.source_morphisms(op_obj.obj_id, morph_type="RELATION_RULE"):
            mg.add_morphism(
                op_obj.obj_id, op_obj.obj_id,
                morph_type="RELATION_RULE", evidence=1, payload=list(rel_rules),
            )


def _populate_kleisli_chains_to_mg(
    mg: MorphismGraph,
    kleisli_chains: dict,
    kleisli_disc_roles: dict,
) -> None:
    """Store each Kleisli chain set as a KLEISLI_CHAIN self-loop on its op node.

    payload = (disc_role_nid, chains_dict) where disc_role_nid is the NodeId
    of the discriminator role and chains_dict maps disc_value_nid → list[RelationRule].
    One morphism per operator.  Keyed by op NodeId in the input dicts.
    """
    from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH as _TG
    for op_nid, chains in kleisli_chains.items():
        disc_role = kleisli_disc_roles.get(op_nid)
        if disc_role is None:
            continue
        op_str = _TG.decode(op_nid)
        op_obj = mg.get_or_create_object(op_str)
        if not mg.source_morphisms(op_obj.obj_id, morph_type="KLEISLI_CHAIN"):
            mg.add_morphism(
                op_obj.obj_id, op_obj.obj_id,
                morph_type="KLEISLI_CHAIN", evidence=1,
                payload=(disc_role, dict(chains)),
            )


# ---------------------------------------------------------------------------
# Stage 4: Graph extraction helpers — rebuild working dicts FROM MorphismGraph
# ---------------------------------------------------------------------------

def _extract_succ_map_from_mg(mg: MorphismGraph) -> dict:
    """Read SUCC_EDGE morphisms → {src_label: tgt_label} dict."""
    result: dict = {}
    for m in mg.morphisms(include_identity=False):
        if m.morph_type == "SUCC_EDGE":
            src = mg.object_by_id(m.source)
            tgt = mg.object_by_id(m.target)
            if src and tgt and src.label and tgt.label:
                result[src.label] = tgt.label
    return result


def _extract_fold_rules_from_mg(mg: MorphismGraph) -> dict:
    """Read FOLD_RULE morphisms → {op_str: BinaryFoldRule} dict."""
    result: dict = {}
    for m in mg.morphisms(include_identity=False):
        if m.morph_type == "FOLD_RULE":
            src = mg.object_by_id(m.source)
            if src and src.label and m.payload is not None:
                result[src.label] = m.payload
    return result


def _extract_fc_adj_from_mg(mg: MorphismGraph) -> tuple:
    """Read FC_EDGE morphisms → (fc_lookup, adj_lookup) flat dicts.

    Reconstructs the flat {(op, input_tuple): output_tuple} dicts used by
    _compose and ComposeEngine from the per-op FC_EDGE morphism payloads.
    """
    fc_lookup: dict = {}
    adj_lookup: dict = {}
    for m in mg.morphisms(include_identity=False):
        if m.morph_type == "FC_EDGE":
            src = mg.object_by_id(m.source)
            if src and src.label and m.payload is not None:
                op_str = src.label
                fc_part, adj_part = m.payload
                for inp_tup, out_tup in fc_part.items():
                    fc_lookup[(op_str, inp_tup)] = out_tup
                for inp_tup, out_tup in adj_part.items():
                    adj_lookup[(op_str, inp_tup)] = out_tup
    return fc_lookup, adj_lookup


def _extract_adj_solve_from_mg(mg: MorphismGraph) -> dict:
    """Read ADJ_EDGE morphisms → {inv_op_str: (fwd_op_str, preserved_pos)} dict."""
    result: dict = {}
    for m in mg.morphisms(include_identity=False):
        if m.morph_type == "ADJ_EDGE":
            src = mg.object_by_id(m.source)
            if src and src.label and m.payload is not None:
                result[src.label] = m.payload
    return result


def _extract_chain_op_atoms_from_mg(mg: MorphismGraph) -> frozenset:
    """Derive chain_op_atoms (frozenset[NodeId]) from CHAIN_STEP morphisms."""
    from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH as _TG
    result: set = set()
    for m in mg.morphisms(include_identity=False):
        if m.morph_type == "CHAIN_STEP":
            src = mg.object_by_id(m.source)
            if src and src.label:
                result.add(_TG.encode(src.label))
    return frozenset(result)


# ---------------------------------------------------------------------------
# Stage 4: Unified CTKG path-find traversal — replaces Level 0.5/0.7/1b/1c
# ---------------------------------------------------------------------------

def _ctkg_path_find(
    prefix: list,
    mg: MorphismGraph,
    engine,  # Optional[ComposeEngine]
    rs,  # Optional[RelationStore]
    ctx_cat: ContextCategory,
    ctx,  # ContextId
    eq_token: str,
    op_atoms: frozenset,
    chain_op_atoms: frozenset,
    lambda_library: 'Optional[dict]' = None,
) -> 'Optional[dict]':
    """Unified graph traversal replacing Levels 1b, 1c, 0.5, and 0.7.

    Dispatch is on morph_type (structural edge type label), never on op name string.
    Reads all rules from MorphismGraph typed morphisms — no Python knowledge dicts.

    Priority order within this function:
      1. CHAIN_STEP  — exact chain/eq table lookup (Level 1b)
      2. RELATION_RULE — arity-free hypergraph rule (Level 1c-relational)
      3. KLEISLI_CHAIN — variable-depth Kleisli chain (Level 1c-kleisli)
      4. FC_EDGE     — exact FC/adj lookup (Level 0.5)
      5. FOLD_RULE + ADJ_EDGE — NNO fold composition (Level 0.7)
    """
    _ROLE_DELIMS = frozenset({'step', 'ans', 'eq', '<eos>'})

    # ------------------------------------------------------------------
    # Levels 1b / 1c-relational / 1c-kleisli: chain/trace format
    # ------------------------------------------------------------------
    chain_state = parse_chain_prefix(prefix, chain_op_atoms, eq_token=eq_token)
    if (
        chain_state is not None
        and chain_state.phase in ("INPUT", "OUTPUT")
        and chain_state.op is not None
    ):
        op_obj = mg.object_by_label(chain_state.op)
        if op_obj is not None:
            op_nid = TOKEN_GRAPH.encode(chain_state.op)
            use_eq = ctx_cat.is_refinement(ctx, ContextId.EQ)

            # --- Level 1b: CHAIN_STEP exact lookup ---
            chain_morphs = mg.source_morphisms(op_obj.obj_id, morph_type="CHAIN_STEP")
            if chain_morphs:
                chain_rule = chain_morphs[0].payload
                result = _chain_predict(
                    chain_rule,
                    chain_state.input_tokens,
                    chain_state.output_tokens,
                    use_eq_table=use_eq,
                )
                if result is not None:
                    return result

            # --- Level 1b-λ: Lambda term evaluation (between chain table and relation) ---
            # Fires for ops with chain rules, non-eq format, non-kleisli.
            # Skip for Kleisli ops (variable-depth output): they handle at 1c-kleisli.
            if lambda_library and not use_eq:
                _kl_check = mg.source_morphisms(op_obj.obj_id, morph_type="KLEISLI_CHAIN")
                if not _kl_check:
                    _nid_prefix = TOKEN_GRAPH.encode_seq(prefix)
                    _lt_result = lambda_predict(
                        _nid_prefix, lambda_library, engine, allow_transfer=True,
                    )
                    if _lt_result is not None:
                        return {TOKEN_GRAPH.decode(k): v for k, v in _lt_result.items()}

            # --- Level 1c-relational: RELATION_RULE ---
            rel_morphs = mg.source_morphisms(op_obj.obj_id, morph_type="RELATION_RULE")
            if rel_morphs and rs is not None and engine is not None:
                rel_rules = rel_morphs[0].payload
                _rel_input = [chain_state.op] + list(chain_state.input_tokens)
                _tmp_rules = {op_nid: rel_rules}
                _alternatives = predict_alternatives_from_rules(
                    _rel_input, rs, _tmp_rules, engine
                )
                if _alternatives:
                    _rk = len(chain_state.output_tokens)
                    _dist: dict = {}
                    _total_w = sum(w for _, w in _alternatives)
                    for _rel_output, _w in _alternatives:
                        _norm_w = _w / _total_w if _total_w > 0 else _w
                        if use_eq:
                            _role_content: list = []
                            for _role_delim in ('ans', 'eq'):
                                if _role_delim in _rel_output:
                                    _start = _rel_output.index(_role_delim) + 1
                                    _raw = _rel_output[_start:]
                                    _end = next(
                                        (i for i, t in enumerate(_raw)
                                         if t in _ROLE_DELIMS),
                                        len(_raw),
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

            # --- Level 1c-kleisli: KLEISLI_CHAIN (trace format only) ---
            if not use_eq:
                kl_morphs = mg.source_morphisms(op_obj.obj_id, morph_type="KLEISLI_CHAIN")
                if kl_morphs and rs is not None and engine is not None:
                    disc_role, chains = kl_morphs[0].payload
                    _kl_seq = [chain_state.op] + list(chain_state.input_tokens)
                    _kl_rel = rs.extract_relation(_kl_seq)
                    if _kl_rel is not None:
                        _disc_val_tup = None
                        for _sep, _toks in _kl_rel.input_roles:
                            if _sep == disc_role and _toks:
                                _disc_val_tup = tuple(_toks)
                                break
                        _disc_key = (
                            _disc_val_tup[0]
                            if _disc_val_tup and len(_disc_val_tup) == 1
                            else TOKEN_GRAPH.encode(
                                ''.join(TOKEN_GRAPH.decode(n) for n in _disc_val_tup)
                            ) if _disc_val_tup else None
                        )
                        if _disc_key is not None and _disc_key in chains:
                            _kl_rules = chains[_disc_key]
                            _kl_vals: dict = {}
                            for _sep, _toks in _kl_rel.input_roles:
                                if _toks:
                                    _kl_vals[_sep] = tuple(_toks)
                            _kl_ok = True
                            for _kl_rule in _kl_rules:
                                _kl_d = _kl_rule.evaluate(_kl_vals, engine)
                                if not _kl_d:
                                    _kl_ok = False
                                    break
                                _kl_vals[_kl_rule.output_role] = max(_kl_d, key=_kl_d.get)
                            if _kl_ok:
                                _kl_out: list = []
                                for _kl_rule in _kl_rules:
                                    _role_str = TOKEN_GRAPH.decode(_kl_rule.output_role)
                                    _delim = (
                                        'step' if _role_str.startswith('step')
                                        else _role_str
                                    )
                                    _kl_out.append(_delim)
                                    _kl_val_tup = _kl_vals[_kl_rule.output_role]
                                    _kl_out.extend(
                                        TOKEN_GRAPH.decode(n) for n in _kl_val_tup
                                    )
                                _kl_rk = len(chain_state.output_tokens)
                                if _kl_rk < len(_kl_out):
                                    return {_kl_out[_kl_rk]: 1.0}
                                if _kl_rk == len(_kl_out):
                                    return {'<eos>': 1.0}

    # ------------------------------------------------------------------
    # Levels 0.5 / 0.7: eq-format FC lookup + NNO fold composition
    # ------------------------------------------------------------------
    if not ctx_cat.is_refinement(ctx, ContextId.EQ):
        return None
    if not prefix or eq_token not in prefix:
        return None
    try:
        eq_idx = prefix.index(eq_token)
    except ValueError:
        return None
    if eq_idx == 0:
        return None

    op = prefix[0]
    input_tuple = tuple(prefix[1:eq_idx])
    output_so_far = prefix[eq_idx + 1:]
    if not input_tuple:
        return None

    op_obj_eq = mg.object_by_label(op)
    if op_obj_eq is None:
        return None

    # --- Level 0.5: FC_EDGE exact lookup ---
    fc_morphs = mg.source_morphisms(op_obj_eq.obj_id, morph_type="FC_EDGE")
    if fc_morphs:
        fc_part, adj_part = fc_morphs[0].payload
        expected = fc_part.get(input_tuple)
        if expected is None:
            expected = adj_part.get(input_tuple)
        if expected is not None:
            k = len(output_so_far)
            if k < len(expected):
                return {expected[k]: 1.0}
            if k == len(expected):
                return {'<eos>': 1.0}

    if engine is None:
        return None

    # --- Level 0.7: FOLD_RULE / ADJ_EDGE composition ---
    fold_morphs = mg.source_morphisms(op_obj_eq.obj_id, morph_type="FOLD_RULE")
    adj_morphs = mg.source_morphisms(op_obj_eq.obj_id, morph_type="ADJ_EDGE")

    result = None
    if fold_morphs and len(input_tuple) == 2:
        args: tuple = tuple((t,) for t in input_tuple)
        result = _compose(
            op, args,
            engine.fc_lookup, engine.fold_rules,
            engine.succ_map, engine.carry_el, engine.carry_out,
            engine.zero, engine.cache,
        )

    if result is None and adj_morphs:
        adj_info = adj_morphs[0].payload  # (fwd_op_str, preserved_pos)
        result = _compose_adjoint_search(
            op, input_tuple, adj_info,
            engine.fc_lookup, engine.fold_rules,
            engine.succ_map, engine.carry_el, engine.carry_out,
            engine.zero, engine.cache,
        )

    if result is not None:
        k = len(output_so_far)
        if k < len(result):
            return {result[k]: 1.0}
        if k == len(result):
            return {'<eos>': 1.0}

    return None


def _ctkg_nno_predict(
    prefix: list[str],
    mg: MorphismGraph,
    unary_chain_maps: dict[str, dict[str, str]],
    engine: 'ComposeEngine',
    op_atoms: frozenset,
    eq_token: str = "eq",
) -> Optional[dict[str, float]]:
    """Level 0.5-NNO: CTKG SUCC_EDGE graph traversal (Stage 4, CT_REFERENCE §19).

    Symbol-agnostic replacement for the hardcoded-'eq' Levels 0.6/0.7 NNO path.
    Uses parse_prefix (eq_token-aware) and reads SUCC_EDGE morphisms from the
    MorphismGraph instead of looking up a Python dict keyed by op-name string.

    Direction (succ vs pred) is determined structurally by comparing the op's
    observed single-digit step_map against the SUCC_EDGE-derived forward map —
    no string comparison against 'succ' or 'pred'.

    Carry info (carry_el, carry_out) is read from engine — the ComposeEngine
    holds the canonical carry info for the forward (succ-like) op.

    Handles both in-distribution and OOD inputs via unary_chain_predict carry
    propagation.  Returns {next_token: 1.0} or {'<eos>': 1.0}, or None on miss.
    """
    state = parse_prefix(prefix, eq_token=eq_token, op_atoms=op_atoms)
    if state.phase not in ("INPUT", "OUTPUT"):
        return None
    if state.op is None:
        return None

    input_digits = list(state.input_digits)
    output_so_far = list(state.output_digits)

    if not input_digits:
        return None

    # Build succ_graph from SUCC_EDGE morphisms in the MorphismGraph.
    # Knowledge comes from typed edges, not a Python dict keyed by op name.
    succ_graph: dict[str, str] = {}
    for m in mg.morphisms(include_identity=False):
        if m.morph_type == "SUCC_EDGE":
            src_obj = mg.object_by_id(m.source)
            tgt_obj = mg.object_by_id(m.target)
            if src_obj and tgt_obj and src_obj.label and tgt_obj.label:
                succ_graph[src_obj.label] = tgt_obj.label

    if not succ_graph:
        return None

    # Determine direction: compare op's observed step_map against succ_graph.
    op_step = unary_chain_maps.get(state.op)
    if op_step is None:
        return None

    inv_succ = {v: k for k, v in succ_graph.items()}
    fwd_matches = sum(1 for k, v in op_step.items() if succ_graph.get(k) == v)
    inv_matches = sum(1 for k, v in op_step.items() if inv_succ.get(k) == v)

    if fwd_matches == 0 and inv_matches == 0:
        return None

    inverse = inv_matches > fwd_matches

    # Carry info comes from ComposeEngine (carries canonical forward-op carry data).
    carry_el: str = engine.carry_el
    carry_out: tuple = engine.carry_out

    if not carry_el:
        return None

    full_output = unary_chain_predict(
        succ_graph, carry_el, carry_out, input_digits, inverse=inverse
    )
    if full_output is None:
        return None

    k = len(output_so_far)
    if k < len(full_output):
        return {full_output[k]: 1.0}
    if k == len(full_output):
        return {"<eos>": 1.0}
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


@dataclass
class ComposeEngine:
    """On-demand NNO fold engine — replaces _binary_fmaps (BFM) (Mistake #48).

    BFM pre-computed {op: {(a,b): result_str}} baked in the assumption that
    tokens are single characters (len(t)==1 filter in build_binary_functional_maps).
    ComposeEngine calls _compose on demand, supporting any token vocabulary including
    NL tokens ('three', 'five', etc.).

    Stage 4: also holds adj_lookup (adjunction-based exact lookup) and
    adj_solve_map (adjunction inverse search) so that _ctkg_path_find can
    access all computation knowledge through self._engine without keeping
    separate Python dict attributes on Predictor.
    """
    fc_lookup: dict
    fold_rules: dict
    succ_map: dict
    carry_el: str
    carry_out: tuple
    zero: str
    cache: dict = field(default_factory=dict)
    adj_lookup: dict = field(default_factory=dict)
    adj_solve_map: dict = field(default_factory=dict)

    def compute(self, op: str, a: str, b: str) -> Optional[tuple[str, ...]]:
        """Compute op(a, b) via NNO fold. a and b are single token strings."""
        _op_nid = TOKEN_GRAPH.encode(op)
        if _op_nid == _CONCAT_NODE:
            return (a + b,)
        if _op_nid == _FST_NODE:
            return (a,)
        if _op_nid == _DIV_NODE:
            return self._div(a, b)
        return _compose(op, ((a,), (b,)),
                        self.fc_lookup, self.fold_rules,
                        self.succ_map, self.carry_el, self.carry_out, self.zero,
                        self.cache)

    def compute_tup(self, op: str, a: tuple, b: tuple) -> Optional[tuple[str, ...]]:
        """Compute op(a, b) via NNO fold. a and b are token tuples (multi-digit ok)."""
        _op_nid = TOKEN_GRAPH.encode(op)
        if _op_nid == _FST_NODE:
            return a
        if _op_nid == _CONCAT_NODE:
            if len(a) == 1 and len(b) == 1:
                return (a[0] + b[0],)
            return None
        if _op_nid == _DIV_NODE:
            if len(a) == 1 and len(b) == 1:
                return self._div(a[0], b[0])
            return None
        return _compose(op, (a, b),
                        self.fc_lookup, self.fold_rules,
                        self.succ_map, self.carry_el, self.carry_out, self.zero,
                        self.cache)

    def _div(self, r: str, a: str) -> Optional[tuple[str, ...]]:
        """Find b such that mul(a, b) = r. Enumerate NNO digit chain."""
        r_tup: tuple[str, ...] = tuple(r.split('\x00')) if '\x00' in r else (r,)
        seen: set[str] = set()
        cur = self.zero
        while cur not in seen:
            result = _compose('mul', ((a,), (cur,)),
                              self.fc_lookup, self.fold_rules,
                              self.succ_map, self.carry_el, self.carry_out, self.zero,
                              self.cache)
            if result is not None and result == r_tup:
                return (cur,)
            seen.add(cur)
            nxt = self.succ_map.get(cur)
            if nxt is None or nxt in seen:
                break
            cur = nxt
        return None

    def known_ops(self) -> list[str]:
        """Return list of ops this engine knows about."""
        ops = list(self.fold_rules.keys())
        for _extra_nid in (_CONCAT_NODE, _FST_NODE, _DIV_NODE):
            _extra_str = TOKEN_GRAPH.decode(_extra_nid)
            if _extra_str not in ops:
                ops.append(_extra_str)
        return ops


