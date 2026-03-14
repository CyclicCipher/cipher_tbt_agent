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
    apply_process_rule,
    build_fc_lookup,
    build_adj_lookup,
    build_unary_chain_maps,
    build_unary_carry_maps,
    complete_succ_map,
    discover_binary_fold_rules,
    unary_chain_predict,
)
from dataclasses import dataclass, field
from experiments.symbolic_ai_v2.ctkg.core.kan_extension import KanExtension
from experiments.symbolic_ai_v2.ctkg.core.working_memory import (
    parse_prefix,
    parse_chain_prefix,
    WorkingMemory,
)
from experiments.symbolic_ai_v2.ctkg.core.spine import Spine


@dataclass
class TraceProgram:
    """Discovered computation program for a trace-format op (Level 1c).

    For trace ops like eval/linsolve, the output has the form:
        [step, *step_tokens, ans, *ans_tokens]
    or for two-step ops like bernoulli:
        [step, *step1_tokens, step, *step2_tokens, ans, *ans_tokens]

    All indices are into the list of a-prefixed operands extracted from the
    input_tokens (tokens matching 'a[0-9]+'; structural tokens like 'x', 'at'
    are ignored).  No operator name or semantic is hardcoded.
    """

    op_atom: str
    # Step computation: step = step_op(arg0, arg1)
    step_op: str
    step_arg0_idx: tuple               # indices into a_operands forming arg0
    step_arg1_idx: tuple               # indices into a_operands forming arg1
    # Ans computation
    ans_is_adj_search: bool            # True → enumerate x s.t. ans_op(other, x)=step
    ans_op: str                        # fold rule op (or fwd op for adj_search)
    ans_step_is_arg0: bool             # True → ans=fold(step, other); False → fold(other, step)
    ans_other_idx: tuple               # a-operand indices for the non-step arg
    evidence: int                      # training examples verified
    # Fix 1: zero-padding widths for output formatting
    step_width: int = 1                # zero-pad step output to this many tokens
    ans_width: int = 1                 # zero-pad ans output to this many tokens
    # Fix 3b: two-step extension (step2_op == "" means single-step program)
    step2_op: str = ""
    step2_arg0_idx: tuple = ()
    step2_arg1_idx: tuple = ()
    # For two-step: ans = ans_outer_op(inner, step_outer)
    #   inner = ans_op(step_inner_or_arg, arg_or_step_inner)
    #   step_inner is step[ans_inner_step] (1 or 2)
    #   step_outer is step[ans_outer_step] (1 or 2)
    ans_inner_step: int = 1            # which step feeds into inner fold
    ans_outer_op: str = ""             # outer fold op; "" = no outer (single-step ans)
    ans_outer_step: int = 2            # which step is outer fold arg
    ans_outer_inner_is_left: bool = True  # True: outer=outer_op(inner, step_outer)
    # Fix 3: ECHO_INPUT direct-fold ans (step_op == 'ECHO_INPUT')
    # When non-empty, ans = ans_op(a_ops[ans_other_idx], a_ops[ans_arg1_idx])
    ans_arg1_idx: tuple = ()

# Fixpoint iteration parameters (architecture §Prediction step 3)
_FP_MAX_ITER: int = 20
_FP_EPS: float = 1e-4
_FP_CYCLE_K: int = 5   # number of snapshots to keep for cycle detection


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
            # Level 1c: Trace program synthesis — discover computation structure
            # of trace-format ops (eval, linsolve, …) from training chain_table.
            # dict[op_atom → dict[n_operands → TraceProgram]] to handle variable arity.
            # Called once at init; result used as fallback when chain_table misses.
            self._trace_programs: dict[str, dict[int, TraceProgram]] = _discover_trace_programs(
                self._chain_rules,
                self._fold_rules,
                self._fc_lookup,
                self._compose_succ_map,
                self._compose_carry_el,
                self._compose_carry_out,
                self._compose_zero,
            )
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
            self._trace_programs: dict[str, dict[int, TraceProgram]] = {}

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

                # Level 1c: Trace program fallback (generalization for novel inputs).
                # Fires when chain_table lookup returned None (test input not in training).
                # Synthesized trace program computes step/ans via fold rules + adj_search.
                # Applies to both step-format and eq-format sequences (eq-format sequences
                # like linear_eval still use step/ans in their output).
                if self._trace_programs:
                    prog_map = self._trace_programs.get(chain_state.op)
                    # For eq-format sequences, skip trace programs when the op
                    # already has a fold rule (pow, add, mul, …) — fold rules are
                    # exact and structurally correct for those ops.  For ops without
                    # a fold rule (eval, linsolve, cs4, bern_p1, …), fire the trace
                    # program with ans_only=True so it generalises eq-format OOD.
                    op_has_fold = chain_state.op in self._fold_rules
                    if prog_map is not None and (not use_eq or not op_has_fold):
                        n_ops = sum(1 for t in chain_state.input_tokens if t.isdigit())
                        prog = prog_map.get(n_ops) if n_ops > 0 else None
                        if prog is not None:
                            trace_result = _trace_program_predict(
                                prog, chain_state.input_tokens,
                                chain_state.output_tokens,
                                self._fc_lookup, self._fold_rules,
                                self._compose_succ_map, self._compose_carry_el,
                                self._compose_carry_out, self._compose_zero,
                                self._compose_cache,
                                ans_only=use_eq,
                            )
                            if trace_result is not None:
                                return trace_result

        # Pass discovered op_atoms — not a hardcoded list
        state = parse_prefix(prefix, op_atoms=self._op_atoms)

        # Level 1a: Process rule (deterministic, fold-type)
        if state.phase == "OUTPUT" and state.op in self._rules:
            rule = self._rules[state.op]
            result = _process_predict(
                rule, state.input_digits, state.output_digits, self._rules
            )
            if result is not None:
                return result

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


# ---------------------------------------------------------------------------
# Internal helpers: process rule
# ---------------------------------------------------------------------------

def _process_predict(
    rule: ProcessRule,
    input_digits: list[str],
    output_digits_so_far: list[str],
    rules_dict: Optional[dict[str, ProcessRule]] = None,
) -> Optional[dict[str, float]]:
    """Return {next_digit: 1.0} or {'<eos>': 1.0} from process rule, or None."""
    full_output = apply_process_rule(rule, input_digits, rules_dict or {})
    if full_output is None:
        return None
    k = len(output_digits_so_far)
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
# Trace program synthesis (Level 1c) — CT_REFERENCE §19
# ---------------------------------------------------------------------------

def _lz_strip(toks: tuple) -> tuple:
    """Strip leading '0' tokens from a digit tuple (for zero-padded comparison).

    Examples:
        _lz_strip(('0', '8')) -> ('8',)
        _lz_strip(('0', '0')) -> ('0',)   # keep at least one digit
        _lz_strip(('1', '3')) -> ('1', '3')
    """
    i = 0
    while i < len(toks) - 1 and toks[i] == '0':
        i += 1
    return toks[i:]


def _zpad(toks: tuple, width: int) -> tuple:
    """Zero-pad a digit tuple to the given width.

    Examples:
        _zpad(('8',), 2)      -> ('0', '8')
        _zpad(('1', '3'), 2)  -> ('1', '3')
        _zpad(('5',), 1)      -> ('5',)
    """
    if len(toks) >= width:
        return toks
    return ('0',) * (width - len(toks)) + toks


def _discover_trace_programs(
    chain_rules: dict,
    fold_rules: dict,
    fc_lookup: dict,
    succ_map: dict,
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
) -> dict:
    """Synthesize TraceProgram for each trace-format chain op.

    For each op in chain_rules with a non-empty chain_table, analyzes
    training (input → step/ans output) examples to discover which fold
    rule compositions explain the step and ans tokens.

    No operator names or semantics are hardcoded.  Discovery is purely
    data-driven: we try all (fold_op × arg_positions) combinations and
    keep those that are 100% consistent with training examples.

    Returns dict mapping op_atom → dict[n_operands → TraceProgram].
    """
    cache: dict = {}
    programs: dict = {}

    for op_atom, cr in chain_rules.items():
        if not cr.chain_table:
            continue

        # --- Parse training samples, splitting by step-count format ---
        parsed_1step: list = []   # (a_ops, step_toks, ans_toks)
        parsed_2step: list = []   # (a_ops, step1_toks, step2_toks, ans_toks)

        for input_key, output_toks in cr.chain_table.items():
            if "ans" not in output_toks:
                continue
            ans_idx = output_toks.index("ans")
            ans_toks = tuple(output_toks[ans_idx + 1:])

            # Collect all 'step' positions before 'ans'
            step_positions = [
                i for i, t in enumerate(output_toks[:ans_idx]) if t == "step"
            ]

            # operands: digit tokens in the input (positional)
            a_ops = [tok for tok in input_key if tok.isdigit()]
            if len(a_ops) < 2 or not ans_toks:
                continue

            if len(step_positions) == 1:
                step_toks = tuple(output_toks[step_positions[0] + 1:ans_idx])
                if step_toks:
                    parsed_1step.append((a_ops, step_toks, ans_toks))
            elif len(step_positions) == 2:
                s1 = step_positions[0]
                s2 = step_positions[1]
                step1_toks = tuple(output_toks[s1 + 1:s2])
                step2_toks = tuple(output_toks[s2 + 1:ans_idx])
                if step1_toks and step2_toks:
                    parsed_2step.append((a_ops, step1_toks, step2_toks, ans_toks))
            # Ops with 0 or >2 steps are not yet handled

        from collections import Counter as _Counter

        # Discover single-step programs
        if len(parsed_1step) >= 3:
            n_counts = _Counter(len(s[0]) for s in parsed_1step)
            for n, n_count in n_counts.items():
                if n < 2 or n_count < 3:
                    continue
                _discover_one_arity(
                    op_atom, n,
                    [s for s in parsed_1step if len(s[0]) == n],
                    fold_rules, fc_lookup,
                    succ_map, carry_el, carry_out, zero_digit,
                    cache, programs,
                )

        # Discover two-step programs
        if len(parsed_2step) >= 3:
            n_counts = _Counter(len(s[0]) for s in parsed_2step)
            for n, n_count in n_counts.items():
                if n < 2 or n_count < 3:
                    continue
                _discover_two_step_arity(
                    op_atom, n,
                    [s for s in parsed_2step if len(s[0]) == n],
                    fold_rules, fc_lookup,
                    succ_map, carry_el, carry_out, zero_digit,
                    cache, programs,
                )

    return programs


def _discover_one_arity(
    op_atom: str,
    n: int,
    parsed: list,
    fold_rules: dict,
    fc_lookup: dict,
    succ_map: dict,
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
    cache: dict,
    programs: dict,
) -> None:
    """Discover and register a single-step TraceProgram for one specific arity n.

    Uses _lz_strip for comparisons so zero-padded training outputs (e.g. '08')
    match unpadded _compose results (e.g. '8').  Allows i==j for self-application
    like sq/bernoulli step.  Also tries both-2-token-groups for step (Fix 4).
    """

    # Determine expected step_width from training data.
    # Only use a non-trivial width (>1) if ALL training examples have the
    # same step length — that signals intentional zero-padding (e.g. _v2()).
    # Mixed lengths mean the step is not zero-padded; use width=1.
    all_step_lens = [len(s[1]) for s in parsed]
    if all_step_lens and len(set(all_step_lens)) == 1:
        step_width = all_step_lens[0]
    else:
        step_width = 1

    # Same logic for ans_width.
    all_ans_lens = [len(s[2]) for s in parsed]
    if all_ans_lens and len(set(all_ans_lens)) == 1:
        ans_width = all_ans_lens[0]
    else:
        ans_width = 1

    def _step_matches(result, expected):
        """Compare _compose result against (possibly zero-padded) training step."""
        if result is None:
            return False
        return _lz_strip(result) == _lz_strip(expected)

    # --- Discover step program ---
    # First check echo patterns before trying fold rules.
    # ECHO_INPUT: step tokens == the operand tokens verbatim (e.g. sq echoes its arg)
    # ECHO_ARG:   step tokens == (a_ops[k],) for some single operand k
    step_prog = None

    if all(s_toks == tuple(ao)
           for ao, s_toks, _ in parsed):
        step_prog = ('ECHO_INPUT', (), ())

    if step_prog is None:
        for k in range(n):
            if all(_lz_strip((ao[k],)) == _lz_strip(s_toks)
                   for ao, s_toks, _ in parsed):
                step_prog = ('ECHO_ARG', (k,), ())
                break

    # Try (fold_op, arg_i, arg_j) with single-token args (allow i==j for sq/bern).
    if step_prog is None:
        for fop in fold_rules:
            for i in range(n):
                for j in range(n):
                    # Fix 3a: allow i==j for self-application like mul(v, v)
                    ev = sum(
                        1 for a_ops, step, _ in parsed
                        if _step_matches(
                            _compose(
                                fop, ((a_ops[i],), (a_ops[j],)),
                                fc_lookup, fold_rules,
                                succ_map, carry_el, carry_out, zero_digit, cache,
                            ),
                            step,
                        )
                    )
                    if ev == len(parsed):
                        step_prog = (fop, (i,), (j,))
                        break
                if step_prog:
                    break
            if step_prog:
                break

    # Try 2-token grouped args: one group + one single.
    if step_prog is None:
        for fop in fold_rules:
            for gi in range(n - 1):
                for j in range(n):
                    if j in (gi, gi + 1):
                        continue
                    # Group (gi, gi+1) as arg0, j as arg1
                    ev = sum(
                        1 for a_ops, step, _ in parsed
                        if _step_matches(
                            _compose(
                                fop,
                                ((a_ops[gi], a_ops[gi + 1]), (a_ops[j],)),
                                fc_lookup, fold_rules,
                                succ_map, carry_el, carry_out, zero_digit, cache,
                            ),
                            step,
                        )
                    )
                    if ev == len(parsed):
                        step_prog = (fop, (gi, gi + 1), (j,))
                        break
                    # j as arg0, group (gi, gi+1) as arg1
                    ev = sum(
                        1 for a_ops, step, _ in parsed
                        if _step_matches(
                            _compose(
                                fop,
                                ((a_ops[j],), (a_ops[gi], a_ops[gi + 1])),
                                fc_lookup, fold_rules,
                                succ_map, carry_el, carry_out, zero_digit, cache,
                            ),
                            step,
                        )
                    )
                    if ev == len(parsed):
                        step_prog = (fop, (j,), (gi, gi + 1))
                        break
                if step_prog:
                    break
            if step_prog:
                break

    # Fix 4: Try both args as 2-token groups (e.g. cs1/cs2: add(c, d)).
    if step_prog is None:
        for fop in fold_rules:
            for gi in range(n - 1):
                for gj in range(gi + 2, n - 1):
                    ev = sum(
                        1 for a_ops, step, _ in parsed
                        if _step_matches(
                            _compose(
                                fop,
                                (
                                    (a_ops[gi], a_ops[gi + 1]),
                                    (a_ops[gj], a_ops[gj + 1]),
                                ),
                                fc_lookup, fold_rules,
                                succ_map, carry_el, carry_out, zero_digit, cache,
                            ),
                            step,
                        )
                    )
                    if ev == len(parsed):
                        step_prog = (fop, (gi, gi + 1), (gj, gj + 1))
                        break
                if step_prog:
                    break
            if step_prog:
                break

    if step_prog is None:
        return

    step_op_name, step_a0_idx, step_a1_idx = step_prog
    step_is_echo_input = (step_op_name == 'ECHO_INPUT')
    step_is_echo_arg   = (step_op_name == 'ECHO_ARG')

    # Pre-compute actual step results (unpadded) for all samples.
    # ECHO_INPUT: step = literal input tokens (not digit tuples) — None here.
    # ECHO_ARG:   step = (a_ops[k],) for the specified k.
    # Otherwise:  step = _compose(fold_op, ...).
    step_results = []
    for a_ops, _, _ in parsed:
        if step_is_echo_arg:
            sr = (a_ops[step_a0_idx[0]],)
        elif step_is_echo_input:
            sr = None  # not a digit tuple; handled in direct-fold ans synthesis
        else:
            a0 = tuple(a_ops[i] for i in step_a0_idx)
            a1 = tuple(a_ops[i] for i in step_a1_idx)
            sr = _compose(step_op_name, (a0, a1), fc_lookup, fold_rules,
                          succ_map, carry_el, carry_out, zero_digit, cache)
        step_results.append(sr)

    # --- Discover ans program (given step) ---
    ans_prog = None

    # ECHO_INPUT ans: step is not a digit tuple — try fold_op(a_group0, a_group1)
    # directly from a_ops (ignoring the step result). Creates program directly.
    if step_is_echo_input:
        for fop in fold_rules:
            found = False
            for w0 in (1, 2):
                for s0 in range(n - w0 + 1):
                    idx0 = tuple(range(s0, s0 + w0))
                    for w1 in (1, 2):
                        for s1 in range(n - w1 + 1):
                            idx1 = tuple(range(s1, s1 + w1))
                            ev = sum(
                                1 for a_ops, _, ans_t in parsed
                                if _step_matches(
                                    _compose(
                                        fop,
                                        (tuple(a_ops[i] for i in idx0),
                                         tuple(a_ops[i] for i in idx1)),
                                        fc_lookup, fold_rules,
                                        succ_map, carry_el, carry_out, zero_digit, cache,
                                    ),
                                    ans_t,
                                )
                            )
                            if ev == len(parsed):
                                found = True
                                prog = TraceProgram(
                                    op_atom=op_atom,
                                    step_op='ECHO_INPUT',
                                    step_arg0_idx=(),
                                    step_arg1_idx=(),
                                    ans_is_adj_search=False,
                                    ans_op=fop,
                                    ans_step_is_arg0=False,
                                    ans_other_idx=idx0,
                                    ans_arg1_idx=idx1,
                                    evidence=len(parsed),
                                    step_width=step_width,
                                    ans_width=ans_width,
                                )
                                programs.setdefault(op_atom, {})[n] = prog
                                return
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
        return  # ECHO_INPUT but no ans found

    # Fold-rule ans synthesis (for non-ECHO_INPUT steps):
    # Try fold(step, a-op[k]) and fold(a-op[k], step) — single-token k
    for fop in fold_rules:
        for k in range(n):
            # step first
            ev = sum(
                1 for (a_ops, _, ans), sr in zip(parsed, step_results)
                if sr is not None and _step_matches(
                    _compose(
                        fop, (sr, (a_ops[k],)),
                        fc_lookup, fold_rules,
                        succ_map, carry_el, carry_out, zero_digit, cache,
                    ),
                    ans,
                )
            )
            if ev == len(parsed):
                ans_prog = (fop, True, (k,), False, 1)  # b_width=1
                break
            # other first
            ev = sum(
                1 for (a_ops, _, ans), sr in zip(parsed, step_results)
                if sr is not None and _step_matches(
                    _compose(
                        fop, ((a_ops[k],), sr),
                        fc_lookup, fold_rules,
                        succ_map, carry_el, carry_out, zero_digit, cache,
                    ),
                    ans,
                )
            )
            if ev == len(parsed):
                ans_prog = (fop, False, (k,), False, 1)
                break
        if ans_prog:
            break

    # Fix 4: Try fold with 2-token group as other arg
    if ans_prog is None:
        for fop in fold_rules:
            for gi in range(n - 1):
                other_idx = (gi, gi + 1)
                # step first, 2-token other
                ev = sum(
                    1 for (a_ops, _, ans), sr in zip(parsed, step_results)
                    if sr is not None and _step_matches(
                        _compose(
                            fop,
                            (sr, (a_ops[gi], a_ops[gi + 1])),
                            fc_lookup, fold_rules,
                            succ_map, carry_el, carry_out, zero_digit, cache,
                        ),
                        ans,
                    )
                )
                if ev == len(parsed):
                    ans_prog = (fop, True, other_idx, False, 2)
                    break
                # other first, 2-token other
                ev = sum(
                    1 for (a_ops, _, ans), sr in zip(parsed, step_results)
                    if sr is not None and _step_matches(
                        _compose(
                            fop,
                            ((a_ops[gi], a_ops[gi + 1]), sr),
                            fc_lookup, fold_rules,
                            succ_map, carry_el, carry_out, zero_digit, cache,
                        ),
                        ans,
                    )
                )
                if ev == len(parsed):
                    ans_prog = (fop, False, other_idx, False, 2)
                    break
            if ans_prog:
                break

    # Try adj_search: enumerate x s.t. fold(x, b) = step; b is single-token
    if ans_prog is None:
        for fop in fold_rules:
            for k in range(n):
                ev = sum(
                    1 for (a_ops, _, ans), sr in zip(parsed, step_results)
                    if sr is not None and _step_matches(
                        _compose_adjoint_search(
                            "adj_search",
                            sr + (a_ops[k],),
                            (fop, 1),
                            fc_lookup, fold_rules,
                            succ_map, carry_el, carry_out, zero_digit, cache,
                        ),
                        ans,
                    )
                )
                if ev == len(parsed):
                    ans_prog = (fop, False, (k,), True, 1)
                    break
            if ans_prog:
                break

    # Fix 4: adj_search with 2-token preserved arg
    if ans_prog is None:
        for fop in fold_rules:
            for gi in range(n - 1):
                other_idx = (gi, gi + 1)
                ev = sum(
                    1 for (a_ops, _, ans), sr in zip(parsed, step_results)
                    if sr is not None and _step_matches(
                        _compose_adjoint_search(
                            "adj_search",
                            sr + (a_ops[gi], a_ops[gi + 1]),
                            (fop, 1),
                            fc_lookup, fold_rules,
                            succ_map, carry_el, carry_out, zero_digit, cache,
                            b_width=2,
                        ),
                        ans,
                    )
                )
                if ev == len(parsed):
                    ans_prog = (fop, False, other_idx, True, 2)
                    break
            if ans_prog:
                break

    if ans_prog is None:
        return

    ans_op_name, ans_step_first, ans_other_idx, ans_is_adj, _b_width = ans_prog
    prog = TraceProgram(
        op_atom=op_atom,
        step_op=step_op_name,
        step_arg0_idx=step_a0_idx,
        step_arg1_idx=step_a1_idx,
        ans_is_adj_search=ans_is_adj,
        ans_op=ans_op_name,
        ans_step_is_arg0=ans_step_first,
        ans_other_idx=ans_other_idx,
        evidence=len(parsed),
        step_width=step_width,
        ans_width=ans_width,
    )
    # Store keyed by (op_atom, n) to support variable-arity ops
    programs.setdefault(op_atom, {})[n] = prog


def _discover_two_step_arity(
    op_atom: str,
    n: int,
    parsed: list,   # (a_ops, step1_toks, step2_toks, ans_toks)
    fold_rules: dict,
    fc_lookup: dict,
    succ_map: dict,
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
    cache: dict,
    programs: dict,
) -> None:
    """Discover and register a two-step TraceProgram for one arity n.

    Handles outputs with format: step <step1> step <step2> ans <ans>
    e.g. Bernoulli: step V1^2 step V2^2 ans P2

    Synthesizes step1, step2, and an ans chain:
        inner = ans_op(step_inner, other) or ans_op(other, step_inner)
        ans   = ans_outer_op(inner, step_outer) or ans_outer_op(step_outer, inner)
    """

    def _matches(result, expected):
        if result is None:
            return False
        return _lz_strip(result) == _lz_strip(expected)

    all_ans_lens_2s = [len(s[3]) for s in parsed]
    if all_ans_lens_2s and len(set(all_ans_lens_2s)) == 1:
        ans_width = all_ans_lens_2s[0]
    else:
        ans_width = 1

    # ---- Step1 synthesis (allow i==j) ----
    step1_prog = None
    for fop in fold_rules:
        for i in range(n):
            for j in range(n):
                ev = sum(
                    1 for a_ops, s1, s2, ans in parsed
                    if _matches(
                        _compose(fop, ((a_ops[i],), (a_ops[j],)),
                                 fc_lookup, fold_rules,
                                 succ_map, carry_el, carry_out, zero_digit, cache),
                        s1,
                    )
                )
                if ev == len(parsed):
                    step1_prog = (fop, (i,), (j,))
                    break
            if step1_prog:
                break
        if step1_prog:
            break

    # ECHO_ARG fallback for step1: step = a_ops[k] for some k
    if step1_prog is None:
        for k in range(n):
            if all(_lz_strip((a_ops[k],)) == _lz_strip(s1)
                   for a_ops, s1, s2, ans in parsed):
                step1_prog = ('ECHO_ARG', (k,), ())
                break

    if step1_prog is None:
        return

    s1_op, s1_a0_idx, s1_a1_idx = step1_prog

    # ---- Step2 synthesis (allow i==j) ----
    step2_prog = None
    for fop in fold_rules:
        for i in range(n):
            for j in range(n):
                ev = sum(
                    1 for a_ops, s1, s2, ans in parsed
                    if _matches(
                        _compose(fop, ((a_ops[i],), (a_ops[j],)),
                                 fc_lookup, fold_rules,
                                 succ_map, carry_el, carry_out, zero_digit, cache),
                        s2,
                    )
                )
                if ev == len(parsed):
                    step2_prog = (fop, (i,), (j,))
                    break
            if step2_prog:
                break
        if step2_prog:
            break

    if step2_prog is None:
        return

    s2_op, s2_a0_idx, s2_a1_idx = step2_prog

    # Pre-compute step results
    s1_is_echo_arg = (s1_op == 'ECHO_ARG')
    s2_is_echo_arg = (s2_op == 'ECHO_ARG')
    step1_results = []
    step2_results = []
    for a_ops, _, _, _ in parsed:
        if s1_is_echo_arg:
            step1_results.append((a_ops[s1_a0_idx[0]],))
        else:
            a0 = tuple(a_ops[i] for i in s1_a0_idx)
            a1 = tuple(a_ops[i] for i in s1_a1_idx)
            step1_results.append(_compose(s1_op, (a0, a1), fc_lookup, fold_rules,
                                          succ_map, carry_el, carry_out, zero_digit, cache))
        if s2_is_echo_arg:
            step2_results.append((a_ops[s2_a0_idx[0]],))
        else:
            b0 = tuple(a_ops[i] for i in s2_a0_idx)
            b1 = tuple(a_ops[i] for i in s2_a1_idx)
            step2_results.append(_compose(s2_op, (b0, b1), fc_lookup, fold_rules,
                                          succ_map, carry_el, carry_out, zero_digit, cache))

    # ---- Ans synthesis: inner = inner_op(step_j, other) outer = outer_op(inner, step_k) ----
    # Try all combinations: inner_step ∈ {1,2}, outer_step ∈ {1,2}, inner_step ≠ outer_step,
    # inner_op ∈ fold_rules, outer_op ∈ fold_rules, k ∈ range(n), left/right orderings.
    ans_prog = None

    step_results_by_idx = {1: step1_results, 2: step2_results}

    for inner_step_idx in (1, 2):
        outer_step_idx = 3 - inner_step_idx  # the other one
        inner_sr_list = step_results_by_idx[inner_step_idx]
        outer_sr_list = step_results_by_idx[outer_step_idx]

        for inner_op in fold_rules:
            for k in range(n):
                # inner: inner_op(step_inner, a[k])
                inner_vals_a = [
                    _compose(inner_op, (sr, (a_ops[k],)),
                             fc_lookup, fold_rules,
                             succ_map, carry_el, carry_out, zero_digit, cache)
                    for (a_ops, _, _, _), sr in zip(parsed, inner_sr_list)
                ]
                # inner: inner_op(a[k], step_inner)
                inner_vals_b = [
                    _compose(inner_op, ((a_ops[k],), sr),
                             fc_lookup, fold_rules,
                             succ_map, carry_el, carry_out, zero_digit, cache)
                    for (a_ops, _, _, _), sr in zip(parsed, inner_sr_list)
                ]

                for inner_step_is_arg0, inner_vals in (
                    (True, inner_vals_a), (False, inner_vals_b)
                ):
                    for outer_op in fold_rules:
                        # outer: outer_op(inner, step_outer)
                        ev = sum(
                            1 for (_, _, _, ans), iv, osr in
                            zip(parsed, inner_vals, outer_sr_list)
                            if iv is not None and osr is not None and _matches(
                                _compose(outer_op, (iv, osr),
                                         fc_lookup, fold_rules,
                                         succ_map, carry_el, carry_out, zero_digit, cache),
                                ans,
                            )
                        )
                        if ev == len(parsed):
                            ans_prog = (inner_op, inner_step_idx, inner_step_is_arg0, (k,),
                                        outer_op, outer_step_idx, True)
                            break
                        # outer: outer_op(step_outer, inner)
                        ev = sum(
                            1 for (_, _, _, ans), iv, osr in
                            zip(parsed, inner_vals, outer_sr_list)
                            if iv is not None and osr is not None and _matches(
                                _compose(outer_op, (osr, iv),
                                         fc_lookup, fold_rules,
                                         succ_map, carry_el, carry_out, zero_digit, cache),
                                ans,
                            )
                        )
                        if ev == len(parsed):
                            ans_prog = (inner_op, inner_step_idx, inner_step_is_arg0, (k,),
                                        outer_op, outer_step_idx, False)
                            break
                    if ans_prog:
                        break
                if ans_prog:
                    break
            if ans_prog:
                break
        if ans_prog:
            break

    # Passthrough inner: inner = step directly (no fold), then outer = outer_op(inner, step_outer)
    # Handles cases like pow e=3: ans = mul(step1=base, step2=base^2) = mul(step2, step1)
    if ans_prog is None:
        for inner_step_idx in (1, 2):
            outer_step_idx = 3 - inner_step_idx
            inner_sr_list = step_results_by_idx[inner_step_idx]
            outer_sr_list = step_results_by_idx[outer_step_idx]
            for outer_op in fold_rules:
                ev = sum(
                    1 for (_, _, _, ans), iv, osr in
                    zip(parsed, inner_sr_list, outer_sr_list)
                    if iv is not None and osr is not None and _matches(
                        _compose(outer_op, (iv, osr),
                                 fc_lookup, fold_rules,
                                 succ_map, carry_el, carry_out, zero_digit, cache),
                        ans,
                    )
                )
                if ev == len(parsed):
                    ans_prog = ('', inner_step_idx, True, (),
                                outer_op, outer_step_idx, True)
                    break
                ev = sum(
                    1 for (_, _, _, ans), iv, osr in
                    zip(parsed, inner_sr_list, outer_sr_list)
                    if iv is not None and osr is not None and _matches(
                        _compose(outer_op, (osr, iv),
                                 fc_lookup, fold_rules,
                                 succ_map, carry_el, carry_out, zero_digit, cache),
                        ans,
                    )
                )
                if ev == len(parsed):
                    ans_prog = ('', inner_step_idx, True, (),
                                outer_op, outer_step_idx, False)
                    break
            if ans_prog:
                break

    # 2-token inner_other_idx: handles Bernoulli-style P1 which is a 2-digit argument
    if ans_prog is None:
        for inner_step_idx in (1, 2):
            outer_step_idx = 3 - inner_step_idx
            inner_sr_list = step_results_by_idx[inner_step_idx]
            outer_sr_list = step_results_by_idx[outer_step_idx]
            for inner_op in fold_rules:
                for gi in range(n - 1):
                    inner_other_idx = (gi, gi + 1)
                    inner_vals_a = [
                        _compose(inner_op, (sr, (a_ops[gi], a_ops[gi + 1])),
                                 fc_lookup, fold_rules,
                                 succ_map, carry_el, carry_out, zero_digit, cache)
                        for (a_ops, _, _, _), sr in zip(parsed, inner_sr_list)
                    ]
                    inner_vals_b = [
                        _compose(inner_op, ((a_ops[gi], a_ops[gi + 1]), sr),
                                 fc_lookup, fold_rules,
                                 succ_map, carry_el, carry_out, zero_digit, cache)
                        for (a_ops, _, _, _), sr in zip(parsed, inner_sr_list)
                    ]
                    for inner_step_is_arg0, inner_vals in (
                        (True, inner_vals_a), (False, inner_vals_b)
                    ):
                        for outer_op in fold_rules:
                            ev = sum(
                                1 for (_, _, _, ans), iv, osr in
                                zip(parsed, inner_vals, outer_sr_list)
                                if iv is not None and osr is not None and _matches(
                                    _compose(outer_op, (iv, osr),
                                             fc_lookup, fold_rules,
                                             succ_map, carry_el, carry_out, zero_digit, cache),
                                    ans,
                                )
                            )
                            if ev == len(parsed):
                                ans_prog = (inner_op, inner_step_idx, inner_step_is_arg0,
                                            inner_other_idx, outer_op, outer_step_idx, True)
                                break
                            ev = sum(
                                1 for (_, _, _, ans), iv, osr in
                                zip(parsed, inner_vals, outer_sr_list)
                                if iv is not None and osr is not None and _matches(
                                    _compose(outer_op, (osr, iv),
                                             fc_lookup, fold_rules,
                                             succ_map, carry_el, carry_out, zero_digit, cache),
                                    ans,
                                )
                            )
                            if ev == len(parsed):
                                ans_prog = (inner_op, inner_step_idx, inner_step_is_arg0,
                                            inner_other_idx, outer_op, outer_step_idx, False)
                                break
                        if ans_prog:
                            break
                    if ans_prog:
                        break
                if ans_prog:
                    break
            if ans_prog:
                break

    if ans_prog is None:
        return

    (inner_op, inner_step_i, inner_step_is_arg0, inner_other_idx,
     outer_op, outer_step_i, outer_inner_is_left) = ans_prog

    prog = TraceProgram(
        op_atom=op_atom,
        step_op=s1_op,
        step_arg0_idx=s1_a0_idx,
        step_arg1_idx=s1_a1_idx,
        ans_is_adj_search=False,
        ans_op=inner_op,
        ans_step_is_arg0=inner_step_is_arg0,
        ans_other_idx=inner_other_idx,
        evidence=len(parsed),
        step_width=1,
        ans_width=ans_width,
        step2_op=s2_op,
        step2_arg0_idx=s2_a0_idx,
        step2_arg1_idx=s2_a1_idx,
        ans_inner_step=inner_step_i,
        ans_outer_op=outer_op,
        ans_outer_step=outer_step_i,
        ans_outer_inner_is_left=outer_inner_is_left,
    )
    programs.setdefault(op_atom, {})[n] = prog


def _trace_program_predict(
    prog: "TraceProgram",
    input_tokens: list,
    output_tokens_so_far: list,
    fc_lookup: dict,
    fold_rules: dict,
    succ_map: dict,
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
    cache: dict,
    ans_only: bool = False,
) -> Optional[dict]:
    """Level 1c: predict next token using a discovered TraceProgram.

    Extracts a-prefixed operands from input_tokens, runs the step/ans
    computation using fold rules / adj_search, then returns a point mass
    for the k-th token of the expected trace output.

    Handles both single-step (step2_op=="") and two-step programs.
    Applies zero-padding to match training format (step_width, ans_width).
    """
    # Operands: digit tokens in the input, extracted by position.
    a_operands = [tok for tok in input_tokens if tok.isdigit()]
    if len(a_operands) < 2:
        return None

    # Validate that all required positions are in bounds
    idx_pool = (
        list(prog.step_arg0_idx) + list(prog.step_arg1_idx) +
        list(prog.ans_other_idx) + list(prog.ans_arg1_idx) +
        list(prog.step2_arg0_idx) + list(prog.step2_arg1_idx)
    )
    if idx_pool and max(idx_pool) >= len(a_operands):
        return None

    # ---- Compute step1 ----
    if prog.step_op == 'ECHO_ARG':
        if not prog.step_arg0_idx:
            return None
        step1_result = (a_operands[prog.step_arg0_idx[0]],)
        step1_padded = _zpad(step1_result, prog.step_width)
    elif prog.step_op == 'ECHO_INPUT':
        step1_result = None   # output is a-tokens, computed below
        step1_padded = ()     # unused in ECHO_INPUT path
    else:
        a0 = tuple(a_operands[i] for i in prog.step_arg0_idx)
        a1 = tuple(a_operands[i] for i in prog.step_arg1_idx)
        step1_result = _compose(prog.step_op, (a0, a1), fc_lookup, fold_rules,
                                succ_map, carry_el, carry_out, zero_digit, cache)
        if step1_result is None:
            return None
        step1_padded = _zpad(step1_result, prog.step_width)

    # ---- Two-step program ----
    if prog.step2_op:
        b0 = tuple(a_operands[i] for i in prog.step2_arg0_idx)
        b1 = tuple(a_operands[i] for i in prog.step2_arg1_idx)
        step2_result = _compose(prog.step2_op, (b0, b1), fc_lookup, fold_rules,
                                succ_map, carry_el, carry_out, zero_digit, cache)
        if step2_result is None:
            return None
        step2_padded = _zpad(step2_result, prog.step_width)

        step_results = {1: step1_result, 2: step2_result}

        # Compute inner: passthrough (no fold) or ans_op(step_inner, other)
        inner_sr = step_results[prog.ans_inner_step]
        if prog.ans_op == '':
            # Passthrough: inner = step directly (no inner fold)
            inner_val = inner_sr
        else:
            other = tuple(a_operands[i] for i in prog.ans_other_idx)
            if prog.ans_step_is_arg0:
                inner_val = _compose(prog.ans_op, (inner_sr, other),
                                     fc_lookup, fold_rules,
                                     succ_map, carry_el, carry_out, zero_digit, cache)
            else:
                inner_val = _compose(prog.ans_op, (other, inner_sr),
                                     fc_lookup, fold_rules,
                                     succ_map, carry_el, carry_out, zero_digit, cache)
        if inner_val is None:
            return None

        # Compute outer: ans_outer_op(inner, step_outer) or reversed
        outer_sr = step_results[prog.ans_outer_step]
        if prog.ans_outer_inner_is_left:
            ans_result = _compose(prog.ans_outer_op, (inner_val, outer_sr),
                                  fc_lookup, fold_rules,
                                  succ_map, carry_el, carry_out, zero_digit, cache)
        else:
            ans_result = _compose(prog.ans_outer_op, (outer_sr, inner_val),
                                  fc_lookup, fold_rules,
                                  succ_map, carry_el, carry_out, zero_digit, cache)
        if ans_result is None:
            return None

        ans_padded = _zpad(ans_result, prog.ans_width)
        full_output = (
            ["step"] + list(step1_padded) +
            ["step"] + list(step2_padded) +
            ["ans"] + list(ans_padded)
        )

    else:
        # ---- Single-step program ----
        if prog.step_op == 'ECHO_INPUT' and prog.ans_arg1_idx:
            # ECHO_INPUT: step tokens are a-prefixed input tokens.
            # Ans computed directly from a_operands (no step result needed).
            other = tuple(a_operands[i] for i in prog.ans_other_idx)
            arg1  = tuple(a_operands[i] for i in prog.ans_arg1_idx)
            ans_result = _compose(prog.ans_op, (other, arg1),
                                  fc_lookup, fold_rules,
                                  succ_map, carry_el, carry_out, zero_digit, cache)
        else:
            other = tuple(a_operands[i] for i in prog.ans_other_idx)
            if prog.ans_is_adj_search:
                b_width = len(prog.ans_other_idx)
                ans_result = _compose_adjoint_search(
                    "adj_search",
                    step1_result + other,
                    (prog.ans_op, 1),
                    fc_lookup, fold_rules,
                    succ_map, carry_el, carry_out, zero_digit, cache,
                    b_width=b_width,
                )
            elif prog.ans_step_is_arg0:
                ans_result = _compose(prog.ans_op, (step1_result, other),
                                      fc_lookup, fold_rules,
                                      succ_map, carry_el, carry_out, zero_digit, cache)
            else:
                ans_result = _compose(prog.ans_op, (other, step1_result),
                                      fc_lookup, fold_rules,
                                      succ_map, carry_el, carry_out, zero_digit, cache)

        if ans_result is None:
            return None

        ans_padded = _zpad(ans_result, prog.ans_width)
        if prog.step_op == 'ECHO_INPUT':
            step1_toks_out = list(a_operands)
        else:
            step1_toks_out = list(step1_padded)
        full_output = ["step"] + step1_toks_out + ["ans"] + list(ans_padded)

    # ans_only mode: for eq-format sequences (no step/ans in output)
    # output only the final answer digits, not the full trace.
    if ans_only:
        output_seq = list(ans_padded)
    else:
        output_seq = full_output

    k = len(output_tokens_so_far)
    if k < len(output_seq):
        return {output_seq[k]: 1.0}
    if k == len(output_seq):
        return {"<eos>": 1.0}
    return None
