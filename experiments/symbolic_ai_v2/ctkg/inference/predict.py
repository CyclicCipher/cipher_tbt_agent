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
class SlotProgram:
    """Discovered computation program using structural pattern matching (Level 1c).

    Replaces TraceProgram with a segment-based representation that:
    - Groups by input structural shape (pattern_key) instead of digit count
    - Handles N-step outputs uniformly (no separate one/two-step code paths)
    - Supports pattern-implied constants in output (e.g. same token in all
      examples with a given structural pattern)

    segments is a list of segment specs, one per variable-width block:
      ('K', tok)              — constant token (same in every training example)
      ('E',)                  — echo all input digit tokens verbatim
      ('G', j)                — single input digit at position j (ECHO_ARG)
      ('F', op, src_a, src_b, w) — fold_rule(src_a, src_b), zero-padded to w tokens
      ('A', op, step_src, oth, w) — adj_search result, zero-padded to w tokens

    src types used inside 'F' and 'A':
      ('I', j0, j1)   — input_digits[j0:j1+1]
      ('S', seg_idx)  — result tuple stored by prior segment seg_idx
    """
    op_atom: str
    pattern_key: tuple          # structural input shape (digits replaced by '_')
    segments: list              # list of segment specs (see above)
    evidence: int               # number of training examples verified

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
            # dict[op_atom → dict[pattern_key → list[SlotProgram]]] for variable structures.
            # Called once at init; result used as fallback when chain_table misses.
            self._trace_programs: dict[str, list] = _discover_trace_programs(
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
            self._trace_programs: dict[str, list] = {}

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
                        # Iterate all programs for this op; use _pattern_key_matches
                        # (supports literal digits at structural positions) to find
                        # compatible programs.  The first consistent one wins.
                        for prog in prog_map:
                            if not _pattern_key_matches(
                                prog.pattern_key, chain_state.input_tokens
                            ):
                                continue
                            trace_result = _slot_program_predict(
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


def _parse_into_value_blocks(output_toks: list) -> list:
    """Split output token list into constant ('K') and value ('V') segments.

    Returns a list of:
      ('K', tok)         — structural/constant token (step, ans, <eos>, or any
                           non-digit token embedded in a value region, e.g. 'mul', 'x')
      ('V', value_tuple) — run of consecutive digit tokens

    Non-digit tokens within a value region (e.g. 'mul' in 'ans mul 6 x')
    are treated as structural constants so that the discovery algorithm can
    work on the purely numeric sub-segments independently.
    """
    result = []
    i = 0
    while i < len(output_toks):
        tok = output_toks[i]
        if not tok.isdigit():
            result.append(('K', tok))
            i += 1
        else:
            j = i
            while j < len(output_toks) and output_toks[j].isdigit():
                j += 1
            result.append(('V', tuple(output_toks[i:j])))
            i = j
    return result


def _discover_value_segment(
    value_tuples: list,
    in_digits_list: list,
    seg_results: dict,
    fold_rules: dict,
    fc_lookup: dict,
    succ_map: dict,
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
    cache: dict,
) -> tuple:
    """Find what computation produces a variable value segment.

    Tries (in order):
      1. All-constant (all examples same value)
      2. Echo all input digits verbatim
      3. Echo single input digit at position j
      4. fold(in_slice, in_slice) — widths 1 and 2
      5. fold(seg_result, in_slice) and fold(in_slice, seg_result) — prior segments
      5b. fold(fold(seg, in_slice), seg) — nested fold for bernoulli-style 3-operand ans
      6. fold(seg_i, seg_j) — two prior computed segments
      7. adj_search(seg_result + in_slice) — adjoint search

    Returns (segment_spec, computed_results) or (None, None).
    Width w in 'F'/'A' specs is the minimum observed token count (for zero-padding).
    Using minimum avoids over-padding when ans widths vary (e.g., sq: 1..3 digits).
    """
    if not value_tuples:
        return None, None

    # Width for zero-padding: use consistent width only when ALL training examples
    # agree.  For variable-length outputs (e.g. sq ans, pow ans) use width=1 so
    # _zpad never adds spurious leading zeros.
    widths = [len(v) for v in value_tuples]
    width = widths[0] if len(set(widths)) == 1 else 1
    actual_n = len(in_digits_list[0]) if in_digits_list else 0

    def _matches(result, expected):
        if result is None:
            return False
        return _lz_strip(result) == _lz_strip(expected)

    # 1. All-constant block — only when ≥2 examples agree; deferred to last resort
    #    for single-example groups (where any value appears constant trivially).
    const_candidate = None
    if all(v == value_tuples[0] for v in value_tuples) and len(value_tuples) >= 2:
        const_candidate = value_tuples[0]
        return ('CONST_BLOCK', const_candidate), [const_candidate for _ in value_tuples]

    # 2. Echo all input digits verbatim
    if all(tuple(in_d) == v for in_d, v in zip(in_digits_list, value_tuples)):
        return ('E',), [tuple(in_d) for in_d in in_digits_list]

    # 3. Echo single input digit at position j
    for j in range(actual_n):
        if all(_lz_strip((in_d[j],)) == _lz_strip(v)
               for in_d, v in zip(in_digits_list, value_tuples)):
            computed = [(in_d[j],) for in_d in in_digits_list]
            return ('G', j), computed

    # 3b. pred / succ of a single input digit (NNO predecessor/successor).
    #     Covers ans = N-1 in derivative_trace pow x N output.
    if succ_map:
        pred_map = {v: k for k, v in succ_map.items()}
        for j in range(actual_n):
            # pred: pred_map[in[j]]
            pred_vals = [pred_map.get(in_d[j]) for in_d in in_digits_list]
            if all(pv is not None and _lz_strip((pv,)) == _lz_strip(v)
                   for pv, v in zip(pred_vals, value_tuples)):
                return ('P', j), [(pv,) for pv in pred_vals]
            # succ: succ_map[in[j]]
            succ_vals = [succ_map.get(in_d[j]) for in_d in in_digits_list]
            if all(sv is not None and _lz_strip((sv,)) == _lz_strip(v)
                   for sv, v in zip(succ_vals, value_tuples)):
                return ('SC', j), [(sv,) for sv in succ_vals]

    # 4. fold(in_slice, in_slice)
    for fop in fold_rules:
        for w0 in (1, 2):
            for s0 in range(actual_n - w0 + 1):
                src_a = ('I', s0, s0 + w0 - 1)
                for w1 in (1, 2):
                    for s1 in range(actual_n - w1 + 1):
                        src_b = ('I', s1, s1 + w1 - 1)
                        computed = [
                            _compose(fop,
                                     (tuple(in_d[s0:s0 + w0]), tuple(in_d[s1:s1 + w1])),
                                     fc_lookup, fold_rules,
                                     succ_map, carry_el, carry_out, zero_digit, cache)
                            for in_d in in_digits_list
                        ]
                        if all(_matches(c, v) for c, v in zip(computed, value_tuples)):
                            return ('F', fop, src_a, src_b, width), computed

    # 4b. adj_search(in_slice, K_const): find x where fold_op(x, K) = in[j:j+w].
    #     Covers integral step = R where mul(R, m) = RM (m is pattern-implied).
    #     Tries small constants K ∈ {'2'..'9'} as the divisor.
    for fop in fold_rules:
        for w0 in (1, 2):
            for s0 in range(actual_n - w0 + 1):
                src = ('I', s0, s0 + w0 - 1)
                for k_digit in ('2', '3', '4', '5', '6', '7', '8', '9'):
                    k_tuple = (k_digit,)
                    computed = []
                    ok = True
                    for in_d, val in zip(in_digits_list, value_tuples):
                        in_slice = tuple(in_d[s0:s0 + w0])
                        result = _compose_adjoint_search(
                            "adj_search", in_slice + k_tuple, (fop, 1),
                            fc_lookup, fold_rules,
                            succ_map, carry_el, carry_out, zero_digit, cache,
                            b_width=1,
                        )
                        if result is None or not _matches(result, val):
                            ok = False
                            break
                        computed.append(result)
                    if ok and computed:
                        return ('AK', fop, src, k_digit, width), computed

    # 5. fold(seg, in_slice) and fold(in_slice, seg)
    for seg_idx, seg_result_list in seg_results.items():
        if any(r is None for r in seg_result_list):
            continue
        for fop in fold_rules:
            for w1 in (1, 2):
                for s1 in range(actual_n - w1 + 1):
                    src_b = ('I', s1, s1 + w1 - 1)
                    computed = [
                        _compose(fop, (sr, tuple(in_d[s1:s1 + w1])),
                                 fc_lookup, fold_rules,
                                 succ_map, carry_el, carry_out, zero_digit, cache)
                        for sr, in_d in zip(seg_result_list, in_digits_list)
                    ]
                    if all(_matches(c, v) for c, v in zip(computed, value_tuples)):
                        return ('F', fop, ('S', seg_idx), src_b, width), computed
                    computed = [
                        _compose(fop, (tuple(in_d[s1:s1 + w1]), sr),
                                 fc_lookup, fold_rules,
                                 succ_map, carry_el, carry_out, zero_digit, cache)
                        for sr, in_d in zip(seg_result_list, in_digits_list)
                    ]
                    if all(_matches(c, v) for c, v in zip(computed, value_tuples)):
                        return ('F', fop, src_b, ('S', seg_idx), width), computed

    # 5b. fold(fold(seg_i, in_slice), seg_j) — nested fold for bernoulli-style 3-operand ans
    # Handles: ans = outer_op(inner_op(seg_i, in_slice), seg_j) and variants
    seg_idxs_nf = [k for k, v in seg_results.items() if not any(r is None for r in v)]
    for seg_idx_i in seg_idxs_nf:
        si_list = seg_results[seg_idx_i]
        for fop_inner in fold_rules:
            for w1 in (1, 2):
                for s1 in range(actual_n - w1 + 1):
                    oth_src = ('I', s1, s1 + w1 - 1)
                    for inner_left in (True, False):
                        if inner_left:
                            inner_vals = [
                                _compose(fop_inner, (sri, tuple(in_d[s1:s1 + w1])),
                                         fc_lookup, fold_rules,
                                         succ_map, carry_el, carry_out, zero_digit, cache)
                                for sri, in_d in zip(si_list, in_digits_list)
                            ]
                            inner_src_a = ('S', seg_idx_i)
                            inner_src_b = oth_src
                        else:
                            inner_vals = [
                                _compose(fop_inner, (tuple(in_d[s1:s1 + w1]), sri),
                                         fc_lookup, fold_rules,
                                         succ_map, carry_el, carry_out, zero_digit, cache)
                                for sri, in_d in zip(si_list, in_digits_list)
                            ]
                            inner_src_a = oth_src
                            inner_src_b = ('S', seg_idx_i)
                        if any(v is None for v in inner_vals):
                            continue
                        # Outer: fold_outer(inner, seg_j) or fold_outer(seg_j, inner)
                        for seg_idx_j in seg_idxs_nf:
                            sj_list = seg_results[seg_idx_j]
                            for fop_outer in fold_rules:
                                computed = [
                                    _compose(fop_outer, (iv, srj),
                                             fc_lookup, fold_rules,
                                             succ_map, carry_el, carry_out, zero_digit, cache)
                                    for iv, srj in zip(inner_vals, sj_list)
                                ]
                                if all(_matches(c, v) for c, v in zip(computed, value_tuples)):
                                    inner_spec = ('F', fop_inner, inner_src_a, inner_src_b)
                                    return ('F', fop_outer, inner_spec, ('S', seg_idx_j), width), computed
                                computed = [
                                    _compose(fop_outer, (srj, iv),
                                             fc_lookup, fold_rules,
                                             succ_map, carry_el, carry_out, zero_digit, cache)
                                    for iv, srj in zip(inner_vals, sj_list)
                                ]
                                if all(_matches(c, v) for c, v in zip(computed, value_tuples)):
                                    inner_spec = ('F', fop_inner, inner_src_a, inner_src_b)
                                    return ('F', fop_outer, ('S', seg_idx_j), inner_spec, width), computed

    # 6. fold(seg_i, seg_j) — two prior segments
    seg_idxs = [k for k, v in seg_results.items() if not any(r is None for r in v)]
    for ii in range(len(seg_idxs)):
        si = seg_idxs[ii]
        si_list = seg_results[si]
        for jj in range(ii, len(seg_idxs)):
            sj = seg_idxs[jj]
            sj_list = seg_results[sj]
            for fop in fold_rules:
                computed = [
                    _compose(fop, (sri, srj),
                             fc_lookup, fold_rules,
                             succ_map, carry_el, carry_out, zero_digit, cache)
                    for sri, srj in zip(si_list, sj_list)
                ]
                if all(_matches(c, v) for c, v in zip(computed, value_tuples)):
                    return ('F', fop, ('S', si), ('S', sj), width), computed
                computed = [
                    _compose(fop, (srj, sri),
                             fc_lookup, fold_rules,
                             succ_map, carry_el, carry_out, zero_digit, cache)
                    for sri, srj in zip(si_list, sj_list)
                ]
                if all(_matches(c, v) for c, v in zip(computed, value_tuples)):
                    return ('F', fop, ('S', sj), ('S', si), width), computed

    # 7. adj_search: fold(x, b) = seg_result; b from in_slice
    for seg_idx, seg_result_list in seg_results.items():
        if any(r is None for r in seg_result_list):
            continue
        for fop in fold_rules:
            for w1 in (1, 2):
                for s1 in range(actual_n - w1 + 1):
                    oth_src = ('I', s1, s1 + w1 - 1)
                    b_width = w1
                    computed = [
                        _compose_adjoint_search(
                            "adj_search",
                            sr + tuple(in_d[s1:s1 + w1]),
                            (fop, 1),
                            fc_lookup, fold_rules,
                            succ_map, carry_el, carry_out, zero_digit, cache,
                            b_width=b_width,
                        )
                        for sr, in_d in zip(seg_result_list, in_digits_list)
                    ]
                    if all(_matches(c, v) for c, v in zip(computed, value_tuples)):
                        return ('A', fop, ('S', seg_idx), oth_src, width), computed

    # 8. Last-resort: all-constant block (for single-example groups where the
    #    early check was skipped because len(value_tuples) < 2).
    if all(v == value_tuples[0] for v in value_tuples):
        toks = value_tuples[0]
        return ('CONST_BLOCK', toks), [toks for _ in value_tuples]

    return None, None


def _discover_slot_program(
    op_atom: str,
    pattern_key: tuple,
    parsed: list,
    fold_rules: dict,
    fc_lookup: dict,
    succ_map: dict,
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
    cache: dict,
) -> "Optional[SlotProgram]":
    """Discover a SlotProgram for one (op_atom, pattern_key) group.

    parsed is a list of (in_digits, output_tokens) pairs, one per training example.
    Returns a SlotProgram or None if no consistent program is found.
    """
    if not parsed:
        return None

    # Parse all training examples into segment structure
    parsed_segs = [(in_d, _parse_into_value_blocks(out_toks))
                   for in_d, out_toks in parsed]

    # Verify all examples share the same segment-type skeleton
    skeleton = [s[0] for s in parsed_segs[0][1]]
    for _, segs in parsed_segs[1:]:
        if [s[0] for s in segs] != skeleton:
            return None

    n_segs = len(parsed_segs[0][1])
    result_segments: list = []
    seg_results: dict = {}  # seg_idx -> list of result tuples (one per example)

    for seg_idx in range(n_segs):
        seg_type = skeleton[seg_idx]

        if seg_type == 'K':
            tok = parsed_segs[0][1][seg_idx][1]
            if not all(segs[seg_idx][1] == tok for _, segs in parsed_segs):
                return None
            result_segments.append(('K', tok))
            seg_results[seg_idx] = [None] * len(parsed)
        else:
            # 'V': variable value block — discover the computation
            value_tuples = [segs[seg_idx][1] for _, segs in parsed_segs]
            in_digits_list = [in_d for in_d, _ in parsed_segs]
            prior_results = {k: v for k, v in seg_results.items()
                             if k < seg_idx and v[0] is not None}

            spec, computed = _discover_value_segment(
                value_tuples, in_digits_list, prior_results,
                fold_rules, fc_lookup, succ_map, carry_el, carry_out, zero_digit, cache,
            )
            if spec is None:
                return None
            result_segments.append(spec)
            seg_results[seg_idx] = computed if computed is not None else [None] * len(parsed)

    return SlotProgram(
        op_atom=op_atom,
        pattern_key=pattern_key,
        segments=result_segments,
        evidence=len(parsed),
    )


def _refine_pattern_key(pattern_key: tuple, examples: list) -> tuple:
    """Embed structural (constant) digit values back into the pattern_key.

    A wildcard '_' at position p is structural if ALL training examples share
    the same digit value at the corresponding in_digits index.  Those positions
    are replaced with the specific digit value so that the SlotProgram's
    pattern_key uniquely identifies the computation structure.

    Example: pow b e group with e=2 → ('_','2') instead of ('_','_'),
    enabling unambiguous dispatch without needing partial output.
    """
    wild_indices = [i for i, t in enumerate(pattern_key) if t == '_']
    if not wild_indices:
        return pattern_key
    refined = list(pattern_key)
    for j, pk_idx in enumerate(wild_indices):
        vals = {in_digits[j] for in_digits, _ in examples if j < len(in_digits)}
        if len(vals) == 1:
            refined[pk_idx] = next(iter(vals))
    return tuple(refined)


def _pattern_key_matches(pattern_key: tuple, input_tokens: list) -> bool:
    """Check if input_tokens are compatible with a (possibly refined) pattern_key.

    '_' is a wildcard matching any single digit token.
    Literal non-'_' entries must match exactly.
    Length must match.
    """
    if len(pattern_key) != len(input_tokens):
        return False
    for pk_tok, in_tok in zip(pattern_key, input_tokens):
        if pk_tok == '_':
            if not in_tok.isdigit():
                return False
        elif pk_tok != in_tok:
            return False
    return True


def _discover_trace_programs(
    chain_rules: dict,
    fold_rules: dict,
    fc_lookup: dict,
    succ_map: dict,
    carry_el: str,
    carry_out: tuple,
    zero_digit: str,
) -> dict:
    """Synthesize SlotProgram for each trace-format chain op.

    Groups training examples by (pattern_key, skeleton_key), then refines
    pattern_keys: digit positions that are constant within a skeleton group
    (structural positions — e.g. exponent in pow b e) are embedded as literals.
    This disambiguates programs at dispatch time without requiring partial output.

    Returns dict mapping op_atom → list[SlotProgram] (flat list per op).
    Dispatch uses _pattern_key_matches() to find the right program for any input.
    """
    cache: dict = {}
    programs: dict = {}

    for op_atom, cr in chain_rules.items():
        if not cr.chain_table:
            continue

        # Group by (pattern_key, skeleton_key)
        # pattern_key: replace ALL digits with '_' (base structural shape).
        # skeleton_key: ALL non-digit tokens in output — distinguishes different
        #   output shapes (e.g. 'mul _ x' vs 'mul _ sq x' vs 'mul _ pow x _').
        # After grouping, refine pattern_keys: digit positions that are CONSTANT
        #   within a skeleton group are structural (they determine the output shape)
        #   and are embedded back as literal values — e.g. exponent '2' in pow b 2.
        groups: dict = {}
        for input_key, output_toks in cr.chain_table.items():
            if 'ans' not in output_toks:
                continue
            in_digits = [tok for tok in input_key if tok.isdigit()]
            if not in_digits:
                continue
            pattern_key = tuple('_' if t.isdigit() else t for t in input_key)
            skeleton_key = tuple(t for t in output_toks if not t.isdigit())
            groups.setdefault((pattern_key, skeleton_key), []).append(
                (in_digits, list(output_toks))
            )

        for (pattern_key, _skel), examples in groups.items():
            if len(examples) < 1:
                continue
            refined_pk = _refine_pattern_key(pattern_key, examples)
            prog = _discover_slot_program(
                op_atom, refined_pk, examples,
                fold_rules, fc_lookup, succ_map, carry_el, carry_out, zero_digit, cache,
            )
            if prog is not None:
                programs.setdefault(op_atom, []).append(prog)

        # Sort programs by specificity: most-specific pattern_key first.
        # A pattern_key with more literal (non-'_') tokens is more specific and
        # should be tried before wildcard patterns to avoid premature matches.
        # Example: ('_','2') before ('_','_') for pow b 2 vs pow b e dispatch.
        if op_atom in programs:
            programs[op_atom].sort(
                key=lambda p: sum(1 for t in p.pattern_key if t != '_'),
                reverse=True,
            )

    return programs


def _slot_program_predict(
    prog: "SlotProgram",
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
    """Level 1c: predict next token using a discovered SlotProgram.

    Evaluates each segment of the program against input_tokens, assembles
    the full expected output sequence, and returns a point mass for the
    k-th token (where k = len(output_tokens_so_far)).

    In ans_only mode (eq-format sequences), strips everything up to and
    including 'ans' so only the final answer tokens are returned.
    """
    in_digits = [tok for tok in input_tokens if tok.isdigit()]

    def _eval_src(src):
        if src[0] == 'I':
            _, j0, j1 = src
            return tuple(in_digits[j0:j1 + 1])
        elif src[0] == 'S':
            return seg_computed.get(src[1])
        elif src[0] == 'F':
            # Nested fold spec: ('F', op, src_a, src_b) — 4 elements, no width
            _, op, src_a, src_b = src
            a = _eval_src(src_a)
            b = _eval_src(src_b)
            if a is None or b is None:
                return None
            return _compose(op, (a, b), fc_lookup, fold_rules,
                            succ_map, carry_el, carry_out, zero_digit, cache)
        return None

    seg_computed: dict = {}  # seg_idx -> result tuple
    output_seq: list = []

    for seg_idx, seg in enumerate(prog.segments):
        if seg[0] == 'K':
            output_seq.append(seg[1])
            seg_computed[seg_idx] = None
        elif seg[0] == 'CONST_BLOCK':
            output_seq.extend(seg[1])
            seg_computed[seg_idx] = seg[1]
        elif seg[0] == 'E':
            output_seq.extend(in_digits)
            seg_computed[seg_idx] = tuple(in_digits)
        elif seg[0] == 'G':
            j = seg[1]
            if j >= len(in_digits):
                return None
            tok = in_digits[j]
            output_seq.append(tok)
            seg_computed[seg_idx] = (tok,)
        elif seg[0] == 'F':
            _, op, src_a_spec, src_b_spec, width = seg
            src_a = _eval_src(src_a_spec)
            src_b = _eval_src(src_b_spec)
            if src_a is None or src_b is None:
                return None
            result = _compose(op, (src_a, src_b), fc_lookup, fold_rules,
                              succ_map, carry_el, carry_out, zero_digit, cache)
            if result is None:
                return None
            padded = _zpad(result, width)
            output_seq.extend(padded)
            seg_computed[seg_idx] = result
        elif seg[0] == 'A':
            _, op, step_src_spec, oth_src_spec, width = seg
            step_src = _eval_src(step_src_spec)
            oth_src = _eval_src(oth_src_spec)
            if step_src is None or oth_src is None:
                return None
            b_width = len(oth_src)
            result = _compose_adjoint_search(
                "adj_search",
                step_src + oth_src,
                (op, 1),
                fc_lookup, fold_rules,
                succ_map, carry_el, carry_out, zero_digit, cache,
                b_width=b_width,
            )
            if result is None:
                return None
            padded = _zpad(result, width)
            output_seq.extend(padded)
            seg_computed[seg_idx] = result
        elif seg[0] == 'P':
            # Predecessor of input digit at position j (inverse of succ_map).
            j = seg[1]
            if j >= len(in_digits):
                return None
            pred_map_local = {v: k for k, v in succ_map.items()} if succ_map else {}
            tok = pred_map_local.get(in_digits[j])
            if tok is None:
                return None
            output_seq.append(tok)
            seg_computed[seg_idx] = (tok,)
        elif seg[0] == 'SC':
            # Successor of input digit at position j.
            j = seg[1]
            if j >= len(in_digits):
                return None
            tok = succ_map.get(in_digits[j]) if succ_map else None
            if tok is None:
                return None
            output_seq.append(tok)
            seg_computed[seg_idx] = (tok,)
        elif seg[0] == 'AK':
            # adj_search(in_slice, K_const): find x where fold_op(x, K) = in[j:j+w].
            _, op, src_spec, k_digit, width = seg
            in_slice = _eval_src(src_spec)
            if in_slice is None:
                return None
            k_tuple = (k_digit,)
            result = _compose_adjoint_search(
                "adj_search",
                in_slice + k_tuple,
                (op, 1),
                fc_lookup, fold_rules,
                succ_map, carry_el, carry_out, zero_digit, cache,
                b_width=1,
            )
            if result is None:
                return None
            padded = _zpad(result, width)
            output_seq.extend(padded)
            seg_computed[seg_idx] = result

    if ans_only:
        if 'ans' in output_seq:
            ans_start = output_seq.index('ans') + 1
            output_seq = output_seq[ans_start:]
        else:
            return None

    k = len(output_tokens_so_far)
    # Consistency check: the program's output must agree with all tokens generated so far.
    # This filters out programs whose skeleton doesn't match the partial output
    # (e.g. an e=2 program that predicts 'step' when output already shows 'ans').
    if k > 0:
        if list(output_seq[:k]) != list(output_tokens_so_far):
            return None
    if k < len(output_seq):
        return {output_seq[k]: 1.0}
    if k == len(output_seq):
        return {'<eos>': 1.0}
    return None
