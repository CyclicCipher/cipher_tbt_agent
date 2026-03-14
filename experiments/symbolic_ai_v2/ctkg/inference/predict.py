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
) -> Optional[tuple]:
    """Solve op(input_flat) via adjunction + composition.

    For an adjunction left_op ⊣ op (preserved_position p):
        op(c_flat + b_flat) = a  iff  left_op(a_flat + b_flat) = c_flat

    where b_flat is the preserved argument (1 token at the end of input_flat)
    and c_flat is the target output (all preceding tokens).

    Enumerates candidate a_flat from zero_digit via succ_map, computes
    left_op(a, b) using the composition engine, and returns the first a
    where the result equals c_flat.  Handles multi-digit c_flat correctly
    (e.g. sub(11, 4) = 7 iff add(7, 4) = 11).

    Returns the answer as a flat tuple (e.g. ('7',)), or None if not found.
    """
    fwd_op, _preserved_pos = adj_info

    # b = last 1 token (preserved single-digit arg of the forward op)
    # c = all preceding tokens (target output of the forward op)
    if len(input_flat) < 2:
        return None
    b_tuple: tuple = input_flat[-1:]   # e.g. ('4',) for sub(11, 4)
    c_tuple: tuple = input_flat[:-1]   # e.g. ('1', '1') for sub(11, 4)

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

    # Cache key: (op, flat_inputs) — same format as fc_lookup
    flat_inputs = tuple(t for arg in args for t in arg)
    cache_key = (op, flat_inputs)

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # FC direct lookup (only unambiguous when all args are single-token)
    all_single = all(len(a) == 1 for a in args)
    if all_single:
        direct = fc_lookup.get(cache_key)
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
