"""
Phase 5 redesign: CT-faithful operator discovery via free category + quotient.

Architecture (Option B — confirmed user choice):
  Dissolves the old ad-hoc fold-type rule discovery into the existing
  FCA / MorphismGraph / KanExtension / Operad pipeline.

  Old approach: exact lookup tables (FOLD_RIGHT, BINARY_FOLD, etc.) built by
  backtracking carry inference.  Exact for seen inputs; 0% for novel inputs.
  Also violated CT_REFERENCE §19 throughout: string matching on token labels
  ("0", "1", carry_mode=="add") instead of universal property tests.

  New approach (CT_REFERENCE §3,4,6,8,9,16,17,19):
    1.  Build the free category FC(G) from training sequences (§17).
        Each eq-delimited sequence is one morphism edge.  Arity is discovered
        from data (len(inputs)), not hardcoded as unary/binary.
    2.  Discover equations E: contexts producing identical outputs (§17).
        The quotient FC(G)/E is the learned category.
    3.  Detect NNO-like structure by universal property (§19): tokens with a
        total self-morphism s: D→D and a unique initial object z: 1→D.
        No string matching on token labels.
    4.  Detect adjunctions (§4): F⊣G pairs where F(a,b)=c ↔ G(c,b)=a.
        Covers add⊣sub, succ⊣pred, mul⊣div — no operator names hardcoded.
    5.  Detect natural transformations (§3): commutativity σ: F(A,B)≅F(B,A).
        Checked as naturality squares on observed edges.
    6.  Enrich MorphismGraph with arithmetic morphisms so Level 2+3 of the
        Predictor handle generalization via JSD Kan extension.

  Arithmetic answers are now probabilistic predictions over concept types
  rather than exact digit sequences.  Determinism emerges as coverage
  grows: more training data → sharper concept clusters → exact answers.

  Backward-compatible stubs for the old API (ProcessRule, ChainRule,
  discover_processes, discover_compose_chains, apply_process_rule) are kept
  so that predict.py and all unit tests import without errors.
  discover_processes() and discover_compose_chains() return [] — the
  Predictor falls through to Level 2+3 for all arithmetic.

See CT_REFERENCE.md §3, §4, §6, §9, §16, §17, §19 for the categorical
foundations of each design decision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Free category data structures  (CT_REFERENCE §17)
# ---------------------------------------------------------------------------

@dataclass
class FreeCategoryEdge:
    """One morphism in the free category FC(G).

    Represents one training sequence: op(inputs) → outputs.
    The number of input/output tokens is len(inputs)/len(outputs) — structural
    arity (unary, binary, …) is learned via NNO/adjunction detection, not
    stored as a token count.
    """

    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass
class NNOEvidence:
    """Evidence that an operator token has NNO-like structure (CT_REFERENCE §19).

    Detected by universal property:
      - op is a unary total function s: D → D on its observed domain.
      - There is exactly one element in the domain with no predecessor.
        That element is the candidate for the initial object z: 1 → D.
      - Walking successor from z covers all observed domain elements in a
        linear chain (no branching, no cycle before exhaustion).

    No string matching on token labels.
    """

    op: str                        # the successor-like operator token
    successor_map: dict[str, str]  # d → s(d) for all observed (d, s(d)) pairs
    zero_candidate: str            # the unique element with no predecessor


@dataclass
class AdjunctionPair:
    """Evidence that left_op ⊣ right_op (CT_REFERENCE §4).

    Detected when left_op(a, b) = c AND right_op(c, b) = a for all
    observed triples that have both sides present.

    Covers add⊣sub, succ⊣pred, mul⊣div — no operator names hardcoded.

    preserved_position:
        For unary adjunctions (succ⊣pred): None.
        For n-ary adjunctions (add⊣sub): the index in left_op.inputs that is
        preserved as the last token(s) of right_op.inputs.  Discovered from
        data — the position that maximises verified triple count.
    """

    left_op: str
    right_op: str
    evidence: int                        # number of (a, b, c) triples verified
    preserved_position: Optional[int] = None  # None = unary; discovered from data
    input_len: Optional[int] = None           # token count of left_op inputs for this adjunction;
                                              # None = unary (applies to all input lengths)


@dataclass
class NaturalTransformation:
    """Evidence of a natural transformation between functors (CT_REFERENCE §3).

    Commutativity σ_{A,B}: F(A,B) ≅ F(B,A) is a natural isomorphism.
    Checked as: for all observed (a, b) in the domain of op with a≠b,
    F(a,b) = F(b,a).
    """

    op: str
    kind: str      # "commutative", "associative", ...
    evidence: int  # number of (a, b) pairs verified


@dataclass
class FreeCategoryGraph:
    """The free category FC(G) built from training sequences.

    Objects: distinct token types observed in the corpus.
    Morphisms: one edge per eq-delimited training sequence.

    Structure annotations discovered by universal property tests:
      nno_candidates  — operators with NNO-like (successor-chain) structure
      adjunctions     — operator pairs satisfying the adjunction unit/counit
      nat_transforms  — natural transformations (commutativity etc.)
      equations       — pairs of distinct edges producing identical outputs
    """

    edges: list[FreeCategoryEdge] = field(default_factory=list)
    nno_candidates: list[NNOEvidence] = field(default_factory=list)
    adjunctions: list[AdjunctionPair] = field(default_factory=list)
    nat_transforms: list[NaturalTransformation] = field(default_factory=list)
    equations: list[tuple[FreeCategoryEdge, FreeCategoryEdge]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Free category construction  (CT_REFERENCE §17)
# ---------------------------------------------------------------------------

def build_free_category(corpus: list[list[str]]) -> FreeCategoryGraph:
    """Build free category FC(G) from eq-delimited training sequences.

    Each sequence [op, a1, ..., ak, 'eq', r1, ..., rm] becomes one directed
    edge: op(a1,...,ak) → (r1,...,rm).  Arity = len(inputs) discovered from
    data.  Operator identity is structural: any first-token that appears with
    an 'eq' delimiter.  Trace-format sequences (step/ans) are ignored here;
    they have no 'eq' delimiter separating input from output.

    After building the edge set the function detects NNO structure,
    adjunctions, natural transformations, and equations in-place.

    Parameters
    ----------
    corpus:
        Raw token sequences.  May mix eq-format and trace-format.

    Returns
    -------
    FreeCategoryGraph with all structural annotations populated.
    """
    edges: list[FreeCategoryEdge] = []

    for seq in corpus:
        if len(seq) < 3:
            continue
        op = seq[0]
        # Strip trailing <eos>
        body = seq[1:-1] if seq[-1] == "<eos>" else seq[1:]

        try:
            eq_idx = body.index("eq")
        except ValueError:
            continue

        inputs = tuple(body[:eq_idx])
        outputs = tuple(body[eq_idx + 1:])

        if not inputs or not outputs:
            continue

        edges.append(FreeCategoryEdge(
            op=op,
            inputs=inputs,
            outputs=outputs,
        ))

    fc = FreeCategoryGraph(edges=edges)
    _detect_nno(fc)
    _detect_adjunctions(fc)
    _detect_nat_transforms(fc)
    _discover_equations(fc)
    return fc


# ---------------------------------------------------------------------------
# Universal property tests
# ---------------------------------------------------------------------------

def _detect_nno(fc: FreeCategoryGraph) -> None:
    """Detect NNO-like operators by universal property (CT_REFERENCE §19).

    A unary operator s has NNO structure if:
      1. It defines a total, consistent function D → D on its observed domain.
      2. Exactly one element in the domain has no predecessor (the initial
         object z, satisfying z: 1 → D in CT terms).
      3. Walking the successor chain from z covers all observed domain elements
         in a single linear chain (no branching, no premature cycle).

    No string matching on token labels — purely structural.
    """
    # Accumulate unary maps: op → {d_in: set of d_out}
    raw_maps: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for edge in fc.edges:
        if len(edge.inputs) == 1 and len(edge.outputs) == 1:
            raw_maps[edge.op][edge.inputs[0]].add(edge.outputs[0])

    for op, mapping in raw_maps.items():
        # Consistency: each input maps to exactly one output
        if any(len(outs) > 1 for outs in mapping.values()):
            continue
        succ_map = {d_in: next(iter(outs)) for d_in, outs in mapping.items()}

        sources = set(succ_map.keys())
        targets = set(succ_map.values())

        # Zero candidates: elements in domain but not in range (no predecessor).
        # With an 80/20 train/test split, some succ edges land in the test set,
        # creating spurious gaps that make normally-unique elements look like
        # extra zero candidates.  We try ALL zero candidates and keep the one
        # that produces the longest chain — that is the true initial object.
        zero_candidates = sources - targets
        if not zero_candidates:
            continue

        best_zero: Optional[str] = None
        best_chain: list[str] = []

        for z_candidate in zero_candidates:
            chain: list[str] = [z_candidate]
            seen: set[str] = {z_candidate}
            curr = z_candidate
            valid = True
            for _ in range(len(sources)):
                nxt = succ_map.get(curr)
                if nxt is None:
                    break
                if nxt in seen:
                    valid = False
                    break
                chain.append(nxt)
                seen.add(nxt)
                curr = nxt
            if valid and len(chain) > len(best_chain):
                best_chain = chain
                best_zero = z_candidate

        if best_zero is not None and len(best_chain) >= 2:
            fc.nno_candidates.append(NNOEvidence(
                op=op,
                successor_map=dict(succ_map),
                zero_candidate=best_zero,
            ))


def _detect_adjunctions(fc: FreeCategoryGraph) -> None:
    """Detect adjunction pairs F ⊣ G (CT_REFERENCE §4).

    For each op-pair (F, G) and each candidate preserved-argument position p,
    tests whether: G(F(inputs) + (inputs[p],)) = inputs_without_p holds for
    all observed F-edges that have a matching G-edge.

    Position p is discovered from data by finding the index that maximises
    the number of verified triples.  No arity or position is hardcoded.
    Handles unary (single-token inputs), binary (two-token inputs), and
    multi-token-output edges (e.g. add('9','5') → ('1','4')).

    Covers add⊣sub, succ⊣pred, mul⊣div — no operator names hardcoded.
    """
    # Build lookup: (op, inputs_tuple) → outputs_tuple (first observation wins)
    lookup: dict[tuple[str, tuple], tuple] = {}
    for edge in fc.edges:
        key = (edge.op, edge.inputs)
        if key not in lookup:
            lookup[key] = edge.outputs

    # Group edges by op and by exact input length (no output-length filter)
    edges_by_op: dict[str, list[FreeCategoryEdge]] = defaultdict(list)
    for edge in fc.edges:
        edges_by_op[edge.op].append(edge)

    ops = list(edges_by_op.keys())
    for op_f in ops:
        f_edges = edges_by_op[op_f]
        if not f_edges:
            continue

        for op_g in ops:
            if op_f == op_g:
                continue

            # --- Unary test: use only edges with exactly 1-token input ---
            # Run this whenever op_f has any single-token-input edges,
            # regardless of its most common arity.  This handles NNO operators
            # (succ, pred) that also have multi-digit edges.
            unary_f = [e for e in f_edges if len(e.inputs) == 1]
            if unary_f:
                verified = total = 0
                for edge in unary_f:
                    result = lookup.get((op_g, edge.outputs))
                    if result is None:
                        continue
                    total += 1
                    if result == edge.inputs:
                        verified += 1
                if total >= 2 and verified == total:
                    fc.adjunctions.append(AdjunctionPair(
                        left_op=op_f, right_op=op_g,
                        evidence=verified, preserved_position=None,
                    ))
                    continue  # found unary adjunction; skip n-ary test for this pair

            # --- N-ary test: group edges by exact input length, try each length ---
            # Try every preserved-argument position; pick the one with best coverage.
            input_lengths = set(len(e.inputs) for e in f_edges if len(e.inputs) > 1)
            for arity in input_lengths:
                arity_edges = [e for e in f_edges if len(e.inputs) == arity]
                best_pos = -1
                best_verified = 0
                for p in range(arity):
                    verified = total = 0
                    for edge in arity_edges:
                        pres = (edge.inputs[p],)
                        non_pres = edge.inputs[:p] + edge.inputs[p + 1:]
                        g_key = (op_g, edge.outputs + pres)
                        result = lookup.get(g_key)
                        if result is None:
                            continue
                        total += 1
                        if result == non_pres:
                            verified += 1
                    if total >= 2 and verified == total and verified > best_verified:
                        best_pos, best_verified = p, verified
                if best_pos >= 0:
                    fc.adjunctions.append(AdjunctionPair(
                        left_op=op_f, right_op=op_g,
                        evidence=best_verified, preserved_position=best_pos,
                        input_len=arity,
                    ))


def _detect_nat_transforms(fc: FreeCategoryGraph) -> None:
    """Detect natural transformations (CT_REFERENCE §3).

    Commutativity σ_{A,B}: F(A,B) ≅ F(B,A) — a natural isomorphism.
    Detected by checking naturality squares: for all observed (a, b) with
    a ≠ b, both F(a,b) and F(b,a) must be observed and equal.
    """
    lookup: dict[tuple[str, tuple], tuple] = {}
    for edge in fc.edges:
        key = (edge.op, edge.inputs)
        if key not in lookup:
            lookup[key] = edge.outputs

    binary_ops: set[str] = set()
    for edge in fc.edges:
        if len(edge.inputs) == 2:
            binary_ops.add(edge.op)

    for op in binary_ops:
        verified = total = 0
        for edge in fc.edges:
            if edge.op != op or len(edge.inputs) != 2:
                continue
            a, b = edge.inputs
            if a == b:
                continue
            swapped = lookup.get((op, (b, a)))
            if swapped is None:
                continue
            total += 1
            if swapped == edge.outputs:
                verified += 1

        if total >= 2 and verified == total:
            fc.nat_transforms.append(NaturalTransformation(
                op=op, kind="commutative", evidence=verified
            ))


def _discover_equations(fc: FreeCategoryGraph) -> None:
    """Discover equations E in FC(G): distinct paths with identical outputs (§17).

    Two edges f, g are equated if f.outputs == g.outputs but they differ in
    operator or inputs.  The equation set E defines the quotient FC(G)/E.
    """
    by_output: dict[tuple, list[FreeCategoryEdge]] = defaultdict(list)
    for edge in fc.edges:
        by_output[edge.outputs].append(edge)

    for group in by_output.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                e1, e2 = group[i], group[j]
                if e1.op != e2.op or e1.inputs != e2.inputs:
                    fc.equations.append((e1, e2))


# ---------------------------------------------------------------------------
# MorphismGraph enrichment  (routes FC(G) structure into the pipeline)
# ---------------------------------------------------------------------------

def enrich_morphism_graph(
    fc: FreeCategoryGraph,
    morphism_graph,   # MorphismGraph — duck-typed to avoid circular import
    lattice,          # ConceptLattice
    hankel,           # HankelCount (for vocabulary access; reserved)
) -> tuple[list[NNOEvidence], list[AdjunctionPair], list[NaturalTransformation]]:
    """Enrich MorphismGraph with arithmetic morphisms from FC(G).

    For each unique (op, inputs) → outputs triple in the free category:
      1. Find the MorphismGraph object whose concept has the highest intent
         weight on the operator token (the best proxy for the input context).
      2. Find the object with the highest intent weight on the output tokens.
      3. Add (or strengthen) a CTKGMorphism between them, weighted by the
         number of times this triple was observed in training.

    This gives Level 2 (morphism marginalization) and Level 3 (Kan extension)
    in the Predictor access to arithmetic structure without exact lookup tables.
    Determinism emerges as training coverage grows.

    Parameters
    ----------
    fc:
        The free category built from build_free_category().
    morphism_graph:
        The MorphismGraph from the EM loop.  Modified in-place.
    lattice:
        The ConceptLattice from the EM loop.
    hankel:
        The HankelCount (reserved for future neighbourhood-hash concept lookup).

    Returns
    -------
    (nno_candidates, adjunctions, nat_transforms) — structural metadata
    for logging and downstream use.
    """
    obj_list = morphism_graph.objects(active_only=True)
    if not lattice.concepts or not obj_list:
        return fc.nno_candidates, fc.adjunctions, fc.nat_transforms

    def _best_obj_for_tokens(tokens: tuple[str, ...]):
        """Find the object whose concept has the highest total intent weight
        on the given tokens.  Returns None if all scores are zero."""
        best_obj = None
        best_score = 0.0
        for obj in obj_list:
            score = sum(
                obj.concept.intent_weights.get(t, 0.0) for t in tokens
            )
            if score > best_score:
                best_score = score
                best_obj = obj
        return best_obj  # None when best_score == 0.0

    # Count occurrences of each unique (op, inputs, outputs) triple
    triple_counts: dict[tuple, int] = defaultdict(int)
    for edge in fc.edges:
        triple_counts[(edge.op, edge.inputs, edge.outputs)] += 1

    # morph_type label: one per operator (arity can vary; use first seen)
    morph_type_cache: dict[str, str] = {}

    for (op, inputs, outputs), count in triple_counts.items():
        in_obj = _best_obj_for_tokens((op,))
        out_obj = _best_obj_for_tokens(outputs)

        if in_obj is None or out_obj is None:
            continue
        if in_obj.obj_id == out_obj.obj_id:
            continue

        arity = len(inputs)
        if op not in morph_type_cache:
            morph_type_cache[op] = f"FC_{op}_{arity}ary"
        morph_type = morph_type_cache[op]

        existing = [
            m for m in morphism_graph.hom(
                in_obj.obj_id, out_obj.obj_id, include_identity=False
            )
            if m.morph_type == morph_type
        ]
        if existing:
            morphism_graph.observe(existing[0].morph_id, count)
        else:
            morphism_graph.add_morphism(
                source_id=in_obj.obj_id,
                target_id=out_obj.obj_id,
                evidence=count,
                morph_type=morph_type,
                confidence=0.0,
            )

    return fc.nno_candidates, fc.adjunctions, fc.nat_transforms


# ---------------------------------------------------------------------------
# Backward-compatible stubs
#
# Kept so that predict.py and all unit tests continue to import without errors.
# discover_processes() and discover_compose_chains() return [] — the Predictor
# falls through from the empty Level 1 to Level 2+3 for all predictions.
# ---------------------------------------------------------------------------

@dataclass
class ProcessRule:
    """Stub — kept for API compatibility.  No longer populated."""

    rule_id: int = 0
    op_atom: str = ""
    process_type: str = "FOLD_RIGHT"
    initial_carry: int = 0
    carry_mode: str = "add"
    transition_table: dict = field(default_factory=dict)
    binary_table: dict = field(default_factory=dict)
    arg_split: str = "none"
    compose_op: str = ""
    compose_initial: int = 0
    result_longer: bool = False
    support: int = 0
    zero_token: str = "0"
    overflow_token: str = "1"
    is_commutative: bool = False

    def __repr__(self) -> str:
        return f"ProcessRule(stub, op={self.op_atom!r})"


@dataclass
class ChainRule:
    """Stub — kept for API compatibility.  No longer populated."""

    rule_id: int = 0
    op_atom: str = ""
    process_type: str = "COMPOSE_CHAIN"
    chain_table: dict = field(default_factory=dict)
    eq_table: dict = field(default_factory=dict)
    support: int = 0

    def __repr__(self) -> str:
        return f"ChainRule(stub, op={self.op_atom!r})"


def discover_processes(
    corpus: list[list[str]],
    op_atoms=None,
) -> list[ProcessRule]:
    """Stub — returns [] under the Option B architecture.

    All arithmetic is routed through build_free_category() +
    enrich_morphism_graph().  The Predictor falls through to Level 2+3.
    """
    return []


def discover_compose_chains(
    corpus: list[list[str]],
    eos_token: str = "<eos>",
    digit_alphabet=None,
) -> list[ChainRule]:
    """Discover ChainRule objects from step/ans and eq-format sequences.

    Scans the training corpus for sequences that use step/ans delimiters
    (trace format: [op, inputs..., step, out..., ans, final..., <eos>]) or
    eq delimiters (eq format: [op, inputs..., eq, out..., <eos>]).

    Builds a chain_table and eq_table for each discovered op:
      chain_table[(input_tokens)] = full_output_tokens (step/ans format)
      eq_table[(input_tokens)]    = output_tokens (eq format, after eq)

    These tables let Level 1b predict any trace-format sequence seen during
    training (memorisation baseline).  Test generalisation comes from the
    composition engine (Level 0.7) when the chain_table misses.

    Simple arithmetic ops (add, sub, mul, pow, succ, pred) are excluded to
    avoid shadowing Level 1a (process rules) and Level 0.7 (fold rules).
    """
    from collections import defaultdict

    STEP_TOKEN = "step"
    ANS_TOKEN = "ans"
    EQ_TOKEN = "eq"

    # Ops handled by fold rules / process rules — exclude from chain_rules
    # so Level 1b doesn't shadow Level 1a / Level 0.7 for arithmetic.
    ARITHMETIC_OPS = frozenset({"add", "sub", "mul", "pow", "succ", "pred"})

    chain_tables: dict[str, dict[tuple, list[str]]] = defaultdict(dict)
    eq_tables: dict[str, dict[tuple, list[str]]] = defaultdict(dict)

    for seq in corpus:
        if not seq:
            continue
        op = seq[0]

        # Strip trailing eos
        body = seq[1:]
        if body and body[-1] == eos_token:
            body = body[:-1]

        if not body:
            continue

        # Check for step/ans format (takes priority over eq format).
        # step/ans traces (e.g. power_trace: pow a4 a3 step 4 step 16 ans 64)
        # use namespaced a-prefix inputs and are DISTINCT from the fold-rule
        # eq-format sequences (pow 4 3 eq 64).  Include them in chain_rules
        # even for arithmetic ops so trace programs can be synthesised.
        step_ans_idx = None
        for i, tok in enumerate(body):
            if tok in (STEP_TOKEN, ANS_TOKEN):
                step_ans_idx = i
                break

        if step_ans_idx is not None:
            input_tokens = tuple(body[:step_ans_idx])
            output_tokens = list(body[step_ans_idx:])
            if input_tokens and output_tokens:
                key = input_tokens
                if key not in chain_tables[op]:
                    chain_tables[op][key] = output_tokens
            continue

        # eq format: skip ARITHMETIC_OPS (handled by fold rules / process rules)
        if op in ARITHMETIC_OPS:
            continue
        if EQ_TOKEN in body:
            eq_idx = body.index(EQ_TOKEN)
            input_tokens = tuple(body[:eq_idx])
            output_tokens = list(body[eq_idx + 1:])
            if input_tokens:
                key = input_tokens
                if key not in eq_tables[op]:
                    eq_tables[op][key] = output_tokens

    # Build ChainRule objects (one per op with non-empty tables)
    rules: list[ChainRule] = []
    all_ops = set(chain_tables.keys()) | set(eq_tables.keys())
    for op in sorted(all_ops):
        ct = dict(chain_tables.get(op, {}))
        et = dict(eq_tables.get(op, {}))
        support = len(ct) + len(et)
        if support == 0:
            continue
        rules.append(ChainRule(
            op_atom=op,
            chain_table=ct,
            eq_table=et,
            support=support,
        ))

    return rules


def complete_succ_map(
    succ_map: dict[str, str],
    zero_digit: str,
    carry_element: str,
) -> dict[str, str]:
    """Infer missing edges in a partial successor chain by bridging segments.

    With an 80/20 train/test split, some succ(d)=d+1 edges land in the test
    set, leaving the chain fragmented into two or more disconnected segments.
    The NNO chain is LINEAR by definition (no branching), so the unique way
    to reconnect k segments is to order them as:

        [segment containing zero_digit] → ... → [segment containing carry_element]

    and bridge each segment's tail to the next segment's head.

    The ordering rule:
      - zero_digit segment comes first.
      - carry_element segment comes last.
      - Any middle segments are sorted by their first element (lexicographic,
        as a last resort when we can't determine order structurally).

    Returns a copy of succ_map extended with the inferred bridge edges.
    The original entries are never modified.
    """
    if not succ_map:
        return dict(succ_map)

    # Collect all chain segments starting from elements with no predecessor
    targets = set(succ_map.values())
    segment_starts = (set(succ_map.keys()) - targets)
    # Also include zero_digit as a possible start even if it has a known predecessor
    # in a broken chain (shouldn't happen, but guard against it).
    if zero_digit in succ_map and zero_digit not in segment_starts:
        segment_starts.add(zero_digit)

    segments: list[list[str]] = []
    for start in segment_starts:
        chain: list[str] = []
        curr: Optional[str] = start
        seen: set[str] = set()
        while curr is not None and curr not in seen:
            chain.append(curr)
            seen.add(curr)
            curr = succ_map.get(curr)
        if chain:
            segments.append(chain)

    if len(segments) <= 1:
        return dict(succ_map)  # No gaps to bridge

    # Sort: zero_digit segment first, carry_element segment last, others by head
    def _sort_key(seg: list[str]) -> tuple:
        if zero_digit in seg:
            return (0, seg[0])
        if carry_element in seg:
            return (2, seg[0])
        return (1, seg[0])

    segments.sort(key=_sort_key)

    # Bridge: tail of each segment → head of next segment
    completed = dict(succ_map)
    for i in range(len(segments) - 1):
        tail = segments[i][-1]
        head = segments[i + 1][0]
        if tail not in completed:
            completed[tail] = head

    return completed


def build_unary_chain_maps(fc: FreeCategoryGraph) -> dict[str, dict[str, str]]:
    """Build partial single-token successor chains from FC edges.

    For each unary operator (at least one edge with len(inputs)==1), collects
    single-token-in → single-token-out edges into a partial chain map.

    Returns {op: {d_in: d_out}} for single-token-to-single-token edges only.
    """
    result: dict[str, dict[str, str]] = {}
    for edge in fc.edges:
        if len(edge.inputs) == 1 and len(edge.outputs) == 1:
            op = edge.op
            if op not in result:
                result[op] = {}
            d_in, d_out = edge.inputs[0], edge.outputs[0]
            if d_in not in result[op]:  # First observation wins
                result[op][d_in] = d_out
    return result


def build_unary_carry_maps(fc: FreeCategoryGraph) -> dict[str, tuple]:
    """For each unary op, find the carry element and its multi-token output.

    The carry element is the single-token input whose successor produces
    more than one output token (e.g. '9' for succ, where succ('9')=('1','0')).

    Returns {op: (carry_element, carry_output_tuple)}.
    """
    result: dict[str, tuple] = {}
    for edge in fc.edges:
        if len(edge.inputs) == 1 and len(edge.outputs) > 1:
            op = edge.op
            if op not in result:  # First observation wins
                result[op] = (edge.inputs[0], edge.outputs)
    return result


def unary_chain_predict(
    step_map: dict[str, str],
    carry_element: str,
    carry_out: tuple[str, ...],
    input_digits: list[str],
    inverse: bool = False,
) -> Optional[list[str]]:
    """Predict successor/predecessor of a multi-digit number using chain + carry.

    Uses the discovered single-token chain (step_map) and the carry element
    to compute multi-digit successor (inverse=False) or predecessor (inverse=True)
    without any int() calls — purely structural token composition.

    Parameters
    ----------
    step_map:
        Single-token chain: {d_in: d_out}.  For succ: {'0':'1',...,'8':'9'}.
    carry_element:
        The token that triggers carry/borrow.  For succ: '9' (no successor
        in step_map because succ('9') produces two output tokens).
    carry_out:
        The multi-token output for carry_element.  For succ: ('1','0').
        carry_out[0] = overflow token (prepended on full overflow).
        carry_out[1] = zero token (the wrap-to digit for carry/borrow).
    input_digits:
        The input token list (MSB-first).
    inverse:
        If False: forward successor (carry propagation).
        If True:  inverse predecessor (borrow propagation).

    Returns
    -------
    List of output tokens (MSB-first), or None if any token is unknown.
    """
    if not input_digits:
        return None

    zero_digit = carry_out[1]  # The wrap-to token; learned from carry_out.

    if inverse:
        # Predecessor: build inverse step_map, borrow_trigger = zero_digit
        pred_map = {v: k for k, v in step_map.items()}
        wrap_trigger = zero_digit       # '0' wraps to carry_element
        wrap_to = carry_element         # digit becomes '9'
        forward = False
    else:
        # Successor: use step_map directly, carry_trigger = carry_element
        pred_map = step_map
        wrap_trigger = carry_element    # '9' wraps to zero_digit
        wrap_to = zero_digit            # digit becomes '0'
        forward = True

    result: list[str] = []
    carry = True  # Start with 1 to add/subtract

    for d in reversed(input_digits):
        if carry:
            if d == wrap_trigger:
                # Carry/borrow continues: wrap this digit and propagate
                result.insert(0, wrap_to)
                carry = True
            elif d in pred_map:
                # Apply step, carry/borrow absorbed
                result.insert(0, pred_map[d])
                carry = False
            else:
                return None  # Unknown token in chain
        else:
            result.insert(0, d)  # Digit unchanged

    if carry:
        if forward:
            # Overflow: prepend the overflow token (e.g. '1' for decimal)
            result.insert(0, carry_out[0])
        else:
            # Underflow: pred(0) is undefined
            return None

    if inverse:
        # Strip leading zero_digit (e.g. pred(10) → '09' → '9')
        while len(result) > 1 and result[0] == zero_digit:
            result.pop(0)

    return result


def build_binary_nno_table(
    fc: FreeCategoryGraph,
    prior_tables: Optional[dict[tuple, tuple]] = None,
) -> dict[tuple, tuple]:
    """Complete binary op tables by NNO column-walk (CT_REFERENCE §19).

    For each binary op whose argument tokens live in a discovered NNO domain,
    fills in all entries by walking the succ chain, using known training
    entries as seeds.

    Three-level induction tower (no operator names hardcoded):

    Level 1 — succ-step:  op(succ(d1), d2) = succ(op(d1, d2))
        Discovered when col[succ(d1)] == succ(col[d1]) for observed pairs.
        Covers add-like ops.

    Level 2 — binary-step:  op(succ(d1), d2) = G(op(d1, d2), d2)
        Discovered by finding a unique op G in prior_tables that satisfies
        the recurrence for all observed consecutive pairs.
        Covers mul (G=add), and other ops using a previously-built table.

    Level 3 — reversed orientation (2-input ops only):
        Try d1 = last input token, d2 = first input token.
        Covers pow (induction over exp with base fixed, G=mul).

    Parameters
    ----------
    fc:
        Free category built from training sequences.
    prior_tables:
        Additional (op, inputs_tuple) → outputs_tuple entries to use as
        candidate step functions (e.g. the NNO-completed add table, so
        mul's step "add(prev, d2)" can be looked up).

    Returns
    -------
    {(op, inputs_tuple): outputs_tuple}  — entries derived by induction.
    """
    chain_maps = build_unary_chain_maps(fc)
    carry_maps = build_unary_carry_maps(fc)
    if not chain_maps:
        return {}

    # Combined lookup: FC direct edges + prior NNO tables (for step discovery)
    combined: dict[tuple, tuple] = build_fc_lookup(fc)
    if prior_tables:
        for k, v in prior_tables.items():
            if k not in combined:
                combined[k] = v
    all_ops: set[str] = {k[0] for k in combined}

    result: dict[tuple, tuple] = {}

    def _fill_column(
        col: dict[tuple, tuple],
        d2: str,
        max_col: int,
        store_key_fn,   # (d1_tup, d2) → original (op, inputs) key
        step_map: dict,
        carry_element: str,
        carry_out: tuple,
    ) -> None:
        """Flood-fill a single column using discovered strategy; write to result."""
        strategy = _discover_fill_strategy(
            col, d2, step_map, carry_element, carry_out, combined, all_ops
        )
        if strategy is _SKIP_STEP:
            return  # No valid fill rule; don't add wrong entries

        changed = True
        while changed and len(col) < max_col:
            changed = False
            for d1_tup, c in list(col.items()):
                if len(col) >= max_col:
                    break
                # Forward: succ(d1)
                fwd_d1 = unary_chain_predict(
                    step_map, carry_element, carry_out, list(d1_tup), inverse=False
                )
                if fwd_d1 is not None:
                    nxt = tuple(fwd_d1)
                    if nxt not in col and len(col) < max_col:
                        if strategy is _SUCC_STEP:
                            fwd_c_list = unary_chain_predict(
                                step_map, carry_element, carry_out, list(c), inverse=False
                            )
                            fwd_c = tuple(fwd_c_list) if fwd_c_list is not None else None
                        else:
                            fwd_c = _apply_binary_step(
                                strategy, c, d2, combined,
                                step_map, carry_element, carry_out,
                            )
                        if fwd_c is not None:
                            col[nxt] = fwd_c
                            changed = True
                if len(col) >= max_col:
                    break
                # Backward: pred(d1) — only for succ-step (binary step has no
                # easy inverse; forward fill from d1=zero is sufficient)
                if strategy is _SUCC_STEP:
                    bwd_d1 = unary_chain_predict(
                        step_map, carry_element, carry_out, list(d1_tup), inverse=True
                    )
                    if bwd_d1 is not None:
                        prv = tuple(bwd_d1)
                        if prv not in col:
                            bwd_c_list = unary_chain_predict(
                                step_map, carry_element, carry_out, list(c), inverse=True
                            )
                            if bwd_c_list is not None:
                                col[prv] = tuple(bwd_c_list)
                                changed = True

        for d1_tup, c in col.items():
            key = store_key_fn(d1_tup, d2)
            if key not in result:
                result[key] = c

    for nno_op, step_map in chain_maps.items():
        carry_info = carry_maps.get(nno_op)
        if not carry_info:
            continue
        carry_element, carry_out = carry_info
        domain: set[str] = set(step_map.keys()) | {carry_element}

        eligible_ops = {
            e.op for e in fc.edges
            if len(e.inputs) >= 2 and e.inputs[0] in domain
        }
        for op in eligible_ops:
            for inp_len in {
                len(e.inputs) for e in fc.edges
                if e.op == op and len(e.inputs) >= 2
            }:
                partial: dict[tuple, tuple] = {}
                for edge in fc.edges:
                    if edge.op == op and len(edge.inputs) == inp_len:
                        partial[edge.inputs] = edge.outputs

                # --- Standard orientation: d1 = inputs[:-1], d2 = inputs[-1] ---
                all_d1s = {k[:-1] for k in partial}
                # Use NNO domain size as max_col for single-digit d1 so that OOD
                # entries are filled even when some training entries are in the
                # test split.  For multi-digit d1 keep the observed count as
                # a safe upper bound to prevent unbounded expansion.
                d1_width = inp_len - 1
                if d1_width == 1:
                    max_col_std = len(step_map) + 1  # full NNO domain (e.g. 10 for decimal)
                else:
                    max_col_std = len(all_d1s)  # safe observed cap
                for d2 in {k[-1] for k in partial}:
                    col: dict[tuple, tuple] = {
                        k[:-1]: v for k, v in partial.items() if k[-1] == d2
                    }
                    _fill_column(
                        col, d2, max_col_std,
                        store_key_fn=lambda d1, d2_, op_=op: (op_, d1 + (d2_,)),
                        step_map=step_map,
                        carry_element=carry_element,
                        carry_out=carry_out,
                    )

                # --- Reversed orientation (2-input only): d1 = inputs[-1], d2 = inputs[0] ---
                # Handles ops whose NNO induction is over the last argument
                # (e.g. pow(base, succ(exp)) = mul(pow(base, exp), base)).
                if inp_len == 2:
                    all_d1s_rev = {(k[-1],) for k in partial}
                    # Reversed d1 is always single-token for 2-input ops
                    max_col_rev = len(step_map) + 1

                    for d2_rev in {k[0] for k in partial}:
                        col_rev: dict[tuple, tuple] = {
                            (k[-1],): v
                            for k, v in partial.items() if k[0] == d2_rev
                        }
                        _fill_column(
                            col_rev, d2_rev, max_col_rev,
                            store_key_fn=lambda d1, d2_, op_=op: (op_, (d2_,) + d1),
                            step_map=step_map,
                            carry_element=carry_element,
                            carry_out=carry_out,
                        )

    return result


# ---------------------------------------------------------------------------
# NNO column-walk helpers
# ---------------------------------------------------------------------------

_SUCC_STEP = object()  # sentinel: use succ as step function (consistent with data)
_SKIP_STEP = object()  # sentinel: no consistent fill rule; skip this column


def _discover_fill_strategy(
    col: dict,
    d2: str,
    step_map: dict[str, str],
    carry_element: str,
    carry_out: tuple,
    combined: dict,
    all_ops: set,
):
    """Determine the fill strategy for a binary-op column.

    Returns
    -------
    _SUCC_STEP  — succ-step is consistent with observed consecutive pairs
                  (or there are no pairs to check, so succ is the default).
    str         — unique binary op G such that G(col[d1], d2) = col[succ(d1)]
                  for all observed consecutive pairs.
    _SKIP_STEP  — succ-step is NOT consistent and no unique binary op found;
                  do not flood-fill this column (prevents wrong entries for
                  non-additive ops like pow in standard orientation).
    """
    pairs: list[tuple] = []
    for d1_tup in col:
        fwd = unary_chain_predict(
            step_map, carry_element, carry_out, list(d1_tup), inverse=False
        )
        if fwd is not None:
            succ_d1 = tuple(fwd)
            if succ_d1 in col:
                pairs.append((d1_tup, succ_d1))

    if not pairs:
        return _SUCC_STEP  # No consecutive pairs; default to succ-step

    # Test succ-step consistency: col[succ(d1)] == succ(col[d1]) for all pairs.
    # Skip pairs where succ(r_prev) is None — holes in step_map from the test
    # split should not disqualify an otherwise-consistent column.
    succ_ok = True
    n_checked = 0
    for d1_tup, succ_d1 in pairs:
        r_prev = col[d1_tup]
        r_next = col[succ_d1]
        succ_r = unary_chain_predict(
            step_map, carry_element, carry_out, list(r_prev), inverse=False
        )
        if succ_r is None:
            continue  # hole in step_map; assume consistent
        n_checked += 1
        if tuple(succ_r) != r_next:
            succ_ok = False
            break

    if succ_ok and n_checked > 0:
        return _SUCC_STEP

    # Succ-step genuinely doesn't hold; find unique binary step op G
    candidates: Optional[set] = None
    for d1_tup, succ_d1 in pairs:
        r_prev = col[d1_tup]
        r_next = col[succ_d1]
        satisfying = {
            op for op in all_ops
            if combined.get((op, r_prev + (d2,))) == r_next
        }
        candidates = satisfying if candidates is None else candidates & satisfying
        if candidates is not None and not candidates:
            return _SKIP_STEP

    if candidates and len(candidates) == 1:
        return next(iter(candidates))
    return _SKIP_STEP


def _apply_binary_step(
    step_op: str,
    c: tuple,
    d2: str,
    combined: dict,
    step_map: dict[str, str],
    carry_element: str,
    carry_out: tuple,
) -> Optional[tuple]:
    """Apply binary step G(c, d2) → next result.

    First tries direct lookup G(c, d2) in combined.  If c is multi-digit and
    the direct lookup misses, falls back to digit-by-digit computation using
    the LSB entry G(c[-1], d2) from combined and carry propagation via
    unary_chain_predict.

    The multi-digit fallback is only valid when the carry out of the LSB step
    is a binary carry (carry_out[0]).  This covers additive step ops (e.g.
    add used as step for mul).  For multiplicative step ops (e.g. mul as step
    for pow), only the direct lookup is used, preventing incorrect results.
    """
    # Direct lookup: works for single-digit c or when full entry is in combined
    direct = combined.get((step_op, c + (d2,)))
    if direct is not None:
        return direct

    # Multi-digit fallback: only for len(c) > 1
    if len(c) <= 1:
        return None

    # Compute G(c[-1], d2) for the least-significant digit
    lsb_result = combined.get((step_op, (c[-1], d2)))
    if lsb_result is None:
        # Try commutative order
        lsb_result = combined.get((step_op, (d2, c[-1])))
    if lsb_result is None:
        return None

    if len(lsb_result) == 1:
        # No carry: replace LSB, keep higher digits unchanged
        return c[:-1] + lsb_result

    # Two-token result: only safe when the carry is a binary (0/1) carry,
    # i.e. the high digit equals carry_out[0] (the NNO overflow token).
    # For additive step ops: 9+7=16, carry=1 → lsb_result[0]='1'==carry_out[0] ✓
    # For multiplicative:  6×9=54, carry=5 → lsb_result[0]='5'≠carry_out[0] → skip
    if lsb_result[0] != carry_out[0]:
        return None  # Non-binary carry; can't handle with succ propagation

    new_lsb = (lsb_result[-1],)
    propagated = unary_chain_predict(
        step_map, carry_element, carry_out, list(c[:-1]), inverse=False
    )
    if propagated is not None:
        return tuple(propagated) + new_lsb
    return None


def apply_process_rule(
    rule: ProcessRule,
    input_digits: list[str],
    rules_dict: Optional[dict] = None,
) -> Optional[list[str]]:
    """Stub — returns None, falling through to Level 2+3."""
    return None


# ---------------------------------------------------------------------------
# Binary fold rules — the categorical composition engine (CT_REFERENCE §19)
# ---------------------------------------------------------------------------

@dataclass
class BinaryFoldRule:
    """Discovered NNO fold rule for a binary op (CT_REFERENCE §19).

    Encodes: op(zero, other) = base_result(other)
             op(succ(n), other) = step(op(n, other), other)

    where 'step' is either:
      - None  → apply succ (unary): op(succ(n),m) = succ(op(n,m))
      - str   → apply binary op:    op(succ(n),m) = step_op(op(n,m), m)

    induction_arg:
        0 = induct on first argument (add, mul pattern)
        1 = induct on second argument (pow pattern)

    base_fixed:
        None  = base result equals the OTHER arg (identity: add(0,m)=m)
        str   = base result is a fixed single token (mul(0,m)='0', pow(b,0)='1')
    """

    op: str
    induction_arg: int          # 0 or 1
    base_fixed: Optional[str]   # None = identity; str = fixed token
    step_op: Optional[str]      # None = unary step; str = binary op name
    step_inverse: bool = False  # True → step is pred (inverse of succ); False → succ
    evidence: int = 0           # number of training triples that verified this rule


def discover_binary_fold_rules(fc: FreeCategoryGraph) -> dict[str, BinaryFoldRule]:
    """Discover BinaryFoldRule for each binary op in the FC (CT_REFERENCE §19).

    For each binary op (exactly 2 input tokens), tests whether the NNO fold
    structure holds:
        op(zero, other) = base(other)          ← base case
        op(succ(n), other) = step(op(n, other), other)  ← inductive step

    'step' is discovered by testing candidates:
        1. succ (unary): op(succ(n),m) = succ(op(n,m))
        2. any other binary op G: op(succ(n),m) = G(op(n,m), m)

    Tries both induction_arg=0 and induction_arg=1.  Records the rule with
    the highest evidence count (most verified training triples).

    No operator names hardcoded.  Works on anonymized token sets.
    """
    # Build direct lookup and succ tools
    fc_lookup = build_fc_lookup(fc)
    chain_maps = build_unary_chain_maps(fc)
    carry_maps = build_unary_carry_maps(fc)

    if not chain_maps:
        return {}

    # Find the succ NNO: the op with a full linear chain and a zero_candidate
    succ_op: Optional[str] = None
    succ_step_map: dict[str, str] = {}
    succ_carry: Optional[tuple] = None
    zero_digit: Optional[str] = None

    # Pick the NNO candidate with the longest chain as the true successor.
    # Short-chain candidates (sq, sqrt) are rejected — they have 2-3 entries.
    best_nno = None
    for nno in fc.nno_candidates:
        carry = carry_maps.get(nno.op)
        if carry is not None:
            if best_nno is None or len(nno.successor_map) > len(best_nno.successor_map):
                best_nno = nno
    if best_nno is not None:
        succ_op = best_nno.op
        _raw_map = best_nno.successor_map
        succ_carry = carry_maps[best_nno.op]
        zero_digit = best_nno.zero_candidate
        # Complete the chain: infer edges missing due to train/test split gaps
        succ_step_map = complete_succ_map(_raw_map, zero_digit, succ_carry[0])

    if succ_op is None or zero_digit is None:
        return {}

    carry_element, carry_out = succ_carry
    one_digit = succ_step_map.get(zero_digit)  # succ(zero) = one

    # Collect all binary edges: op → list of (d1, d2, output)
    binary_edges: dict[str, list[tuple]] = {}
    for edge in fc.edges:
        if len(edge.inputs) == 2:
            op = edge.op
            if op == succ_op:
                continue  # skip the succ op itself
            if op not in binary_edges:
                binary_edges[op] = []
            binary_edges[op].append((edge.inputs[0], edge.inputs[1], edge.outputs))

    # For step discovery, build a quick lookup: (step_op, (a, b)) → output
    # Only for entries where both a and b are single tokens
    step_lookup: dict[tuple, tuple] = {}
    for key, val in fc_lookup.items():
        step_lookup[key] = val

    rules: dict[str, BinaryFoldRule] = {}

    def _succ_tuple(t: tuple) -> Optional[tuple]:
        r = unary_chain_predict(succ_step_map, carry_element, carry_out, list(t), inverse=False)
        return tuple(r) if r is not None else None

    def _pred_of(d: str) -> Optional[str]:
        """pred(d) for single digit d."""
        pred_map = {v: k for k, v in succ_step_map.items()}
        return pred_map.get(d)

    for op, triples in binary_edges.items():
        best_rule: Optional[BinaryFoldRule] = None
        best_ev = 0

        for ind_arg in (0, 1):  # try both induction axes
            # Identify base case: op(zero, other) for ind_arg=0, op(other, zero) for ind_arg=1
            base_observations: list[tuple] = []
            for d1, d2, out in triples:
                ind_d = d1 if ind_arg == 0 else d2
                other_d = d2 if ind_arg == 0 else d1
                if ind_d == zero_digit:
                    base_observations.append((other_d, out))

            if not base_observations:
                continue  # No base case observed for this axis

            # Require at least 3 distinct base observations to avoid spurious
            # identity detection from a single data point.  For example,
            # sub(0, b=0) = 0 is the only base observation for sub ind=0,
            # which accidentally looks like "identity" even though sub(0, b)
            # is undefined for b > 0.  add/mul/pow all have 8-10 base obs.
            if len(base_observations) < 3:
                continue

            # Determine base_fixed: is base(other) == (other,) or a fixed token?
            identity_count = sum(1 for other_d, out in base_observations if out == (other_d,))
            fixed_counts: dict[str, int] = {}
            for other_d, out in base_observations:
                if len(out) == 1 and out[0] != other_d:
                    fixed_counts[out[0]] = fixed_counts.get(out[0], 0) + 1

            if identity_count >= len(base_observations) * 0.8:
                base_fixed = None  # identity
            elif fixed_counts:
                base_fixed = max(fixed_counts, key=lambda k: fixed_counts[k])
            else:
                continue  # Can't determine base case

            # Find inductive step: collect consecutive pairs (d_n, d_succ, d2, out_n, out_succ)
            # where d_succ = succ(d_n)
            consec_pairs: list[tuple] = []
            for d1, d2, out in triples:
                ind_d = d1 if ind_arg == 0 else d2
                other_d = d2 if ind_arg == 0 else d1
                if ind_d == zero_digit:
                    continue
                # Look for the entry with ind_d - 1 (pred of ind_d)
                pred_ind = _pred_of(ind_d)
                if pred_ind is None:
                    continue
                pred_inputs = (pred_ind, other_d) if ind_arg == 0 else (other_d, pred_ind)
                out_prev = fc_lookup.get((op, pred_inputs))
                if out_prev is None:
                    continue
                consec_pairs.append((out_prev, out, other_d))

            if not consec_pairs:
                continue

            # Test step candidates
            # Candidate 1a: succ (unary forward step).
            # Skip pairs where _succ_tuple returns None (succ_map gap from
            # the 80/20 split) — require consistency only on checkable pairs.
            succ_checkable = [
                (p, n) for p, n, _ in consec_pairs
                if _succ_tuple(p) is not None
            ]
            succ_ev = sum(1 for p, n in succ_checkable if _succ_tuple(p) == n)
            if succ_checkable and succ_ev == len(succ_checkable):
                rule = BinaryFoldRule(op=op, induction_arg=ind_arg,
                                      base_fixed=base_fixed, step_op=None,
                                      step_inverse=False, evidence=succ_ev)
                if succ_ev > best_ev:
                    best_ev = succ_ev
                    best_rule = rule
                continue  # succ step found; no need to check binary ops

            # Candidate 1b: pred (unary inverse step — for sub-like ops).
            # Tests: op(succ(n), d) = pred(op(n, d)).
            # This is the NNO fold rule for subtraction:
            #   sub(a, 0) = a  and  sub(a, succ(b)) = pred(sub(a, b))
            def _pred_tuple(t: tuple) -> Optional[tuple]:
                r = unary_chain_predict(succ_step_map, carry_element, carry_out, list(t), inverse=True)
                return tuple(r) if r is not None else None

            pred_checkable = [
                (p, n) for p, n, _ in consec_pairs
                if _pred_tuple(p) is not None
            ]
            pred_ev = sum(1 for p, n in pred_checkable if _pred_tuple(p) == n)
            if pred_checkable and pred_ev == len(pred_checkable):
                rule = BinaryFoldRule(op=op, induction_arg=ind_arg,
                                      base_fixed=base_fixed, step_op=None,
                                      step_inverse=True, evidence=pred_ev)
                if pred_ev > best_ev:
                    best_ev = pred_ev
                    best_rule = rule
                continue  # pred step found; no need to check binary ops

            # Candidate 2: binary step op G — G(out_prev, other_d) == out_next
            # Only check entries where out_prev is a single token (in step_lookup)
            testable = [(p, n, m) for p, n, m in consec_pairs if len(p) == 1]
            if not testable:
                continue

            step_candidates = set(binary_edges.keys()) - {op}
            for step_op_cand in step_candidates:
                # Only count pairs where step_lookup has an entry (skip gaps
                # from the 80/20 train/test split, same as succ_checkable above).
                checkable_step = [
                    (p, n, m) for p, n, m in testable
                    if step_lookup.get((step_op_cand, (p[0], m))) is not None
                ]
                ev = sum(
                    1 for out_prev, out_next, other_d in checkable_step
                    if step_lookup.get((step_op_cand, (out_prev[0], other_d))) == out_next
                )
                if checkable_step and ev == len(checkable_step):
                    rule = BinaryFoldRule(op=op, induction_arg=ind_arg,
                                         base_fixed=base_fixed,
                                         step_op=step_op_cand,
                                         evidence=ev)
                    if ev > best_ev:
                        best_ev = ev
                        best_rule = rule

        if best_rule is not None:
            rules[op] = best_rule

    return rules


# ---------------------------------------------------------------------------
# FC lookup table constructors  (for Level 0.5 in predict.py)
# ---------------------------------------------------------------------------

def build_fc_lookup(fc: FreeCategoryGraph) -> dict[tuple, tuple]:
    """Build (op, inputs_tuple) → outputs_tuple direct lookup from FC edges.

    Returns the first observed outputs for each (op, inputs) pair.
    Used by Predictor Level 0.5 for exact-match prediction on in-FC sequences.
    """
    lookup: dict[tuple, tuple] = {}
    for edge in fc.edges:
        key = (edge.op, edge.inputs)
        if key not in lookup:
            lookup[key] = edge.outputs
    return lookup


def build_adj_lookup(fc: FreeCategoryGraph) -> dict[tuple, tuple]:
    """Build adjunction-based inference lookup (CT_REFERENCE §4).

    For each detected adjunction left_op ⊣ right_op, populates two kinds of
    entries so that EITHER side can be predicted from the OTHER side's training
    edges:

    Unary adjunctions (preserved_position=None):
        From each right_op-edge  right_op(c) → a:
            adj_lookup[(left_op, a_tuple)] = c_tuple       [left from right]
        From each left_op-edge   left_op(a) → c:
            adj_lookup[(right_op, c_tuple)] = a_tuple      [right from left]

    N-ary adjunctions (preserved_position=p):
        Discovered preserved argument: inputs[p] appears verbatim in the
        adjoint call's inputs.  Specifically, for left_op(inputs) = outputs:
            G(outputs + (inputs[p],)) = inputs_without_p

        From each left_op-edge: build right_op lookup entry.
        From each right_op-edge: build left_op lookup entry using all possible
        split points (split right_op.inputs into c_tokens + preserved_token for
        every prefix length ≥1).  Wrong-arity keys are never queried.

    Covers succ⊣pred, add⊣sub, mul⊣div — no operator names hardcoded.

    Returns
    -------
    {(op, inputs_tuple): inferred_outputs_tuple}
    """
    result: dict[tuple, tuple] = {}

    for adj in fc.adjunctions:
        left_op, right_op = adj.left_op, adj.right_op
        p = adj.preserved_position

        if p is None:
            # --- Unary adjunction ---
            # From left_op-edges: right_op(outputs) = inputs
            for edge in fc.edges:
                if edge.op != left_op:
                    continue
                key = (right_op, edge.outputs)
                if key not in result:
                    result[key] = edge.inputs
            # From right_op-edges: left_op(outputs) = inputs
            for edge in fc.edges:
                if edge.op != right_op:
                    continue
                key = (left_op, edge.outputs)
                if key not in result:
                    result[key] = edge.inputs

        else:
            # --- N-ary adjunction with discovered preserved position p ---
            # adj.input_len is the exact token count for which this adjunction
            # was verified.  Only apply to edges with that exact input length
            # to avoid cross-arity contamination (e.g. a pp=1 rule discovered
            # on 2-token sub edges must not fire on 3-token sub edges).

            # From left_op-edges (exact input_len): build right_op lookup.
            for edge in fc.edges:
                if edge.op != left_op:
                    continue
                if adj.input_len is not None and len(edge.inputs) != adj.input_len:
                    continue
                if len(edge.inputs) <= p:
                    continue
                pres = (edge.inputs[p],)
                non_pres = edge.inputs[:p] + edge.inputs[p + 1:]
                key = (right_op, edge.outputs + pres)
                if key not in result:
                    result[key] = non_pres

            # From right_op-edges (exact input_len): build left_op lookup.
            # The right_op's input_len is len(left_op outputs) + 1 (for the
            # single preserved token).  We know the preserved token sits at
            # position p in the right_op's inputs (mirroring the left_op
            # structure), so split the right_op inputs at position p and
            # (len-p-1) trailing tokens form the non-preserved part.
            # Rather than hardcoding the split, we match by the registered
            # right_op input_len for THIS adjunction direction, which equals
            # the typical right_op token count for this adjunction.
            # Safe approach: only use right_op edges whose input length
            # corresponds to a registered right_op adjunction's input_len
            # (i.e., edges that have a matching adjunction in the reverse
            # direction — already stored or will be stored).
            # Simpler: use the right_op adj that has left_op as right_op.
            # Since we iterate all adjunctions, the reverse direction will
            # handle itself when we process adj (right_op ⊣ left_op).
            pass  # reverse direction covered when that AdjunctionPair is processed

    return result
