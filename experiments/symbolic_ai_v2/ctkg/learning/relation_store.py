"""Relational tuple store — hypergraph representation for sequence data.

Each training sequence is segmented into named roles using two kinds of
structural tokens:

  OUTPUT_DELIMS  — {eq, step, ans, <eos>} — mark the start of each output
                   phase.  Every token between two consecutive output delimiters
                   belongs to that output role.

  INPUT_SEPS     — tokens that appear at consistent positions in the input
                   segment and act as named separators between input argument
                   groups.  These are LEARNED from data: a token qualifies as
                   an input separator if it appears at the same position in
                   ≥80% of training sequences for that op.  Any token can be
                   a separator — there is NO pre-seeded keyword list.  This is
                   the Iron Rule: separator identity is discovered from
                   distributional statistics alone.

The result is a Relation dict with structured input and output roles:

    Relation('eval',
             input_roles=[('', ['2']), ('x', ['1']), ('at', ['5'])],
             output_roles=[('step', ['1','0']), ('ans', ['1','1'])])

-- Phase XXIII: all string node identifiers replaced with NodeId (int). --

All token strings and role names are now opaque NodeId integers.  String ↔
NodeId conversion lives in TOKEN_GRAPH (ctkg/core/node.py).  No string is
stored in any node above the character level.

The public methods of RelationStore still accept str for convenience: any str
parameter is encoded to NodeId at the method boundary.  All returned data
structures (Relations, RelationRules) use NodeId internally.

Usage in predict.py:

    store = RelationStore()
    store.update_batch(chain_rule_seqs)
    step_corpus = store.eq_corpus_for_role(STEP_NODE)  # → discover_rules(...)
    eq_corpus   = store.eq_corpus_for_role(EQ_NODE)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from experiments.symbolic_ai_v2.ctkg.core.node import (
    NodeId,
    TOKEN_GRAPH,
    OUTPUT_DELIMS,
    STEP_NODE,
    EQ_NODE,
    ANS_NODE,
    EOS_NODE,
    is_positional_role,
)

if TYPE_CHECKING:
    from experiments.symbolic_ai_v2.ctkg.core.dependent_type import TypeTerm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of training sequences that must agree on a position for it to be
# treated as a structural separator.  Any token (not a pre-seeded list) can
# qualify — Iron Rule compliance.
_SEP_THRESHOLD: float = 0.80


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    """Structured representation of one training/test sequence.

    Phase XXIII: all str fields replaced with NodeId.

    Parameters
    ----------
    op : NodeId
        The operator token (first token of the sequence).
    input_roles : list of (role_name_NodeId, list[NodeId])
        Ordered groups of input tokens, separated by input separator tokens.
        The role_name is the separator token that PRECEDES this group, or
        TOKEN_GRAPH.encode('') for the first group (nothing precedes it).
    output_roles : list of (role_name_NodeId, list[NodeId])
        Output phases, keyed by their opening delimiter token
        (EQ_NODE, STEP_NODE, ANS_NODE).  Ordered in sequence order.
    """
    op: NodeId
    input_roles: list[tuple[NodeId, list[NodeId]]] = field(default_factory=list)
    output_roles: list[tuple[NodeId, list[NodeId]]] = field(default_factory=list)
    # Optional type-distribution fields (Phase XI extension).
    # type_dist = dict[ConceptId, float]; kept as plain dict to avoid
    # circular imports with operad.py / concept_lattice.py.
    input_type_dists: list[tuple[NodeId, dict]] = field(default_factory=list)
    output_type_dists: list[tuple[NodeId, dict]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def input_role(self, name: NodeId) -> Optional[list[NodeId]]:
        """Return the token list for the input role named *name*, or None."""
        for rname, toks in self.input_roles:
            if rname == name:
                return toks
        return None

    def output_role(self, name: NodeId) -> Optional[list[NodeId]]:
        """Return the token list for the *first* output role named *name*."""
        for rname, toks in self.output_roles:
            if rname == name:
                return toks
        return None

    def all_output_roles(self, name: NodeId) -> list[list[NodeId]]:
        """Return all output role values for roles named *name* (e.g. multi-step)."""
        return [toks for rname, toks in self.output_roles if rname == name]

    def flat_input(self) -> list[NodeId]:
        """Reconstruct the flat input token sequence from named roles."""
        _EMPTY = TOKEN_GRAPH.encode('')
        result: list[NodeId] = []
        for sep, toks in self.input_roles:
            if sep != _EMPTY:
                result.append(sep)
            result.extend(toks)
        return result


# ---------------------------------------------------------------------------
# RelationStore
# ---------------------------------------------------------------------------

class RelationStore:
    """Learns operator schemas from training sequences and stores Relations.

    Phase XXIII: all internal storage uses NodeId.  Public methods that
    accept str parameters encode them to NodeId automatically.
    """

    def __init__(self) -> None:
        # op → list of Relation objects (all NodeId)
        self._relations: dict[NodeId, list[Relation]] = {}
        # op → learned input schema: sorted [(position_in_input, sep_NodeId), ...]
        # Positional schemas use role NodeIds P0_NODE, P1_NODE, ... as sep_NodeId
        self._schemas: dict[NodeId, list[tuple[int, NodeId]]] = {}
        # op → frozenset of all observed output role NodeIds
        self._output_role_names: dict[NodeId, frozenset[NodeId]] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def update_batch(self, seqs: list[list[str]]) -> None:
        """Add a batch of training sequences and learn operator schemas.

        Each sequence is a list of *string* tokens.  Strings are encoded to
        NodeIds at this boundary — all internal processing uses NodeId.
        """
        # Encode every sequence to NodeId lists
        encoded: list[list[NodeId]] = [
            TOKEN_GRAPH.encode_seq(seq) for seq in seqs if seq
        ]
        if not encoded:
            return

        # Group encoded sequences by op NodeId
        raw: dict[NodeId, list[list[NodeId]]] = {}
        for seq in encoded:
            if not seq:
                continue
            op = seq[0]
            raw.setdefault(op, []).append(seq)

        # Learn input schemas per op
        for op, op_seqs in raw.items():
            self._schemas[op] = _learn_input_schema(op_seqs)

        # Extract and store Relations
        for op, op_seqs in raw.items():
            schema = self._schemas.get(op, [])
            rels = self._relations.setdefault(op, [])
            out_roles: set[NodeId] = set()
            for seq in op_seqs:
                rel = _extract_relation(seq, schema)
                if rel is not None:
                    rels.append(rel)
                    for rname, _ in rel.output_roles:
                        out_roles.add(rname)
            existing = self._output_role_names.get(op, frozenset())
            self._output_role_names[op] = existing | frozenset(out_roles)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_relations(self, op: str) -> list[Relation]:
        """Return all stored Relations for *op* (str boundary method)."""
        return self._relations.get(TOKEN_GRAPH.encode(op), [])

    def get_relations_by_id(self, op: NodeId) -> list[Relation]:
        """Return all stored Relations for op NodeId."""
        return self._relations.get(op, [])

    def get_schema(self, op: str) -> list[tuple[int, NodeId]]:
        """Return the learned input schema for *op* (str boundary method)."""
        return self._schemas.get(TOKEN_GRAPH.encode(op), [])

    def get_schema_by_id(self, op: NodeId) -> list[tuple[int, NodeId]]:
        """Return the learned input schema for op NodeId."""
        return self._schemas.get(op, [])

    def has_input_seps(self, op: str) -> bool:
        """Return True if *op* has learned input separators."""
        return bool(self._schemas.get(TOKEN_GRAPH.encode(op)))

    def all_output_role_names(self, op: str) -> frozenset[NodeId]:
        """Return all output role NodeIds observed for *op* in training data."""
        return self._output_role_names.get(TOKEN_GRAPH.encode(op), frozenset())

    def all_output_role_names_by_id(self, op: NodeId) -> frozenset[NodeId]:
        """Return all output role NodeIds for op NodeId."""
        return self._output_role_names.get(op, frozenset())

    def extract_relation(self, seq: list[str]) -> Optional[Relation]:
        """Extract a Relation from a new (possibly OOD) sequence."""
        if not seq:
            return None
        encoded = TOKEN_GRAPH.encode_seq(seq)
        op = encoded[0]
        schema = self._schemas.get(op, [])
        return _extract_relation(encoded, schema)

    def extract_relation_from_ids(self, seq: list[NodeId]) -> Optional[Relation]:
        """Extract a Relation from an already-encoded NodeId sequence."""
        if not seq:
            return None
        op = seq[0]
        schema = self._schemas.get(op, [])
        return _extract_relation(seq, schema)

    # ------------------------------------------------------------------
    # eq-corpus builders for discover_rules
    # ------------------------------------------------------------------

    def eq_corpus_for_role(
        self,
        role: NodeId,
        ops: Optional[frozenset[NodeId]] = None,
        merge_digits: bool = False,
        nno_atoms: frozenset[NodeId] = frozenset(),
    ) -> list[list[NodeId]]:
        """Build an eq-format corpus targeting *role* across all ops.

        For each relation that has an output role named *role*, emit:
            [op_NodeId] + flat_input_NodeIds + [EQ_NODE] + role_NodeIds

        Parameters
        ----------
        role : NodeId
            Output role to target (e.g. STEP_NODE, ANS_NODE, EQ_NODE).
        ops : frozenset[NodeId], optional
            Restrict to these op NodeIds (default: all ops).
        merge_digits : bool
            If True, merge consecutive NNO-alphabet tokens in the output.
        nno_atoms : frozenset[NodeId]
            NNO alphabet used for digit merging.

        Returns
        -------
        list of NodeId sequences, each ending with the role's tokens.
        """
        result: list[list[NodeId]] = []
        for op, rels in self._relations.items():
            if ops is not None and op not in ops:
                continue
            for rel in rels:
                role_val = rel.output_role(role)
                if role_val is None:
                    continue
                if not role_val:
                    continue
                input_toks = rel.flat_input()
                output_toks = list(role_val)
                if merge_digits and nno_atoms:
                    output_toks = _merge_digit_runs(output_toks, nno_atoms)
                result.append([op] + input_toks + [EQ_NODE] + output_toks)
        return result

    def ops_with_role(self, role: NodeId) -> frozenset[NodeId]:
        """Return the set of op NodeIds that have at least one relation with *role*."""
        result: set[NodeId] = set()
        for op, rels in self._relations.items():
            for rel in rels:
                if rel.output_role(role) is not None:
                    result.add(op)
                    break
        return frozenset(result)

    def ops_with_step(self) -> frozenset[NodeId]:
        """Shorthand: ops with a 'step' output role."""
        return self.ops_with_role(STEP_NODE)

    def ops_with_input_seps(self) -> frozenset[NodeId]:
        """Return ops that have at least one learned input separator."""
        return frozenset(op for op, schema in self._schemas.items() if schema)

    def ops_with_schema(self) -> frozenset[NodeId]:
        """Return ops that have any learned schema (separator or positional)."""
        return frozenset(op for op, schema in self._schemas.items() if schema)

    def ops_with_positional_schema(self) -> frozenset[NodeId]:
        """Return ops that have a positional schema (no separators, fixed length).

        Positional role NodeIds: P0_NODE, P1_NODE, ... as assigned by
        _learn_input_schema when no separator tokens are found.
        """
        result: set[NodeId] = set()
        for op, schema in self._schemas.items():
            if schema and is_positional_role(schema[0][1]):
                result.add(op)
        return frozenset(result)

    # ------------------------------------------------------------------
    # Legacy str-based wrappers (for callers not yet updated to NodeId)
    # ------------------------------------------------------------------

    def ops_with_role_str(self, role: str) -> frozenset[str]:
        """str-boundary wrapper around ops_with_role."""
        ids = self.ops_with_role(TOKEN_GRAPH.encode(role))
        return frozenset(TOKEN_GRAPH.decode(op) for op in ids)

    def ops_with_schema_str(self) -> frozenset[str]:
        """str-boundary wrapper around ops_with_schema."""
        return frozenset(TOKEN_GRAPH.decode(op) for op in self.ops_with_schema())

    def ops_with_positional_schema_str(self) -> frozenset[str]:
        """str-boundary wrapper around ops_with_positional_schema."""
        return frozenset(TOKEN_GRAPH.decode(op) for op in self.ops_with_positional_schema())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _learn_input_schema(seqs: list[list[NodeId]]) -> list[tuple[int, NodeId]]:
    """Identify input-separator positions for an operator.

    Returns a sorted list of (position_in_input_segment, separator_NodeId)
    pairs.  A token qualifies if it appears at the same position in ≥80% of
    training sequences for this op.  Any token can become a separator — no
    pre-seeded keyword list (Iron Rule compliance).

    Positional schemas: when no separator tokens are found and all input
    segments have the same length, assigns P0_NODE, P1_NODE, ... roles.
    """
    if not seqs:
        return []

    # Extract the input segment (everything between op and first OUTPUT_DELIM)
    input_segs: list[list[NodeId]] = []
    for seq in seqs:
        end = len(seq)
        for i, tok in enumerate(seq[1:], 1):
            if tok in OUTPUT_DELIMS:
                end = i
                break
        input_segs.append(seq[1:end])

    if not input_segs:
        return []

    n = len(input_segs)
    threshold = _SEP_THRESHOLD * n

    # Count occurrences of every token at each position.
    # pos_counts[pos][tok_NodeId] = count
    pos_counts: dict[int, dict[NodeId, int]] = {}
    for seg in input_segs:
        for i, tok in enumerate(seg):
            pos_counts.setdefault(i, {}).setdefault(tok, 0)
            pos_counts[i][tok] += 1

    schema: list[tuple[int, NodeId]] = []
    for pos, tok_counts in sorted(pos_counts.items()):
        for tok, cnt in tok_counts.items():
            if cnt >= threshold:
                schema.append((pos, tok))

    if schema:
        return schema

    # No separator tokens found — try positional schema.
    lengths = {len(seg) for seg in input_segs}
    if len(lengths) == 1:
        L = next(iter(lengths))
        if L > 0:
            return [(i, TOKEN_GRAPH.encode(f'p{i}')) for i in range(L)]

    return schema


def _extract_relation(seq: list[NodeId], schema: list[tuple[int, NodeId]]) -> Optional[Relation]:
    """Extract a Relation from *seq* using the given input *schema*.

    Parameters
    ----------
    seq : list[NodeId]
        Raw token sequence (starts with op).
    schema : list of (position, sep_NodeId)
        Input separator schema learned from training data.
    """
    if not seq:
        return None
    op = seq[0]
    body = seq[1:]

    _EMPTY_NODE = TOKEN_GRAPH.encode('')

    # ---- Split body into input segment and output segment ----
    input_end = len(body)
    first_output_delim: Optional[NodeId] = None
    for i, tok in enumerate(body):
        if tok in OUTPUT_DELIMS:
            input_end = i
            first_output_delim = tok
            break

    input_tokens = body[:input_end]

    # ---- Parse input_tokens into roles using schema ----
    input_roles: list[tuple[NodeId, list[NodeId]]] = []
    _is_positional = bool(schema and is_positional_role(schema[0][1]))

    if _is_positional:
        for pos, role_nid in schema:
            if pos < len(input_tokens):
                input_roles.append((role_nid, [input_tokens[pos]]))
    elif schema:
        sorted_schema = sorted(schema, key=lambda x: x[0])
        prev_end = 0
        prev_sep: NodeId = _EMPTY_NODE
        for pos, sep_nid in sorted_schema:
            if pos >= len(input_tokens):
                break
            group = input_tokens[prev_end:pos]
            input_roles.append((prev_sep, group))
            prev_sep = sep_nid
            prev_end = pos + 1  # skip the separator itself
        # Last group after the final separator
        input_roles.append((prev_sep, input_tokens[prev_end:]))
    else:
        # No separators: single group
        input_roles = [(_EMPTY_NODE, input_tokens)]

    # ---- Parse output into roles ----
    output_roles: list[tuple[NodeId, list[NodeId]]] = []
    if first_output_delim is not None:
        output_body = body[input_end:]  # starts with the first delimiter
        current_delim: Optional[NodeId] = None
        current_group: list[NodeId] = []
        for tok in output_body:
            if tok in OUTPUT_DELIMS:
                if current_delim is not None and current_delim != EOS_NODE:
                    output_roles.append((current_delim, current_group))
                current_delim = tok
                current_group = []
            else:
                if current_delim is not None:
                    current_group.append(tok)
        # Flush last group (unless it's EOS_NODE)
        if current_delim is not None and current_delim != EOS_NODE:
            output_roles.append((current_delim, current_group))

    return Relation(op=op, input_roles=input_roles, output_roles=output_roles)


# ---------------------------------------------------------------------------
# Arity-free rule discovery (hypergraph approach)
# ---------------------------------------------------------------------------

@dataclass
class RelationRule:
    """A rule for one output role, discovered from relational tuples.

    Phase XXIII: all str fields replaced with NodeId.

    Represents: output_role_NodeId = engine.compute(op_NodeId, arg1_val, arg2_val)

    where arg1 and arg2 are role NodeIds:
      - input role NodeIds: EMPTY_NODE (first unnamed group), 'x'-NodeId, etc.
      - output role NodeIds: STEP_NODE, ANS_NODE, EQ_NODE

    Phase XXI — dependent type annotations (optional):
        arg1_type, arg2_type, output_type carry the type of each role's value
        as discovered from the NNO chain.  ordinal=None in these fields means
        the rule is universally quantified over all NNO ordinals.

    Phase XXII — probability monad:
        total_obs is the denominator for confidence.
        evaluate() returns a Kleisli morphism A → Dist(B) encoded as
        dict[tuple[NodeId,...], float] instead of Optional[str].
    """
    output_role: NodeId
    op_name: NodeId
    arg1: NodeId  # role NodeId for first argument
    arg2: NodeId  # role NodeId for second argument
    evidence: int = 0
    # Phase XXII
    total_obs: int = 0
    # Phase XXI — type annotations (None = not yet inferred)
    arg1_type: Optional['TypeTerm'] = None   # type: ignore[type-arg]
    arg2_type: Optional['TypeTerm'] = None
    output_type: Optional['TypeTerm'] = None

    @property
    def confidence(self) -> float:
        """Empirical probability that this rule correctly predicts the output."""
        if self.total_obs <= 0:
            return 1.0
        return self.evidence / self.total_obs

    def evaluate(
        self,
        role_values: dict[NodeId, tuple[NodeId, ...]],
        engine,
    ) -> 'dict[tuple[NodeId, ...], float]':
        """Evaluate this rule as a Kleisli morphism A → Dist(B).

        Phase XXIII: role_values maps role NodeId → tuple of NodeIds.
        Returns dict[result_tuple → probability].

        The engine operates on str tokens; NodeIds are decoded at the engine
        boundary and results are encoded back to NodeId tuples.

        Returns {} if inputs are unavailable or the engine lookup misses.
        """
        if engine is None:
            return {}
        v1_tup = role_values.get(self.arg1)
        v2_tup = role_values.get(self.arg2)
        if v1_tup is None or v2_tup is None:
            return {}
        # Decode op NodeId to str (engine is str-based until Stage 9)
        op_str = TOKEN_GRAPH.decode(self.op_name)
        # Engine expects str tokens; decode NodeIds → str, encode results → NodeId
        if len(v1_tup) == 1 and len(v2_tup) == 1:
            a_str = TOKEN_GRAPH.decode(v1_tup[0])
            b_str = TOKEN_GRAPH.decode(v2_tup[0])
            str_result = engine.compute(op_str, a_str, b_str)
        else:
            a_strs = tuple(TOKEN_GRAPH.decode(n) for n in v1_tup)
            b_strs = tuple(TOKEN_GRAPH.decode(n) for n in v2_tup)
            str_result = engine.compute_tup(op_str, a_strs, b_strs)
        if str_result is None:
            return {}
        result_tup = tuple(TOKEN_GRAPH.encode(t) for t in str_result)
        return {result_tup: self.confidence}


def discover_relation_rules(
    relations: list['Relation'],
    engine,
    min_evidence: int = 2,
    unknown_tolerance: float = 0.20,
    mismatch_tolerance: float = 0.0,
    type_context: Optional[dict[NodeId, 'TypeTerm']] = None,
) -> list[RelationRule]:
    """Discover RelationRules from a list of Relations.

    Phase XXIII: all string identifiers replaced with NodeId.
    type_context maps NodeId → TypeTerm (was dict[str, TypeTerm]).

    For each output role in the relations, attempts to find a binary function
    f such that output_role_value = engine.compute(f, role_i_val, role_j_val)
    for all training relations.

    Multi-token role values are stored as tuple[NodeId, ...].

    Returns list of RelationRule sorted by evidence descending.
    """
    if not relations:
        return []
    if engine is None:
        return []

    _EMPTY_NODE = TOKEN_GRAPH.encode('')

    # Collect all input role names from the first relation
    all_input_role_names: list[NodeId] = []
    for sep, toks in relations[0].input_roles:
        all_input_role_names.append(sep)

    # Collect all output role names in order
    all_output_roles: list[NodeId] = []
    seen_out: set[NodeId] = set()
    for rel in relations:
        for rname, _ in rel.output_roles:
            if rname not in seen_out:
                seen_out.add(rname)
                all_output_roles.append(rname)

    discovered: list[RelationRule] = []
    available_source_roles = list(all_input_role_names)

    for target_role in all_output_roles:
        # Collect (role_values, target_value) for each relation
        examples: list[tuple[dict[NodeId, tuple[NodeId, ...]], tuple[NodeId, ...]]] = []
        for rel in relations:
            role_vals: dict[NodeId, tuple[NodeId, ...]] = {}
            for sep, toks in rel.input_roles:
                if toks:
                    role_vals[sep] = tuple(toks)
            for rname, toks in rel.output_roles:
                if toks:
                    role_vals[rname] = tuple(toks)

            target_val_tup = role_vals.get(target_role)
            if target_val_tup is None:
                continue
            examples.append((role_vals, target_val_tup))

        if not examples:
            available_source_roles.append(target_role)
            continue

        # Phase XV (Coproducts): collect ALL qualifying rules
        role_rules: list[RelationRule] = []
        source_roles = list(available_source_roles)

        for op_str in engine.known_ops():
            # Encode op str to NodeId for storage in RelationRule (engine is str-based)
            op_nid = TOKEN_GRAPH.encode(op_str)
            for role_a in source_roles:
                for role_b in source_roles:
                    n_match = 0
                    n_unknown = 0
                    n_mismatch = 0
                    for role_vals, target_val in examples:
                        va_tup = role_vals.get(role_a)
                        vb_tup = role_vals.get(role_b)
                        if va_tup is None or vb_tup is None:
                            n_unknown += 1
                            continue
                        # Decode NodeIds to str for the engine; encode result back
                        if len(va_tup) == 1 and len(vb_tup) == 1:
                            str_res = engine.compute(
                                op_str,
                                TOKEN_GRAPH.decode(va_tup[0]),
                                TOKEN_GRAPH.decode(vb_tup[0]),
                            )
                        else:
                            str_res = engine.compute_tup(
                                op_str,
                                tuple(TOKEN_GRAPH.decode(n) for n in va_tup),
                                tuple(TOKEN_GRAPH.decode(n) for n in vb_tup),
                            )
                        if str_res is None:
                            n_unknown += 1
                            continue
                        result_tup = tuple(TOKEN_GRAPH.encode(t) for t in str_res)
                        if result_tup != target_val:
                            n_mismatch += 1
                        else:
                            n_match += 1
                    n = len(examples)
                    if (n_match >= min_evidence
                            and n_unknown / n <= unknown_tolerance
                            and n_mismatch / n <= mismatch_tolerance):
                        # Phase XXI: infer types from a sample example
                        # type_context maps NodeId → TypeTerm
                        a1_type = a2_type = out_type = None
                        if type_context and examples:
                            sample_rv, sample_tv = examples[0]
                            from experiments.symbolic_ai_v2.ctkg.core.dependent_type import (
                                TypeTerm, token_type,
                            )
                            def _get_role_nid(rv: dict, role_nid: NodeId):
                                tup = rv.get(role_nid)
                                return tup[0] if tup and len(tup) == 1 else None
                            a1_nid = _get_role_nid(sample_rv, role_a)
                            a2_nid = _get_role_nid(sample_rv, role_b)
                            if a1_nid is not None:
                                a1_type = TypeTerm(
                                    tag=token_type(a1_nid, type_context).tag,
                                    ordinal=None,
                                )
                            if a2_nid is not None:
                                a2_type = TypeTerm(
                                    tag=token_type(a2_nid, type_context).tag,
                                    ordinal=None,
                                )
                            first_out = sample_tv[0] if sample_tv else None
                            if first_out is not None:
                                out_tag = token_type(first_out, type_context).tag
                                out_type = TypeTerm(tag=out_tag, ordinal=None)
                        role_rules.append(RelationRule(
                            output_role=target_role,
                            op_name=op_nid,
                            arg1=role_a,
                            arg2=role_b,
                            evidence=n_match,
                            total_obs=n,
                            arg1_type=a1_type,
                            arg2_type=a2_type,
                            output_type=out_type,
                        ))

        discovered.extend(role_rules)
        available_source_roles.append(target_role)

    return sorted(discovered, key=lambda r: -r.evidence)


def predict_from_relation_rules(
    seq: list[str],
    store: 'RelationStore',
    rules_by_op: dict[NodeId, list[RelationRule]],
    engine,
) -> Optional[list[str]]:
    """Predict the full output for *seq* using discovered RelationRules.

    str boundary: accepts list[str], returns list[str].
    Internally operates on NodeId.

    Returns None if the op is not known, no rules were discovered, or
    any required engine lookup fails.
    """
    if not seq:
        return None
    rel = store.extract_relation(seq)
    if rel is None:
        return None
    op = rel.op
    op_rules = rules_by_op.get(op)
    if not op_rules:
        return None

    # Build initial role_values from input roles
    _EMPTY_NODE = TOKEN_GRAPH.encode('')
    role_values: dict[NodeId, tuple[NodeId, ...]] = {}
    for sep, toks in rel.input_roles:
        if toks:
            role_values[sep] = tuple(toks)

    # Apply rules in the discovered order
    output_parts: list[tuple[NodeId, tuple[NodeId, ...]]] = []
    for rule in op_rules:
        result_dist = rule.evaluate(role_values, engine)
        if not result_dist:
            return None
        result_tup = max(result_dist, key=result_dist.get)
        role_values[rule.output_role] = result_tup
        output_parts.append((rule.output_role, result_tup))

    # Build output token list and decode to strings
    output_nids: list[NodeId] = []
    for role_nid, result_tup in output_parts:
        output_nids.append(role_nid)   # e.g. STEP_NODE, ANS_NODE, EQ_NODE
        output_nids.extend(result_tup)

    return TOKEN_GRAPH.decode_seq(output_nids)


def predict_alternatives_from_rules(
    seq: list[str],
    store: 'RelationStore',
    rules_by_op: dict[NodeId, list[RelationRule]],
    engine,
) -> list[tuple[list[str], float]]:
    """Return all consistent output alternatives with evidence weights (Phase XV).

    str boundary: accepts list[str], returns list[(list[str], float)].
    Internally operates on NodeId.
    """
    from collections import defaultdict

    if not seq:
        return []
    rel = store.extract_relation(seq)
    if rel is None:
        return []
    op = rel.op
    op_rules = rules_by_op.get(op)
    if not op_rules:
        return []

    _EMPTY_NODE = TOKEN_GRAPH.encode('')

    # Build initial role_values from input roles
    role_values_base: dict[NodeId, tuple[NodeId, ...]] = {}
    for sep, toks in rel.input_roles:
        if toks:
            role_values_base[sep] = tuple(toks)

    # Group rules by output_role, preserving the dependency order
    rules_by_role: dict[NodeId, list[RelationRule]] = defaultdict(list)
    role_order: list[NodeId] = []
    seen_roles: set[NodeId] = set()
    for rule in sorted(op_rules, key=lambda r: -r.evidence):
        if rule.output_role not in seen_roles:
            seen_roles.add(rule.output_role)
            role_order.append(rule.output_role)
        rules_by_role[rule.output_role].append(rule)

    # Walk the dependency chain, branching on each output role
    paths: list[tuple[dict[NodeId, tuple[NodeId, ...]], float, list[tuple[NodeId, tuple[NodeId, ...]]]]] = [
        (dict(role_values_base), 1.0, [])
    ]

    for role_name in role_order:
        role_rules = rules_by_role[role_name]
        new_paths = []
        for role_vals, weight, output_parts in paths:
            result_evidence: dict[tuple[NodeId, ...], float] = {}
            for rule in role_rules:
                result_dist = rule.evaluate(role_vals, engine)
                for result_tup in result_dist:
                    result_evidence[result_tup] = (
                        result_evidence.get(result_tup, 0.0) + rule.evidence
                    )
            if not result_evidence:
                continue
            total = sum(result_evidence.values())
            for result_tup, ev in result_evidence.items():
                new_rv = dict(role_vals)
                new_rv[role_name] = result_tup
                new_paths.append((
                    new_rv,
                    weight * ev / total,
                    output_parts + [(role_name, result_tup)],
                ))
        paths = new_paths
        if not paths:
            return []

    # Convert paths to (output_token_list, weight) pairs (decoded to str)
    alternatives: list[tuple[list[str], float]] = []
    for _, weight, output_parts in paths:
        output_nids: list[NodeId] = []
        for role_nid, result_tup in output_parts:
            output_nids.append(role_nid)
            output_nids.extend(result_tup)
        alternatives.append((TOKEN_GRAPH.decode_seq(output_nids), weight))

    return alternatives


def discover_kleisli_chains(
    relations: list['Relation'],
    engine,
    min_evidence: int = 2,
    mismatch_tolerance: float = 0.25,
) -> tuple[Optional[NodeId], dict[NodeId, list['RelationRule']]]:
    """Discover Kleisli chain rules for variable-depth output ops.

    Phase XXIII: returns (disc_role_NodeId, {disc_val_NodeId: rules}).
    Returns (None, {}) if no discriminator is found.
    """
    if not relations:
        return None, {}

    _EMPTY_NODE = TOKEN_GRAPH.encode('')

    # Count step tokens per relation
    step_counts = [
        sum(1 for rname, _ in rel.output_roles if rname == STEP_NODE)
        for rel in relations
    ]
    unique_depths = set(step_counts)

    if len(unique_depths) <= 1:
        return None, {}

    if not relations[0].input_roles:
        return None, {}

    input_role_names = [sep for sep, _ in relations[0].input_roles]

    # Find discriminator
    disc_role: Optional[NodeId] = None
    for role_name in input_role_names:
        val_to_depths: dict[tuple[NodeId, ...], set[int]] = {}
        for rel, n_steps in zip(relations, step_counts):
            val_tup = None
            for sep, toks in rel.input_roles:
                if sep == role_name and toks:
                    val_tup = tuple(toks)
                    break
            if val_tup is not None:
                val_to_depths.setdefault(val_tup, set()).add(n_steps)
        if val_to_depths and all(len(v) == 1 for v in val_to_depths.values()):
            disc_role = role_name
            break

    if disc_role is None:
        return None, {}

    # Group relations by discriminator value (tuple[NodeId,...])
    groups: dict[tuple[NodeId, ...], list[Relation]] = {}
    for rel in relations:
        val_tup = None
        for sep, toks in rel.input_roles:
            if sep == disc_role and toks:
                val_tup = tuple(toks)
                break
        if val_tup is not None:
            groups.setdefault(val_tup, []).append(rel)

    result: dict[NodeId, list[RelationRule]] = {}

    # Use the single-token disc value as key when possible (common case)
    _STEP_PATTERN = TOKEN_GRAPH.encode('step')

    for disc_val_tup, group_rels in groups.items():
        if len(group_rels) < min_evidence:
            continue

        # Reindex repeated 'step' output roles: STEP_NODE → step_0, step_1, ...
        def _reindex(rel: 'Relation') -> 'Relation':
            new_out: list[tuple[NodeId, list[NodeId]]] = []
            step_idx = 0
            for rname, toks in rel.output_roles:
                if rname == STEP_NODE:
                    new_out.append((TOKEN_GRAPH.encode(f'step_{step_idx}'), list(toks)))
                    step_idx += 1
                else:
                    new_out.append((rname, list(toks)))
            return Relation(op=rel.op, input_roles=rel.input_roles, output_roles=new_out)

        reindexed = [_reindex(r) for r in group_rels]
        expected_roles: set[NodeId] = {rname for rname, _ in reindexed[0].output_roles}

        rules = discover_relation_rules(
            reindexed, engine,
            min_evidence=min_evidence,
            mismatch_tolerance=mismatch_tolerance,
        )

        if not rules:
            continue

        covered = {r.output_role for r in rules}
        if not expected_roles.issubset(covered):
            continue

        role_order = [rname for rname, _ in reindexed[0].output_roles]
        ordered = sorted(
            rules,
            key=lambda r: role_order.index(r.output_role) if r.output_role in role_order else 999,
        )
        # Use single-token disc val as key for the result dict, or encode the multi-token string
        if len(disc_val_tup) == 1:
            disc_key = disc_val_tup[0]
        else:
            # For multi-token disc values, encode the decoded string
            disc_key = TOKEN_GRAPH.encode(TOKEN_GRAPH.decode_seq(list(disc_val_tup)))
        result[disc_key] = ordered

    return disc_role, result


def _merge_digit_runs(tokens: list[NodeId], nno_atoms: frozenset[NodeId]) -> list[NodeId]:
    """Merge consecutive NNO-alphabet tokens into a single compound token.

    E.g. [enc('1'), enc('2')] → [enc('12')] when both are in nno_atoms.
    Non-NNO tokens act as merge barriers.
    """
    if not nno_atoms or not tokens:
        return list(tokens)
    result: list[NodeId] = []
    run: list[NodeId] = []
    for tok in tokens:
        if tok in nno_atoms:
            run.append(tok)
        else:
            if run:
                compound_str = ''.join(TOKEN_GRAPH.decode(t) for t in run)
                result.append(TOKEN_GRAPH.encode(compound_str))
                run = []
            result.append(tok)
    if run:
        compound_str = ''.join(TOKEN_GRAPH.decode(t) for t in run)
        result.append(TOKEN_GRAPH.encode(compound_str))
    return result


def _split_compound(tok: NodeId, nno_atoms: frozenset[NodeId]) -> list[NodeId]:
    """Split a compound NNO token back to individual single-char tokens.

    Inverse of _merge_digit_runs for single tokens.
    E.g. enc('12') → [enc('1'), enc('2')] when '1' and '2' are in nno_atoms.
    Returns [tok] unchanged if single-char or not all-NNO.
    """
    tok_str = TOKEN_GRAPH.decode(tok)
    if len(tok_str) <= 1:
        return [tok]
    parts = [TOKEN_GRAPH.encode(c) for c in tok_str]
    if all(p in nno_atoms for p in parts):
        return parts
    return [tok]
