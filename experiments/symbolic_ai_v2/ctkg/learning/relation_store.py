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

This eliminates discover_arities() and hardcoded TERMINATOR sets — the
structural tokens are identified from data, not from a hand-crafted list.

Usage in predict.py:

    store = RelationStore()
    store.update_batch(chain_rule_seqs)
    step_corpus = store.eq_corpus_for_role('step')   # → discover_rules(step_corpus)
    ans_corpus  = store.eq_corpus_for_role('ans')    # → discover_rules(ans_corpus)
    eq_corpus   = store.eq_corpus_for_role('eq')     # → discover_rules(eq_corpus)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from experiments.symbolic_ai_v2.ctkg.core.dependent_type import TypeTerm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tokens that mark the start of an output role segment.
OUTPUT_DELIMS: frozenset[str] = frozenset({'eq', 'step', 'ans', '<eos>'})

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

    This is the canonical representation for multi-input operations /
    hyperedges in the CTKG (see CTKG_ARCHITECTURE.md).  It is strictly
    more expressive than MultiMorphism (ctkg/core/operad.py), which is
    the degenerate anonymous/positional special case.

    Parameters
    ----------
    op : str
        The operator token (first token of the sequence).
    input_roles : list of (role_name, tokens)
        Ordered groups of input tokens, separated by input separator tokens.
        The role_name is the separator token that PRECEDES this group, or ''
        for the first group (nothing precedes it).
    output_roles : list of (role_name, tokens)
        Output phases, keyed by their opening delimiter token
        ('eq', 'step', 'ans').  Ordered in sequence order.
    input_type_dists : list of (role_name, type_dist), optional
        Type distributions for each input role (derived from ConceptLattice
        when available).  Parallel to input_roles; empty list = not set.
    output_type_dists : list of (role_name, type_dist), optional
        Type distributions for each output role (derived from ConceptLattice
        when available).  Parallel to output_roles; empty list = not set.
    """
    op: str
    input_roles: list[tuple[str, list[str]]] = field(default_factory=list)
    output_roles: list[tuple[str, list[str]]] = field(default_factory=list)
    # Optional type-distribution fields (Phase XI extension).
    # type_dist = dict[ConceptId, float]; kept as plain dict to avoid
    # circular imports with operad.py / concept_lattice.py.
    input_type_dists: list[tuple[str, dict]] = field(default_factory=list)
    output_type_dists: list[tuple[str, dict]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def input_role(self, name: str) -> Optional[list[str]]:
        """Return the token list for the input role named *name*, or None."""
        for rname, toks in self.input_roles:
            if rname == name:
                return toks
        return None

    def output_role(self, name: str) -> Optional[list[str]]:
        """Return the token list for the *first* output role named *name*."""
        for rname, toks in self.output_roles:
            if rname == name:
                return toks
        return None

    def all_output_roles(self, name: str) -> list[list[str]]:
        """Return all output role values for roles named *name* (e.g. multi-step)."""
        return [toks for rname, toks in self.output_roles if rname == name]

    def flat_input(self) -> list[str]:
        """Reconstruct the flat input token sequence from named roles."""
        result: list[str] = []
        for sep, toks in self.input_roles:
            if sep:
                result.append(sep)
            result.extend(toks)
        return result


# ---------------------------------------------------------------------------
# RelationStore
# ---------------------------------------------------------------------------

class RelationStore:
    """Learns operator schemas from training sequences and stores Relations.

    The store is built in two passes:
      1. ``update_batch(seqs)`` collects all sequences and learns which tokens
         act as input separators for each operator.
      2. ``eq_corpus_for_role(role)`` builds an eq-format corpus for a
         specific output role (e.g. 'step', 'ans', 'eq'), suitable for
         passing to ``discover_rules()``.

    The store handles operators whose input sections contain natural separators
    (e.g. 'x', 'at', 'dx' for symbolic math ops — discovered from data, not
    hard-coded).  For operators with no internal separators (e.g. 'linsolve'
    with concatenated digit arguments), all input tokens are grouped under a
    single unnamed role and the discovery falls back to the existing
    arity-aware pipeline.
    """

    def __init__(self) -> None:
        # op → list of Relation objects
        self._relations: dict[str, list[Relation]] = {}
        # op → learned input schema: sorted [(position_in_input, sep_token), ...]
        # Positional schemas use role names 'p0','p1',... as the sep_token field
        self._schemas: dict[str, list[tuple[int, str]]] = {}
        # op → frozenset of all observed output role names ('step','ans','eq',...)
        self._output_role_names: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def update_batch(self, seqs: list[list[str]]) -> None:
        """Add a batch of training sequences and learn operator schemas."""
        # Group sequences by op
        raw: dict[str, list[list[str]]] = {}
        for seq in seqs:
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
            out_roles: set[str] = set()
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
        """Return all stored Relations for *op*."""
        return self._relations.get(op, [])

    def get_schema(self, op: str) -> list[tuple[int, str]]:
        """Return the learned input schema for *op*."""
        return self._schemas.get(op, [])

    def has_input_seps(self, op: str) -> bool:
        """Return True if *op* has learned input separators."""
        return bool(self._schemas.get(op))

    def all_output_role_names(self, op: str) -> frozenset[str]:
        """Return all output role names observed for *op* in training data."""
        return self._output_role_names.get(op, frozenset())

    def extract_relation(self, seq: list[str]) -> Optional[Relation]:
        """Extract a Relation from a new (possibly OOD) sequence."""
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
        role: str,
        ops: Optional[frozenset[str]] = None,
        merge_digits: bool = False,
        nno_atoms: frozenset[str] = frozenset(),
    ) -> list[list[str]]:
        """Build an eq-format corpus targeting *role* across all ops.

        For each relation that has an output role named *role*, emit:
            [op] + flat_input + ['eq'] + role_tokens

        Parameters
        ----------
        role : str
            Output role to target ('step', 'ans', 'eq', 'step_0', ...).
        ops : frozenset, optional
            Restrict to these ops (default: all ops).
        merge_digits : bool
            If True, merge consecutive NNO-alphabet tokens in the output
            (e.g. ['1','2'] → ['12'] representing the two-digit number 12).
        nno_atoms : frozenset
            NNO alphabet used for digit merging (required when merge_digits=True).

        Returns
        -------
        list of token sequences, each ending with the role's tokens.
        """
        result: list[list[str]] = []
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
                result.append([op] + input_toks + ['eq'] + output_toks)
        return result

    def ops_with_role(self, role: str) -> frozenset[str]:
        """Return the set of ops that have at least one relation with *role*."""
        result: set[str] = set()
        for op, rels in self._relations.items():
            for rel in rels:
                if rel.output_role(role) is not None:
                    result.add(op)
                    break
        return frozenset(result)

    def ops_with_step(self) -> frozenset[str]:
        """Shorthand: ops with a 'step' output role."""
        return self.ops_with_role('step')

    def ops_with_input_seps(self) -> frozenset[str]:
        """Return ops that have at least one learned input separator.

        Only these ops have unambiguous input segmentation and are eligible
        for the per-role rule discovery pipeline.
        """
        return frozenset(op for op, schema in self._schemas.items() if schema)

    def ops_with_schema(self) -> frozenset[str]:
        """Return ops that have any learned schema (separator-based or positional).

        Includes:
        - Ops with named input separators (e.g. 'eval' with 'x','at')
        - Ops with positional schemas (fixed-length separator-free inputs)
        """
        return frozenset(op for op, schema in self._schemas.items() if schema)

    def ops_with_positional_schema(self) -> frozenset[str]:
        """Return ops that have a positional schema (no separators, fixed length).

        Positional role names match the pattern p0, p1, ... as assigned by
        _learn_input_schema when no separator tokens are found.
        """
        result: set[str] = set()
        for op, schema in self._schemas.items():
            if schema and schema[0][1].startswith('p') and schema[0][1][1:].isdigit():
                result.add(op)
        return frozenset(result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _learn_input_schema(seqs: list[list[str]]) -> list[tuple[int, str]]:
    """Identify input-separator positions for an operator.

    Returns a sorted list of (position_in_input_segment, separator_token)
    pairs.  A token qualifies if it appears at the same position in ≥80% of
    training sequences for this op.  Any token can become a separator — no
    pre-seeded keyword list (Iron Rule compliance).

    The *position_in_input_segment* counts tokens WITHIN the input segment
    (i.e. after the op token and before the first OUTPUT_DELIM).
    """
    if not seqs:
        return []

    # Extract the input segment (everything between op and first OUTPUT_DELIM)
    input_segs: list[list[str]] = []
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

    # Count occurrences of EVERY token at each position.
    # Any token that appears at the same position in ≥80% of sequences
    # is a structural separator — discovered purely from data.
    # pos_counts[pos][tok] = count
    pos_counts: dict[int, dict[str, int]] = {}
    for seg in input_segs:
        for i, tok in enumerate(seg):
            pos_counts.setdefault(i, {}).setdefault(tok, 0)
            pos_counts[i][tok] += 1

    schema: list[tuple[int, str]] = []
    for pos, tok_counts in sorted(pos_counts.items()):
        for tok, cnt in tok_counts.items():
            if cnt >= threshold:
                schema.append((pos, tok))

    if schema:
        return schema

    # No separator tokens found.  Check if ALL input segments have the same
    # length — if so, assign positional role names 'p0','p1',...,'p{L-1}'.
    # Positional schemas are detected in _extract_relation by checking whether
    # the first schema entry's role name starts with 'p' and is a digit string.
    lengths = {len(seg) for seg in input_segs}
    if len(lengths) == 1:
        L = next(iter(lengths))
        if L > 0:
            return [(i, f'p{i}') for i in range(L)]

    return schema


def _extract_relation(seq: list[str], schema: list[tuple[int, str]]) -> Optional[Relation]:
    """Extract a Relation from *seq* using the given input *schema*.

    Parameters
    ----------
    seq : list[str]
        Raw token sequence (starts with op).
    schema : list of (position, sep_token)
        Input separator schema learned from training data.
    """
    if not seq:
        return None
    op = seq[0]
    body = seq[1:]

    # ---- Split body into input segment and output segment ----
    input_end = len(body)
    first_output_delim = None
    for i, tok in enumerate(body):
        if tok in OUTPUT_DELIMS:
            input_end = i
            first_output_delim = tok
            break

    input_tokens = body[:input_end]

    # ---- Parse input_tokens into roles using schema ----
    input_roles: list[tuple[str, list[str]]] = []
    _is_positional = bool(
        schema
        and schema[0][1].startswith('p')
        and schema[0][1][1:].isdigit()
    )
    if _is_positional:
        # Positional schema: each schema entry (pos, 'p{pos}') assigns the
        # token at that position to its own role.  Tokens beyond the schema
        # length (e.g. OOD sequences with extra digits) are dropped so that
        # rule evaluation stays within the discovered positional structure.
        for pos, role_name in schema:
            if pos < len(input_tokens):
                input_roles.append((role_name, [input_tokens[pos]]))
    elif schema:
        sorted_schema = sorted(schema, key=lambda x: x[0])
        prev_end = 0
        prev_sep = ''
        for pos, sep_tok in sorted_schema:
            if pos >= len(input_tokens):
                break
            group = input_tokens[prev_end:pos]
            input_roles.append((prev_sep, group))
            prev_sep = sep_tok
            prev_end = pos + 1  # skip the separator itself
        # Last group after the final separator
        input_roles.append((prev_sep, input_tokens[prev_end:]))
    else:
        # No separators: single group
        input_roles = [('', input_tokens)]

    # ---- Parse output into roles ----
    output_roles: list[tuple[str, list[str]]] = []
    if first_output_delim is not None:
        output_body = body[input_end:]  # starts with the first delimiter
        current_delim: Optional[str] = None
        current_group: list[str] = []
        for tok in output_body:
            if tok in OUTPUT_DELIMS:
                if current_delim is not None and current_delim != '<eos>':
                    output_roles.append((current_delim, current_group))
                current_delim = tok
                current_group = []
            else:
                if current_delim is not None:
                    current_group.append(tok)
        # Flush last group (unless it's '<eos>')
        if current_delim is not None and current_delim != '<eos>':
            output_roles.append((current_delim, current_group))

    return Relation(op=op, input_roles=input_roles, output_roles=output_roles)


# ---------------------------------------------------------------------------
# Arity-free rule discovery (hypergraph approach)
# ---------------------------------------------------------------------------

@dataclass
class RelationRule:
    """A rule for one output role, discovered from relational tuples.

    Represents: output_role = bfm_op(arg1_val, arg2_val)

    where arg1 and arg2 are role names:
      - input role names: '' (first unnamed input group), 'x', 'at', 'dx', ...
      - output role names: 'step', 'ans', 'eq'

    No tree parsing, no arities, no prefix notation.
    Just a binary functional map lookup on named role values.

    Phase XXI — dependent type annotations (optional):
        arg1_type, arg2_type, output_type carry the type of each role's value
        as discovered from the NNO chain.  ordinal=None in these fields means
        the rule is universally quantified over all NNO ordinals (the rule
        applies for *any* digit, not just a specific one).

    Phase XXII — probability monad:
        total_obs is the denominator for confidence: the number of training
        examples seen for this (op, output_role) regardless of match/mismatch.
        confidence = evidence / total_obs  ∈ (0, 1].
        evaluate() returns a Kleisli morphism A → Dist(B) encoded as
        dict[str, float] instead of Optional[str].
    """
    output_role: str
    op_name: str
    arg1: str  # role name for first argument
    arg2: str  # role name for second argument
    evidence: int = 0
    # Phase XXII
    total_obs: int = 0
    # Phase XXI — type annotations (None = not yet inferred)
    arg1_type: Optional['TypeTerm'] = None   # type: ignore[type-arg]
    arg2_type: Optional['TypeTerm'] = None
    output_type: Optional['TypeTerm'] = None

    @property
    def confidence(self) -> float:
        """Empirical probability that this rule correctly predicts the output.

        confidence = evidence / total_obs.
        Defaults to 1.0 when total_obs is unset (legacy rules or deterministic
        ops where every observed example matched).
        """
        if self.total_obs <= 0:
            return 1.0
        return self.evidence / self.total_obs

    def evaluate(
        self,
        role_values: dict[str, str],
        bfm: dict[str, dict[tuple, str]],
    ) -> 'dict[str, float]':
        """Evaluate this rule as a Kleisli morphism A → Dist(B).

        Phase XXII: returns a distribution dict[result_str → probability]
        rather than Optional[str].  For a deterministic BFM lookup the
        distribution has at most one entry with probability == self.confidence.

        Returns {} (empty dict) if inputs are unavailable or the bfm lookup
        misses — analogous to the previous None return.
        """
        v1 = role_values.get(self.arg1)
        v2 = role_values.get(self.arg2)
        if v1 is None or v2 is None:
            return {}
        result = bfm.get(self.op_name, {}).get((v1, v2))
        if result is None:
            return {}
        return {result: self.confidence}


def discover_relation_rules(
    relations: list['Relation'],
    bfm: dict[str, dict[tuple, str]],
    min_evidence: int = 2,
    unknown_tolerance: float = 0.20,
    mismatch_tolerance: float = 0.0,
    type_context: Optional[dict[str, 'TypeTerm']] = None,
) -> list[RelationRule]:
    """Discover RelationRules from a list of Relations.

    For each output role in the relations, attempts to find a binary function
    f such that output_role_value = bfm[f][(role_i_val, role_j_val)] for all
    training relations.

    This is COMPLETELY ARITY-FREE: it operates directly on role value strings,
    with no tree parsing, no prefix notation, no discover_arities().

    Multi-digit role values are treated as single strings (e.g. '10' from
    mul('2','5')='10' in the BFM).

    Parameters
    ----------
    relations : list of Relation
        Training relations for a SINGLE operator (mix of different ops not supported).
    bfm : dict
        Binary functional maps {op_name: {(a, b): result_str}}.
    min_evidence : int
        Minimum number of matching examples for a rule to be accepted.
    unknown_tolerance : float
        Maximum fraction of examples that can be 'unknown' (missing from bfm).
    mismatch_tolerance : float
        Maximum fraction of examples that can be 'mismatch' (bfm returns a
        different value than the target).  Default 0.0 = strict (any mismatch
        rejects the rule).  Use a small positive value (e.g. 0.25) for ops
        where the training corpus intentionally mixes different sub-cases
        (e.g. sq(n) for n≥10 produces 3-digit answers not in the BFM).
    type_context : dict[str, TypeTerm], optional
        Phase XXI: mapping from token string to TypeTerm.  When provided,
        each discovered rule is annotated with the type tags of its arg1,
        arg2, and output values (universally quantified — ordinal=None).

    Returns
    -------
    list of RelationRule, sorted by evidence descending.
    """
    if not relations:
        return []

    op = relations[0].op

    # Collect all role names: input roles + previously computed output roles
    all_input_role_names: list[str] = []
    # We use the first relation to determine input role names
    for sep, toks in relations[0].input_roles:
        role_name = sep if sep else ''
        all_input_role_names.append(role_name)

    # Collect all output role names (in order of typical appearance)
    # Use ordering: step before ans, ans before eq
    all_output_roles: list[str] = []
    seen_out: set[str] = set()
    for rel in relations:
        for rname, _ in rel.output_roles:
            if rname not in seen_out:
                seen_out.add(rname)
                all_output_roles.append(rname)

    discovered: list[RelationRule] = []

    # The available "source" roles for rule inputs: all input roles +
    # all output roles computed BEFORE the current target role.
    available_source_roles = list(all_input_role_names)

    for target_role in all_output_roles:
        # Collect (role_values, target_value) for each relation
        examples: list[tuple[dict[str, str], str]] = []
        for rel in relations:
            role_vals: dict[str, str] = {}
            # Input roles: use role name as key, join tokens
            for sep, toks in rel.input_roles:
                rname = sep if sep else ''
                if toks:
                    role_vals[rname] = ''.join(toks)  # e.g. '2' or '12'
            # Output roles (for use as source inputs to later rules)
            for rname, toks in rel.output_roles:
                if toks:
                    role_vals[rname] = ''.join(toks)

            target_val_list = role_vals.get(target_role)
            if target_val_list is None:
                continue
            examples.append((role_vals, target_val_list))

        if not examples:
            available_source_roles.append(target_role)
            continue

        # Try all binary functional combinations.
        # Phase XV (Coproducts): collect ALL qualifying rules, not just the best.
        # Multiple rules for the same output_role represent a coproduct A ⊔ B:
        # the output type is the disjoint union of all supported interpretations.
        # For deterministic ops (arithmetic) this degenerates to a single rule;
        # for genuinely ambiguous ops (NLP) multiple rules carry different weights.
        role_rules: list[RelationRule] = []

        source_roles = list(available_source_roles)

        for op_name, op_map in bfm.items():
            for role_a in source_roles:
                for role_b in source_roles:
                    n_match = 0
                    n_unknown = 0
                    n_mismatch = 0
                    for role_vals, target_val in examples:
                        va = role_vals.get(role_a)
                        vb = role_vals.get(role_b)
                        if va is None or vb is None:
                            n_unknown += 1
                            continue
                        result = op_map.get((va, vb))
                        if result is None:
                            n_unknown += 1
                            continue
                        if result != target_val:
                            n_mismatch += 1
                        else:
                            n_match += 1
                    n = len(examples)
                    if (n_match >= min_evidence
                            and n_unknown / n <= unknown_tolerance
                            and n_mismatch / n <= mismatch_tolerance):
                        # Phase XXI: infer types from a sample example
                        a1_type = a2_type = out_type = None
                        if type_context and examples:
                            sample_rv, sample_tv = examples[0]
                            from experiments.symbolic_ai_v2.ctkg.core.dependent_type import (
                                TypeTerm, rule_type_tag, token_type,
                            )
                            a1_type = rule_type_tag(sample_rv, role_a, type_context)
                            a2_type = rule_type_tag(sample_rv, role_b, type_context)
                            out_type = TypeTerm(
                                tag=token_type(sample_tv, type_context).tag,
                                ordinal=None,  # universally quantified
                            )
                        role_rules.append(RelationRule(
                            output_role=target_role,
                            op_name=op_name,
                            arg1=role_a,
                            arg2=role_b,
                            evidence=n_match,
                            total_obs=n,       # Phase XXII
                            arg1_type=a1_type,  # Phase XXI
                            arg2_type=a2_type,
                            output_type=out_type,
                        ))

        discovered.extend(role_rules)

        # Make this role available as a source for subsequent rules.
        # Use the highest-evidence rule's result for deterministic chaining.
        available_source_roles.append(target_role)

    return sorted(discovered, key=lambda r: -r.evidence)


def predict_from_relation_rules(
    seq: list[str],
    store: 'RelationStore',
    rules_by_op: dict[str, list[RelationRule]],
    bfm: dict[str, dict[tuple, str]],
) -> Optional[list[str]]:
    """Predict the full output for *seq* using discovered RelationRules.

    Extracts input roles from *seq*, applies rules in dependency order
    (step before ans), and returns the complete expected output token list
    (including delimiter tokens like 'step', 'ans').

    Returns None if:
    - The op is not known to the store.
    - No rules were discovered for this op.
    - Any required bfm lookup fails.
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
    role_values: dict[str, str] = {}
    for sep, toks in rel.input_roles:
        rname = sep if sep else ''
        if toks:
            role_values[rname] = ''.join(toks)

    # Apply rules in the discovered order
    output_parts: list[tuple[str, str]] = []  # [(role_name, result_str)]
    for rule in op_rules:
        result_dist = rule.evaluate(role_values, bfm)
        if not result_dist:
            return None  # rule failed — can't complete the prediction
        result = max(result_dist, key=result_dist.get)
        role_values[rule.output_role] = result
        output_parts.append((rule.output_role, result))

    # Build output token list: [delim, tok1, tok2, ...] for each output role
    output_toks: list[str] = []
    for role_name, result_str in output_parts:
        output_toks.append(role_name)  # e.g. 'step', 'ans', 'eq'
        output_toks.extend(list(result_str))  # split compound token back to digits

    return output_toks


def predict_alternatives_from_rules(
    seq: list[str],
    store: 'RelationStore',
    rules_by_op: dict[str, list[RelationRule]],
    bfm: dict[str, dict[tuple, str]],
) -> list[tuple[list[str], float]]:
    """Return all consistent output alternatives with evidence weights (Phase XV).

    This is the coproduct-aware prediction function.  Where
    ``predict_from_relation_rules`` commits to one output (the best rule),
    this function returns ALL outputs that can be produced by ANY combination
    of qualifying rules, weighted by cumulative rule evidence.

    The result is a list of ``(output_token_list, weight)`` pairs:

    - **One element** when all applicable rules for every output role agree on
      the same result (the deterministic case — arithmetic, fully-determined ops).
      The coproduct degenerates to a point; weight = 1.0.

    - **Multiple elements** when competing rules for some output role give
      different results (the genuinely ambiguous case — NLP lexical ambiguity,
      underspecified contexts).  Weights are proportional to evidence and sum
      to 1.0 over the full set of alternatives.

    Categorical structure
    --------------------
    Each output role's set of competing results is an object in the coproduct
    A₁ ⊔ A₂ ⊔ ... ⊔ Aₖ of alternative interpretations.  The evidence weights
    are the Markov kernel (CT_REFERENCE §15) from the input to the coproduct:
    a morphism in Stoch rather than Set.  ``predict_next`` uses these weights
    to produce a soft prediction — a Kleisli morphism in the probability monad
    rather than a deterministic function.

    Returns an empty list if no rules fire.
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

    # Build initial role_values from input roles
    role_values_base: dict[str, str] = {}
    for sep, toks in rel.input_roles:
        rname = sep if sep else ''
        if toks:
            role_values_base[rname] = ''.join(toks)

    # Group rules by output_role, preserving the dependency order (step → ans).
    rules_by_role: dict[str, list[RelationRule]] = defaultdict(list)
    role_order: list[str] = []
    seen_roles: set[str] = set()
    for rule in sorted(op_rules, key=lambda r: -r.evidence):
        if rule.output_role not in seen_roles:
            seen_roles.add(rule.output_role)
            role_order.append(rule.output_role)
        rules_by_role[rule.output_role].append(rule)

    # Walk the dependency chain, branching on each output role that has
    # competing results.  Each "path" is (role_values, cumulative_weight,
    # output_parts).  Branching is the coproduct injection; merging
    # (same result from multiple rules) is the copairing.
    paths: list[tuple[dict[str, str], float, list[tuple[str, str]]]] = [
        (dict(role_values_base), 1.0, [])
    ]

    for role_name in role_order:
        role_rules = rules_by_role[role_name]
        new_paths: list[tuple[dict[str, str], float, list[tuple[str, str]]]] = []
        for role_vals, weight, output_parts in paths:
            # Collect all results this role can produce, summing evidence
            # when multiple rules agree (copairing: f+g when f=g on a branch).
            result_evidence: dict[str, float] = {}
            for rule in role_rules:
                result_dist = rule.evaluate(role_vals, bfm)
                for result_str in result_dist:
                    result_evidence[result_str] = (
                        result_evidence.get(result_str, 0.0) + rule.evidence
                    )
            if not result_evidence:
                continue   # no rule fires — this path is dead
            total = sum(result_evidence.values())
            for result_str, ev in result_evidence.items():
                new_rv = dict(role_vals)
                new_rv[role_name] = result_str
                new_paths.append((
                    new_rv,
                    weight * ev / total,
                    output_parts + [(role_name, result_str)],
                ))
        paths = new_paths
        if not paths:
            return []

    # Convert paths to (output_token_list, weight) pairs.
    alternatives: list[tuple[list[str], float]] = []
    for _, weight, output_parts in paths:
        output_toks: list[str] = []
        for role_name, result_str in output_parts:
            output_toks.append(role_name)
            output_toks.extend(list(result_str))
        alternatives.append((output_toks, weight))

    return alternatives


def discover_kleisli_chains(
    relations: list['Relation'],
    bfm: dict[str, dict[tuple, str]],
    min_evidence: int = 2,
    mismatch_tolerance: float = 0.25,
) -> tuple[Optional[str], dict[str, list['RelationRule']]]:
    """Discover Kleisli chain rules for variable-depth output ops.

    For ops where different inputs produce different numbers of 'step' tokens
    (e.g. pow(base, exp) has exp-1 intermediate steps), groups relations by
    the value of a discriminator role — the input role whose value uniquely
    determines the step count.  For each group, reindexes repeated 'step'
    output roles as 'step_0', 'step_1', ..., then discovers RelationRules
    per depth group.

    Returns
    -------
    (disc_role_name, {disc_val: ordered_rules})
    Returns (None, {}) if the op has fixed depth or no discriminator is found.
    """
    if not relations:
        return None, {}

    # Count step tokens per relation
    step_counts = [
        sum(1 for rname, _ in rel.output_roles if rname == 'step')
        for rel in relations
    ]
    unique_depths = set(step_counts)

    # Fixed depth — standard discover_relation_rules suffices
    if len(unique_depths) <= 1:
        return None, {}

    if not relations[0].input_roles:
        return None, {}

    input_role_names = [sep if sep else '' for sep, _ in relations[0].input_roles]

    # Find discriminator: the input role whose value uniquely determines step count
    disc_role: Optional[str] = None
    for role_name in input_role_names:
        val_to_depths: dict[str, set[int]] = {}
        for rel, n_steps in zip(relations, step_counts):
            val = None
            for sep, toks in rel.input_roles:
                rn = sep if sep else ''
                if rn == role_name and toks:
                    val = ''.join(toks)
                    break
            if val is not None:
                val_to_depths.setdefault(val, set()).add(n_steps)
        if val_to_depths and all(len(v) == 1 for v in val_to_depths.values()):
            disc_role = role_name
            break

    if disc_role is None:
        return None, {}

    # Group relations by discriminator value
    groups: dict[str, list[Relation]] = {}
    for rel in relations:
        val = None
        for sep, toks in rel.input_roles:
            rn = sep if sep else ''
            if rn == disc_role and toks:
                val = ''.join(toks)
                break
        if val is not None:
            groups.setdefault(val, []).append(rel)

    result: dict[str, list[RelationRule]] = {}

    for disc_val, group_rels in groups.items():
        if len(group_rels) < min_evidence:
            continue

        # Reindex repeated 'step' output roles: step → step_0, step_1, ...
        def _reindex(rel: 'Relation') -> 'Relation':
            new_out: list[tuple[str, list[str]]] = []
            step_idx = 0
            for rname, toks in rel.output_roles:
                if rname == 'step':
                    new_out.append((f'step_{step_idx}', list(toks)))
                    step_idx += 1
                else:
                    new_out.append((rname, list(toks)))
            return Relation(op=rel.op, input_roles=rel.input_roles, output_roles=new_out)

        reindexed = [_reindex(r) for r in group_rels]

        # Expected output roles for this group (from the first relation)
        expected_roles: set[str] = {rname for rname, _ in reindexed[0].output_roles}

        rules = discover_relation_rules(
            reindexed, bfm,
            min_evidence=min_evidence,
            mismatch_tolerance=mismatch_tolerance,
        )

        if not rules:
            continue

        covered = {r.output_role for r in rules}
        if not expected_roles.issubset(covered):
            continue  # incomplete chain — would produce wrong output structure

        # Sort rules in dependency order (step_0, step_1, ..., ans)
        role_order = [rname for rname, _ in reindexed[0].output_roles]
        ordered = sorted(
            rules,
            key=lambda r: role_order.index(r.output_role) if r.output_role in role_order else 999,
        )
        result[disc_val] = ordered

    return disc_role, result


def _merge_digit_runs(tokens: list[str], nno_atoms: frozenset[str]) -> list[str]:
    """Merge consecutive NNO-alphabet tokens into a single compound token.

    E.g. ['1', '2'] → ['12'] when both '1' and '2' are in nno_atoms.
    Non-NNO tokens (like 'mul', 'x') are kept as-is and act as merge barriers.
    """
    if not nno_atoms or not tokens:
        return list(tokens)
    result: list[str] = []
    run: list[str] = []
    for tok in tokens:
        if tok in nno_atoms:
            run.append(tok)
        else:
            if run:
                result.append(''.join(run))
                run = []
            result.append(tok)
    if run:
        result.append(''.join(run))
    return result


def _split_compound(tok: str, nno_atoms: frozenset[str]) -> list[str]:
    """Split a compound NNO token back to individual single-char tokens.

    Inverse of _merge_digit_runs for single tokens.
    E.g. '12' → ['1', '2'] when '1' and '2' are in nno_atoms.
    Returns [tok] unchanged if single-char or not all-NNO.
    """
    if len(tok) <= 1:
        return [tok]
    if all(c in nno_atoms for c in tok):
        return list(tok)
    return [tok]
