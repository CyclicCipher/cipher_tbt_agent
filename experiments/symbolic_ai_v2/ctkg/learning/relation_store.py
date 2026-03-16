"""Relational tuple store — hypergraph representation for sequence data.

Each training sequence is segmented into named roles using two kinds of
structural tokens:

  OUTPUT_DELIMS  — {eq, step, ans, <eos>} — mark the start of each output
                   phase.  Every token between two consecutive output delimiters
                   belongs to that output role.

  INPUT_SEPS     — tokens like 'x', 'at', 'dx' that appear at consistent
                   positions in the input segment and act as named separators
                   between input argument groups.  These are LEARNED from data:
                   a token qualifies as an input separator if it appears at the
                   same position in ≥80% of training sequences for that op.

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
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tokens that mark the start of an output role segment.
OUTPUT_DELIMS: frozenset[str] = frozenset({'eq', 'step', 'ans', '<eos>'})

# Candidate tokens for input separators.  A token must be in this set AND
# appear at a consistent position in the training data to become a separator.
# This set is pre-seeded with known structural keywords; it does NOT include
# digit tokens or operator names.
KNOWN_INPUT_SEPS: frozenset[str] = frozenset({'x', 'at', 'dx'})

# Fraction of training sequences that must agree on a position for it to be
# treated as a structural separator.
_SEP_THRESHOLD: float = 0.80


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    """Structured representation of one training/test sequence.

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
    """
    op: str
    input_roles: list[tuple[str, list[str]]] = field(default_factory=list)
    output_roles: list[tuple[str, list[str]]] = field(default_factory=list)

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
    ('x', 'at', 'dx').  For operators with no internal separators (e.g.
    'linsolve' with concatenated digit arguments), all input tokens are grouped
    under a single unnamed role and the discovery falls back to the existing
    arity-aware pipeline.
    """

    def __init__(self) -> None:
        # op → list of Relation objects
        self._relations: dict[str, list[Relation]] = {}
        # op → learned input schema: sorted [(position_in_input, sep_token), ...]
        self._schemas: dict[str, list[tuple[int, str]]] = {}

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
            for seq in op_seqs:
                rel = _extract_relation(seq, schema)
                if rel is not None:
                    rels.append(rel)

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _learn_input_schema(seqs: list[list[str]]) -> list[tuple[int, str]]:
    """Identify input-separator positions for an operator.

    Returns a sorted list of (position_in_input_segment, separator_token)
    pairs.  A token qualifies if it is in KNOWN_INPUT_SEPS and appears at
    the same position in ≥80% of training sequences for this op.

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

    # Count occurrences of each KNOWN_INPUT_SEP at each position
    # pos_counts[pos][tok] = count
    pos_counts: dict[int, dict[str, int]] = {}
    for seg in input_segs:
        for i, tok in enumerate(seg):
            if tok in KNOWN_INPUT_SEPS:
                pos_counts.setdefault(i, {}).setdefault(tok, 0)
                pos_counts[i][tok] += 1

    schema: list[tuple[int, str]] = []
    for pos, tok_counts in sorted(pos_counts.items()):
        for tok, cnt in tok_counts.items():
            if cnt >= threshold:
                schema.append((pos, tok))

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
    if schema:
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
    """
    output_role: str
    op_name: str
    arg1: str  # role name for first argument
    arg2: str  # role name for second argument
    evidence: int = 0

    def evaluate(
        self,
        role_values: dict[str, str],
        bfm: dict[str, dict[tuple, str]],
    ) -> Optional[str]:
        """Evaluate this rule given a dict of role_name → value_string.

        Returns the result string, or None if inputs are unavailable or
        the bfm lookup misses.
        """
        v1 = role_values.get(self.arg1)
        v2 = role_values.get(self.arg2)
        if v1 is None or v2 is None:
            return None
        return bfm.get(self.op_name, {}).get((v1, v2))


def discover_relation_rules(
    relations: list['Relation'],
    bfm: dict[str, dict[tuple, str]],
    min_evidence: int = 2,
    unknown_tolerance: float = 0.20,
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

        # Try all binary functional combinations
        best_rule: Optional[RelationRule] = None
        best_evidence = 0

        source_roles = list(available_source_roles)

        for op_name, op_map in bfm.items():
            for role_a in source_roles:
                for role_b in source_roles:
                    n_match = 0
                    n_unknown = 0
                    mismatch = False
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
                            mismatch = True
                            break
                        n_match += 1
                    n = len(examples)
                    if (not mismatch
                            and n_match >= min_evidence
                            and n_unknown / n <= unknown_tolerance):
                        if n_match > best_evidence:
                            best_evidence = n_match
                            best_rule = RelationRule(
                                output_role=target_role,
                                op_name=op_name,
                                arg1=role_a,
                                arg2=role_b,
                                evidence=n_match,
                            )

        if best_rule is not None:
            discovered.append(best_rule)

        # Make this role available as a source for subsequent rules
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
        result = rule.evaluate(role_values, bfm)
        if result is None:
            return None  # rule failed — can't complete the prediction
        role_values[rule.output_role] = result
        output_parts.append((rule.output_role, result))

    # Build output token list: [delim, tok1, tok2, ...] for each output role
    output_toks: list[str] = []
    for role_name, result_str in output_parts:
        output_toks.append(role_name)  # e.g. 'step', 'ans', 'eq'
        output_toks.extend(list(result_str))  # split compound token back to digits

    return output_toks


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
