"""
Phase 4: GraphZip graph grammar induction -> compression trees.

For path-graph inputs (text, digit sequences), GraphZip degenerates to
SEQUITUR (Nevill-Manning & Witten 1997).  This module implements a batch
SEQUITUR variant on typed concept-ID sequences and provides utilities to
map the resulting grammar to CTKG morphism candidates.

SEQUITUR invariants (enforced after each pass):
1. Digram uniqueness: no ordered pair of symbols appears more than once
   across all rule bodies simultaneously.
2. Rule utility: every grammar rule (except S) is used at least twice.

The batch implementation (vs. the online version in the original paper)
gives an equivalent grammar for our use case.  Rules are created by
repeatedly replacing the most frequent digram until no digram appears more
than once.  Utility is enforced after each replacement pass.

Description length (MDL):
    DL = bits(grammar) + bits(corpus | grammar)
    bits(grammar)  = Σ_rule (|body| * log2(|alphabet|))
    bits(corpus|g) = start_rule_length * log2(|alphabet|)

Type assignment (prerequisite for SEQUITUR):
    assign_types(sequence, hankel, lattice, r) maps each atom at position i
    to a concept ID by matching the neighbourhood hash against extent_weights.
    Fallback: highest dot-product between atom one-hot and concept centroid.

See CTKG_ARCHITECTURE.md §Phase 4 for the full specification.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import ConceptLattice, ConceptId


# ---------------------------------------------------------------------------
# Type assignment
# ---------------------------------------------------------------------------

def assign_types(
    sequence: Sequence[str],
    hankel: HankelCount,
    lattice: ConceptLattice,
    r: int = 1,
) -> list[ConceptId]:
    """Map each atom in `sequence` to a concept ID from `lattice`.

    Algorithm (per position i):
    1. Compute neighbourhood hash h = HankelCount._neighbourhood_key(seq, i, r).
    2. For each concept C in lattice: score = C.extent_weights.get(h, 0.0).
    3. Assign type(i) = concept with highest score.
    4. Fallback (h not in any extent): assign to concept with highest
       centroid_vector[atom_one_hot] — i.e. concept that assigns highest
       probability to this atom.
    """
    seq = list(sequence)
    atoms = lattice.atoms
    atom_idx: dict[str, int] = {a: j for j, a in enumerate(atoms)}
    concepts = lattice.concepts

    types: list[ConceptId] = []
    for i, atom in enumerate(seq):
        h = HankelCount._neighbourhood_key(seq, i, r)
        best_cid: ConceptId = -1
        best_score: float = -1.0
        for c in concepts:
            score = c.extent_weights.get(h, 0.0)
            if score > best_score:
                best_score = score
                best_cid = c.concept_id

        if best_score > 0.0:
            types.append(best_cid)
            continue

        # Fallback: highest centroid probability for this atom
        j = atom_idx.get(atom, -1)
        if j >= 0 and concepts:
            best_cid_fb = max(concepts, key=lambda c: float(c.centroid_vector[j]))
            types.append(best_cid_fb.concept_id)
        elif concepts:
            types.append(max(concepts, key=lambda c: c.support).concept_id)
        else:
            types.append(0)

    return types


# ---------------------------------------------------------------------------
# Symbol encoding
# ---------------------------------------------------------------------------
# Symbols in rule bodies use a sign convention:
#   sym >= 0  →  non-terminal (rule_id)
#   sym < 0   →  terminal, concept_id = -(sym + 1)

def _t(concept_id: int) -> int:
    """Encode a terminal concept_id as a negative symbol."""
    assert concept_id >= 0
    return -(concept_id + 1)


def _nt(rule_id: int) -> int:
    """Encode a non-terminal rule_id as a non-negative symbol."""
    assert rule_id >= 0
    return rule_id


def _is_terminal(sym: int) -> bool:
    return sym < 0


def _is_nonterminal(sym: int) -> bool:
    return sym >= 0


def _terminal_id(sym: int) -> ConceptId:
    assert sym < 0
    return -(sym + 1)


def _nonterminal_id(sym: int) -> int:
    assert sym >= 0
    return sym


# ---------------------------------------------------------------------------
# Grammar data structures
# ---------------------------------------------------------------------------

@dataclass
class GrammarRule:
    """One rule in the SEQUITUR grammar.

    rule_id >= 0.
    body: encoded symbols (terminal or non-terminal).
    use_count: number of references in other rules' bodies (0 for start rule S).
    """

    rule_id: int
    body: list[int] = field(default_factory=list)
    use_count: int = 0

    def __repr__(self) -> str:
        syms = []
        for s in self.body:
            syms.append(f"t{_terminal_id(s)}" if _is_terminal(s) else f"R{s}")
        return f"Rule({self.rule_id}: {' '.join(syms)})"


@dataclass
class Grammar:
    """A context-free grammar produced by SEQUITUR."""

    rules: dict[int, GrammarRule] = field(default_factory=dict)
    start: int = 0

    def terminals_in_rule(self, rule_id: int) -> list[ConceptId]:
        """Recursively expand rule to its terminal concept IDs."""
        rule = self.rules[rule_id]
        result: list[ConceptId] = []
        for sym in rule.body:
            if _is_terminal(sym):
                result.append(_terminal_id(sym))
            else:
                result.extend(self.terminals_in_rule(_nonterminal_id(sym)))
        return result

    def encode_length(self, n_terminals: int) -> float:
        """Approximate MDL description length in bits."""
        alphabet = n_terminals + len(self.rules)
        if alphabet <= 1:
            return 0.0
        bps = math.log2(alphabet)
        return sum(len(r.body) * bps for r in self.rules.values())

    def n_rules(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return f"Grammar(rules={len(self.rules)}, start={self.start})"


# ---------------------------------------------------------------------------
# Batch SEQUITUR
# ---------------------------------------------------------------------------

def _digram_counts(body: list[int]) -> Counter:
    """Count all adjacent digrams in body."""
    c: Counter = Counter()
    for i in range(len(body) - 1):
        c[(body[i], body[i + 1])] += 1
    return c


def _replace_in_body(body: list[int], a: int, b: int, replacement: int) -> list[int]:
    """Replace all non-overlapping occurrences of [a, b] in body with [replacement]."""
    result: list[int] = []
    i = 0
    while i < len(body):
        if i + 1 < len(body) and body[i] == a and body[i + 1] == b:
            result.append(replacement)
            i += 2
        else:
            result.append(body[i])
            i += 1
    return result


def sequitur(typed_sequence: list[int]) -> Grammar:
    """Run batch SEQUITUR on a single pre-typed sequence of concept IDs.

    Parameters
    ----------
    typed_sequence:
        List of terminal concept IDs (non-negative integers).

    Returns
    -------
    Grammar with start rule encoding the compressed sequence.
    """
    # Start: one rule S → [_t(c) for c in typed_sequence]
    start_body = [_t(c) for c in typed_sequence]
    rules: dict[int, GrammarRule] = {0: GrammarRule(rule_id=0, body=start_body)}
    next_rid = 1

    MAX_ITERATIONS = 200  # guard against infinite loops

    for _ in range(MAX_ITERATIONS):
        # Collect all digram counts across all rule bodies
        all_digrams: Counter = Counter()
        for r in rules.values():
            all_digrams.update(_digram_counts(r.body))

        # Find most frequent digram (must appear >= 2 times)
        if not all_digrams:
            break
        best_digram, best_count = all_digrams.most_common(1)[0]
        if best_count < 2:
            break

        a, b = best_digram

        # Check if an existing rule already has exactly body=[a, b]
        existing_rid: int | None = None
        for rid, rule in rules.items():
            if rid == 0:
                continue
            if rule.body == [a, b]:
                existing_rid = rid
                break

        if existing_rid is not None:
            new_sym = _nt(existing_rid)
        else:
            # Create new rule
            new_rule = GrammarRule(rule_id=next_rid, body=[a, b], use_count=0)
            rules[next_rid] = new_rule
            new_sym = _nt(next_rid)
            next_rid += 1

        # Replace all occurrences of [a, b] with new_sym across all rules
        # EXCEPT in the newly-created rule itself (its body IS [a, b] by definition;
        # replacing inside it would produce a self-referential rule R → [R]).
        new_rid_for_this = _nonterminal_id(new_sym) if _is_nonterminal(new_sym) else -1
        for rid in list(rules.keys()):
            if rid == new_rid_for_this:
                continue  # never modify the rule we just created
            old_body = rules[rid].body
            new_body = _replace_in_body(old_body, a, b, new_sym)
            rules[rid].body = new_body

        # Update use_counts from scratch (cheapest correct approach)
        for rid in rules:
            rules[rid].use_count = 0
        for rid, rule in rules.items():
            for sym in rule.body:
                if _is_nonterminal(sym) and sym != 0 and sym in rules:
                    rules[sym].use_count += 1

        # Enforce Rule Utility: inline rules used exactly once
        changed = True
        while changed:
            changed = False
            for rid in list(rules.keys()):
                if rid == 0:
                    continue
                rule = rules.get(rid)
                if rule is None or rule.use_count != 1:
                    continue
                # Inline this rule everywhere
                nt_sym = _nt(rid)
                for other_rid, other_rule in rules.items():
                    if nt_sym in other_rule.body:
                        new_body = []
                        for sym in other_rule.body:
                            if sym == nt_sym:
                                new_body.extend(rule.body)
                            else:
                                new_body.append(sym)
                        other_rule.body = new_body
                del rules[rid]
                # Recompute use_counts after inlining
                for r in rules.values():
                    r.use_count = 0
                for r in rules.values():
                    for sym in r.body:
                        if _is_nonterminal(sym) and sym != 0 and sym in rules:
                            rules[sym].use_count += 1
                changed = True
                break  # restart the while loop

    return Grammar(rules=rules, start=0)


def corpus_grammar(typed_corpus: list[list[int]]) -> Grammar:
    """Build a shared SEQUITUR grammar over a collection of typed sequences.

    All sequences are concatenated with a unique separator between them.
    The separator (concept_id = 32766) prevents cross-sequence digrams.
    """
    SEP = 32766  # concept_id reserved as separator; never appears in practice
    combined: list[int] = []
    for idx, seq in enumerate(typed_corpus):
        combined.extend(seq)
        if idx < len(typed_corpus) - 1:
            combined.append(SEP)
    if not combined:
        return Grammar(rules={0: GrammarRule(rule_id=0, body=[])}, start=0)
    return sequitur(combined)


def grammar_description_length(
    grammar: Grammar,
    typed_corpus: list[list[int]],
) -> float:
    """Approximate MDL description length in bits.

    DL = bits(grammar) + bits(corpus|grammar)
    """
    n_terms = 0
    for r in grammar.rules.values():
        for sym in r.body:
            if _is_terminal(sym):
                cid = _terminal_id(sym)
                if cid > n_terms:
                    n_terms = cid
    n_terms += 1  # vocab_size upper bound
    return grammar.encode_length(n_terms)


def typed_corpus_from_lattice(
    corpus: list[list[str]],
    hankel: HankelCount,
    lattice: ConceptLattice,
    r: int = 1,
) -> list[list[int]]:
    """Assign types to all sequences in corpus using `lattice` at radius `r`."""
    return [assign_types(seq, hankel, lattice, r=r) for seq in corpus]
