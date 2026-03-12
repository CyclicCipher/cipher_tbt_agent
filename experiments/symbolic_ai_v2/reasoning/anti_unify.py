"""anti_unify.py — Anti-unification (least general generalisation) engine.

Phase 22 of the reasoning layer redesign (ROADMAP_REDESIGN.md).

Anti-unification (Plotkin 1970, Reynolds 1970) computes the *least general
generalisation* (lgg) of two or more sequences: the most specific pattern that
subsumes all of them.  It is the categorical dual of Robinson unification:
where unification computes a coequalizer (most-specific common instance),
anti-unification computes an equalizer (most-specific common generalisation).

This module is deliberately independent of MorphismGraph.  All operations
work on plain Python lists of integers (atom IDs) and Variable objects.
The only coupling to the rest of the system occurs in Phase 25 (templates.py).

Public API
----------
Variable                 — frozen dataclass marking a free slot in a pattern
Pattern                  — type alias: list[int | Variable]
Bindings                 — type alias: dict[Variable, int]
lgg(a, b)                — least general generalisation of two sequences
lgg_all(seqs)            — lgg folded over a list of sequences
match(pattern, seq)      — test whether seq is an instance of pattern
instantiate(pattern, b)  — apply bindings to produce a concrete sequence
n_vars(pattern)          — count distinct Variables in a pattern
is_ground(pattern)       — True iff pattern contains no Variables
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


# ── Variable ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Variable:
    """A free slot in a pattern.  Two Variables are equal iff their ids match.

    Variables with the same id within a single pattern must be bound to the
    same atom in any valid match (consistency requirement).

    Ids are assigned locally within each lgg / lgg_all call and carry no
    global meaning.  Callers that merge patterns from different lgg calls
    must renumber variables to avoid accidental id collisions.
    """
    id: int

    def __repr__(self) -> str:
        return f"?{self.id}"


# ── Type aliases ───────────────────────────────────────────────────────────────

PatternElem = Union[int, Variable]
Pattern     = list[PatternElem]
Bindings    = dict[Variable, int]


# ── Core operations ────────────────────────────────────────────────────────────

def lgg(seq_a: list[int], seq_b: list[int]) -> tuple[Pattern, Bindings, Bindings]:
    """Compute the least general generalisation of two atom-id sequences.

    Both sequences must have the same length; raises ValueError otherwise.

    Returns
    -------
    (pattern, bindings_a, bindings_b)
      pattern    : the lgg — each position is either a shared constant
                   atom_id or a fresh Variable where the two sequences differ.
      bindings_a : maps each Variable in pattern to its value in seq_a.
      bindings_b : maps each Variable in pattern to its value in seq_b.

    Example
    -------
    >>> lgg([1, 2, 3], [1, 9, 3])
    ([1, ?0, 3], {?0: 2}, {?0: 9})
    """
    if len(seq_a) != len(seq_b):
        raise ValueError(
            f"lgg requires equal-length sequences, got {len(seq_a)} and {len(seq_b)}"
        )

    pattern:    Pattern  = []
    bindings_a: Bindings = {}
    bindings_b: Bindings = {}
    var_counter = 0

    for a, b in zip(seq_a, seq_b):
        if a == b:
            pattern.append(a)
        else:
            v = Variable(var_counter)
            var_counter += 1
            pattern.append(v)
            bindings_a[v] = a
            bindings_b[v] = b

    return pattern, bindings_a, bindings_b


def lgg_all(seqs: list[list[int]]) -> tuple[Optional[Pattern], list[Bindings]]:
    """Compute the lgg of an arbitrary collection of atom-id sequences.

    All sequences must have the same length.  If they differ, or if seqs is
    empty, returns (None, []).

    Returns
    -------
    (pattern, bindings_list)
      pattern       : the lgg pattern (None if seqs is empty or mixed-length).
      bindings_list : one Bindings dict per input sequence, mapping each
                      Variable in pattern to that sequence's atom at the
                      variable's position.

    Algorithm
    ---------
    Left-fold: initialise pattern = seqs[0] (all atoms).  For each subsequent
    sequence, convert any position where pattern[i] != seq[i] to a Variable.
    After the fold, rebuild bindings for all sequences in one linear pass.
    """
    if not seqs:
        return None, []

    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        return None, []

    n = len(seqs[0])
    # Start with the first sequence as all-atom pattern
    pattern: Pattern = list(seqs[0])
    var_counter = 0

    for seq in seqs[1:]:
        for i in range(n):
            elem = pattern[i]
            if isinstance(elem, Variable):
                # Already generalised at this position — keep it
                continue
            if elem != seq[i]:
                pattern[i] = Variable(var_counter)
                var_counter += 1

    # Build bindings for every input sequence
    all_bindings: list[Bindings] = []
    for seq in seqs:
        b: Bindings = {}
        for i in range(n):
            elem = pattern[i]
            if isinstance(elem, Variable):
                b[elem] = seq[i]
        all_bindings.append(b)

    return pattern, all_bindings


def match(pattern: Pattern, seq: list[int]) -> Optional[Bindings]:
    """Test whether seq is an instance of pattern; return bindings or None.

    A match succeeds iff:
      - len(pattern) == len(seq), AND
      - for every constant position: pattern[i] == seq[i], AND
      - for every Variable: all positions with the same Variable are bound to
        the same atom (consistency).

    Returns
    -------
    Bindings dict on success, None on failure.
    """
    if len(pattern) != len(seq):
        return None

    bindings: Bindings = {}
    for elem, val in zip(pattern, seq):
        if isinstance(elem, Variable):
            if elem in bindings:
                if bindings[elem] != val:
                    return None   # inconsistent binding
            else:
                bindings[elem] = val
        else:
            if elem != val:
                return None   # constant mismatch

    return bindings


def instantiate(pattern: Pattern, bindings: Bindings) -> Optional[list[int]]:
    """Apply bindings to pattern, producing a concrete atom-id sequence.

    Returns None if any Variable in pattern is absent from bindings.
    """
    result: list[int] = []
    for elem in pattern:
        if isinstance(elem, Variable):
            v = bindings.get(elem)
            if v is None:
                return None   # unbound variable
            result.append(v)
        else:
            result.append(elem)   # type: ignore[arg-type]
    return result


# ── Inspection helpers ─────────────────────────────────────────────────────────

def n_vars(pattern: Pattern) -> int:
    """Count distinct Variable objects in pattern."""
    return len({e for e in pattern if isinstance(e, Variable)})


def is_ground(pattern: Pattern) -> bool:
    """Return True iff pattern contains no Variables (fully instantiated)."""
    return all(isinstance(e, int) for e in pattern)
