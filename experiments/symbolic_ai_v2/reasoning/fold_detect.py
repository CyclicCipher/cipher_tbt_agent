"""fold_detect.py — Catamorphism (fold) synthesis from input-output examples.

Phase 23 of the reasoning layer redesign (ROADMAP_REDESIGN.md).

A fold (catamorphism) over a list has the form:

    fold([], base)       = base
    fold(x :: xs, base)  = step(x, fold(xs, base))   [right fold]

Given a collection of (input_path, output_atom_id) observations, this module
tests whether the computation is a right fold and extracts the step pattern
via anti-unification.

The step pattern is represented as a Pattern over a 2-element sequence
[current_elem, accumulator] → result.  Anti-unification is used instead of
polynomial fitting, so this works for any discrete token domain — not just
integer-valued atoms.

Public API
----------
FoldRule                 — discovered fold: base + step pattern
fold_detect(obs)         — detect fold from I/O pairs; returns FoldRule or None

Internal helpers (exposed for testing)
---------------------------------------
_step_constraints(obs)   — extract (elem, acc, result) triples from obs
_consistent_step(triples)— check a fixed step function against all triples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .anti_unify import Pattern, Bindings, lgg_all, match, instantiate, n_vars


# ── FoldRule ───────────────────────────────────────────────────────────────────

@dataclass
class FoldRule:
    """A discovered right-fold computation.

    Attributes
    ----------
    base         : atom_id of the base-case output (fold over empty list).
    step_pattern : anti-unified pattern over 2-element sequences
                   [current_elem, accumulator].  Each position is either a
                   constant atom_id or a Variable.
    step_lookup  : explicit lookup table {(elem_id, acc_id): result_id} built
                   from all observed step applications.  Used when the step
                   pattern alone is ambiguous (e.g. multiple Variables in rhs).
    coverage     : number of training (input_path, output) pairs confirmed.
    arity        : length of input_paths that were observed (min / max).
    """
    base:         int
    step_pattern: Pattern           # lgg of all (elem, acc) → result mappings
    step_lookup:  dict[tuple[int, int], int] = field(default_factory=dict)
    coverage:     int = 0
    min_arity:    int = 0
    max_arity:    int = 0

    def apply(self, elem_id: int, acc_id: int) -> Optional[int]:
        """Apply the step function to (elem, acc), returning the result atom_id.

        Tries the lookup table first (exact match), then the step_pattern
        (variable generalisation).
        """
        result = self.step_lookup.get((elem_id, acc_id))
        if result is not None:
            return result

        bindings = match(self.step_pattern, [elem_id, acc_id])
        if bindings is None:
            return None

        # The output is encoded as the 3rd column in the extended pattern:
        # step_pattern is built from 3-element [elem, acc, result] sequences,
        # so we need the corresponding output pattern.
        # (See fold_detect: step_pattern stores the 2-element input side only;
        #  the result side is handled by step_lookup + _result_pattern.)
        return None   # fallback: lookup only


@dataclass
class _FoldRuleFull:
    """Internal: holds both input and output patterns for fold synthesis."""
    base:            int
    input_pattern:   Pattern
    output_pattern:  Pattern
    step_lookup:     dict[tuple[int, int], int]
    coverage:        int
    min_arity:       int
    max_arity:       int

    def to_fold_rule(self) -> FoldRule:
        return FoldRule(
            base         = self.base,
            step_pattern = self.input_pattern,
            step_lookup  = self.step_lookup,
            coverage     = self.coverage,
            min_arity    = self.min_arity,
            max_arity    = self.max_arity,
        )

    def apply(self, elem_id: int, acc_id: int) -> Optional[int]:
        """Predict result for (elem, acc) using lookup then pattern."""
        result = self.step_lookup.get((elem_id, acc_id))
        if result is not None:
            return result
        b = match(self.input_pattern, [elem_id, acc_id])
        if b is None:
            return None
        out = instantiate(self.output_pattern, b)
        if out is None or len(out) != 1:
            return None
        return out[0]


# ── Step constraint extraction ─────────────────────────────────────────────────

def _step_constraints(
    obs: list[tuple[list[int], int]],
) -> Optional[list[tuple[int, int, int]]]:
    """Derive (elem_id, acc_id, result_id) triples from right-fold observations.

    A right fold satisfies:
        fold([], base)      = base
        fold(x :: xs, base) = step(x, fold(xs, base))

    Algorithm
    ---------
    1. Find base from the empty-path observation.  Absent → None.
    2. Process observations in ascending length order.
    3. For each path [p0, p1, ..., pk] → result:
       a. Compute acc = fold_right(path[1:], base) using already-known step values.
       b. Add the new constraint: step(p0, acc) = result.
    4. Report any inconsistency (same (elem, acc) maps to two different results).

    Processing in ascending length order ensures each length-k observation can
    use constraints derived from all shorter paths.
    """
    # Group by length
    by_len: dict[int, list[tuple[list[int], int]]] = {}
    for path, result in obs:
        by_len.setdefault(len(path), []).append((path, result))

    # Base case
    if 0 not in by_len:
        return None
    base_results = [r for (_, r) in by_len[0]]
    if len(set(base_results)) != 1:
        return None   # inconsistent base
    base = base_results[0]

    known: dict[tuple[int, int], int] = {}   # (elem, acc) → result

    for length in sorted(by_len.keys()):
        if length == 0:
            continue
        for path, final_result in by_len[length]:
            # Unfold path[1:] from right to compute acc = fold_right(path[1:], base)
            acc = base
            ok  = True
            for elem in reversed(path[1:]):
                key = (elem, acc)
                if key not in known:
                    ok = False
                    break
                acc = known[key]

            if not ok:
                continue  # can't determine inner acc yet — skip

            # Add the outermost step constraint: step(path[0], acc) = final_result
            outermost = (path[0], acc)
            if outermost in known:
                if known[outermost] != final_result:
                    return None   # inconsistency
            else:
                known[outermost] = final_result

    triples = [(e, a, r) for (e, a), r in known.items()]
    return triples if triples else None


# ── Main detector ──────────────────────────────────────────────────────────────

def fold_detect(
    obs: list[tuple[list[int], int]],
) -> Optional[_FoldRuleFull]:
    """Detect a right fold from input-output observations.

    Parameters
    ----------
    obs : list of (input_path, output_atom_id).
          input_path is a list of atom IDs (may be empty for the base case).

    Returns
    -------
    _FoldRuleFull on success, None if no consistent fold is found.

    A fold is found when:
      1. A base case (empty path) exists.
      2. All length-k observations are consistent with a single step function.
      3. Anti-unification of the step constraints yields a non-trivial pattern
         (i.e. not all Variables — there must be some consistent structure).
    """
    if not obs:
        return None

    constraints = _step_constraints(obs)
    if constraints is None:
        return None

    # Retrieve base
    by_len: dict[int, list[tuple[list[int], int]]] = {}
    for path, result in obs:
        by_len.setdefault(len(path), []).append((path, result))
    base = by_len[0][0][1]

    # Build step lookup
    step_lookup: dict[tuple[int, int], int] = {
        (e, a): r for (e, a, r) in constraints
    }

    # Anti-unify the input sides: [elem, acc] sequences
    input_seqs  = [[e, a] for (e, a, _) in constraints]
    output_seqs = [[r]    for (_, _, r) in constraints]

    in_pattern, in_bindings = lgg_all(input_seqs)
    if in_pattern is None:
        return None

    out_pattern, out_bindings = lgg_all(output_seqs)
    if out_pattern is None:
        return None

    # Reject maximally general patterns (all variables — no structure discovered)
    if in_pattern is not None and n_vars(in_pattern) == len(in_pattern):
        # Every position is a variable → no constant structure found
        # Still valid if we have a non-empty lookup table (memorised fold)
        if not step_lookup:
            return None

    coverage = sum(len(v) for v in by_len.values())
    min_arity = min(by_len.keys())
    max_arity = max(by_len.keys())

    return _FoldRuleFull(
        base           = base,
        input_pattern  = in_pattern,
        output_pattern = out_pattern,
        step_lookup    = step_lookup,
        coverage       = coverage,
        min_arity      = min_arity,
        max_arity      = max_arity,
    )
