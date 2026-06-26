"""Recursive residual modelling — the ONE general mechanism for structured deviations (RESEARCH.md R7).

A bespoke detector per structure type (orbits for products, a holonomy for carry, ...) is the maximal-work
path. Instead: model the transitions, take the prediction ERRORS (the residual), model the residual with the
SAME machinery, and recurse until the residual has no compressing structure (the MDL stop). Carry, context-
dependence, exceptions and (nested) hierarchy are then ONE problem — structure in the residual — captured by
one loop, not N detectors.

Concretely, over a state's factored COORDINATES (what disentanglement provides): each action's effect is a
coordinate DELTA. The model is a per-action DECISION LIST of (predicate, delta) rules, learned by residual
peeling — the dominant delta is the base rule; the states it mispredicts are the residual; the most common
residual delta plus the simplest predicate over the coordinates that selects those states (and breaks none of
the already-correct ones) is the next rule, prepended so it is checked first; recurse on what is still wrong.
If a residual delta-group shares NO coordinate value, there is no compressing predicate — the loop STOPS
rather than memorise a per-state lookup (the MDL stop). Pure stdlib."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product


def _delta(cs, cn):
    return tuple(b - a for a, b in zip(cs, cn))


def _mode(items):
    return Counter(items).most_common(1)[0][0]


def _apply(dl, cs):
    for pred, _desc, delta in dl:                            # decision list, specific rules first, base last
        if pred is None or pred(cs):
            return tuple(a + d for a, d in zip(cs, delta))
    return cs


def _coord_literals(need, d):
    """Per-coordinate literals on dim d that cover ALL of `need`: ==value (if `need` share it), and the bound
    literals >=min / <=max (which cover all of need by construction). The >=/<= give RANGES, so wraps and other
    interval structure are expressible, not just point matches."""
    vals = {cs[d] for cs in need}
    lits = []
    if len(vals) == 1:
        v = next(iter(vals))
        lits.append((lambda cs, d=d, v=v: cs[d] == v, f"c{d}=={v}"))
    lo, hi = min(vals), max(vals)
    lits.append((lambda cs, d=d, lo=lo: cs[d] >= lo, f"c{d}>={lo}"))
    lits.append((lambda cs, d=d, hi=hi: cs[d] <= hi, f"c{d}<={hi}"))
    return lits


def _find_predicate(need, correct):
    """The SIMPLEST predicate (a conjunction of per-coordinate literals ==/>=/<=) that covers ALL of `need` and
    matches NONE of the already-correct states. None if no such predicate exists (no compression -> an MDL
    stop, not a per-state lookup). == literals are tried before ranges, fewer coordinates before more."""
    correct, ndim = list(correct), len(need[0])
    for dims in range(1, ndim + 1):                         # fewest coordinates first (Occam / MDL)
        for combo in combinations(range(ndim), dims):
            for choice in product(*[_coord_literals(need, d) for d in combo]):
                lits = [p for p, _ in choice]
                pred = lambda cs, lits=lits: all(p(cs) for p in lits)    # covers all need by construction
                if not any(pred(cs) for cs in correct):     # must break none of the already-correct states
                    return pred, " & ".join(desc for _, desc in choice)
    return None, None


def recursive_residual(coords, transitions, actions, max_levels=16):
    """Learn, per action, a decision list of (predicate, description, delta) rules by recursive residual
    peeling over the states' factored `coords`. `transitions` = [(state, action, next_state)]."""
    rules = {}
    for a in actions:
        edges = [(coords[s], coords[sn]) for (s, act, sn) in transitions if act == a]
        dl = [(None, "base", _mode([_delta(cs, cn) for cs, cn in edges]))]   # base = the dominant delta
        for _ in range(max_levels):
            residual = [(cs, cn) for cs, cn in edges if _apply(dl, cs) != cn]
            if not residual:
                break
            correct = [cs for cs, cn in edges if _apply(dl, cs) == cn]
            added = False
            for target, _ in Counter(_delta(cs, cn) for cs, cn in residual).most_common():
                need = [cs for cs, cn in residual if _delta(cs, cn) == target]
                pred, desc = _find_predicate(need, correct)                  # model the first COMPRESSIBLE group
                if pred is not None:
                    dl.insert(0, (pred, desc, target))                       # prepend: checked before the base
                    added = True
                    break
            if not added:
                break                                                        # no residual group compresses → MDL stop
        rules[a] = dl
    return rules


def predict(rules, coords, s, a):
    return _apply(rules[a], coords[s])
