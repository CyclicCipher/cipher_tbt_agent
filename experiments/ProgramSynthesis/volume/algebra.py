"""Operations on box concepts — the *algebra* (docs/phase2/VOLUME_CONCEPTS.md §1).

A deliberate structural echo of the owner's open question (§0 framing question): the CONCEPTS live
in `box.py` (regions = graph *nodes*); the OPERATIONS that act on them live *here* (maps over
regions). `meet` and `entails` relate or produce concepts — they are **not themselves concepts**.
(A functional *law* — §10A — is a third kind again: a constraint between quantity-spaces, not
expressible as a box meet. So this module is evidence for the distinction, not against it.)

- meet(a, b)    = AND / context-narrowing — the conjunction region over the union of subspaces.
- entails(a, b) = IS-A / specificity — a ⊆ b. Asymmetric, which is exactly the relation a *point*
                  embedding cannot express, and the reason regions beat points for hierarchy/binding.
- volume(a)     = a measure of generality (within one subspace; cross-subspace is apples-to-oranges).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .box import BoxConcept

_NEG_INF, _POS_INF = float("-inf"), float("inf")


def _bounds(c: BoxConcept) -> Dict[int, Tuple[float, float]]:
    return {d: (c.lo[i], c.hi[i]) for i, d in enumerate(c.dims)}


def meet(a: BoxConcept, b: BoxConcept) -> Optional[BoxConcept]:
    """Intersection over the UNION of the two subspaces: on a shared dimension take the tighter
    interval; on a dimension only one constrains, keep it. Returns None if a shared dimension's
    constraints are disjoint — an empty region, i.e. a context that cannot be satisfied."""
    ab, bb = _bounds(a), _bounds(b)
    dims, lo, hi = [], [], []
    for d in sorted(set(ab) | set(bb)):
        al, ah = ab.get(d, (_NEG_INF, _POS_INF))
        bl, bh = bb.get(d, (_NEG_INF, _POS_INF))
        l, h = max(al, bl), min(ah, bh)
        if l > h:
            return None                       # disjoint on this dim -> empty concept
        dims.append(d); lo.append(l); hi.append(h)
    return BoxConcept(tuple(dims), tuple(lo), tuple(hi))


def entails(a: BoxConcept, b: BoxConcept) -> bool:
    """a ⊆ b: a is at least as specific as b. For every dimension b constrains, a must constrain it
    too and lie within b's interval. a may constrain *extra* dimensions — that only makes it more
    specific. Asymmetric: a broader concept does not entail a narrower one."""
    ab = _bounds(a)
    for i, d in enumerate(b.dims):
        if d not in ab:                       # b constrains a dim a leaves open -> a is broader there
            return False
        al, ah = ab[d]
        if al < b.lo[i] or ah > b.hi[i]:
            return False
    return True


def volume(a: BoxConcept) -> float:
    """Product of side lengths over a's own subspace (a k-dim measure of generality). Comparable
    only within the same subspace; across subspaces it is apples-to-oranges (different dimensions)."""
    v = 1.0
    for i in range(a.n_dims):
        v *= (a.hi[i] - a.lo[i])
    return v
