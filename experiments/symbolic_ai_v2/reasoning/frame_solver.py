"""frame_solver.py — Operator frame extraction and adjunction-based prediction.

Discovers operator frames of the form

    op  arg₁ … argₙ  eq  →  answer

from a state space's raw_dist, then detects three adjunction types:

  Type U  (arity 1):  op1(a)   = b  ↔  op2(b)   = a      e.g.  succ ⊣ pred
  Type A  (arity 2):  op1(a,b) = c  ↔  op2(c,b) = a      e.g.  add  ⊣ sub
  Type B  (arity 2):  op(a,b)  = c  ↔  op(a,c)  = b      e.g.  sub self-adjoint

No hardcoded rules.  No type checks.  An operator named 'succ' is treated
identically to one named 'red' — only the co-occurrence geometry matters
(Yoneda lemma).  The same algorithm discovers ordinal chains in integer tokens,
musical pitch sequences, month names, or any domain with an inverse structure.

Adjunction detection implements the categorical unit/counit conditions:
  unit   coverage: for what fraction of op1 training pairs does the adjoint
                   op2 pair exist in training?
  counit coverage: symmetric check in the other direction
Both must meet `min_coverage` for the adjunction to be declared.

Public API
----------
FrameSolver.build(raw_dist)   — extract frames, detect adjunctions
FrameSolver.predict(ctx)      — {answer: 1.0} or {} if no adjunction fires
FrameSolver.adjunctions       — list of discovered Adjunction objects (inspect)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── Adjunction record ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Adjunction:
    """A discovered adjunction relationship between two operator frames.

    Type U  (arity 1):  op1(a)   = b  ↔  op2(b)   = a   (e.g. succ ⊣ pred)
    Type A  (arity 2):  op1(a,b) = c  ↔  op2(c,b) = a   (e.g. add  ⊣ sub )
    Type B  (arity 2):  op(a,b)  = c  ↔  op(a,c)  = b   (sub self-adjoint  )

    Attributes
    ----------
    op1, op2    : operator token strings
    arity       : number of arguments (1 or 2)
    swap        : 'U', 'A', or 'B'
    coverage    : fraction of op1 training pairs whose adjoint op2 pair exists
    """
    op1:      str
    op2:      str
    arity:    int
    swap:     str
    coverage: float


# ── FrameSolver ───────────────────────────────────────────────────────────────

class FrameSolver:
    """Predict via operator frames and discovered adjunctions.

    Construction
    ------------
    fs = FrameSolver.build(sp.ss.raw_dist)

    Prediction
    ----------
    dist = fs.predict(('sub', '13', '7', 'eq'))   # → {'6': 1.0}
    dist = fs.predict(('succ', '5', 'eq'))         # → {'6': 1.0}
    dist = fs.predict(('add', '6', '7', 'eq'))     # → {'13': 1.0}  (if not in training)

    Returns {} when no adjunction fires — the caller should try the next
    prediction level.
    """

    def __init__(
        self,
        frames:      dict,
        adjunctions: list,
        _rev_u:      dict,
        _rev_a:      dict,
        _rev_b:      dict,
    ) -> None:
        self.frames      = frames        # op -> arity -> {args_tuple: answer}
        self.adjunctions = adjunctions   # list[Adjunction]
        self._rev_u      = _rev_u        # op -> {answer: arg}
        self._rev_a      = _rev_a        # op -> {(answer, arg2): arg1}
        self._rev_b      = _rev_b        # op -> {(answer, arg1): arg2}

    # ── Class-method constructor ───────────────────────────────────────────

    @classmethod
    def build(
        cls,
        raw_dist:       dict,
        min_coverage:   float = 0.75,
        min_frame_size: int   = 3,
    ) -> 'FrameSolver':
        """Build a FrameSolver from a SpectralPredictor's state space raw_dist.

        Parameters
        ----------
        raw_dist       : StateSpace.raw_dist — {ctx_tuple: {token: prob}}
        min_coverage   : adjunction threshold — fraction of op1 pairs that must
                         have a matching op2 pair (unit/counit coverage)
        min_frame_size : minimum (args, answer) pairs in a frame before it is
                         considered for adjunction detection (noise guard)
        """
        frames             = _extract_frames(raw_dist)
        rev_u, rev_a, rev_b = _build_reverses(frames)
        adjs               = _detect_adjunctions(
            frames, rev_u, rev_a, rev_b, min_coverage, min_frame_size,
        )
        return cls(frames, adjs, rev_u, rev_a, rev_b)

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, ctx: tuple) -> dict:
        """Return {answer: 1.0} if an adjunction fires for this context.

        Returns {} when the context shape is not an operator frame query
        (i.e. does not end with 'eq') or when no adjunction applies.

        The caller should fall through to the next prediction level on {}.
        """
        if not ctx or ctx[-1] != 'eq':
            return {}
        op   = ctx[0]
        args = ctx[1:-1]
        n    = len(args)
        if n == 0:
            return {}

        # Direct frame lookup (Level 1 should have caught this, but guard)
        direct = self.frames.get(op, {}).get(n, {}).get(args)
        if direct is not None:
            return {direct: 1.0}

        # Adjunction lookup — try every detected adjunction
        for adj in self.adjunctions:
            if adj.arity != n:
                continue
            ans = _apply_adjunction(adj, op, args, self._rev_u, self._rev_a, self._rev_b)
            if ans is not None:
                return {ans: 1.0}

        return {}


# ── Frame extraction ──────────────────────────────────────────────────────────

def _extract_frames(raw_dist: dict) -> dict:
    """Extract deterministic operator frames from raw_dist.

    Scans for rows of the form (op, arg1, …, argN, 'eq') → answer where the
    predicted distribution is near-deterministic (best_prob ≥ 0.95).

    Returns
    -------
    frames[op][arity][(arg1, …, argN)] = answer_str
    """
    frames: dict = {}
    for ctx, dist in raw_dist.items():
        if len(ctx) < 3 or ctx[-1] != 'eq':
            continue
        best_tok = max(dist, key=dist.get)
        if dist[best_tok] < 0.95:
            continue
        op   = ctx[0]
        args = ctx[1:-1]
        if not args:
            continue
        n = len(args)
        frames.setdefault(op, {}).setdefault(n, {})[args] = best_tok
    return frames


# ── Reverse index construction ────────────────────────────────────────────────

def _build_reverses(frames: dict):
    """Build reverse lookup tables for all operator frames.

    rev_u[op][answer]          = arg        (arity 1)
    rev_a[op][(answer, arg2)]  = arg1       (arity 2, type-A: 1st arg ↔ answer)
    rev_b[op][(answer, arg1)]  = arg2       (arity 2, type-B: 2nd arg ↔ answer)
    """
    rev_u: dict = {}
    rev_a: dict = {}
    rev_b: dict = {}

    for op, arities in frames.items():
        for n, mapping in arities.items():
            if n == 1:
                rev_u.setdefault(op, {})
                for (a,), ans in mapping.items():
                    rev_u[op][ans] = a
            elif n == 2:
                rev_a.setdefault(op, {})
                rev_b.setdefault(op, {})
                for (a, b), c in mapping.items():
                    # Type A reverse: op(a,b)=c  →  rev_a[op][(c,b)] = a
                    rev_a[op][(c, b)] = a
                    # Type B reverse: op(a,b)=c  →  rev_b[op][(c,a)] = b
                    rev_b[op][(c, a)] = b

    return rev_u, rev_a, rev_b


# ── Adjunction detection ──────────────────────────────────────────────────────

def _detect_adjunctions(
    frames:         dict,
    rev_u:          dict,
    rev_a:          dict,
    rev_b:          dict,
    min_coverage:   float,
    min_frame_size: int,
) -> list:
    """Detect U, A, and B adjunctions between all operator pairs.

    For each candidate (op1, op2, swap_type):
      coverage = |{op1 training pairs whose adjoint op2 pair exists}| / |op1 pairs|

    Declared an adjunction when coverage ≥ min_coverage.

    Both (op1, op2) and (op2, op1) are checked as separate candidates so
    that predict() can fire on queries to either operator.
    """
    adjs: list = []
    seen: set  = set()

    def _add(adj: Adjunction) -> None:
        key = (adj.op1, adj.op2, adj.swap)
        if key not in seen:
            adjs.append(adj)
            seen.add(key)

    for op1, arities1 in frames.items():

        # ── Type U ────────────────────────────────────────────────────────
        f1_u = arities1.get(1, {})
        if len(f1_u) >= min_frame_size:
            for op2, arities2 in frames.items():
                f2_u = arities2.get(1, {})
                if len(f2_u) < min_frame_size:
                    continue
                # Unit coverage: for each (a,)→b in op1, does op2((b,))==a?
                hits = sum(1 for (a,), b in f1_u.items() if f2_u.get((b,)) == a)
                cov  = hits / len(f1_u)
                if cov >= min_coverage:
                    _add(Adjunction(op1=op1, op2=op2, arity=1, swap='U', coverage=cov))

        # ── Types A and B ─────────────────────────────────────────────────
        f1_2 = arities1.get(2, {})
        if len(f1_2) < min_frame_size:
            continue

        # Type A: op1(a,b)=c  ↔  op2(c,b)=a
        for op2, arities2 in frames.items():
            f2_2 = arities2.get(2, {})
            if len(f2_2) < min_frame_size:
                continue
            hits = sum(1 for (a, b), c in f1_2.items() if f2_2.get((c, b)) == a)
            cov  = hits / len(f1_2)
            if cov >= min_coverage:
                _add(Adjunction(op1=op1, op2=op2, arity=2, swap='A', coverage=cov))

        # Type B: op(a,b)=c  ↔  op(a,c)=b  (self-adjoint)
        hits = sum(1 for (a, b), c in f1_2.items() if f1_2.get((a, c)) == b)
        cov  = hits / len(f1_2)
        if cov >= min_coverage:
            _add(Adjunction(op1=op1, op2=op1, arity=2, swap='B', coverage=cov))

    return adjs


# ── Adjunction application ────────────────────────────────────────────────────

def _apply_adjunction(
    adj:   Adjunction,
    op:    str,
    args:  tuple,
    rev_u: dict,
    rev_a: dict,
    rev_b: dict,
) -> Optional[str]:
    """Try to answer op(args) using one adjunction record.

    Returns the predicted answer string, or None if this adjunction does not
    apply to the (op, args) pair.

    Type U:  op1(a)=b  ↔  op2(b)=a
      predict op1(a): find b where op2(b)=a  →  rev_u[op2][a]
      predict op2(b): find a where op1(a)=b  →  rev_u[op1][b]

    Type A:  op1(a,b)=c  ↔  op2(c,b)=a
      predict op2(c,b): find a where op1(a,b)=c  →  rev_a[op1][(c,b)]
      predict op1(a,b): find c where op2(c,b)=a  →  rev_a[op2][(a,b)]

    Type B:  op(a,b)=c  ↔  op(a,c)=b   (self-adjoint, op1==op2)
      predict op(a,b): find c where op(a,c)=b  →  rev_b[op][(b,a)]
    """
    swap = adj.swap

    if swap == 'U':
        a = args[0]
        if op == adj.op1:
            # predict op1(a): find b in op2's reverse where op2(b)=a
            return rev_u.get(adj.op2, {}).get(a)
        if op == adj.op2:
            # predict op2(b): find a in op1's reverse where op1(a)=b
            return rev_u.get(adj.op1, {}).get(a)

    elif swap == 'A':
        if len(args) != 2:
            return None
        a, b = args
        if op == adj.op2:
            # predict op2(c,b): find a where op1(a,b)=c  →  rev_a[op1][(c,b)]
            return rev_a.get(adj.op1, {}).get((a, b))
        if op == adj.op1:
            # predict op1(a,b): find c where op2(c,b)=a  →  rev_a[op2][(a,b)]
            return rev_a.get(adj.op2, {}).get((a, b))

    elif swap == 'B':
        if len(args) != 2 or adj.op1 != adj.op2:
            return None
        if op == adj.op1:
            a, b = args
            # predict op(a,b): find c where op(a,c)=b  →  rev_b[op][(b,a)]
            return rev_b.get(op, {}).get((b, a))

    return None
