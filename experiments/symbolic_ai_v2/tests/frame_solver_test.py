"""frame_solver_test.py — Unit tests for FrameSolver adjunction detection.

Tests cover all three adjunction types:
  Type U  (unary):   succ ⊣ pred
  Type A  (binary):  add  ⊣ sub
  Type B  (binary):  sub self-adjoint

Each test constructs a minimal raw_dist by hand, builds a FrameSolver,
and verifies that predict() returns the correct answer for held-out queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.symbolic_ai_v2.reasoning.frame_solver import FrameSolver, Adjunction


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_dist(facts: list[tuple]) -> dict:
    """Build a minimal raw_dist from a list of (context_tuple, answer) facts."""
    raw: dict = {}
    for ctx, ans in facts:
        raw[ctx] = {ans: 1.0}
    return raw


def _succ_pred_raw(n_max: int = 20, holdout: set | None = None) -> dict:
    """succ and pred facts for 0..n_max with optional holdouts."""
    holdout = holdout or set()
    facts = []
    for n in range(n_max):
        if ('succ', n) not in holdout:
            facts.append((('succ', str(n), 'eq'), str(n + 1)))
    for n in range(1, n_max + 1):
        if ('pred', n) not in holdout:
            facts.append((('pred', str(n), 'eq'), str(n - 1)))
    return _make_raw_dist(facts)


def _add_sub_raw(a_max: int = 5, b_max: int = 5,
                 holdout_add: set | None = None,
                 holdout_sub: set | None = None) -> dict:
    """add and sub facts, with optional holdouts."""
    holdout_add = holdout_add or set()
    holdout_sub = holdout_sub or set()
    facts = []
    for a in range(a_max + 1):
        for b in range(b_max + 1):
            c = a + b
            if (a, b) not in holdout_add:
                facts.append((('add', str(a), str(b), 'eq'), str(c)))
            if (c, b) not in holdout_sub:
                facts.append((('sub', str(c), str(b), 'eq'), str(a)))
    return _make_raw_dist(facts)


# ── Type U tests (succ ⊣ pred) ────────────────────────────────────────────────

class TestTypeU:
    def test_adjunction_detected(self):
        """succ⊣pred and pred⊣succ are detected from joint data."""
        raw = _succ_pred_raw()
        fs  = FrameSolver.build(raw)
        ops  = {(a.op1, a.op2, a.swap) for a in fs.adjunctions}
        assert ('succ', 'pred', 'U') in ops
        assert ('pred', 'succ', 'U') in ops

    def test_predict_succ_held_out(self):
        """succ(5) is predicted via pred(6)=5 when succ(5) is held out."""
        raw = _succ_pred_raw(holdout={('succ', 5)})
        fs  = FrameSolver.build(raw)
        assert fs.predict(('succ', '5', 'eq')) == {'6': 1.0}

    def test_predict_pred_held_out(self):
        """pred(2) is predicted via succ(1)=2 when pred(2) is held out."""
        raw = _succ_pred_raw(holdout={('pred', 2)})
        fs  = FrameSolver.build(raw)
        assert fs.predict(('pred', '2', 'eq')) == {'1': 1.0}

    def test_no_false_positive_on_seen_fact(self):
        """predict() on a seen fact returns the direct frame answer."""
        raw = _succ_pred_raw()
        fs  = FrameSolver.build(raw)
        # succ(3) is in training; should return direct answer
        assert fs.predict(('succ', '3', 'eq')) == {'4': 1.0}

    def test_returns_empty_for_unknown_input(self):
        """predict() returns {} when the input token has no adjoint entry."""
        raw = _succ_pred_raw(n_max=10)
        fs  = FrameSolver.build(raw)
        # succ(99) is unknown — neither direct nor via adjunction
        assert fs.predict(('succ', '99', 'eq')) == {}

    def test_coverage_threshold_filters_noise(self):
        """A near-random frame pair does not spuriously trigger detection."""
        # Build two unrelated unary operators 'foo' and 'bar'
        facts = []
        for i in range(10):
            facts.append((('foo', str(i), 'eq'), str(i * 3 % 7)))
            facts.append((('bar', str(i), 'eq'), str((i + 5) % 7)))
        raw = _make_raw_dist(facts)
        fs  = FrameSolver.build(raw, min_coverage=0.75)
        ops = {(a.op1, a.op2, a.swap) for a in fs.adjunctions}
        assert ('foo', 'bar', 'U') not in ops
        assert ('bar', 'foo', 'U') not in ops


# ── Type A tests (add ⊣ sub) ──────────────────────────────────────────────────

class TestTypeA:
    def test_adjunction_detected(self):
        """add⊣sub and sub⊣add detected from joint add/sub data."""
        raw = _add_sub_raw()
        fs  = FrameSolver.build(raw)
        ops = {(a.op1, a.op2, a.swap) for a in fs.adjunctions}
        assert ('add', 'sub', 'A') in ops
        assert ('sub', 'add', 'A') in ops

    def test_predict_sub_via_add(self):
        """sub(7,4)=3 predicted via add(3,4)=7 when sub(7,4) held out."""
        raw = _add_sub_raw(holdout_sub={(7, 4)})
        fs  = FrameSolver.build(raw)
        assert fs.predict(('sub', '7', '4', 'eq')) == {'3': 1.0}

    def test_predict_add_via_sub(self):
        """add(3,4)=7 predicted via sub(7,4)=3 when add(3,4) held out."""
        raw = _add_sub_raw(holdout_add={(3, 4)})
        fs  = FrameSolver.build(raw)
        assert fs.predict(('add', '3', '4', 'eq')) == {'7': 1.0}

    def test_different_second_arg(self):
        """Adjunction correctly keeps the second argument fixed."""
        raw = _add_sub_raw(a_max=9, holdout_sub={(9, 2)})
        fs  = FrameSolver.build(raw)
        # sub(9,2)=7 via add(7,2)=9
        assert fs.predict(('sub', '9', '2', 'eq')) == {'7': 1.0}

    def test_no_answer_when_adjoint_also_missing(self):
        """Returns {} when both sub(c,b) and add(a,b) are held out and type-B path blocked."""
        # Hold out sub(7,4) [direct], add(3,4) [type-A adjoint], and sub(7,3) [type-B adjoint]
        raw = _add_sub_raw(holdout_add={(3, 4)}, holdout_sub={(7, 4), (7, 3)})
        fs  = FrameSolver.build(raw)
        # sub(7,4): direct missing; add(3,4): also missing; sub(7,3): also missing → no path
        assert fs.predict(('sub', '7', '4', 'eq')) == {}


# ── Type B tests (sub self-adjoint) ───────────────────────────────────────────

class TestTypeB:
    def test_self_adjunction_detected(self):
        """sub's type-B self-adjunction detected from subtraction-only data."""
        # Only subtraction — no addition data
        facts = [
            (('sub', '7', '3', 'eq'), '4'),
            (('sub', '7', '4', 'eq'), '3'),
            (('sub', '8', '3', 'eq'), '5'),
            (('sub', '8', '5', 'eq'), '3'),
            (('sub', '9', '4', 'eq'), '5'),
            (('sub', '9', '5', 'eq'), '4'),
            (('sub', '6', '2', 'eq'), '4'),
            (('sub', '6', '4', 'eq'), '2'),
            (('sub', '5', '1', 'eq'), '4'),
            (('sub', '5', '4', 'eq'), '1'),
        ]
        raw = _make_raw_dist(facts)
        fs  = FrameSolver.build(raw, min_frame_size=3)
        ops = {(a.op1, a.op2, a.swap) for a in fs.adjunctions}
        assert ('sub', 'sub', 'B') in ops

    def test_predict_sub_via_type_b(self):
        """sub(7,4) predicted via sub(7,3)=4 using type-B self-adjunction."""
        facts = [
            (('sub', '7', '3', 'eq'), '4'),   # sub(7,3)=4  ↔  sub(7,4)=3
            (('sub', '8', '3', 'eq'), '5'),
            (('sub', '8', '5', 'eq'), '3'),
            (('sub', '9', '4', 'eq'), '5'),
            (('sub', '9', '5', 'eq'), '4'),
            (('sub', '6', '2', 'eq'), '4'),
            (('sub', '6', '4', 'eq'), '2'),
        ]
        raw = _make_raw_dist(facts)
        fs  = FrameSolver.build(raw, min_frame_size=3)
        # sub(7,4)=3 is not in training, but sub(7,3)=4 IS → type B fires
        assert fs.predict(('sub', '7', '4', 'eq')) == {'3': 1.0}


# ── Non-eq contexts ───────────────────────────────────────────────────────────

class TestNonEquality:
    def test_non_eq_context_returns_empty(self):
        """predict() returns {} for contexts not ending in 'eq'."""
        raw = _succ_pred_raw()
        fs  = FrameSolver.build(raw)
        assert fs.predict(('succ', '3')) == {}
        assert fs.predict(('3',)) == {}
        assert fs.predict(()) == {}

    def test_single_token_context_returns_empty(self):
        """predict() returns {} for a single-token context (op, eq) — no args."""
        raw = _succ_pred_raw()
        fs  = FrameSolver.build(raw)
        assert fs.predict(('succ', 'eq')) == {}
