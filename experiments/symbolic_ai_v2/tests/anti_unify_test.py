"""anti_unify_test.py — Phase 22: Tests for the anti-unification engine.

8 tests verifying:
  1. lgg of identical sequences returns the sequence with no variables.
  2. lgg produces a Variable at every differing position.
  3. lgg of arithmetic pairs produces a constant-offset pattern.
  4. lgg_all folds correctly over more than two sequences.
  5. match succeeds on a concrete instance of a pattern.
  6. match fails on a constant mismatch.
  7. match enforces consistent variable binding.
  8. Round-trip: match(lgg(a,b)[0], a) always succeeds; instantiate recovers a.
  9. n_vars and is_ground helpers are correct.
  10. lgg_all on mixed-length sequences returns (None, []).
"""

from __future__ import annotations

import time
import pytest

from experiments.symbolic_ai_v2.reasoning.anti_unify import (
    Variable, lgg, lgg_all, match, instantiate, n_vars, is_ground,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def atom(x: int) -> int:
    """Just returns x — atom IDs are plain ints in tests."""
    return x


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLgg:
    def test_identical_sequences_no_variables(self):
        """lgg of two identical sequences is the sequence itself, no Variables."""
        pattern, ba, bb = lgg([1, 2, 3], [1, 2, 3])
        assert pattern == [1, 2, 3]
        assert ba == {}
        assert bb == {}

    def test_variable_at_differing_position(self):
        """lgg introduces a Variable exactly at each position where seqs differ."""
        pattern, ba, bb = lgg([1, 2, 3], [1, 9, 3])
        assert pattern[0] == 1
        assert isinstance(pattern[1], Variable)
        assert pattern[2] == 3
        assert len(ba) == 1
        v = pattern[1]
        assert ba[v] == 2
        assert bb[v] == 9

    def test_arithmetic_offset_pattern(self):
        """lgg of (3, 7) and (2, 6) gives pattern [?0, ?1] — two vars.

        The *relationship* between ?0 and ?1 is encoded by the step function
        in fold_detect, not by lgg directly.  lgg's job is only to find shared
        structure; for completely distinct pairs it gives two variables.
        """
        pattern, ba, bb = lgg([3, 7], [2, 6])
        # Both positions differ → two Variables
        assert all(isinstance(e, Variable) for e in pattern)
        # But the constant-offset relationship is preserved by the step lookup
        # (tested in fold_detect_test.py)

    def test_mixed_constant_and_variable(self):
        """lgg of [op, 3, 4, 7] and [op, 2, 5, 7] shares op and 7."""
        op = 99
        pattern, ba, bb = lgg([op, 3, 4, 7], [op, 2, 5, 7])
        assert pattern[0] == op        # shared constant
        assert isinstance(pattern[1], Variable)
        assert isinstance(pattern[2], Variable)
        assert pattern[3] == 7         # shared constant

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            lgg([1, 2], [1, 2, 3])


class TestLggAll:
    def test_single_sequence_returns_it(self):
        pattern, bindings = lgg_all([[10, 20, 30]])
        assert pattern == [10, 20, 30]
        assert bindings == [{}]

    def test_three_sequences_correct_vars(self):
        """lgg_all of [1,2,3], [1,9,3], [1,5,3] → [1, ?0, 3]."""
        pattern, bindings = lgg_all([[1, 2, 3], [1, 9, 3], [1, 5, 3]])
        assert pattern is not None
        assert pattern[0] == 1
        assert isinstance(pattern[1], Variable)
        assert pattern[2] == 3
        v = pattern[1]
        assert bindings[0][v] == 2
        assert bindings[1][v] == 9
        assert bindings[2][v] == 5

    def test_all_same_no_variables(self):
        pattern, bindings = lgg_all([[7, 8], [7, 8], [7, 8]])
        assert pattern == [7, 8]
        assert all(b == {} for b in bindings)

    def test_mixed_lengths_returns_none(self):
        pattern, bindings = lgg_all([[1, 2], [1, 2, 3]])
        assert pattern is None
        assert bindings == []

    def test_empty_list_returns_none(self):
        pattern, bindings = lgg_all([])
        assert pattern is None
        assert bindings == []


class TestMatch:
    def test_match_ground_pattern(self):
        """A ground pattern matches only the identical sequence."""
        assert match([1, 2, 3], [1, 2, 3]) == {}
        assert match([1, 2, 3], [1, 9, 3]) is None

    def test_match_with_variable(self):
        """A variable matches any atom and records the binding."""
        v = Variable(0)
        b = match([1, v, 3], [1, 42, 3])
        assert b == {v: 42}

    def test_match_consistent_variable(self):
        """The same Variable at two positions must match the same atom."""
        v = Variable(0)
        # Both positions bound to 5 → success
        assert match([v, 2, v], [5, 2, 5]) == {v: 5}
        # Inconsistent binding → failure
        assert match([v, 2, v], [5, 2, 9]) is None

    def test_match_length_mismatch(self):
        assert match([1, 2], [1, 2, 3]) is None

    def test_match_returns_empty_dict_on_ground_match(self):
        result = match([1, 2, 3], [1, 2, 3])
        assert result == {}
        assert isinstance(result, dict)


class TestInstantiate:
    def test_instantiate_replaces_vars(self):
        v0 = Variable(0)
        v1 = Variable(1)
        pattern = [1, v0, v1, 4]
        bindings = {v0: 10, v1: 20}
        assert instantiate(pattern, bindings) == [1, 10, 20, 4]

    def test_instantiate_unbound_returns_none(self):
        v0 = Variable(0)
        assert instantiate([v0], {}) is None

    def test_instantiate_ground_pattern(self):
        assert instantiate([1, 2, 3], {}) == [1, 2, 3]


class TestRoundTrip:
    def test_match_lgg_always_succeeds_for_input_a(self):
        """match(lgg(a,b)[0], a) must always succeed."""
        pairs = [
            ([1, 2, 3], [4, 5, 6]),
            ([1, 2, 3], [1, 2, 3]),
            ([7, 8],    [7, 9]),
            ([1],       [2]),
        ]
        for a, b in pairs:
            pattern, ba, _ = lgg(a, b)
            result = match(pattern, a)
            assert result is not None, f"match failed for a={a}, b={b}, pattern={pattern}"
            recovered = instantiate(pattern, result)
            assert recovered == a, f"instantiate failed for a={a}"

    def test_match_lgg_always_succeeds_for_input_b(self):
        """match(lgg(a,b)[0], b) must always succeed."""
        pairs = [([1, 2, 3], [4, 5, 6]), ([7, 8], [7, 9])]
        for a, b in pairs:
            pattern, _, bb = lgg(a, b)
            result = match(pattern, b)
            assert result is not None
            recovered = instantiate(pattern, result)
            assert recovered == b


class TestHelpers:
    def test_n_vars(self):
        v0 = Variable(0)
        v1 = Variable(1)
        assert n_vars([1, v0, v1, v0]) == 2   # v0 counted once
        assert n_vars([1, 2, 3]) == 0

    def test_is_ground(self):
        assert is_ground([1, 2, 3])
        assert not is_ground([1, Variable(0), 3])

    def test_performance_lgg_all_1000_pairs(self):
        """lgg_all on 1000 pairs of length-5 sequences completes in < 1 s."""
        seqs = [[i % 10, (i + 1) % 10, 99, i % 7, 42] for i in range(1000)]
        t0 = time.perf_counter()
        pattern, bindings = lgg_all(seqs)
        elapsed = time.perf_counter() - t0
        assert pattern is not None
        assert elapsed < 1.0, f"lgg_all took {elapsed:.2f}s, expected < 1s"
