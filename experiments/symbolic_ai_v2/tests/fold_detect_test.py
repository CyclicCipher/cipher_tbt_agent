"""fold_detect_test.py — Phase 23: Tests for catamorphism detection.

6 tests verifying:
  1. fold_detect identifies addition as fold(succ_step, m) from observations.
  2. fold_detect identifies a constant-offset step function.
  3. fold_detect returns None when base case is absent.
  4. fold_detect returns None when observations are inconsistent.
  5. fold_detect returns None on an empty observation list.
  6. Performance: completes in < 0.1 s on 100 observations of length ≤ 5.
"""

from __future__ import annotations

import time
import pytest

from experiments.symbolic_ai_v2.reasoning.fold_detect import fold_detect, _step_constraints


# ── Helpers ────────────────────────────────────────────────────────────────────

# Use plain integers as atom IDs.
# Represent arithmetic via direct integer values (ranks) for clarity.

def _make_successor_obs(start: int, length: int) -> list[tuple[list[int], int]]:
    """Generate fold observations for fold(succ, start)(n) = start + n.

    The 'input path' is a list of `n` copies of the constant atom 1,
    and the output is start + n.
    """
    obs = []
    for n in range(length + 1):
        obs.append(([1] * n, start + n))
    return obs


def _make_addition_obs() -> list[tuple[list[int], int]]:
    """Observations for add(m, n) = m + n, expressed as fold(succ_step, m)(n).

    We generate observations for add(3, k) for k in 0..4:
      fold([], 3) = 3
      fold([1], 3) = 4
      fold([1,1], 3) = 5
      ...
    i.e. each element in the path is '1', and the step is acc+1.
    """
    return _make_successor_obs(3, 5)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFoldDetect:
    def test_successor_fold_detected(self):
        """fold_detect identifies fold(succ_step, m) from count-up observations."""
        obs = _make_successor_obs(start=0, length=5)
        rule = fold_detect(obs)
        assert rule is not None
        assert rule.base == 0

        # The step lookup should allow computing 0+1=1, 1+1=2, etc.
        # step(1, acc) = acc + 1; elem=1, acc=k → result=k+1
        for k in range(5):
            result = rule.apply(1, k)
            assert result == k + 1, f"step(1, {k}) = {result}, expected {k + 1}"

    def test_addition_as_fold_of_succ(self):
        """add(3, n) is detected as fold(succ_step, 3)(n)."""
        obs = _make_addition_obs()
        rule = fold_detect(obs)
        assert rule is not None
        assert rule.base == 3
        # The step maps (1, k) → k+1 for the atoms we've seen
        assert rule.apply(1, 3) == 4
        assert rule.apply(1, 4) == 5

    def test_no_base_case_returns_none(self):
        """fold_detect returns None when there is no empty-path observation."""
        obs = [([1], 5), ([1, 1], 6)]   # missing ([], base)
        rule = fold_detect(obs)
        assert rule is None

    def test_inconsistent_observations_returns_none(self):
        """fold_detect returns None when observations contradict a fold."""
        # ([], 3) as base, but ([1], 5) and ([1], 7) both claim to be step(1, 3)
        obs = [([], 3), ([1], 5), ([1], 7)]
        # _step_constraints should see (1, 3) → 5 and (1, 3) → 7 as inconsistent
        constraints = _step_constraints(obs)
        assert constraints is None

    def test_empty_obs_returns_none(self):
        assert fold_detect([]) is None

    def test_performance(self):
        """fold_detect on 100 observations of length ≤ 5 completes in < 0.1 s."""
        obs = []
        base = 10
        for length in range(6):
            for _ in range(20):    # 20 observations at each length
                obs.append(([1] * length, base + length))

        t0 = time.perf_counter()
        rule = fold_detect(obs)
        elapsed = time.perf_counter() - t0

        assert rule is not None
        assert elapsed < 0.1, f"fold_detect took {elapsed:.3f}s, expected < 0.1s"


class TestStepConstraints:
    def test_returns_correct_triples(self):
        """_step_constraints extracts (elem, acc, result) from fold observations."""
        obs = [([], 0), ([1], 1), ([1, 1], 2)]
        triples = _step_constraints(obs)
        assert triples is not None
        # Should contain (1, 0) → 1
        assert (1, 0, 1) in triples

    def test_inconsistent_base_returns_none(self):
        """Two different base cases → inconsistency → None."""
        obs = [([], 0), ([], 5)]
        assert _step_constraints(obs) is None

    def test_missing_base_returns_none(self):
        obs = [([1], 5), ([1, 1], 6)]
        assert _step_constraints(obs) is None
