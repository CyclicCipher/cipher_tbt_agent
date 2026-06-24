"""Smoke + correctness tests for Step 4 (continual learning: ShiftSeq scale-shift + harness).

Run: ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests/test_continual_smoke.py -q
"""

from __future__ import annotations

import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import PartitionedModel  # noqa: E402
from tasks import ShiftSeq  # noqa: E402
from train_continual import ARMS, ContinualConfig, run_continual  # noqa: E402


def test_offset_shift_is_delta_invisible():
    """Two v0 regimes (same delta_step) -> IDENTICAL delta inputs and targets; absolute differs.
    This is *why* delta should resist forgetting under the offset shift."""
    task = ShiftSeq(length=8, n_deltas=4)
    a = task.sample(32, 0.0, 0.0, rng=random.Random(7), delta_step=1)
    b = task.sample(32, 1000.0, 1000.0, rng=random.Random(7), delta_step=1)
    assert torch.allclose(a.delta_input, b.delta_input)      # delta sees no shift
    assert torch.equal(a.target, b.target)
    assert not torch.allclose(a.abs_input, b.abs_input)      # absolute shifts by 1000


def test_scale_shift_is_delta_visible():
    """Doubling the increments DOES change the delta input -> delta cannot annihilate it
    (the control that keeps the experiment honest)."""
    task = ShiftSeq(length=8, n_deltas=4, max_step=2)
    a = task.sample(64, 0.0, 100.0, rng=random.Random(3), delta_step=1)
    b = task.sample(64, 0.0, 100.0, rng=random.Random(3), delta_step=2)
    # same v0 sequence/randomness, but B's increments are 2x -> different delta inputs
    assert not torch.allclose(a.delta_input, b.delta_input)
    # B's deltas are even; its total-change targets are larger on average
    assert b.target.float().sum() > a.target.float().sum()


def test_max_step_sizes_the_head_and_default_is_unchanged():
    assert ShiftSeq(8, 4).target_classes == (4 - 1) * 8 + 1          # default unchanged (Stage 2)
    assert ShiftSeq(8, 4, max_step=2).target_classes == (4 - 1) * 8 * 2 + 1


def test_partitioned_allocate_freezes_old_and_routes():
    m = PartitionedModel(lambda: nn.Linear(4, 3), mode="oracle")
    old = m.experts[0]
    assert m.n_experts() == 1
    new = m.allocate()
    assert m.n_experts() == 2 and m.active == 1
    assert all(not p.requires_grad for p in old.parameters())       # old expert frozen
    assert all(p.requires_grad for p in new.parameters())           # new expert trainable
    x = torch.randn(2, 4)
    assert torch.equal(m(x, expert=0), old(x))                      # explicit routing
    assert torch.equal(m(x), new(x))                                # active = new


def test_partitioned_surprise_fires_on_spike_only():
    m = PartitionedModel(lambda: nn.Linear(2, 2), mode="surprise", k=4.0, cooldown=5)
    for _ in range(20):
        assert m.observe(0.10) is None                              # stable -> no allocation
    assert m.observe(5.0) is not None                               # spike -> allocate
    assert m.n_experts() == 2


def test_partitioned_single_never_allocates():
    m = PartitionedModel(lambda: nn.Linear(2, 2), mode="single")
    for v in (0.1, 9.0, 0.1, 12.0):
        assert m.observe(v) is None
    assert m.n_experts() == 1


def test_run_continual_smoke_reports_forgetting_and_experts():
    out = run_continual(ContinualConfig(smoke=True))
    assert len(ARMS) == 12                                          # shift x encoding x mode
    for arm in ARMS:
        r = out["arms"][arm]
        for k in ("acc_A_end_of_A", "acc_A_final", "forgetting", "acc_B_final", "n_experts"):
            assert k in r
        assert any(h["phase"] == "A" for h in r["history"])
        assert any(h["phase"] == "B" for h in r["history"])
    # oracle always partitions; single never does
    assert out["arms"]["offset_absolute_oracle"]["n_experts"] == 2
    assert out["arms"]["offset_absolute_single"]["n_experts"] == 1
