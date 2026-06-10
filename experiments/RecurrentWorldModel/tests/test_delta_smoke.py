"""Smoke + correctness tests for Stage 2 (ShiftSeq + Δ-encoding, continuous-input model).

Run:  ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests -q
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from tasks import ShiftSeq  # noqa: E402
from train_delta import DeltaConfig, run_delta  # noqa: E402


def test_shiftseq_target_is_total_change():
    task = ShiftSeq(length=8, n_deltas=4, seed=0)
    b = task.sample(64, 0.0, 100.0, rng=random.Random(0))
    last = task.seq_len - 1
    assert torch.all(b.loss_mask.sum(dim=1) == 1.0)
    for i in range(64):
        # total change = last absolute value - first (v0); equals the target
        change = round(b.abs_input[i, last].item() - b.abs_input[i, 0].item())
        assert b.target[i, last].item() == change
        # and equals the sum of the delta inputs
        assert b.target[i, last].item() == int(round(b.delta_input[i].sum().item()))


def test_delta_input_is_shift_invariant_but_absolute_is_not():
    task = ShiftSeq(length=8, n_deltas=4, seed=0)
    # same rng seed, same deltas -> delta inputs + targets identical across v0 ranges,
    # absolute inputs differ by the shift
    a = task.sample(32, 0.0, 0.0, rng=random.Random(7))      # v0 = 0
    b = task.sample(32, 1000.0, 1000.0, rng=random.Random(7))  # v0 = 1000
    assert torch.allclose(a.delta_input, b.delta_input)        # deltas don't see the shift
    assert torch.equal(a.target, b.target)                     # target is shift-invariant
    assert not torch.allclose(a.abs_input, b.abs_input)        # absolutes shift by 1000
    assert torch.allclose(b.abs_input - a.abs_input, torch.full_like(a.abs_input, 1000.0))


def test_continuous_input_model_runs():
    task = ShiftSeq(seed=0)
    m = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.target_classes, dim=32, n_layers=2,
                                               max_seq=task.seq_len, pos_mode="pope",
                                               continuous_input=True))
    assert m.embed is None and m.input_proj is not None and m.head_proj is not None
    b = task.sample(4, 0.0, 100.0, rng=random.Random(0))
    out = m(b.abs_input)                       # real-valued input
    assert out.shape == (4, task.seq_len, task.target_classes)


def test_run_delta_smoke():
    out = run_delta(DeltaConfig(smoke=True))
    for arm in ("absolute", "delta"):
        assert arm in out["arms"] and "acc_ood" in out["arms"][arm]["final"]
