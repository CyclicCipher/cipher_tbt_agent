"""Smoke + correctness tests for the temporal fork (EventStream + continuous-time PoPE).

Run:  ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests -q
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from tasks import EventStream  # noqa: E402
from train_temporal import TempConfig, run_temporal  # noqa: E402


def test_eventstream_target_is_correct_decay():
    task = EventStream(n_levels=6, n_events=8, max_gap=3, noise_frac=0.3, seed=0)
    b = task.sample(64, rng=random.Random(0))
    # exactly one scored position (the query) per example
    assert torch.all(b.loss_mask.sum(dim=1) == 1.0)
    # timestamps are strictly increasing (monotone time) and != position index in general
    assert torch.all(b.timestamps[:, 1:] > b.timestamps[:, :-1])
    # recompute the target from the events and check it matches (the decay rule)
    for i in range(16):
        q = task.n_events
        last_k, last_tau = None, None
        for j in range(task.n_events):
            tok = b.input_ids[i, j].item()
            if 2 <= tok < 2 + task.V:          # a VAL(k) event
                last_k, last_tau = tok - 2, b.timestamps[i, j].item()
        elapsed = b.timestamps[i, q].item() - last_tau if last_k is not None else 0
        cur = 0 if last_k is None else max(0, last_k - int(elapsed // task.decay_per))
        assert b.targets[i, q].item() == 2 + cur


def test_continuous_pope_uses_timestamps():
    # feeding timestamps as the PoPE coordinate changes the output (vs integer positions)
    task = EventStream(seed=0)
    torch.manual_seed(0)
    m = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.vocab_size, dim=32, n_layers=2,
                                               max_seq=task.seq_len, pos_mode="pope"))
    b = task.sample(4, rng=random.Random(0))
    out_int = m(b.input_ids)                         # integer positions
    out_cont = m(b.input_ids, coord=b.timestamps)    # continuous-time PoPE
    assert out_int.shape == out_cont.shape
    assert not torch.allclose(out_int, out_cont, atol=1e-4)


def test_time_input_arm_adds_params_and_runs():
    task = EventStream(seed=0)
    base = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.vocab_size, dim=32, n_layers=2,
                                                  max_seq=task.seq_len, pos_mode="pope"))
    ti = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.vocab_size, dim=32, n_layers=2,
                                                max_seq=task.seq_len, pos_mode="pope", time_input=True))
    from core.model import count_parameters
    assert count_parameters(ti) > count_parameters(base)   # the time projection
    b = task.sample(4, rng=random.Random(0))
    assert ti(b.input_ids, time_feat=b.timestamps).shape[-1] == task.vocab_size


def test_run_temporal_smoke_all_arms():
    out = run_temporal(TempConfig(smoke=True))
    for arm in ("integer", "time_input", "continuous"):
        assert arm in out["arms"] and "acc_ood" in out["arms"][arm]["final"]
