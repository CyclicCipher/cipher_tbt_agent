"""Smoke tests for the Stage 0 experiment pipeline. No real training -- shapes,
gradient flow, the param-matched baseline, the task generator, and a 2-step CPU
run of the whole harness.

Run:  ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests -q
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import FixedDepthConfig, FixedDepthTransformer, matched_baseline  # noqa: E402
from core.model import SettlingLM, SettlingLMConfig, count_parameters  # noqa: E402
from tasks import ModularChain  # noqa: E402
from train_stage0 import Stage0Config, lm_loss, run_stage0  # noqa: E402


def test_modular_chain_shapes_and_correctness():
    task = ModularChain(modulus=7, n_ops=8, max_len=6, seed=0)
    rng = random.Random(0)
    b = task.sample(32, 1, 4, rng)
    assert b.input_ids.shape == (32, task.seq_len)
    assert b.targets.shape == b.input_ids.shape
    # exactly one scored position per example (the EQ position)
    assert torch.all(b.loss_mask.sum(dim=1) == 1.0)
    # the target at the EQ position must be the correct answer token
    for i in range(32):
        eq_pos = int(b.loss_mask[i].argmax())
        assert b.input_ids[i, eq_pos].item() == 1  # EQ
        assert b.targets[i, eq_pos].item() == b.input_ids[i, eq_pos + 1].item()


def test_settling_lm_forward_and_grad():
    task = ModularChain(seed=0)
    cfg = SettlingLMConfig(vocab_size=task.vocab_size, dim=32, n_heads=4, max_seq=task.seq_len,
                           n_supervision_segments=2)
    model = SettlingLM(cfg)
    rng = random.Random(0)
    b = task.sample(8, 1, 4, rng)
    final, seg_logits, infos = model(b.input_ids)
    assert final.shape == (8, task.seq_len, task.vocab_size)
    assert len(seg_logits) == 2 and len(infos) == 2
    loss = lm_loss(seg_logits, b.targets, b.loss_mask)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_matched_baseline_close_in_params():
    task = ModularChain(seed=0)
    ref = SettlingLM(SettlingLMConfig(vocab_size=task.vocab_size, dim=64, n_heads=4,
                                      max_seq=task.seq_len))
    target = count_parameters(ref)
    model, cfg, n = matched_baseline(target, vocab_size=task.vocab_size, n_heads=4,
                                     n_layers=4, max_seq=task.seq_len)
    assert isinstance(model, FixedDepthTransformer)
    # within 15% of the settling model's parameter budget
    assert abs(n - target) / target < 0.15
    out = model(task.sample(4, 1, 3, random.Random(0)).input_ids)
    assert out.shape == (4, task.seq_len, task.vocab_size)


def test_fixed_depth_forward():
    task = ModularChain(seed=0)
    m = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.vocab_size, dim=32, n_layers=3,
                                               max_seq=task.seq_len))
    out = m(task.sample(4, 1, 3, random.Random(0)).input_ids)
    assert out.shape == (4, task.seq_len, task.vocab_size)


def test_run_stage0_smoke():
    out = run_stage0(Stage0Config(smoke=True, baseline=True))
    assert out["n_params"] > 0
    assert out["history"], "no eval recorded"
    final = out["final"]
    assert "acc_in" in final and "acc_ood" in final
    assert 0.0 <= final["acc_in"] <= 1.0
    assert "convergence_rate" in final["convergence"]
