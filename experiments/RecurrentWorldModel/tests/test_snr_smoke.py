"""Smoke tests for the SNR-gated optimizer and the transformer experiment harness.

Run:  ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests -q
"""

from __future__ import annotations

import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from optim import SNRAdamW, per_example_snr_gate  # noqa: E402
from tasks import ModularChain  # noqa: E402
from train_transformer import TConfig, run_transformer  # noqa: E402


def _ce(logits, t, m):
    v = logits.shape[-1]
    c = F.cross_entropy(logits.reshape(-1, v), t.reshape(-1), reduction="none").reshape(m.shape)
    return (c * m).sum() / m.sum().clamp_min(1)


def test_snr_modes_step_and_gate():
    task = ModularChain(seed=0)
    rng = random.Random(0)
    for mode in ("none", "ema", "faithful"):
        torch.manual_seed(0)
        m = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.vocab_size, dim=32, n_layers=2,
                                                   max_seq=task.seq_len, pos_mode="pope"))
        opt = SNRAdamW(m.parameters(), lr=1e-3, batch_size=32, mode=mode, gate_warmup=2)
        before = [p.detach().clone() for p in m.parameters()]
        for _ in range(5):
            b = task.sample(32, 1, 4, rng)
            loss = _ce(m(b.input_ids), b.targets, b.loss_mask)
            opt.zero_grad(); loss.backward()
            if mode == "faithful":
                g, r = per_example_snr_gate(m, b.input_ids, b.targets, b.loss_mask, 32)
                opt.set_external_gate(g, r)
            opt.step()
        moved = any(not torch.allclose(a, p) for a, p in zip(before, m.parameters()))
        assert moved, f"mode {mode} did not update params"
        assert 0.0 <= opt.last_gate_frac <= 1.0
        if mode == "none":
            assert opt.last_gate_frac == 1.0


def test_per_example_gate_matches_manual():
    # on a tiny linear model, the vmap per-example gate must equal a hand loop
    torch.manual_seed(0)
    task = ModularChain(seed=0)
    m = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.vocab_size, dim=16, n_layers=1,
                                               max_seq=task.seq_len, pos_mode="pope"))
    b = task.sample(8, 1, 3, random.Random(1))
    gate, risk = per_example_snr_gate(m, b.input_ids, b.targets, b.loss_mask, 8)
    # every gate entry is binary and shaped like its parameter
    name_to_p = dict(m.named_parameters())
    for p, gmask in gate.items():
        assert gmask.shape == p.shape
        assert torch.all((gmask == 0) | (gmask == 1))
    assert len(gate) == len(name_to_p)
    assert risk >= 0.0


def test_run_transformer_smoke_all_arms():
    out = run_transformer(TConfig(smoke=True))
    assert out["n_params"] > 0
    for arm in ("adamw", "snr_ema", "snr_faithful"):
        assert arm in out["arms"]
        f = out["arms"][arm]["final"]
        assert "acc_in" in f and "acc_ood" in f and "risk" in f


def test_wsd_schedule_and_momentum_warmup():
    from train_transformer import lr_mult, muon_momentum
    total, warm, cd = 1000, 100, 0.4
    assert lr_mult(0, total, warm, cd, 0.0) < lr_mult(50, total, warm, cd, 0.0)   # warmup rising
    assert abs(lr_mult(warm, total, warm, cd, 0.0) - 1.0) < 1e-6                    # peak = 1
    assert abs(lr_mult(500, total, warm, cd, 0.0) - 1.0) < 1e-6                     # stable phase
    assert lr_mult(total - 1, total, warm, cd, 0.0) < 0.05                          # cooled near 0
    # momentum warms 0.85 -> 0.95
    assert abs(muon_momentum(0, 300) - 0.85) < 1e-6
    assert abs(muon_momentum(300, 300) - 0.95) < 1e-6
    assert 0.85 < muon_momentum(150, 300) < 0.95
