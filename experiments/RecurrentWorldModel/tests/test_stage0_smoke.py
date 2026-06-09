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
    s = out["settling"]
    assert s["n_params"] > 0 and s["history"], "no settling eval recorded"
    assert "acc_in" in s["final"] and "acc_ood" in s["final"]
    assert 0.0 <= s["final"]["acc_in"] <= 1.0
    assert "convergence_rate" in s["final"]["convergence"]
    # the baseline arm is actually trained + evaluated now (the gate)
    b = out["baseline"]
    assert b["n_params"] > 0 and b["history"]
    assert "acc_ood" in b["final"]


def test_bptt_grad_flows_through_all_steps():
    from core.deq import DEQConfig, DEQFixedPoint
    from core.block import SettlingBlock, SettlingBlockConfig
    block = SettlingBlock(SettlingBlockConfig(dim=32, n_heads=4))
    deq = DEQFixedPoint(block, DEQConfig(grad_mode="bptt", bptt_iters=6))
    x = torch.randn(2, 8, 32, requires_grad=True)
    h, info = deq(x)
    assert info.iters == 6
    h.pow(2).mean().backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_state_norm_bounds_the_state():
    from core.deq import DEQConfig, DEQFixedPoint
    from core.block import SettlingBlock, SettlingBlockConfig
    block = SettlingBlock(SettlingBlockConfig(dim=64, n_heads=4))
    x = torch.randn(2, 8, 64)
    deq = DEQFixedPoint(block, DEQConfig(state_norm=True, max_iter=50))
    with torch.no_grad():
        h, _ = deq(x)
    # unit RMS per element => mean(h^2) ~ 1 along the feature dim
    rms = h.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_run_stage0_bptt_smoke():
    out = run_stage0(Stage0Config(smoke=True, grad_mode="bptt", bptt_iters=6, state_norm=True))
    assert out["settling"]["history"] and "acc_in" in out["settling"]["final"]


def test_ift_gradient_matches_bptt_reference():
    """The implicit (IFT) gradient must equal the full-BPTT gradient AT the fixed
    point. state_norm makes the map contractive so both the forward and the
    adjoint solve converge."""
    from core.deq import DEQConfig, DEQFixedPoint
    from core.block import SettlingBlock, SettlingBlockConfig
    bcfg = SettlingBlockConfig(dim=32, n_heads=4)
    x = torch.randn(2, 6, 32)
    target = torch.randn(2, 6, 32)

    def grads(mode, **kw):
        torch.manual_seed(0)
        block = SettlingBlock(bcfg)
        deq = DEQFixedPoint(block, DEQConfig(grad_mode=mode, state_norm=True,
                                             max_iter=150, tol=1e-6, **kw))
        h, info = deq(x)
        (h - target).pow(2).mean().backward()
        g = torch.cat([p.grad.flatten() for p in block.parameters()])
        return g, info

    g_ift, info = grads("ift")
    g_ref, _ = grads("bptt", bptt_iters=150)  # full BPTT to equilibrium = reference
    assert info.converged
    cos = torch.nn.functional.cosine_similarity(g_ift, g_ref, dim=0).item()
    assert cos > 0.999, f"IFT gradient diverges from reference (cos={cos:.4f})"


def test_rope_attention_runs_and_extrapolates_positions():
    from core.block import SettlingBlock, SettlingBlockConfig
    block = SettlingBlock(SettlingBlockConfig(dim=32, n_heads=4, rope=True, max_seq=16))
    # rope has no learned position params, so a longer sequence than any "training"
    # length still runs (the extrapolation property we want)
    for t in (4, 12):
        h = torch.randn(2, t, 32)
        out = block(h, torch.zeros_like(h))
        assert out.shape == (2, t, 32) and torch.isfinite(out).all()


def test_rope_model_has_no_absolute_pos_params():
    task = ModularChain(seed=0)
    rope = SettlingLM(SettlingLMConfig(vocab_size=task.vocab_size, dim=32, n_heads=4,
                                       max_seq=task.seq_len, use_rope=True))
    abso = SettlingLM(SettlingLMConfig(vocab_size=task.vocab_size, dim=32, n_heads=4,
                                       max_seq=task.seq_len, use_rope=False))
    assert rope.pos is None and abso.pos is not None
    # rope model has fewer params (no max_seq x dim position table)
    assert count_parameters(rope) < count_parameters(abso)
