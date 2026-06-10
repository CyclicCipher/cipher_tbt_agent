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


def test_pope_and_rope_attention_run_and_extrapolate():
    from core.block import SettlingBlock, SettlingBlockConfig
    for pos in ("pope", "rope"):
        block = SettlingBlock(SettlingBlockConfig(dim=32, n_heads=4, pos_enc=pos, max_seq=16))
        # no learned position params, so a longer sequence than any "training" length
        # still runs (the extrapolation property we want)
        for t in (4, 12):
            h = torch.randn(2, t, 32)
            out = block(h, torch.zeros_like(h))
            assert out.shape == (2, t, 32) and torch.isfinite(out).all()


def test_pope_decouples_content_from_position():
    """PoPE's claim: content lives in magnitude (position-independent), and the
    attention score depends only on RELATIVE position. Verify both directly."""
    from core.block import PolarPositionalEmbedding
    import torch.nn.functional as F
    torch.manual_seed(0)
    pe = PolarPositionalEmbedding(head_dim=8, max_seq=64)
    x = torch.randn(1, 1, 5, 8)  # (b, h, t, head_dim) content features

    # (a) content = magnitude is the SAME regardless of position: encode(x) at any
    #     position has per-feature magnitude sqrt(cos^2+sin^2)*softplus(x) = softplus(x).
    enc = pe.encode(x, is_key=False)              # (1,1,5,16)
    mag = (enc[..., :8] ** 2 + enc[..., 8:] ** 2).sqrt()
    assert torch.allclose(mag, F.softplus(x), atol=1e-5)

    # (b) score is translation-invariant: same content, shift both q,k coords by
    #     the same delta -> identical score (depends only on s - t).
    pe.delta.data.zero_()  # isolate position from the learnable phase bias
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    def score(qpos, kpos):
        qe = pe.encode(q, is_key=False, coord=torch.tensor([[float(qpos)]]))
        ke = pe.encode(k, is_key=True, coord=torch.tensor([[float(kpos)]]))
        return (qe * ke).sum().item()

    assert abs(score(2, 5) - score(4, 7)) < 1e-4   # both have s - t = 3
    # continuous coordinates: the score tracks the REAL gap, not token count
    assert abs(score(0.0, 3.0) - score(10.0, 13.0)) < 1e-4   # both gap = 3.0
    assert abs(score(0, 1) - score(0, 3)) > 1e-3             # different gaps differ


def test_pope_coord_none_equals_integer_positions():
    from core.block import PolarPositionalEmbedding
    pe = PolarPositionalEmbedding(head_dim=8, max_seq=16)
    x = torch.randn(2, 1, 5, 8)
    default = pe.encode(x, is_key=False)                       # coord=None -> integers
    explicit = pe.encode(x, is_key=False,
                         coord=torch.arange(5).float()[None].expand(2, 5))
    assert torch.allclose(default, explicit, atol=1e-6)


def test_warm_start_modes_run_and_converge_faster():
    task = ModularChain(seed=0)
    rng = random.Random(0)
    b = task.sample(8, 1, 4, rng)
    iters = {}
    for mode in ("zeros", "input", "proposal"):
        m = SettlingLM(SettlingLMConfig(vocab_size=task.vocab_size, dim=32, n_heads=4,
                                        max_seq=task.seq_len, pos_mode="pope", warm_start=mode))
        final, seg, infos = m(b.input_ids)
        assert final.shape == (8, task.seq_len, task.vocab_size)
        with torch.no_grad():
            _, info = m.deq(m._inject(b.input_ids), h0=m._warm_h0(m._inject(b.input_ids)))
        iters[mode] = info.iters
    # a proposal warm-start should never need MORE iterations than a cold start
    assert iters["proposal"] <= iters["zeros"]


def test_residual_gate_makes_forward_converge_fast():
    # the gate's purpose: a small init makes the settling map strongly contractive
    from core.deq import DEQConfig
    task = ModularChain(seed=0)
    rng = random.Random(0)
    x_ids = task.sample(8, 1, 4, rng).input_ids
    iters = {}
    for gate in (False, True):
        torch.manual_seed(0)
        m = SettlingLM(SettlingLMConfig(vocab_size=task.vocab_size, dim=64, n_heads=4,
                                        max_seq=task.seq_len, pos_mode="pope",
                                        residual_gate=gate, gate_init=0.1,
                                        deq=DEQConfig(max_iter=100, tol=1e-3, state_norm=True)))
        with torch.no_grad():
            _, info = m.deq(m._inject(x_ids))
        iters[gate] = info.iters
    # gated converges in far fewer iterations than ungated (which usually hits the cap)
    assert iters[True] < iters[False]


def test_pope_with_qknorm_still_runs():
    # PoPE now applies QK-Norm to raw features before softplus -- make sure the path runs
    from core.block import SettlingBlock, SettlingBlockConfig
    block = SettlingBlock(SettlingBlockConfig(dim=32, n_heads=4, pos_enc="pope", qk_norm=True))
    out = block(torch.randn(2, 6, 32), torch.zeros(2, 6, 32))
    assert out.shape == (2, 6, 32) and torch.isfinite(out).all()


def test_pope_model_has_no_absolute_pos_params():
    task = ModularChain(seed=0)
    pope = SettlingLM(SettlingLMConfig(vocab_size=task.vocab_size, dim=32, n_heads=4,
                                       max_seq=task.seq_len, pos_mode="pope"))
    learned = SettlingLM(SettlingLMConfig(vocab_size=task.vocab_size, dim=32, n_heads=4,
                                          max_seq=task.seq_len, pos_mode="learned"))
    assert pope.pos is None and learned.pos is not None
    # pope carries position in attention -> no max_seq x dim absolute table
    assert count_parameters(pope) < count_parameters(learned)
