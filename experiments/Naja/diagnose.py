#!/usr/bin/env python3
"""
Diagnostic script for Naja architecture.

Isolates each component, measures timing, verifies correctness.
Runs fast (small tensors, few iterations) — safe on any GPU.

Usage:
    python experiments/Naja/diagnose.py
    python experiments/Naja/diagnose.py --device cpu   # force CPU
"""

import argparse
import sys
import os
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.Naja.naja import (
    NajaConfig, NajaLM, NajaMixer, NajaBlock,
    delta_recurrence, delta_recurrence_chunkwise, delta_recurrence_wy,
    apply_pope, apply_pope_perp,
)


def fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds*1e6:.0f}us"
    elif seconds < 1.0:
        return f"{seconds*1e3:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def time_fn(fn, warmup=2, repeats=5, sync_cuda=False, device=None):
    """Time a function, return median time in seconds."""
    for _ in range(warmup):
        fn()
    if sync_cuda and device and device.type == 'cuda':
        torch.cuda.synchronize(device)

    times = []
    for _ in range(repeats):
        if sync_cuda and device and device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        result = fn()
        if sync_cuda and device and device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], result  # median


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return passed


# ---------------------------------------------------------------------------
# Test 1: Forward pass timing breakdown
# ---------------------------------------------------------------------------

def test_timing(device, batch=8, seq_len=64):
    section("TIMING BREAKDOWN")

    config = NajaConfig(
        d_model=128, d_state=64, n_layer=4, headdim=64,
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        use_chunkwise=False, chunk_size=64,
    )
    model = NajaLM(config, vocab_size=16).to(device)
    x = torch.randint(0, 16, (batch, seq_len), device=device)

    use_cuda = device.type == 'cuda'

    # Full forward
    def fwd():
        return model(x)
    t_fwd, _ = time_fn(fwd, warmup=2, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Forward (full model):    {fmt_time(t_fwd)}")

    # Forward + backward
    def fwd_bwd():
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 16), x[:, 1:].reshape(-1))
        loss.backward()
        model.zero_grad()
        return loss.item()
    t_fwdbwd, _ = time_fn(fwd_bwd, warmup=1, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Forward + backward:      {fmt_time(t_fwdbwd)}")

    # Same with chunkwise
    config_chunk = NajaConfig(
        d_model=128, d_state=64, n_layer=4, headdim=64,
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        use_chunkwise=True, chunk_size=16,
    )
    model_chunk = NajaLM(config_chunk, vocab_size=16).to(device)

    def fwd_chunk():
        return model_chunk(x)
    t_fwd_c, _ = time_fn(fwd_chunk, warmup=2, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Forward (chunkwise=16):  {fmt_time(t_fwd_c)}")

    def fwd_bwd_chunk():
        logits = model_chunk(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 16), x[:, 1:].reshape(-1))
        loss.backward()
        model_chunk.zero_grad()
        return loss.item()
    t_fwdbwd_c, _ = time_fn(fwd_bwd_chunk, warmup=1, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Fwd+bwd (chunkwise=16):  {fmt_time(t_fwdbwd_c)}")

    # WY chunkwise (Phase 5a — real parallelism)
    config_wy = NajaConfig(
        d_model=128, d_state=64, n_layer=4, headdim=64,
        use_delta_rule=True, use_pope_perp=False, per_channel_decay=True,
        use_wy_chunkwise=True, chunk_size=seq_len,
    )
    model_wy = NajaLM(config_wy, vocab_size=16).to(device)

    def fwd_wy():
        return model_wy(x)
    t_fwd_wy, _ = time_fn(fwd_wy, warmup=2, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Forward (WY chunkwise):  {fmt_time(t_fwd_wy)}")

    def fwd_bwd_wy():
        logits = model_wy(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 16), x[:, 1:].reshape(-1))
        loss.backward()
        model_wy.zero_grad()
        return loss.item()
    t_fwdbwd_wy, _ = time_fn(fwd_bwd_wy, warmup=1, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Fwd+bwd (WY chunkwise):  {fmt_time(t_fwdbwd_wy)}")

    # Estimate epoch time
    n_train = 5000
    n_batches = n_train // batch
    est_epoch_chunk = t_fwdbwd_c * n_batches
    est_epoch_wy = t_fwdbwd_wy * n_batches
    print(f"\n  Estimated epoch time (grad-checkpoint, {n_batches} batches): {fmt_time(est_epoch_chunk)}")
    print(f"  Estimated epoch time (WY chunkwise, {n_batches} batches): {fmt_time(est_epoch_wy)}")
    if est_epoch_wy < est_epoch_chunk:
        speedup = est_epoch_chunk / max(est_epoch_wy, 1e-9)
        print(f"  WY speedup vs grad-checkpoint: {speedup:.1f}x")
    if est_epoch_chunk > 60:
        print(f"  WARNING: >1 min/epoch — this is why training appears to hang!")
        print(f"  The first epoch line won't print until all {n_batches} batches complete.")

    return t_fwd, t_fwdbwd, t_fwdbwd_c, t_fwdbwd_wy


# ---------------------------------------------------------------------------
# Test 2: Recurrence isolation
# ---------------------------------------------------------------------------

def test_recurrence_isolation(device, batch=8, seq_len=64):
    section("RECURRENCE ISOLATION")

    nheads, headdim, d_state, r = 4, 64, 64, 1

    # Create dummy recurrence inputs
    x_write = torch.randn(batch, seq_len, nheads, headdim, r, device=device)
    x_write_prev = torch.randn_like(x_write)
    B1 = torch.randn(batch, seq_len, r, d_state, device=device)
    B1_prev = torch.randn_like(B1)
    B2 = torch.randn_like(B1)
    C = torch.randn_like(B1)
    alpha = torch.sigmoid(torch.randn(batch, seq_len, nheads, d_state, device=device))
    beta1 = torch.sigmoid(torch.randn(batch, seq_len, nheads, 1, device=device))
    beta2 = torch.sigmoid(torch.randn_like(beta1))
    lam = torch.sigmoid(torch.randn(batch, seq_len, 1, 1, device=device))

    use_cuda = device.type == 'cuda'

    # Naive sequential
    def run_naive():
        return delta_recurrence(
            x_write, x_write_prev, B1, B1_prev, B2, C, alpha,
            beta1, beta2, lam, use_trapezoidal=True, use_delta=True)
    t_naive, _ = time_fn(run_naive, warmup=1, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Naive sequential (L={seq_len}):      {fmt_time(t_naive)}")

    # Chunkwise
    def run_chunk():
        return delta_recurrence_chunkwise(
            x_write, x_write_prev, B1, B1_prev, B2, C, alpha,
            beta1, beta2, lam, use_trapezoidal=True, use_delta=True,
            chunk_size=16)
    t_chunk, _ = time_fn(run_chunk, warmup=1, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Chunkwise (chunk=16):                {fmt_time(t_chunk)}")

    # WY chunkwise
    def run_wy():
        return delta_recurrence_wy(
            x_write, x_write_prev, B1, B1_prev, None, C, alpha,
            beta1, None, lam, use_trapezoidal=True, use_delta=True,
            chunk_size=seq_len)
    t_wy, _ = time_fn(run_wy, warmup=1, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  WY chunkwise (chunk={seq_len}):          {fmt_time(t_wy)}")

    # Without delta rule
    def run_no_delta():
        return delta_recurrence(
            x_write, x_write_prev, B1, B1_prev, None, C, alpha,
            None, None, lam, use_trapezoidal=True, use_delta=False)
    t_no_delta, _ = time_fn(run_no_delta, warmup=1, repeats=3, sync_cuda=use_cuda, device=device)
    print(f"  Naive no-delta (L={seq_len}):        {fmt_time(t_no_delta)}")

    # Percentage in recurrence: rough estimate
    # (Full model has 4 layers × 1 recurrence each, plus projections)
    print(f"\n  Recurrence is the dominant cost in the sequential loop.")
    print(f"  Each layer runs {seq_len} sequential timesteps with ~20 CUDA kernel launches each.")
    total_kernels = seq_len * 20 * 4  # 4 layers
    print(f"  Estimated CUDA kernels per batch (4 layers): ~{total_kernels}")
    print(f"  At ~30us overhead each: ~{total_kernels * 30 / 1000:.0f}ms of launch overhead alone")


# ---------------------------------------------------------------------------
# Test 3: Architecture correctness
# ---------------------------------------------------------------------------

def test_architecture(device, batch=4, seq_len=32):
    section("ARCHITECTURE CORRECTNESS")

    vocab_size = 16
    all_pass = True

    # --- 3a: Forward produces valid output ---
    config = NajaConfig(
        d_model=128, d_state=64, n_layer=4, headdim=64,
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
    )
    model = NajaLM(config, vocab_size).to(device)
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    with torch.no_grad():
        logits = model(x)

    all_pass &= check("Output shape",
        logits.shape == (batch, seq_len, vocab_size),
        f"got {logits.shape}")

    all_pass &= check("No NaN in output",
        not torch.isnan(logits).any().item())

    all_pass &= check("No Inf in output",
        not torch.isinf(logits).any().item())

    # Check logits are not all identical (model isn't collapsed)
    logit_std = logits.std(dim=-1).mean().item()
    all_pass &= check("Output diversity (std > 0.01)",
        logit_std > 0.01,
        f"mean logit std={logit_std:.4f}")

    # --- 3b: Gradient flow ---
    logits = model(x)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size), x[:, 1:].reshape(-1))
    loss.backward()

    grad_norms = {}
    any_zero = False
    any_nan = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms[name] = gn
            if gn == 0:
                any_zero = True
            if torch.isnan(p.grad).any():
                any_nan = True

    all_pass &= check("Gradients exist",
        len(grad_norms) > 0,
        f"{len(grad_norms)} params with grad")

    all_pass &= check("No NaN gradients", not any_nan)

    zero_grads = [n for n, g in grad_norms.items() if g == 0]
    all_pass &= check("No zero gradients",
        not any_zero,
        f"zero: {zero_grads}" if any_zero else f"all {len(grad_norms)} non-zero")

    model.zero_grad()

    # --- 3c: Chunkwise matches naive ---
    config_plain = NajaConfig(
        d_model=128, d_state=64, n_layer=1, headdim=64,
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        use_chunkwise=False,
    )
    config_chunk = NajaConfig(
        d_model=128, d_state=64, n_layer=1, headdim=64,
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        use_chunkwise=True, chunk_size=16,
    )

    m_plain = NajaLM(config_plain, vocab_size).to(device)
    m_chunk = NajaLM(config_chunk, vocab_size).to(device)
    m_chunk.load_state_dict(m_plain.state_dict())  # same weights

    with torch.no_grad():
        out_plain = m_plain(x)
        out_chunk = m_chunk(x)

    diff = (out_plain - out_chunk).abs().max().item()
    all_pass &= check("Chunkwise == naive (max diff < 1e-4)",
        diff < 1e-4,
        f"max_diff={diff:.2e}")

    return all_pass


# ---------------------------------------------------------------------------
# Test 3b: WY chunkwise correctness (compare to naive sequential)
# ---------------------------------------------------------------------------

def test_wy_correctness(device, batch=4, seq_len=32):
    section("WY CHUNKWISE CORRECTNESS")

    all_pass = True
    vocab_size = 16

    # Test WY matches naive on a single-layer model without B2 (Phase 5a limitation)
    # We use use_pope_perp=False since WY doesn't support B2 yet
    config_naive = NajaConfig(
        d_model=128, d_state=64, n_layer=1, headdim=64,
        use_delta_rule=True, use_pope_perp=False, per_channel_decay=True,
        use_chunkwise=False, use_wy_chunkwise=False,
    )
    config_wy = NajaConfig(
        d_model=128, d_state=64, n_layer=1, headdim=64,
        use_delta_rule=True, use_pope_perp=False, per_channel_decay=True,
        use_chunkwise=False, use_wy_chunkwise=True, chunk_size=seq_len,
    )

    m_naive = NajaLM(config_naive, vocab_size).to(device)
    m_wy = NajaLM(config_wy, vocab_size).to(device)
    m_wy.load_state_dict(m_naive.state_dict())  # same weights

    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    with torch.no_grad():
        out_naive = m_naive(x)
        out_wy = m_wy(x)

    diff = (out_naive - out_wy).abs().max().item()
    mean_diff = (out_naive - out_wy).abs().mean().item()
    all_pass &= check("WY vs naive: max diff < 1e-2",
        diff < 1e-2,
        f"max_diff={diff:.2e}, mean_diff={mean_diff:.2e}")

    all_pass &= check("WY vs naive: max diff < 1e-3",
        diff < 1e-3,
        f"max_diff={diff:.2e}")

    all_pass &= check("WY vs naive: max diff < 1e-4",
        diff < 1e-4,
        f"max_diff={diff:.2e}")

    # Also test with chunk_size < seq_len (multiple chunks)
    if seq_len > 8:
        config_wy_multi = NajaConfig(
            d_model=128, d_state=64, n_layer=1, headdim=64,
            use_delta_rule=True, use_pope_perp=False, per_channel_decay=True,
            use_chunkwise=False, use_wy_chunkwise=True, chunk_size=8,
        )
        m_wy_multi = NajaLM(config_wy_multi, vocab_size).to(device)
        m_wy_multi.load_state_dict(m_naive.state_dict())

        with torch.no_grad():
            out_wy_multi = m_wy_multi(x)

        diff_multi = (out_naive - out_wy_multi).abs().max().item()
        all_pass &= check("WY (multi-chunk, C=8) vs naive: max diff < 1e-2",
            diff_multi < 1e-2,
            f"max_diff={diff_multi:.2e}")

    # Test gradient flow through WY
    m_wy.zero_grad()
    logits_wy = m_wy(x)
    loss_wy = F.cross_entropy(
        logits_wy[:, :-1].reshape(-1, vocab_size), x[:, 1:].reshape(-1))
    loss_wy.backward()

    n_grads = sum(1 for p in m_wy.parameters() if p.grad is not None and p.grad.norm() > 0)
    all_pass &= check("WY backward: gradients flow",
        n_grads > 0,
        f"{n_grads} params with non-zero grad")

    any_nan = any(torch.isnan(p.grad).any() for p in m_wy.parameters() if p.grad is not None)
    all_pass &= check("WY backward: no NaN gradients", not any_nan)

    m_wy.zero_grad()
    return all_pass


# ---------------------------------------------------------------------------
# Test 4: Delta rule effect
# ---------------------------------------------------------------------------

def test_delta_effect(device, batch=4, seq_len=32):
    section("DELTA RULE EFFECT")

    vocab_size = 16
    all_pass = True

    # Create two configs: with and without delta
    config_delta = NajaConfig(
        d_model=128, d_state=64, n_layer=1, headdim=64,
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
    )
    config_nodelta = NajaConfig(
        d_model=128, d_state=64, n_layer=1, headdim=64,
        use_delta_rule=False, use_pope_perp=False, per_channel_decay=True,
    )

    m_delta = NajaLM(config_delta, vocab_size).to(device)
    m_nodelta = NajaLM(config_nodelta, vocab_size).to(device)

    # They have different param counts because delta adds beta projections
    n_delta = sum(p.numel() for p in m_delta.parameters())
    n_nodelta = sum(p.numel() for p in m_nodelta.parameters())
    print(f"  Params with delta:    {n_delta:,}")
    print(f"  Params without delta: {n_nodelta:,}")
    all_pass &= check("Delta adds parameters",
        n_delta > n_nodelta,
        f"diff={n_delta - n_nodelta}")

    # Run same input through both — they should produce different outputs
    # (even after init, because delta model has beta gates that modify the recurrence)
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    with torch.no_grad():
        out_d = m_delta(x)
        out_nd = m_nodelta(x)

    # Both should produce valid output
    all_pass &= check("Delta model: no NaN",
        not torch.isnan(out_d).any().item())
    all_pass &= check("No-delta model: no NaN",
        not torch.isnan(out_nd).any().item())

    return all_pass


# ---------------------------------------------------------------------------
# Test 5: PoPE orthogonality
# ---------------------------------------------------------------------------

def test_pope_orthogonality(device):
    section("PoPE ORTHOGONALITY")

    batch, seq_len = 4, 32
    d_bc = 32  # d_state // 2
    all_pass = True

    x = torch.randn(batch, seq_len, d_bc, device=device)
    theta = torch.randn(batch, seq_len, d_bc, device=device)
    delta = torch.zeros(d_bc, device=device)

    B1 = apply_pope(x, theta, delta)       # (batch, seqlen, d_state)
    B2 = apply_pope_perp(x, theta, delta)  # (batch, seqlen, d_state)

    # B1 and B2 should be orthogonal: dot product ≈ 0
    dot = (B1 * B2).sum(dim=-1)  # (batch, seqlen)
    max_dot = dot.abs().max().item()
    mean_dot = dot.abs().mean().item()
    all_pass &= check("B1 · B2 ≈ 0 (orthogonal)",
        max_dot < 1e-5,
        f"max|dot|={max_dot:.2e}, mean|dot|={mean_dot:.2e}")

    # Both should have same magnitude (same mu)
    norm1 = B1.norm(dim=-1)
    norm2 = B2.norm(dim=-1)
    norm_diff = (norm1 - norm2).abs().max().item()
    all_pass &= check("||B1|| == ||B2|| (same magnitude)",
        norm_diff < 1e-5,
        f"max_diff={norm_diff:.2e}")

    return all_pass


# ---------------------------------------------------------------------------
# Test 6: Per-channel decay sanity
# ---------------------------------------------------------------------------

def test_decay(device):
    section("PER-CHANNEL DECAY")

    all_pass = True

    config = NajaConfig(
        d_model=128, d_state=64, n_layer=1, headdim=64,
        per_channel_decay=True,
    )
    mixer = NajaMixer(config).to(device)

    batch, seq_len = 4, 32
    u = torch.randn(batch, seq_len, config.d_model, device=device)

    # Extract decay alphas
    with torch.no_grad():
        decay_hidden = F.silu(mixer.decay_down(u))
        alpha_logits = mixer.decay_up(decay_hidden) + mixer.decay_bias
        alpha_logits = alpha_logits.reshape(batch, seq_len, config.nheads, config.d_state)
        alpha = torch.sigmoid(alpha_logits)

    alpha_mean = alpha.mean().item()
    alpha_min = alpha.min().item()
    alpha_max = alpha.max().item()
    alpha_std = alpha.std().item()

    print(f"  alpha stats: mean={alpha_mean:.3f} min={alpha_min:.3f} "
          f"max={alpha_max:.3f} std={alpha_std:.3f}")

    all_pass &= check("Alpha in [0, 1]",
        alpha_min >= 0 and alpha_max <= 1)

    all_pass &= check("Alpha not collapsed (std > 0.01)",
        alpha_std > 0.01,
        f"std={alpha_std:.4f}")

    # Check initial bias produces reasonable retention
    all_pass &= check("Alpha mean in [0.3, 0.99] (reasonable retention)",
        0.3 < alpha_mean < 0.99,
        f"mean={alpha_mean:.3f}")

    return all_pass


# ---------------------------------------------------------------------------
# Test 7: Memory usage
# ---------------------------------------------------------------------------

def test_memory(device, batch=8, seq_len=64):
    section("MEMORY USAGE")

    if device.type != 'cuda':
        print("  (Skipped — CPU mode)")
        return True

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    config = NajaConfig(
        d_model=128, d_state=64, n_layer=4, headdim=64,
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        use_chunkwise=True, chunk_size=16,
    )
    model = NajaLM(config, vocab_size=16).to(device)
    x = torch.randint(0, 16, (batch, seq_len), device=device)

    # Forward + backward
    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 16), x[:, 1:].reshape(-1))
    loss.backward()
    model.zero_grad()

    torch.cuda.synchronize(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)

    print(f"  Peak VRAM: {peak_mb:.0f} MB / {total_mb:.0f} MB "
          f"({100 * peak_mb / total_mb:.1f}%)")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    passed = peak_mb < total_mb * 0.9
    check("Peak VRAM < 90% capacity", passed,
          f"{peak_mb:.0f}MB < {total_mb * 0.9:.0f}MB")
    return passed


# ---------------------------------------------------------------------------
# Test 8: AMP compatibility
# ---------------------------------------------------------------------------

def test_amp(device, batch=4, seq_len=32):
    section("AMP COMPATIBILITY")

    if device.type != 'cuda':
        print("  (Skipped — CPU mode)")
        return True

    all_pass = True
    config = NajaConfig(
        d_model=128, d_state=64, n_layer=4, headdim=64,
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        use_chunkwise=True, chunk_size=16,
    )
    model = NajaLM(config, vocab_size=16).to(device)
    x = torch.randint(0, 16, (batch, seq_len), device=device)
    scaler = torch.amp.GradScaler('cuda')

    try:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 16), x[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(torch.optim.SGD(model.parameters(), lr=0.01))
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))

        all_pass &= check("AMP forward: no error", True)
        all_pass &= check("AMP backward: no error", True)
        all_pass &= check("AMP output: no NaN",
            not torch.isnan(logits).any().item())
    except Exception as e:
        all_pass &= check("AMP forward+backward", False, str(e))

    model.zero_grad()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Naja diagnostic')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Naja Diagnostic — device={device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")

    results = {}

    # Run tests
    results['timing'] = test_timing(device)
    results['recurrence'] = test_recurrence_isolation(device)
    results['architecture'] = test_architecture(device)
    results['wy_correctness'] = test_wy_correctness(device)
    results['delta'] = test_delta_effect(device)
    results['pope'] = test_pope_orthogonality(device)
    results['decay'] = test_decay(device)
    results['memory'] = test_memory(device)
    results['amp'] = test_amp(device)

    # Summary
    section("SUMMARY")
    t_fwd, t_fwdbwd, t_fwdbwd_chunk, t_fwdbwd_wy = results['timing']
    n_batches = 5000 // 8
    est_chunk = t_fwdbwd_chunk * n_batches
    est_wy = t_fwdbwd_wy * n_batches
    print(f"  Forward (naive):     {fmt_time(t_fwd)}")
    print(f"  Fwd+bwd (naive):     {fmt_time(t_fwdbwd)}")
    print(f"  Fwd+bwd (grad-ckpt): {fmt_time(t_fwdbwd_chunk)}")
    print(f"  Fwd+bwd (WY chunk):  {fmt_time(t_fwdbwd_wy)}")
    print(f"  Est. epoch (grad-ckpt): {fmt_time(est_chunk)}")
    print(f"  Est. epoch (WY):        {fmt_time(est_wy)}")
    if est_wy > 0:
        speedup = est_chunk / max(est_wy, 1e-9)
        print(f"  WY speedup:             {speedup:.1f}x")
    print()

    if est_wy > 120:
        print("  DIAGNOSIS: WY chunkwise is still slow on this hardware.")
        print("  Consider reducing n_train, seq_len, or n_layer.")
    elif est_wy > 30:
        print("  WY chunkwise: slow but workable (~1-2 min/epoch).")
    else:
        print("  WY chunkwise: training speed looks reasonable.")

    if est_chunk > 120 and est_wy < est_chunk:
        print()
        print("  The naive/grad-checkpoint paths are very slow due to")
        print("  sequential CUDA kernel launches. Use --use_wy_chunkwise")
        print("  for real parallelism.")


if __name__ == '__main__':
    main()
