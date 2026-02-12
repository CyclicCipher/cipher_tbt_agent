"""Diagnostic: isolate which operation in ssd_trapz produces NaN on HVP.

Tests second-derivative (Hessian-vector product) through progressively
smaller sub-operations of the Mamba3 computation to pinpoint the exact
source of NaN in the CG optimizer.

Usage:
    python experiments/Mamba3/diag_hvp.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F

from experiments.Mamba3.mamba3_block import (
    Mamba3Config, Mamba3Block, Mamba3Mixer, RMSNorm, segsum, ssd_trapz,
    apply_rope,
)


def test_hvp(name, fn, x_shape, device, seed=42):
    """Test if HVP through fn produces NaN.

    Creates e (error), computes loss = fn(x + e).sum(),
    then checks if Hd has NaN where d = -grad.

    Returns True if NaN detected.
    """
    torch.manual_seed(seed)
    x = torch.randn(x_shape, device=device, dtype=torch.float32).detach()
    e = torch.zeros(x_shape, device=device, dtype=torch.float32,
                    requires_grad=True)

    inp = x + e
    try:
        out = fn(inp)
    except Exception as ex:
        print(f"  {name:55s}: FORWARD FAILED: {ex}")
        return True

    loss = out.sum()

    g = torch.autograd.grad(loss, e, create_graph=True)[0]
    g_norm = torch.linalg.vector_norm(g).item()
    g_nan = torch.isnan(g).any().item()

    if g_nan:
        print(f"  {name:55s}: GRAD NaN (1st derivative broken)")
        return True

    d = -g.detach()
    s = torch.sum(g * d)

    if not s.requires_grad:
        print(f"  {name:55s}: OK   |g|={g_norm:.4g}  |Hd|=0 (H=0, linear fn)")
        return False

    Hd_tuple = torch.autograd.grad(s, e, allow_unused=True)
    Hd = Hd_tuple[0]

    if Hd is None:
        print(f"  {name:55s}: OK   |g|={g_norm:.4g}  |Hd|=0 (unused in graph)")
        return False

    hd_nan = torch.isnan(Hd).any().item()
    hd_inf = torch.isinf(Hd).any().item()
    hd_norm = torch.linalg.vector_norm(Hd).item()
    nan_count = torch.isnan(Hd.reshape(-1)).sum().item()
    total = Hd.reshape(-1).numel()

    status = "NaN!" if hd_nan else ("Inf!" if hd_inf else "OK")
    print(f"  {name:55s}: {status:4s} |g|={g_norm:.4g}  |Hd|={hd_norm:.4g}"
          f"  nan={nan_count}/{total}")
    return hd_nan or hd_inf


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    config = Mamba3Config(d_model=128, d_state=64, n_layer=4, chunk_size=64)
    B, L, D = 4, 64, config.d_model
    nheads = config.nheads
    headdim = config.headdim
    d_state = config.d_state
    chunk_size = config.chunk_size
    chunks = L // chunk_size

    # ================================================================
    print("=" * 70)
    print("PART 1: Basic operations (sanity checks)")
    print("=" * 70)

    linear = torch.nn.Linear(D, D, bias=False).to(device).float()
    test_hvp("Linear", linear, (B, L, D), device)

    test_hvp("SiLU", F.silu, (B, L, D), device)

    norm = RMSNorm(D).to(device).float()
    for p in norm.parameters():
        p.requires_grad_(False)
    test_hvp("RMSNorm", norm, (B, L, D), device)

    test_hvp("Softplus", F.softplus, (B, L, 4), device)
    test_hvp("Sigmoid", torch.sigmoid, (B, L, 1), device)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 2: segsum + exp variants")
    print("=" * 70)

    A_shape = (B, nheads, chunks, chunk_size)

    def segsum_only(x):
        return segsum(x.reshape(A_shape))
    test_hvp("segsum (no exp)", segsum_only,
             (B * nheads * chunks * chunk_size,), device)

    def segsum_exp(x):
        return torch.exp(segsum(x.reshape(A_shape)))
    test_hvp("segsum + exp", segsum_exp,
             (B * nheads * chunks * chunk_size,), device)

    def cumsum_exp(x):
        return torch.exp(torch.cumsum(x.reshape(A_shape), dim=-1))
    test_hvp("cumsum + exp (no mask)", cumsum_exp,
             (B * nheads * chunks * chunk_size,), device)

    def exp_of_cumsum_diff(x):
        A_c = x.reshape(A_shape)
        A_cum = torch.cumsum(A_c, dim=-1)
        return torch.exp(A_cum[:, :, :, -1:] - A_cum)
    test_hvp("exp(cumsum[-1] - cumsum)", exp_of_cumsum_diff,
             (B * nheads * chunks * chunk_size,), device)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 3: SSD sub-steps (e through A directly)")
    print("=" * 70)

    # Step 1: L = exp(segsum(A_c))
    def ssd_step1(A_flat):
        A_c = A_flat.reshape(A_shape)
        return torch.exp(segsum(A_c))
    test_hvp("Step1: L = exp(segsum(A))", ssd_step1,
             (B * nheads * chunks * chunk_size,), device)

    # Step 1 + einsum with fixed operands
    torch.manual_seed(42)
    C_fixed = torch.randn(B, chunks, chunk_size, 1, d_state, device=device)
    B_fixed = torch.randn(B, chunks, chunk_size, 1, d_state, device=device)
    x_fixed = torch.randn(B, chunks, chunk_size, nheads, headdim, device=device)

    def ssd_step1_einsum(A_flat):
        A_c = A_flat.reshape(A_shape)
        L = torch.exp(segsum(A_c))
        Y = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp",
                          C_fixed, B_fixed, L, x_fixed)
        return Y
    test_hvp("Step1+einsum: Y_diag", ssd_step1_einsum,
             (B * nheads * chunks * chunk_size,), device)

    # Step 2: decay_states = exp(A_cumsum[-1:] - A_cumsum)
    def ssd_step2(A_flat):
        A_c = A_flat.reshape(A_shape)
        A_cumsum = torch.cumsum(A_c, dim=-1)
        return torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    test_hvp("Step2: decay_states", ssd_step2,
             (B * nheads * chunks * chunk_size,), device)

    # Step 3: decay_chunk = exp(segsum(pad(A_cumsum[-1])))
    def ssd_step3(A_flat):
        A_c = A_flat.reshape(A_shape)
        A_cumsum = torch.cumsum(A_c, dim=-1)
        return torch.exp(
            segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
        )
    test_hvp("Step3: decay_chunk", ssd_step3,
             (B * nheads * chunks * chunk_size,), device)

    # Step 4: state_decay_out = exp(A_cumsum)
    def ssd_step4(A_flat):
        A_c = A_flat.reshape(A_shape)
        A_cumsum = torch.cumsum(A_c, dim=-1)
        return torch.exp(A_cumsum)
    test_hvp("Step4: state_decay_out = exp(A_cumsum)", ssd_step4,
             (B * nheads * chunks * chunk_size,), device)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 4: ssd_trapz with e on specific inputs")
    print("=" * 70)

    torch.manual_seed(42)
    x_ssd = torch.randn(B, L, nheads, headdim, device=device)
    A_ssd = torch.randn(B, L, nheads, device=device) * 0.1 - 0.5
    B_ssd = torch.randn(B, L, 1, d_state, device=device)
    C_ssd = torch.randn(B, L, 1, d_state, device=device)
    lam_ssd = torch.sigmoid(torch.randn(B, L, 1, 1, device=device))
    xp_ssd = F.pad(x_ssd[:, :-1], (0, 0, 0, 0, 1, 0))
    Bp_ssd = F.pad(B_ssd[:, :-1], (0, 0, 0, 0, 1, 0))

    def ssd_e_A(e_flat):
        e = e_flat.reshape(B, L, nheads)
        y, _ = ssd_trapz(x_ssd, xp_ssd, A_ssd + e,
                          B_ssd, Bp_ssd, C_ssd, lam_ssd, chunk_size)
        return y
    test_hvp("ssd_trapz: e through A only", ssd_e_A,
             (B * L * nheads,), device)

    def ssd_e_x(e_flat):
        e = e_flat.reshape(B, L, nheads, headdim)
        y, _ = ssd_trapz(x_ssd + e, xp_ssd, A_ssd,
                          B_ssd, Bp_ssd, C_ssd, lam_ssd, chunk_size)
        return y
    test_hvp("ssd_trapz: e through x_curr only", ssd_e_x,
             (B * L * nheads * headdim,), device)

    def ssd_e_B(e_flat):
        e = e_flat.reshape(B, L, 1, d_state)
        y, _ = ssd_trapz(x_ssd, xp_ssd, A_ssd,
                          B_ssd + e, Bp_ssd, C_ssd, lam_ssd, chunk_size)
        return y
    test_hvp("ssd_trapz: e through B_curr only", ssd_e_B,
             (B * L * d_state,), device)

    def ssd_e_C(e_flat):
        e = e_flat.reshape(B, L, 1, d_state)
        y, _ = ssd_trapz(x_ssd, xp_ssd, A_ssd,
                          B_ssd, Bp_ssd, C_ssd + e, lam_ssd, chunk_size)
        return y
    test_hvp("ssd_trapz: e through C only", ssd_e_C,
             (B * L * d_state,), device)

    def ssd_e_all(e_flat):
        """e added to all inputs simultaneously (like real model)."""
        n = B * L
        # Split e into contributions for each input
        eA = e_flat[:n * nheads].reshape(B, L, nheads) * 0.01
        eB = e_flat[:n * d_state].reshape(B, L, 1, d_state) * 0.01
        eC = e_flat[:n * d_state].reshape(B, L, 1, d_state) * 0.01
        y, _ = ssd_trapz(x_ssd, xp_ssd, A_ssd + eA,
                          B_ssd + eB, Bp_ssd, C_ssd + eC, lam_ssd,
                          chunk_size)
        return y
    test_hvp("ssd_trapz: e through A+B+C", ssd_e_all,
             (B * L * d_state,), device)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 5: Full Mixer decomposed")
    print("=" * 70)

    torch.manual_seed(42)
    mixer = Mamba3Mixer(config).to(device).float()
    for p in mixer.parameters():
        p.requires_grad_(False)

    def mixer_proj_only(u):
        """Just linear projection, no nonlinearities."""
        return mixer.in_proj(u)
    test_hvp("Mixer: in_proj only (linear)", mixer_proj_only,
             (B, L, D), device)

    def mixer_proj_silu(u):
        cfg = config
        proj = mixer.in_proj(u)
        z, x, Bv, Cv, dt, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state,
             cfg.nheads, cfg.d_state // 2, 1], dim=-1)
        return F.silu(x)
    test_hvp("Mixer: proj + SiLU(x)", mixer_proj_silu, (B, L, D), device)

    def mixer_proj_dt(u):
        cfg = config
        proj = mixer.in_proj(u)
        z, x, Bv, Cv, dt, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state,
             cfg.nheads, cfg.d_state // 2, 1], dim=-1)
        return F.softplus(dt + mixer.dt_bias)
    test_hvp("Mixer: proj + softplus(dt)", mixer_proj_dt,
             (B, L, D), device)

    def mixer_proj_BC_rope(u):
        cfg = config
        proj = mixer.in_proj(u)
        z, x, Bv, Cv, dt, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state,
             cfg.nheads, cfg.d_state // 2, 1], dim=-1)
        Bv = mixer.B_norm(Bv + mixer.B_bias)
        Cv = mixer.C_norm(Cv + mixer.C_bias)
        theta_cum = torch.cumsum(theta, dim=1)
        Bv = apply_rope(Bv, theta_cum)
        Cv = apply_rope(Cv, theta_cum)
        return torch.cat([Bv, Cv], dim=-1)
    test_hvp("Mixer: proj + BC norm + RoPE", mixer_proj_BC_rope,
             (B, L, D), device)

    def mixer_up_to_ssd(u):
        """Everything up to ssd_trapz output (before D skip + gating)."""
        cfg = config
        batch, seqlen, _ = u.shape
        A = -torch.exp(mixer.A_log)
        proj = mixer.in_proj(u)
        z, x, Bv, Cv, dt, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state,
             cfg.nheads, cfg.d_state // 2, 1], dim=-1)
        dt = F.softplus(dt + mixer.dt_bias)
        lam = torch.sigmoid(lam_logit)
        x = F.silu(x)
        Bv = mixer.B_norm(Bv + mixer.B_bias)
        Cv = mixer.C_norm(Cv + mixer.C_bias)
        theta_cum = torch.cumsum(theta, dim=1)
        Bv = apply_rope(Bv, theta_cum)
        Cv = apply_rope(Cv, theta_cum)
        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        x_dt = x * dt.unsqueeze(-1)
        x_prev = F.pad(x_dt[:, :-1], (0, 0, 0, 0, 1, 0))
        B_prev = F.pad(Bv[:, :-1].unsqueeze(2), (0, 0, 0, 0, 1, 0))
        B_curr = Bv.unsqueeze(2)
        C_curr = Cv.unsqueeze(2)
        lam_exp = lam.unsqueeze(-1)
        y, _ = ssd_trapz(x_dt, x_prev, A * dt, B_curr, B_prev,
                          C_curr, lam_exp, cfg.chunk_size)
        return y
    test_hvp("Mixer: full up to ssd_trapz output", mixer_up_to_ssd,
             (B, L, D), device)

    def mixer_full(u):
        return mixer(u)
    test_hvp("Mixer: FULL forward", mixer_full, (B, L, D), device)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 6: Full Block and MLP")
    print("=" * 70)

    torch.manual_seed(42)
    block = Mamba3Block(config).to(device).float()
    for p in block.parameters():
        p.requires_grad_(False)

    def block_mixer_only(u):
        return block.mixer(block.mixer_norm(u))
    test_hvp("Block: norm + mixer only", block_mixer_only,
             (B, L, D), device)

    def block_mlp_only(u):
        return block.mlp(block.mlp_norm(u))
    test_hvp("Block: norm + MLP only", block_mlp_only,
             (B, L, D), device)

    def block_full(u):
        return block(u)
    test_hvp("Block: FULL (mixer + MLP + residuals)", block_full,
             (B, L, D), device)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 7: Per-layer error in full ePC model")
    print("=" * 70)

    from experiments.Mamba3.epc_model import ePCMamba3LM

    torch.manual_seed(42)
    model = ePCMamba3LM(
        config, vocab_size=16, iters=1, e_lr=0.02,
        error_optim='cg', damping=0.1,
        precision_mode='geometric', precision_base=3.0,
    ).to(device).float()

    inputs = torch.randint(1, 16, (B, L), device=device)
    targets = torch.zeros(B, L, dtype=torch.long, device=device)
    targets[:, L // 2:] = inputs[:, :L // 2]

    pce = model.pce
    out_proj = model.out_proj

    for p in pce.layers.parameters():
        p.requires_grad_(False)
    for p in pce.out_norm.parameters():
        p.requires_grad_(False)
    out_proj.requires_grad_(False)

    x = model.embedding(inputs).detach().float()
    pce.init_zero_errors(x)

    # Test HVP for each error individually
    for idx in range(len(pce.errors)):
        e = pce.errors[idx]

        E = pce.E(x, targets, out_proj)
        g = torch.autograd.grad(E, [e], create_graph=True)[0]
        g_nan = torch.isnan(g).any().item()

        if g_nan:
            print(f"  error[{idx}] grad: NaN in first derivative!")
            # Reset for next iter
            pce.init_zero_errors(x)
            continue

        d = -g.detach()
        s = torch.sum(g * d)
        g_norm = torch.linalg.vector_norm(g).item()

        if not s.requires_grad:
            print(f"  error[{idx}] HVP: OK   |g|={g_norm:.4g}  |Hd|=0 (H=0)")
            pce.init_zero_errors(x)
            continue

        Hd_tuple = torch.autograd.grad(s, [e], allow_unused=True)
        Hd = Hd_tuple[0]

        if Hd is None:
            print(f"  error[{idx}] HVP: OK   |g|={g_norm:.4g}  |Hd|=0 (unused)")
            pce.init_zero_errors(x)
            continue

        hd_nan = torch.isnan(Hd).any().item()
        hd_norm = torch.linalg.vector_norm(Hd).item()
        nan_count = torch.isnan(Hd.reshape(-1)).sum().item()
        total = Hd.numel()

        status = "NaN!" if hd_nan else "OK"
        print(f"  error[{idx}] HVP: {status:4s}  |g|={g_norm:.4g}"
              f"  |Hd|={hd_norm:.4g}  nan={nan_count}/{total}")

        # Reset errors for next test
        pce.init_zero_errors(x)

    # Unfreeze
    for p in pce.layers.parameters():
        p.requires_grad_(True)
    for p in pce.out_norm.parameters():
        p.requires_grad_(True)
    out_proj.requires_grad_(True)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 8: Pinpoint NaN in ePC chain")
    print("=" * 70)
    print("\n  --- 8a-d: Single error at end of N-block chain ---")
    print("  (other blocks run with no_grad, error only at last position)")

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    layers = list(pce.layers)
    out_norm_fn = pce.out_norm
    out_proj_fn = model.out_proj

    for n_blocks in range(len(layers) + 1):
        # Run first n_blocks with no_grad to get s_base
        with torch.no_grad():
            s = x.clone()
            for i in range(n_blocks):
                s = layers[i](s)
        s_base = s.detach()

        # Add error, then out_norm → out_proj → CE
        e_test = torch.zeros_like(s_base, requires_grad=True)
        s_out = out_norm_fn(s_base + e_test)
        logits = out_proj_fn(s_out)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            targets.reshape(-1), reduction='sum')

        g = torch.autograd.grad(loss, e_test, create_graph=True)[0]
        g_norm = torch.linalg.vector_norm(g).item()
        g_nan = torch.isnan(g).any().item()
        if g_nan:
            print(f"  {n_blocks} blocks → e → norm → proj → CE        "
                  f": GRAD NaN!")
            continue

        d = -g.detach()
        s = torch.sum(g * d)
        if not s.requires_grad:
            print(f"  {n_blocks} blocks → e → norm → proj → CE        "
                  f": OK   |g|={g_norm:.4g}  H=0")
            continue

        Hd = torch.autograd.grad(s, e_test, allow_unused=True)[0]
        if Hd is None:
            print(f"  {n_blocks} blocks → e → norm → proj → CE        "
                  f": OK   |g|={g_norm:.4g}  Hd=unused")
            continue

        hd_nan = torch.isnan(Hd).any().item()
        hd_norm = torch.linalg.vector_norm(Hd).item()
        nan_count = torch.isnan(Hd.reshape(-1)).sum().item()
        total = Hd.numel()
        status = "NaN!" if hd_nan else "OK"
        print(f"  {n_blocks} blocks → e → norm → proj → CE        "
              f": {status:4s} |g|={g_norm:.4g}  |Hd|={hd_norm:.4g}"
              f"  nan={nan_count}/{total}")

    # --- 8e: Single error with blocks AFTER it (in the graph) ---
    print("\n  --- 8e: Error at position i, remaining blocks in graph ---")

    for err_pos in range(len(layers)):
        # Run blocks before err_pos with no_grad
        with torch.no_grad():
            s = x.clone()
            for i in range(err_pos):
                s = layers[i](s)
        s_base = s.detach()

        # Add error, then run remaining blocks WITH grad
        e_test = torch.zeros_like(s_base, requires_grad=True)
        s_curr = s_base + e_test
        for i in range(err_pos, len(layers)):
            s_curr = layers[i](s_curr)

        s_out = out_norm_fn(s_curr)
        logits = out_proj_fn(s_out)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            targets.reshape(-1), reduction='sum')

        g = torch.autograd.grad(loss, e_test, create_graph=True)[0]
        g_norm = torch.linalg.vector_norm(g).item()
        if torch.isnan(g).any():
            label = f"e → blocks[{err_pos}:{len(layers)}] → norm → CE"
            print(f"  {label:50s}: GRAD NaN!")
            continue

        d = -g.detach()
        sv = torch.sum(g * d)
        if not sv.requires_grad:
            label = f"e → blocks[{err_pos}:{len(layers)}] → norm → CE"
            print(f"  {label:50s}: OK   |g|={g_norm:.4g}  H=0")
            continue

        Hd = torch.autograd.grad(sv, e_test, allow_unused=True)[0]
        if Hd is None:
            label = f"e → blocks[{err_pos}:{len(layers)}] → norm → CE"
            print(f"  {label:50s}: OK   |g|={g_norm:.4g}  Hd=unused")
            continue

        hd_nan = torch.isnan(Hd).any().item()
        hd_norm = torch.linalg.vector_norm(Hd).item()
        nan_count = torch.isnan(Hd.reshape(-1)).sum().item()
        total = Hd.numel()
        status = "NaN!" if hd_nan else "OK"
        label = f"e → blocks[{err_pos}:{len(layers)}] → norm → CE"
        print(f"  {label:50s}: {status:4s} |g|={g_norm:.4g}"
              f"  |Hd|={hd_norm:.4g}  nan={nan_count}/{total}")

    # --- 8f: Full ePC chain with all errors vs single error ---
    print("\n  --- 8f: Full chain, all errors in graph vs single error ---")

    # All errors in graph (replicating pce.E exactly)
    pce.init_zero_errors(x)
    E = pce.E(x, targets, out_proj_fn)
    g_all = torch.autograd.grad(E, pce.errors, create_graph=True)
    g0_norm = torch.linalg.vector_norm(g_all[0]).item()
    d_all = [-gl.detach() for gl in g_all]
    sv = sum(torch.sum(gl * dl) for gl, dl in zip(g_all, d_all))
    if sv.requires_grad:
        Hd_all = torch.autograd.grad(sv, pce.errors, allow_unused=True)
        hd0 = Hd_all[0]
        if hd0 is not None:
            hd_nan = torch.isnan(hd0).any().item()
            hd_norm = torch.linalg.vector_norm(hd0).item()
            status = "NaN!" if hd_nan else "OK"
            print(f"  All errors in graph, grad ALL, HVP ALL  : {status:4s}"
                  f"  |g[0]|={g0_norm:.4g}  |Hd[0]|={hd_norm:.4g}")
        else:
            print(f"  All errors in graph, grad ALL, HVP ALL  : Hd=None")
    else:
        print(f"  All errors in graph, grad ALL, HVP ALL  : H=0")

    # All errors in graph, but grad+HVP only for error[0]
    pce.init_zero_errors(x)
    E = pce.E(x, targets, out_proj_fn)
    g_one = torch.autograd.grad(E, [pce.errors[0]], create_graph=True)[0]
    g_one_norm = torch.linalg.vector_norm(g_one).item()
    d_one = -g_one.detach()
    sv = torch.sum(g_one * d_one)
    if sv.requires_grad:
        Hd_one = torch.autograd.grad(sv, [pce.errors[0]],
                                      allow_unused=True)[0]
        if Hd_one is not None:
            hd_nan = torch.isnan(Hd_one).any().item()
            hd_norm = torch.linalg.vector_norm(Hd_one).item()
            status = "NaN!" if hd_nan else "OK"
            print(f"  All errors in graph, grad e[0], HVP e[0]: {status:4s}"
                  f"  |g|={g_one_norm:.4g}  |Hd|={hd_norm:.4g}")
        else:
            print(f"  All errors in graph, grad e[0], HVP e[0]: Hd=None")
    else:
        print(f"  All errors in graph, grad e[0], HVP e[0]: H=0")

    # --- 8g: .sum() loss instead of CE (control) ---
    print("\n  --- 8g: .sum() loss instead of CE ---")

    pce.init_zero_errors(x)
    y_pred = pce.y_pred(x)
    logits = out_proj_fn(y_pred)
    loss_sum = logits.sum()  # sum loss instead of CE

    g_sum = torch.autograd.grad(loss_sum, pce.errors, create_graph=True)
    g0_norm = torch.linalg.vector_norm(g_sum[0]).item()
    d_sum = [-gl.detach() for gl in g_sum]
    sv = sum(torch.sum(gl * dl) for gl, dl in zip(g_sum, d_sum))
    if sv.requires_grad:
        Hd_sum = torch.autograd.grad(sv, pce.errors, allow_unused=True)
        hd0 = Hd_sum[0]
        if hd0 is not None:
            hd_nan = torch.isnan(hd0).any().item()
            hd_norm = torch.linalg.vector_norm(hd0).item()
            status = "NaN!" if hd_nan else "OK"
            print(f"  Full chain, .sum() loss, HVP ALL        : {status:4s}"
                  f"  |g[0]|={g0_norm:.4g}  |Hd[0]|={hd_norm:.4g}")
        else:
            print(f"  Full chain, .sum() loss, HVP ALL        : Hd=None")
    else:
        print(f"  Full chain, .sum() loss, HVP ALL        : H=0")

    # --- 8h: Numerical HVP (finite differences) as ground truth ---
    print("\n  --- 8h: Numerical HVP (finite differences) ---")

    pce.init_zero_errors(x)
    eps_fd = 1e-3

    def compute_grad_at(errors_data):
        """Compute gradient at given error values."""
        for e, ed in zip(pce.errors, errors_data):
            e.data.copy_(ed)
        E = pce.E(x, targets, out_proj_fn)
        g = torch.autograd.grad(E, pce.errors)
        return [gl.detach().clone() for gl in g]

    errors_zero = [torch.zeros_like(e) for e in pce.errors]
    g_base = compute_grad_at(errors_zero)

    # Direction: d = -g (but g at zero)
    d_fd = [-gl.clone() for gl in g_base]
    d_norm = sum(torch.dot(dl.reshape(-1), dl.reshape(-1)).item()
                 for dl in d_fd) ** 0.5

    # g(0 + eps*d) - finite differences
    errors_plus = [eps_fd * dl for dl in d_fd]
    g_plus = compute_grad_at(errors_plus)

    # Hd ≈ (g(eps*d) - g(0)) / eps
    Hd_fd = [(gp - gb) / eps_fd for gp, gb in zip(g_plus, g_base)]
    hd0_norm = torch.linalg.vector_norm(Hd_fd[0]).item()
    hd0_nan = torch.isnan(Hd_fd[0]).any().item()
    status = "NaN!" if hd0_nan else "OK"
    print(f"  Numerical HVP (eps={eps_fd})               : {status:4s}"
          f"  |Hd[0]|={hd0_norm:.4g}")

    # Reset errors to zero
    for e in pce.errors:
        e.data.zero_()

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad_(True)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
