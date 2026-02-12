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
    Hd_tuple = torch.autograd.grad(s, e, allow_unused=True)
    Hd = Hd_tuple[0]

    if Hd is None:
        print(f"  {name:55s}: OK   |g|={g_norm:.4g}  |Hd|=0 (linear, no 2nd deriv)")
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
        Hd_tuple = torch.autograd.grad(s, [e], allow_unused=True)
        Hd = Hd_tuple[0]

        g_norm = torch.linalg.vector_norm(g).item()
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
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
