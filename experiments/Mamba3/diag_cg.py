"""Diagnostic script for CG optimizer NaN investigation.

Traces every intermediate value through one CG step + weight update
to pinpoint exactly where NaN first appears.

Usage:
    python experiments/Mamba3/diag_cg.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from experiments.Mamba3.mamba3_block import Mamba3Config
from experiments.Mamba3.epc_model import ePCMamba3LM


def check(name, tensor, detail=False):
    """Print NaN/Inf check for a tensor."""
    if isinstance(tensor, (int, float)):
        is_nan = tensor != tensor
        is_inf = abs(tensor) == float('inf')
        print(f"  {name:40s}: value={tensor:.6g}, nan={is_nan}, inf={is_inf}")
        return
    if not isinstance(tensor, torch.Tensor):
        print(f"  {name:40s}: type={type(tensor).__name__}, value={tensor}")
        return
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    norm = torch.linalg.vector_norm(tensor.float()).item()
    dtype = tensor.dtype
    tag = ""
    if has_nan:
        tag += " *** NaN ***"
    if has_inf:
        tag += " *** Inf ***"
    print(f"  {name:40s}: norm={norm:.6g}, dtype={dtype}, "
          f"shape={list(tensor.shape)}, nan={has_nan}, inf={has_inf}{tag}")
    if detail and (has_nan or has_inf):
        flat = tensor.reshape(-1)
        nan_count = torch.isnan(flat).sum().item()
        inf_count = torch.isinf(flat).sum().item()
        print(f"    -> {nan_count}/{flat.numel()} NaN, {inf_count}/{flat.numel()} Inf")
        finite = flat[torch.isfinite(flat)]
        if finite.numel() > 0:
            print(f"    -> finite range: [{finite.min().item():.6g}, {finite.max().item():.6g}]")


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = Mamba3Config(d_model=128, d_state=64, n_layer=4, chunk_size=64)

    model = ePCMamba3LM(
        config, vocab_size=16, iters=1, e_lr=0.02,
        error_optim='cg', damping=0.1,
        precision_mode='geometric', precision_base=3.0,
    ).to(device)

    # Synthetic data (1 batch)
    B, L = 4, 64  # small batch for diagnostics
    inputs = torch.randint(1, 16, (B, L), device=device)
    targets = torch.zeros(B, L, dtype=torch.long, device=device)
    targets[:, L // 2:] = inputs[:, :L // 2]

    use_amp = device.type == 'cuda'
    autocast_ctx = (torch.amp.autocast(device.type, dtype=torch.bfloat16)
                    if use_amp else nullcontext())

    print(f"\n{'='*70}")
    print("TEST 1: Forward pass (no errors) — does the model produce NaN?")
    print('='*70)
    with torch.no_grad():
        with autocast_ctx:
            logits = model(inputs)
            check("logits (no errors)", logits, detail=True)
            loss = F.cross_entropy(
                logits.reshape(-1, 16), targets.reshape(-1))
            check("CE loss (no errors)", loss)

    print(f"\n{'='*70}")
    print("TEST 2: Autocast nesting — does enabled=False actually work?")
    print('='*70)
    with autocast_ctx:
        x_bf16 = model.embedding(inputs)
        check("x under autocast", x_bf16)

        dt = x_bf16.device.type
        with torch.amp.autocast(dt, enabled=False):
            x_fp32 = x_bf16.float()
            # Run a layer in fp32
            layer0 = model.pce.layers[0]
            y_inner = layer0(x_fp32)
            check("layer0 output (autocast=False)", y_inner, detail=True)

            # Compare: run same layer WITH autocast
        y_outer = layer0(x_bf16)
        check("layer0 output (autocast=True)", y_outer, detail=True)

    print(f"\n{'='*70}")
    print("TEST 3: Manual CG step — trace every intermediate")
    print('='*70)

    pce = model.pce
    out_proj = model.out_proj

    with autocast_ctx:
        x_raw = model.embedding(inputs).detach()
        check("x_raw (embedding under autocast)", x_raw)

    # Freeze weights
    for p in pce.layers.parameters():
        p.requires_grad_(False)
    for p in pce.out_norm.parameters():
        p.requires_grad_(False)
    out_proj.requires_grad_(False)

    # Init errors
    pce.init_zero_errors(x_raw)
    print(f"\nErrors initialized: {len(pce.errors)} layers")
    for i, e in enumerate(pce.errors):
        check(f"  error[{i}] (init)", e)

    # CG loop manually, in fp32, NO autocast
    x = x_raw.float()
    check("x (cast to fp32)", x)

    device_type = x.device.type
    print(f"\nRunning CG with torch.amp.autocast({device_type}, enabled=False):")

    with torch.amp.autocast(device_type, enabled=False):
        # Check: is autocast actually disabled?
        test_a = torch.randn(4, 4, device=device)
        test_b = torch.randn(4, 4, device=device)
        test_c = test_a @ test_b
        print(f"  matmul dtype inside autocast=False: {test_c.dtype} "
              f"(expect float32)")

        # Forward
        print("\n--- Forward pass (E) ---")
        y_pred = pce.y_pred(x)
        check("y_pred", y_pred, detail=True)

        logits = out_proj(y_pred)
        check("logits", logits, detail=True)

        E_errors = 0.5 * sum(
            pi * torch.linalg.vector_norm(e, ord=2, dim=None) ** 2
            for pi, e in zip(pce.precisions, pce.errors)
        )
        check("E_errors", E_errors)

        b, l, v = logits.shape
        E_output = F.cross_entropy(
            logits.reshape(b * l, v), targets.reshape(b * l),
            reduction='sum')
        check("E_output (CE)", E_output)

        E = E_errors + E_output
        check("E (total)", E)

        # Gradient
        print("\n--- Gradient (g = dE/de, create_graph=True) ---")
        g = torch.autograd.grad(E, pce.errors, create_graph=True)
        for i, gl in enumerate(g):
            check(f"g[{i}]", gl, detail=True)

        # Search direction
        d = [(-gl).detach() for gl in g]
        rTr = sum(torch.dot(rl.reshape(-1), rl.reshape(-1)).item()
                  for rl in d)
        check("rTr (= ||g||²)", rTr)

        # HVP: s = g^T d
        print("\n--- HVP computation ---")
        d_det = [dl.detach() for dl in d]
        s = sum(torch.sum(gl * dl) for gl, dl in zip(g, d_det))
        check("s (= -||g||²)", s)

        print("\n--- torch.autograd.grad(s, errors) for HVP ---")
        try:
            Hd = torch.autograd.grad(s, pce.errors)
            for i, hd in enumerate(Hd):
                check(f"Hd[{i}]", hd, detail=True)

            dTHd = sum(
                torch.dot(dl.reshape(-1), hdl.reshape(-1)).item()
                for dl, hdl in zip(d, Hd))
            check("dTHd (curvature along d)", dTHd)

            if dTHd <= 0:
                alpha = 0.02  # fallback
                print(f"  alpha = {alpha} (negative curvature fallback)")
            else:
                alpha = rTr / dTHd
                print(f"  alpha = rTr/dTHd = {rTr:.6g}/{dTHd:.6g} = {alpha:.6g}")

        except Exception as ex:
            print(f"  *** HVP FAILED: {ex}")
            alpha = 0.02

        # Update errors
        print("\n--- Error update: e += alpha * d ---")
        with torch.no_grad():
            for i, (e, dl) in enumerate(zip(pce.errors, d)):
                e.data.add_(dl.detach(), alpha=alpha)
                check(f"error[{i}] (after CG step)", e, detail=True)

    # Test 4: Weight update phase
    print(f"\n{'='*70}")
    print("TEST 4: Weight update (E_local under autocast with CG errors)")
    print('='*70)

    # Unfreeze
    for p in pce.layers.parameters():
        p.requires_grad_(True)
    for p in pce.out_norm.parameters():
        p.requires_grad_(True)
    out_proj.requires_grad_(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    with autocast_ctx:
        x_for_weight = model.embedding(inputs)
        check("x_for_weight (embedding)", x_for_weight)

        # E_local step by step
        E_local = 0.0
        s_i = x_for_weight
        for i, (pi, e_i, layer_i) in enumerate(
                zip(pce.precisions, pce.errors, pce.layers)):
            s_i_pred = layer_i(s_i)
            check(f"E_local layer {i} s_i_pred", s_i_pred, detail=True)

            s_i_plus_e = s_i_pred + e_i
            check(f"E_local layer {i} s_i_pred + e_i", s_i_plus_e, detail=True)

            s_i = s_i_plus_e.detach()
            mse = pi * 0.5 * F.mse_loss(s_i_pred, s_i, reduction='sum')
            check(f"E_local layer {i} MSE (pi={pi:.2f})", mse)
            E_local = E_local + mse

        # Output
        s_out = pce.out_norm(s_i)
        check("E_local s_out (normed)", s_out, detail=True)
        logits_w = out_proj(s_out)
        check("E_local logits", logits_w, detail=True)

        b, l, v = logits_w.shape
        ce_loss = F.cross_entropy(
            logits_w.reshape(b * l, v), targets.reshape(b * l),
            reduction='sum')
        check("E_local CE", ce_loss)

        total = (E_local + ce_loss) / (B * 1.0)  # energy_scale=1 for CG
        check("weight_loss", total)

    print("\n--- Backward + gradient check ---")
    total.backward()
    param_nan = False
    grad_nan = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                grad_nan = True
                check(f"GRAD {name}", p.grad, detail=True)
    if not grad_nan:
        print("  All parameter gradients are finite.")
    else:
        print("  *** NaN in parameter gradients! ***")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)


if __name__ == '__main__':
    main()
