"""
Diagnostic script for the flat energy landscape problem in ePC-Mamba3.

Measures:
  1. Error gradient decomposition: penalty grad vs output-loss grad per layer
  2. Jacobian ∂ŷ/∂ε_i singular values (top-k via power iteration)
  3. Per-block Jacobian norms (∂block_out/∂block_in)
  4. RMSNorm attenuation factor
  5. Residual stream dominance (error magnitude vs hidden state magnitude)
  6. Mean reduction scaling analysis

Tests four hypotheses for the flat landscape:
  H1: RMSNorm attenuation
  H2: Residual stream dominance
  H3: Mamba3 internal saturation (small block Jacobians)
  H4: Mean reduction scaling mismatch

Usage:
  python experiments/Mamba3/diagnose_energy.py --task copy
  python experiments/Mamba3/diagnose_energy.py --task 1b
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3Block, RMSNorm
from experiments.Mamba3.epc_model import ePCMamba3LM


# ---------------------------------------------------------------------------
# Data generation (copied from train_epc.py to keep this self-contained)
# ---------------------------------------------------------------------------

def generate_copy_data(n_samples, seq_len, vocab_size):
    k = seq_len // 2
    data = torch.randint(1, vocab_size, (n_samples, k))
    full_seq = torch.zeros(n_samples, 2 * k + 1, dtype=torch.long)
    full_seq[:, :k] = data
    full_seq[:, k + 1:] = data
    inputs = full_seq[:, :-1].contiguous()
    targets = full_seq[:, 1:].contiguous()
    targets[:, :k] = -100
    return inputs, targets


def generate_nextstep_data(task, n_samples, seq_len, vocab_size):
    from experiments.energy_reasoning.data_gen import (
        generate_arithmetic, generate_multi_rule, generate_interleaved,
    )
    generators = {'1a': generate_arithmetic, '1b': generate_multi_rule,
                  '1c': generate_interleaved}
    seqs = generators[task](n_samples, seq_len + 1, vocab_size)
    return seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous()


# ---------------------------------------------------------------------------
# Diagnostic: Error gradient decomposition
# ---------------------------------------------------------------------------

def diagnose_error_gradient_decomposition(model, inputs, targets, device):
    """Decompose ε_i.grad into penalty component and output-loss component.

    After E.backward(), each ε_i.grad = ∂E/∂ε_i = ε_i (penalty) + ∂L/∂ε_i (output-loss).
    Since ε_i starts at zero, the penalty gradient is also zero at t=0.
    We measure after 1 SGD step and after T steps to see the decomposition.

    Returns dict with per-layer gradient norms.
    """
    pce = model.pce
    out_proj = model.out_proj

    x = model.embedding(inputs).detach()

    # Freeze weights
    for p in pce.layers.parameters():
        p.requires_grad_(False)
    for p in pce.out_norm.parameters():
        p.requires_grad_(False)
    out_proj.requires_grad_(False)

    results = {'t0': {}, 't1': {}, 'tT': {}}

    # --- t=0: errors are zero, so penalty grad = 0, total grad = output-loss grad ---
    pce.init_zero_errors(x)
    optim = torch.optim.SGD(pce.errors, lr=pce.e_lr)

    optim.zero_grad()
    E = pce.E(x, targets, out_proj)
    E.backward()

    results['E_at_t0'] = E.item()
    for i, e in enumerate(pce.errors):
        total_grad = e.grad.clone()
        penalty_grad = e.clone()  # = 0 since errors are zero
        output_loss_grad = total_grad - penalty_grad

        results['t0'][f'layer_{i}'] = {
            'total_grad_norm': total_grad.norm().item(),
            'penalty_grad_norm': penalty_grad.norm().item(),
            'output_loss_grad_norm': output_loss_grad.norm().item(),
            'total_grad_mean_abs': total_grad.abs().mean().item(),
            'output_loss_grad_mean_abs': output_loss_grad.abs().mean().item(),
            'error_norm': e.norm().item(),
        }

    # --- t=1: after one SGD step ---
    optim.step()
    optim.zero_grad()
    E = pce.E(x, targets, out_proj)
    E.backward()

    results['E_at_t1'] = E.item()
    for i, e in enumerate(pce.errors):
        total_grad = e.grad.clone()
        # Penalty gradient: ∂(½ mean(ε²))/∂ε = ε / N (where N = numel)
        N = e.numel()
        penalty_grad = e / N
        output_loss_grad = total_grad - penalty_grad

        results['t1'][f'layer_{i}'] = {
            'total_grad_norm': total_grad.norm().item(),
            'penalty_grad_norm': penalty_grad.norm().item(),
            'output_loss_grad_norm': output_loss_grad.norm().item(),
            'penalty_to_output_ratio': (
                penalty_grad.norm().item() /
                (output_loss_grad.norm().item() + 1e-30)),
            'error_norm': e.norm().item(),
            'error_mean_abs': e.abs().mean().item(),
        }

    # --- Run remaining T-2 steps to get to convergence ---
    for t in range(2, pce.iters):
        optim.step()
        optim.zero_grad()
        E = pce.E(x, targets, out_proj)
        E.backward()

    results['E_at_tT'] = E.item()
    for i, e in enumerate(pce.errors):
        total_grad = e.grad.clone()
        N = e.numel()
        penalty_grad = e / N
        output_loss_grad = total_grad - penalty_grad

        results['tT'][f'layer_{i}'] = {
            'total_grad_norm': total_grad.norm().item(),
            'penalty_grad_norm': penalty_grad.norm().item(),
            'output_loss_grad_norm': output_loss_grad.norm().item(),
            'penalty_to_output_ratio': (
                penalty_grad.norm().item() /
                (output_loss_grad.norm().item() + 1e-30)),
            'error_norm': e.norm().item(),
            'error_mean_abs': e.abs().mean().item(),
        }

    # Unfreeze
    for p in pce.layers.parameters():
        p.requires_grad_(True)
    for p in pce.out_norm.parameters():
        p.requires_grad_(True)
    out_proj.requires_grad_(True)

    return results


# ---------------------------------------------------------------------------
# Diagnostic: Jacobian singular values via power iteration
# ---------------------------------------------------------------------------

def diagnose_jacobian_singular_values(model, inputs, device, top_k=3,
                                      n_power_iters=20):
    """Estimate top-k singular values of ∂logits/∂ε_i for each layer.

    Uses power iteration on J^T J to find the top singular value.
    Repeats with deflation for top-k (approximate).

    Returns dict with per-layer singular values.
    """
    pce = model.pce
    out_proj = model.out_proj
    x = model.embedding(inputs).detach()

    # Zero errors, no grad on weights
    for p in pce.layers.parameters():
        p.requires_grad_(False)
    for p in pce.out_norm.parameters():
        p.requires_grad_(False)
    out_proj.requires_grad_(False)

    pce.init_zero_errors(x)

    results = {}

    for layer_idx in range(len(pce.layers)):
        # We want ∂logits/∂ε_layer_idx
        # Use the Jacobian-vector product (JVP) trick via backward
        e = pce.errors[layer_idx]
        shape = e.shape  # (batch, seq, d_model)
        flat_dim = e.numel()

        svs = []
        deflation_vecs = []

        for k_idx in range(min(top_k, 3)):
            # Initialize random vector
            v = torch.randn_like(e)
            v = v / (v.norm() + 1e-10)

            # Deflate against previously found vectors
            for prev_v in deflation_vecs:
                v = v - (v * prev_v).sum() * prev_v
                v = v / (v.norm() + 1e-10)

            for _ in range(n_power_iters):
                # Compute J @ v via forward-mode AD (or finite diff)
                # Use finite differences: J @ v ≈ (f(ε + h*v) - f(ε)) / h
                h = 1e-4

                # f(ε)
                pce.errors[layer_idx] = e.detach().clone().requires_grad_(True)
                logits_base = out_proj(pce.y_pred(x))

                # f(ε + h*v)
                e_pert = (e.detach() + h * v.detach()).requires_grad_(True)
                pce.errors[layer_idx] = e_pert
                logits_pert = out_proj(pce.y_pred(x))

                Jv = (logits_pert - logits_base) / h  # J @ v

                # Compute J^T @ (J @ v) via backward
                # Reset error
                pce.errors[layer_idx] = e.detach().clone().requires_grad_(True)
                logits2 = out_proj(pce.y_pred(x))

                # J^T @ Jv: backward with Jv as the upstream gradient
                pce.errors[layer_idx].grad = None
                (logits2 * Jv.detach()).sum().backward()
                JtJv = pce.errors[layer_idx].grad.clone()

                # Deflate
                for prev_v in deflation_vecs:
                    JtJv = JtJv - (JtJv * prev_v).sum() * prev_v

                # Update v
                norm = JtJv.norm()
                if norm > 1e-30:
                    v = JtJv / norm
                else:
                    break

            # Singular value = sqrt(v^T J^T J v) = ||J v||
            # Recompute Jv with final v
            pce.errors[layer_idx] = e.detach().clone().requires_grad_(True)
            logits_base = out_proj(pce.y_pred(x))
            e_pert = (e.detach() + h * v.detach()).requires_grad_(True)
            pce.errors[layer_idx] = e_pert
            logits_pert = out_proj(pce.y_pred(x))
            Jv_final = (logits_pert - logits_base) / h
            sv = Jv_final.norm().item() / max(v.norm().item(), 1e-30)
            svs.append(sv)
            deflation_vecs.append(v.detach())

        # Reset error
        pce.errors[layer_idx] = torch.zeros_like(e, requires_grad=True)

        results[f'layer_{layer_idx}'] = {
            'singular_values': svs,
            'top_sv': svs[0] if svs else 0.0,
        }

    # Unfreeze
    for p in pce.layers.parameters():
        p.requires_grad_(True)
    for p in pce.out_norm.parameters():
        p.requires_grad_(True)
    out_proj.requires_grad_(True)

    return results


# ---------------------------------------------------------------------------
# Diagnostic: Per-block Jacobian norms (H3: internal saturation)
# ---------------------------------------------------------------------------

def diagnose_block_jacobians(model, inputs, device):
    """Measure ||∂block_out/∂block_in|| for each Mamba3Block.

    Uses finite differences: ||J|| ≈ ||f(x+h*v) - f(x)|| / (h * ||v||)
    averaged over several random directions v.

    This tests H3: if block Jacobian norms are small, the block's
    internal nonlinearities (SSD, SwiGLU, gating) are saturated.
    """
    pce = model.pce
    x = model.embedding(inputs).detach()

    results = {}
    n_dirs = 5  # random directions to average
    h = 1e-4

    s = x.clone()
    for layer_idx, layer in enumerate(pce.layers):
        layer.eval()

        # Measure ||∂layer(s)/∂s|| via finite differences
        jac_norms = []
        with torch.no_grad():
            out_base = layer(s)

            for _ in range(n_dirs):
                v = torch.randn_like(s)
                v = v / v.norm()
                out_pert = layer(s + h * v)
                Jv_norm = (out_pert - out_base).norm().item() / h
                jac_norms.append(Jv_norm)

        # Also measure the sub-components:
        # Mamba3Block: x = x + Mixer(norm(x)); x = x + MLP(norm(x))
        with torch.no_grad():
            # Mixer branch Jacobian
            mixer_jac_norms = []
            normed = layer.mixer_norm(s)
            mixer_base = layer.mixer(normed)
            for _ in range(n_dirs):
                v = torch.randn_like(s)
                v = v / v.norm()
                mixer_pert = layer.mixer(layer.mixer_norm(s + h * v))
                Jv_norm = (mixer_pert - mixer_base).norm().item() / h
                mixer_jac_norms.append(Jv_norm)

            # MLP branch Jacobian
            s_after_mixer = s + mixer_base
            mlp_jac_norms = []
            normed2 = layer.mlp_norm(s_after_mixer)
            mlp_base = layer.mlp(normed2)
            for _ in range(n_dirs):
                v = torch.randn_like(s_after_mixer)
                v = v / v.norm()
                mlp_pert = layer.mlp(layer.mlp_norm(s_after_mixer + h * v))
                Jv_norm = (mlp_pert - mlp_base).norm().item() / h
                mlp_jac_norms.append(Jv_norm)

        results[f'layer_{layer_idx}'] = {
            'block_jac_norm_mean': sum(jac_norms) / len(jac_norms),
            'block_jac_norm_max': max(jac_norms),
            'mixer_jac_norm_mean': sum(mixer_jac_norms) / len(mixer_jac_norms),
            'mlp_jac_norm_mean': sum(mlp_jac_norms) / len(mlp_jac_norms),
            'hidden_state_norm': s.norm().item(),
            'hidden_state_mean_abs': s.abs().mean().item(),
        }

        # Propagate through layer for next iteration
        with torch.no_grad():
            s = out_base

    return results


# ---------------------------------------------------------------------------
# Diagnostic: RMSNorm attenuation (H1)
# ---------------------------------------------------------------------------

def diagnose_rmsnorm_attenuation(model, inputs, device):
    """Measure how much RMSNorm attenuates small perturbations.

    Computes the ratio ||norm(s + δ) - norm(s)|| / ||δ|| for
    various perturbation sizes δ. If this ratio << 1, RMSNorm
    is crushing the error signal.

    Also compares output change with and without RMSNorm.
    """
    pce = model.pce
    out_proj = model.out_proj
    x = model.embedding(inputs).detach()

    # Get hidden state right before out_norm
    with torch.no_grad():
        s = x
        for layer in pce.layers:
            s = layer(s)

    results = {}

    # Test different perturbation scales
    for scale_name, scale in [('tiny', 1e-4), ('small', 1e-2),
                               ('medium', 0.1), ('large', 1.0)]:
        delta = torch.randn_like(s) * scale

        with torch.no_grad():
            # With RMSNorm
            norm_s = pce.out_norm(s)
            norm_s_pert = pce.out_norm(s + delta)
            output_change_with_norm = (norm_s_pert - norm_s).norm().item()

            # Without RMSNorm (identity)
            output_change_without_norm = delta.norm().item()

            # Attenuation ratio
            attenuation = output_change_with_norm / (output_change_without_norm + 1e-30)

            # End-to-end: how does perturbation affect logits?
            logits_base = out_proj(norm_s)
            logits_pert = out_proj(norm_s_pert)
            logit_change = (logits_pert - logits_base).norm().item()

            logits_no_norm = out_proj(s)
            logits_no_norm_pert = out_proj(s + delta)
            logit_change_no_norm = (logits_no_norm_pert - logits_no_norm).norm().item()

        results[scale_name] = {
            'perturbation_norm': delta.norm().item(),
            'output_change_with_norm': output_change_with_norm,
            'output_change_without_norm': output_change_without_norm,
            'attenuation_ratio': attenuation,
            'logit_change_with_norm': logit_change,
            'logit_change_without_norm': logit_change_no_norm,
            'logit_attenuation': logit_change / (logit_change_no_norm + 1e-30),
        }

    # Also measure the RMS of the hidden state (the denominator in RMSNorm)
    with torch.no_grad():
        rms = torch.sqrt(s.pow(2).mean(-1, keepdim=True) + 1e-5)
        results['hidden_state_rms'] = {
            'mean': rms.mean().item(),
            'min': rms.min().item(),
            'max': rms.max().item(),
        }

    return results


# ---------------------------------------------------------------------------
# Diagnostic: Residual stream dominance (H2)
# ---------------------------------------------------------------------------

def diagnose_residual_dominance(model, inputs, device):
    """Measure how much each layer's non-residual contribution is
    relative to the full residual stream.

    For x = x + Mixer(norm(x)); x = x + MLP(norm(x)):
    - residual_ratio = ||Mixer(norm(x))|| / ||x|| (should be << 1 for dominance)
    - error_to_hidden_ratio = ||ε_i|| / ||x_i|| after error optimization

    If residual dominates, ε_i is a tiny perturbation to a large residual
    stream, meaning ∂output/∂ε_i ≈ (∂output/∂x_i) which is spread across
    the full d_model dimensions.
    """
    pce = model.pce
    x = model.embedding(inputs).detach()

    results = {}

    with torch.no_grad():
        s = x.clone()
        for layer_idx, layer in enumerate(pce.layers):
            # Measure residual vs non-residual
            normed = layer.mixer_norm(s)
            mixer_out = layer.mixer(normed)
            s_after_mixer = s + mixer_out

            normed2 = layer.mlp_norm(s_after_mixer)
            mlp_out = layer.mlp(normed2)
            s_after_mlp = s_after_mixer + mlp_out

            results[f'layer_{layer_idx}'] = {
                'input_norm': s.norm().item(),
                'mixer_output_norm': mixer_out.norm().item(),
                'mixer_to_input_ratio': mixer_out.norm().item() / (s.norm().item() + 1e-30),
                'mlp_output_norm': mlp_out.norm().item(),
                'mlp_to_input_ratio': mlp_out.norm().item() / (s_after_mixer.norm().item() + 1e-30),
                'full_output_norm': s_after_mlp.norm().item(),
                'growth_factor': s_after_mlp.norm().item() / (s.norm().item() + 1e-30),
            }

            s = s_after_mlp

    return results


# ---------------------------------------------------------------------------
# Diagnostic: Mean reduction scaling analysis (H4)
# ---------------------------------------------------------------------------

def diagnose_mean_reduction_scaling(model, inputs, targets, device):
    """Analyze the scaling of penalty vs output-loss gradients under
    mean reduction.

    Key question: does mean reduction make the error penalty coefficient
    effectively 1/N (where N = numel), which is too weak?

    Compares:
    - ∂(½ mean(ε²))/∂ε = ε/N  (penalty gradient)
    - ∂L_output/∂ε             (output-loss gradient)
    - The ratio penalty/output at different error magnitudes
    """
    pce = model.pce
    out_proj = model.out_proj
    x = model.embedding(inputs).detach()

    # Freeze weights
    for p in pce.layers.parameters():
        p.requires_grad_(False)
    for p in pce.out_norm.parameters():
        p.requires_grad_(False)
    out_proj.requires_grad_(False)

    results = {}

    for init_scale_name, init_scale in [('zero', 0.0), ('tiny', 1e-4),
                                         ('small', 1e-2), ('medium', 0.1),
                                         ('large', 1.0)]:
        pce.init_zero_errors(x)
        # Set errors to specific scale
        for e in pce.errors:
            e.data.copy_(torch.randn_like(e) * init_scale)

        # Compute E and backward
        for e in pce.errors:
            if e.grad is not None:
                e.grad.zero_()

        E = pce.E(x, targets, out_proj)

        # Compute penalty and output-loss separately
        E_penalty = 0.5 * sum(e.pow(2).mean() for e in pce.errors)
        E_output = E.item() - E_penalty.item()

        E.backward()

        layer_data = {}
        for i, e in enumerate(pce.errors):
            total_grad = e.grad.clone()
            N = e.numel()
            penalty_grad = e.detach() / N  # ∂(½ mean(ε²))/∂ε
            output_loss_grad = total_grad - penalty_grad

            layer_data[f'layer_{i}'] = {
                'N_elements': N,
                'error_scale': e.abs().mean().item(),
                'total_grad_norm': total_grad.norm().item(),
                'penalty_grad_norm': penalty_grad.norm().item(),
                'output_loss_grad_norm': output_loss_grad.norm().item(),
                'penalty_grad_per_element': penalty_grad.abs().mean().item(),
                'output_grad_per_element': output_loss_grad.abs().mean().item(),
                'ratio_penalty_to_output': (
                    penalty_grad.norm().item() /
                    (output_loss_grad.norm().item() + 1e-30)),
            }

        results[init_scale_name] = {
            'E_total': E.item(),
            'E_penalty': E_penalty.item(),
            'E_output': E_output,
            'layers': layer_data,
        }

    # Unfreeze
    for p in pce.layers.parameters():
        p.requires_grad_(True)
    for p in pce.out_norm.parameters():
        p.requires_grad_(True)
    out_proj.requires_grad_(True)

    return results


# ---------------------------------------------------------------------------
# Diagnostic: E_init and E_final tracking over a few batches
# ---------------------------------------------------------------------------

def diagnose_energy_trajectory(model, data_loader, device, n_batches=5):
    """Run ePC error optimization on a few batches and track
    E at every iteration (not just init/final).

    Returns per-batch energy trajectories.
    """
    pce = model.pce
    out_proj = model.out_proj

    trajectories = []

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= n_batches:
            break

        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        x = model.embedding(inputs).detach()

        # Freeze weights
        for p in pce.layers.parameters():
            p.requires_grad_(False)
        for p in pce.out_norm.parameters():
            p.requires_grad_(False)
        out_proj.requires_grad_(False)

        pce.init_zero_errors(x)
        optim = torch.optim.SGD(pce.errors, lr=pce.e_lr)

        energy_trace = []
        error_norm_trace = []

        for t in range(pce.iters):
            optim.zero_grad()
            E = pce.E(x, targets, out_proj)
            E_val = E.item()
            energy_trace.append(E_val)

            norms = [e.norm().item() for e in pce.errors]
            error_norm_trace.append(norms)

            E.backward()
            optim.step()

        # Final E after last step
        with torch.no_grad():
            E_final = pce.E(x, targets, out_proj).item()
            energy_trace.append(E_final)

        trajectories.append({
            'energy_trace': energy_trace,
            'error_norm_trace': error_norm_trace,
            'E_init': energy_trace[0],
            'E_final': E_final,
            'convergence': energy_trace[0] - E_final,
        })

        # Unfreeze
        for p in pce.layers.parameters():
            p.requires_grad_(True)
        for p in pce.out_norm.parameters():
            p.requires_grad_(True)
        out_proj.requires_grad_(True)

    return trajectories


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subsection(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Diagnose flat energy landscape in ePC-Mamba3')
    parser.add_argument('--task', type=str, default='copy',
                        choices=['copy', '1a', '1b', '1c'])
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Smaller batch for diagnostics (memory)')
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--e_lr', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.seed > 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Task: {args.task}")

    # Config
    chunk_size = min(64, args.seq_len)
    while args.seq_len % chunk_size != 0 and chunk_size > 1:
        chunk_size -= 1

    config = Mamba3Config(
        d_model=args.d_model, d_state=args.d_state,
        n_layer=args.n_layer, chunk_size=chunk_size,
    )

    # Data
    if args.task == 'copy':
        train_x, train_y = generate_copy_data(500, args.seq_len, args.vocab_size)
        test_x, test_y = generate_copy_data(100, args.seq_len, args.vocab_size)
    else:
        train_x, train_y = generate_nextstep_data(
            args.task, 500, args.seq_len, args.vocab_size)
        test_x, test_y = generate_nextstep_data(
            args.task, 100, args.seq_len, args.vocab_size)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Model (untrained — we're diagnosing architecture, not learned weights)
    model = ePCMamba3LM(
        config, vocab_size=args.vocab_size,
        iters=args.iters, e_lr=args.e_lr,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ePC-Mamba3 (T={args.iters}, e_lr={args.e_lr})")
    print(f"Config: d_model={config.d_model}, n_layer={config.n_layer}, "
          f"d_state={config.d_state}")
    print(f"Parameters: {num_params:,}")

    # Get one batch for single-batch diagnostics
    batch = next(iter(train_loader))
    inputs = batch[0].to(device)
    targets = batch[1].to(device)
    print(f"Batch shape: inputs={inputs.shape}, targets={targets.shape}")
    error_numel = args.batch_size * args.seq_len * args.d_model
    print(f"Error numel per layer: {error_numel:,} "
          f"(batch={args.batch_size} x seq={args.seq_len} x d={args.d_model})")

    # =====================================================================
    # DIAGNOSTIC 1: Energy trajectory
    # =====================================================================
    print_section("DIAGNOSTIC 1: Energy Trajectory (E_init → E_final)")

    trajectories = diagnose_energy_trajectory(model, train_loader, device, n_batches=3)

    for i, traj in enumerate(trajectories):
        print(f"\nBatch {i}:")
        print(f"  E_init  = {traj['E_init']:.6f}")
        print(f"  E_final = {traj['E_final']:.6f}")
        print(f"  Convergence (E_init - E_final) = {traj['convergence']:.6f}")
        print(f"  Relative convergence = {traj['convergence'] / (abs(traj['E_init']) + 1e-10):.6f}")
        print(f"  Energy trace: {[f'{e:.4f}' for e in traj['energy_trace'][:6]]} "
              f"... {[f'{e:.4f}' for e in traj['energy_trace'][-3:]]}")
        if traj['error_norm_trace']:
            last_norms = traj['error_norm_trace'][-1]
            print(f"  Final error norms: {[f'{n:.6f}' for n in last_norms]}")

    # =====================================================================
    # DIAGNOSTIC 2: Error gradient decomposition
    # =====================================================================
    print_section("DIAGNOSTIC 2: Error Gradient Decomposition")
    print("(penalty grad = ε/N, output-loss grad = ∂L/∂ε)")

    grad_results = diagnose_error_gradient_decomposition(
        model, inputs, targets, device)

    print_subsection("t=0 (errors = 0, penalty grad = 0)")
    print(f"E = {grad_results['E_at_t0']:.6f}")
    for layer_key, data in sorted(grad_results['t0'].items()):
        print(f"  {layer_key}:")
        print(f"    total_grad_norm    = {data['total_grad_norm']:.8f}")
        print(f"    output_loss_grad   = {data['output_loss_grad_norm']:.8f}")
        print(f"    output_grad_mean   = {data['output_loss_grad_mean_abs']:.10f}")

    print_subsection("t=1 (after 1 SGD step)")
    print(f"E = {grad_results['E_at_t1']:.6f}")
    for layer_key, data in sorted(grad_results['t1'].items()):
        print(f"  {layer_key}:")
        print(f"    error_norm         = {data['error_norm']:.8f}")
        print(f"    penalty_grad_norm  = {data['penalty_grad_norm']:.10f}")
        print(f"    output_loss_grad   = {data['output_loss_grad_norm']:.10f}")
        print(f"    penalty/output     = {data['penalty_to_output_ratio']:.4f}")

    print_subsection(f"t=T (after {model.pce.iters} SGD steps)")
    print(f"E = {grad_results['E_at_tT']:.6f}")
    for layer_key, data in sorted(grad_results['tT'].items()):
        print(f"  {layer_key}:")
        print(f"    error_norm         = {data['error_norm']:.8f}")
        print(f"    penalty_grad_norm  = {data['penalty_grad_norm']:.10f}")
        print(f"    output_loss_grad   = {data['output_loss_grad_norm']:.10f}")
        print(f"    penalty/output     = {data['penalty_to_output_ratio']:.4f}")

    # =====================================================================
    # DIAGNOSTIC 3: Jacobian singular values
    # =====================================================================
    print_section("DIAGNOSTIC 3: Jacobian ∂logits/∂ε_i Singular Values")
    print("(top singular values via power iteration)")

    jac_results = diagnose_jacobian_singular_values(
        model, inputs, device, top_k=3, n_power_iters=20)

    for layer_key, data in sorted(jac_results.items()):
        svs = data['singular_values']
        print(f"  {layer_key}: top SVs = {[f'{s:.4f}' for s in svs]}")

    # Check: are deeper layers' SVs smaller?
    top_svs = [jac_results[f'layer_{i}']['top_sv']
               for i in range(config.n_layer)]
    print(f"\n  Top SV by layer: {[f'{s:.4f}' for s in top_svs]}")
    if top_svs[0] > 0 and top_svs[-1] > 0:
        print(f"  Ratio layer_0/layer_{config.n_layer-1}: "
              f"{top_svs[0] / top_svs[-1]:.2f}x")

    # =====================================================================
    # DIAGNOSTIC 4: Per-block Jacobian norms (H3: saturation)
    # =====================================================================
    print_section("DIAGNOSTIC 4: Per-Block Jacobian Norms (H3: Saturation)")
    print("(||∂block_out/∂block_in|| via finite differences)")

    block_results = diagnose_block_jacobians(model, inputs, device)

    for layer_key, data in sorted(block_results.items()):
        print(f"  {layer_key}:")
        print(f"    block_jac_norm     = {data['block_jac_norm_mean']:.4f} "
              f"(max: {data['block_jac_norm_max']:.4f})")
        print(f"    mixer_jac_norm     = {data['mixer_jac_norm_mean']:.4f}")
        print(f"    mlp_jac_norm       = {data['mlp_jac_norm_mean']:.4f}")
        print(f"    hidden_state_norm  = {data['hidden_state_norm']:.4f}")

    # =====================================================================
    # DIAGNOSTIC 5: RMSNorm attenuation (H1)
    # =====================================================================
    print_section("DIAGNOSTIC 5: RMSNorm Attenuation (H1)")
    print("(||norm(s+δ) - norm(s)|| / ||δ||)")

    norm_results = diagnose_rmsnorm_attenuation(model, inputs, device)

    for scale_name, data in norm_results.items():
        if scale_name == 'hidden_state_rms':
            print(f"\n  Hidden state RMS: mean={data['mean']:.4f}, "
                  f"min={data['min']:.4f}, max={data['max']:.4f}")
        else:
            print(f"\n  Perturbation '{scale_name}' (||δ||={data['perturbation_norm']:.4f}):")
            print(f"    RMSNorm attenuation   = {data['attenuation_ratio']:.6f}")
            print(f"    Logit attenuation      = {data['logit_attenuation']:.6f}")
            print(f"    Change with norm       = {data['output_change_with_norm']:.6f}")
            print(f"    Change without norm    = {data['output_change_without_norm']:.6f}")

    # =====================================================================
    # DIAGNOSTIC 6: Residual stream dominance (H2)
    # =====================================================================
    print_section("DIAGNOSTIC 6: Residual Stream Dominance (H2)")
    print("(mixer/MLP output norm vs residual stream norm)")

    res_results = diagnose_residual_dominance(model, inputs, device)

    for layer_key, data in sorted(res_results.items()):
        print(f"  {layer_key}:")
        print(f"    input_norm         = {data['input_norm']:.4f}")
        print(f"    mixer_to_input     = {data['mixer_to_input_ratio']:.4f}")
        print(f"    mlp_to_input       = {data['mlp_to_input_ratio']:.4f}")
        print(f"    growth_factor      = {data['growth_factor']:.4f}")

    # =====================================================================
    # DIAGNOSTIC 7: Mean reduction scaling (H4)
    # =====================================================================
    print_section("DIAGNOSTIC 7: Mean Reduction Scaling (H4)")
    print("(penalty vs output-loss gradient at different error scales)")

    scale_results = diagnose_mean_reduction_scaling(
        model, inputs, targets, device)

    for scale_name, data in scale_results.items():
        print(f"\n  Error scale '{scale_name}':")
        print(f"    E_total   = {data['E_total']:.6f}")
        print(f"    E_penalty = {data['E_penalty']:.8f}")
        print(f"    E_output  = {data['E_output']:.6f}")
        for layer_key, ldata in sorted(data['layers'].items()):
            print(f"    {layer_key}:")
            print(f"      N = {ldata['N_elements']:,}")
            print(f"      penalty_grad_norm   = {ldata['penalty_grad_norm']:.8f}")
            print(f"      output_grad_norm    = {ldata['output_loss_grad_norm']:.8f}")
            print(f"      ratio penalty/output = {ldata['ratio_penalty_to_output']:.4f}")
            print(f"      penalty per element  = {ldata['penalty_grad_per_element']:.12f}")
            print(f"      output per element   = {ldata['output_grad_per_element']:.12f}")

    # =====================================================================
    # Summary & Hypothesis Verdict
    # =====================================================================
    print_section("SUMMARY: Hypothesis Verdicts")

    # H1: RMSNorm
    tiny_attn = norm_results.get('tiny', {}).get('attenuation_ratio', 1.0)
    small_attn = norm_results.get('small', {}).get('attenuation_ratio', 1.0)
    print(f"\n  H1 (RMSNorm attenuation):")
    print(f"    Attenuation at tiny scale: {tiny_attn:.6f}")
    print(f"    Attenuation at small scale: {small_attn:.6f}")
    if tiny_attn < 0.1:
        print(f"    VERDICT: LIKELY CAUSE — RMSNorm crushes small perturbations by "
              f"{1/tiny_attn:.0f}x")
    elif tiny_attn < 0.5:
        print(f"    VERDICT: CONTRIBUTING — RMSNorm reduces by {1/tiny_attn:.1f}x")
    else:
        print(f"    VERDICT: NOT A MAJOR FACTOR — ratio near 1.0")

    # H2: Residual dominance
    mixer_ratios = [res_results[f'layer_{i}']['mixer_to_input_ratio']
                    for i in range(config.n_layer)]
    avg_mixer_ratio = sum(mixer_ratios) / len(mixer_ratios)
    print(f"\n  H2 (Residual stream dominance):")
    print(f"    Average mixer/input ratio: {avg_mixer_ratio:.4f}")
    if avg_mixer_ratio < 0.1:
        print(f"    VERDICT: LIKELY CAUSE — non-residual branch is {1/avg_mixer_ratio:.0f}x "
              f"smaller than residual")
    elif avg_mixer_ratio < 0.5:
        print(f"    VERDICT: CONTRIBUTING — branch is {1/avg_mixer_ratio:.1f}x smaller")
    else:
        print(f"    VERDICT: NOT A MAJOR FACTOR")

    # H3: Saturation
    block_jacs = [block_results[f'layer_{i}']['block_jac_norm_mean']
                  for i in range(config.n_layer)]
    avg_block_jac = sum(block_jacs) / len(block_jacs)
    print(f"\n  H3 (Mamba3 internal saturation):")
    print(f"    Average block Jacobian norm: {avg_block_jac:.4f}")
    if avg_block_jac < 0.1:
        print(f"    VERDICT: LIKELY CAUSE — blocks are nearly constant functions")
    elif avg_block_jac < 0.5:
        print(f"    VERDICT: CONTRIBUTING — blocks have weak response")
    else:
        print(f"    VERDICT: NOT A MAJOR FACTOR — blocks have reasonable sensitivity")

    # H4: Scaling
    # Compare penalty vs output at 'small' error scale (most realistic)
    if 'small' in scale_results:
        small_data = scale_results['small']
        ratios = [small_data['layers'][f'layer_{i}']['ratio_penalty_to_output']
                  for i in range(config.n_layer)]
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n  H4 (Mean reduction scaling):")
        print(f"    Penalty/output ratio at 'small' scale: {avg_ratio:.4f}")
        if avg_ratio > 10:
            print(f"    VERDICT: LIKELY CAUSE — penalty {avg_ratio:.0f}x stronger than output signal")
        elif avg_ratio > 2:
            print(f"    VERDICT: CONTRIBUTING — penalty {avg_ratio:.1f}x stronger")
        elif avg_ratio < 0.01:
            print(f"    VERDICT: OPPOSITE — penalty too weak ({avg_ratio:.4f}x), "
                  f"errors may grow unchecked")
        else:
            print(f"    VERDICT: NOT A MAJOR FACTOR — roughly balanced")

    # Overall Jacobian health
    print(f"\n  Overall Jacobian ∂logits/∂ε health:")
    for i in range(config.n_layer):
        sv = jac_results[f'layer_{i}']['top_sv']
        print(f"    Layer {i}: top SV = {sv:.4f}", end="")
        if sv < 0.01:
            print(" [DEAD — errors at this layer cannot influence output]")
        elif sv < 0.1:
            print(" [WEAK — errors have very limited influence]")
        elif sv < 1.0:
            print(" [MODERATE — errors have some influence]")
        else:
            print(" [HEALTHY — errors can steer output]")

    print(f"\nDone. Run on GPU for accurate measurements.")


if __name__ == '__main__':
    main()
