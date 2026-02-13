"""
Diagnostic script for the flat energy landscape problem in ePC-JEPA.

Measures:
  1. Error gradient decomposition: penalty grad vs output-loss grad per layer
  2. Jacobian ∂logits/∂ε_i singular values (top-k via power iteration)
  3. Per-block Jacobian norms (∂block_out/∂block_in)
  4. RMSNorm attenuation factor
  5. Residual stream dominance (error magnitude vs hidden state magnitude)
  6. Mean reduction scaling analysis
  7. E_init/E_final tracking with full iteration traces

Tests four hypotheses for the flat landscape:
  H1: RMSNorm attenuation
  H2: Residual stream dominance
  H3: Mamba3 internal saturation (small block Jacobians)
  H4: Mean reduction scaling mismatch

The ePC-JEPA model has additional components vs plain ePC-Mamba3:
  - JEPA loss (cosine similarity)
  - Decoder CE loss
  - VICReg regularization
  - Predictor network between encoder and loss
  - Target encoder (EMA)

Usage:
  python experiments/energy_reasoning/diagnose_energy.py --stage 1b
  python experiments/energy_reasoning/diagnose_energy.py --stage 1a
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
from experiments.energy_reasoning.epc_jepa_model import ePCJEPAModel
from experiments.energy_reasoning.data_gen import get_stage_data


# ---------------------------------------------------------------------------
# Diagnostic: Error gradient decomposition
# ---------------------------------------------------------------------------

def diagnose_error_gradient_decomposition(model, seqs, s_target, device):
    """Decompose ε_i.grad into penalty component and output-loss component.

    In ePC-JEPA, E = ½ Σ mean(ε_i²) + JEPA_loss + λ*CE + VICReg.
    At t=0, errors=0 so penalty grad=0, total grad = output-loss grad.

    Returns dict with per-layer gradient norms at t=0, t=1, t=T.
    """
    x_emb = model.embedding(seqs).detach()
    s_target = s_target.detach()

    model._freeze_all_weights()

    results = {'t0': {}, 't1': {}, 'tT': {}}

    # --- t=0: errors are zero ---
    model.init_zero_errors(x_emb)
    optim = torch.optim.SGD(model.errors, lr=model.e_lr)

    optim.zero_grad()
    E = model.E(x_emb, seqs, s_target)
    E.backward()

    results['E_at_t0'] = E.item()

    # Decompose E into components at t=0
    with torch.no_grad():
        E_penalty_t0 = 0.5 * sum(e.pow(2).mean() for e in model.errors).item()

    for i, e in enumerate(model.errors):
        total_grad = e.grad.clone()
        penalty_grad = e.clone()  # = 0 at t=0
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
    E = model.E(x_emb, seqs, s_target)
    E.backward()

    results['E_at_t1'] = E.item()
    for i, e in enumerate(model.errors):
        total_grad = e.grad.clone()
        N = e.numel()
        penalty_grad = e.detach() / N  # ∂(½ mean(ε²))/∂ε
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

    # --- Run remaining T-2 steps to convergence ---
    for t in range(2, model.iters):
        optim.step()
        optim.zero_grad()
        E = model.E(x_emb, seqs, s_target)
        E.backward()

    results['E_at_tT'] = E.item()
    for i, e in enumerate(model.errors):
        total_grad = e.grad.clone()
        N = e.numel()
        penalty_grad = e.detach() / N
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

    model._unfreeze_all_weights()
    return results


# ---------------------------------------------------------------------------
# Diagnostic: Jacobian singular values via power iteration
# ---------------------------------------------------------------------------

def diagnose_jacobian_singular_values(model, seqs, s_target, device,
                                      top_k=3, n_power_iters=20):
    """Estimate top-k singular values of ∂(pred_output)/∂ε_i for each layer.

    In ePC-JEPA, the chain is:
      ε_i → encoder layers i..N → out_norm → predictor → s_pred
    We measure the Jacobian of the predictor output w.r.t. each error.
    """
    x_emb = model.embedding(seqs).detach()
    s_target_det = s_target.detach()

    model._freeze_all_weights()
    model.init_zero_errors(x_emb)

    results = {}

    for layer_idx in range(len(model.layers)):
        e = model.errors[layer_idx]
        h = 1e-4
        svs = []
        deflation_vecs = []

        for k_idx in range(min(top_k, 3)):
            v = torch.randn_like(e)
            v = v / (v.norm() + 1e-10)

            for prev_v in deflation_vecs:
                v = v - (v * prev_v).sum() * prev_v
                v = v / (v.norm() + 1e-10)

            for _ in range(n_power_iters):
                # f(ε): encoder → predictor output
                model.errors[layer_idx] = e.detach().clone().requires_grad_(True)
                s_ctx_base = model.encode_context(x_emb)
                pred_base = model.predictor(s_ctx_base)

                # f(ε + h*v)
                e_pert = (e.detach() + h * v.detach()).requires_grad_(True)
                model.errors[layer_idx] = e_pert
                s_ctx_pert = model.encode_context(x_emb)
                pred_pert = model.predictor(s_ctx_pert)

                Jv = (pred_pert - pred_base) / h

                # J^T @ Jv via backward
                model.errors[layer_idx] = e.detach().clone().requires_grad_(True)
                s_ctx2 = model.encode_context(x_emb)
                pred2 = model.predictor(s_ctx2)

                model.errors[layer_idx].grad = None
                (pred2 * Jv.detach()).sum().backward()
                JtJv = model.errors[layer_idx].grad.clone()

                for prev_v in deflation_vecs:
                    JtJv = JtJv - (JtJv * prev_v).sum() * prev_v

                norm = JtJv.norm()
                if norm > 1e-30:
                    v = JtJv / norm
                else:
                    break

            # Final singular value
            model.errors[layer_idx] = e.detach().clone().requires_grad_(True)
            s_ctx_base = model.encode_context(x_emb)
            pred_base = model.predictor(s_ctx_base)
            e_pert = (e.detach() + h * v.detach()).requires_grad_(True)
            model.errors[layer_idx] = e_pert
            s_ctx_pert = model.encode_context(x_emb)
            pred_pert = model.predictor(s_ctx_pert)
            Jv_final = (pred_pert - pred_base) / h
            sv = Jv_final.norm().item() / max(v.norm().item(), 1e-30)
            svs.append(sv)
            deflation_vecs.append(v.detach())

        model.errors[layer_idx] = torch.zeros_like(e, requires_grad=True)

        results[f'layer_{layer_idx}'] = {
            'singular_values': svs,
            'top_sv': svs[0] if svs else 0.0,
        }

    model._unfreeze_all_weights()
    return results


# ---------------------------------------------------------------------------
# Diagnostic: Per-block Jacobian norms (H3: internal saturation)
# ---------------------------------------------------------------------------

def diagnose_block_jacobians(model, seqs, device):
    """Measure ||∂block_out/∂block_in|| for each encoder block."""
    x = model.embedding(seqs).detach()

    results = {}
    n_dirs = 5
    h = 1e-4

    s = x.clone()
    for layer_idx, layer in enumerate(model.layers):
        layer.eval()

        jac_norms = []
        with torch.no_grad():
            out_base = layer(s)
            for _ in range(n_dirs):
                v = torch.randn_like(s)
                v = v / v.norm()
                out_pert = layer(s + h * v)
                Jv_norm = (out_pert - out_base).norm().item() / h
                jac_norms.append(Jv_norm)

            # Sub-components
            mixer_jac_norms = []
            mixer_base = layer.mixer(layer.mixer_norm(s))
            for _ in range(n_dirs):
                v = torch.randn_like(s)
                v = v / v.norm()
                mixer_pert = layer.mixer(layer.mixer_norm(s + h * v))
                Jv_norm = (mixer_pert - mixer_base).norm().item() / h
                mixer_jac_norms.append(Jv_norm)

            s_after_mixer = s + mixer_base
            mlp_jac_norms = []
            mlp_base = layer.mlp(layer.mlp_norm(s_after_mixer))
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

        with torch.no_grad():
            s = out_base

    return results


# ---------------------------------------------------------------------------
# Diagnostic: RMSNorm attenuation (H1)
# ---------------------------------------------------------------------------

def diagnose_rmsnorm_attenuation(model, seqs, device):
    """Measure how much out_norm (RMSNorm) attenuates perturbations.

    Also tests the predictor's sensitivity to perturbations at its input.
    """
    x = model.embedding(seqs).detach()

    with torch.no_grad():
        s = x
        for layer in model.layers:
            s = layer(s)

    results = {}

    for scale_name, scale in [('tiny', 1e-4), ('small', 1e-2),
                               ('medium', 0.1), ('large', 1.0)]:
        delta = torch.randn_like(s) * scale

        with torch.no_grad():
            # RMSNorm attenuation
            norm_s = model.out_norm(s)
            norm_s_pert = model.out_norm(s + delta)
            change_with_norm = (norm_s_pert - norm_s).norm().item()
            change_without_norm = delta.norm().item()
            attenuation = change_with_norm / (change_without_norm + 1e-30)

            # Predictor sensitivity: does the predictor amplify or attenuate?
            pred_base = model.predictor(norm_s)
            pred_pert = model.predictor(norm_s_pert)
            pred_change = (pred_pert - pred_base).norm().item()

            # Decoder sensitivity
            dec_base = model.decoder(pred_base[:, :-1])
            dec_pert = model.decoder(pred_pert[:, :-1])
            dec_change = (dec_pert - dec_base).norm().item()

        results[scale_name] = {
            'perturbation_norm': delta.norm().item(),
            'norm_attenuation': attenuation,
            'pred_change': pred_change,
            'pred_amplification': pred_change / (change_with_norm + 1e-30),
            'decoder_change': dec_change,
            'end_to_end_ratio': dec_change / (change_without_norm + 1e-30),
        }

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

def diagnose_residual_dominance(model, seqs, device):
    """Measure non-residual contribution vs full residual stream."""
    x = model.embedding(seqs).detach()

    results = {}

    with torch.no_grad():
        s = x.clone()
        for layer_idx, layer in enumerate(model.layers):
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
# Diagnostic: Mean reduction scaling (H4)
# ---------------------------------------------------------------------------

def diagnose_mean_reduction_scaling(model, seqs, s_target, device):
    """Analyze penalty vs output-loss gradient scaling under mean reduction."""
    x_emb = model.embedding(seqs).detach()
    s_target = s_target.detach()

    model._freeze_all_weights()

    results = {}

    for init_scale_name, init_scale in [('zero', 0.0), ('tiny', 1e-4),
                                         ('small', 1e-2), ('medium', 0.1),
                                         ('large', 1.0)]:
        model.init_zero_errors(x_emb)
        for e in model.errors:
            e.data.copy_(torch.randn_like(e) * init_scale)

        for e in model.errors:
            if e.grad is not None:
                e.grad.zero_()

        E = model.E(x_emb, seqs, s_target)
        E_penalty = 0.5 * sum(e.pow(2).mean() for e in model.errors)
        E_output = E.item() - E_penalty.item()
        E.backward()

        layer_data = {}
        for i, e in enumerate(model.errors):
            total_grad = e.grad.clone()
            N = e.numel()
            penalty_grad = e.detach() / N
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

    model._unfreeze_all_weights()
    return results


# ---------------------------------------------------------------------------
# Diagnostic: Full energy trajectory with loss decomposition
# ---------------------------------------------------------------------------

def diagnose_energy_trajectory(model, data_loader, device, n_batches=3):
    """Run error optimization and track E, E_penalty, E_output at every step.

    Also decomposes the output loss into JEPA, decode CE, and VICReg.
    """
    trajectories = []

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= n_batches:
            break

        seqs = batch[0].to(device)
        x_emb = model.embedding(seqs).detach()

        with torch.no_grad():
            s_target = model.encode_target(seqs)

        model._freeze_all_weights()
        model.init_zero_errors(x_emb)
        optim = torch.optim.SGD(model.errors, lr=model.e_lr)

        energy_trace = []
        penalty_trace = []
        output_trace = []
        error_norm_trace = []
        jepa_trace = []
        decode_trace = []

        for t in range(model.iters):
            optim.zero_grad()
            E = model.E(x_emb, seqs, s_target)
            E_val = E.item()

            # Decompose
            with torch.no_grad():
                E_pen = 0.5 * sum(e.pow(2).mean() for e in model.errors).item()
                s_ctx = model.encode_context(x_emb)
                s_pred = model.predictor(s_ctx)
                jepa_l = model._jepa_loss_nextstep(s_pred, s_target).item()
                logits = model.decoder(s_pred[:, :-1])
                dec_l = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    seqs[:, 1:].reshape(-1)).item()

            energy_trace.append(E_val)
            penalty_trace.append(E_pen)
            output_trace.append(E_val - E_pen)
            error_norm_trace.append([e.norm().item() for e in model.errors])
            jepa_trace.append(jepa_l)
            decode_trace.append(dec_l)

            E.backward()
            optim.step()

        # Final values
        with torch.no_grad():
            E_final = model.E(x_emb, seqs, s_target).item()
            energy_trace.append(E_final)

        trajectories.append({
            'energy_trace': energy_trace,
            'penalty_trace': penalty_trace,
            'output_trace': output_trace,
            'error_norm_trace': error_norm_trace,
            'jepa_trace': jepa_trace,
            'decode_trace': decode_trace,
            'E_init': energy_trace[0],
            'E_final': E_final,
            'convergence': energy_trace[0] - E_final,
        })

        model._unfreeze_all_weights()

    return trajectories


# ---------------------------------------------------------------------------
# Diagnostic: Predictor sensitivity (unique to JEPA)
# ---------------------------------------------------------------------------

def diagnose_predictor_sensitivity(model, seqs, device):
    """Measure how sensitive the predictor is to changes in encoder output.

    The error gradient flows: ε → encoder → out_norm → predictor → loss.
    If the predictor has low sensitivity to its input, errors can't steer
    the loss even if the encoder passes the perturbation through.
    """
    x = model.embedding(seqs).detach()

    with torch.no_grad():
        s = x
        for layer in model.layers:
            s = layer(s)
        s_context = model.out_norm(s)

    results = {}

    for scale_name, scale in [('tiny', 1e-4), ('small', 1e-2),
                               ('medium', 0.1), ('large', 1.0)]:
        delta = torch.randn_like(s_context) * scale

        with torch.no_grad():
            pred_base = model.predictor(s_context)
            pred_pert = model.predictor(s_context + delta)
            pred_change = (pred_pert - pred_base).norm().item()

            # JEPA loss change
            s_target = model.encode_target(seqs)
            jepa_base = model._jepa_loss_nextstep(pred_base, s_target).item()
            jepa_pert = model._jepa_loss_nextstep(pred_pert, s_target).item()

        results[scale_name] = {
            'input_perturbation_norm': delta.norm().item(),
            'predictor_output_change': pred_change,
            'amplification_ratio': pred_change / (delta.norm().item() + 1e-30),
            'jepa_loss_base': jepa_base,
            'jepa_loss_perturbed': jepa_pert,
            'jepa_loss_change': abs(jepa_pert - jepa_base),
        }

    return results


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
        description='Diagnose flat energy landscape in ePC-JEPA')
    parser.add_argument('--stage', type=str, default='1b',
                        choices=['1a', '1b', '1c'])
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Smaller batch for diagnostics')
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--d_pred', type=int, default=64)
    parser.add_argument('--n_pred_layer', type=int, default=2)
    parser.add_argument('--d_z', type=int, default=64)
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
    print(f"Stage: {args.stage}")

    # Config
    chunk_size = min(64, args.seq_len)
    while args.seq_len % chunk_size != 0 and chunk_size > 1:
        chunk_size -= 1

    enc_config = Mamba3Config(
        d_model=args.d_model, d_state=args.d_state,
        n_layer=args.n_layer, chunk_size=chunk_size,
    )
    pred_config = Mamba3Config(
        d_model=args.d_pred,
        d_state=min(args.d_state, args.d_pred),
        n_layer=args.n_pred_layer,
        chunk_size=chunk_size,
        headdim=min(64, args.d_pred),
    )

    # Data
    data = get_stage_data(
        stage=args.stage, n_train=500, n_test=100,
        seq_len=args.seq_len, vocab_size=args.vocab_size,
    )
    train_loader = DataLoader(
        TensorDataset(data['train_seqs']),
        batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Model (untrained — diagnosing architecture)
    model = ePCJEPAModel(
        enc_config=enc_config, pred_config=pred_config,
        vocab_size=args.vocab_size, d_z=args.d_z,
        iters=args.iters, e_lr=args.e_lr,
    ).to(device)

    trainable = sum(p.numel() for p in model.get_trainable_params())
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: ePC-JEPA (T={args.iters}, e_lr={args.e_lr})")
    print(f"Encoder: d={enc_config.d_model}, layers={enc_config.n_layer}")
    print(f"Predictor: d={pred_config.d_model}, layers={pred_config.n_layer}")
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    # Get one batch
    batch = next(iter(train_loader))
    seqs = batch[0].to(device)

    with torch.no_grad():
        s_target = model.encode_target(seqs)

    error_numel = args.batch_size * args.seq_len * args.d_model
    print(f"Batch shape: seqs={seqs.shape}")
    print(f"Error numel per layer: {error_numel:,}")

    # =====================================================================
    # DIAGNOSTIC 1: Energy Trajectory with Loss Decomposition
    # =====================================================================
    print_section("DIAGNOSTIC 1: Energy Trajectory (E_init -> E_final)")

    trajectories = diagnose_energy_trajectory(model, train_loader, device, n_batches=3)

    for i, traj in enumerate(trajectories):
        print(f"\nBatch {i}:")
        print(f"  E_init  = {traj['E_init']:.6f}")
        print(f"  E_final = {traj['E_final']:.6f}")
        print(f"  Convergence = {traj['convergence']:.6f}")
        print(f"  Relative convergence = "
              f"{traj['convergence'] / (abs(traj['E_init']) + 1e-10):.6f}")
        print(f"  Energy:  {[f'{e:.4f}' for e in traj['energy_trace'][:5]]} "
              f"... {[f'{e:.4f}' for e in traj['energy_trace'][-3:]]}")
        print(f"  Penalty: {[f'{p:.6f}' for p in traj['penalty_trace'][:5]]}")
        print(f"  Output:  {[f'{o:.4f}' for o in traj['output_trace'][:5]]}")
        print(f"  JEPA:    {[f'{j:.4f}' for j in traj['jepa_trace'][:5]]}")
        print(f"  Decode:  {[f'{d:.4f}' for d in traj['decode_trace'][:5]]}")
        if traj['error_norm_trace']:
            final_norms = traj['error_norm_trace'][-1]
            print(f"  Final error norms: {[f'{n:.6f}' for n in final_norms]}")

    # =====================================================================
    # DIAGNOSTIC 2: Error Gradient Decomposition
    # =====================================================================
    print_section("DIAGNOSTIC 2: Error Gradient Decomposition")
    print("(penalty grad = ε/N, output-loss grad = ∂L/∂ε)")

    grad_results = diagnose_error_gradient_decomposition(
        model, seqs, s_target, device)

    print_subsection("t=0 (errors = 0, penalty grad = 0)")
    print(f"E = {grad_results['E_at_t0']:.6f}")
    for layer_key, d in sorted(grad_results['t0'].items()):
        print(f"  {layer_key}:")
        print(f"    total_grad_norm    = {d['total_grad_norm']:.8f}")
        print(f"    output_loss_grad   = {d['output_loss_grad_norm']:.8f}")
        print(f"    output_grad_mean   = {d['output_loss_grad_mean_abs']:.10f}")

    print_subsection("t=1 (after 1 SGD step)")
    print(f"E = {grad_results['E_at_t1']:.6f}")
    for layer_key, d in sorted(grad_results['t1'].items()):
        print(f"  {layer_key}:")
        print(f"    error_norm         = {d['error_norm']:.8f}")
        print(f"    penalty_grad_norm  = {d['penalty_grad_norm']:.10f}")
        print(f"    output_loss_grad   = {d['output_loss_grad_norm']:.10f}")
        print(f"    penalty/output     = {d['penalty_to_output_ratio']:.4f}")

    print_subsection(f"t=T (after {model.iters} SGD steps)")
    print(f"E = {grad_results['E_at_tT']:.6f}")
    for layer_key, d in sorted(grad_results['tT'].items()):
        print(f"  {layer_key}:")
        print(f"    error_norm         = {d['error_norm']:.8f}")
        print(f"    penalty_grad_norm  = {d['penalty_grad_norm']:.10f}")
        print(f"    output_loss_grad   = {d['output_loss_grad_norm']:.10f}")
        print(f"    penalty/output     = {d['penalty_to_output_ratio']:.4f}")

    # =====================================================================
    # DIAGNOSTIC 3: Jacobian Singular Values
    # =====================================================================
    print_section("DIAGNOSTIC 3: Jacobian ∂(predictor_output)/∂ε_i Singular Values")

    jac_results = diagnose_jacobian_singular_values(
        model, seqs, s_target, device, top_k=3, n_power_iters=20)

    for layer_key, d in sorted(jac_results.items()):
        svs = d['singular_values']
        print(f"  {layer_key}: top SVs = {[f'{s:.4f}' for s in svs]}")

    top_svs = [jac_results[f'layer_{i}']['top_sv']
               for i in range(enc_config.n_layer)]
    print(f"\n  Top SV by layer: {[f'{s:.4f}' for s in top_svs]}")
    if top_svs[0] > 0 and top_svs[-1] > 0:
        print(f"  Ratio layer_0/layer_{enc_config.n_layer-1}: "
              f"{top_svs[0] / top_svs[-1]:.2f}x")

    # =====================================================================
    # DIAGNOSTIC 4: Per-Block Jacobian Norms (H3)
    # =====================================================================
    print_section("DIAGNOSTIC 4: Per-Block Jacobian Norms (H3: Saturation)")

    block_results = diagnose_block_jacobians(model, seqs, device)

    for layer_key, d in sorted(block_results.items()):
        print(f"  {layer_key}:")
        print(f"    block_jac_norm     = {d['block_jac_norm_mean']:.4f} "
              f"(max: {d['block_jac_norm_max']:.4f})")
        print(f"    mixer_jac_norm     = {d['mixer_jac_norm_mean']:.4f}")
        print(f"    mlp_jac_norm       = {d['mlp_jac_norm_mean']:.4f}")
        print(f"    hidden_state_norm  = {d['hidden_state_norm']:.4f}")

    # =====================================================================
    # DIAGNOSTIC 5: RMSNorm Attenuation (H1)
    # =====================================================================
    print_section("DIAGNOSTIC 5: RMSNorm Attenuation (H1)")

    norm_results = diagnose_rmsnorm_attenuation(model, seqs, device)

    for scale_name, d in norm_results.items():
        if scale_name == 'hidden_state_rms':
            print(f"\n  Hidden state RMS: mean={d['mean']:.4f}, "
                  f"min={d['min']:.4f}, max={d['max']:.4f}")
        else:
            print(f"\n  Perturbation '{scale_name}' "
                  f"(||δ||={d['perturbation_norm']:.4f}):")
            print(f"    RMSNorm attenuation    = {d['norm_attenuation']:.6f}")
            print(f"    Predictor amplification = {d['pred_amplification']:.6f}")
            print(f"    End-to-end ratio        = {d['end_to_end_ratio']:.6f}")

    # =====================================================================
    # DIAGNOSTIC 6: Residual Stream Dominance (H2)
    # =====================================================================
    print_section("DIAGNOSTIC 6: Residual Stream Dominance (H2)")

    res_results = diagnose_residual_dominance(model, seqs, device)

    for layer_key, d in sorted(res_results.items()):
        print(f"  {layer_key}:")
        print(f"    input_norm         = {d['input_norm']:.4f}")
        print(f"    mixer_to_input     = {d['mixer_to_input_ratio']:.4f}")
        print(f"    mlp_to_input       = {d['mlp_to_input_ratio']:.4f}")
        print(f"    growth_factor      = {d['growth_factor']:.4f}")

    # =====================================================================
    # DIAGNOSTIC 7: Mean Reduction Scaling (H4)
    # =====================================================================
    print_section("DIAGNOSTIC 7: Mean Reduction Scaling (H4)")

    scale_results = diagnose_mean_reduction_scaling(
        model, seqs, s_target, device)

    for scale_name, d in scale_results.items():
        print(f"\n  Error scale '{scale_name}':")
        print(f"    E_total   = {d['E_total']:.6f}")
        print(f"    E_penalty = {d['E_penalty']:.8f}")
        print(f"    E_output  = {d['E_output']:.6f}")
        for layer_key, ld in sorted(d['layers'].items()):
            print(f"    {layer_key}:")
            print(f"      N = {ld['N_elements']:,}")
            print(f"      penalty_grad_norm    = {ld['penalty_grad_norm']:.8f}")
            print(f"      output_grad_norm     = {ld['output_loss_grad_norm']:.8f}")
            print(f"      ratio penalty/output = {ld['ratio_penalty_to_output']:.4f}")
            print(f"      penalty per element  = {ld['penalty_grad_per_element']:.12f}")
            print(f"      output per element   = {ld['output_grad_per_element']:.12f}")

    # =====================================================================
    # DIAGNOSTIC 8: Predictor Sensitivity (JEPA-specific)
    # =====================================================================
    print_section("DIAGNOSTIC 8: Predictor Sensitivity (JEPA-specific)")

    pred_results = diagnose_predictor_sensitivity(model, seqs, device)

    for scale_name, d in pred_results.items():
        print(f"\n  Input perturbation '{scale_name}' "
              f"(||δ||={d['input_perturbation_norm']:.4f}):")
        print(f"    Predictor output change = {d['predictor_output_change']:.6f}")
        print(f"    Amplification ratio     = {d['amplification_ratio']:.6f}")
        print(f"    JEPA loss change        = {d['jepa_loss_change']:.8f}")
        print(f"    JEPA base/perturbed     = {d['jepa_loss_base']:.4f} / "
              f"{d['jepa_loss_perturbed']:.4f}")

    # =====================================================================
    # Summary & Hypothesis Verdicts
    # =====================================================================
    print_section("SUMMARY: Hypothesis Verdicts")

    # H1: RMSNorm
    tiny_attn = norm_results.get('tiny', {}).get('norm_attenuation', 1.0)
    small_attn = norm_results.get('small', {}).get('norm_attenuation', 1.0)
    print(f"\n  H1 (RMSNorm attenuation):")
    print(f"    Attenuation at tiny scale: {tiny_attn:.6f}")
    print(f"    Attenuation at small scale: {small_attn:.6f}")
    if tiny_attn < 0.1:
        print(f"    VERDICT: LIKELY CAUSE - RMSNorm crushes perturbations by "
              f"{1/tiny_attn:.0f}x")
    elif tiny_attn < 0.5:
        print(f"    VERDICT: CONTRIBUTING - reduces by {1/tiny_attn:.1f}x")
    else:
        print(f"    VERDICT: NOT A MAJOR FACTOR")

    # H2: Residual dominance
    mixer_ratios = [res_results[f'layer_{i}']['mixer_to_input_ratio']
                    for i in range(enc_config.n_layer)]
    avg_mixer_ratio = sum(mixer_ratios) / len(mixer_ratios)
    print(f"\n  H2 (Residual stream dominance):")
    print(f"    Average mixer/input ratio: {avg_mixer_ratio:.4f}")
    if avg_mixer_ratio < 0.1:
        print(f"    VERDICT: LIKELY CAUSE - branch is {1/avg_mixer_ratio:.0f}x "
              f"smaller than residual")
    elif avg_mixer_ratio < 0.5:
        print(f"    VERDICT: CONTRIBUTING - branch is {1/avg_mixer_ratio:.1f}x smaller")
    else:
        print(f"    VERDICT: NOT A MAJOR FACTOR")

    # H3: Saturation
    block_jacs = [block_results[f'layer_{i}']['block_jac_norm_mean']
                  for i in range(enc_config.n_layer)]
    avg_block_jac = sum(block_jacs) / len(block_jacs)
    print(f"\n  H3 (Mamba3 internal saturation):")
    print(f"    Average block Jacobian norm: {avg_block_jac:.4f}")
    if avg_block_jac < 0.1:
        print(f"    VERDICT: LIKELY CAUSE - blocks are nearly constant")
    elif avg_block_jac < 0.5:
        print(f"    VERDICT: CONTRIBUTING - weak response")
    else:
        print(f"    VERDICT: NOT A MAJOR FACTOR - reasonable sensitivity")

    # H4: Scaling
    if 'small' in scale_results:
        small_data = scale_results['small']
        ratios = [small_data['layers'][f'layer_{i}']['ratio_penalty_to_output']
                  for i in range(enc_config.n_layer)]
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n  H4 (Mean reduction scaling):")
        print(f"    Penalty/output ratio at 'small' scale: {avg_ratio:.4f}")
        if avg_ratio > 10:
            print(f"    VERDICT: LIKELY CAUSE - penalty {avg_ratio:.0f}x stronger")
        elif avg_ratio > 2:
            print(f"    VERDICT: CONTRIBUTING - penalty {avg_ratio:.1f}x stronger")
        elif avg_ratio < 0.01:
            print(f"    VERDICT: OPPOSITE - penalty too weak ({avg_ratio:.4f}x)")
        else:
            print(f"    VERDICT: NOT A MAJOR FACTOR - roughly balanced")

    # Jacobian health
    print(f"\n  Overall Jacobian ∂(predictor)/∂ε health:")
    for i in range(enc_config.n_layer):
        sv = jac_results[f'layer_{i}']['top_sv']
        print(f"    Layer {i}: top SV = {sv:.4f}", end="")
        if sv < 0.01:
            print(" [DEAD]")
        elif sv < 0.1:
            print(" [WEAK]")
        elif sv < 1.0:
            print(" [MODERATE]")
        else:
            print(" [HEALTHY]")

    # Predictor sensitivity
    pred_tiny = pred_results.get('tiny', {}).get('amplification_ratio', 1.0)
    print(f"\n  Predictor sensitivity:")
    print(f"    Amplification at tiny scale: {pred_tiny:.4f}")
    if pred_tiny < 0.1:
        print(f"    VERDICT: BOTTLENECK - predictor attenuates by "
              f"{1/pred_tiny:.0f}x")
    elif pred_tiny > 2.0:
        print(f"    VERDICT: HEALTHY - predictor amplifies by {pred_tiny:.1f}x")
    else:
        print(f"    VERDICT: NEUTRAL")

    print(f"\nDone. Run on GPU for accurate measurements.")


if __name__ == '__main__':
    main()
