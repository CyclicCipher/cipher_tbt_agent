"""
JEPA training for energy-based reasoning experiments (backprop).

Stage 1 (--stage 1a/1b/1c): Sequence prediction on structured sequences.
  Validates that the JEPA encoder learns useful representations.
  Default: next-step prediction (aligned with Mamba's causal nature).

Stage 2 (--stage 2): Pattern induction (few-shot rule discovery).
  Validates that z + Langevin improve predictions beyond single-pass.
  Evaluates four ablation variants at test time:
    no_z, random_z, langevin, oracle
  Key metric: Langevin gap = accuracy(langevin) - accuracy(no-z).

Usage:
  # Stage 1b: multi-rule with next-step prediction (default)
  python experiments/energy_reasoning/train_jepa.py --stage 1b

  # Stage 1b with masked prediction (not recommended for Mamba, see Mistake #34)
  python experiments/energy_reasoning/train_jepa.py --stage 1b --prediction_mode masked

  # Stage 2: pattern induction
  python experiments/energy_reasoning/train_jepa.py --stage 2 --epochs 50
"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.Mamba3.mamba3_block import Mamba3Config
from experiments.energy_reasoning.jepa_model import (
    JEPAModel, LangevinDynamics, vicreg_loss,
)
from experiments.energy_reasoning.data_gen import (
    get_stage_data, generate_mask, generate_causal_mask,
    generate_last_token_mask, DEFAULT_RULES,
)


# ---------------------------------------------------------------------------
# Evaluation — Stage 1
# ---------------------------------------------------------------------------

def compute_masked_accuracy(logits, tokens, mask):
    """Token-level accuracy on masked positions only."""
    preds = logits.argmax(dim=-1)
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == tokens[mask]).float().mean().item()


def compute_nextstep_accuracy(logits_shift, tokens_shift):
    """Next-token prediction accuracy."""
    preds = logits_shift.argmax(dim=-1)
    return (preds == tokens_shift).float().mean().item()


def evaluate_stage1_nextstep(model, test_loader, device, amp_ctx):
    """Evaluate Stage 1 with next-step prediction."""
    model.eval()
    total_jepa = 0.0
    total_decode = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            seqs = batch[0].to(device)
            with amp_ctx():
                result = model.forward_train_nextstep(seqs, z=None)
                logits_shift = model.decode(result['s_pred'][:, :-1])
            total_jepa += result['L_jepa'].item()
            total_decode += result['L_decode'].item()

            tokens_shift = seqs[:, 1:]
            total_acc += compute_nextstep_accuracy(logits_shift, tokens_shift)
            n_batches += 1

    return {
        'jepa_loss': total_jepa / n_batches,
        'decode_loss': total_decode / n_batches,
        'accuracy': total_acc / n_batches,
    }


def evaluate_stage1_masked(model, test_loader, mask_ratio, device, amp_ctx,
                           mask_mode='causal'):
    """Evaluate Stage 1 with masked prediction."""
    model.eval()
    total_jepa = 0.0
    total_decode = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            seqs = batch[0].to(device)
            if mask_mode == 'causal':
                mask = generate_causal_mask(
                    seqs.shape[0], seqs.shape[1], mask_ratio, device)
            else:
                mask = generate_mask(
                    seqs.shape[0], seqs.shape[1], mask_ratio, device)

            with amp_ctx():
                result = model.forward_train(seqs, mask, z=None)
                logits = model.decode(result['s_pred'])
            total_jepa += result['L_jepa'].item()
            total_decode += result['L_decode'].item()
            total_acc += compute_masked_accuracy(logits, seqs, mask)
            n_batches += 1

    return {
        'jepa_loss': total_jepa / n_batches,
        'decode_loss': total_decode / n_batches,
        'accuracy': total_acc / n_batches,
    }


# ---------------------------------------------------------------------------
# Evaluation — Stage 2 (multi-variant)
# ---------------------------------------------------------------------------

def evaluate_stage2(model, test_loader, device, n_rules, oracle_proj,
                    langevin_T=5, langevin_eta=0.01,
                    langevin_sigma=0.1):
    """Evaluate pattern induction with four ablation variants.

    Returns dict with per-variant accuracy and per-rule breakdown.
    """
    model.eval()

    variants = ['no_z', 'random_z', 'langevin', 'oracle']
    correct = {v: 0 for v in variants}
    total = 0
    rule_correct = {v: [0] * n_rules for v in variants}
    rule_total = [0] * n_rules
    energy_trajectories = []

    for batch in test_loader:
        seqs = batch[0].to(device)
        targets = batch[1].to(device)
        rule_idx = batch[2].to(device)
        B = seqs.shape[0]

        with torch.no_grad():
            s_target = model.encode_target(seqs)

        # --- Variant 1: no z ---
        with torch.no_grad():
            logits = model.forward_eval_nextstep(seqs, z=None)
            pred_no_z = logits[:, -1].argmax(dim=-1)

        # --- Variant 2: random z ---
        with torch.no_grad():
            z_rand = torch.randn(B, model.d_z, device=device)
            logits = model.forward_eval_nextstep(seqs, z=z_rand)
            pred_rand = logits[:, -1].argmax(dim=-1)

        # --- Variant 3: Langevin-refined z ---
        z_star, eng_traj = model.langevin_refine(
            seqs, s_target,
            T=langevin_T, eta=langevin_eta,
            sigma_max=langevin_sigma,
            noise_at_test=False,
        )
        if len(energy_trajectories) < 10:
            energy_trajectories.append(eng_traj)
        with torch.no_grad():
            logits = model.forward_eval_nextstep(seqs, z=z_star)
            pred_lang = logits[:, -1].argmax(dim=-1)

        # --- Variant 4: oracle z ---
        with torch.no_grad():
            z_oracle = oracle_proj[rule_idx]
            logits = model.forward_eval_nextstep(seqs, z=z_oracle)
            pred_oracle = logits[:, -1].argmax(dim=-1)

        # Tally
        preds = {
            'no_z': pred_no_z, 'random_z': pred_rand,
            'langevin': pred_lang, 'oracle': pred_oracle,
        }
        for v in variants:
            c = (preds[v] == targets).long()
            correct[v] += c.sum().item()
            for r in range(n_rules):
                mask_r = (rule_idx == r)
                rule_correct[v][r] += (c * mask_r.long()).sum().item()

        total += B
        for r in range(n_rules):
            rule_total[r] += (rule_idx == r).sum().item()

    acc = {v: correct[v] / max(total, 1) for v in variants}
    rule_acc = {}
    for v in variants:
        rule_acc[v] = [
            rule_correct[v][r] / max(rule_total[r], 1) for r in range(n_rules)
        ]

    return {
        'accuracy': acc,
        'rule_accuracy': rule_acc,
        'rule_total': rule_total,
        'langevin_gap': acc['langevin'] - acc['no_z'],
        'energy_trajectories': energy_trajectories,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class JEPADiagnostics:
    """Collect and plot training diagnostics for JEPA experiments."""

    def __init__(self, stage, z_mode='none'):
        self.stage = stage
        self.z_mode = z_mode
        self.reset()

    def reset(self):
        self.train_jepa = []
        self.train_decode = []
        self.train_acc = []
        self.train_var = []
        self.train_cov = []
        self.ms_per_batch = []
        self.test_jepa = []
        self.test_decode = []
        self.test_acc = []

    def update_train(self, jepa_loss, decode_loss, acc, L_var, L_cov, ms):
        self.train_jepa.append(jepa_loss)
        self.train_decode.append(decode_loss)
        self.train_acc.append(acc)
        self.train_var.append(L_var)
        self.train_cov.append(L_cov)
        self.ms_per_batch.append(ms)

    def update_test(self, metrics):
        self.test_jepa.append(metrics.get('jepa_loss', 0))
        self.test_decode.append(metrics.get('decode_loss', 0))
        self.test_acc.append(metrics.get('accuracy', 0))

    def plot(self, save_path, epoch, config_str=''):
        nrows = 2
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
        fig.suptitle(
            f'JEPA Stage {self.stage} — Epoch {epoch} {config_str}',
            fontsize=13)

        epochs = list(range(1, len(self.test_acc) + 1))

        def _epoch_avg(data, n_epochs):
            n_per = len(data) // max(n_epochs, 1)
            if n_per == 0:
                return []
            return [np.mean(data[i * n_per:(i + 1) * n_per])
                    for i in range(n_epochs)]

        n_ep = len(epochs)

        # [0,0] JEPA loss
        ax = axes[0, 0]
        if self.test_jepa:
            ax.plot(epochs, self.test_jepa, 'r--', label='Test', linewidth=2)
        train_avg = _epoch_avg(self.train_jepa, n_ep)
        if train_avg:
            ax.plot(epochs[:len(train_avg)], train_avg, 'b-',
                    label='Train', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('JEPA Prediction Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [0,1] Accuracy
        ax = axes[0, 1]
        if self.test_acc:
            ax.plot(epochs, self.test_acc, 'r--', label='Test', linewidth=2)
        train_acc_avg = _epoch_avg(self.train_acc, n_ep)
        if train_acc_avg:
            ax.plot(epochs[:len(train_acc_avg)], train_acc_avg, 'b-',
                    label='Train', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Token Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [0,2] Speed
        ax = axes[0, 2]
        ms_avg = _epoch_avg(self.ms_per_batch, n_ep)
        if ms_avg:
            ax.plot(epochs[:len(ms_avg)], ms_avg, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ms/batch')
        ax.set_title('Speed')
        ax.grid(True, alpha=0.3)

        # [1,0] VICReg variance
        ax = axes[1, 0]
        var_avg = _epoch_avg(self.train_var, n_ep)
        cov_avg = _epoch_avg(self.train_cov, n_ep)
        if var_avg:
            ax.plot(epochs[:len(var_avg)], var_avg, 'b-',
                    label='Variance', linewidth=2)
        if cov_avg:
            ax2 = ax.twinx()
            ax2.plot(epochs[:len(cov_avg)], cov_avg, 'r-',
                     label='Covariance', linewidth=2)
            ax2.set_ylabel('L_cov', color='r')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L_var', color='b')
        ax.set_title('VICReg Components')
        ax.grid(True, alpha=0.3)

        # [1,1] Decode CE loss
        ax = axes[1, 1]
        if self.test_decode:
            ax.plot(epochs, self.test_decode, 'r--', label='Test', linewidth=2)
        dec_avg = _epoch_avg(self.train_decode, n_ep)
        if dec_avg:
            ax.plot(epochs[:len(dec_avg)], dec_avg, 'b-',
                    label='Train', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CE Loss')
        ax.set_title('Decode Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [1,2] Summary
        ax = axes[1, 2]
        ax.axis('off')
        lines = [f'Stage: {self.stage}', f'z_mode: {self.z_mode}']
        if self.test_acc:
            lines.append(f'Best Test Acc: {max(self.test_acc):.2%}')
            lines.append(f'Final Test Acc: {self.test_acc[-1]:.2%}')
        if self.ms_per_batch:
            lines.append(f'Avg ms/batch: {np.mean(self.ms_per_batch):.1f}')
        ax.text(0.1, 0.5, '\n'.join(lines), fontsize=12,
                verticalalignment='center', family='monospace')

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f'  [Saved {save_path}]')


# ---------------------------------------------------------------------------
# Stage 2 Plot
# ---------------------------------------------------------------------------

def plot_stage2(save_path, epoch, config_str, history, last_metrics,
                rule_names, args):
    """Plot Stage 2 diagnostics: ablation comparison, per-rule, energy."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f'JEPA Stage 2 (Pattern Induction) - Epoch {epoch} {config_str}',
        fontsize=13)

    epochs = list(range(1, len(history['no_z']) + 1))

    colors = {'no_z': 'blue', 'random_z': 'gray',
              'langevin': 'red', 'oracle': 'green'}
    labels = {'no_z': 'No z', 'random_z': 'Random z',
              'langevin': 'Langevin', 'oracle': 'Oracle'}

    # [0,0] Ablation accuracy over time
    ax = axes[0, 0]
    for v in ['no_z', 'random_z', 'langevin', 'oracle']:
        ax.plot(epochs, history[v], color=colors[v],
                label=labels[v], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Ablation: Answer Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # [0,1] Langevin gap over time
    ax = axes[0, 1]
    ax.plot(epochs, history['gap'], 'red', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.10, color='green', linestyle=':', alpha=0.5,
               label='10% target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Langevin - No z')
    ax.set_title('Langevin Gap')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # [0,2] Per-rule accuracy (last epoch)
    ax = axes[0, 2]
    ra = last_metrics['rule_accuracy']
    x_pos = np.arange(len(rule_names))
    width = 0.2
    for i, v in enumerate(['no_z', 'langevin', 'oracle']):
        vals = ra[v]
        ax.bar(x_pos + i * width, vals, width, label=labels[v],
               color=colors[v], alpha=0.8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(rule_names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Rule Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] Energy trajectory (last epoch)
    ax = axes[1, 0]
    if history['energy_traj']:
        traj = history['energy_traj'][-1]
        ax.plot(range(len(traj)), traj, 'r-o', linewidth=2, markersize=4)
    ax.set_xlabel('Langevin Step')
    ax.set_ylabel('E_pred(z)')
    ax.set_title('Energy Trajectory (Last Epoch)')
    ax.grid(True, alpha=0.3)

    # [1,1] Energy trajectories across epochs
    ax = axes[1, 1]
    n_traj = len(history['energy_traj'])
    if n_traj > 0:
        indices = [0, n_traj // 4, n_traj // 2, 3 * n_traj // 4, n_traj - 1]
        indices = sorted(set(max(0, min(i, n_traj - 1)) for i in indices))
        for idx in indices:
            traj = history['energy_traj'][idx]
            ax.plot(range(len(traj)), traj, '-o', markersize=3,
                    label=f'Epoch {idx + 1}', alpha=0.8)
    ax.set_xlabel('Langevin Step')
    ax.set_ylabel('E_pred(z)')
    ax.set_title('Energy Trajectories Over Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,2] Summary text
    ax = axes[1, 2]
    ax.axis('off')
    a = last_metrics['accuracy']
    gap = last_metrics['langevin_gap']
    lines = [
        f"Stage 2 Results (Epoch {epoch})",
        f"",
        f"No z:      {a['no_z']:.4f}",
        f"Random z:  {a['random_z']:.4f}",
        f"Langevin:  {a['langevin']:.4f}",
        f"Oracle:    {a['oracle']:.4f}",
        f"",
        f"Langevin gap: {gap:+.4f}",
        f"",
        f"Config:",
        f"  Langevin T={args.langevin_T}",
        f"  eta={args.langevin_eta}",
        f"  sigma={args.langevin_sigma}",
        f"  d_z={args.d_z}",
        f"  Train w/ z: {args.train_with_z}",
    ]
    ax.text(0.1, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f'  [Saved {save_path}]')


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def _print_profile(prof):
    """Print per-phase timing breakdown."""
    total = sum(prof[k] for k in ['forward', 'backward', 'ema_update',
                                   'diagnostics', 'eval'])
    n = prof['n_batches']
    ne = prof['n_epochs']
    print(f"\n{'='*55}")
    print(f"  PROFILE ({ne} epochs, {n} batches)")
    print(f"{'='*55}")
    for label, key in [('Forward + loss', 'forward'),
                       ('Backward + optim', 'backward'),
                       ('EMA update', 'ema_update'),
                       ('Train diagnostics', 'diagnostics'),
                       ('Evaluation', 'eval')]:
        t = prof[key]
        pct = 100 * t / max(total, 1e-9)
        per_batch = 1000 * t / max(n, 1)
        print(f"  {label:<20s} {t:7.2f}s  ({pct:5.1f}%)  "
              f"{per_batch:6.1f} ms/b")
    print(f"  {'TOTAL':<20s} {total:7.2f}s  "
          f"{total/max(ne,1):6.1f} s/epoch")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='JEPA energy reasoning training (backprop)')

    # Stage
    parser.add_argument('--stage', type=str, default='1b',
                        choices=['1a', '1b', '1c', '2'],
                        help='Experiment stage')

    # Architecture
    parser.add_argument('--d_model', type=int, default=128,
                        help='Encoder model dimension')
    parser.add_argument('--d_state', type=int, default=64,
                        help='SSM state dimension')
    parser.add_argument('--n_layer', type=int, default=4,
                        help='Encoder layers')
    parser.add_argument('--d_pred', type=int, default=64,
                        help='Predictor bottleneck dimension')
    parser.add_argument('--n_pred_layer', type=int, default=2,
                        help='Predictor layers')
    parser.add_argument('--d_z', type=int, default=64,
                        help='Latent variable dimension')

    # Data
    parser.add_argument('--seq_len', type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--mask_ratio', type=float, default=0.2,
                        help='Fraction of positions to mask (masked mode)')
    parser.add_argument('--n_examples', type=int, default=5,
                        help='Example pairs in pattern induction (Stage 2)')
    parser.add_argument('--n_rules', type=int, default=5,
                        help='Number of rules (Stage 2)')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Peak learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=4,
                        help='LR warmup epochs (linear ramp from lr/5 to lr)')
    parser.add_argument('--w_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')

    # JEPA / VICReg
    parser.add_argument('--ema_tau_start', type=float, default=0.996)
    parser.add_argument('--ema_tau_end', type=float, default=1.0)
    parser.add_argument('--jepa_loss', type=str, default='l2',
                        choices=['cosine', 'l2'],
                        help='JEPA loss: cosine or l2')
    parser.add_argument('--lambda_decode', type=float, default=1.0,
                        help='Decode CE loss weight')
    parser.add_argument('--lambda_var', type=float, default=1.0,
                        help='VICReg variance loss weight')
    parser.add_argument('--lambda_cov', type=float, default=0.04,
                        help='VICReg covariance loss weight')
    parser.add_argument('--mask_mode', type=str, default='causal',
                        choices=['causal', 'random'],
                        help='Mask mode for masked prediction')
    parser.add_argument('--prediction_mode', type=str, default='next_step',
                        choices=['masked', 'next_step'],
                        help='Prediction mode: next_step (default, aligns '
                             'with Mamba causal nature) or masked')

    # z / Langevin (Stage 2)
    parser.add_argument('--langevin_T', type=int, default=5,
                        help='Langevin steps')
    parser.add_argument('--langevin_eta', type=float, default=0.01,
                        help='Langevin step size')
    parser.add_argument('--langevin_sigma', type=float, default=0.1,
                        help='Langevin max noise scale')
    parser.add_argument('--langevin_threshold', type=float, default=1e-3,
                        help='Langevin adaptive stopping threshold')
    parser.add_argument('--train_with_z', action='store_true', default=True,
                        help='Train with oracle z (Stage 2, teaches predictor '
                             'to use z)')
    parser.add_argument('--no_train_with_z', dest='train_with_z',
                        action='store_false',
                        help='Train without z (Stage 2 baseline)')

    # Performance
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision (AMP)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (PyTorch 2+)')
    parser.add_argument('--profile', action='store_true',
                        help='Run 5-epoch profiling with per-phase timing')

    # Misc
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot_every', type=int, default=10)
    args = parser.parse_args()

    # Seed
    if args.seed > 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Seed: {args.seed}")

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Stage 2 smart defaults
    # -----------------------------------------------------------------------

    is_stage2 = (args.stage == '2')
    if is_stage2:
        if args.n_train == 5000:
            args.n_train = 10000
        if args.n_test == 1000:
            args.n_test = 2000
        if args.epochs == 30:
            args.epochs = 50
        if args.seq_len == 64:
            task_len = 2 * args.n_examples + 2
            args.seq_len = max(task_len, 16)
        if args.batch_size == 32:
            args.batch_size = 128
        # Stage 2 always uses next_step prediction
        prediction_mode = 'next_step'
    else:
        prediction_mode = args.prediction_mode

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------

    chunk_size = min(64, args.seq_len)
    while args.seq_len % chunk_size != 0 and chunk_size > 1:
        chunk_size -= 1

    enc_config = Mamba3Config(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layer=args.n_layer,
        chunk_size=chunk_size,
    )
    pred_config = Mamba3Config(
        d_model=args.d_pred,
        d_state=min(args.d_state, args.d_pred),
        n_layer=args.n_pred_layer,
        chunk_size=chunk_size,
        headdim=min(64, args.d_pred),
    )

    print(f"Encoder: d={enc_config.d_model}, layers={enc_config.n_layer}, "
          f"chunk={chunk_size}")
    print(f"Predictor: d={pred_config.d_model}, layers={pred_config.n_layer}")
    print(f"seq_len={args.seq_len}, batch_size={args.batch_size}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    print(f"\nGenerating Stage {args.stage} data...")
    data = get_stage_data(
        stage=args.stage,
        n_train=args.n_train,
        n_test=args.n_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        n_examples=args.n_examples,
        n_rules=args.n_rules,
    )

    if is_stage2:
        train_loader = DataLoader(
            TensorDataset(data['train_seqs'], data['train_targets'],
                          data['train_rules']),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            TensorDataset(data['test_seqs'], data['test_targets'],
                          data['test_rules']),
            batch_size=args.batch_size, shuffle=False, drop_last=False)
        # Oracle z projection (fixed random, shared across train/test)
        torch.manual_seed(0)
        oracle_proj = torch.randn(args.n_rules, args.d_z, device=device) * 0.5
        if args.seed > 0:
            torch.manual_seed(args.seed)
        rule_names = [r[0] for r in DEFAULT_RULES[:args.n_rules]]
        print(f"Rules: {rule_names}")
        print(f"Train with z: {args.train_with_z}")
    else:
        train_loader = DataLoader(
            TensorDataset(data['train_seqs']),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            TensorDataset(data['test_seqs']),
            batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"Train: {data['train_seqs'].shape}, Test: {data['test_seqs'].shape}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = JEPAModel(
        enc_config=enc_config,
        pred_config=pred_config,
        vocab_size=args.vocab_size,
        d_z=args.d_z,
        ema_tau_start=args.ema_tau_start,
        ema_tau_end=args.ema_tau_end,
        jepa_loss_type=args.jepa_loss,
    ).to(device)

    num_params = sum(p.numel() for p in model.get_trainable_params())
    num_params_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {num_params:,} / Total: {num_params_total:,}")

    # --- AMP (mixed precision) ---
    use_amp = (not args.no_amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        def amp_ctx():
            return torch.amp.autocast('cuda', dtype=torch.float16)
    else:
        amp_ctx = nullcontext
    print(f"AMP: {'ON' if use_amp else 'OFF'}")

    # --- torch.compile ---
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: ON")
        except Exception as e:
            print(f"torch.compile: FAILED ({e}), continuing without")
    else:
        print("torch.compile: OFF (use --compile to enable)")

    print(f"JEPA loss: {args.jepa_loss}, Prediction: {prediction_mode}")
    print(f"EMA tau: {args.ema_tau_start} -> {args.ema_tau_end}")
    print(f"LR: {args.lr}, warmup: {args.warmup_epochs} epochs")

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr)

    # Cosine LR schedule with linear warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return 0.2 + 0.8 * epoch / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(
            args.epochs - args.warmup_epochs, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Diagnostics
    diagnostics = JEPADiagnostics(args.stage)
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    best_test_acc = 0.0

    # Profiling accumulators
    prof = {'forward': 0, 'backward': 0, 'ema_update': 0,
            'diagnostics': 0, 'eval': 0, 'n_batches': 0, 'n_epochs': 0}

    if is_stage2:
        # ==================================================================
        # Stage 2: Pattern Induction Training Loop
        # ==================================================================
        print(f"\n{'='*65}")
        print(f"JEPA Stage 2 | Langevin T={args.langevin_T}, "
              f"eta={args.langevin_eta}, sigma={args.langevin_sigma}")
        print(f"{'='*65}")
        print(f"{'Epoch':>5} {'JEPA':>8} {'DecCE':>8} {'TrAcc':>7} "
              f"{'NoZ':>7} {'RndZ':>7} {'Lang':>7} {'Orac':>7} "
              f"{'Gap':>7} {'ms/b':>7} {'ep_s':>6}")
        print("-" * 90)

        s2_history = {
            'no_z': [], 'random_z': [], 'langevin': [], 'oracle': [],
            'gap': [], 'energy_traj': [],
        }

        for epoch in range(1, args.epochs + 1):
            model.train()
            ep_jepa = ep_dec = ep_acc = ep_ms = 0.0
            n_batches = 0
            epoch_t0 = time.perf_counter()

            for batch in train_loader:
                seqs = batch[0].to(device)
                targets = batch[1].to(device)
                rule_idx = batch[2].to(device)
                B = seqs.shape[0]
                t0 = time.perf_counter()

                # Oracle z for training
                z_train = None
                if args.train_with_z:
                    z_train = oracle_proj[rule_idx]

                # --- Forward ---
                _t = time.perf_counter()
                with amp_ctx():
                    result = model.forward_train_nextstep(seqs, z=z_train)
                    L_jepa = result['L_jepa']
                    L_decode = result['L_decode']
                    L_var = result['L_var']
                    L_cov = result['L_cov']

                    L_total = (L_jepa
                               + args.lambda_decode * L_decode
                               + args.lambda_var * L_var
                               + args.lambda_cov * L_cov)
                prof['forward'] += time.perf_counter() - _t

                # --- Backward + update ---
                _t = time.perf_counter()
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(L_total).backward()
                    scaler.unscale_(optimizer)
                    if args.w_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.get_trainable_params(),
                            max_norm=args.w_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    L_total.backward()
                    if args.w_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.get_trainable_params(),
                            max_norm=args.w_clip)
                    optimizer.step()
                prof['backward'] += time.perf_counter() - _t

                # --- EMA update ---
                _t = time.perf_counter()
                model.update_target_encoder()
                prof['ema_update'] += time.perf_counter() - _t

                # --- Accuracy ---
                _t = time.perf_counter()
                with torch.no_grad():
                    logits_shift = model.forward_eval_nextstep(
                        seqs, z=z_train)
                    pred_last = logits_shift[:, -1].argmax(dim=-1)
                    acc = (pred_last == targets).float().mean().item()
                prof['diagnostics'] += time.perf_counter() - _t

                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000

                ep_jepa += L_jepa.item()
                ep_dec += L_decode.item()
                ep_acc += acc
                ep_ms += ms
                n_batches += 1
                prof['n_batches'] += 1

            # --- End-of-epoch ---
            scheduler.step()
            model.set_ema_progress(epoch / args.epochs)

            # Evaluate all 4 ablation variants
            _t = time.perf_counter()
            s2_metrics = evaluate_stage2(
                model, test_loader, device, args.n_rules, oracle_proj,
                langevin_T=args.langevin_T,
                langevin_eta=args.langevin_eta,
                langevin_sigma=args.langevin_sigma,
            )
            prof['eval'] += time.perf_counter() - _t
            prof['n_epochs'] += 1
            epoch_time = time.perf_counter() - epoch_t0
            a = s2_metrics['accuracy']
            gap = s2_metrics['langevin_gap']

            best_test_acc = max(best_test_acc, a['langevin'])

            for v in ['no_z', 'random_z', 'langevin', 'oracle']:
                s2_history[v].append(a[v])
            s2_history['gap'].append(gap)
            if s2_metrics['energy_trajectories']:
                s2_history['energy_traj'].append(
                    s2_metrics['energy_trajectories'][0])

            avg_jepa = ep_jepa / n_batches
            avg_dec = ep_dec / n_batches
            avg_acc = ep_acc / n_batches
            avg_ms = ep_ms / n_batches

            print(f"{epoch:5d} {avg_jepa:8.4f} {avg_dec:8.4f} "
                  f"{avg_acc:7.4f} {a['no_z']:7.4f} "
                  f"{a['random_z']:7.4f} {a['langevin']:7.4f} "
                  f"{a['oracle']:7.4f} {gap:+7.4f} "
                  f"{avg_ms:7.1f} {epoch_time:6.1f}")

            if args.profile and epoch == 5:
                _print_profile(prof)
                print("(--profile mode: stopping after 5 epochs)")
                break

            if epoch % args.plot_every == 0 or epoch == args.epochs:
                config_str = (
                    f'(Langevin T={args.langevin_T}, '
                    f'eta={args.langevin_eta})')
                chart_path = os.path.join(
                    save_dir,
                    f'jepa_s2_epoch_{epoch:03d}.png')
                plot_stage2(chart_path, epoch, config_str, s2_history,
                            s2_metrics, rule_names, args)

        # --- Stage 2 Summary ---
        final_epoch = min(epoch, args.epochs)
        print(f"\n{'='*65}")
        print(f"Stage 2 Final Results (Epoch {final_epoch})")
        print(f"{'='*65}")
        a = s2_metrics['accuracy']
        print(f"  No z:      {a['no_z']:.4f}")
        print(f"  Random z:  {a['random_z']:.4f}")
        print(f"  Langevin:  {a['langevin']:.4f}")
        print(f"  Oracle:    {a['oracle']:.4f}")
        print(f"  Langevin gap: {gap:+.4f}")
        print()
        print(f"Per-rule accuracy (Langevin):")
        for r, name in enumerate(rule_names):
            ra = s2_metrics['rule_accuracy']['langevin'][r]
            print(f"  {name:>12s}: {ra:.4f} "
                  f"(n={s2_metrics['rule_total'][r]})")
        print()
        if gap > 0.10:
            print("PASS: Langevin gap > 10%!")
        elif gap > 0.02:
            print(f"PROMISING: Langevin gap {gap:.1%}")
        elif gap > 0:
            print(f"MARGINAL: Langevin gap {gap:.1%}")
        else:
            print(f"NO EFFECT: Langevin gap {gap:.1%}")

    else:
        # ==================================================================
        # Stage 1: Sequence Prediction Training Loop
        # ==================================================================
        print(f"\n{'='*65}")
        print(f"JEPA Stage {args.stage} | {prediction_mode} | "
              f"JEPA loss: {args.jepa_loss}")
        print(f"{'='*65}")
        print(f"{'Epoch':>5} {'JEPA':>10} {'Decode':>10} {'Acc':>10} "
              f"{'TestAcc':>10} {'ms/b':>8} {'ep_s':>6}")
        print("-" * 65)

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_t0 = time.perf_counter()
            ep_jepa = 0.0
            ep_decode = 0.0
            ep_acc = 0.0
            ep_ms = 0.0
            n_batches = 0

            for batch in train_loader:
                seqs = batch[0].to(device)
                B = seqs.shape[0]
                t0 = time.perf_counter()

                # --- Mask (only for masked mode) ---
                mask = None
                if prediction_mode == 'masked':
                    if args.mask_mode == 'causal':
                        mask = generate_causal_mask(
                            B, seqs.shape[1], args.mask_ratio, device)
                    else:
                        mask = generate_mask(
                            B, seqs.shape[1], args.mask_ratio, device)

                # --- Forward ---
                _t = time.perf_counter()
                with amp_ctx():
                    if prediction_mode == 'next_step':
                        result = model.forward_train_nextstep(seqs, z=None)
                    else:
                        result = model.forward_train(seqs, mask, z=None)
                    L_jepa = result['L_jepa']
                    L_decode = result['L_decode']
                    L_var = result['L_var']
                    L_cov = result['L_cov']

                    L_total = (L_jepa
                               + args.lambda_decode * L_decode
                               + args.lambda_var * L_var
                               + args.lambda_cov * L_cov)
                prof['forward'] += time.perf_counter() - _t

                # --- Backward + update ---
                _t = time.perf_counter()
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(L_total).backward()
                    scaler.unscale_(optimizer)
                    if args.w_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.get_trainable_params(),
                            max_norm=args.w_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    L_total.backward()
                    if args.w_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.get_trainable_params(),
                            max_norm=args.w_clip)
                    optimizer.step()
                prof['backward'] += time.perf_counter() - _t

                # --- EMA update ---
                _t = time.perf_counter()
                model.update_target_encoder()
                prof['ema_update'] += time.perf_counter() - _t

                # --- Accuracy ---
                _t = time.perf_counter()
                with torch.no_grad(), amp_ctx():
                    if prediction_mode == 'next_step':
                        logits_shift = model.decode(
                            result['s_pred'][:, :-1])
                        tokens_shift = seqs[:, 1:]
                        acc = compute_nextstep_accuracy(
                            logits_shift, tokens_shift)
                    else:
                        logits = model.decode(result['s_pred'])
                        acc = compute_masked_accuracy(logits, seqs, mask)
                prof['diagnostics'] += time.perf_counter() - _t

                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000

                diagnostics.update_train(
                    L_jepa.item(), L_decode.item(), acc,
                    L_var.item(), L_cov.item(), ms)

                ep_jepa += L_jepa.item()
                ep_decode += L_decode.item()
                ep_acc += acc
                ep_ms += ms
                n_batches += 1
                prof['n_batches'] += 1

            # --- End-of-epoch ---
            scheduler.step()
            model.set_ema_progress(epoch / args.epochs)

            # Evaluate
            _t = time.perf_counter()
            if prediction_mode == 'next_step':
                test_metrics = evaluate_stage1_nextstep(
                    model, test_loader, device, amp_ctx)
            else:
                test_metrics = evaluate_stage1_masked(
                    model, test_loader, args.mask_ratio, device, amp_ctx,
                    mask_mode=args.mask_mode)
            prof['eval'] += time.perf_counter() - _t
            prof['n_epochs'] += 1

            diagnostics.update_test(test_metrics)
            best_test_acc = max(best_test_acc, test_metrics['accuracy'])

            avg_jepa = ep_jepa / n_batches
            avg_dec = ep_decode / n_batches
            avg_acc = ep_acc / n_batches
            avg_ms = ep_ms / n_batches

            epoch_time = time.perf_counter() - epoch_t0
            print(f"{epoch:5d} {avg_jepa:10.4f} {avg_dec:10.4f} "
                  f"{avg_acc:10.4f} {test_metrics['accuracy']:10.4f} "
                  f"{avg_ms:8.1f} {epoch_time:6.1f}")

            if args.profile and epoch == 5:
                _print_profile(prof)
                print("(--profile mode: stopping after 5 epochs)")
                break

            # Plot
            if epoch % args.plot_every == 0 or epoch == args.epochs:
                config_str = (f'(loss={args.jepa_loss}, '
                              f'pred={prediction_mode})')
                chart_path = os.path.join(
                    save_dir,
                    f'jepa_s{args.stage}_{prediction_mode}'
                    f'_epoch_{epoch:03d}.png')
                diagnostics.plot(chart_path, epoch, config_str)

        # --- Stage 1 Summary ---
        print(f"\nBest test accuracy: {best_test_acc:.4f}")
        if best_test_acc >= 0.80:
            print(f"PASS: Stage {args.stage} JEPA backbone works!")
        elif best_test_acc >= 0.40:
            print(f"PROMISING: Stage {args.stage} at {best_test_acc:.1%}")
        else:
            print(f"Needs work: {best_test_acc:.1%}")


if __name__ == '__main__':
    main()
