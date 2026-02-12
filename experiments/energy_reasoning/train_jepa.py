"""
JEPA training for energy-based reasoning experiments.

Stage 1 (--stage 1a/1b/1c): Masked prediction on structured sequences.
  Validates that the JEPA encoder learns useful representations.
  No z, no Langevin — pure self-supervised backbone training.

Stage 2 (--stage 2): Pattern induction (few-shot rule discovery).
  Validates that z + Langevin improve predictions beyond single-pass.
  Supports four ablation modes:
    --z_mode none      : No z (baseline)
    --z_mode random    : Random z, not optimized
    --z_mode langevin  : Langevin-optimized z (full system)
    --z_mode oracle    : z = projected rule label (upper bound)

Usage:
  # Stage 1a: arithmetic sequences
  python experiments/energy_reasoning/train_jepa.py --stage 1a

  # Stage 2: pattern induction with Langevin
  python experiments/energy_reasoning/train_jepa.py --stage 2 --z_mode langevin

  # Stage 2: ablation comparison
  python experiments/energy_reasoning/train_jepa.py --stage 2 --z_mode none
  python experiments/energy_reasoning/train_jepa.py --stage 2 --z_mode langevin
"""

import argparse
import os
import sys
import time

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
    get_stage_data, generate_mask, generate_last_token_mask,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_masked_accuracy(logits, tokens, mask):
    """Token-level accuracy on masked positions only.

    Args:
        logits: (batch, seq_len, vocab_size).
        tokens: (batch, seq_len) ground truth.
        mask: (batch, seq_len) boolean.

    Returns:
        Accuracy (float) on masked positions.
    """
    preds = logits.argmax(dim=-1)  # (batch, seq_len)
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == tokens[mask]).float().mean().item()


def evaluate_stage1(model, test_loader, mask_ratio, device):
    """Evaluate Stage 1: masked prediction without z."""
    model.eval()
    total_jepa = 0.0
    total_decode = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            seqs = batch[0].to(device)
            mask = generate_mask(seqs.shape[0], seqs.shape[1], mask_ratio, device)

            result = model.forward_train(seqs, mask, z=None)
            total_jepa += result['L_jepa'].item()
            total_decode += result['L_decode'].item()

            logits = model.decode(result['s_pred'])
            total_acc += compute_masked_accuracy(logits, seqs, mask)
            n_batches += 1

    return {
        'jepa_loss': total_jepa / n_batches,
        'decode_loss': total_decode / n_batches,
        'accuracy': total_acc / n_batches,
    }


def evaluate_stage2(model, test_loader, langevin, z_mode, device,
                    d_z, n_rules):
    """Evaluate Stage 2: pattern induction with ablation modes.

    Args:
        model: JEPAModel.
        test_loader: DataLoader yielding (seqs, targets, rule_indices).
        langevin: LangevinDynamics instance (used if z_mode='langevin').
        z_mode: 'none', 'random', 'langevin', or 'oracle'.
        device: Torch device.
        d_z: Latent dimension.
        n_rules: Number of rules (for oracle mode).

    Returns:
        dict with accuracy, jepa_loss, per-step energies (if langevin).
    """
    model.eval()
    total_acc = 0.0
    total_jepa = 0.0
    all_energies = []
    n_batches = 0

    for batch in test_loader:
        seqs = batch[0].to(device)
        targets = batch[1].to(device)
        rule_indices = batch[2].to(device)
        B = seqs.shape[0]

        mask = generate_last_token_mask(B, seqs.shape[1], device)

        # Choose z based on mode
        if z_mode == 'none':
            z = None
        elif z_mode == 'random':
            z = torch.randn(B, d_z, device=device)
        elif z_mode == 'langevin':
            with torch.no_grad():
                s_context = model.encode(seqs, mask=mask)
                s_target = model.encode_target(seqs)
            # Langevin needs grad for z
            with torch.enable_grad():
                z, energies = langevin.sample(
                    model.predictor, s_context, s_target, mask, device)
            all_energies.append(energies)
        elif z_mode == 'oracle':
            # One-hot rule index projected to d_z
            one_hot = F.one_hot(rule_indices, n_rules).float()
            # Simple linear projection (not learned — fixed random)
            if not hasattr(model, '_oracle_proj'):
                model._oracle_proj = torch.randn(
                    n_rules, d_z, device=device) * 0.5
            z = one_hot @ model._oracle_proj
        else:
            raise ValueError(f"Unknown z_mode: {z_mode}")

        with torch.no_grad():
            if z_mode == 'langevin':
                # Already have s_context, s_target
                s_pred = model.predict(s_context, z)
                logits = model.decode(s_pred)
            else:
                logits = model.forward_eval(seqs, mask, z)

        # Accuracy on answer position (last token)
        pred_tokens = logits[:, -1, :].argmax(dim=-1)
        acc = (pred_tokens == targets).float().mean().item()
        total_acc += acc
        n_batches += 1

    result = {
        'accuracy': total_acc / n_batches,
    }
    if all_energies:
        # Average energy per step across batches
        max_steps = max(len(e) for e in all_energies)
        avg_energies = []
        for t in range(max_steps):
            vals = [e[t] for e in all_energies if t < len(e)]
            avg_energies.append(np.mean(vals))
        result['energy_trajectory'] = avg_energies
    return result


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
        # Per-batch
        self.train_jepa = []
        self.train_decode = []
        self.train_acc = []
        self.train_var = []
        self.train_cov = []
        self.ms_per_batch = []
        self.langevin_energies = []  # list of energy trajectories
        # Per-epoch
        self.test_jepa = []
        self.test_decode = []
        self.test_acc = []

    def update_train(self, jepa_loss, decode_loss, acc, L_var, L_cov, ms,
                     langevin_energies=None):
        self.train_jepa.append(jepa_loss)
        self.train_decode.append(decode_loss)
        self.train_acc.append(acc)
        self.train_var.append(L_var)
        self.train_cov.append(L_cov)
        self.ms_per_batch.append(ms)
        if langevin_energies is not None:
            self.langevin_energies.append(langevin_energies)

    def update_test(self, metrics):
        self.test_jepa.append(metrics.get('jepa_loss', 0))
        self.test_decode.append(metrics.get('decode_loss', 0))
        self.test_acc.append(metrics.get('accuracy', 0))

    def plot(self, save_path, epoch, config_str=''):
        is_stage2 = self.stage == '2'
        nrows = 3 if is_stage2 else 2
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
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

        # --- Row 0: Loss, Accuracy, Speed ---

        # [0,0] JEPA loss
        ax = axes[0, 0]
        if self.test_jepa:
            ax.plot(epochs, self.test_jepa, 'r--', label='Test', linewidth=2)
        train_avg = _epoch_avg(self.train_jepa, n_ep)
        if train_avg:
            ax.plot(epochs[:len(train_avg)], train_avg, 'b-',
                    label='Train', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L2 Loss')
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
        ax.set_title('Masked Token Accuracy')
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

        # --- Row 1: VICReg, Decode Loss, Summary ---

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

        # --- Row 2: Langevin diagnostics (Stage 2 only) ---
        if is_stage2:
            # [2,0] Langevin energy trajectory (recent batches)
            ax = axes[2, 0]
            if self.langevin_energies:
                recent = self.langevin_energies[-50:]
                for i, traj in enumerate(recent):
                    alpha = 0.1 + 0.9 * (i / len(recent))
                    ax.plot(range(len(traj)), traj, alpha=alpha,
                            linewidth=0.8, color='blue')
                # Average trajectory
                max_t = max(len(e) for e in recent)
                avg = []
                for t in range(max_t):
                    vals = [e[t] for e in recent if t < len(e)]
                    avg.append(np.mean(vals))
                ax.plot(range(len(avg)), avg, 'r-', linewidth=2,
                        label='Mean')
                ax.legend()
            ax.set_xlabel('Langevin Step')
            ax.set_ylabel('Energy')
            ax.set_title('Langevin Energy per Step')
            ax.grid(True, alpha=0.3)

            # [2,1] Accuracy over training
            ax = axes[2, 1]
            if self.train_acc:
                ax.plot(self.train_acc, alpha=0.3, linewidth=0.5)
                if len(self.train_acc) > 50:
                    w = min(50, len(self.train_acc) // 5)
                    ma = np.convolve(self.train_acc,
                                     np.ones(w) / w, mode='valid')
                    ax.plot(range(w - 1, len(self.train_acc)), ma,
                            linewidth=1.5, color='red', label=f'MA-{w}')
                    ax.legend(fontsize=8)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training Accuracy (per batch)')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            # [2,2] Langevin steps used
            ax = axes[2, 2]
            if self.langevin_energies:
                steps_used = [len(e) for e in self.langevin_energies]
                ax.plot(steps_used, alpha=0.5, linewidth=0.5)
                if len(steps_used) > 20:
                    w = min(20, len(steps_used) // 5)
                    ma = np.convolve(steps_used,
                                     np.ones(w) / w, mode='valid')
                    ax.plot(range(w - 1, len(steps_used)), ma,
                            linewidth=1.5, color='red')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Steps')
            ax.set_title('Langevin Steps (adaptive)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f'  [Saved {save_path}]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='JEPA energy reasoning training')

    # Stage
    parser.add_argument('--stage', type=str, default='1a',
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
                        help='Fraction of positions to mask (Stage 1)')
    parser.add_argument('--n_examples', type=int, default=5,
                        help='Example pairs in pattern induction (Stage 2)')
    parser.add_argument('--n_rules', type=int, default=5,
                        help='Number of rules (Stage 2)')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--w_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')

    # JEPA / VICReg
    parser.add_argument('--ema_tau', type=float, default=0.996,
                        help='Target encoder EMA coefficient')
    parser.add_argument('--lambda_decode', type=float, default=0.1,
                        help='Decode CE loss weight')
    parser.add_argument('--lambda_var', type=float, default=1.0,
                        help='VICReg variance loss weight')
    parser.add_argument('--lambda_cov', type=float, default=0.04,
                        help='VICReg covariance loss weight')

    # z / Langevin (Stage 2)
    parser.add_argument('--z_mode', type=str, default='none',
                        choices=['none', 'random', 'langevin', 'oracle'],
                        help='How z is computed during training')
    parser.add_argument('--langevin_T', type=int, default=5,
                        help='Langevin steps')
    parser.add_argument('--langevin_eta', type=float, default=0.01,
                        help='Langevin step size')
    parser.add_argument('--langevin_sigma', type=float, default=0.1,
                        help='Langevin max noise scale')
    parser.add_argument('--langevin_threshold', type=float, default=1e-3,
                        help='Langevin adaptive stopping threshold')

    # Misc
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot_every', type=int, default=5)
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
    # Config
    # -----------------------------------------------------------------------

    # For Stage 2, seq_len = 2*n_examples + 2, padded to chunk-aligned size
    if args.stage == '2':
        task_len = 2 * args.n_examples + 2
        # Round up to nearest power of 2 for chunk alignment (min 8)
        seq_len = max(8, 1 << (task_len - 1).bit_length())
        args.seq_len = seq_len
        print(f"Stage 2: task_len={task_len}, padded seq_len={seq_len}")

    # Chunk size must divide seq_len
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

    if args.stage == '2':
        train_loader = DataLoader(
            TensorDataset(data['train_seqs'], data['train_targets'],
                          data['train_rules']),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            TensorDataset(data['test_seqs'], data['test_targets'],
                          data['test_rules']),
            batch_size=args.batch_size, shuffle=False, drop_last=True)
    else:
        train_loader = DataLoader(
            TensorDataset(data['train_seqs']),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            TensorDataset(data['test_seqs']),
            batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"Train: {data['train_seqs'].shape}, Test: {data['test_seqs'].shape}")
    print(f"Seq example: {data['train_seqs'][0].tolist()}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    # Stage 1: z_mode forced to 'none'
    z_mode = args.z_mode if args.stage == '2' else 'none'
    use_z = z_mode != 'none'

    model = JEPAModel(
        enc_config=enc_config,
        pred_config=pred_config,
        vocab_size=args.vocab_size,
        d_z=args.d_z,
        ema_tau=args.ema_tau,
    ).to(device)

    num_params = sum(p.numel() for p in model.get_trainable_params())
    num_params_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {num_params:,} / Total: {num_params_total:,}")

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr)

    # Langevin sampler (Stage 2)
    langevin = LangevinDynamics(
        d_z=args.d_z,
        eta=args.langevin_eta,
        sigma_max=args.langevin_sigma,
        T=args.langevin_T,
        adaptive_threshold=args.langevin_threshold,
    ) if z_mode == 'langevin' else None

    # Diagnostics
    diagnostics = JEPADiagnostics(args.stage, z_mode)
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    print(f"\nStage {args.stage}, z_mode={z_mode}")
    print(f"{'Epoch':>5} {'JEPA':>10} {'Decode':>10} {'Acc':>10} "
          f"{'TestAcc':>10} {'ms/b':>8}")
    print("-" * 58)

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_jepa = 0.0
        ep_decode = 0.0
        ep_acc = 0.0
        ep_ms = 0.0
        n_batches = 0

        for batch in train_loader:
            seqs = batch[0].to(device)
            B = seqs.shape[0]

            t0 = time.perf_counter()

            # --- Generate mask ---
            if args.stage == '2':
                mask = generate_last_token_mask(B, seqs.shape[1], device)
            else:
                mask = generate_mask(B, seqs.shape[1], args.mask_ratio, device)

            # --- Compute z ---
            z = None
            langevin_e = None

            if z_mode == 'random':
                z = torch.randn(B, args.d_z, device=device)

            elif z_mode == 'langevin':
                # Pre-encode (detached) for Langevin search
                with torch.no_grad():
                    s_ctx = model.encode(seqs, mask=mask)
                    s_tgt = model.encode_target(seqs)
                with torch.enable_grad():
                    z, langevin_e = langevin.sample(
                        model.predictor, s_ctx, s_tgt, mask, device)

            elif z_mode == 'oracle':
                rule_indices = batch[2].to(device)
                one_hot = F.one_hot(rule_indices, args.n_rules).float()
                if not hasattr(model, '_oracle_proj'):
                    model._oracle_proj = torch.randn(
                        args.n_rules, args.d_z, device=device) * 0.5
                z = one_hot @ model._oracle_proj

            # --- Forward pass ---
            result = model.forward_train(seqs, mask, z)
            L_jepa = result['L_jepa']
            L_decode = result['L_decode']
            L_var = result['L_var']
            L_cov = result['L_cov']

            L_total = (L_jepa
                       + args.lambda_decode * L_decode
                       + args.lambda_var * L_var
                       + args.lambda_cov * L_cov)

            # --- Backward + update ---
            optimizer.zero_grad()
            L_total.backward()
            if args.w_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_params(), max_norm=args.w_clip)
            optimizer.step()

            # --- EMA update ---
            model.update_target_encoder()

            # --- Accuracy ---
            with torch.no_grad():
                logits = model.decode(result['s_pred'])
                if args.stage == '2':
                    targets = batch[1].to(device)
                    pred_tok = logits[:, -1, :].argmax(dim=-1)
                    acc = (pred_tok == targets).float().mean().item()
                else:
                    acc = compute_masked_accuracy(logits, seqs, mask)

            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000

            # --- Diagnostics ---
            diagnostics.update_train(
                L_jepa.item(), L_decode.item(), acc,
                L_var.item(), L_cov.item(), ms,
                langevin_energies=langevin_e)

            ep_jepa += L_jepa.item()
            ep_decode += L_decode.item()
            ep_acc += acc
            ep_ms += ms
            n_batches += 1

        # --- End-of-epoch evaluation ---
        if args.stage == '2':
            test_metrics = evaluate_stage2(
                model, test_loader, langevin, z_mode, device,
                args.d_z, args.n_rules)
        else:
            test_metrics = evaluate_stage1(
                model, test_loader, args.mask_ratio, device)

        diagnostics.update_test(test_metrics)
        best_test_acc = max(best_test_acc, test_metrics['accuracy'])

        avg_jepa = ep_jepa / n_batches
        avg_dec = ep_decode / n_batches
        avg_acc = ep_acc / n_batches
        avg_ms = ep_ms / n_batches

        print(f"{epoch:5d} {avg_jepa:10.4f} {avg_dec:10.4f} "
              f"{avg_acc:10.4f} {test_metrics['accuracy']:10.4f} "
              f"{avg_ms:8.1f}")

        # Energy trajectory logging (Stage 2 + Langevin)
        if z_mode == 'langevin' and epoch % 5 == 0:
            et = test_metrics.get('energy_trajectory', [])
            if et:
                e_str = ' → '.join(f'{e:.4f}' for e in et)
                print(f"  Langevin energy: {e_str}")

        # Save plot
        if epoch % args.plot_every == 0 or epoch == args.epochs:
            config_str = (f'(d={args.d_model}, pred_d={args.d_pred}, '
                          f'z={z_mode})')
            chart_path = os.path.join(
                save_dir,
                f'jepa_s{args.stage}_{z_mode}_epoch_{epoch:03d}.png')
            diagnostics.plot(chart_path, epoch, config_str)

    # --- Summary ---
    print(f"\nBest test accuracy: {best_test_acc:.4f}")

    if args.stage == '2':
        if best_test_acc >= 0.90:
            print(f"PASS: {z_mode} achieves {best_test_acc:.1%} on "
                  f"pattern induction!")
        elif best_test_acc >= 0.50:
            print(f"PROMISING: {z_mode} at {best_test_acc:.1%}")
        else:
            chance = 1.0 / args.vocab_size
            print(f"Needs work: {best_test_acc:.1%} "
                  f"(chance={chance:.1%})")
    else:
        if best_test_acc >= 0.80:
            print(f"PASS: Stage {args.stage} JEPA backbone works!")
        elif best_test_acc >= 0.40:
            print(f"PROMISING: Stage {args.stage} at {best_test_acc:.1%}")
        else:
            print(f"Needs work: {best_test_acc:.1%} on Stage {args.stage}")


if __name__ == '__main__':
    main()
