"""
ePC-JEPA training: predictive coding encoder + JEPA framework.

The first-ever combination of error-based predictive coding with
Joint Embedding Predictive Architecture. The encoder learns LOCALLY
(each block from its own prediction error), while the predictor and
decoder learn GLOBALLY (from JEPA representation loss and decode CE).

Stage 1 (--stage 1a/1b/1c): Sequence prediction validation.
  Next-step prediction mode (aligned with Mamba's causal nature).
  Validates that ePC can train an encoder to produce useful JEPA
  representations via local learning only.

Training modes:
  --ipc         : Incremental PC (interleave error + weight steps)
  (default)     : Standard ePC (T error steps, then 1 weight step)

Usage:
  # Stage 1b: multi-rule with ePC-JEPA
  python experiments/energy_reasoning/train_epc_jepa.py --stage 1b

  # With iPC mode
  python experiments/energy_reasoning/train_epc_jepa.py --stage 1b --ipc

  # Compare with standard JEPA baseline
  python experiments/energy_reasoning/train_jepa.py --stage 1b --prediction_mode next_step
"""

import argparse
import math
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
from experiments.energy_reasoning.epc_jepa_model import ePCJEPAModel
from experiments.energy_reasoning.jepa_model import vicreg_loss
from experiments.energy_reasoning.data_gen import get_stage_data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, device):
    """Evaluate next-step prediction accuracy."""
    model.eval()
    total_acc = 0.0
    total_jepa = 0.0
    total_decode = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            seqs = batch[0].to(device)

            # Target encoder
            s_target = model.encode_target(seqs)

            # Standard forward (no errors)
            logits_shift = model.forward_eval(seqs)  # (B, seq_len-1, V)
            tokens_shift = seqs[:, 1:]

            # Accuracy
            preds = logits_shift.argmax(dim=-1)
            acc = (preds == tokens_shift).float().mean().item()
            total_acc += acc

            # JEPA loss (for monitoring)
            x = model.embedding(seqs)
            for layer in model.layers:
                x = layer(x)
            s_ctx = model.out_norm(x)
            s_pred = model.predictor(s_ctx)
            total_jepa += model._jepa_loss_nextstep(s_pred, s_target).item()

            # Decode loss
            total_decode += F.cross_entropy(
                logits_shift.reshape(-1, model.vocab_size),
                tokens_shift.reshape(-1),
            ).item()

            n_batches += 1

    return {
        'accuracy': total_acc / n_batches,
        'jepa_loss': total_jepa / n_batches,
        'decode_loss': total_decode / n_batches,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class ePCJEPADiagnostics:
    """Collect and plot training diagnostics."""

    def __init__(self, stage):
        self.stage = stage
        self.reset()

    def reset(self):
        # Per-batch
        self.train_energy_init = []
        self.train_energy_final = []
        self.train_jepa = []
        self.train_decode = []
        self.train_acc = []
        self.train_error_norms = []
        self.ms_per_batch = []
        self.actual_iters = []
        self.rep_std = []  # representation std (collapse monitor)
        self.train_var = []   # VICReg variance loss
        self.train_cov = []   # VICReg covariance loss
        # Per-epoch
        self.test_acc = []
        self.test_jepa = []
        self.test_decode = []

    def update_train(self, diag, jepa_loss, decode_loss, acc, ms,
                     rep_std=0.0, L_var=0.0, L_cov=0.0):
        self.train_energy_init.append(diag['E_initial'])
        self.train_energy_final.append(diag['E_final'])
        self.train_jepa.append(jepa_loss)
        self.train_decode.append(decode_loss)
        self.train_acc.append(acc)
        self.train_error_norms.append(diag['error_norms'])
        self.ms_per_batch.append(ms)
        self.actual_iters.append(diag['actual_iters'])
        self.rep_std.append(rep_std)
        self.train_var.append(L_var)
        self.train_cov.append(L_cov)

    def update_test(self, metrics):
        self.test_acc.append(metrics['accuracy'])
        self.test_jepa.append(metrics.get('jepa_loss', 0))
        self.test_decode.append(metrics.get('decode_loss', 0))

    def plot(self, save_path, epoch, config_str=''):
        fig, axes = plt.subplots(3, 3, figsize=(16, 15))
        fig.suptitle(
            f'ePC-JEPA Stage {self.stage} - Epoch {epoch} {config_str}',
            fontsize=13)

        epochs = list(range(1, len(self.test_acc) + 1))

        def _epoch_avg(data, n_epochs):
            n_per = len(data) // max(n_epochs, 1)
            if n_per == 0:
                return []
            return [np.mean(data[i * n_per:(i + 1) * n_per])
                    for i in range(n_epochs)]

        n_ep = len(epochs)

        # --- Row 0: Accuracy, JEPA loss, Decode loss ---

        # [0,0] Accuracy
        ax = axes[0, 0]
        if self.test_acc:
            ax.plot(epochs, self.test_acc, 'r--', label='Test', linewidth=2)
        train_avg = _epoch_avg(self.train_acc, n_ep)
        if train_avg:
            ax.plot(epochs[:len(train_avg)], train_avg, 'b-',
                    label='Train', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Next-Step Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [0,1] JEPA loss
        ax = axes[0, 1]
        if self.test_jepa:
            ax.plot(epochs, self.test_jepa, 'r--', label='Test', linewidth=2)
        jepa_avg = _epoch_avg(self.train_jepa, n_ep)
        if jepa_avg:
            ax.plot(epochs[:len(jepa_avg)], jepa_avg, 'b-',
                    label='Train', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('JEPA Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [0,2] Decode loss
        ax = axes[0, 2]
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

        # --- Row 1: Energy, Error norms, Speed ---

        # [1,0] Energy (init vs final)
        ax = axes[1, 0]
        ei_avg = _epoch_avg(self.train_energy_init, n_ep)
        ef_avg = _epoch_avg(self.train_energy_final, n_ep)
        if ei_avg:
            ax.plot(epochs[:len(ei_avg)], ei_avg, 'b-',
                    label='E_initial', linewidth=2)
        if ef_avg:
            ax.plot(epochs[:len(ef_avg)], ef_avg, 'r-',
                    label='E_final', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Energy')
        ax.set_title('Error Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [1,1] Per-layer error norms
        ax = axes[1, 1]
        if self.train_error_norms:
            n_layers = len(self.train_error_norms[0])
            for layer_idx in range(n_layers):
                norms = [en[layer_idx] for en in self.train_error_norms
                         if layer_idx < len(en)]
                norms_avg = _epoch_avg(norms, n_ep)
                if norms_avg:
                    ax.plot(epochs[:len(norms_avg)], norms_avg,
                            label=f'Layer {layer_idx}', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('||e_i||')
        ax.set_title('Error Norms per Layer')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [1,2] Speed + iterations
        ax = axes[1, 2]
        ms_avg = _epoch_avg(self.ms_per_batch, n_ep)
        if ms_avg:
            ax.plot(epochs[:len(ms_avg)], ms_avg, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ms/batch')
        ax.set_title('Speed')
        ax.grid(True, alpha=0.3)

        # --- Row 2: Representation health, Convergence, Summary ---

        # [2,0] Representation std (collapse monitor)
        ax = axes[2, 0]
        std_avg = _epoch_avg(self.rep_std, n_ep)
        if std_avg:
            ax.plot(epochs[:len(std_avg)], std_avg, 'purple', linewidth=2)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                        label='Target std=1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Std')
        ax.set_title('Representation Std (collapse=0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [2,1] Energy convergence (E_init - E_final)
        ax = axes[2, 1]
        conv = [ei - ef for ei, ef in zip(
            self.train_energy_init, self.train_energy_final)]
        conv_avg = _epoch_avg(conv, n_ep)
        if conv_avg:
            ax.plot(epochs[:len(conv_avg)], conv_avg, 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('E_init - E_final')
        ax.set_title('Error Convergence')
        ax.grid(True, alpha=0.3)

        # [2,2] VICReg Components
        ax = axes[2, 2]
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

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f'  [Saved {save_path}]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='ePC-JEPA training (predictive coding + JEPA)')

    # Stage
    parser.add_argument('--stage', type=str, default='1b',
                        choices=['1a', '1b', '1c'],
                        help='Experiment stage')

    # Architecture
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--d_pred', type=int, default=64,
                        help='Predictor bottleneck dimension')
    parser.add_argument('--n_pred_layer', type=int, default=2)
    parser.add_argument('--d_z', type=int, default=64)

    # Data
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)

    # ePC parameters
    parser.add_argument('--iters', type=int, default=5,
                        help='Error optimization iterations (T)')
    parser.add_argument('--e_lr', type=float, default=0.1,
                        help='Error learning rate (PC literature uses 0.1-1.0)')
    parser.add_argument('--error_optim', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--precision_mode', type=str, default='geometric',
                        choices=['none', 'linear', 'geometric'])
    parser.add_argument('--precision_base', type=float, default=3.0)
    parser.add_argument('--early_stop_rtol', type=float, default=1e-3,
                        help='Early stopping tolerance for error energy '
                             '(0 to disable)')
    parser.add_argument('--min_iters', type=int, default=2,
                        help='Minimum error iterations before early stopping')
    parser.add_argument('--ipc', action='store_true',
                        help='Use incremental PC (interleaved steps)')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Weight optimizer learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=4)
    parser.add_argument('--w_clip', type=float, default=1.0)

    # JEPA
    parser.add_argument('--ema_tau_start', type=float, default=0.996)
    parser.add_argument('--ema_tau_end', type=float, default=1.0)
    parser.add_argument('--jepa_loss', type=str, default='cosine',
                        choices=['cosine', 'l2'])
    parser.add_argument('--lambda_decode', type=float, default=1.0)
    parser.add_argument('--lambda_var', type=float, default=1.0,
                        help='VICReg variance loss weight')
    parser.add_argument('--lambda_cov', type=float, default=0.04,
                        help='VICReg covariance loss weight')

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

    print(f"Encoder: d={enc_config.d_model}, layers={enc_config.n_layer}")
    print(f"Predictor: d={pred_config.d_model}, layers={pred_config.n_layer}")
    print(f"ePC: T={args.iters}, e_lr={args.e_lr}, "
          f"optim={args.error_optim}, prec={args.precision_mode}")
    print(f"Early stop: rtol={args.early_stop_rtol}, "
          f"min_iters={args.min_iters}")

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
    )
    train_loader = DataLoader(
        TensorDataset(data['train_seqs']),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        TensorDataset(data['test_seqs']),
        batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"Train: {data['train_seqs'].shape}, Test: {data['test_seqs'].shape}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = ePCJEPAModel(
        enc_config=enc_config,
        pred_config=pred_config,
        vocab_size=args.vocab_size,
        d_z=args.d_z,
        iters=args.iters,
        e_lr=args.e_lr,
        error_optim=args.error_optim,
        precision_mode=args.precision_mode,
        precision_base=args.precision_base,
        ema_tau_start=args.ema_tau_start,
        ema_tau_end=args.ema_tau_end,
        jepa_loss_type=args.jepa_loss,
        lambda_decode=args.lambda_decode,
        lambda_var=args.lambda_var,
        lambda_cov=args.lambda_cov,
    ).to(device)

    trainable = sum(p.numel() for p in model.get_trainable_params())
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / Total: {total:,}")
    print(f"Precisions: {[f'{p:.2f}' for p in model.precisions]}")
    print(f"Energy scale: {model.energy_scale:.4f}")

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr)

    # Cosine LR with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return 0.2 + 0.8 * epoch / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(
            args.epochs - args.warmup_epochs, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Diagnostics
    diagnostics = ePCJEPADiagnostics(args.stage)
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    mode_str = 'iPC' if args.ipc else 'ePC'
    print(f"\n{'='*65}")
    print(f"ePC-JEPA Stage {args.stage} | {mode_str} | "
          f"T={args.iters} | e_lr={args.e_lr}")
    print(f"{'='*65}")
    print(f"{'Epoch':>5} {'E_init':>8} {'E_fin':>8} {'JEPA':>8} "
          f"{'DecCE':>8} {'Acc':>8} {'TestAcc':>8} {'ms/b':>7} {'iters':>5}")
    print("-" * 70)

    best_test_acc = 0.0

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_ei = 0.0
        ep_ef = 0.0
        ep_jepa = 0.0
        ep_dec = 0.0
        ep_acc = 0.0
        ep_ms = 0.0
        ep_iters = 0.0
        n_batches = 0

        for batch in train_loader:
            seqs = batch[0].to(device)
            B = seqs.shape[0]
            t0 = time.perf_counter()

            # --- Encode target (EMA, no grad) ---
            s_target = model.encode_target(seqs)

            # --- Phase 1 + Phase 2 ---
            if args.ipc:
                model.ipc_train_step(
                    seqs, s_target, optimizer, B, w_clip=args.w_clip,
                    early_stop_rtol=args.early_stop_rtol,
                    min_iters=args.min_iters)
            else:
                # Phase 1: error optimization
                model.minimize_error_energy(
                    seqs, s_target,
                    early_stop_rtol=args.early_stop_rtol,
                    min_iters=args.min_iters)

                # Phase 2: weight optimization
                optimizer.zero_grad()
                w_loss = model.compute_weight_loss(seqs, s_target, B)
                w_loss.backward()
                if args.w_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.get_trainable_params(), max_norm=args.w_clip)
                optimizer.step()

            # --- Phase 3: EMA update ---
            model.update_target()

            # --- Diagnostics ---
            diag = model.get_diagnostics()

            with torch.no_grad():
                # Accuracy
                logits_shift = model.forward_eval(seqs)
                tokens_shift = seqs[:, 1:]
                preds = logits_shift.argmax(dim=-1)
                acc = (preds == tokens_shift).float().mean().item()

                # JEPA loss (monitoring)
                x = model.embedding(seqs)
                for layer in model.layers:
                    x = layer(x)
                s_ctx = model.out_norm(x)
                s_pred = model.predictor(s_ctx)
                jepa_l = model._jepa_loss_nextstep(s_pred, s_target).item()

                # Decode CE
                dec_l = F.cross_entropy(
                    logits_shift.reshape(-1, model.vocab_size),
                    tokens_shift.reshape(-1),
                ).item()

                # Representation std (collapse monitor)
                std = torch.sqrt(
                    s_ctx.reshape(-1, s_ctx.shape[-1]).var(dim=0) + 1e-4
                ).mean().item()

                # VICReg components (monitoring)
                L_var, L_cov = vicreg_loss(s_ctx)
                L_var_val = L_var.item()
                L_cov_val = L_cov.item()

            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000

            diagnostics.update_train(diag, jepa_l, dec_l, acc, ms, std,
                                     L_var_val, L_cov_val)

            ep_ei += diag['E_initial']
            ep_ef += diag['E_final']
            ep_jepa += jepa_l
            ep_dec += dec_l
            ep_acc += acc
            ep_ms += ms
            ep_iters += diag['actual_iters']
            n_batches += 1

        # --- End-of-epoch ---
        scheduler.step()
        model.set_ema_progress(epoch / args.epochs)

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        diagnostics.update_test(test_metrics)
        best_test_acc = max(best_test_acc, test_metrics['accuracy'])

        avg_ei = ep_ei / n_batches
        avg_ef = ep_ef / n_batches
        avg_jepa = ep_jepa / n_batches
        avg_dec = ep_dec / n_batches
        avg_acc = ep_acc / n_batches
        avg_ms = ep_ms / n_batches
        avg_it = ep_iters / n_batches

        print(f"{epoch:5d} {avg_ei:8.4f} {avg_ef:8.4f} {avg_jepa:8.4f} "
              f"{avg_dec:8.4f} {avg_acc:8.4f} {test_metrics['accuracy']:8.4f} "
              f"{avg_ms:7.1f} {avg_it:5.1f}")

        # Plot
        if epoch % args.plot_every == 0 or epoch == args.epochs:
            config_str = (f'({mode_str}, T={args.iters}, '
                          f'e_lr={args.e_lr}, prec={args.precision_mode})')
            chart_path = os.path.join(
                save_dir,
                f'epc_jepa_s{args.stage}_{mode_str}_epoch_{epoch:03d}.png')
            diagnostics.plot(chart_path, epoch, config_str)

    # --- Summary ---
    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    if best_test_acc >= 0.80:
        print(f"PASS: ePC-JEPA Stage {args.stage} works! "
              f"Local learning produces useful JEPA representations.")
    elif best_test_acc >= 0.40:
        print(f"PROMISING: {best_test_acc:.1%} — learning but not converged")
    else:
        print(f"Needs work: {best_test_acc:.1%}")


if __name__ == '__main__':
    main()
