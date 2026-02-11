"""
ePC-Mamba3 training for synthetic tasks.

Tests ePC with deeper Mamba3 architecture (4+ layers).
Key difference from ePC_Mamba: Mamba3 blocks have internal residuals
(Mixer + MLP), trapezoidal discretization, and data-dependent RoPE.

Usage:
  python experiments/Mamba3/train_epc.py
  python experiments/Mamba3/train_epc.py --n_layer 8 --epochs 50
  python experiments/Mamba3/train_epc.py --baseline  # backprop comparison
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3LM
from experiments.Mamba3.epc_model import ePCMamba3LM


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_copy_data(n_samples, seq_len, vocab_size, copy_len=None):
    """Generate copy task data. PAD=0, data tokens in [1, vocab_size-1]."""
    if copy_len is None:
        copy_len = seq_len // 2
    assert copy_len <= seq_len // 2

    inputs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)

    data = torch.randint(1, vocab_size, (n_samples, copy_len))
    inputs[:, :copy_len] = data
    targets[:, seq_len - copy_len:] = data

    return inputs, targets


def compute_accuracy(logits, targets, ignore_pad=True):
    """Compute token-level accuracy."""
    preds = logits.argmax(dim=-1)
    if ignore_pad:
        mask = targets != 0
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == targets[mask]).float().mean().item()
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, device, use_epc=True):
    """Evaluate returning both accuracy and loss."""
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            if use_epc:
                logits = model(inputs)  # targets=None → feedforward
            else:
                logits = model(inputs)
            acc = compute_accuracy(logits, targets)
            b, l, v = logits.shape
            loss = F.cross_entropy(
                logits.reshape(b * l, v), targets.reshape(b * l)
            ).item()
            total_acc += acc
            total_loss += loss
            n_batches += 1

    return total_acc / n_batches, total_loss / n_batches


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class Diagnostics:
    """Collect and plot training diagnostics for ePC-Mamba3."""

    def __init__(self, num_error_layers, use_epc=True):
        self.num_error_layers = num_error_layers
        self.use_epc = use_epc
        self.reset()

    def reset(self):
        self.train_accs = []
        self.train_losses = []
        self.test_accs = []
        self.test_losses = []
        self.ms_per_batch = []
        # ePC-specific
        self.layer_energies = [[] for _ in range(self.num_error_layers)]
        self.error_norms = [[] for _ in range(self.num_error_layers)]
        self.inference_convergence = []
        self.newton_rank1_ratio = []
        self.newton_coeff = []

    def update_train_epc(self, acc, loss, diagnostics, ms):
        self.train_accs.append(acc)
        self.train_losses.append(loss)
        self.ms_per_batch.append(ms)
        self.inference_convergence.append(diagnostics['convergence'])
        self.newton_rank1_ratio.append(diagnostics.get('newton_rank1_ratio', 0))
        self.newton_coeff.append(diagnostics.get('newton_coeff', 0))
        for i, energy in enumerate(diagnostics.get('layer_energies', [])):
            if i < self.num_error_layers:
                self.layer_energies[i].append(energy)
        for i, norm in enumerate(diagnostics.get('error_norms', [])):
            if i < self.num_error_layers:
                self.error_norms[i].append(norm)

    def update_train_baseline(self, acc, loss, ms):
        self.train_accs.append(acc)
        self.train_losses.append(loss)
        self.ms_per_batch.append(ms)

    def update_test(self, acc, loss):
        self.test_accs.append(acc)
        self.test_losses.append(loss)

    def plot(self, save_path, epoch, config_str=''):
        nrows = 3 if self.use_epc else 2
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
        mode = 'ePC-Mamba3' if self.use_epc else 'Backprop Mamba3'
        fig.suptitle(f'{mode} Training — Epoch {epoch} {config_str}', fontsize=13)

        epochs = list(range(1, len(self.test_accs) + 1))

        # --- Row 0 ---

        # [0,0] Loss (per epoch)
        ax = axes[0, 0]
        if self.test_losses:
            ax.plot(epochs, [self.test_losses[i] for i in range(len(epochs))],
                    'r--', label='Test', linewidth=2)
        # Train loss: average per epoch from batch data
        if self.train_losses:
            n_batches_per_epoch = len(self.train_losses) // max(len(epochs), 1)
            if n_batches_per_epoch > 0:
                epoch_train_loss = []
                for ep in range(len(epochs)):
                    start = ep * n_batches_per_epoch
                    end = start + n_batches_per_epoch
                    epoch_train_loss.append(np.mean(self.train_losses[start:end]))
                ax.plot(epochs[:len(epoch_train_loss)], epoch_train_loss,
                        'b-', label='Train', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (CE)')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [0,1] Accuracy (per epoch)
        ax = axes[0, 1]
        if self.test_accs:
            ax.plot(epochs, self.test_accs, 'r--', label='Test', linewidth=2)
        if self.train_accs:
            n_batches_per_epoch = len(self.train_accs) // max(len(epochs), 1)
            if n_batches_per_epoch > 0:
                epoch_train_acc = []
                for ep in range(len(epochs)):
                    start = ep * n_batches_per_epoch
                    end = start + n_batches_per_epoch
                    epoch_train_acc.append(np.mean(self.train_accs[start:end]))
                ax.plot(epochs[:len(epoch_train_acc)], epoch_train_acc,
                        'b-', label='Train', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Token Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [0,2] Speed (per epoch)
        ax = axes[0, 2]
        if self.ms_per_batch:
            n_batches_per_epoch = len(self.ms_per_batch) // max(len(epochs), 1)
            if n_batches_per_epoch > 0:
                epoch_ms = []
                for ep in range(len(epochs)):
                    start = ep * n_batches_per_epoch
                    end = start + n_batches_per_epoch
                    epoch_ms.append(np.mean(self.ms_per_batch[start:end]))
                ax.plot(epochs[:len(epoch_ms)], epoch_ms, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ms/batch')
        ax.set_title('Speed')
        ax.grid(True, alpha=0.3)

        # --- Row 1: Generalization + summary ---

        # [1,0] Generalization gap
        ax = axes[1, 0]
        if self.test_accs and self.train_accs:
            n_batches_per_epoch = len(self.train_accs) // max(len(epochs), 1)
            if n_batches_per_epoch > 0:
                epoch_train_acc = []
                for ep in range(len(epochs)):
                    start = ep * n_batches_per_epoch
                    end = start + n_batches_per_epoch
                    epoch_train_acc.append(np.mean(self.train_accs[start:end]))
                gap = [tr - te for tr, te in
                       zip(epoch_train_acc[:len(self.test_accs)], self.test_accs)]
                ax.plot(epochs[:len(gap)], gap, 'm-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train - Test Acc')
        ax.set_title('Generalization Gap')
        ax.grid(True, alpha=0.3)

        # [1,1] Summary text
        ax = axes[1, 1]
        ax.axis('off')
        lines = [f'Mode: {mode}']
        if self.test_accs:
            lines.append(f'Best Test Acc: {max(self.test_accs):.2%}')
            lines.append(f'Final Test Acc: {self.test_accs[-1]:.2%}')
        if self.ms_per_batch:
            lines.append(f'Avg ms/batch: {np.mean(self.ms_per_batch):.1f}')
        if self.use_epc and self.inference_convergence:
            recent = self.inference_convergence[-200:]
            lines.append(f'Avg convergence: {np.mean(recent):.2f}')
        ax.text(0.1, 0.5, '\n'.join(lines), fontsize=12,
                verticalalignment='center', family='monospace')

        # [1,2] Per-layer error stats
        ax = axes[1, 2]
        ax.axis('off')
        if self.use_epc:
            lines = ['Error Stats (last 200 batches):']
            for i, norms in enumerate(self.error_norms):
                if norms:
                    recent = norms[-200:]
                    lines.append(f'  Layer {i+1}: ||e|| mean={np.mean(recent):.4f}, '
                                 f'max={np.max(recent):.4f}')
            if self.inference_convergence:
                recent = self.inference_convergence[-200:]
                lines.append(f'\nConvergence: mean={np.mean(recent):.2f}')
        else:
            lines = ['Backprop baseline — no ePC diagnostics']
        ax.text(0.05, 0.5, '\n'.join(lines), fontsize=10,
                verticalalignment='center', family='monospace')

        # --- Row 2: ePC-specific ---
        if self.use_epc:
            # [2,0] Per-layer energies
            ax = axes[2, 0]
            for i, energies in enumerate(self.layer_energies):
                if energies:
                    ax.plot(energies, label=f'Layer {i+1}', alpha=0.7, linewidth=0.8)
            if any(e for e in self.layer_energies):
                ax.set_yscale('log')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Energy')
            ax.set_title('Per-Layer Energies (0.5 ||e_i||²)')
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            # [2,1] Error magnitudes
            ax = axes[2, 1]
            for i, norms in enumerate(self.error_norms):
                if norms:
                    ax.plot(norms, label=f'Layer {i+1}', alpha=0.7, linewidth=0.8)
            if any(n for n in self.error_norms):
                ax.set_yscale('log')
            ax.set_xlabel('Batch')
            ax.set_ylabel('||e_i||')
            ax.set_title('Error Magnitudes')
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            # [2,2] Inference convergence
            ax = axes[2, 2]
            if self.inference_convergence:
                ax.plot(self.inference_convergence, alpha=0.5, linewidth=0.5)
                if len(self.inference_convergence) > 50:
                    window = min(50, len(self.inference_convergence) // 5)
                    ma = np.convolve(self.inference_convergence,
                                     np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(self.inference_convergence)), ma,
                            linewidth=1.5, color='red', label=f'MA-{window}')
                    ax.legend(fontsize=8)
            ax.set_xlabel('Batch')
            ax.set_ylabel('E_initial - E_final')
            ax.set_title('Inference Convergence')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f'  [Saved {save_path}]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ePC-Mamba3 synthetic training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--iters', type=int, default=2,
                        help='Error optimization iterations (T)')
    parser.add_argument('--error_optim', type=str, default='newton',
                        choices=['sgd', 'adam', 'newton'],
                        help='Error optimizer: sgd, adam, or newton')
    parser.add_argument('--e_lr', type=float, default=0.02,
                        help='Error learning rate (for sgd/adam)')
    parser.add_argument('--damping', type=float, default=0.1,
                        help='Newton damping factor')
    parser.add_argument('--precision_mode', type=str, default='geometric',
                        choices=['none', 'linear', 'geometric'],
                        help='Per-layer precision weighting mode')
    parser.add_argument('--precision_base', type=float, default=3.0,
                        help='Base for geometric precision (ignored if not geometric)')
    parser.add_argument('--w_clip', type=float, default=1.0,
                        help='Weight gradient clipping max norm (0 to disable)')
    parser.add_argument('--ipc', action='store_true', default=True,
                        help='Incremental PC: weight update every Newton step')
    parser.add_argument('--no_ipc', action='store_true',
                        help='Disable iPC (standard ePC: T error steps then 1 weight step)')
    parser.add_argument('--init_scale', type=float, default=1.0,
                        help='Scale block output projections for larger initial Jacobian')
    parser.add_argument('--mhc', action='store_true',
                        help='Use manifold-constrained hyperconnections (mHC)')
    parser.add_argument('--n_streams', type=int, default=2,
                        help='Number of parallel residual streams for mHC')
    parser.add_argument('--mupc', action='store_true',
                        help='Use Depth-muP scaling (muPC, Innocenti 2025)')
    parser.add_argument('--conv_threshold', type=float, default=0.0,
                        help='Convergence threshold for early stopping error optimization '
                             '(0=disabled). Stops when relative energy change < threshold.')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--baseline', action='store_true',
                        help='Run backprop baseline instead of ePC')
    parser.add_argument('--use_conv', action='store_true',
                        help='Enable optional short causal convolution')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--plot_every', type=int, default=10,
                        help='Save diagnostic plots every N epochs')
    args = parser.parse_args()

    if args.no_ipc:
        args.ipc = False

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Config
    chunk_size = min(64, args.seq_len)
    while args.seq_len % chunk_size != 0 and chunk_size > 1:
        chunk_size -= 1

    config = Mamba3Config(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layer=args.n_layer,
        chunk_size=chunk_size,
        use_conv=args.use_conv,
    )
    print(f"Config: d_model={config.d_model}, d_inner={config.d_inner}, "
          f"nheads={config.nheads}, n_layer={config.n_layer}, "
          f"d_state={config.d_state}")

    # Data
    print(f"\nGenerating copy task data...")
    train_x, train_y = generate_copy_data(
        args.n_train, args.seq_len, args.vocab_size)
    test_x, test_y = generate_copy_data(
        args.n_test, args.seq_len, args.vocab_size)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    # Model
    use_epc = not args.baseline

    if use_epc:
        model = ePCMamba3LM(
            config, vocab_size=args.vocab_size,
            iters=args.iters, e_lr=args.e_lr,
            error_optim=args.error_optim, damping=args.damping,
            precision_mode=args.precision_mode,
            precision_base=args.precision_base,
            use_mhc=args.mhc, n_streams=args.n_streams,
            use_mupc=args.mupc,
            convergence_threshold=args.conv_threshold,
        ).to(device)
        optim_str = args.error_optim.upper()
        if args.error_optim == 'newton':
            print(f"Model: ePC-Mamba3 (T={args.iters}, Newton, damping={args.damping})")
        else:
            print(f"Model: ePC-Mamba3 (T={args.iters}, {optim_str}, e_lr={args.e_lr}, "
                  f"energy_scale={model.pce.energy_scale:.4f})")
        if args.ipc:
            print(f"  Mode: iPC (weight update every Newton step, {args.iters}x faster)")
        if args.mhc:
            print(f"  mHC: {args.n_streams} streams, Sinkhorn-constrained mixing")
        if args.mupc:
            alpha = model.pce.mupc_alpha
            print(f"  muPC: Depth-muP alpha={alpha:.4f} "
                  f"(1/sqrt({args.d_model}*{2*args.n_layer}))")
        if args.init_scale != 1.0:
            with torch.no_grad():
                for block in model.pce.layers:
                    block.mixer.out_proj.weight.mul_(args.init_scale)
                    block.mlp.down_proj.weight.mul_(args.init_scale)
            print(f"  Init scale: {args.init_scale}x on mixer/MLP output projections")
        print(f"  Adaptive damping: enabled (doubles on energy overshoot)")
        if args.conv_threshold > 0:
            print(f"  Convergence early stopping: threshold={args.conv_threshold}")
        if args.precision_mode != 'none':
            pi_str = ', '.join(f'{p:.2f}' for p in model.pce.precisions)
            print(f"Precisions ({args.precision_mode}): [{pi_str}]")
    else:
        model = Mamba3LM(config, vocab_size=args.vocab_size).to(device)
        print("Model: Backprop Mamba3 (baseline)")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Diagnostics
    num_error_layers = config.n_layer if use_epc else 0
    diagnostics = Diagnostics(num_error_layers, use_epc=use_epc)

    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    # Training
    best_test_acc = 0.0
    print(f"\n{'Epoch':>5} {'Loss':>10} {'Train Acc':>10} {'Test Acc':>10} {'ms/batch':>10}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_time = 0.0
        n_batches = 0

        for batch in train_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            batch_size = inputs.shape[0]

            if use_epc:
                t0 = time.perf_counter()

                if args.ipc:
                    # iPC: interleaved error + weight steps
                    E_val = model.ipc_train_step(
                        inputs, targets, optimizer, batch_size, args.w_clip)
                    loss_val = E_val
                else:
                    # Standard ePC: Phase 1 inference, Phase 2 weight update
                    model(inputs, targets)

                    optimizer.zero_grad()
                    weight_loss = model.compute_weight_loss(
                        inputs, targets, batch_size)
                    weight_loss.backward()

                    if args.w_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=args.w_clip)

                    optimizer.step()
                    loss_val = weight_loss.item()

                # Collect diagnostics BEFORE accuracy eval (#23)
                diag = model.get_diagnostics()

                # Get accuracy (feedforward — resets errors)
                with torch.no_grad():
                    logits = model(inputs)
                    acc = compute_accuracy(logits, targets)

                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000

                diagnostics.update_train_epc(
                    acc=acc, loss=loss_val, diagnostics=diag, ms=ms)

            else:
                t0 = time.perf_counter()

                optimizer.zero_grad()
                logits = model(inputs)
                b, l, v = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(b * l, v), targets.reshape(b * l))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)

                optimizer.step()

                loss_val = loss.item()
                with torch.no_grad():
                    acc = compute_accuracy(logits, targets)

                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000

                diagnostics.update_train_baseline(
                    acc=acc, loss=loss_val, ms=ms)

            epoch_loss += loss_val
            epoch_acc += acc
            epoch_time += ms
            n_batches += 1

        # End-of-epoch evaluation
        test_acc, test_loss = evaluate(model, test_loader, device, use_epc=use_epc)
        diagnostics.update_test(test_acc, test_loss)
        best_test_acc = max(best_test_acc, test_acc)

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        avg_time = epoch_time / n_batches

        print(f"{epoch:5d} {avg_loss:10.4f} {avg_acc:10.4f} "
              f"{test_acc:10.4f} {avg_time:10.1f}")

        # Periodic ePC diagnostics
        if use_epc and epoch % 10 == 0:
            d = model.get_diagnostics()
            eff_damp = d.get('effective_damping', args.damping)
            damp_str = f", eff_damp={eff_damp:.2f}" if eff_damp != args.damping else ""
            iters_str = f", iters={d['iters_used']}" if d['iters_used'] != args.iters else ""
            print(f"  ePC: E_init={d['E_initial']:.2f}, "
                  f"E_final={d['E_final']:.2f}, "
                  f"convergence={d['convergence']:.2f}"
                  f"{damp_str}{iters_str}")
            if d['error_norms']:
                norms_str = ', '.join(f'{n:.4f}' for n in d['error_norms'])
                print(f"  Error norms: [{norms_str}]")

        # Save plot
        if epoch % args.plot_every == 0 or epoch == args.epochs:
            mode = 'epc' if use_epc else 'baseline'
            if use_epc:
                if args.error_optim == 'newton':
                    config_str = f'(T={args.iters}, Newton, damp={args.damping})'
                else:
                    config_str = f'(T={args.iters}, {args.error_optim.upper()}, e_lr={args.e_lr})'
                if args.precision_mode != 'none':
                    config_str += f', prec={args.precision_mode}'
            else:
                config_str = ''
            chart_path = os.path.join(
                save_dir, f'mamba3_{mode}_epoch_{epoch:03d}.png')
            diagnostics.plot(chart_path, epoch, config_str)

        # Early success
        if best_test_acc >= 0.99 and epoch >= 5:
            print(f"\nSuccess! Test accuracy {best_test_acc:.4f} >= 99% at epoch {epoch}")
            mode = 'epc' if use_epc else 'baseline'
            chart_path = os.path.join(
                save_dir, f'mamba3_{mode}_epoch_{epoch:03d}.png')
            diagnostics.plot(chart_path, epoch)
            break

    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    if best_test_acc >= 0.95:
        print("PASS: Works on copy task!")
    elif best_test_acc >= 0.90:
        print("PROMISING: 90%+ accuracy.")
    else:
        print(f"Needs work: {best_test_acc:.1%} accuracy.")


if __name__ == '__main__':
    main()
