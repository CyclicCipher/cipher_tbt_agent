"""
Synthetic task training for ePC-Mamba (Phase 1).

Tasks:
  1. Copy: input [a,b,c,PAD,PAD,PAD] → output [PAD,PAD,PAD,a,b,c]
  2. Selective copy: copy only tokens marked with a flag
  3. Sequence classification: predict label from sequence content

Usage:
  python experiments/ePC_Mamba/train_synthetic.py [--task copy|selective|classify]
  python experiments/ePC_Mamba/train_synthetic.py --profile  # with timing breakdown
  python experiments/ePC_Mamba/train_synthetic.py --baseline  # backprop comparison
"""

import argparse
import math
import os
import sys
import time

# Add project root to path (MISTAKES.md #7)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.ePC_Mamba.mamba_block import Mamba2Config, Mamba2Block, RMSNorm
from experiments.ePC_Mamba.epc_model import PCESequence


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_copy_data(n_samples: int, seq_len: int, vocab_size: int,
                       copy_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate copy task data.

    Input:  [a, b, c, ..., PAD, PAD, PAD, ...]  (tokens in first half)
    Target: [PAD, PAD, PAD, ..., a, b, c, ...]  (tokens in second half)

    PAD token = 0, data tokens in [1, vocab_size-1].
    """
    if copy_len is None:
        copy_len = seq_len // 2
    assert copy_len <= seq_len // 2, "copy_len must be <= seq_len // 2"

    tokens = torch.randint(1, vocab_size, (n_samples, copy_len))

    inputs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)

    inputs[:, :copy_len] = tokens
    targets[:, seq_len - copy_len:] = tokens

    return inputs, targets


def generate_selective_copy_data(n_samples: int, seq_len: int,
                                 vocab_size: int, n_markers: int = 4
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate selective copy task data.

    Marker tokens are placed at random positions. The model must copy
    the tokens that FOLLOW each marker to the end of the sequence.
    """
    marker_token = vocab_size - 1

    inputs = torch.randint(1, vocab_size - 1, (n_samples, seq_len))
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)

    for i in range(n_samples):
        inputs[i] = torch.randint(1, vocab_size - 1, (seq_len,))
        valid_positions = torch.arange(0, seq_len - 1)
        marker_positions = valid_positions[torch.randperm(len(valid_positions))[:n_markers]].sort().values
        inputs[i, marker_positions] = marker_token
        for j, pos in enumerate(marker_positions):
            if pos + 1 < seq_len:
                targets[i, seq_len - n_markers + j] = inputs[i, pos + 1]

    return inputs, targets


def generate_classify_data(n_samples: int, seq_len: int, vocab_size: int,
                           n_classes: int = 4
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate sequence classification data. Label = (sum of tokens) mod n_classes."""
    inputs = torch.randint(1, vocab_size, (n_samples, seq_len))
    targets = (inputs.sum(dim=1) % n_classes).long()
    return inputs, targets


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class ePCMambaSynthetic(nn.Module):
    """ePC-Mamba model for synthetic sequence tasks."""

    def __init__(self, config: Mamba2Config, vocab_size: int,
                 task: str = 'copy', n_classes: int = 4,
                 iters: int = 2, e_lr: float = 0.01,
                 error_optim: str = 'newton', damping: float = 1.0,
                 output_loss: str = 'mse'):
        super().__init__()
        self.config = config
        self.task = task

        self.embedding = nn.Embedding(vocab_size, config.d_model)

        if task in ('copy', 'selective'):
            self.pce = PCESequence(
                config, iters=iters, e_lr=e_lr, error_optim=error_optim,
                damping=damping, output_loss=output_loss,
            )
            self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)
        elif task == 'classify':
            self.pce = PCESequence(
                config, iters=iters, e_lr=e_lr, error_optim=error_optim,
                damping=damping, output_loss=output_loss,
            )
            self.out_proj = nn.Linear(config.d_model, n_classes)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        if targets is not None:
            if self.task == 'classify':
                return self._forward_classify_epc(x, targets)
            else:
                return self.pce.minimize_error_energy(x, targets, self.out_proj)
        else:
            self.pce.errors = [0.0] * (len(self.pce.layers) - 1)
            hidden = self.pce.y_pred(x)
            if self.task == 'classify':
                hidden = hidden.mean(dim=1)
            return self.out_proj(hidden)

    def _forward_classify_epc(self, x, targets):
        class _PooledProj(nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.proj = proj
            def forward(self, x):
                return self.proj(x.mean(dim=1))
        return self.pce.minimize_error_energy(x, targets, _PooledProj(self.out_proj))

    def compute_weight_loss(self, input_ids, targets, batch_size):
        x = self.embedding(input_ids)
        if self.task == 'classify':
            class _PooledProj(nn.Module):
                def __init__(self, proj):
                    super().__init__()
                    self.proj = proj
                def forward(self, x):
                    return self.proj(x.mean(dim=1))
            return self.pce.E_local(x, targets, _PooledProj(self.out_proj)) / batch_size
        return self.pce.E_local(x, targets, self.out_proj) / batch_size

    def get_diagnostics(self):
        return self.pce.get_diagnostics()


class BackpropMambaBaseline(nn.Module):
    """Standard backprop Mamba model (same architecture, no ePC)."""

    def __init__(self, config: Mamba2Config, vocab_size: int,
                 task: str = 'copy', n_classes: int = 4):
        super().__init__()
        self.config = config
        self.task = task

        self.embedding = nn.Embedding(vocab_size, config.d_model)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(config.n_layer):
            self.norms.append(RMSNorm(config.d_model))
            self.layers.append(Mamba2Block(config))

        self.out_norm = RMSNorm(config.d_model)

        if task in ('copy', 'selective'):
            self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)
        elif task == 'classify':
            self.out_proj = nn.Linear(config.d_model, n_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        x = self.out_norm(x)
        if self.task == 'classify':
            x = x.mean(dim=1)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(logits, targets, pad_token=0, task='copy'):
    """Compute accuracy, ignoring PAD positions for copy tasks."""
    if task == 'classify':
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()
    else:
        preds = logits.argmax(dim=-1)
        mask = targets != pad_token
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == targets[mask]).float().mean().item()


# ---------------------------------------------------------------------------
# Diagnostics (adapted from ePC_ResNet/train_cifar10.py)
# ---------------------------------------------------------------------------

def _sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def get_weight_magnitudes(model, use_epc=True):
    """Extract max absolute weight per Mamba block."""
    mags = {}
    if use_epc:
        for i, layer in enumerate(model.pce.layers):
            max_val = 0.0
            for p in layer.parameters():
                max_val = max(max_val, p.data.abs().max().item())
            mags[f'L{i+1}'] = max_val
    else:
        for i, layer in enumerate(model.layers):
            max_val = 0.0
            for p in layer.parameters():
                max_val = max(max_val, p.data.abs().max().item())
            mags[f'L{i+1}'] = max_val
    return mags


class Diagnostics:
    """Collect and plot training diagnostics for ePC-Mamba."""

    def __init__(self, num_error_layers, task='copy'):
        self.num_error_layers = num_error_layers
        self.task = task
        self.reset()

    def reset(self):
        self.train_accs = []
        self.train_losses = []
        self.test_accs = []
        self.test_losses = []
        self.layer_energies = [[] for _ in range(self.num_error_layers)]
        self.inference_convergence = []
        self.iters_used = []
        self.error_norms = [[] for _ in range(self.num_error_layers)]
        self.learning_rates = []
        self.weight_magnitudes = {}
        # Hypothesis diagnostics
        self.newton_rank1_ratio = []    # H1: How much curvature rank-1 captures
        self.newton_coeff = []          # H1: Woodbury coefficient magnitude
        self.causal_early_late = []     # H2: Early/late position error ratio
        self.rmsnorm_ratios = [[] for _ in range(self.num_error_layers + 1)]  # H3
        self.gate_frac_near_zero = [[] for _ in range(self.num_error_layers + 1)]  # H5
        self.per_position_snapshots = []  # H2: occasional full snapshots
        self.timing = {
            'data_transfer': [],
            'inference': [],
            'inference_init': [],
            'inference_forward': [],
            'inference_backward': [],
            'inference_step': [],
            'weight_forward': [],
            'weight_backward': [],
            'optimizer_step': [],
        }

    def update_train(self, acc, loss, diagnostics, lr, weight_mags):
        self.train_accs.append(acc)
        self.train_losses.append(loss)
        self.inference_convergence.append(diagnostics['convergence'])
        self.iters_used.append(diagnostics.get('iters_used', 0))
        self.learning_rates.append(lr)

        for i, energy in enumerate(diagnostics['layer_energies']):
            if i < self.num_error_layers:
                self.layer_energies[i].append(energy)

        for i, norm in enumerate(diagnostics['error_norms']):
            if i < self.num_error_layers:
                self.error_norms[i].append(norm)

        for name, mag in weight_mags.items():
            if name not in self.weight_magnitudes:
                self.weight_magnitudes[name] = []
            self.weight_magnitudes[name].append(mag)

    def update_train_baseline(self, acc, loss, lr, weight_mags):
        """For backprop baseline (no ePC diagnostics)."""
        self.train_accs.append(acc)
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        for name, mag in weight_mags.items():
            if name not in self.weight_magnitudes:
                self.weight_magnitudes[name] = []
            self.weight_magnitudes[name].append(mag)

    def update_hypothesis(self, hyp_diag, save_snapshot=False):
        """Update hypothesis-specific diagnostics."""
        self.newton_rank1_ratio.append(hyp_diag.get('newton_rank1_ratio', 0))
        self.newton_coeff.append(hyp_diag.get('newton_coeff', 0))
        self.causal_early_late.append(hyp_diag.get('causal_early_late_ratio', 0))

        for i, ratio in enumerate(hyp_diag.get('rmsnorm_ratios', [])):
            if i < len(self.rmsnorm_ratios):
                self.rmsnorm_ratios[i].append(ratio)

        for i, gs in enumerate(hyp_diag.get('gate_stats', [])):
            if i < len(self.gate_frac_near_zero):
                self.gate_frac_near_zero[i].append(gs['frac_near_zero'])

        if save_snapshot and hyp_diag.get('per_position_error_norms'):
            self.per_position_snapshots.append(
                hyp_diag['per_position_error_norms']
            )

    def update_test(self, acc, loss):
        self.test_accs.append(acc)
        self.test_losses.append(loss)

    def update_timing(self, timing_dict):
        for key, val in timing_dict.items():
            if key in self.timing:
                self.timing[key].append(val)

    def has_timing(self):
        return len(self.timing.get('inference', [])) > 0

    def print_performance_report(self, epoch, batch_size):
        """Print a performance breakdown table."""
        if not self.has_timing():
            return

        n = len(self.timing['inference'])
        def avg(key):
            vals = self.timing.get(key, [])
            return np.mean(vals) if vals else 0.0

        t_data = avg('data_transfer')
        t_inf = avg('inference')
        t_init = avg('inference_init')
        t_fwd = avg('inference_forward')
        t_bwd = avg('inference_backward')
        t_step = avg('inference_step')
        t_wfwd = avg('weight_forward')
        t_wbwd = avg('weight_backward')
        t_opt = avg('optimizer_step')
        t_total = t_data + t_inf + t_wfwd + t_wbwd + t_opt

        def pct(v):
            return v / t_total * 100 if t_total > 0 else 0

        throughput = batch_size / (t_total / 1000) if t_total > 0 else 0

        print()
        print("=" * 66)
        print(f"  PERFORMANCE REPORT (Epoch {epoch}, {n} batches profiled)")
        print("=" * 66)
        print(f"  {'Phase':<24} {'Avg (ms)':>9} {'%':>6}")
        print("-" * 66)
        print(f"  {'Data transfer':<24} {t_data:>9.1f} {pct(t_data):>5.1f}%")
        print(f"  {'Inference (total)':<24} {t_inf:>9.1f} {pct(t_inf):>5.1f}%")
        print(f"    {'Error init':<22} {t_init:>9.1f} {pct(t_init):>5.1f}%")
        print(f"    {'Forward (E)':<22} {t_fwd:>9.1f} {pct(t_fwd):>5.1f}%")
        print(f"    {'Backward':<22} {t_bwd:>9.1f} {pct(t_bwd):>5.1f}%")
        print(f"    {'Newton step':<22} {t_step:>9.1f} {pct(t_step):>5.1f}%")
        print(f"  {'Weight forward':<24} {t_wfwd:>9.1f} {pct(t_wfwd):>5.1f}%")
        print(f"  {'Weight backward':<24} {t_wbwd:>9.1f} {pct(t_wbwd):>5.1f}%")
        print(f"  {'Optimizer step':<24} {t_opt:>9.1f} {pct(t_opt):>5.1f}%")
        print("-" * 66)
        print(f"  {'TOTAL per batch':<24} {t_total:>9.1f}")
        print(f"  {'Throughput':<24} {throughput:>9.0f} samples/sec")
        avg_T = np.mean(self.iters_used) if self.iters_used else 0
        print(f"  {'Avg T (iters used)':<24} {avg_T:>9.2f}")
        print("=" * 66)

    def plot(self, save_path, epoch=None, num_epochs=None, use_epc=True,
             batch_size=32, task='copy', iters=2, error_optim='newton',
             e_lr=0.01, damping=1.0):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        has_hyp = use_epc and len(self.newton_rank1_ratio) > 0
        nrows = 3
        if self.has_timing():
            nrows = 4
        if has_hyp:
            nrows += 1
        fig, axes = plt.subplots(nrows, 3, figsize=(18, 5 * nrows))

        # --- Row 0 ---

        # [0,0] Accuracy
        ax = axes[0, 0]
        if self.train_accs:
            ax.plot(self.train_accs, alpha=0.4, linewidth=0.5, label='Train (batch)')
            if len(self.train_accs) > 50:
                window = min(50, len(self.train_accs) // 5)
                ma = np.convolve(self.train_accs, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.train_accs)), ma,
                        linewidth=1.5, label=f'Train (MA-{window})')
        if self.test_accs:
            n_train = len(self.train_accs)
            n_test = len(self.test_accs)
            if n_test > 0:
                test_x = [(i + 1) * n_train / n_test for i in range(n_test)]
                ax.plot(test_x, self.test_accs, 'o-', linewidth=2,
                        markersize=4, label='Test')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [0,1] Output Loss
        ax = axes[0, 1]
        if self.train_losses:
            ax.plot(self.train_losses, alpha=0.4, linewidth=0.5, label='Train (batch)')
            if len(self.train_losses) > 50:
                window = min(50, len(self.train_losses) // 5)
                ma = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.train_losses)), ma,
                        linewidth=1.5, label=f'Train (MA-{window})')
        if self.test_losses:
            n_train = len(self.train_losses)
            n_test = len(self.test_losses)
            if n_test > 0:
                test_x = [(i + 1) * n_train / n_test for i in range(n_test)]
                ax.plot(test_x, self.test_losses, 'o-', linewidth=2,
                        markersize=4, label='Test')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Output Loss (CE)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [0,2] Per-Layer Energies
        ax = axes[0, 2]
        if use_epc:
            for i, energies in enumerate(self.layer_energies):
                if energies:
                    ax.plot(energies, label=f'Layer {i+1}', alpha=0.7, linewidth=0.8)
            ax.set_yscale('log')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Energy')
        ax.set_title('Per-Layer Energies (0.5 ||e_i||^2)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # --- Row 1 ---

        # [1,0] Inference Convergence
        ax = axes[1, 0]
        if use_epc and self.inference_convergence:
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
        ax.set_title('Inference Convergence (E_0 - E_T)')
        ax.grid(True, alpha=0.3)

        # [1,1] Error Magnitudes
        ax = axes[1, 1]
        if use_epc:
            for i, norms in enumerate(self.error_norms):
                if norms:
                    ax.plot(norms, label=f'Layer {i+1}', alpha=0.7, linewidth=0.8)
            if any(norms for norms in self.error_norms):
                ax.set_yscale('log')
        ax.set_xlabel('Batch')
        ax.set_ylabel('||e_i||')
        ax.set_title('Error Magnitudes (per layer)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # [1,2] Learning Rate
        ax = axes[1, 2]
        if self.learning_rates:
            ax.plot(self.learning_rates, linewidth=1.5)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)

        # --- Row 2 ---

        # [2,0] Weight Magnitudes
        ax = axes[2, 0]
        for name, mags in self.weight_magnitudes.items():
            if mags:
                ax.plot(mags, label=name, alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Batch')
        ax.set_ylabel('max|W|')
        ax.set_title('Weight Magnitudes (per layer)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # [2,1] Summary
        ax = axes[2, 1]
        ax.axis('off')
        lines = []
        if self.test_accs:
            lines.append(f"Best Test Acc: {max(self.test_accs):.2%}")
            lines.append(f"Final Test Acc: {self.test_accs[-1]:.2%}")
        if self.train_accs:
            lines.append(f"Final Train Acc (batch): {self.train_accs[-1]:.2%}")
        if epoch is not None and num_epochs is not None:
            lines.append(f"\nEpoch: {epoch}/{num_epochs}")
        lines.append(f"Task: {task}")
        if use_epc:
            if error_optim == 'newton':
                lines.append(f"Mode: ePC (T={iters}, {error_optim}, damp={damping})")
            else:
                lines.append(f"Mode: ePC (T={iters}, {error_optim}, e_lr={e_lr})")
        else:
            lines.append("Mode: Backprop")
        if use_epc and self.inference_convergence:
            avg_conv = np.mean(self.inference_convergence[-100:])
            lines.append(f"Avg Convergence (last 100): {avg_conv:.2f}")
        ax.text(0.1, 0.5, '\n'.join(lines), fontsize=11,
                verticalalignment='center', family='monospace')

        # [2,2] Per-layer error stats
        ax = axes[2, 2]
        ax.axis('off')
        lines = []
        if use_epc:
            for i, norms in enumerate(self.error_norms):
                if norms:
                    recent = norms[-100:] if len(norms) >= 100 else norms
                    lines.append(f"Layer {i+1}: ||e|| mean={np.mean(recent):.3f}, "
                                 f"max={np.max(recent):.3f}")
            if self.inference_convergence:
                recent = self.inference_convergence[-100:]
                lines.append(f"\nConvergence: mean={np.mean(recent):.2f}, "
                             f"min={np.min(recent):.2f}")
            if self.iters_used:
                recent_iters = self.iters_used[-100:] if len(self.iters_used) >= 100 else self.iters_used
                lines.append(f"Avg iters used: {np.mean(recent_iters):.2f}")
                early_stop_rate = sum(1 for i in recent_iters if i < max(recent_iters)) / len(recent_iters)
                lines.append(f"Early stop rate: {early_stop_rate:.0%}")
        else:
            lines.append("Backprop baseline — no ePC diagnostics")
        ax.text(0.05, 0.5, '\n'.join(lines), fontsize=9,
                verticalalignment='center', family='monospace')

        # --- Row 3: Profiling ---
        if self.has_timing():
            phase_keys = [
                ('data_transfer', 'Data xfer', '#e0e0e0'),
                ('inference_init', 'Error init', '#1f77b4'),
                ('inference_forward', 'Inf forward', '#ff7f0e'),
                ('inference_backward', 'Inf backward', '#2ca02c'),
                ('inference_step', 'Newton step', '#d62728'),
                ('weight_forward', 'Wt forward', '#9467bd'),
                ('weight_backward', 'Wt backward', '#8c564b'),
                ('optimizer_step', 'Optim step', '#e377c2'),
            ]
            n_timing = min(len(v) for v in self.timing.values() if v)

            # [3,0] Stacked area chart
            ax = axes[3, 0]
            window = 1
            if n_timing > 0:
                window = max(1, min(20, n_timing // 10))
                smoothed = {}
                for key, _, _ in phase_keys:
                    vals = np.array(self.timing[key][:n_timing], dtype=np.float64)
                    if window > 1 and len(vals) >= window:
                        kernel = np.ones(window) / window
                        smoothed[key] = np.convolve(vals, kernel, mode='valid')
                    else:
                        smoothed[key] = vals
                n_smooth = min(len(v) for v in smoothed.values())
                x_smooth = np.arange(n_smooth)
                bottoms = np.zeros(n_smooth)
                for key, label, color in phase_keys:
                    vals = smoothed[key][:n_smooth]
                    ax.fill_between(x_smooth, bottoms, bottoms + vals,
                                    alpha=0.7, label=label, color=color)
                    bottoms += vals
            ax.set_xlabel('Batch')
            ax.set_ylabel('Time (ms)')
            ax.set_title(f'Per-Batch Timing (MA-{window})')
            ax.legend(fontsize=6, ncol=2, loc='upper right')
            ax.grid(True, alpha=0.3)

            # [3,1] Average breakdown (horizontal bar)
            ax = axes[3, 1]
            last_n = min(200, n_timing) if n_timing > 0 else 0
            if last_n > 0:
                labels = []
                values = []
                colors = []
                for key, label, color in phase_keys:
                    vals = self.timing[key][-last_n:]
                    avg_val = np.mean(vals) if vals else 0
                    labels.append(label)
                    values.append(avg_val)
                    colors.append(color)
                y_pos = np.arange(len(labels))
                ax.barh(y_pos, values, color=colors, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_xlabel('Avg time (ms)')
                ax.set_title(f'Avg Phase Timing (last {last_n} batches)')
                for i, v in enumerate(values):
                    if v > 0:
                        ax.text(v + 0.5, i, f'{v:.1f}ms', va='center', fontsize=7)
                ax.grid(True, alpha=0.3, axis='x')

            # [3,2] Performance summary text
            ax = axes[3, 2]
            ax.axis('off')
            if last_n > 0:
                def tavg(key):
                    v = self.timing.get(key, [])[-last_n:]
                    return np.mean(v) if v else 0
                t_total = sum(tavg(k) for k, _, _ in phase_keys)
                t_wt = tavg('weight_forward') + tavg('weight_backward')
                throughput = batch_size / (t_total / 1000) if t_total > 0 else 0
                plines = [
                    f"Avg batch:    {t_total:.0f} ms",
                    f"Throughput:   {throughput:.0f} samples/sec",
                    f"",
                    f"Inference:    {tavg('inference'):.0f} ms "
                    f"({tavg('inference')/t_total*100:.0f}%)" if t_total > 0 else "",
                    f"  Init:       {tavg('inference_init'):.0f} ms",
                    f"  Forward:    {tavg('inference_forward'):.0f} ms",
                    f"  Backward:   {tavg('inference_backward'):.0f} ms",
                    f"  Newton:     {tavg('inference_step'):.0f} ms",
                    f"",
                    f"Weight phase: {t_wt:.0f} ms "
                    f"({t_wt/t_total*100:.0f}%)" if t_total > 0 else "",
                    f"Optim step:   {tavg('optimizer_step'):.0f} ms",
                ]
                ax.text(0.05, 0.5, '\n'.join(plines), fontsize=9,
                        verticalalignment='center', family='monospace')

        # --- Hypothesis row (when ePC data available) ---
        if has_hyp:
            hyp_row = nrows - 1  # always the last row

            # [hyp,0] Newton quality: rank-1 ratio + Woodbury coefficient
            ax = axes[hyp_row, 0]
            ax2 = ax.twinx()
            if self.newton_rank1_ratio:
                ax.plot(self.newton_rank1_ratio, alpha=0.5, linewidth=0.5,
                        color='blue', label='||u||^2/||g||^2')
                if len(self.newton_rank1_ratio) > 50:
                    window = min(50, len(self.newton_rank1_ratio) // 5)
                    ma = np.convolve(self.newton_rank1_ratio,
                                     np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(self.newton_rank1_ratio)), ma,
                            linewidth=1.5, color='blue')
            if self.newton_coeff:
                ax2.plot(self.newton_coeff, alpha=0.3, linewidth=0.5,
                         color='red', label='Woodbury coeff')
                if len(self.newton_coeff) > 50:
                    window = min(50, len(self.newton_coeff) // 5)
                    ma = np.convolve(self.newton_coeff,
                                     np.ones(window)/window, mode='valid')
                    ax2.plot(range(window-1, len(self.newton_coeff)), ma,
                             linewidth=1.5, color='red')
                ax2.set_ylabel('Woodbury coeff', color='red', fontsize=8)
                ax2.tick_params(axis='y', labelcolor='red', labelsize=7)
            ax.set_xlabel('Batch')
            ax.set_ylabel('||u||^2 / ||g||^2', color='blue', fontsize=8)
            ax.tick_params(axis='y', labelcolor='blue', labelsize=7)
            ax.set_title('H1: Newton Quality (rank-1 capture)')
            ax.grid(True, alpha=0.3)

            # [hyp,1] Causal asymmetry + per-position snapshot
            ax = axes[hyp_row, 1]
            if self.causal_early_late:
                ax.plot(self.causal_early_late, alpha=0.5, linewidth=0.5, color='green')
                if len(self.causal_early_late) > 50:
                    window = min(50, len(self.causal_early_late) // 5)
                    ma = np.convolve(self.causal_early_late,
                                     np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(self.causal_early_late)), ma,
                            linewidth=1.5, color='green', label=f'MA-{window}')
                ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                           label='Symmetric (1.0)')
                ax.legend(fontsize=7)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Early / Late error norm')
            ax.set_title('H2: Causal Asymmetry (early/late ratio)')
            ax.grid(True, alpha=0.3)

            # [hyp,2] Summary of all hypotheses
            ax = axes[hyp_row, 2]
            ax.axis('off')
            lines = ['HYPOTHESIS STATUS:']
            # H1
            if self.newton_rank1_ratio:
                recent = self.newton_rank1_ratio[-100:]
                r1 = np.mean(recent)
                status = 'OK' if r1 < 2.0 else 'CONCERN' if r1 < 10.0 else 'BAD'
                lines.append(f"\nH1 Newton rank-1: ratio={r1:.3f} [{status}]")
                lines.append(f"   (>2 means rank-1 misses curvature)")
            if self.newton_coeff:
                recent = self.newton_coeff[-100:]
                lines.append(f"   Woodbury coeff: {np.mean(recent):.4f}")
            # H2
            if self.causal_early_late:
                recent = self.causal_early_late[-100:]
                ratio = np.mean(recent)
                status = 'OK' if 0.5 < ratio < 2.0 else 'CONCERN'
                lines.append(f"\nH2 Causal asym: ratio={ratio:.3f} [{status}]")
                lines.append(f"   (>>1 = early dominates, <<1 = late)")
            # H3
            for i, ratios in enumerate(self.rmsnorm_ratios):
                if ratios:
                    recent = ratios[-100:]
                    r = np.mean(recent)
                    status = 'OK' if 0.5 < r < 2.0 else 'CONCERN'
                    lines.append(f"\nH3 RMSNorm L{i+1}: ratio={r:.3f} [{status}]")
            # H5
            for i, fracs in enumerate(self.gate_frac_near_zero):
                if fracs:
                    recent = fracs[-100:]
                    f = np.mean(recent)
                    status = 'OK' if f < 0.1 else 'CONCERN' if f < 0.3 else 'BAD'
                    lines.append(f"\nH5 SiLU gate L{i+1}: {f:.1%} near-zero [{status}]")

            ax.text(0.02, 0.95, '\n'.join(lines), fontsize=8,
                    verticalalignment='top', family='monospace',
                    transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Diagnostics saved to {save_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_loss(model, test_loader, device, task, use_epc=True):
    """Evaluate returning both accuracy and loss."""
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            if use_epc:
                logits = model(inputs)
            else:
                logits = model(inputs)

            acc = compute_accuracy(logits, targets, task=task)
            if task == 'classify':
                loss = F.cross_entropy(logits, targets).item()
            else:
                b, l, v = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(b * l, v), targets.reshape(b * l)
                ).item()

            total_acc += acc
            total_loss += loss
            n_batches += 1

    return total_acc / n_batches, total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ePC-Mamba synthetic task training')
    parser.add_argument('--task', choices=['copy', 'selective', 'classify'],
                        default='copy', help='Which synthetic task')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--iters', type=int, default=2, help='Error optimization iterations (T)')
    parser.add_argument('--e_lr', type=float, default=0.01,
                        help='Error optimization learning rate (SGD/Adam)')
    parser.add_argument('--error_optim', choices=['newton', 'adam', 'sgd'],
                        default='newton', help='Error optimization method')
    parser.add_argument('--damping', type=float, default=0.1,
                        help='Newton damping (0.1 matches ePC-ResNet)')
    parser.add_argument('--output_loss', choices=['ce', 'mse'], default='mse',
                        help='Output loss function (mse matches ePC paper)')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--baseline', action='store_true',
                        help='Run backprop baseline instead of ePC')
    parser.add_argument('--profile', action='store_true',
                        help='Enable per-phase timing profiling')
    parser.add_argument('--chart_interval', type=int, default=10,
                        help='Save diagnostic chart every N epochs')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Config
    config = Mamba2Config(
        d_model=args.d_model,
        d_state=64,
        headdim=64,
        expand=2,
        chunk_size=min(64, args.seq_len),
        n_layer=args.n_layer,
    )
    print(f"Config: d_model={config.d_model}, d_inner={config.d_inner}, "
          f"nheads={config.nheads}, n_layer={config.n_layer}")

    # Generate data
    print(f"\nGenerating {args.task} task data...")
    if args.task == 'copy':
        train_x, train_y = generate_copy_data(
            args.n_train, args.seq_len, args.vocab_size)
        test_x, test_y = generate_copy_data(
            args.n_test, args.seq_len, args.vocab_size)
    elif args.task == 'selective':
        train_x, train_y = generate_selective_copy_data(
            args.n_train, args.seq_len, args.vocab_size)
        test_x, test_y = generate_selective_copy_data(
            args.n_test, args.seq_len, args.vocab_size)
    elif args.task == 'classify':
        train_x, train_y = generate_classify_data(
            args.n_train, args.seq_len, args.vocab_size)
        test_x, test_y = generate_classify_data(
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
        model = ePCMambaSynthetic(
            config, vocab_size=args.vocab_size, task=args.task,
            iters=args.iters, e_lr=args.e_lr,
            error_optim=args.error_optim, damping=args.damping,
            output_loss=args.output_loss,
        ).to(device)
        model.pce.profiling = args.profile
        eopt = args.error_optim
        if eopt == 'newton':
            print(f"Model: ePC-Mamba (T={args.iters}, {eopt}, damping={args.damping})")
        else:
            print(f"Model: ePC-Mamba (T={args.iters}, {eopt}, e_lr={args.e_lr})")
    else:
        model = BackpropMambaBaseline(
            config, vocab_size=args.vocab_size, task=args.task,
        ).to(device)
        print("Model: Backprop-Mamba (baseline)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Diagnostics
    num_error_layers = config.n_layer - 1 if use_epc else 0
    diagnostics = Diagnostics(num_error_layers, task=args.task)

    # Output directory
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Training
    print(f"\n{'Epoch':>5} {'Loss':>10} {'Train Acc':>10} {'Test Acc':>10} {'ms/batch':>10}")
    print("-" * 50)

    best_test_acc = 0.0
    prof = args.profile

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_time = 0.0
        n_batches = 0

        for batch in train_loader:
            if prof:
                _t = _sync_time()

            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            batch_size = inputs.shape[0]

            if prof:
                _t2 = _sync_time()
                t_data = (_t2 - _t) * 1000
                _t = _t2

            if use_epc:
                # Phase 1: Inference (optimize errors)
                t_batch_start = time.perf_counter()
                model(inputs, targets)

                if prof:
                    _t2 = _sync_time()
                    t_inference = (_t2 - _t) * 1000
                    _t = _t2

                # Phase 2: Weight update via E_local
                optimizer.zero_grad()

                if prof:
                    _tw = _sync_time()

                weight_loss = model.compute_weight_loss(inputs, targets, batch_size)

                if prof:
                    _t2 = _sync_time()
                    t_wfwd = (_t2 - _tw) * 1000
                    _tw = _t2

                weight_loss.backward()

                if prof:
                    _t2 = _sync_time()
                    t_wbwd = (_t2 - _tw) * 1000
                    _tw = _t2

                optimizer.step()

                if prof:
                    _t2 = _sync_time()
                    t_optim = (_t2 - _tw) * 1000

                loss_val = weight_loss.item()

                # Collect diagnostics BEFORE accuracy eval (which resets errors)
                diag = model.get_diagnostics()
                weight_mags = get_weight_magnitudes(model, use_epc=True)

                # Hypothesis diagnostics (every batch — lightweight)
                # Must run while errors are still populated as tensors
                x_detached = model.embedding(inputs).detach()
                hyp_diag = model.pce.get_hypothesis_diagnostics(x_detached)
                # Save full per-position snapshot every 50 batches
                save_snap = (n_batches % 50 == 0)

                # Get accuracy (feedforward without ePC for clean eval)
                # NOTE: This sets pce.errors = [0.0, ...], destroying tensor errors.
                # All diagnostics that need tensor errors must be collected above.
                with torch.no_grad():
                    logits = model(inputs)
                    acc = compute_accuracy(logits, targets, task=args.task)

                t_batch_end = time.perf_counter()

                diagnostics.update_train(
                    acc=acc, loss=loss_val, diagnostics=diag,
                    lr=args.lr, weight_mags=weight_mags,
                )
                diagnostics.update_hypothesis(hyp_diag, save_snapshot=save_snap)

                if prof:
                    inf_profile = model.pce._profile if hasattr(model.pce, '_profile') else {}
                    diagnostics.update_timing({
                        'data_transfer': t_data,
                        'inference': t_inference,
                        'inference_init': inf_profile.get('init_ms', 0),
                        'inference_forward': inf_profile.get('forward_ms', 0),
                        'inference_backward': inf_profile.get('backward_ms', 0),
                        'inference_step': inf_profile.get('step_ms', 0),
                        'weight_forward': t_wfwd,
                        'weight_backward': t_wbwd,
                        'optimizer_step': t_optim,
                    })

                epoch_time += (t_batch_end - t_batch_start)

            else:
                # Standard backprop baseline
                t_batch_start = time.perf_counter()
                optimizer.zero_grad()
                logits = model(inputs)
                if args.task == 'classify':
                    loss = F.cross_entropy(logits, targets)
                else:
                    b, l, v = logits.shape
                    loss = F.cross_entropy(
                        logits.reshape(b * l, v), targets.reshape(b * l))
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                with torch.no_grad():
                    acc = compute_accuracy(logits, targets, task=args.task)

                t_batch_end = time.perf_counter()
                weight_mags = get_weight_magnitudes(model, use_epc=False)
                diagnostics.update_train_baseline(
                    acc=acc, loss=loss_val,
                    lr=args.lr, weight_mags=weight_mags,
                )
                epoch_time += (t_batch_end - t_batch_start)

            epoch_loss += loss_val
            epoch_acc += acc
            n_batches += 1

        # End-of-epoch evaluation
        test_acc, test_loss = evaluate_with_loss(
            model, test_loader, device,
            task=args.task, use_epc=use_epc,
        )
        diagnostics.update_test(test_acc, test_loss)
        best_test_acc = max(best_test_acc, test_acc)

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        avg_time = epoch_time * 1000 / n_batches

        print(f"{epoch:5d} {avg_loss:10.4f} {avg_acc:10.4f} {test_acc:10.4f} {avg_time:10.1f}")

        # Periodic ePC diagnostics
        if use_epc and epoch % 10 == 0:
            diag = model.get_diagnostics()
            print(f"  ePC: E_init={diag['E_initial']:.2f}, "
                  f"E_final={diag['E_final']:.2f}, "
                  f"convergence={diag['convergence']:.2f}, "
                  f"iters={diag['iters_used']}")
            if diag['error_norms']:
                norms_str = ', '.join(f'{n:.4f}' for n in diag['error_norms'])
                print(f"  Error norms: [{norms_str}]")

        # Performance report
        if prof and epoch % 10 == 0:
            diagnostics.print_performance_report(epoch, args.batch_size)

        # Save diagnostic chart
        if epoch % args.chart_interval == 0 or epoch == args.epochs:
            mode = 'epc' if use_epc else 'baseline'
            chart_path = os.path.join(
                out_dir, f'diagnostics_{args.task}_{mode}_epoch_{epoch}.png')
            diagnostics.plot(
                chart_path, epoch=epoch, num_epochs=args.epochs,
                use_epc=use_epc, batch_size=args.batch_size,
                task=args.task, iters=args.iters, error_optim=args.error_optim,
                e_lr=args.e_lr, damping=args.damping,
            )

        # Early success
        if best_test_acc >= 0.99 and epoch >= 5:
            print(f"\nSuccess! Test accuracy {best_test_acc:.4f} >= 99% at epoch {epoch}")
            break

    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    if best_test_acc >= 0.95:
        print("PASS: ePC-Mamba works with SSM blocks!")
    elif best_test_acc >= 0.90:
        print("PROMISING: 90%+ accuracy, may need more epochs or tuning.")
    else:
        print("FAIL: Did not reach 90% accuracy. Debug needed.")

    # Final chart
    mode = 'epc' if use_epc else 'baseline'
    chart_path = os.path.join(out_dir, f'diagnostics_{args.task}_{mode}_final.png')
    diagnostics.plot(
        chart_path, epoch=args.epochs, num_epochs=args.epochs,
        use_epc=use_epc, batch_size=args.batch_size,
        task=args.task, iters=args.iters, error_optim=args.error_optim,
        e_lr=args.e_lr, damping=args.damping,
    )


if __name__ == '__main__':
    main()
