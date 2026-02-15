#!/usr/bin/env python3
"""
Training script for Naja language model.

Supports:
  - Stage 1a/1b/1c/2 from data_gen.py (shared with JEPA)
  - Ablation tasks from tasks.py
  - Phase 4: Surprise-gated training (two-pass forward)
  - Phase 6: KL divergence surprise at inference time
  - AMP, torch.compile, profiling
  - Per-feature ablation via config toggles

Usage:
  # Stage 1b (default, sanity check)
  python train_naja.py --stage 1b --epochs 30

  # Stage 2 (pattern induction)
  python train_naja.py --stage 2

  # Ablation task: associative recall with delta rule ON
  python train_naja.py --task associative_recall --epochs 50

  # Base Mamba3 config (ablation control)
  python train_naja.py --task parity --preset mamba3_base

  # Full Naja with surprise gating
  python train_naja.py --stage 1b --use_surprise_gate

Do NOT run full training on CPU (Mistake #36).
"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- Path setup (same pattern as jepa_model.py) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.energy_reasoning.data_gen import get_stage_data
from experiments.Naja.naja import (
    NajaConfig, NajaLM, KLSurpriseTracker, mamba3_base_config,
)
from experiments.Naja.tasks import get_task_data, ABLATION_TASKS


# ---------------------------------------------------------------------------
# Presets: named config bundles for common ablation variants
# ---------------------------------------------------------------------------

PRESETS = {
    'naja_full': dict(
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=1,
    ),
    'mamba3_base': dict(
        use_delta_rule=False, use_pope_perp=False, per_channel_decay=False,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=1,
    ),
    'mamba3_rope': dict(
        use_delta_rule=False, use_pope_perp=False, per_channel_decay=False,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=1,
        use_pope=False,
    ),
    'delta_only': dict(
        use_delta_rule=True, use_pope_perp=False, per_channel_decay=False,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=1,
    ),
    'pope_perp_only': dict(
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=False,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=1,
    ),
    'per_channel_only': dict(
        use_delta_rule=False, use_pope_perp=False, per_channel_decay=True,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=1,
    ),
    'delta_per_channel': dict(
        use_delta_rule=True, use_pope_perp=False, per_channel_decay=True,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=1,
    ),
    'stable_reparam': dict(
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        stable_reparam=True, use_surprise_gate=False, mimo_rank=1,
    ),
    'surprise': dict(
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        stable_reparam=False, use_surprise_gate=True, mimo_rank=1,
    ),
    'mimo2': dict(
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=2,
    ),
    'mimo4': dict(
        use_delta_rule=True, use_pope_perp=True, per_channel_decay=True,
        stable_reparam=False, use_surprise_gate=False, mimo_rank=4,
    ),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train Naja language model')

    # Stage / task selection
    g = p.add_mutually_exclusive_group()
    g.add_argument('--stage', choices=['1a', '1b', '1c', '2'], default=None,
                   help='Data generation stage (from data_gen.py)')
    g.add_argument('--task', choices=list(ABLATION_TASKS.keys()), default=None,
                   help='Ablation task (from tasks.py)')

    # Architecture
    p.add_argument('--preset', choices=list(PRESETS.keys()), default='naja_full',
                   help='Config preset (default: naja_full)')
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_state', type=int, default=64)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--headdim', type=int, default=64)
    p.add_argument('--expand', type=int, default=2)
    p.add_argument('--mimo_rank', type=int, default=None,
                   help='Override MIMO rank from preset')

    # Feature toggles (override preset)
    p.add_argument('--use_delta_rule', action='store_true', default=None)
    p.add_argument('--no_delta_rule', action='store_true')
    p.add_argument('--use_pope_perp', action='store_true', default=None)
    p.add_argument('--no_pope_perp', action='store_true')
    p.add_argument('--no_pope', action='store_true',
                   help='Use RoPE instead of PoPE for B/C encoding')
    p.add_argument('--per_channel_decay', action='store_true', default=None)
    p.add_argument('--no_per_channel_decay', action='store_true')
    p.add_argument('--stable_reparam', action='store_true', default=None)
    p.add_argument('--use_surprise_gate', action='store_true', default=None)
    p.add_argument('--use_chunkwise', action='store_true', default=False,
                   help='Legacy gradient-checkpointed chunking')
    p.add_argument('--use_wy_chunkwise', action='store_true', default=True,
                   help='WY chunkwise parallelism (default)')
    p.add_argument('--no_wy_chunkwise', action='store_true',
                   help='Disable WY chunkwise (use naive sequential)')
    p.add_argument('--chunk_size', type=int, default=64)

    # Data
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--vocab_size', type=int, default=16)
    p.add_argument('--n_train', type=int, default=5000)
    p.add_argument('--n_test', type=int, default=1000)
    p.add_argument('--n_examples', type=int, default=5, help='Stage 2: example pairs')
    p.add_argument('--n_rules', type=int, default=5, help='Stage 2: number of rules')

    # Training
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup_epochs', type=int, default=4)
    p.add_argument('--w_clip', type=float, default=1.0, help='Gradient clipping max norm')
    p.add_argument('--weight_decay', type=float, default=0.01)

    # Performance
    p.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    p.add_argument('--compile', action='store_true', help='torch.compile the model')
    p.add_argument('--profile', action='store_true', help='Print timing breakdown (5 epochs)')

    # Diagnostics
    p.add_argument('--diag_every', type=int, default=5,
                   help='Save diagnostic charts every N epochs (0 to disable)')
    p.add_argument('--diag_dir', type=str, default=None,
                   help='Directory for diagnostic charts (default: auto)')

    # Output
    p.add_argument('--results_file', type=str, default=None,
                   help='Append JSON result line to this file after training')

    # Misc
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--print_every', type=int, default=1, help='Print every N epochs')

    args = p.parse_args()

    # Default to stage 1b if neither stage nor task specified
    if args.stage is None and args.task is None:
        args.stage = '1b'

    # Stage 2 smart defaults
    if args.stage == '2':
        if args.n_train == 5000:
            args.n_train = 10000
        if args.n_test == 1000:
            args.n_test = 2000
        if args.epochs == 30:
            args.epochs = 50
        if args.batch_size == 32:
            args.batch_size = 128
        args.seq_len = max(2 * args.n_examples + 2, args.seq_len)

    # Handle --no_wy_chunkwise
    if args.no_wy_chunkwise:
        args.use_wy_chunkwise = False

    # VRAM-aware defaults for batch size
    if args.device == 'auto' or args.device.startswith('cuda'):
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb <= 4.0:
                    # Naja is very memory-efficient (~200MB peak at batch=32).
                    # Only reduce for very small GPUs (≤4GB).
                    if args.stage == '2' and args.batch_size == 128:
                        args.batch_size = 64
                    # chunk_size=64 is fine: with virtual tokens (Cs=128)
                    # the WY matrices are only ~16MB per head.
                    print(f"VRAM: {vram_gb:.1f}GB detected — "
                          f"batch_size={args.batch_size}, "
                          f"chunk_size={args.chunk_size}")
        except ImportError:
            pass

    return args


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args) -> NajaConfig:
    """Build NajaConfig from preset + CLI overrides."""
    preset = PRESETS[args.preset].copy()

    # CLI overrides
    if args.use_delta_rule is not None:
        preset['use_delta_rule'] = True
    if args.no_delta_rule:
        preset['use_delta_rule'] = False
    if args.use_pope_perp is not None:
        preset['use_pope_perp'] = True
    if args.no_pope_perp:
        preset['use_pope_perp'] = False
    if args.per_channel_decay is not None:
        preset['per_channel_decay'] = True
    if args.no_per_channel_decay:
        preset['per_channel_decay'] = False
    if args.stable_reparam is not None:
        preset['stable_reparam'] = True
    if args.use_surprise_gate is not None:
        preset['use_surprise_gate'] = True
    if args.no_pope:
        preset['use_pope'] = False
    if args.mimo_rank is not None:
        preset['mimo_rank'] = args.mimo_rank

    return NajaConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layer=args.n_layer,
        headdim=args.headdim,
        expand=args.expand,
        chunk_size=args.chunk_size,
        use_chunkwise=args.use_chunkwise,
        use_wy_chunkwise=args.use_wy_chunkwise,
        **preset,
    )


# ---------------------------------------------------------------------------
# LR schedule (same as train_jepa.py)
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.2 + 0.8 * epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_nextstep(model, loader, device, amp_ctx):
    """Evaluate next-step prediction accuracy (Stage 1 / ablation tasks)."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            seqs = batch[0].to(device)
            with amp_ctx():
                logits = model(seqs)
            # Next-step: logits[:, :-1] predicts seqs[:, 1:]
            preds = logits[:, :-1].argmax(dim=-1)
            targets = seqs[:, 1:]
            # Ignore PAD (token 0) positions in target
            mask = targets != 0
            correct += (preds[mask] == targets[mask]).sum().item()
            total += mask.sum().item()

    return correct / max(total, 1)


def evaluate_last_token(model, loader, device, amp_ctx):
    """Evaluate accuracy predicting the answer token.

    The answer is at seqs[:, -1], so the genuine prediction is at
    logits[:, -2] (which sees everything up to and including the query
    token but NOT the answer).  Using logits[:, -1] would be trivial
    copy-last-token — see Mistake #41.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            seqs = batch[0].to(device)
            targets = batch[1].to(device)
            with amp_ctx():
                logits = model(seqs)
            pred_last = logits[:, -2].argmax(dim=-1)
            correct += (pred_last == targets).sum().item()
            total += targets.shape[0]

    return correct / max(total, 1)


def evaluate_with_kl_surprise(model, loader, device, amp_ctx, kl_tracker):
    """Evaluate with Phase 6 KL surprise at inference time."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            seqs = batch[0].to(device)
            targets = batch[1].to(device)
            with amp_ctx():
                logits_p1 = model(seqs)
                # Compute KL surprise from EMA distribution
                surprise = kl_tracker(logits_p1)
                # Re-run with surprise signal
                logits = model(seqs, surprise=surprise.detach())
            pred_last = logits[:, -2].argmax(dim=-1)
            correct += (pred_last == targets).sum().item()
            total += targets.shape[0]

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def collect_diagnostics(model, config, epoch, device, amp_ctx):
    """Collect diagnostic snapshot from model state."""
    diag = {'epoch': epoch}

    # Alpha (decay) distribution
    model.eval()
    with torch.no_grad():
        all_alpha = []
        all_beta1 = []
        all_beta2 = []
        grad_norms = {}
        weight_norms = {}

        for i, layer in enumerate(model.layers):
            mixer = layer.mixer
            # Weight norms per layer
            weight_norms[f'L{i}_in_proj'] = mixer.in_proj.weight.norm().item()
            weight_norms[f'L{i}_out_proj'] = mixer.out_proj.weight.norm().item()

            # Gradient norms (if available)
            for name, p in mixer.named_parameters():
                if p.grad is not None:
                    grad_norms[f'L{i}_{name}'] = p.grad.norm().item()

        diag['weight_norms'] = weight_norms
        diag['grad_norms'] = grad_norms

        # Run a forward pass to capture alpha/beta values
        dummy = torch.randint(0, 16, (4, 64), device=device)
        hooks = []
        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Access the mixer's internal state after forward
                pass
            return hook_fn

        # Simpler: directly access the mixer parameters for alpha stats
        for i, layer in enumerate(model.layers):
            mixer = layer.mixer
            if hasattr(mixer, 'decay_bias'):
                # Per-channel decay: the bias determines typical alpha range
                bias = mixer.decay_bias.detach()
                if config.stable_reparam:
                    alpha_approx = 1.0 - 1.0 / (bias.pow(2) + 0.5)
                else:
                    alpha_approx = torch.sigmoid(bias)
                all_alpha.append(alpha_approx.cpu().flatten())

            if hasattr(mixer, 'beta1_proj'):
                # Beta1 bias gives typical gate value
                b1_bias = mixer.beta1_proj.bias.detach() if mixer.beta1_proj.bias is not None else torch.zeros(1)
                all_beta1.append(torch.sigmoid(b1_bias).cpu().flatten())

            if hasattr(mixer, 'beta2_proj'):
                b2_bias = mixer.beta2_proj.bias.detach() if mixer.beta2_proj.bias is not None else torch.zeros(1)
                all_beta2.append(torch.sigmoid(b2_bias).cpu().flatten())

        if all_alpha:
            diag['alpha'] = torch.cat(all_alpha).numpy()
        if all_beta1:
            diag['beta1'] = torch.cat(all_beta1).numpy()
        if all_beta2:
            diag['beta2'] = torch.cat(all_beta2).numpy()

    return diag


def save_diagnostic_charts(history, diag, save_dir, epoch):
    """Save diagnostic charts as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [diagnostics] matplotlib not available, skipping charts")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Naja Diagnostics — Epoch {epoch}', fontsize=14)

    # 1. Loss curve
    ax = axes[0, 0]
    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]
    ax.plot(epochs, losses, 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    # Mark NaN losses
    nan_epochs = [e for e, l in zip(epochs, losses) if math.isnan(l)]
    if nan_epochs:
        ax.axvline(x=nan_epochs[0], color='r', linestyle='--', label=f'NaN at epoch {nan_epochs[0]}')
        ax.legend()

    # 2. Accuracy curves
    ax = axes[0, 1]
    train_acc = [h['train_acc'] for h in history]
    test_acc = [h['test_acc'] for h in history]
    ax.plot(epochs, train_acc, 'b-', label='Train', linewidth=1.5)
    ax.plot(epochs, test_acc, 'r--', label='Test', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train / Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 3. Alpha (decay) distribution
    ax = axes[0, 2]
    if 'alpha' in diag:
        ax.hist(diag['alpha'], bins=50, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(x=diag['alpha'].mean(), color='red', linestyle='--',
                    label=f'mean={diag["alpha"].mean():.3f}')
        ax.set_xlabel('Alpha (decay)')
        ax.set_title('Per-Channel Decay Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # 4. Gradient norms per layer
    ax = axes[1, 0]
    if diag.get('grad_norms'):
        names = list(diag['grad_norms'].keys())
        values = list(diag['grad_norms'].values())
        # Group by layer
        layer_norms = {}
        for n, v in zip(names, values):
            layer = n.split('_')[0]  # "L0", "L1", etc.
            layer_norms.setdefault(layer, []).append(v)
        layer_means = {k: sum(v)/len(v) for k, v in layer_norms.items()}
        ax.bar(list(layer_means.keys()), list(layer_means.values()), color='coral')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Gradient Norm')
        ax.set_title('Gradient Norms by Layer')
    else:
        ax.text(0.5, 0.5, 'No gradients', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # 5. Weight norms per layer
    ax = axes[1, 1]
    if diag.get('weight_norms'):
        names = list(diag['weight_norms'].keys())
        values = list(diag['weight_norms'].values())
        ax.barh(names, values, color='mediumseagreen')
        ax.set_xlabel('Weight Norm')
        ax.set_title('Weight Norms')
    ax.grid(True, alpha=0.3)

    # 6. Gradient norm history
    ax = axes[1, 2]
    if any('grad_norm' in h for h in history):
        gn_epochs = [h['epoch'] for h in history if 'grad_norm' in h]
        gn_values = [h['grad_norm'] for h in history if 'grad_norm' in h]
        ax.plot(gn_epochs, gn_values, 'g-', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Grad Norm (clipped)')
        ax.set_title('Gradient Norm Over Time')
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / f'epoch_{epoch:03d}.png'
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  [diagnostics] saved {path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # --- Seed ---
    torch.manual_seed(args.seed)

    # --- Device ---
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # --- Config ---
    config = build_config(args)

    # --- Data ---
    has_targets = False
    if args.task is not None:
        data = get_task_data(args.task, n_train=args.n_train, n_test=args.n_test)
        vocab_size = data['vocab_size']
        has_targets = True
        train_ds = TensorDataset(data['train_seqs'], data['train_targets'])
        test_ds = TensorDataset(data['test_seqs'], data['test_targets'])
        task_name = args.task
    elif args.stage in ('1a', '1b', '1c'):
        data = get_stage_data(
            args.stage, n_train=args.n_train, n_test=args.n_test,
            seq_len=args.seq_len, vocab_size=args.vocab_size,
        )
        vocab_size = args.vocab_size
        train_ds = TensorDataset(data['train_seqs'])
        test_ds = TensorDataset(data['test_seqs'])
        task_name = f'stage_{args.stage}'
    else:  # stage 2
        data = get_stage_data(
            '2', n_train=args.n_train, n_test=args.n_test,
            seq_len=args.seq_len, vocab_size=args.vocab_size,
            n_examples=args.n_examples, n_rules=args.n_rules,
        )
        vocab_size = args.vocab_size
        has_targets = True
        train_ds = TensorDataset(data['train_seqs'], data['train_targets'], data['train_rules'])
        test_ds = TensorDataset(data['test_seqs'], data['test_targets'], data['test_rules'])
        task_name = 'stage_2'

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = NajaLM(config, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Phase 6: KL surprise tracker (inference-time only)
    kl_tracker = KLSurpriseTracker(vocab_size).to(device) if has_targets else None

    # --- AMP ---
    use_amp = (not args.no_amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        def amp_ctx():
            return torch.amp.autocast('cuda', dtype=torch.float16)
    else:
        amp_ctx = nullcontext

    # --- torch.compile ---
    if args.compile:
        if config.use_wy_chunkwise:
            print("torch.compile: SKIPPED (incompatible with WY chunkwise)")
        else:
            try:
                model = torch.compile(model)
                print("torch.compile: ON")
            except Exception as e:
                print(f"torch.compile: FAILED ({e}), continuing without")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.warmup_epochs, args.epochs),
    )

    # --- Config summary ---
    features = []
    if not config.use_pope:
        features.append('rope')
    if config.use_delta_rule:
        features.append('delta')
    if config.use_pope_perp:
        features.append('pope_perp')
    if config.per_channel_decay:
        features.append('per_ch_decay')
    if config.stable_reparam:
        features.append('stable')
    if config.use_surprise_gate:
        features.append('surprise')
    if config.mimo_rank > 1:
        features.append(f'mimo_r{config.mimo_rank}')
    if config.use_wy_chunkwise:
        features.append(f'wy_chunk{config.chunk_size}')
    elif config.use_chunkwise:
        features.append(f'chunk{config.chunk_size}')
    feat_str = '+'.join(features) if features else 'base_mamba3'

    print(f"Naja [{feat_str}] | {task_name} | {n_params:,} params | {device}")
    print(f"  d_model={config.d_model} d_state={config.d_state} "
          f"n_layer={config.n_layer} nheads={config.nheads} "
          f"mimo_r={config.mimo_rank}")
    print(f"  epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"vocab={vocab_size} seq_len={train_ds[0][0].shape[0]}")
    if use_amp:
        print("  AMP: fp16")
    print()

    # --- Header ---
    if has_targets:
        print(f"{'Epoch':>5}  {'Loss':>8}  {'TrAcc':>7}  {'TstAcc':>7}  "
              f"{'KL_Acc':>7}  {'ms/b':>6}  {'ep_s':>6}")
    else:
        print(f"{'Epoch':>5}  {'Loss':>8}  {'TrAcc':>7}  {'TstAcc':>7}  "
              f"{'ms/b':>6}  {'ep_s':>6}")

    # --- Profiling ---
    if args.profile:
        prof = {'forward': 0.0, 'backward': 0.0, 'step': 0.0, 'eval': 0.0}
        profile_epochs = min(5, args.epochs)

    # --- Diagnostics ---
    history = []
    diag_dir = args.diag_dir or os.path.join(
        os.path.dirname(__file__), 'diagnostics', f'{task_name}_{feat_str}')

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0
        n_nan_batches = 0
        epoch_grad_norm = 0.0
        t_epoch = time.perf_counter()

        for batch in train_loader:
            seqs = batch[0].to(device)

            t0 = time.perf_counter()

            # --- Forward ---
            with amp_ctx():
                if config.use_surprise_gate:
                    # Phase 4: two-pass forward with surprise
                    logits, _surprise = model.forward_with_surprise(seqs)
                else:
                    logits = model(seqs)

                # Loss: next-step prediction
                if has_targets:
                    # Predict the answer token: logits[:, -2] sees [prefix..., Q]
                    # but NOT the answer at seqs[:, -1].  See Mistake #41.
                    targets = batch[1].to(device)
                    loss = F.cross_entropy(logits[:, -2], targets)
                    # Also add next-step CE on the full sequence for representation learning
                    loss_ns = F.cross_entropy(
                        logits[:, :-1].reshape(-1, vocab_size),
                        seqs[:, 1:].reshape(-1),
                        ignore_index=0,  # ignore PAD
                    )
                    loss = loss + 0.5 * loss_ns
                else:
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, vocab_size),
                        seqs[:, 1:].reshape(-1),
                        ignore_index=0,
                    )

            # Skip NaN/Inf losses (log but don't crash)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                n_nan_batches += 1
                optimizer.zero_grad()
                # Still count accuracy from valid logits
                with torch.no_grad():
                    if has_targets:
                        pred = logits[:, -2].argmax(dim=-1)
                        valid = ~torch.isnan(logits[:, -2].sum(dim=-1))
                        epoch_correct += (pred[valid] == targets[valid]).sum().item()
                        epoch_total += valid.sum().item()
                    else:
                        preds = logits[:, :-1].argmax(dim=-1)
                        tgts = seqs[:, 1:]
                        mask = (tgts != 0) & ~torch.isnan(logits[:, :-1].sum(dim=-1))
                        epoch_correct += (preds[mask] == tgts[mask]).sum().item()
                        epoch_total += mask.sum().item()
                n_batches += 1
                continue

            if args.profile and epoch <= profile_epochs:
                torch.cuda.synchronize() if device.type == 'cuda' else None
                prof['forward'] += time.perf_counter() - t0
                t0 = time.perf_counter()

            # --- Backward ---
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.w_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.w_clip)
                    epoch_grad_norm += gn.item()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.w_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.w_clip)
                    epoch_grad_norm += gn.item()
                optimizer.step()

            if args.profile and epoch <= profile_epochs:
                torch.cuda.synchronize() if device.type == 'cuda' else None
                prof['backward'] += time.perf_counter() - t0

            # --- Train accuracy ---
            with torch.no_grad():
                if has_targets:
                    pred = logits[:, -2].argmax(dim=-1)
                    epoch_correct += (pred == targets).sum().item()
                    epoch_total += targets.shape[0]
                else:
                    preds = logits[:, :-1].argmax(dim=-1)
                    tgts = seqs[:, 1:]
                    mask = tgts != 0
                    epoch_correct += (preds[mask] == tgts[mask]).sum().item()
                    epoch_total += mask.sum().item()

            epoch_loss += loss_val
            n_batches += 1

        scheduler.step()
        train_acc = epoch_correct / max(epoch_total, 1)
        valid_batches = n_batches - n_nan_batches
        avg_loss = epoch_loss / max(valid_batches, 1)
        avg_grad_norm = epoch_grad_norm / max(valid_batches, 1)
        ms_per_batch = (time.perf_counter() - t_epoch) / max(n_batches, 1) * 1000

        # --- Evaluation ---
        t_eval = time.perf_counter()
        if has_targets:
            test_acc = evaluate_last_token(model, test_loader, device, amp_ctx)
            # Phase 6: KL surprise evaluation
            if kl_tracker is not None and config.use_surprise_gate:
                kl_acc = evaluate_with_kl_surprise(
                    model, test_loader, device, amp_ctx, kl_tracker)
            else:
                kl_acc = None
        else:
            test_acc = evaluate_nextstep(model, test_loader, device, amp_ctx)
            kl_acc = None

        if args.profile and epoch <= profile_epochs:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            prof['eval'] += time.perf_counter() - t_eval

        ep_s = time.perf_counter() - t_epoch

        # --- Record history ---
        record = {
            'epoch': epoch, 'loss': avg_loss, 'train_acc': train_acc,
            'test_acc': test_acc, 'grad_norm': avg_grad_norm,
            'nan_batches': n_nan_batches,
        }
        history.append(record)

        # --- Print ---
        if epoch % args.print_every == 0 or epoch == args.epochs:
            nan_str = f" ({n_nan_batches} nan)" if n_nan_batches > 0 else ""
            if has_targets:
                kl_str = f"{kl_acc:7.4f}" if kl_acc is not None else "    ---"
                print(f"{epoch:5d}  {avg_loss:8.4f}  {train_acc:7.4f}  {test_acc:7.4f}  "
                      f"{kl_str}  {ms_per_batch:6.1f}  {ep_s:6.1f}{nan_str}")
            else:
                print(f"{epoch:5d}  {avg_loss:8.4f}  {train_acc:7.4f}  {test_acc:7.4f}  "
                      f"{ms_per_batch:6.1f}  {ep_s:6.1f}{nan_str}")

        # --- Diagnostic charts ---
        if args.diag_every > 0 and (epoch % args.diag_every == 0 or epoch == args.epochs):
            diag = collect_diagnostics(model, config, epoch, device, amp_ctx)
            save_diagnostic_charts(history, diag, diag_dir, epoch)

        # Profile breakdown after N epochs
        if args.profile and epoch == profile_epochs:
            total_t = sum(prof.values())
            print(f"\n--- Profile ({profile_epochs} epochs) ---")
            for k, v in prof.items():
                print(f"  {k:12s}: {v:7.2f}s  ({100*v/max(total_t,1e-9):5.1f}%)")
            print(f"  {'total':12s}: {total_t:7.2f}s")
            print()

    # --- Final summary ---
    print(f"\nFinal: train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
          f"loss={avg_loss:.4f}")
    if kl_acc is not None:
        print(f"  KL surprise test_acc={kl_acc:.4f}")
    print(f"Config: {feat_str}")
    total_nan = sum(h['nan_batches'] for h in history)
    if total_nan > 0:
        print(f"  WARNING: {total_nan} NaN batches across all epochs")

    result = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'kl_acc': kl_acc,
        'loss': avg_loss,
        'config': feat_str,
        'task': task_name,
        'preset': args.preset,
        'n_params': n_params,
        'epochs': args.epochs,
    }

    # Append JSON line to results file if requested
    if args.results_file:
        import json
        with open(args.results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    train(args)
