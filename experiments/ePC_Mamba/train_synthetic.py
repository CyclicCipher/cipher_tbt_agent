"""
Synthetic task training for ePC-Mamba (Phase 1).

Tasks:
  1. Copy: input [a,b,c,PAD,PAD,PAD] → output [PAD,PAD,PAD,a,b,c]
  2. Selective copy: copy only tokens marked with a flag
  3. Sequence classification: predict label from sequence content

Success criteria: >95% accuracy on the copy task proves ePC works
with Mamba's recurrent dynamics. If this fails, there's a fundamental
incompatibility to debug.

Usage:
  python experiments/ePC_Mamba/train_synthetic.py [--task copy|selective|classify]
"""

import argparse
import os
import sys
import time

# Add project root to path (MISTAKES.md #7)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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

    Args:
        n_samples: number of sequences.
        seq_len: total sequence length (must be even).
        vocab_size: number of tokens (including PAD=0).
        copy_len: how many tokens to copy (default: seq_len // 2).

    Returns:
        inputs: (n_samples, seq_len) long tensor.
        targets: (n_samples, seq_len) long tensor.
    """
    if copy_len is None:
        copy_len = seq_len // 2
    assert copy_len <= seq_len // 2, "copy_len must be <= seq_len // 2"

    # Random tokens in [1, vocab_size-1] (0 is PAD)
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

    Random tokens with markers scattered throughout. Output should
    contain only the marked tokens (in order) at fixed positions,
    PAD elsewhere.

    Uses two special tokens: PAD=0, MARKER=vocab_size-1.
    Data tokens in [1, vocab_size-2].

    Args:
        n_samples: number of sequences.
        seq_len: total sequence length.
        vocab_size: number of tokens.
        n_markers: number of tokens to mark for copying.
    """
    marker_token = vocab_size - 1

    # Random data tokens
    inputs = torch.randint(1, vocab_size - 1, (n_samples, seq_len))
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)

    # Place markers at random positions
    for i in range(n_samples):
        marker_positions = torch.randperm(seq_len)[:n_markers].sort().values
        marked_tokens = inputs[i, marker_positions].clone()
        # Set marker flag: replace the position BEFORE with marker token
        # Actually simpler: use a second channel. But for single-sequence:
        # mark by replacing with (token + marker_offset) — too complex.
        # Simplest approach: marker precedes the token to copy.
        # But this changes the input. Let's just use a paired input.
        # For Phase 1 simplicity: mark positions by adding marker_token after them.
        # Actually, let's just mark them in the input directly:
        # input has marker tokens at certain positions, output copies the
        # tokens that FOLLOW each marker.

        # Reset: all random tokens
        inputs[i] = torch.randint(1, vocab_size - 1, (seq_len,))
        # Pick positions for markers (leaving room for token after marker)
        valid_positions = torch.arange(0, seq_len - 1)
        marker_positions = valid_positions[torch.randperm(len(valid_positions))[:n_markers]].sort().values
        inputs[i, marker_positions] = marker_token
        # Targets: the tokens following each marker, packed at the end
        for j, pos in enumerate(marker_positions):
            if pos + 1 < seq_len:
                targets[i, seq_len - n_markers + j] = inputs[i, pos + 1]

    return inputs, targets


def generate_classify_data(n_samples: int, seq_len: int, vocab_size: int,
                           n_classes: int = 4
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate sequence classification data.

    Label = (sum of all tokens) mod n_classes.
    Simple but requires the model to attend to all positions.

    Args:
        n_samples: number of sequences.
        seq_len: total sequence length.
        vocab_size: number of tokens.
        n_classes: number of classes.

    Returns:
        inputs: (n_samples, seq_len) long tensor.
        targets: (n_samples,) long tensor of class indices.
    """
    inputs = torch.randint(1, vocab_size, (n_samples, seq_len))
    targets = (inputs.sum(dim=1) % n_classes).long()
    return inputs, targets


# ---------------------------------------------------------------------------
# Model wrapper for synthetic tasks
# ---------------------------------------------------------------------------

class ePCMambaSynthetic(nn.Module):
    """ePC-Mamba model for synthetic sequence tasks.

    For copy/selective: predicts tokens at each position (CE loss).
    For classify: mean-pools hidden states → linear classifier.

    Args:
        config: Mamba2Config.
        vocab_size: token vocabulary size.
        task: 'copy', 'selective', or 'classify'.
        n_classes: number of classes (for classify task).
        iters: Newton iterations.
        damping: Newton damping.
    """

    def __init__(self, config: Mamba2Config, vocab_size: int,
                 task: str = 'copy', n_classes: int = 4,
                 iters: int = 2, damping: float = 1.0):
        super().__init__()
        self.config = config
        self.task = task

        self.embedding = nn.Embedding(vocab_size, config.d_model)

        if task in ('copy', 'selective'):
            self.pce = PCESequence(
                config, iters=iters, damping=damping, output_loss='ce',
            )
            self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)
        elif task == 'classify':
            self.pce = PCESequence(
                config, iters=iters, damping=damping, output_loss='ce',
            )
            self.out_proj = nn.Linear(config.d_model, n_classes)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, input_ids: torch.Tensor,
                targets: torch.Tensor | None = None):
        """Forward pass.

        For copy/selective: targets is (batch, seqlen) token indices.
        For classify: targets is (batch,) class indices.
        """
        x = self.embedding(input_ids)

        if targets is not None:
            if self.task == 'classify':
                # For classification, we need a different output path.
                # Use mean pooling before the classifier.
                return self._forward_classify_epc(x, targets)
            else:
                return self.pce.minimize_error_energy(
                    x, targets, self.out_proj
                )
        else:
            self.pce.errors = [0.0] * (len(self.pce.layers) - 1)
            hidden = self.pce.y_pred(x)
            if self.task == 'classify':
                hidden = hidden.mean(dim=1)  # mean pool over sequence
            return self.out_proj(hidden)

    def _forward_classify_epc(self, x: torch.Tensor,
                              targets: torch.Tensor) -> float:
        """ePC inference for classification (mean-pool then classify).

        We define a custom output projection that mean-pools then
        applies the linear classifier, so it fits the PCESequence API.
        """
        # Wrap out_proj to include mean pooling
        class _PooledProj(nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.proj = proj

            def forward(self, x):
                # x: (batch, seqlen, d_model) → mean pool → (batch, d_model)
                return self.proj(x.mean(dim=1))

        pooled_proj = _PooledProj(self.out_proj)
        return self.pce.minimize_error_energy(x, targets, pooled_proj)

    def compute_weight_loss(self, input_ids: torch.Tensor,
                            targets: torch.Tensor,
                            batch_size: int) -> torch.Tensor:
        """Compute E_local for weight update."""
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

    def get_diagnostics(self) -> dict:
        return self.pce.get_diagnostics()


# ---------------------------------------------------------------------------
# Backprop baseline for comparison
# ---------------------------------------------------------------------------

class BackpropMambaBaseline(nn.Module):
    """Standard backprop Mamba model (same architecture, no ePC).

    For comparing convergence speed and final accuracy.
    """

    def __init__(self, config: Mamba2Config, vocab_size: int,
                 task: str = 'copy', n_classes: int = 4):
        super().__init__()
        self.config = config
        self.task = task

        self.embedding = nn.Embedding(vocab_size, config.d_model)

        # Pre-norm Mamba layers with residual connections
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))  # pre-norm residual
        x = self.out_norm(x)
        if self.task == 'classify':
            x = x.mean(dim=1)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                     pad_token: int = 0, task: str = 'copy') -> float:
    """Compute accuracy, ignoring PAD positions for copy tasks."""
    if task == 'classify':
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()
    else:
        # For copy tasks: only evaluate on non-PAD target positions
        preds = logits.argmax(dim=-1)
        mask = targets != pad_token
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == targets[mask]).float().mean().item()


def train_epoch(model, dataloader, optimizer, device, task, use_epc=True):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    total_time = 0.0

    for batch in dataloader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        batch_size = inputs.shape[0]
        t0 = time.perf_counter()

        if use_epc:
            # Phase 1: Inference (optimize errors)
            model(inputs, targets)

            # Phase 2: Weight update via E_local
            optimizer.zero_grad()
            weight_loss = model.compute_weight_loss(inputs, targets, batch_size)
            weight_loss.backward()
            optimizer.step()

            loss_val = weight_loss.item()

            # Get accuracy (feedforward without ePC for clean eval)
            with torch.no_grad():
                logits = model(inputs)
                acc = compute_accuracy(logits, targets, task=task)
        else:
            # Standard backprop baseline
            optimizer.zero_grad()
            logits = model(inputs)
            if task == 'classify':
                loss = F.cross_entropy(logits, targets)
            else:
                b, l, v = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(b * l, v), targets.reshape(b * l)
                )
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            with torch.no_grad():
                acc = compute_accuracy(logits, targets, task=task)

        t1 = time.perf_counter()
        total_loss += loss_val
        total_acc += acc
        n_batches += 1
        total_time += (t1 - t0)

    return {
        'loss': total_loss / n_batches,
        'accuracy': total_acc / n_batches,
        'time_ms': total_time * 1000 / n_batches,
    }


def evaluate(model, dataloader, device, task, use_epc=True):
    """Evaluate on a dataset."""
    model.eval()
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            if use_epc:
                logits = model(inputs)
            else:
                logits = model(inputs)
            acc = compute_accuracy(logits, targets, task=task)
            total_acc += acc
            n_batches += 1

    return total_acc / n_batches


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
    parser.add_argument('--iters', type=int, default=2, help='Newton iterations (T)')
    parser.add_argument('--damping', type=float, default=1.0)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--baseline', action='store_true',
                        help='Run backprop baseline instead of ePC')
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
        chunk_size=min(64, args.seq_len),  # chunk_size <= seq_len
        n_layer=args.n_layer,
    )
    print(f"Config: d_model={config.d_model}, d_inner={config.d_inner}, "
          f"nheads={config.nheads}, n_layer={config.n_layer}")

    # Generate data
    print(f"\nGenerating {args.task} task data...")
    if args.task == 'copy':
        train_x, train_y = generate_copy_data(
            args.n_train, args.seq_len, args.vocab_size
        )
        test_x, test_y = generate_copy_data(
            args.n_test, args.seq_len, args.vocab_size
        )
    elif args.task == 'selective':
        train_x, train_y = generate_selective_copy_data(
            args.n_train, args.seq_len, args.vocab_size
        )
        test_x, test_y = generate_selective_copy_data(
            args.n_test, args.seq_len, args.vocab_size
        )
    elif args.task == 'classify':
        train_x, train_y = generate_classify_data(
            args.n_train, args.seq_len, args.vocab_size
        )
        test_x, test_y = generate_classify_data(
            args.n_test, args.seq_len, args.vocab_size
        )

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
            iters=args.iters, damping=args.damping,
        ).to(device)
        print(f"Model: ePC-Mamba (T={args.iters}, damping={args.damping})")
    else:
        model = BackpropMambaBaseline(
            config, vocab_size=args.vocab_size, task=args.task,
        ).to(device)
        print("Model: Backprop-Mamba (baseline)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n{'Epoch':>5} {'Loss':>10} {'Train Acc':>10} {'Test Acc':>10} {'ms/batch':>10}")
    print("-" * 50)

    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            task=args.task, use_epc=use_epc,
        )
        test_acc = evaluate(
            model, test_loader, device,
            task=args.task, use_epc=use_epc,
        )

        best_test_acc = max(best_test_acc, test_acc)

        print(f"{epoch:5d} {train_metrics['loss']:10.4f} "
              f"{train_metrics['accuracy']:10.4f} {test_acc:10.4f} "
              f"{train_metrics['time_ms']:10.1f}")

        # Print ePC diagnostics every 10 epochs
        if use_epc and epoch % 10 == 0:
            diag = model.get_diagnostics()
            print(f"  ePC: E_init={diag['E_initial']:.2f}, "
                  f"E_final={diag['E_final']:.2f}, "
                  f"convergence={diag['convergence']:.2f}, "
                  f"iters={diag['iters_used']}")
            if diag['error_norms']:
                norms_str = ', '.join(f'{n:.4f}' for n in diag['error_norms'])
                print(f"  Error norms: [{norms_str}]")

        # Early success
        if best_test_acc >= 0.95 and epoch >= 5:
            print(f"\nSuccess! Test accuracy {best_test_acc:.4f} >= 95% at epoch {epoch}")
            break

    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    if best_test_acc >= 0.95:
        print("PASS: ePC-Mamba works with SSM blocks!")
    else:
        print("FAIL: Did not reach 95% accuracy. Debug needed.")
        print("Check: error norms, convergence, per-position accuracy")


if __name__ == '__main__':
    main()
