"""
State Space Attention — WikiText-2 character-level experiment.

PURPOSE
-------
Not to beat Mamba3 on BPC (we don't expect a meaningful difference
without extensive training). The goal is to test whether the K workspace
slots develop specialisation — different slots active for different
character classes, structural positions, and boundary types.

The slot diagnostics ARE the primary result. BPC is a sanity check
that the model is learning something.

MODES
-----
  python train_wikitext2.py            # SSA (with slot attention)
  python train_wikitext2.py --baseline # ablation: no slot attention

MEMORY (4 GB VRAM budget)
-------------------------
  - fp16 mixed precision  (halves activation footprint)
  - Gradient checkpointing (trades one extra forward per block for
    not storing activations; ~1.4-1.6x slower, saves ~n_layer x savings)
  - seq_len=512 (must be multiple of chunk_size=64)
  - batch_size=2, grad_accum=4  (effective batch = 8)
  - GradScaler for fp16 loss-scale stability
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from slot_workspace import SlotConfig, SlotLM


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_path: str = "../language/data/wikitext2_train.txt"
    val_fraction: float = 0.1      # fraction of text held out for validation
    seq_len: int = 512             # characters per training example
                                   # MUST be a multiple of chunk_size (64)

    # Model (SSA)
    d_model: int = 128
    n_slots: int = 8              # K workspace slots  (headdim = 256/8 = 32)
    d_state: int = 32
    n_layer: int = 4
    intra_attn_heads: int = 4     # 0 = baseline / ablation mode

    # Training
    batch_size: int = 16          # 4 GB VRAM → 1.2 GB used with batch=2, so batch=16 is safe
    grad_accum: int = 1           # no accumulation needed with larger batches
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 20
    warmup_steps: int = 300       # linear LR warmup
    grad_clip: float = 1.0

    # Memory
    use_amp: bool = True          # fp16 mixed precision
    use_grad_ckpt: bool = False   # off: 4 GB is enough; checkpointing was adding overhead

    # Data loading
    num_workers: int = 2          # parallel data loading workers
    prefetch_factor: int = 2      # batches to prefetch per worker
    persistent_workers: bool = True  # keep workers alive between epochs

    # Diagnostics
    diag_every_n_epochs: int = 5  # run slot diagnostics every N epochs
    diag_n_chars: int = 20_000    # characters to analyse for diagnostics
    plot_dir: str = "plots"       # directory for saved figures

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    save_every_n_epochs: int = 5

    # Misc
    seed: int = 42
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    @property
    def chunk_size(self) -> int:
        return 64   # must match SlotConfig / SSD

    def __post_init__(self):
        assert self.seq_len % self.chunk_size == 0, (
            f"seq_len={self.seq_len} must be divisible by chunk_size={self.chunk_size}"
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_vocab(text: str) -> dict[str, int]:
    """Character vocabulary from training text, sorted for reproducibility."""
    chars = sorted(set(text))
    vocab = {c: i + 1 for i, c in enumerate(chars)}  # 0 = padding / unknown
    vocab['<unk>'] = 0
    return vocab


class CharDataset(Dataset):
    """Non-overlapping fixed-length character windows."""

    def __init__(self, text: str, seq_len: int, vocab: dict[str, int]):
        self.seq_len = seq_len
        tokens = [vocab.get(c, 0) for c in text]
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        # Each example: input [i .. i+seq_len), target [i+1 .. i+seq_len+1)
        self.n = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        s = idx * self.seq_len
        return self.tokens[s:s + self.seq_len], self.tokens[s + 1:s + self.seq_len + 1]


def load_data(cfg: TrainConfig) -> tuple[str, str]:
    """Read raw text, split into train and val by article boundaries."""
    path = Path(__file__).parent / cfg.data_path
    text = path.read_text(encoding='utf-8')

    # Split at article boundaries (lines beginning with ' =') to avoid
    # cutting an article mid-sentence.
    lines = text.splitlines(keepends=True)
    # Find article-start lines (WikiText-2 format: ' = Title = \n')
    splits = [i for i, l in enumerate(lines) if l.startswith(' = ') and l.strip().endswith('=')]

    # Put the last val_fraction of articles in the val set
    if len(splits) > 1:
        split_idx = splits[max(1, int(len(splits) * (1 - cfg.val_fraction)))]
        train_text = ''.join(lines[:split_idx])
        val_text   = ''.join(lines[split_idx:])
    else:
        # Fallback: split by character count
        n = int(len(text) * (1 - cfg.val_fraction))
        train_text, val_text = text[:n], text[n:]

    return train_text, val_text


# ---------------------------------------------------------------------------
# Learning rate schedule (cosine with linear warmup)
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: SlotLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    cfg: TrainConfig,
    epoch: int,
) -> float:
    model.train()
    device = cfg.device
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # Mixed precision forward
        with torch.amp.autocast(device_type=device, dtype=torch.float16,
                                enabled=cfg.use_amp):
            logits = model(x)                       # (B, T, vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=0,                     # skip padding / unknown
            ) / cfg.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only advance LR schedule when optimizer actually ran.
            # scaler.step() is a no-op (skips optimizer) when it detects
            # inf/nan gradients, which is common at the first fp16 step.
            # Calling scheduler.step() in that case triggers PyTorch's
            # "step before optimizer" warning and skips the first LR value.
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad()

        total_loss   += loss.item() * cfg.grad_accum
        total_tokens += (y != 0).sum().item()

    # Handle trailing partial accumulation
    if (step + 1) % cfg.grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        optimizer.zero_grad()

    avg_nll = total_loss / len(loader)
    bpc = avg_nll / math.log(2)
    return bpc


@torch.no_grad()
def eval_epoch(
    model: SlotLM,
    loader: DataLoader,
    cfg: TrainConfig,
) -> float:
    model.eval()
    device = cfg.device
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device, dtype=torch.float16,
                                enabled=cfg.use_amp):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=0,
            )
        total_loss += loss.item()

    bpc = (total_loss / len(loader)) / math.log(2)
    return bpc


# ---------------------------------------------------------------------------
# Slot diagnostics
# ---------------------------------------------------------------------------

def collect_slot_activations(
    model: SlotLM,
    text: str,
    vocab: dict[str, int],
    cfg: TrainConfig,
) -> tuple[list[Tensor], list[str]] | None:
    """
    Collect IntraSlotAttention outputs for cfg.diag_n_chars characters.

    Returns:
        layer_acts: list of (N, K, headdim) tensors, one per model layer
                    N = number of characters processed
        chars:      list of N characters (aligned to layer_acts positions)

    Returns None if model has no slot attention (baseline mode).
    """
    device = cfg.device
    seq_len = cfg.seq_len

    # Check if slot attention exists
    has_slot_attn = any(
        layer.mixer.slot_attn is not None for layer in model.layers
    )
    if not has_slot_attn:
        return None

    # Register hooks on every IntraSlotAttention in every layer
    # Hook captures output: (batch=1, seqlen, K, headdim)
    n_layers = len(model.layers)
    layer_buffers: list[list[Tensor]] = [[] for _ in range(n_layers)]

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            # output: (1, seqlen, K, headdim) — squeeze batch dim
            layer_buffers[layer_idx].append(output[0].detach().cpu())
        return hook

    hooks = []
    for i, layer in enumerate(model.layers):
        if layer.mixer.slot_attn is not None:
            hooks.append(
                layer.mixer.slot_attn.register_forward_hook(make_hook(i))
            )

    # Tokenise text and run in non-overlapping windows
    n_chars = min(cfg.diag_n_chars, len(text) - seq_len - 1)
    tokens  = [vocab.get(c, 0) for c in text]

    model.eval()
    with torch.no_grad():
        for start in range(0, n_chars, seq_len):
            end = start + seq_len
            if end + 1 > len(tokens):
                break
            x = torch.tensor(
                tokens[start:end], dtype=torch.long
            ).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device, dtype=torch.float16,
                                    enabled=cfg.use_amp):
                model(x)

    for h in hooks:
        h.remove()

    # Concatenate along sequence dimension: (N, K, headdim) per layer
    layer_acts = []
    for buf in layer_buffers:
        if buf:
            layer_acts.append(torch.cat(buf, dim=0).float())  # (N, K, D)
        else:
            layer_acts.append(None)

    aligned_chars = list(text[:len(layer_acts[0])])
    return layer_acts, aligned_chars


def run_diagnostics(
    model: SlotLM,
    val_text: str,
    vocab: dict[str, int],
    cfg: TrainConfig,
    epoch: int,
) -> None:
    """Compute and save slot specialisation diagnostics."""
    print(f"\n  [diag] collecting slot activations ({cfg.diag_n_chars} chars)...")
    result = collect_slot_activations(model, val_text, vocab, cfg)

    if result is None:
        print("  [diag] baseline mode — no slot attention, skipping diagnostics.")
        return

    layer_acts, chars = result
    plot_dir = Path(__file__).parent / cfg.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Use the last layer's activations (most refined representations)
    acts = layer_acts[-1]   # (N, K, headdim)
    K    = acts.shape[1]
    N    = acts.shape[0]

    # Slot norms: (N, K) — L2 norm of each slot vector at each position
    slot_norms = acts.norm(dim=-1)   # (N, K)

    # ── 1. Character-class specialisation ─────────────────────────────────
    char_classes = {
        'vowel':     set('aeiouAEIOU'),
        'consonant': set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'),
        'space':     {' ', '\t'},
        'newline':   {'\n'},
        'digit':     set('0123456789'),
        'punct':     set('.,;:!?\'"()-[]{}'),
        'upper':     set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
    }

    # Mean slot norm per character class: (K, n_classes)
    class_names = list(char_classes.keys())
    class_means = torch.zeros(K, len(class_names))
    for ci, (name, char_set) in enumerate(char_classes.items()):
        mask = torch.tensor([c in char_set for c in chars], dtype=torch.bool)
        if mask.any():
            class_means[:, ci] = slot_norms[mask].mean(dim=0)

    _save_heatmap(
        class_means.numpy(),
        row_labels=[f"slot {k}" for k in range(K)],
        col_labels=class_names,
        title=f"Slot activation norm by character class (epoch {epoch}, last layer)",
        xlabel="Character class",
        ylabel="Slot",
        path=plot_dir / f"slot_class_epoch{epoch:03d}.png",
    )

    # ── 2. Word-boundary specialisation ───────────────────────────────────
    # Is position immediately after a space? (word-start heuristic)
    at_word_start = torch.tensor(
        [i > 0 and chars[i-1] == ' ' for i in range(N)], dtype=torch.bool
    )
    at_mid_word = torch.tensor(
        [
            i > 0 and chars[i-1] not in {' ', '\n'} and chars[i] not in {' ', '\n'}
            for i in range(N)
        ],
        dtype=torch.bool,
    )

    boundary_means = torch.stack([
        slot_norms[at_word_start].mean(dim=0) if at_word_start.any() else torch.zeros(K),
        slot_norms[at_mid_word].mean(dim=0)   if at_mid_word.any()   else torch.zeros(K),
    ], dim=1)   # (K, 2)

    _save_heatmap(
        boundary_means.numpy(),
        row_labels=[f"slot {k}" for k in range(K)],
        col_labels=['word-start', 'mid-word'],
        title=f"Slot activation at word boundaries (epoch {epoch}, last layer)",
        xlabel="Position type",
        ylabel="Slot",
        path=plot_dir / f"slot_boundary_epoch{epoch:03d}.png",
    )

    # ── 3. Slot correlation matrix ─────────────────────────────────────────
    # Pairwise Pearson correlation between slot norm time-series.
    # Anti-correlated slots = competing (good specialisation).
    # Highly correlated slots = redundant.
    corr = _pearson_corr(slot_norms.T)   # (K, K)

    _save_heatmap(
        corr.numpy(),
        row_labels=[f"slot {k}" for k in range(K)],
        col_labels=[f"slot {k}" for k in range(K)],
        title=f"Slot correlation matrix (epoch {epoch}, last layer)",
        xlabel="Slot",
        ylabel="Slot",
        vmin=-1.0, vmax=1.0,
        cmap='RdBu_r',
        path=plot_dir / f"slot_corr_epoch{epoch:03d}.png",
    )

    # ── 4. Slot persistence (volatility) ──────────────────────────────────
    # delta[k] = mean |norm(slot_k, t) - norm(slot_k, t-1)|
    # Low  delta → slot holds stable information (entity tracking)
    # High delta → slot rapidly updated (local feature detector)
    delta = (slot_norms[1:] - slot_norms[:-1]).abs().mean(dim=0)   # (K,)
    _save_bar(
        delta.numpy(),
        labels=[f"slot {k}" for k in range(K)],
        title=f"Slot persistence (lower = more stable, epoch {epoch})",
        ylabel="Mean |delta norm|",
        path=plot_dir / f"slot_persistence_epoch{epoch:03d}.png",
    )

    # ── Print summary ──────────────────────────────────────────────────────
    most_stable   = delta.argmin().item()
    most_volatile = delta.argmax().item()
    top_vowel     = class_means[:, class_names.index('vowel')].argmax().item()
    top_space     = class_means[:, class_names.index('space')].argmax().item()
    top_boundary  = (boundary_means[:, 0] - boundary_means[:, 1]).argmax().item()

    print(f"  [diag] most stable slot:        slot {most_stable}"
          f"  (delta={delta[most_stable]:.4f})")
    print(f"  [diag] most volatile slot:      slot {most_volatile}"
          f"  (delta={delta[most_volatile]:.4f})")
    print(f"  [diag] most active for vowels:  slot {top_vowel}")
    print(f"  [diag] most active for spaces:  slot {top_space}")
    print(f"  [diag] most boundary-sensitive: slot {top_boundary}")
    print(f"  [diag] plots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# Plot helpers (headless matplotlib)
# ---------------------------------------------------------------------------

def _save_heatmap(
    data,
    row_labels, col_labels, title,
    xlabel='', ylabel='',
    vmin=None, vmax=None,
    cmap='viridis',
    path: Path = None,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4, len(col_labels)), max(3, len(row_labels) * 0.7)))
    im = ax.imshow(data, aspect='auto', cmap=cmap,
                   vmin=vmin if vmin is not None else data.min(),
                   vmax=vmax if vmax is not None else data.max())
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def _save_bar(data, labels, title, ylabel, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4, len(labels)), 3))
    ax.bar(range(len(data)), data)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def _pearson_corr(x: Tensor) -> Tensor:
    """Pearson correlation matrix for rows of x: (K, N) -> (K, K)."""
    x = x - x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
    x = x / std
    return (x @ x.T) / x.shape[1]


# ---------------------------------------------------------------------------
# BPC log (appended to a CSV for easy plotting)
# ---------------------------------------------------------------------------

def log_bpc(path: Path, epoch: int, train_bpc: float, val_bpc: float,
            mode: str) -> None:
    write_header = not path.exists()
    with open(path, 'a') as f:
        if write_header:
            f.write("mode,epoch,train_bpc,val_bpc\n")
        f.write(f"{mode},{epoch},{train_bpc:.5f},{val_bpc:.5f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step-time profiler
# ---------------------------------------------------------------------------

@torch.no_grad()
def _warmup(model, loader, device, use_amp, n=5):
    """Run a few forward passes to let CUDA JIT and cuDNN autotune settle."""
    model.eval()
    it = iter(loader)
    for _ in range(n):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device, dtype=torch.float16,
                                enabled=use_amp):
            model(x)
    if device == 'cuda':
        torch.cuda.synchronize()


def profile_throughput(model, loader, cfg, n_steps: int = 60) -> None:
    """Measure and report time breakdown: data loading vs. GPU compute.

    Uses torch.cuda.synchronize() around each phase to get wall-clock GPU
    times, not just CPU-side scheduling times.

    Reports:
      - ms per step for each phase
      - estimated GPU utilisation
      - chars/sec throughput
      - recommendation if data loading is the bottleneck
    """
    device = cfg.device
    use_amp = cfg.use_amp
    n_warmup = 10

    print(f"\nProfiling {n_steps} steps "
          f"(batch={cfg.batch_size}, seq={cfg.seq_len}, "
          f"num_workers={cfg.num_workers})...")

    # Temporary optimizer for backward timing (SGD = no moment buffers to init)
    tmp_opt = torch.optim.SGD(model.parameters(), lr=0.0)

    _warmup(model, loader, device, use_amp, n=n_warmup)
    model.train()

    times_data: list[float] = []
    times_h2d:  list[float] = []
    times_fwd:  list[float] = []
    times_bwd:  list[float] = []

    loader_it = iter(loader)

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    for i in range(n_steps + n_warmup):
        # ── Data loading (CPU) ────────────────────────────────────────────
        sync()
        t0 = time.perf_counter()
        try:
            x, y = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            x, y = next(loader_it)
        t_data = time.perf_counter() - t0

        # ── Host→Device transfer ──────────────────────────────────────────
        sync()
        t0 = time.perf_counter()
        x = x.to(device, non_blocking=False)   # blocking so timing is clean
        y = y.to(device, non_blocking=False)
        sync()
        t_h2d = time.perf_counter() - t0

        # ── Forward ───────────────────────────────────────────────────────
        sync()
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type=device, dtype=torch.float16,
                                enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=0,
            )
        sync()
        t_fwd = time.perf_counter() - t0

        # ── Backward ──────────────────────────────────────────────────────
        sync()
        t0 = time.perf_counter()
        loss.backward()
        sync()
        t_bwd = time.perf_counter() - t0
        tmp_opt.zero_grad()

        if i >= n_warmup:
            times_data.append(t_data * 1e3)
            times_h2d.append(t_h2d  * 1e3)
            times_fwd.append(t_fwd  * 1e3)
            times_bwd.append(t_bwd  * 1e3)

    def ms(lst):
        return sum(lst) / len(lst)

    d  = ms(times_data)
    h  = ms(times_h2d)
    f  = ms(times_fwd)
    b  = ms(times_bwd)
    total = d + h + f + b
    gpu   = f + b
    chars_per_sec = (cfg.batch_size * cfg.seq_len) / (total / 1e3)

    print(f"\n  {'Phase':<20} {'ms/step':>8}  {'%':>5}")
    print(f"  {'-'*38}")
    print(f"  {'Data loading (CPU)':<20} {d:8.2f}  {100*d/total:5.1f}%")
    print(f"  {'H2D transfer':<20} {h:8.2f}  {100*h/total:5.1f}%")
    print(f"  {'Forward':<20} {f:8.2f}  {100*f/total:5.1f}%")
    print(f"  {'Backward':<20} {b:8.2f}  {100*b/total:5.1f}%")
    print(f"  {'-'*38}")
    print(f"  {'Total':<20} {total:8.2f}")
    print(f"\n  GPU utilisation (est.): {100*gpu/total:.0f}%")
    print(f"  Throughput:             {chars_per_sec:,.0f} chars/sec")

    if device == 'cuda':
        alloc  = torch.cuda.memory_allocated()  / 1024**3
        reserv = torch.cuda.memory_reserved()   / 1024**3
        print(f"  VRAM allocated:         {alloc:.2f} GB  "
              f"(reserved: {reserv:.2f} GB)")

    print()
    if d / total > 0.15:
        print(f"  BOTTLENECK: data loading is {100*d/total:.0f}% of step time.")
        print(f"  Fix: increase --num_workers (currently {cfg.num_workers})")
    elif gpu / total > 0.90:
        print(f"  GPU-bound (good). To go faster: larger model or longer sequences.")
    else:
        print(f"  Mixed bottleneck. Check H2D transfer and data loading together.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SSA WikiText-2 experiment")
    parser.add_argument('--baseline', action='store_true',
                        help='Ablation: disable slot attention (intra_attn_heads=0)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_ckpt', action='store_true',
                        help='Disable gradient checkpointing')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader worker processes (default: cfg.num_workers=2)')
    parser.add_argument('--profile', type=int, default=None, metavar='N',
                        help='Profile throughput for N steps then exit (e.g. --profile 60)')
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.baseline:
        cfg.intra_attn_heads = 0
    if args.epochs      is not None: cfg.max_epochs   = args.epochs
    if args.lr          is not None: cfg.lr            = args.lr
    if args.batch       is not None: cfg.batch_size    = args.batch
    if args.seq_len     is not None: cfg.seq_len       = args.seq_len
    if args.num_workers is not None: cfg.num_workers   = args.num_workers
    if args.no_amp:                  cfg.use_amp        = False
    if args.no_ckpt:                 cfg.use_grad_ckpt  = False

    mode = 'baseline' if cfg.intra_attn_heads == 0 else 'ssa'
    torch.manual_seed(cfg.seed)

    # cuDNN autotuner — picks fastest conv/matmul algorithm for fixed input shapes.
    # Safe because all batches have the same (batch, seq_len) shape.
    if cfg.device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    train_text, val_text = load_data(cfg)
    vocab = build_vocab(train_text)
    vocab_size = max(vocab.values()) + 1
    print(f"  train: {len(train_text):,} chars | "
          f"val: {len(val_text):,} chars | "
          f"vocab: {vocab_size} chars")

    train_ds = CharDataset(train_text, cfg.seq_len, vocab)
    val_ds   = CharDataset(val_text,   cfg.seq_len, vocab)
    nw = cfg.num_workers
    pf = cfg.prefetch_factor if nw > 0 else None
    pw = cfg.persistent_workers and nw > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        pin_memory=(cfg.device == 'cuda'), num_workers=nw,
        prefetch_factor=pf, persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=(cfg.device == 'cuda'), num_workers=nw,
        prefetch_factor=pf, persistent_workers=pw,
    )
    print(f"  train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    slot_cfg = SlotConfig(
        d_model          = cfg.d_model,
        n_slots          = cfg.n_slots,
        d_state          = cfg.d_state,
        n_layer          = cfg.n_layer,
        expand           = 2,
        chunk_size       = cfg.chunk_size,
        intra_attn_heads = cfg.intra_attn_heads,
        use_pope         = True,
        stable_ssm       = True,
        use_scroll_back  = False,
    )
    model = SlotLM(slot_cfg, vocab_size).to(cfg.device)

    if cfg.use_grad_ckpt:
        model.enable_gradient_checkpointing()

    counts = model.parameter_count()
    print(f"\nModel [{mode}]")
    print(f"  slots: {cfg.n_slots} x headdim={slot_cfg.headdim}  |  "
          f"d_state={cfg.d_state}  |  layers={cfg.n_layer}")
    print(f"  total params:    {counts['total']:,}")
    print(f"  slot_attention:  {counts['slot_attention']:,}")
    print(f"  grad checkpt:    {cfg.use_grad_ckpt}")
    print(f"  mixed precision: {cfg.use_amp}")
    print(f"  device:          {cfg.device}")

    # ── Profile mode (measure throughput, then exit) ──────────────────────
    if args.profile is not None:
        profile_throughput(model, train_loader, cfg, n_steps=args.profile)
        return

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = (len(train_loader) // cfg.grad_accum) * cfg.max_epochs
    scheduler   = make_scheduler(optimizer, cfg.warmup_steps, total_steps)
    scaler      = torch.amp.GradScaler(cfg.device, enabled=cfg.use_amp)

    # ── Logging setup ─────────────────────────────────────────────────────
    ckpt_dir = Path(__file__).parent / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(__file__).parent / f"bpc_log_{mode}.csv"

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nTraining for {cfg.max_epochs} epochs...")
    best_val_bpc = float('inf')

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        train_bpc = train_epoch(model, train_loader, optimizer, scheduler,
                                scaler, cfg, epoch)
        val_bpc   = eval_epoch(model, val_loader, cfg)
        elapsed   = time.time() - t0
        lr_now    = scheduler.get_last_lr()[0]

        print(f"epoch {epoch:3d}/{cfg.max_epochs}  "
              f"train_bpc={train_bpc:.4f}  val_bpc={val_bpc:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.1f}s")

        log_bpc(log_path, epoch, train_bpc, val_bpc, mode)

        # Checkpoint
        if val_bpc < best_val_bpc:
            best_val_bpc = val_bpc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'cfg': slot_cfg,
                'vocab': vocab,
                'val_bpc': val_bpc,
            }, ckpt_dir / f"best_{mode}.pt")

        if epoch % cfg.save_every_n_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'cfg': slot_cfg,
                'vocab': vocab,
                'val_bpc': val_bpc,
            }, ckpt_dir / f"{mode}_epoch{epoch:03d}.pt")

        # Slot diagnostics
        if epoch % cfg.diag_every_n_epochs == 0:
            run_diagnostics(model, val_text, vocab, cfg, epoch)

        # VRAM cleanup between epochs
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nDone. Best val BPC: {best_val_bpc:.4f}")
    print(f"BPC log: {log_path}")

    # Final diagnostics
    print("\nRunning final diagnostics...")
    run_diagnostics(model, val_text, vocab, cfg, epoch)


if __name__ == "__main__":
    main()
