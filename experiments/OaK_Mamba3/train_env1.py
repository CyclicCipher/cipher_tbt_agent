"""
train_env1.py — MDLM training on Environment 1 (ARC-1 Analog).

Trains OaK-Mamba3 (BiMamba + MDLM objective) to infer and apply
transformation rules from K example (input, output) grid pairs, then
predict the masked test output.

Usage:
    python experiments/OaK_Mamba3/train_env1.py [--difficulty 2] [--steps 2000]

Objective (MDLM):
    At each training step a random fraction of test output tokens are
    replaced with MASK_TOKEN (13). The model sees the masked sequence and
    must predict the original token at every masked position.

    Example output tokens are independently given a 30% chance of having
    20% of their tokens also masked (prevents over-reliance on example
    outputs).

    Input tokens and SEP/QUERY tokens are never masked.

Loss:
    task_loss  : CE on masked-position logits vs original tokens (primary)
    gvf_loss_0 : MSE of GVF-0 predictions vs per-token CE errors at
                 masked positions (auxiliary)

Inference (iterative unmasking):
    Start with all test output positions masked, run the model, unmask the
    K most confident positions, repeat for n_steps=10 rounds until every
    position is filled.

Evaluation:
    Exact-match accuracy (all H*W cells correct) using iterative unmasking.
    Also reports per-cell accuracy.
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'Mamba3'))

from oak_model import (
    OaKConfig, OaKModel, OaKOutput,
    SEP_TOKEN, QUERY_TOKEN, PAD_TOKEN, MASK_TOKEN, NUM_COLORS,
)
from env1 import sample_episode


# ---------------------------------------------------------------------------
# Episode encoding (no masking — masking applied separately in make_batch)
# ---------------------------------------------------------------------------

def encode_episode(
    episode: Dict,
    H: int,
    W: int,
) -> Tuple[List[int], List[Tuple[int, int, int, int]], int]:
    """Encode one episode into a flat token sequence.

    Sequence format:
        [SEP] [input_1 tokens] [SEP] [output_1 tokens]
        [SEP] [input_2 tokens] [SEP] [output_2 tokens]
        ...
        [SEP] [test_input tokens] [QUERY] [test_output tokens]

    No masking is applied here — masking is handled in apply_masking /
    make_batch so that the original tokens remain available as labels.

    Returns:
        tokens:        list of int token ids (unmasked originals)
        grid_segments: list of (start_idx, H, W, seg_id)
                       seg_id: 0=input, 1=output, 2=test-output
        label_start:   index of the first test output token in tokens
    """
    tokens: List[int] = []
    grid_segments: List[Tuple[int, int, int, int]] = []

    # Example pairs
    for ig, og in zip(episode['input_grids'], episode['output_grids']):
        tokens.append(SEP_TOKEN)
        start = len(tokens)
        grid_segments.append((start, H, W, 0))          # input grid
        tokens.extend(ig.flatten().tolist())

        tokens.append(SEP_TOKEN)
        start = len(tokens)
        grid_segments.append((start, H, W, 1))          # output grid
        tokens.extend(og.flatten().tolist())

    # Test input
    tokens.append(SEP_TOKEN)
    start = len(tokens)
    grid_segments.append((start, H, W, 0))              # test input (seg 0)
    tokens.extend(episode['test_input'].flatten().tolist())

    # QUERY + test output
    tokens.append(QUERY_TOKEN)
    label_start = len(tokens)
    grid_segments.append((label_start, H, W, 2))        # test output (seg 2)
    tokens.extend(episode['test_output'].flatten().tolist())

    return tokens, grid_segments, label_start


# ---------------------------------------------------------------------------
# MDLM masking
# ---------------------------------------------------------------------------

def apply_masking(
    tokens: List[int],
    grid_segments: List[Tuple[int, int, int, int]],
    label_start: int,
    HW: int,
    rng: np.random.Generator,
    test_mask_rate: Optional[float] = None,
    ex_mask_prob: float = 0.3,
    ex_mask_rate: float = 0.2,
) -> Tuple[List[int], np.ndarray]:
    """Apply MDLM masking to a token sequence.

    Rules:
      - Test output positions (label_start .. label_start+HW-1):
            mask_rate ~ Uniform(0.5, 1.0) fraction are masked.
      - Example output grid positions (seg_id == 1):
            each segment independently has ex_mask_prob chance of masking
            ex_mask_rate fraction of its tokens.
      - Input grid positions (seg_id == 0), SEP, QUERY: never masked.

    Returns:
        masked_tokens: copy of tokens with MASK_TOKEN at masked positions
        mask_array:    bool ndarray of length len(tokens);
                       True = position is masked (needs to be predicted)
    """
    T = len(tokens)
    masked = list(tokens)
    mask_array = np.zeros(T, dtype=bool)

    # Determine test mask rate
    if test_mask_rate is None:
        test_mask_rate = float(rng.uniform(0.5, 1.0))

    # Mask test output positions
    test_end = label_start + HW
    test_indices = np.arange(label_start, min(test_end, T))
    n_test_mask = max(1, int(round(len(test_indices) * test_mask_rate)))
    chosen_test = rng.choice(test_indices, size=n_test_mask, replace=False)
    for idx in chosen_test:
        masked[idx] = MASK_TOKEN
        mask_array[idx] = True

    # Optionally mask some example output positions (seg_id == 1)
    for start, h, w, seg_id in grid_segments:
        if seg_id != 1:
            continue
        if rng.random() < ex_mask_prob:
            seg_indices = np.arange(start, start + h * w)
            n_mask = max(1, int(round(len(seg_indices) * ex_mask_rate)))
            chosen_ex = rng.choice(seg_indices, size=n_mask, replace=False)
            for idx in chosen_ex:
                if idx < T:
                    masked[idx] = MASK_TOKEN
                    mask_array[idx] = True

    return masked, mask_array


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def make_batch(
    H: int,
    W: int,
    K: int,
    difficulty: int,
    batch_size: int,
    chunk_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Dict:
    """Sample a batch of episodes, apply MDLM masking, and collate into tensors.

    All episodes share the same H, W, K so that grid_segments is identical
    across the batch.

    Batch dict keys:
        tokens:        (B, T) masked token ids (MASK_TOKEN at masked positions)
        tokens_orig:   (B, T) original unmasked token ids (label source)
        mask:          (B, T) bool tensor — True at positions to predict
        grid_segments: list of (start, H, W, seg_id) shared across batch
        label_starts:  list[int] (same value for all episodes in batch)
        labels:        (B, HW) original test output token ids
        H, W, K:       int
    """
    HW = H * W

    raw_tokens_masked: List[List[int]] = []
    raw_tokens_orig:   List[List[int]] = []
    mask_arrays:       List[np.ndarray] = []
    label_starts:      List[int] = []
    all_labels:        List[List[int]] = []
    grid_segments_shared = None

    for _ in range(batch_size):
        ep = sample_episode(H, W, K, difficulty, rng)
        tokens, grid_segments, label_start = encode_episode(ep, H, W)

        masked_tokens, mask_array = apply_masking(
            tokens, grid_segments, label_start, HW, rng
        )

        raw_tokens_masked.append(masked_tokens)
        raw_tokens_orig.append(tokens)
        mask_arrays.append(mask_array)
        label_starts.append(label_start)
        all_labels.append(ep['test_output'].flatten().tolist())

        if grid_segments_shared is None:
            grid_segments_shared = grid_segments  # identical for all episodes

    # Pad to multiple of chunk_size using PAD_TOKEN (not MASK_TOKEN)
    max_T = max(len(t) for t in raw_tokens_masked)
    T = ((max_T + chunk_size - 1) // chunk_size) * chunk_size

    padded_masked = [t + [PAD_TOKEN] * (T - len(t)) for t in raw_tokens_masked]
    padded_orig   = [t + [PAD_TOKEN] * (T - len(t)) for t in raw_tokens_orig]
    padded_masks  = [
        np.concatenate([m, np.zeros(T - len(m), dtype=bool)])
        for m in mask_arrays
    ]

    tokens_t      = torch.tensor(padded_masked, dtype=torch.long,  device=device)
    tokens_orig_t = torch.tensor(padded_orig,   dtype=torch.long,  device=device)
    mask_t        = torch.tensor(np.stack(padded_masks), dtype=torch.bool, device=device)
    labels_t      = torch.tensor(all_labels, dtype=torch.long, device=device)

    return {
        'tokens':        tokens_t,           # (B, T)
        'tokens_orig':   tokens_orig_t,      # (B, T)
        'mask':          mask_t,             # (B, T) bool
        'grid_segments': grid_segments_shared,
        'label_starts':  label_starts,
        'labels':        labels_t,           # (B, HW)
        'H': H, 'W': W, 'K': K,
    }


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_losses(
    outputs: OaKOutput,
    batch: Dict,
    lambda_gvf: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute MDLM task loss and GVF-0 auxiliary loss.

    task_loss:
        CE between model predictions at masked positions and the original
        token ids. Only masked positions (batch['mask'] == True) contribute.
        If no positions are masked (shouldn't happen in practice), returns
        zero task loss.

    gvf_loss_0:
        MSE of GVF-0 scalar predictions vs per-token CE errors at the
        same masked positions.

    Returns:
        (total_loss, task_loss, gvf_loss)
    """
    mask       = batch['mask']             # (B, T) bool
    orig       = batch['tokens_orig']      # (B, T)
    task_logits = outputs.task_logits      # (B, T, NUM_COLORS)
    gvf0_preds  = outputs.gvf_vals[..., 0]  # (B, T)

    # Select masked positions
    # flat indices where mask is True
    flat_mask    = mask.reshape(-1)            # (B*T,)
    flat_logits  = task_logits.reshape(-1, NUM_COLORS)  # (B*T, C)
    flat_orig    = orig.reshape(-1)            # (B*T,)
    flat_gvf0    = gvf0_preds.reshape(-1)     # (B*T,)

    masked_logits = flat_logits[flat_mask]    # (M, C)
    masked_labels = flat_orig[flat_mask]      # (M,)
    masked_gvf0   = flat_gvf0[flat_mask]     # (M,)

    if masked_logits.shape[0] == 0:
        zero = task_logits.new_tensor(0.0)
        return zero, zero, zero

    task_loss = F.cross_entropy(masked_logits, masked_labels)

    # GVF-0 targets: per-token CE errors at masked positions (no gradient)
    with torch.no_grad():
        per_tok_ce = F.cross_entropy(
            masked_logits,
            masked_labels,
            reduction='none',
        )   # (M,)

    gvf_loss   = F.mse_loss(masked_gvf0, per_tok_ce)
    total_loss = task_loss + lambda_gvf * gvf_loss
    return total_loss, task_loss, gvf_loss


# ---------------------------------------------------------------------------
# Iterative unmasking inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def iterative_unmask(
    model: OaKModel,
    tokens_masked: Tensor,
    tokens_orig: Tensor,
    grid_segments: List,
    label_start: int,
    HW: int,
    n_steps: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, float]:
    """Iterative unmasking inference for one episode.

    Starts from a fully-masked test output (ignoring how training masked it),
    then in each round unmasks the top-K most confident positions until all
    positions are filled.

    Args:
        tokens_masked:  (1, T) or (T,) masked token sequence (unused except
                        for its non-test-output tokens which are left alone)
        tokens_orig:    (1, T) or (T,) original tokens (ground truth for EM)
        grid_segments:  list of (start, H, W, seg_id)
        label_start:    index of first test output token
        HW:             H * W (number of test output cells)
        n_steps:        number of unmasking rounds
        device:         torch device

    Returns:
        predictions: (HW,) int tensor of predicted color tokens
        exact_match: 1.0 if all cells correct, 0.0 otherwise
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Ensure batch dimension
    if tokens_masked.dim() == 1:
        tokens_masked = tokens_masked.unsqueeze(0)
    if tokens_orig.dim() == 1:
        tokens_orig = tokens_orig.unsqueeze(0)

    tokens_masked = tokens_masked.to(device)
    tokens_orig   = tokens_orig.to(device)

    T = tokens_masked.shape[1]

    # Build fully-masked version: copy the input but replace test output with MASK
    current = tokens_masked.clone()
    test_end = label_start + HW
    current[0, label_start:min(test_end, T)] = MASK_TOKEN

    # Track which test output positions are still masked
    still_masked = torch.ones(HW, dtype=torch.bool, device=device)

    # K positions to unmask per round
    K_per_round = math.ceil(HW / n_steps)

    for _ in range(n_steps):
        if not still_masked.any():
            break

        outputs = model(current, grid_segments)
        logits  = outputs.task_logits[0, label_start:label_start + HW, :]  # (HW, C)

        # Confidence = max softmax probability
        probs      = F.softmax(logits, dim=-1)         # (HW, C)
        confidence = probs.max(dim=-1).values          # (HW,)
        best_token = probs.argmax(dim=-1)              # (HW,)

        # Mask confidence at already-unmasked positions so they aren't re-chosen
        confidence_masked = confidence.clone()
        confidence_masked[~still_masked] = -1.0

        # Select top-K still-masked positions by confidence
        n_unmask = min(K_per_round, still_masked.sum().item())
        topk_rel = confidence_masked.topk(int(n_unmask)).indices  # relative to label_start

        # Fill chosen positions in the running sequence
        for rel_idx in topk_rel:
            abs_idx = label_start + rel_idx.item()
            if abs_idx < T:
                current[0, abs_idx] = best_token[rel_idx]
            still_masked[rel_idx] = False

    # Final forward pass to get predictions for all positions
    outputs  = model(current, grid_segments)
    logits   = outputs.task_logits[0, label_start:label_start + HW, :]
    # Use whatever is in current (filled by iterative unmasking) for already-placed
    # tokens, but take argmax for any remaining masked positions
    final_preds = logits.argmax(dim=-1)   # (HW,)

    # For positions that were placed during unmasking, use placed values
    placed_mask = ~still_masked
    if placed_mask.any():
        placed_vals = current[0, label_start:label_start + HW]
        final_preds[placed_mask] = placed_vals[placed_mask]

    # Ground truth for exact match
    gt = tokens_orig[0, label_start:label_start + HW]   # (HW,)
    exact_match = float((final_preds == gt).all().item())

    return final_preds, exact_match


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def exact_match_accuracy(
    model: OaKModel,
    H: int,
    W: int,
    K: int,
    difficulty: int,
    n_episodes: int,
    chunk_size: int,
    rng: np.random.Generator,
    device: torch.device,
    use_iterative: bool = True,
    n_steps: int = 10,
) -> Tuple[float, float]:
    """Compute exact-match and per-cell accuracy over n_episodes episodes.

    If use_iterative=True, uses iterative_unmask for inference (fully masked
    start, n_steps rounds).  Otherwise uses single-pass teacher-forced
    decoding (MASK on test output positions, argmax at those positions).

    Returns:
        (exact_match_acc, per_cell_acc) both in [0, 1]
    """
    model.eval()
    HW = H * W

    exact_hits  = 0
    cell_correct = 0
    cell_total   = 0

    for _ in range(n_episodes):
        batch = make_batch(H, W, K, difficulty, 1, chunk_size, rng, device)
        label_start  = batch['label_starts'][0]
        tokens_orig  = batch['tokens_orig']   # (1, T)

        if use_iterative:
            # Reconstruct fully-masked version for eval (ignore training mask)
            tokens_eval = tokens_orig.clone()
            tokens_eval[0, label_start:label_start + HW] = MASK_TOKEN

            preds, em = iterative_unmask(
                model,
                tokens_eval,
                tokens_orig,
                batch['grid_segments'],
                label_start,
                HW,
                n_steps=n_steps,
                device=device,
            )
            gt = tokens_orig[0, label_start:label_start + HW]
        else:
            # Single-pass teacher-forced: mask all test output positions
            tokens_eval = tokens_orig.clone()
            tokens_eval[0, label_start:label_start + HW] = MASK_TOKEN

            outputs = model(tokens_eval, batch['grid_segments'])
            logits  = outputs.task_logits[0, label_start:label_start + HW, :]
            preds   = logits.argmax(dim=-1)
            gt      = tokens_orig[0, label_start:label_start + HW]
            em      = float((preds == gt).all().item())

        exact_hits   += int(em)
        cell_correct += (preds == gt).sum().item()
        cell_total   += HW

    model.train()
    return exact_hits / n_episodes, cell_correct / cell_total


# ---------------------------------------------------------------------------
# Difficulty-level curriculum helpers
# ---------------------------------------------------------------------------

def _sample_hwk(difficulty: int, rng: np.random.Generator) -> Tuple[int, int, int]:
    """Sample H, W, K from difficulty-level ranges.

    difficulty 1: H=W ~ Uniform(3,6),  K ~ Uniform(2,3)
    difficulty 2: H=W ~ Uniform(4,10), K ~ Uniform(2,5)
    difficulty 3: H=W ~ Uniform(5,13), K ~ Uniform(3,8)
    """
    if difficulty == 1:
        hw = int(rng.integers(3, 7))       # [3, 6]
        k  = int(rng.integers(2, 4))       # [2, 3]
    elif difficulty == 2:
        hw = int(rng.integers(4, 11))      # [4, 10]
        k  = int(rng.integers(2, 6))       # [2, 5]
    else:
        hw = int(rng.integers(5, 14))      # [5, 13]
        k  = int(rng.integers(3, 9))       # [3, 8]
    return hw, hw, k


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Model ---
    config = OaKConfig(
        d_model     = args.d_model,
        d_state     = 64,
        expand      = 2,
        headdim     = 64,
        chunk_size  = args.chunk_size,
        n_layer     = args.n_layer,
        mlp_expand  = 4,
        stable_ssm  = True,
        num_options = 1,    # Phase 1: single option
        d_option    = 32,
        n_gvfs      = 5,
    )
    model = OaKModel(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = 1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    rng = np.random.default_rng(args.seed)

    print(f'\nMDLM Training — difficulty={args.difficulty}')
    print(f'Steps={args.steps}, batch={args.batch_size}, chunk={args.chunk_size}')
    print(f'H/W/K sampled per batch from difficulty range\n')

    t0          = time.time()
    loss_window = []

    for step in range(1, args.steps + 1):
        model.train()

        # Sample H, W, K fresh each batch
        H, W, K = _sample_hwk(args.difficulty, rng)

        batch = make_batch(
            H, W, K, args.difficulty,
            args.batch_size, config.chunk_size,
            rng, device,
        )

        outputs = model(batch['tokens'], batch['grid_segments'])

        total_loss, task_loss, gvf_loss = compute_losses(
            outputs, batch, lambda_gvf=args.lambda_gvf
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_window.append(task_loss.item())
        if len(loss_window) > 50:
            loss_window.pop(0)

        if step % args.log_interval == 0:
            elapsed  = time.time() - t0
            avg_loss = sum(loss_window) / len(loss_window)
            lr_now   = scheduler.get_last_lr()[0]
            print(
                f'step {step:6d} | loss {avg_loss:.4f} | '
                f'gvf {gvf_loss.item():.4f} | '
                f'lr {lr_now:.2e} | H={H} W={W} K={K} | {elapsed:.0f}s'
            )

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('EVALUATION — iterative unmasking (n_steps=10)')
    print('=' * 60)

    eval_rng = np.random.default_rng(args.seed + 999)

    # Use a fixed representative (H, W, K) for evaluation
    # Middle of the difficulty range
    if args.difficulty == 1:
        H_eval, W_eval, K_eval = 4, 4, 2
    elif args.difficulty == 2:
        H_eval, W_eval, K_eval = 6, 6, 3
    else:
        H_eval, W_eval, K_eval = 8, 8, 5

    em_acc, cell_acc = exact_match_accuracy(
        model, H_eval, W_eval, K_eval, args.difficulty,
        n_episodes   = args.eval_episodes,
        chunk_size   = config.chunk_size,
        rng          = eval_rng,
        device       = device,
        use_iterative = True,
        n_steps      = 10,
    )
    print(f'Exact-match accuracy : {em_acc * 100:.1f}%  ({args.eval_episodes} episodes)')
    print(f'Per-cell accuracy    : {cell_acc * 100:.1f}%')
    print(f'H={H_eval} W={W_eval} K={K_eval} difficulty={args.difficulty}')
    print(f'Total training time  : {time.time() - t0:.0f}s')
    print('=' * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OaK-Mamba3 MDLM Training (env1)')
    parser.add_argument('--d_model',      type=int,   default=128,
                        help='Model width (default 128 for fast smoke-test)')
    parser.add_argument('--n_layer',      type=int,   default=4)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--steps',        type=int,   default=2000)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--difficulty',   type=int,   default=2,
                        choices=[1, 2, 3],
                        help='Task difficulty (1=easy, 3=hard)')
    parser.add_argument('--chunk_size',   type=int,   default=64)
    parser.add_argument('--log_interval', type=int,   default=100)
    parser.add_argument('--eval_episodes',type=int,   default=200)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--lambda_gvf',   type=float, default=0.1,
                        help='Weight on GVF-0 auxiliary loss')
    args = parser.parse_args()

    train(args)
