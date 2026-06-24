"""Learnability metrics — the same signatures used in RWM's data point #1.

- action-match accuracy (raw and masked to the valid action set)
- the train-vs-held-out generalization gap (the shift-invisibility signal)
- time-to-threshold over a training history
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

# LockPath only uses the four directional actions (GameAction values 1..4).
VALID_ACTIONS = (1, 2, 3, 4)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str,
    valid_actions: Sequence[int] = VALID_ACTIONS,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Return loss, raw argmax accuracy, and accuracy masked to valid actions."""
    was_training = model.training
    model.eval()
    valid = torch.tensor(valid_actions, device=device)
    loss_sum = correct = masked_correct = total = 0
    for bi, (fw, aw, tg) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        fw, aw, tg = fw.to(device), aw.to(device), tg.to(device)
        logits, _ = model(fw, aw)
        loss_sum += F.cross_entropy(logits, tg, reduction="sum").item()
        correct += (logits.argmax(-1) == tg).sum().item()
        masked_pred = valid[logits[:, valid].argmax(-1)]
        masked_correct += (masked_pred == tg).sum().item()
        total += tg.numel()
    if was_training:
        model.train()
    total = max(total, 1)
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "masked_acc": masked_correct / total,
    }


@torch.no_grad()
def evaluate_tensors(
    model,
    frames: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    device: str,
    valid_actions: Sequence[int] = VALID_ACTIONS,
    chunk: int = 2048,
) -> Dict[str, float]:
    """Evaluate over on-device tensors in chunks, syncing once at the end."""
    was_training = model.training
    model.eval()
    valid = torch.tensor(valid_actions, device=device)
    n = targets.shape[0]
    loss = correct = masked = None
    for s in range(0, n, chunk):
        fw = frames[s:s + chunk].long()
        logits, _ = model(fw, actions[s:s + chunk])
        tg = targets[s:s + chunk]
        l = torch.nn.functional.cross_entropy(logits, tg, reduction="sum")
        c = (logits.argmax(-1) == tg).sum()
        m = (valid[logits[:, valid].argmax(-1)] == tg).sum()
        loss = l if loss is None else loss + l
        correct = c if correct is None else correct + c
        masked = m if masked is None else masked + m
    if was_training:
        model.train()
    denom = max(n, 1)
    return {
        "loss": float(loss) / denom,
        "acc": float(correct) / denom,
        "masked_acc": float(masked) / denom,
    }


def time_to_threshold(
    history: List[Dict[str, float]], key: str, thresh: float
) -> Optional[int]:
    """First logged step at which `history[i][key] >= thresh`, else None."""
    for h in history:
        if h.get(key, 0.0) >= thresh:
            return h["step"]
    return None
