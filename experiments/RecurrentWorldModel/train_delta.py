"""Stage 2 of the temporal fork: Δ-encoding vs absolute encoding under distribution shift.

Two arms on ShiftSeq (target = total change, shift-invariant), identical architecture
(continuous-input PoPE transformer), differing ONLY in the input representation:
  * absolute -- the running absolute values (the change is a small signal buried in
    large absolutes; at test v0 SHIFTS to an unseen range)
  * delta    -- the increments (independent of v0 => identical train/test)

Eval: in-distribution (v0 in train range) and OOD-shift (v0 in a far, unseen range).
Hypothesis (Makushkin / compress-the-source): delta generalizes to the shift for free;
absolute must difference large shifted values and should degrade OOD. A *negative*
result (absolute also generalizes by learning clean differencing) is equally
informative -- it constrains when Δ-encoding actually helps.

GPU job (Mistake #36):
    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_delta.py --steps 4000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from core.model import count_parameters  # noqa: E402
from tasks import ShiftSeq  # noqa: E402

ARMS = ("absolute", "delta")
EARLY_EVALS = (10, 25, 50, 100, 150)


@dataclass
class DeltaConfig:
    length: int = 8
    n_deltas: int = 4
    train_v0: tuple[float, float] = (0.0, 100.0)
    shift_v0: tuple[float, float] = (1000.0, 1100.0)   # OOD: far, unseen absolute range
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 6
    steps: int = 4000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 100
    eval_batch: int = 256
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    arms: tuple[str, ...] = ARMS
    smoke: bool = False


def ce_loss(logits, targets, mask):
    v = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, v), targets.reshape(-1), reduction="none")
    return (ce.reshape(mask.shape) * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def accuracy(logits, targets, mask):
    pred = logits.argmax(dim=-1)
    return ((pred == targets).float() * mask).sum().item() / mask.sum().clamp_min(1.0).item()


def _input(arm, batch):
    return batch.abs_input if arm == "absolute" else batch.delta_input


@torch.no_grad()
def evaluate(arm, model, task, cfg, rng):
    model.eval()
    dev = cfg.device
    b_in = task.sample(cfg.eval_batch, *cfg.train_v0, rng=rng).to(dev)
    b_ood = task.sample(cfg.eval_batch, *cfg.shift_v0, rng=rng).to(dev)
    ai = accuracy(model(_input(arm, b_in)), b_in.target, b_in.loss_mask)
    ao = accuracy(model(_input(arm, b_ood)), b_ood.target, b_ood.loss_mask)
    model.train()
    return ai, ao


def train_arm(arm, cfg, task):
    torch.manual_seed(cfg.seed)
    model = FixedDepthTransformer(FixedDepthConfig(
        vocab_size=task.target_classes, dim=cfg.dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        max_seq=task.seq_len, pos_mode="pope", continuous_input=True,
    )).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    eval_rng = random.Random(cfg.seed + 999)
    history = []
    model.train()
    for step in range(1, cfg.steps + 1):
        b = task.sample(cfg.batch_size, *cfg.train_v0, rng=rng).to(cfg.device)
        loss = ce_loss(model(_input(arm, b)), b.target, b.loss_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in EARLY_EVALS or step % cfg.eval_every == 0 or step == cfg.steps:
            ai, ao = evaluate(arm, model, task, cfg, eval_rng)
            history.append({"step": step, "loss": loss.item(), "acc_in": ai, "acc_ood": ao})
            print(f"[{arm:9}] step {step:>5} loss {loss.item():.3f} acc_in {ai:.3f} acc_ood {ao:.3f}")
    return history


def run_delta(cfg: DeltaConfig) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.n_layers = 32, 2
        cfg.steps, cfg.batch_size, cfg.eval_batch, cfg.eval_every = 2, 16, 16, 2
        cfg.device = "cpu"

    task = ShiftSeq(cfg.length, cfg.n_deltas, seed=cfg.seed)
    torch.manual_seed(cfg.seed)
    print(f"[delta] ShiftSeq L={cfg.length} D={cfg.n_deltas} | target classes {task.target_classes} "
          f"| chance ~{1/task.target_classes:.3f} | train_v0={cfg.train_v0} shift_v0={cfg.shift_v0} "
          f"| device={cfg.device}")
    result = {"chance": 1 / task.target_classes, "arms": {}}
    for arm in cfg.arms:
        h = train_arm(arm, cfg, task)
        result["arms"][arm] = {"history": h, "final": h[-1] if h else {}}

    print("\n[compare] final  acc_in / acc_ood (the shift is the test):")
    for arm in cfg.arms:
        f = result["arms"][arm]["final"]
        if f:
            print(f"  {arm:9}  in {f['acc_in']:.3f}  ood {f['acc_ood']:.3f}")

    if not cfg.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"delta_shift_seed{cfg.seed}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[delta] wrote metrics to {path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2: Δ-encoding vs absolute under shift")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run_delta(DeltaConfig(steps=a.steps, dim=a.dim, n_layers=a.layers, batch_size=a.batch_size,
                          arms=tuple(a.arms), seed=a.seed, smoke=a.smoke))


if __name__ == "__main__":
    main()
