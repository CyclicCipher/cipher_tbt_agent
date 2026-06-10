"""Stage 1 of the temporal fork: does CONTINUOUS-TIME PoPE let a transformer use
real elapsed time that integer-position PoPE cannot?

Three arms on the EventStream task (answer depends on real elapsed time; token count
!= time), identical data + init:
  * integer     -- PoPE with integer positions, no timing info at all (control floor)
  * time_input  -- integer PoPE + a learned log(1+time) input feature ("time as content")
  * continuous  -- continuous-time PoPE: timestamps drive the phase ("time as position")

The decisive comparison is time_input vs continuous: both have the timing; is encoding
it in the POSITIONAL (relative-phase) channel better than as a content feature? Plus
an OOD-by-gap eval (larger gaps at test = longer elapsed times unseen in training) to
test whether each timing representation EXTRAPOLATES.

GPU job (Mistake #36):
    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_temporal.py --steps 4000
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
from tasks import EventStream  # noqa: E402

ARMS = ("integer", "time_input", "continuous")


# dense early evals so we catch the fast initial learning (most happens by ~step 400)
EARLY_EVALS = (10, 25, 50, 100, 150)


@dataclass
class TempConfig:
    n_levels: int = 10            # value range (bigger => answers not 0-dominated)
    n_events: int = 10
    train_max_gap: int = 8
    ood_max_gap: int = 16          # longer elapsed times than trained => extrapolation test
    noise_frac: float = 0.3
    decay_per: int = 3
    fixed_dist: int = 1            # CLEAN isolation: value event at constant token distance
                                   # => token count carries 0 info, only real timing solves it
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


@torch.no_grad()
def nonzero_accuracy(logits, targets, mask, zero_token):
    """Accuracy restricted to non-zero answers -- the discriminative signal the
    0-dominated base rate hides (the decayed-to-0 cases are the trivial majority)."""
    pred = logits.argmax(dim=-1)
    m = mask * (targets != zero_token).float()
    return ((pred == targets).float() * m).sum().item() / m.sum().clamp_min(1.0).item()


def _forward(arm, model, batch):
    if arm == "integer":
        return model(batch.input_ids)
    if arm == "time_input":
        return model(batch.input_ids, time_feat=batch.timestamps)
    return model(batch.input_ids, coord=batch.timestamps)         # continuous


def build(arm, cfg, vocab, max_seq):
    return FixedDepthTransformer(FixedDepthConfig(
        vocab_size=vocab, dim=cfg.dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        max_seq=max_seq, pos_mode="pope", time_input=(arm == "time_input"),
    ))


@torch.no_grad()
def evaluate(arm, model, task, cfg, rng):
    model.eval()
    dev = cfg.device
    zt = 2  # VAL(0) token
    b_in = task.sample(cfg.eval_batch, gap_max=cfg.train_max_gap, rng=rng).to(dev)
    b_ood = task.sample(cfg.eval_batch, gap_max=cfg.ood_max_gap, rng=rng).to(dev)
    li, lo = _forward(arm, model, b_in), _forward(arm, model, b_ood)
    m = {
        "acc_in": accuracy(li, b_in.targets, b_in.loss_mask),
        "acc_ood": accuracy(lo, b_ood.targets, b_ood.loss_mask),
        "nz_in": nonzero_accuracy(li, b_in.targets, b_in.loss_mask, zt),
        "nz_ood": nonzero_accuracy(lo, b_ood.targets, b_ood.loss_mask, zt),
    }
    model.train()
    return m


def train_arm(arm, cfg, task):
    torch.manual_seed(cfg.seed)
    model = build(arm, cfg, task.vocab_size, task.seq_len).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    eval_rng = random.Random(cfg.seed + 999)
    history = []
    model.train()
    for step in range(1, cfg.steps + 1):
        b = task.sample(cfg.batch_size, gap_max=cfg.train_max_gap, rng=rng).to(cfg.device)
        loss = ce_loss(_forward(arm, model, b), b.targets, b.loss_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in EARLY_EVALS or step % cfg.eval_every == 0 or step == cfg.steps:
            m = evaluate(arm, model, task, cfg, eval_rng)
            m["step"] = step
            m["loss"] = loss.item()
            history.append(m)
            print(f"[{arm:11}] step {step:>5} loss {loss.item():.3f} "
                  f"acc_in {m['acc_in']:.3f} acc_ood {m['acc_ood']:.3f} "
                  f"| nz_in {m['nz_in']:.3f} nz_ood {m['nz_ood']:.3f}")
    return history


def run_temporal(cfg: TempConfig) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.n_layers = 32, 2
        cfg.steps, cfg.batch_size, cfg.eval_batch, cfg.eval_every = 2, 16, 16, 2
        cfg.device = "cpu"

    task = EventStream(n_levels=cfg.n_levels, n_events=cfg.n_events, max_gap=cfg.train_max_gap,
                       noise_frac=cfg.noise_frac, decay_per=cfg.decay_per,
                       fixed_dist=cfg.fixed_dist, seed=cfg.seed)
    torch.manual_seed(cfg.seed)
    print(f"[temporal] EventStream V={cfg.n_levels} fixed_dist={cfg.fixed_dist} | "
          f"chance ~{1/cfg.n_levels:.3f} | vocab {task.vocab_size} seq_len {task.seq_len} | "
          f"device={cfg.device} | arms={cfg.arms}")
    result = {"chance": 1 / cfg.n_levels, "arms": {}}
    for arm in cfg.arms:
        h = train_arm(arm, cfg, task)
        result["arms"][arm] = {"history": h, "final": h[-1] if h else {}}

    print("\n[compare] final  acc (in/ood)  |  non-zero acc (in/ood) <- the discriminative signal:")
    for arm in cfg.arms:
        f = result["arms"][arm]["final"]
        if f:
            print(f"  {arm:11}  {f['acc_in']:.3f}/{f['acc_ood']:.3f}  |  {f['nz_in']:.3f}/{f['nz_ood']:.3f}")

    if not cfg.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"temporal_pope_seed{cfg.seed}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[temporal] wrote metrics to {path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1: continuous-time PoPE on EventStream")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--ood_max_gap", type=int, default=6)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run_temporal(TempConfig(steps=a.steps, dim=a.dim, n_layers=a.layers, batch_size=a.batch_size,
                            ood_max_gap=a.ood_max_gap, arms=tuple(a.arms), seed=a.seed, smoke=a.smoke))


if __name__ == "__main__":
    main()
