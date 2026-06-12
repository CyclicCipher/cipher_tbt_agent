"""TBAF drift test: does a single common-mode-rejecting activation reduce free-running
rollout drift in a PoPE transformer?

Task = MotifEcho (periodic, deterministic -> any rollout drift is the architecture's).
Metric = the rollout-decay curve: free-running next-token accuracy vs rollout depth.

Arms (one inserted ActivationFFN sublayer each, except `baseline`):
  baseline       -- plain PoPE transformer (no inserted sublayer)
  gelu           -- + one GELU FFN sublayer       (capacity-matched control)
  tbaf           -- + one corrected per-token TBAF (the test)
  tbaf_verbatim  -- + one verbatim repo-TBAF       (artifact control; batch-coupled)
  commonmode     -- + one mean-subtraction sublayer (invariance without the nonlinearity)

The question: does `tbaf` flatten the decay curve vs `gelu` (same sublayer, different act)?
And does `tbaf_verbatim` break / only "work" by going constant?

GPU job (Mistake #36):
    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_motif.py --steps 4000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from tasks import MotifEcho  # noqa: E402
from train_field import _wsd_lambda  # noqa: E402

ARMS = ("baseline", "gelu", "tbaf", "tbaf_verbatim", "commonmode")
EARLY_EVALS = (10, 25, 50, 100, 150)
_ACT = {"baseline": "none", "gelu": "gelu", "tbaf": "tbaf",
        "tbaf_verbatim": "tbaf_verbatim", "commonmode": "commonmode"}


@dataclass
class MotifConfig:
    vocab_size: int = 16
    motif_min: int = 2
    motif_max: int = 6
    context_len: int = 18
    horizon: int = 60
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 6
    steps: int = 4000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 250
    eval_batch: int = 256
    schedule: bool = True
    warmup_frac: float = 0.05
    cooldown_frac: float = 0.2
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    arms: tuple[str, ...] = ARMS
    smoke: bool = False


def _build(arm: str, cfg: MotifConfig, task: MotifEcho) -> FixedDepthTransformer:
    return FixedDepthTransformer(FixedDepthConfig(
        vocab_size=task.V, dim=cfg.dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        max_seq=task.seq_len, pos_mode="pope", inject_act=_ACT[arm]))


@torch.no_grad()
def rollout_decay(model, task, cfg, gen):
    """Free-running generation from the fixed context; per-step accuracy vs the true periodic
    continuation -> the decay curve (length = horizon)."""
    model.eval()
    b = task.sample(cfg.eval_batch, generator=gen).to(cfg.device)
    seq = b.tokens[:, :task.context_len].clone()
    acc = []
    for k in range(task.horizon):
        nxt = model(seq)[:, -1].argmax(-1)                  # (B,) greedy next token
        acc.append((nxt == b.tokens[:, task.context_len + k]).float().mean().item())
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    model.train()
    return acc


def train_arm(arm, cfg, task):
    torch.manual_seed(cfg.seed)
    model = _build(arm, cfg, task).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _wsd_lambda(cfg)) if cfg.schedule else None
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 999)
    warmup = cfg.motif_max                                    # first period is unpredictable
    history = []
    model.train()
    for step in range(1, cfg.steps + 1):
        b = task.sample(cfg.batch_size, generator=gen).to(cfg.device)
        logits = model(b.tokens[:, :-1])                     # predict next token
        loss = F.cross_entropy(logits[:, warmup:].reshape(-1, task.V),
                               b.tokens[:, warmup + 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        if step in EARLY_EVALS or step % cfg.eval_every == 0 or step == cfg.steps:
            decay = rollout_decay(model, task, cfg, eval_gen)
            # summary scalars: accuracy at the first rolled step, and averaged over the horizon
            rec = {"step": step, "loss": loss.item(), "decay": decay,
                   "acc_step1": decay[0], "acc_mean": sum(decay) / len(decay),
                   "acc_last": decay[-1]}
            history.append(rec)
            print(f"[{arm:13}] step {step:>5} loss {loss.item():.3f} | rollout acc "
                  f"step1 {decay[0]:.3f}  mean {rec['acc_mean']:.3f}  last {decay[-1]:.3f}")
    return history


def run_motif(cfg: MotifConfig) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.n_layers = 24, 2
        cfg.steps, cfg.batch_size, cfg.eval_batch, cfg.eval_every = 2, 16, 16, 2
        cfg.horizon, cfg.device = 8, "cpu"

    task = MotifEcho(vocab_size=cfg.vocab_size, motif_min=cfg.motif_min, motif_max=cfg.motif_max,
                     context_len=cfg.context_len, horizon=cfg.horizon, seed=cfg.seed)
    torch.manual_seed(cfg.seed)
    print(f"[motif] V={task.V} period={cfg.motif_min}-{cfg.motif_max} context={task.context_len} "
          f"horizon={task.horizon} | device={cfg.device}")
    result = {"horizon": task.horizon, "context_len": task.context_len, "arms": {}}
    for arm in cfg.arms:
        h = train_arm(arm, cfg, task)
        result["arms"][arm] = {"history": h, "final": h[-1] if h else {}}

    print("\n[compare] final rollout accuracy (step1 / horizon-mean / last):")
    for arm in cfg.arms:
        f = result["arms"][arm]["final"]
        if f:
            print(f"  {arm:13}  {f['acc_step1']:.3f} / {f['acc_mean']:.3f} / {f['acc_last']:.3f}")

    if not cfg.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"motif_seed{cfg.seed}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[motif] wrote metrics to {path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="TBAF drift test on MotifEcho")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_schedule", action="store_false", dest="schedule")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run_motif(MotifConfig(steps=a.steps, horizon=a.horizon, arms=tuple(a.arms), seed=a.seed,
                          schedule=a.schedule, smoke=a.smoke))


if __name__ == "__main__":
    main()
