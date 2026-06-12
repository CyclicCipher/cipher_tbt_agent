"""TBAF drift test, mechanism-matched: the corrected common-mode-rejecting activation
inserted into LeWM's PREDICTOR (the iterated latent path), on the stable PeriodicField task.

Unlike the token transformer (no carried latent), LeWM's rollout feeds its own predicted
latents back, so phase/amplitude drift can actually accumulate while a perfect model would
track the bounded periodic wave forever. The question: does `tbaf` flatten the rollout-decay
curve vs a capacity-matched `gelu` sublayer? `tbaf_verbatim` is the artifact control.

GPU job (Mistake #36):
    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_lewm_periodic.py --steps 3000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import LeWorldModel  # noqa: E402
from tasks import PeriodicField  # noqa: E402
from train_field import _wsd_lambda  # noqa: E402

ARMS = ("baseline", "gelu", "tbaf", "commonmode", "tbaf_verbatim")
EARLY_EVALS = (10, 25, 50, 100, 150)
_ACT = {"baseline": "none", "gelu": "gelu", "tbaf": "tbaf",
        "commonmode": "commonmode", "tbaf_verbatim": "tbaf_verbatim"}


@dataclass
class PeriodicLeWMConfig:
    n_obs: int = 20
    t_obs: float = 15.0
    t_max: float = 30.0
    t_bins: int = 30
    v_min: float = 0.0
    v_max: float = 24.0
    v_bins: int = 24
    period_range: tuple[float, float] = (3.0, 7.0)
    dim: int = 128
    heads: int = 4
    depth: int = 4
    reg: str = "sigreg"              # cheap reg; the activation is the variable under test
    num_proj: int = 256
    lam: float = 1.0
    steps: int = 3000
    batch_size: int = 128
    lr: float = 5e-4
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


def _task(cfg: PeriodicLeWMConfig) -> PeriodicField:
    return PeriodicField(n_obs=cfg.n_obs, t_obs=cfg.t_obs, t_max=cfg.t_max, t_bins=cfg.t_bins,
                         v_min=cfg.v_min, v_max=cfg.v_max, v_bins=cfg.v_bins,
                         period_range=cfg.period_range, seed=cfg.seed)


@torch.no_grad()
def evaluate(model, task, cfg, centers, gen):
    model.eval()
    b = task.sample(cfg.eval_batch, generator=gen).to(cfg.device)
    what = model.rollout_what(b.obs_v, b.obs_t, centers, cfg.t_obs)      # (B, Tb, Vb)
    acc_bin = (what.argmax(-1) == b.field_target.argmax(-1)).float().mean(0)  # (Tb,)
    ood = task.time_mask.to(cfg.device) == 0
    model.train()
    return {"per_bin": acc_bin.tolist(), "acc_in": acc_bin[~ood].mean().item(),
            "acc_ood": acc_bin[ood].mean().item()}


def train_arm(arm, cfg, task, centers):
    torch.manual_seed(cfg.seed)
    model = LeWorldModel(dim=cfg.dim, heads=cfg.heads, depth=cfg.depth, v_bins=cfg.v_bins,
                         v_min=cfg.v_min, v_max=cfg.v_max, num_proj=cfg.num_proj, lam=cfg.lam,
                         reg=cfg.reg, inject_act=_ACT[arm]).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _wsd_lambda(cfg)) if cfg.schedule else None
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 999)
    history = []
    model.train()
    for step in range(1, cfg.steps + 1):
        b = task.sample(cfg.batch_size, generator=gen).to(cfg.device)
        out = model.losses(b.obs_v, b.obs_t)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        if step in EARLY_EVALS or step % cfg.eval_every == 0 or step == cfg.steps:
            m = evaluate(model, task, cfg, centers, eval_gen)
            history.append({"step": step, "mse": out["mse"].item(), **m})
            print(f"[{arm:13}] step {step:>5} mse {out['mse'].item():.3f} | what "
                  f"in {m['acc_in']:.3f}  ood(rollout) {m['acc_ood']:.3f}")
    return history


def run(cfg: PeriodicLeWMConfig) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.depth, cfg.num_proj = 24, 2, 64
        cfg.steps, cfg.batch_size, cfg.eval_batch, cfg.eval_every = 2, 16, 16, 2
        cfg.device = "cpu"

    task = _task(cfg)
    torch.manual_seed(cfg.seed)
    centers = task.t_centers.to(torch.float32).to(cfg.device)
    print(f"[periodic] P={cfg.period_range} obs_win=[0,{cfg.t_obs}] horizon=[0,{cfg.t_max}] "
          f"| reg={cfg.reg} | device={cfg.device}")
    result = {"t_centers": task.t_centers.tolist(), "time_mask": task.time_mask.tolist(),
              "t_obs": cfg.t_obs, "arms": {}}
    for arm in cfg.arms:
        h = train_arm(arm, cfg, task, centers)
        result["arms"][arm] = {"history": h, "final": h[-1] if h else {}}

    print("\n[compare] final what in / ood(rollout):")
    for arm in cfg.arms:
        f = result["arms"][arm]["final"]
        if f:
            print(f"  {arm:13}  {f['acc_in']:.3f} / {f['acc_ood']:.3f}")

    if not cfg.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"periodic_lewm_seed{cfg.seed}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[periodic] wrote metrics to {path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="TBAF drift test in LeWM's predictor (PeriodicField)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--reg", choices=("sigreg", "subjepa"), default="sigreg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_schedule", action="store_false", dest="schedule")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(PeriodicLeWMConfig(steps=a.steps, arms=tuple(a.arms), reg=a.reg, seed=a.seed,
                           schedule=a.schedule, smoke=a.smoke))


if __name__ == "__main__":
    main()
