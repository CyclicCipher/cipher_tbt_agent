"""LeWorldModel on DriftField -- step (1) of the (2)->(1) comparison.

A latent world model (encoder + autoregressive AdaLN predictor + SIGReg, faithful to
arXiv:2603.19312 / lucas-maes/le-wm) trained with MSE(next-latent, detached) + lambda*SIGReg
+ a decode probe. Evaluation rolls the latent forward across the query-time grid: OOD time
bins are reached by COMPOSING in-distribution steps, the mechanism the supervised readouts
(train_field.py) lacked. Metrics are identical to train_field (accuracy for deterministic,
KL=CE-H for stochastic; censoring-masked `when`), so the numbers drop straight into the
data-point-#3 comparison.

GPU job (Mistake #36):
    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_lewm.py --sigma 0 --steps 4000
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
from tasks import DriftField  # noqa: E402
from train_field import _split_metric, _wsd_lambda, per_bin_acc  # noqa: E402  (reuse metric + schedule)

EARLY_EVALS = (10, 25, 50, 100, 150)


@dataclass
class LeWMConfig:
    sigma: float = 0.0
    n_obs: int = 12
    t_obs: float = 10.0
    t_max: float = 20.0
    t_bins: int = 20
    v_min: float = 0.0
    v_max: float = 64.0
    v_bins: int = 32
    v0_range: tuple[float, float] = (0.0, 20.0)
    mu_range: tuple[float, float] = (0.5, 2.0)
    dim: int = 128
    heads: int = 4
    depth: int = 4
    reg: str = "subjepa"             # default: Sub-JEPA subspace reg; "sigreg" = ambient SIGReg
    num_subspaces: int = 32
    num_proj: int = 256
    lam: float = 1.0                 # regularizer weight
    steps: int = 4000
    batch_size: int = 128
    lr: float = 5e-4                 # le-wm default
    weight_decay: float = 0.01
    eval_every: int = 100
    eval_batch: int = 256
    schedule: bool = True
    warmup_frac: float = 0.05
    cooldown_frac: float = 0.2
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    smoke: bool = False


def _task(cfg: LeWMConfig) -> DriftField:
    return DriftField(n_obs=cfg.n_obs, t_obs=cfg.t_obs, t_max=cfg.t_max, t_bins=cfg.t_bins,
                      v_min=cfg.v_min, v_max=cfg.v_max, v_bins=cfg.v_bins,
                      v0_range=cfg.v0_range, mu_range=cfg.mu_range, sigma=cfg.sigma, seed=cfg.seed)


@torch.no_grad()
def evaluate(model, task, cfg, centers, gen):
    model.eval()
    b = task.sample(cfg.eval_batch, generator=gen).to(cfg.device)
    what, when = model.what_when(b.obs_v, b.obs_t, centers, cfg.t_obs)
    wm = _split_metric(what, b.field_target, b.time_mask)
    nm = _split_metric(when, b.when_target, b.thr_mask, valid=b.when_valid)
    model.train()
    return {"what": wm, "when": nm, "what_per_bin": per_bin_acc(what, b.field_target)}


def run_lewm(cfg: LeWMConfig) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.depth, cfg.num_proj, cfg.num_subspaces = 32, 2, 64, 8
        cfg.steps, cfg.batch_size, cfg.eval_batch, cfg.eval_every = 2, 16, 16, 2
        cfg.device = "cpu"

    task = _task(cfg)
    torch.manual_seed(cfg.seed)
    regime = "stochastic" if cfg.sigma > 0 else "deterministic"
    key = "kl" if cfg.sigma > 0 else "acc"
    centers = task.t_centers.to(torch.float32).to(cfg.device)
    model = LeWorldModel(dim=cfg.dim, heads=cfg.heads, depth=cfg.depth, v_bins=cfg.v_bins,
                         v_min=cfg.v_min, v_max=cfg.v_max, num_proj=cfg.num_proj, lam=cfg.lam,
                         reg=cfg.reg, num_subspaces=cfg.num_subspaces).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _wsd_lambda(cfg)) if cfg.schedule else None
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 999)
    print(f"[lewm] DriftField {regime} sigma={cfg.sigma} | reg={cfg.reg} "
          f"(subspaces={cfg.num_subspaces}) lam={cfg.lam} | obs_win=[0,{cfg.t_obs}] "
          f"horizon=[0,{cfg.t_max}] | device={cfg.device}")

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
            rec = {"step": step, "mse": out["mse"].item(), "sigreg": out["sigreg"].item(),
                   "decode": out["decode"].item(), **m}
            history.append(rec)
            w, n = m["what"], m["when"]
            print(f"[lewm] step {step:>5} mse {out['mse'].item():.3f} sig {out['sigreg'].item():.2f} "
                  f"dec {out['decode'].item():.3f} | what {key}_in {w[key+'_in']:.3f} "
                  f"{key}_ood {w[key+'_ood']:.3f} | when {key}_in {n[key+'_in']:.3f} "
                  f"{key}_ood {n[key+'_ood']:.3f}")

    result = {"sigma": cfg.sigma, "regime": regime, "model": "lewm", "reg": cfg.reg,
              "n_obs": cfg.n_obs, "t_centers": task.t_centers.tolist(),
              "time_mask": task.time_mask.tolist(),
              "history": history, "final": history[-1] if history else {}}
    if not cfg.smoke:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"lewm_{regime}_{cfg.reg}_n{cfg.n_obs}_seed{cfg.seed}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[lewm] wrote metrics to {path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Step 1: LeWorldModel latent world model on DriftField")
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--reg", choices=("subjepa", "sigreg"), default="subjepa")
    p.add_argument("--num_subspaces", type=int, default=32)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_obs", type=int, default=12, help="observations per trajectory (denser = better mu estimate)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_schedule", action="store_false", dest="schedule")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run_lewm(LeWMConfig(sigma=a.sigma, steps=a.steps, dim=a.dim, depth=a.depth, reg=a.reg,
                        num_subspaces=a.num_subspaces, lam=a.lam, batch_size=a.batch_size,
                        n_obs=a.n_obs, seed=a.seed, schedule=a.schedule, smoke=a.smoke))


if __name__ == "__main__":
    main()
