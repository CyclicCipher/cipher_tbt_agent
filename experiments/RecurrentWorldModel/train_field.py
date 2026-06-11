"""Stage 3 of the temporal fork: unified (time x value) field vs separate heads.

Four readout arms on an identical continuous-input PoPE trunk; the question is
whether predicting ONE field (and reading both `what` and `when` off it) learns
better than two specialised query heads, and whether the inverse query (`when`)
comes FREE from a field trained only on the forward query (`what`) -- control C1.

  unified       -- one field; loss = what + when (when read is parameter-free)
  unified_fwd   -- one field; loss = what ONLY; `when` read cold at eval (C1: the
                   headline unification test -- does the inverse come for free?)
  separate      -- two query heads; loss = what + when
  separate_fwd  -- two query heads; loss = what ONLY (the `when` head is never
                   trained -> structural C1 baseline, expected chance)

Dynamics via `--sigma`: 0 = deterministic (sharp ridge; metric = accuracy),
>0 = stochastic (diffuse band; metric = KL(true||model) = CE - H(p_true), the
closed-form floor; accuracy is meaningless on a distribution). OOD = horizon
extrapolation: only time bins within the observation window are supervised; bins
in the unobserved future are eval-only.

GPU job (Mistake #36):
    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_field.py --sigma 0 --steps 4000
    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_field.py --sigma 2 --steps 4000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FunctionalFieldModel, SeparateHeadsModel, UnifiedFieldModel  # noqa: E402
from tasks import DriftField  # noqa: E402

# Full registry (any selectable via --arms). The DEFAULT run is the lean set that answers
# step (2): fixed-grid vs functional readout, each both-loss and forward-only. The `separate`
# arms (the decoupled-head baseline, characterised in data point #3) stay available but are
# off by default.
ALL_ARMS = ("unified", "unified_fwd", "separate", "separate_fwd", "functional", "functional_fwd")
ARMS = ("unified", "unified_fwd", "functional", "functional_fwd")
EARLY_EVALS = (10, 25, 50, 100, 150)


@dataclass
class FieldConfig:
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
    n_heads: int = 4
    n_layers: int = 6
    steps: int = 4000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 100
    eval_batch: int = 256
    schedule: bool = True            # WSD: linear warmup -> constant -> linear cooldown
    warmup_frac: float = 0.05
    cooldown_frac: float = 0.2
    ff_freqs: int = 6                # functional readout: # Fourier frequencies per coordinate
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    arms: tuple[str, ...] = ARMS
    smoke: bool = False


def _make_task(cfg: FieldConfig) -> DriftField:
    return DriftField(n_obs=cfg.n_obs, t_obs=cfg.t_obs, t_max=cfg.t_max, t_bins=cfg.t_bins,
                      v_min=cfg.v_min, v_max=cfg.v_max, v_bins=cfg.v_bins,
                      v0_range=cfg.v0_range, mu_range=cfg.mu_range, sigma=cfg.sigma, seed=cfg.seed)


def _build(arm: str, cfg: FieldConfig, task: DriftField):
    if arm.startswith("unified"):
        return UnifiedFieldModel(cfg.dim, cfg.n_heads, cfg.n_layers, cfg.n_obs,
                                 cfg.t_bins, cfg.v_bins)
    tcn = (task.t_centers / cfg.t_max).to(torch.float32)
    vcn = (task.v_centers / cfg.v_max).to(torch.float32)
    if arm.startswith("functional"):
        return FunctionalFieldModel(cfg.dim, cfg.n_heads, cfg.n_layers, cfg.n_obs,
                                    cfg.t_bins, cfg.v_bins, tcn, vcn, n_freq=cfg.ff_freqs)
    return SeparateHeadsModel(cfg.dim, cfg.n_heads, cfg.n_layers, cfg.n_obs,
                              cfg.t_bins, cfg.v_bins, tcn, vcn)


def _soft_ce(logp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-sum target*logp over the class dim -> (B, nbins)."""
    return -(target * logp).sum(-1)


def _split_metric(logp, target, bin_mask, valid=None):
    """Per-bin accuracy and KL=CE-H, averaged over in-dist (mask=1) and OOD (mask=0) bins.

    `valid` (B, nbins), if given, additionally restricts to non-trivial entries (the
    `when` metric uses it to drop censored / already-crossed thresholds whose answer is a
    trivial constant -- otherwise OOD is dominated by 'never reached -> last bin')."""
    ce = _soft_ce(logp, target)                              # (B, nbins)
    acc = (logp.argmax(-1) == target.argmax(-1)).float()     # (B, nbins)
    kl = ce - DriftField.entropy(target)                     # (B, nbins)
    m = bin_mask.bool()                                      # (nbins,)
    in_sel = m[None, :].expand_as(acc).clone()
    ood_sel = (~m)[None, :].expand_as(acc).clone()
    if valid is not None:
        v = valid.bool()
        in_sel &= v
        ood_sel &= v
    pick = lambda x, s: x[s].mean().item() if s.any() else float("nan")
    return {"acc_in": pick(acc, in_sel), "acc_ood": pick(acc, ood_sel),
            "kl_in": pick(kl, in_sel), "kl_ood": pick(kl, ood_sel)}


def per_bin_acc(logp, target):
    """Accuracy at each time bin, averaged over the batch -> (Tb,) list. Lets us see whether
    OOD accuracy *falls with horizon* (the informational-ceiling signature) vs flat."""
    return (logp.argmax(-1) == target.argmax(-1)).float().mean(0).tolist()


@torch.no_grad()
def evaluate(arm, model, task, cfg, gen):
    model.eval()
    b = task.sample(cfg.eval_batch, generator=gen).to(cfg.device)
    what_logp = model.what_logp(b.obs_v, b.obs_t)
    when_logp = model.when_logp(b.obs_v, b.obs_t)
    what = _split_metric(what_logp, b.field_target, b.time_mask)
    when = _split_metric(when_logp, b.when_target, b.thr_mask, valid=b.when_valid)
    field_ent = DriftField.entropy(what_logp.exp()).mean().item()
    model.train()
    return {"what": what, "when": when, "field_entropy": field_ent,
            "what_per_bin": per_bin_acc(what_logp, b.field_target)}


def _wsd_lambda(cfg: FieldConfig):
    """WSD multiplier: linear warmup -> constant 1.0 -> linear cooldown to 0."""
    warmup = max(1, int(cfg.warmup_frac * cfg.steps))
    cooldown = max(1, int(cfg.cooldown_frac * cfg.steps))

    def mult(step: int) -> float:
        if step < warmup:
            return step / warmup
        if step > cfg.steps - cooldown:
            return max(0.0, (cfg.steps - step) / cooldown)
        return 1.0
    return mult


def train_arm(arm, cfg, task):
    torch.manual_seed(cfg.seed)
    model = _build(arm, cfg, task).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, _wsd_lambda(cfg)) if cfg.schedule else None)
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 999)
    forward_only = arm.endswith("_fwd")
    stoch = cfg.sigma > 0.0
    key = "kl" if stoch else "acc"
    history = []
    model.train()
    for step in range(1, cfg.steps + 1):
        b = task.sample(cfg.batch_size, generator=gen).to(cfg.device)
        what_loss = (_soft_ce(model.what_logp(b.obs_v, b.obs_t), b.field_target)
                     * b.time_mask).sum() / b.time_mask.sum().clamp_min(1.0) / b.obs_v.shape[0]
        loss = what_loss
        if not forward_only:
            when_loss = (_soft_ce(model.when_logp(b.obs_v, b.obs_t), b.when_target)
                         * b.thr_mask).sum() / b.thr_mask.sum().clamp_min(1.0) / b.obs_v.shape[0]
            loss = loss + when_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        if step in EARLY_EVALS or step % cfg.eval_every == 0 or step == cfg.steps:
            m = evaluate(arm, model, task, cfg, eval_gen)
            history.append({"step": step, "loss": loss.item(), **m})
            w, n = m["what"], m["when"]
            print(f"[{arm:13}] step {step:>5} loss {loss.item():.3f} | "
                  f"what {key}_in {w[key+'_in']:.3f} {key}_ood {w[key+'_ood']:.3f} | "
                  f"when {key}_in {n[key+'_in']:.3f} {key}_ood {n[key+'_ood']:.3f}")
    return history


def run_field(cfg: FieldConfig) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.n_layers = 32, 2
        cfg.steps, cfg.batch_size, cfg.eval_batch, cfg.eval_every = 2, 16, 16, 2
        cfg.device = "cpu"

    task = _make_task(cfg)
    torch.manual_seed(cfg.seed)
    regime = "stochastic" if cfg.sigma > 0 else "deterministic"
    print(f"[field] DriftField {regime} sigma={cfg.sigma} | Tb={cfg.t_bins} Vb={cfg.v_bins} "
          f"| obs_win=[0,{cfg.t_obs}] horizon=[0,{cfg.t_max}] | device={cfg.device}")
    result = {"sigma": cfg.sigma, "regime": regime, "n_obs": cfg.n_obs,
              "t_centers": task.t_centers.tolist(), "time_mask": task.time_mask.tolist(),
              "arms": {}}
    for arm in cfg.arms:
        h = train_arm(arm, cfg, task)
        result["arms"][arm] = {"history": h, "final": h[-1] if h else {}}

    key = "kl" if cfg.sigma > 0 else "acc"
    print(f"\n[compare] final {key} (what / when, in / ood):")
    for arm in cfg.arms:
        f = result["arms"][arm]["final"]
        if f:
            w, n = f["what"], f["when"]
            print(f"  {arm:13}  what {w[key+'_in']:.3f}/{w[key+'_ood']:.3f}   "
                  f"when {n[key+'_in']:.3f}/{n[key+'_ood']:.3f}")

    if not cfg.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"field_{regime}_n{cfg.n_obs}_seed{cfg.seed}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[field] wrote metrics to {path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 3: unified field vs separate heads")
    p.add_argument("--sigma", type=float, default=0.0, help="0=deterministic, >0=stochastic")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ALL_ARMS)
    p.add_argument("--n_obs", type=int, default=12, help="observations per trajectory (denser = better mu estimate)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_schedule", action="store_false", dest="schedule",
                   help="disable the WSD LR schedule (constant LR)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run_field(FieldConfig(sigma=a.sigma, steps=a.steps, dim=a.dim, n_layers=a.layers,
                          batch_size=a.batch_size, arms=tuple(a.arms), n_obs=a.n_obs,
                          seed=a.seed, schedule=a.schedule, smoke=a.smoke))


if __name__ == "__main__":
    main()
