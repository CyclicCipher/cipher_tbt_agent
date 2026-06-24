"""Step 4 of the temporal fork: novelty-partition continual learning.

The hypothesis, pre-loaded by data point #2: Δ-encoding made the distribution shift
*invisible* (OOD inputs were literally in-distribution). If so, the same representation
should also prevent CATASTROPHIC FORGETTING under that shift -- train regime A, then regime
B, and because B looks identical to A in delta space, learning B cannot overwrite A. Absolute
encoding, where B looks alien, should forget A badly.

The 2x2 keeps it honest (so it's not tautological):
  encoding {absolute, delta}  x  shift {offset, scale}
  * offset shift  -- B differs from A only in v0 (the additive offset).  Δ-INVISIBLE.
  * scale  shift  -- B doubles the increments themselves (delta_step=2). Δ-VISIBLE.
Prediction: forgetting is LOW only for (delta, offset). The scale row is the control showing
delta's resistance is *invisibility* (Prediction P1's integration-constant tradeoff), not magic.

Protocol: train phase_steps on A, then phase_steps on B; track accuracy on a held-out A test
throughout. Forgetting = acc_A(end of phase A) - acc_A(end of phase B).

GPU job (Mistake #36):
    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_continual.py --phase_steps 1500
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

from baselines import FixedDepthConfig, FixedDepthTransformer, PartitionedModel  # noqa: E402
from tasks import ShiftSeq  # noqa: E402

# arm = {shift}_{encoding}_{mode}: the full ablation. mode=single is the forgetful baseline;
# oracle/surprise are the partition (non-overwriting) mechanism.
ARMS = tuple(f"{s}_{e}_{m}" for s in ("offset", "scale")
             for e in ("absolute", "delta") for m in ("single", "oracle", "surprise"))
EARLY_EVALS = (10, 25, 50, 100)


@dataclass
class ContinualConfig:
    length: int = 8
    n_deltas: int = 4
    regimeA_v0: tuple[float, float] = (0.0, 100.0)
    regimeB_v0: tuple[float, float] = (1000.0, 1100.0)   # offset shift
    scale_step: int = 2                                  # scale shift: B doubles the increments
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 6
    phase_steps: int = 1500                              # steps per phase (A then B)
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 100
    eval_batch: int = 256
    surprise_k: float = 4.0          # z-score threshold for surprise-gated allocation
    surprise_cooldown: int = 50      # min steps between allocations
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    arms: tuple[str, ...] = ARMS
    smoke: bool = False


def _input(encoding, batch):
    return batch.abs_input if encoding == "absolute" else batch.delta_input


def ce_loss(logits, targets, mask):
    v = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, v), targets.reshape(-1), reduction="none")
    return (ce.reshape(mask.shape) * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def accuracy(model, encoding, batch, expert=None):
    logits = model(_input(encoding, batch), expert=expert)
    pred = logits.argmax(-1)
    m = batch.loss_mask
    return ((pred == batch.target).float() * m).sum().item() / m.sum().clamp_min(1.0).item()


def _wsd_mult(step: int, total: int, warmup_frac: float = 0.05, cooldown_frac: float = 0.2) -> float:
    warmup = max(1, int(warmup_frac * total))
    cooldown = max(1, int(cooldown_frac * total))
    if step < warmup:
        return step / warmup
    if step > total - cooldown:
        return max(0.0, (total - step) / cooldown)
    return 1.0


def _samplers(shift_type, cfg, task):
    """Return (A_regime, B_regime) sampler closures: (batch, rng) -> ShiftBatch."""
    def a_regime(b, rng):
        return task.sample(b, *cfg.regimeA_v0, rng=rng, delta_step=1)
    if shift_type == "offset":
        def b_regime(b, rng):
            return task.sample(b, *cfg.regimeB_v0, rng=rng, delta_step=1)
    else:  # scale: same v0 range, larger increments -> delta-visible
        def b_regime(b, rng):
            return task.sample(b, *cfg.regimeA_v0, rng=rng, delta_step=cfg.scale_step)
    return a_regime, b_regime


def run_arm(arm, cfg):
    shift_type, encoding, mode = arm.split("_")
    max_step = cfg.scale_step if shift_type == "scale" else 1
    task = ShiftSeq(cfg.length, cfg.n_deltas, max_step=max_step)
    torch.manual_seed(cfg.seed)

    def make_expert():
        return FixedDepthTransformer(FixedDepthConfig(
            vocab_size=task.target_classes, dim=cfg.dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            max_seq=task.seq_len, pos_mode="pope", continuous_input=True)).to(cfg.device)

    model = PartitionedModel(make_expert, mode=mode, k=cfg.surprise_k, cooldown=cfg.surprise_cooldown)
    opt = torch.optim.AdamW(model.experts[0].parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def add_expert(expert):
        opt.add_param_group({"params": list(expert.parameters())})

    a_regime, b_regime = _samplers(shift_type, cfg, task)
    rng = random.Random(cfg.seed)
    eval_rng = random.Random(cfg.seed + 999)
    total = 2 * cfg.phase_steps
    history = []
    model.train()
    for step in range(1, total + 1):
        phase = "A" if step <= cfg.phase_steps else "B"
        regime = a_regime if phase == "A" else b_regime
        if mode == "oracle" and step == cfg.phase_steps + 1 and model.n_experts() == 1:
            add_expert(model.allocate())                      # known boundary
        b = regime(cfg.batch_size, rng).to(cfg.device)
        loss = ce_loss(model(_input(encoding, b)), b.target, b.loss_mask)
        allocated = model.observe(loss.item())                # surprise: detect BEFORE training,
        if allocated is not None:                             # so the old expert isn't corrupted
            add_expert(allocated)                             # new expert trains from next batch
        else:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * _wsd_mult(step, total)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.experts[model.active].parameters(), 1.0)
            opt.step()
        if step in EARLY_EVALS or step % cfg.eval_every == 0 or step == cfg.phase_steps or step == total:
            model.eval()
            b_expert = min(1, model.n_experts() - 1)          # oracle routing: A->0, B->allocated
            acc_a = accuracy(model, encoding, a_regime(cfg.eval_batch, eval_rng).to(cfg.device), expert=0)
            acc_b = accuracy(model, encoding, b_regime(cfg.eval_batch, eval_rng).to(cfg.device), expert=b_expert)
            model.train()
            history.append({"step": step, "phase": phase, "acc_A": acc_a, "acc_B": acc_b,
                            "n_experts": model.n_experts()})
    end_of_A = [h["acc_A"] for h in history if h["step"] <= cfg.phase_steps][-1]
    end = history[-1]["acc_A"]
    return {"history": history, "acc_A_end_of_A": end_of_A, "acc_A_final": end,
            "forgetting": end_of_A - end, "acc_B_final": history[-1]["acc_B"],
            "n_experts": model.n_experts()}


def run_continual(cfg: ContinualConfig) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.n_layers = 32, 2
        cfg.phase_steps, cfg.batch_size, cfg.eval_batch, cfg.eval_every = 2, 16, 16, 2
        cfg.device = "cpu"

    print(f"[continual] A={cfg.regimeA_v0} -> B offset={cfg.regimeB_v0} / scale x{cfg.scale_step} "
          f"| phase_steps={cfg.phase_steps} | device={cfg.device}")
    result = {"arms": {}}
    for arm in cfg.arms:
        r = run_arm(arm, cfg)
        result["arms"][arm] = r
        print(f"[{arm:24}] acc_A {r['acc_A_end_of_A']:.3f}->{r['acc_A_final']:.3f}  "
              f"forget {r['forgetting']:+.3f}  acc_B {r['acc_B_final']:.3f}  experts {r['n_experts']}")

    print("\n[ablation] forgetting by mode (single / oracle / surprise) -- lower = better retention:")
    for shift in ("offset", "scale"):
        for enc in ("absolute", "delta"):
            cells = []
            for m in ("single", "oracle", "surprise"):
                a = f"{shift}_{enc}_{m}"
                if a in result["arms"]:
                    r = result["arms"][a]
                    cells.append(f"{m}:{r['forgetting']:+.3f}(e{r['n_experts']})")
            if cells:
                print(f"  {shift:7} {enc:9}  " + "  ".join(cells))

    if not cfg.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"continual_seed{cfg.seed}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[continual] wrote metrics to {path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Step 4: novelty-partition continual learning")
    p.add_argument("--phase_steps", type=int, default=1500)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run_continual(ContinualConfig(phase_steps=a.phase_steps, arms=tuple(a.arms), seed=a.seed,
                                  smoke=a.smoke))


if __name__ == "__main__":
    main()
