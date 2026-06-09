"""Stage 0 experiment harness -- the settling core vs a matched fixed-depth baseline.

What this runs (implementation_plan.md §2, training_environment.md §2.1):
  * Trains ``SettlingLM`` on ModularChain (compositional reasoning, difficulty = chain length).
  * Logs the Risk-1 convergence diagnostics: convergence rate, iterations distribution,
    oscillation rate, difficulty->iterations correlation (adaptive compute), basin
    consistency (spurious attractors).
  * Evaluates in-distribution and OOD-by-difficulty accuracy.
  * Optionally trains a parameter-matched fixed-depth transformer for the gate:
    does recurrent depth beat fixed depth at equal params and equal data?

THIS DOES NOT RUN ON IMPORT. Run it on the GPU (never train on the dev machine --
Mistake #36):

    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_stage0.py --steps 4000 --baseline

A tiny CPU smoke path exists for tests only: run_stage0(Stage0Config(smoke=True)).
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

from baselines import matched_baseline  # noqa: E402
from core.deq import DEQConfig  # noqa: E402
from core.model import SettlingLM, SettlingLMConfig, count_parameters  # noqa: E402
from probes import ConvergenceMonitor, basin_consistency  # noqa: E402
from tasks import ModularChain  # noqa: E402


@dataclass
class Stage0Config:
    # task
    modulus: int = 7
    n_ops: int = 8
    max_len: int = 8
    train_len_min: int = 1
    train_len_max: int = 4
    ood_len_min: int = 5
    ood_len_max: int = 6
    # model
    dim: int = 128
    n_heads: int = 4
    segments: int = 2          # deep-supervision segments
    deq_max_iter: int = 40
    deq_tol: float = 1e-3
    grad_mode: str = "one_step"   # one_step | unrolled | bptt | ift
    bptt_iters: int = 12          # iterations for grad_mode=bptt
    state_norm: bool = False      # RMS-normalize state each iteration (contraction aid)
    pos_mode: str = "pope"        # pope | rope | learned (positional scheme, both arms)
    warm_start: str = "zeros"     # zeros | input | proposal (settle init; Solve-the-Loop)
    residual_gate: bool = False   # LayerScale contraction gate (helps forward converge -> IFT)
    gate_init: float = 0.1
    # training
    steps: int = 2000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 200
    eval_batch: int = 256
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # baseline
    baseline: bool = False
    baseline_layers: int = 6
    # outputs: metrics JSON is written here. "diagnostics" is git-ignored
    # (experiments/*/diagnostics/) so run artifacts never get committed.
    out_dir: str = "diagnostics"
    # smoke (tests only): tiny + 2 steps, no real training
    smoke: bool = False


def lm_loss(seg_logits: list[torch.Tensor], targets, mask) -> torch.Tensor:
    """Mean over deep-supervision segments of masked next-token cross-entropy."""
    v = seg_logits[0].shape[-1]
    denom = mask.sum().clamp_min(1.0)
    total = seg_logits[0].new_zeros(())
    for logits in seg_logits:
        ce = F.cross_entropy(logits.reshape(-1, v), targets.reshape(-1), reduction="none")
        total = total + (ce.reshape(mask.shape) * mask).sum() / denom
    return total / len(seg_logits)


@torch.no_grad()
def accuracy(logits, targets, mask) -> float:
    pred = logits.argmax(dim=-1)
    correct = ((pred == targets).float() * mask).sum()
    return (correct / mask.sum().clamp_min(1.0)).item()


@torch.no_grad()
def adaptive_compute_probe(model, task, cfg: Stage0Config, rng) -> dict:
    """Risk-1 adaptive-compute test: do iterations-to-converge rise with difficulty?

    Solve a pure-difficulty-n batch for each n and correlate n with the solver's
    iteration count. Positive correlation = the core spends more compute on harder
    inputs (the central justification for recurrent depth).
    """
    mon = ConvergenceMonitor()
    diffs: list[float] = []
    series: list[tuple[int, int]] = []
    for n in range(1, task.max_len + 1):
        batch = task.sample(64, n, n, rng).to(cfg.device)
        _, info = model.deq(model._inject(batch.input_ids))
        mon.record(info)
        diffs.append(float(n))
        series.append((n, info.iters))
    return {"series": series, "iters_vs_difficulty_r": mon.difficulty_correlation(diffs)}


@torch.no_grad()
def evaluate(model, task, cfg: Stage0Config, rng) -> dict:
    model.eval()
    dev = cfg.device
    # in-distribution
    b_in = task.sample(cfg.eval_batch, cfg.train_len_min, cfg.train_len_max, rng).to(dev)
    logits_in, _, infos_in = model(b_in.input_ids)
    # out-of-distribution by difficulty
    b_ood = task.sample(cfg.eval_batch, cfg.ood_len_min, cfg.ood_len_max, rng).to(dev)
    logits_ood, _, _ = model(b_ood.input_ids)

    # Risk-1 convergence diagnostics over the in-dist eval batch
    mon = ConvergenceMonitor()
    for info in infos_in:
        mon.record(info)
    # iterations-vs-difficulty correlation needs per-example iters; recompute
    # one solve per example would be costly, so report the summary + a basin probe
    basin = basin_consistency(model.deq, model._inject(b_in.input_ids[:8]), n_restarts=4)

    adaptive = adaptive_compute_probe(model, task, cfg, rng)

    model.train()
    return {
        "acc_in": accuracy(logits_in, b_in.targets, b_in.loss_mask),
        "acc_ood": accuracy(logits_ood, b_ood.targets, b_ood.loss_mask),
        "convergence": mon.summary(),
        "basin": basin,
        "adaptive": adaptive,
    }


def build_model(cfg: Stage0Config, vocab: int, max_seq: int) -> SettlingLM:
    lm_cfg = SettlingLMConfig(
        vocab_size=vocab, dim=cfg.dim, n_heads=cfg.n_heads, max_seq=max_seq,
        n_supervision_segments=cfg.segments, pos_mode=cfg.pos_mode,
        warm_start=cfg.warm_start, residual_gate=cfg.residual_gate, gate_init=cfg.gate_init,
        deq=DEQConfig(
            max_iter=cfg.deq_max_iter, tol=cfg.deq_tol, grad_mode=cfg.grad_mode,
            bptt_iters=cfg.bptt_iters, state_norm=cfg.state_norm,
        ),
    )
    return SettlingLM(lm_cfg)


def _eval_accuracy(model, forward, task, cfg: Stage0Config, rng) -> tuple[float, float]:
    """In-dist + OOD accuracy for any model exposing ``forward(model, ids) -> [logits]``."""
    model.eval()
    dev = cfg.device
    with torch.no_grad():
        b_in = task.sample(cfg.eval_batch, cfg.train_len_min, cfg.train_len_max, rng).to(dev)
        b_ood = task.sample(cfg.eval_batch, cfg.ood_len_min, cfg.ood_len_max, rng).to(dev)
        ai = accuracy(forward(model, b_in.input_ids)[-1], b_in.targets, b_in.loss_mask)
        ao = accuracy(forward(model, b_ood.input_ids)[-1], b_ood.targets, b_ood.loss_mask)
    model.train()
    return ai, ao


def _fit(model, forward, task, cfg: Stage0Config, rng, label: str, full_eval: bool) -> list[dict]:
    """Train one model. ``forward(model, ids)`` returns a list of segment logits
    (settling: deep-supervision segments; baseline: a single-element list).
    ``full_eval`` toggles the settling-only convergence diagnostics."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict] = []
    model.train()
    for step in range(1, cfg.steps + 1):
        batch = task.sample(cfg.batch_size, cfg.train_len_min, cfg.train_len_max, rng).to(cfg.device)
        seg_logits = forward(model, batch.input_ids)
        loss = lm_loss(seg_logits, batch.targets, batch.loss_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            if full_eval:
                m = evaluate(model, task, cfg, rng)
            else:
                ai, ao = _eval_accuracy(model, forward, task, cfg, rng)
                m = {"acc_in": ai, "acc_ood": ao}
            m["step"] = step
            m["loss"] = loss.item()
            history.append(m)
            if full_eval:
                c = m["convergence"]
                r = m["adaptive"]["iters_vs_difficulty_r"]
                print(f"[{label}] step {step:>5} loss {m['loss']:.3f} "
                      f"acc_in {m['acc_in']:.3f} acc_ood {m['acc_ood']:.3f} "
                      f"| conv {c['convergence_rate']:.2f} iters~{c['iters_mean']:.1f} "
                      f"osc {c['oscillation_rate']:.2f} adaptive_r {r:+.2f} "
                      f"basin_dist {m['basin']['mean_pairwise_rel_dist']:.3f}")
            else:
                print(f"[{label}] step {step:>5} loss {m['loss']:.3f} "
                      f"acc_in {m['acc_in']:.3f} acc_ood {m['acc_ood']:.3f}")
    return history


def run_stage0(cfg: Stage0Config) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.n_heads, cfg.segments = 32, 4, 2
        cfg.steps, cfg.batch_size, cfg.eval_batch = 2, 16, 16
        cfg.deq_max_iter, cfg.eval_every, cfg.device = 8, 2, "cpu"

    torch.manual_seed(cfg.seed)
    task = ModularChain(cfg.modulus, cfg.n_ops, cfg.max_len, seed=cfg.seed)

    # --- settling core (full convergence diagnostics) ---
    model = build_model(cfg, task.vocab_size, task.seq_len).to(cfg.device)
    n_params = count_parameters(model)
    print(f"[stage0] settling model: {n_params:,} params | device={cfg.device} "
          f"| grad_mode={cfg.grad_mode} state_norm={cfg.state_norm} pos={cfg.pos_mode} "
          f"warm={cfg.warm_start} gate={cfg.residual_gate}")
    hist_s = _fit(model, lambda m, ids: m(ids)[1], task, cfg,
                  random.Random(cfg.seed), "settling", full_eval=True)
    result = {"settling": {"n_params": n_params, "history": hist_s,
                           "final": hist_s[-1] if hist_s else {}}}

    # --- matched fixed-depth baseline: actually train it, then compare (the gate) ---
    if cfg.baseline:
        bl, bl_cfg, bl_params = matched_baseline(
            n_params, vocab_size=task.vocab_size, n_heads=cfg.n_heads,
            n_layers=cfg.baseline_layers, max_seq=task.seq_len, pos_mode=cfg.pos_mode,
            residual_gate=cfg.residual_gate, gate_init=cfg.gate_init,
        )
        bl = bl.to(cfg.device)
        print(f"[stage0] matched baseline: {bl_params:,} params "
              f"({cfg.baseline_layers} layers, dim={bl_cfg.dim})")
        # same seed => identical training batches => an equal-data comparison
        hist_b = _fit(bl, lambda m, ids: [m(ids)], task, cfg,
                      random.Random(cfg.seed), "baseline", full_eval=False)
        result["baseline"] = {"n_params": bl_params, "history": hist_b,
                              "final": hist_b[-1] if hist_b else {}}
        s, b = result["settling"]["final"], result["baseline"]["final"]
        if s and b:
            delta = s["acc_ood"] - b["acc_ood"]
            verdict = ("settling wins" if delta > 0.02
                       else "baseline wins" if delta < -0.02
                       else "TIE -- recurrent depth buys nothing")
            print(f"[gate] OOD acc  settling {s['acc_ood']:.3f}  vs  baseline {b['acc_ood']:.3f}"
                  f"  => delta {delta:+.3f}  [{verdict}]")

    # write the run artifact to the (git-ignored) outputs dir -- not during smoke
    if not cfg.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.out_dir)
        os.makedirs(out, exist_ok=True)
        tag = (f"{cfg.grad_mode}{'_sn' if cfg.state_norm else ''}"
               f"_{cfg.pos_mode}{'' if cfg.warm_start == 'zeros' else '_warm-' + cfg.warm_start}"
               f"{'_gate' if cfg.residual_gate else ''}_seed{cfg.seed}")
        path = os.path.join(out, f"stage0_{tag}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[stage0] wrote metrics to {path}")

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 0: settling core vs fixed-depth baseline")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--segments", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--deq_max_iter", type=int, default=40)
    p.add_argument("--grad_mode", choices=["one_step", "unrolled", "bptt", "ift"], default="one_step")
    p.add_argument("--bptt_iters", type=int, default=12)
    p.add_argument("--state_norm", action="store_true", help="RMS-normalize state each iteration")
    p.add_argument("--pos", choices=["pope", "rope", "learned"], default="pope", help="positional scheme")
    p.add_argument("--warm_start", choices=["zeros", "input", "proposal"], default="zeros",
                   help="settle init (Solve-the-Loop warm-start)")
    p.add_argument("--residual_gate", action="store_true",
                   help="LayerScale contraction gate on attn+ffn residuals (helps IFT)")
    p.add_argument("--gate_init", type=float, default=0.1)
    p.add_argument("--baseline", action="store_true", help="also build a param-matched baseline")
    p.add_argument("--baseline_layers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny CPU smoke run (no real training)")
    a = p.parse_args()
    cfg = Stage0Config(
        steps=a.steps, dim=a.dim, n_heads=a.heads, segments=a.segments,
        batch_size=a.batch_size, lr=a.lr, deq_max_iter=a.deq_max_iter,
        grad_mode=a.grad_mode, bptt_iters=a.bptt_iters, state_norm=a.state_norm,
        pos_mode=a.pos, warm_start=a.warm_start,
        residual_gate=a.residual_gate, gate_init=a.gate_init,
        baseline=a.baseline, baseline_layers=a.baseline_layers, seed=a.seed, smoke=a.smoke,
    )
    run_stage0(cfg)


if __name__ == "__main__":
    main()
