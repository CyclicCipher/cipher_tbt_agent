"""PoPE-transformer experiment harness: test the item-52 SNR-gated optimizer.

Trains the *base PoPE transformer* (FixedDepthTransformer) on ModularChain, with
three optimizer arms on IDENTICAL data and identical init, for a clean A/B:
  * adamw         -- plain AdamW (control)
  * snr_ema       -- SNR gate, cheap across-step EMA variance (Litman & Guo 2026)
  * snr_faithful  -- SNR gate, true within-batch per-example variance (torch.func)

Logs train (in-dist) accuracy, OOD-by-length accuracy, loss, the fraction of
parameters the gate keeps, and the paper's validation-free generalization signal
(predicted population-risk-improvement rate). The question: does suppressing the
memorization (noise) directions narrow the train/test gap on the same task -- and
does the validation-free signal track real OOD?

THIS DOES NOT RUN ON IMPORT. GPU job (Mistake #36):

    ./venv/Scripts/python.exe experiments/RecurrentWorldModel/train_transformer.py --steps 4000

Tiny CPU smoke: run_transformer(TConfig(smoke=True)).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from core.model import count_parameters  # noqa: E402
from optim import SNRAdamW, per_example_snr_gate  # noqa: E402
from tasks import ModularChain  # noqa: E402

ARM_MODE = {"adamw": "none", "snr_ema": "ema", "snr_faithful": "faithful"}


@dataclass
class TConfig:
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
    n_layers: int = 6
    pos_mode: str = "pope"
    # training
    steps: int = 4000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    var_beta: float = 0.99
    gate_warmup: int = 100
    faithful_subsample: int = 32  # sub-batch size for the faithful variance estimate (0 = full)
    eval_every: int = 200
    eval_batch: int = 256
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    arms: tuple[str, ...] = ("adamw", "snr_ema", "snr_faithful")
    smoke: bool = False


def ce_loss(logits, targets, mask):
    v = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, v), targets.reshape(-1), reduction="none")
    return (ce.reshape(mask.shape) * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def accuracy(logits, targets, mask):
    pred = logits.argmax(dim=-1)
    return ((pred == targets).float() * mask).sum().item() / mask.sum().clamp_min(1.0).item()


def build(cfg: TConfig, vocab: int, max_seq: int) -> FixedDepthTransformer:
    return FixedDepthTransformer(FixedDepthConfig(
        vocab_size=vocab, dim=cfg.dim, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        max_seq=max_seq, pos_mode=cfg.pos_mode,
    ))


@torch.no_grad()
def evaluate(model, task, cfg, rng):
    model.eval()
    dev = cfg.device
    b_in = task.sample(cfg.eval_batch, cfg.train_len_min, cfg.train_len_max, rng).to(dev)
    b_ood = task.sample(cfg.eval_batch, cfg.ood_len_min, cfg.ood_len_max, rng).to(dev)
    ai = accuracy(model(b_in.input_ids), b_in.targets, b_in.loss_mask)
    ao = accuracy(model(b_ood.input_ids), b_ood.targets, b_ood.loss_mask)
    model.train()
    return ai, ao


def train_arm(arm: str, cfg: TConfig, task) -> list[dict]:
    torch.manual_seed(cfg.seed)                       # identical init across arms
    model = build(cfg, task.vocab_size, task.seq_len).to(cfg.device)
    opt = SNRAdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                   batch_size=cfg.batch_size, var_beta=cfg.var_beta,
                   mode=ARM_MODE[arm], gate_warmup=cfg.gate_warmup)
    rng = random.Random(cfg.seed)                     # identical data stream across arms
    eval_rng = random.Random(cfg.seed + 999)
    history: list[dict] = []
    model.train()
    for step in range(1, cfg.steps + 1):
        b = task.sample(cfg.batch_size, cfg.train_len_min, cfg.train_len_max, rng).to(cfg.device)
        logits = model(b.input_ids)
        loss = ce_loss(logits, b.targets, b.loss_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if ARM_MODE[arm] == "faithful":
            sub = cfg.faithful_subsample or None
            gate, risk = per_example_snr_gate(model, b.input_ids, b.targets, b.loss_mask,
                                              cfg.batch_size, subsample=sub)
            opt.set_external_gate(gate, risk)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ai, ao = evaluate(model, task, cfg, eval_rng)
            history.append({"step": step, "loss": loss.item(), "acc_in": ai, "acc_ood": ao,
                            "risk": opt.last_risk, "gate_frac": opt.last_gate_frac})
            print(f"[{arm:12}] step {step:>5} loss {loss.item():.3f} "
                  f"acc_in {ai:.3f} acc_ood {ao:.3f} "
                  f"gate {opt.last_gate_frac:.2f} risk {opt.last_risk:.3g}")
    return history


def run_transformer(cfg: TConfig) -> dict:
    if cfg.smoke:
        cfg.dim, cfg.n_layers = 32, 2
        cfg.steps, cfg.batch_size, cfg.eval_batch = 2, 16, 16
        cfg.eval_every, cfg.gate_warmup, cfg.device = 2, 1, "cpu"

    task = ModularChain(cfg.modulus, cfg.n_ops, cfg.max_len, seed=cfg.seed)
    torch.manual_seed(cfg.seed)
    n_params = count_parameters(build(cfg, task.vocab_size, task.seq_len))
    print(f"[transformer] PoPE, {n_params:,} params | device={cfg.device} | arms={cfg.arms}")

    result = {"n_params": n_params, "arms": {}}
    for arm in cfg.arms:
        result["arms"][arm] = {"history": train_arm(arm, cfg, task)}
        result["arms"][arm]["final"] = result["arms"][arm]["history"][-1] if result["arms"][arm]["history"] else {}

    # headline comparison
    print("\n[compare] final  acc_in / acc_ood  (validation-free risk):")
    for arm in cfg.arms:
        f = result["arms"][arm]["final"]
        if f:
            print(f"  {arm:12}  in {f['acc_in']:.3f}  ood {f['acc_ood']:.3f}  risk {f['risk']:.3g}")

    if not cfg.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"transformer_snr_seed{cfg.seed}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[transformer] wrote metrics to {path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="PoPE transformer: AdamW vs SNR-gated AdamW")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gate_warmup", type=int, default=100)
    p.add_argument("--faithful_subsample", type=int, default=32,
                   help="sub-batch for the faithful variance estimate (0 = full batch)")
    p.add_argument("--arms", nargs="+", default=["adamw", "snr_ema", "snr_faithful"],
                   choices=["adamw", "snr_ema", "snr_faithful"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cfg = TConfig(steps=a.steps, dim=a.dim, n_layers=a.layers, batch_size=a.batch_size,
                  lr=a.lr, gate_warmup=a.gate_warmup, faithful_subsample=a.faithful_subsample,
                  arms=tuple(a.arms), seed=a.seed, smoke=a.smoke)
    run_transformer(cfg)


if __name__ == "__main__":
    main()
