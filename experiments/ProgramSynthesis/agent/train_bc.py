"""Phase 1 — behavior-cloning sweep over binding channels.

The supervised learnability microscope (LEARNING_AGENT.md §1, §5): train each
binding arm to imitate the BFS oracle, holding the trunk/optimizer/data fixed, and
log per-step the train-layout and held-out-layout accuracy. This isolates:

  P1  channel dominance     — which binding reaches competence fastest / highest
  P2  shift invisibility    — does the train-vs-held-out gap stay ~0 (pope) or open
                              (content/none) on unseen layouts of the same mechanic

Compose mode (train on the constituent mechanics, test on `compose`) is P3 — the
A∘B composition test for a learned agent.

This is a GPU job. Run, e.g.:
    python -m agent.train_bc --train-mechanics key_door --steps 4000
    python -m agent.train_bc --train-mechanics nav,key_door,block_pad \
        --test-mechanic compose --steps 6000        # P3 composition transfer
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from agent.dataset import build_dataset, materialize
from agent.encoders import BINDINGS
from agent.layouts import Layout, sample_layouts
from agent.metrics import evaluate_tensors, time_to_threshold
from agent.trunk import build_model

torch.set_float32_matmul_precision("high")  # enable TF32 for any fp32 matmuls


def make_layout_splits(
    train_mechanics: List[str],
    test_mechanic: str,
    n_train: int,
    n_test: int,
    seed: int,
) -> Tuple[List[Layout], List[Layout]]:
    """Train layouts (union over train mechanics) and disjoint held-out test layouts.

    If `test_mechanic` is one of the train mechanics, its pool is sampled once and
    split so train/test are disjoint position-configs (the P2 shift test). Otherwise
    test layouts are sampled independently (the P3 composition transfer).
    """
    train_layouts: List[Layout] = []
    test_layouts: Optional[List[Layout]] = None
    for m in train_mechanics:
        if m == test_mechanic:
            pool = sample_layouts(m, n_train + n_test, seed=seed)
            train_layouts += pool[:n_train]
            test_layouts = pool[n_train:]
        else:
            train_layouts += sample_layouts(m, n_train, seed=seed)
    if test_layouts is None:
        test_layouts = sample_layouts(test_mechanic, n_test, seed=seed + 9973)
    return train_layouts, test_layouts


def train_one(
    binding: str,
    train_ds,
    test_ds,
    *,
    steps: int,
    batch: int,
    lr: float,
    device: str,
    eval_every: int,
    window: int,
    seed: int,
    patch: int = 2,
    crop: int = 16,
    use_compile: bool = False,
    amp: bool = True,
) -> List[Dict[str, float]]:
    """Train one binding arm; return the eval history."""
    torch.manual_seed(seed)                       # same init seed across arms
    model = build_model(binding, window=window, patch=patch, crop=crop).to(device)
    if use_compile and device == "cuda":
        # Plain fusing compile, NOT mode="reduce-overhead": CUDA graphs re-record per
        # distinct input shape, and the varying eval-chunk sizes across seeds caused a
        # graph-recording storm (the multi-seed slowdown). Fusion alone is robust.
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = device == "cuda" and amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Stage the whole (small) dataset on-device and batch by random index. For a
    # model this size the DataLoader/CPU path dominates and starves the GPU; this
    # removes it entirely.
    ftr, atr, ttr = (t.to(device) for t in materialize(train_ds))
    fte, ate, tte = (t.to(device) for t in materialize(test_ds))
    n_tr = ttr.shape[0]
    bsz = min(batch, n_tr)

    history: List[Dict[str, float]] = []
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            tr = evaluate_tensors(model, ftr, atr, ttr, device)
            te = evaluate_tensors(model, fte, ate, tte, device)
            history.append({
                "step": step,
                "train_loss": round(tr["loss"], 4),
                "train_masked": round(tr["masked_acc"], 4),
                "test_masked": round(te["masked_acc"], 4),
                "train_acc": round(tr["acc"], 4),
                "test_acc": round(te["acc"], 4),
                "gap": round(tr["masked_acc"] - te["masked_acc"], 4),
            })
        if step == steps:
            break
        idx = torch.randint(0, n_tr, (bsz,), device=device)
        fw, aw, tg = ftr[idx].long(), atr[idx], ttr[idx]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(fw, aw)
            loss = F.cross_entropy(logits, tg)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return history


def run_sweep(
    bindings: List[str],
    train_ds,
    test_ds,
    *,
    tag: str = "",
    **kw,
) -> Dict[str, List[Dict[str, float]]]:
    results = {}
    prefix = f"[{tag}] " if tag else ""
    for binding in bindings:
        history = train_one(binding, train_ds, test_ds, **kw)
        last = history[-1]
        tt90 = time_to_threshold(history, "test_masked", 0.9)
        print(f"  {prefix}{binding:9s} held-out {last['test_masked']:.3f}  "
              f"gap {last['gap']:+.3f}  tt90 {tt90}")
        results[binding] = history
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 binding-channel BC sweep")
    ap.add_argument("--train-mechanics", default="key_door",
                    help="comma-separated: nav,key_door,block_pad,compose")
    ap.add_argument("--test-mechanic", default=None,
                    help="default = first train mechanic (held-out positions); "
                         "set to 'compose' for the P3 composition transfer")
    ap.add_argument("--bindings", default=",".join(BINDINGS))
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--steps", type=int, default=1500)   # ~270 epochs; was 700-epoch overkill
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--crop", type=int, default=16,
                    help="crop frames to top-left crop×crop (LockPath boards fit 16)")
    ap.add_argument("--compile", action="store_true",
                    help="wrap the model in torch.compile (CUDA graphs) — ~2x+")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable fp16 autocast (faster if casts dominate the profile)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="average over this many seeds (seed, seed+1, ...) with fresh "
                         "layouts + init each, for error bars")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="path to write the history JSON")
    args = ap.parse_args()

    train_mechanics = [m.strip() for m in args.train_mechanics.split(",") if m.strip()]
    test_mechanic = args.test_mechanic or train_mechanics[0]
    bindings = [b.strip() for b in args.bindings.split(",") if b.strip()]

    print(f"device={args.device} | train={train_mechanics} test={test_mechanic} | "
          f"bindings={bindings} | seeds={args.seeds}")

    common = dict(
        steps=args.steps, batch=args.batch, lr=args.lr, device=args.device,
        eval_every=args.eval_every, window=args.window,
        patch=args.patch, crop=args.crop,
        use_compile=args.compile, amp=not args.no_amp,
    )

    seed_finals: Dict[str, List[float]] = {b: [] for b in bindings}
    seed0_results = None
    for s in range(args.seeds):
        seed = args.seed + s
        train_layouts, test_layouts = make_layout_splits(
            train_mechanics, test_mechanic, args.n_train, args.n_test, seed
        )
        train_ds = build_dataset(train_layouts, window=args.window)
        test_ds = build_dataset(test_layouts, window=args.window)
        if s == 0:
            print(f"train: {len(train_layouts)} layouts -> {len(train_ds)} decisions | "
                  f"held-out: {len(test_layouts)} layouts -> {len(test_ds)} decisions")
        tag = f"seed {seed}" if args.seeds > 1 else ""
        results = run_sweep(bindings, train_ds, test_ds, tag=tag, seed=seed, **common)
        for b in bindings:
            seed_finals[b].append(round(results[b][-1]["test_masked"], 4))
        if s == 0:
            seed0_results = results

    print(f"\n=== held-out masked acc: mean +/- std over {args.seeds} seed(s) ===")
    for b in bindings:
        vals = seed_finals[b]
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {b:9s} {m:.3f} +/- {sd:.3f}   [{' '.join(f'{v:.3f}' for v in vals)}]")

    out = args.out or os.path.join(
        os.path.dirname(__file__), "runs",
        f"bc_{'+'.join(train_mechanics)}_to_{test_mechanic}.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "config": vars(args),
            "train_mechanics": train_mechanics,
            "test_mechanic": test_mechanic,
            "seed_finals": seed_finals,
            "results": seed0_results,
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
