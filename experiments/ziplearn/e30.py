"""E30 — the depth loop under attention residuals (DESIGN §18; pre-registered 2026-09-21). The gradient arm, for insight
only: which routes across passes a looped transformer learns when it may read any earlier pass, whether that beats the
fixed boundary operator (norm(last pass) + alpha * anchor), and whether either extrapolates to more passes than trained.

Cells, all RoPE, 3200 steps, seed 0, E0's task and budget, two loop shapes each (tied K = 4; untied growth 2 -> 4, E26's
best): (a) the fixed boundary operator; (b) attention residuals across passes with the mixers tied to the core (routing by
content); (c) one mixer query per pass; (d) attention residuals windowed to the anchor + the last 2 passes. Measured:
trained / held-out compositions solved and mean accuracy (E26: 13/17, held-out 0.12-0.20); the routes (mean weight per
source at each pass); accuracy at 2x the trained passes. Pass: some attention-residual cell at or above E26's best held-out
accuracy with routes that are not uniform. Refute: every attention-residual cell below the fixed operator.

    python experiments/ziplearn/e30.py           (GPU, ~20 min; run detached) -> runs/e30/summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
H1 = HERE.parent / "transformers" / "h1_lid.py"
SHAPES = {"tied_k4": ["--loops", "4"], "untied_grow2to4": ["--loops", "4", "--untied", "1", "--grow_at", "0.5"]}
SCHEMES = {"fixed": ["--loop_res", "std"],
           "attnres_tied": ["--loop_res", "attnres", "--mix", "tied"],
           "attnres_per_pass": ["--loop_res", "attnres", "--mix", "per_pass"],
           "attnres_window2": ["--loop_res", "attnres", "--mix", "tied", "--window", "2"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--out", default=str(HERE / "runs" / "e30"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for shape, sflags in SHAPES.items():
        for scheme, rflags in SCHEMES.items():
            name = f"{scheme}__{shape}"
            js = out / f"{name}.json"
            if js.exists():
                results[name] = json.load(open(js))
                print(f"{name}: reusing", flush=True)
                continue
            t0 = time.time()
            cmd = [sys.executable, str(H1), "--pos", "rope", "--res", "loop", "--steps", str(args.steps), "--extrap", "2",
                   "--json", str(js)] + sflags + rflags
            with open(out / f"{name}.log", "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(H1.parent)).returncode
            if rc != 0 or not js.exists():
                print(f"{name}: FAILED rc={rc}", flush=True)
                continue
            r = json.load(open(js))
            results[name] = r
            ex = r.get("extrap") or {}
            print(f"{name}: {time.time() - t0:.0f}s | trained {r['trained_solved']}/{r['n_trained']} (acc {r['trained_mean_acc']:.2f}) | "
                  f"held-out {r['held_solved']}/{r['n_held']} (acc {r['held_mean_acc']:.2f}) | at {ex.get('passes')} passes: trained acc "
                  f"{ex.get('trained_mean_acc', float('nan')):.2f}, held-out acc {ex.get('held_mean_acc', float('nan')):.2f} | loss {r['train_loss']:.3f}", flush=True)
            if r.get("routes"):
                for mname, calls in r["routes"].items():
                    last = calls[-1]
                    print(f"      {mname:<14} last call: " + " ".join(f"{x:.2f}" for x in last), flush=True)
    json.dump(results, open(out / "summary.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
