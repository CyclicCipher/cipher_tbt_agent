"""E26 — the looped transformer on the composition task (after Chen et al., arXiv:2609.19107).

The gradient arm as a looped transformer: a prelude block, a core applied K times with the boundary operator between
passes (RMSNorm the stream, re-inject the prelude's output), a coda block. Cells, all RoPE, 3200 steps, seed 0, the
same task and budget as E0: vanilla d3 (E0's run); tied loops K = 2, 4, 6 (executed depth 4, 6, 8 with the parameters
of three blocks); untied K = 4 (a six-block stack with the operator); tied and untied growth 2 -> 4 at half of training.
Measured: trained compositions solved and mean accuracy; held-out compositions solved and mean accuracy (E0: 0/8
everywhere). Pre-registered: more executed depth raises trained solve rate; any cell solving a held-out composition
is the result to look for. Refute: no cell better than vanilla.

    python experiments/ziplearn/e26.py           (GPU, ~15 min; run detached) -> runs/e26/summary.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
H1 = HERE.parent / "transformers" / "h1_lid.py"
CELLS = [("loop_tied_k2", ["--res", "loop", "--loops", "2"]),
         ("loop_tied_k4", ["--res", "loop", "--loops", "4"]),
         ("loop_tied_k6", ["--res", "loop", "--loops", "6"]),
         ("loop_untied_k4", ["--res", "loop", "--loops", "4", "--untied", "1"]),
         ("loop_tied_grow2to4", ["--res", "loop", "--loops", "4", "--grow_at", "0.5"]),
         ("loop_untied_grow2to4", ["--res", "loop", "--loops", "4", "--untied", "1", "--grow_at", "0.5"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--out", default=str(HERE / "runs" / "e26"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    vanilla = HERE / "runs" / "e0" / "rope_std.json"
    if vanilla.exists():
        shutil.copy(vanilla, out / "vanilla_d3.json")
        results["vanilla_d3"] = json.load(open(vanilla))
        r = results["vanilla_d3"]
        print(f"vanilla_d3 (E0): trained {r['trained_solved']}/{r['n_trained']} (acc {r['trained_mean_acc']:.2f}) | "
              f"held-out {r['held_solved']}/{r['n_held']} (acc {r['held_mean_acc']:.2f})", flush=True)
    for name, flags in CELLS:
        js = out / f"{name}.json"
        if js.exists():
            results[name] = json.load(open(js))
            print(f"{name}: reusing", flush=True)
            continue
        t0 = time.time()
        cmd = [sys.executable, str(H1), "--pos", "rope", "--steps", str(args.steps), "--json", str(js)] + flags
        with open(out / f"{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(H1.parent)).returncode
        if rc != 0 or not js.exists():
            print(f"{name}: FAILED rc={rc}", flush=True)
            continue
        r = json.load(open(js))
        results[name] = r
        print(f"{name}: {time.time() - t0:.0f}s | trained {r['trained_solved']}/{r['n_trained']} (acc {r['trained_mean_acc']:.2f}) | "
              f"held-out {r['held_solved']}/{r['n_held']} (acc {r['held_mean_acc']:.2f}) | loss {r['train_loss']:.3f}", flush=True)
    json.dump(results, open(out / "summary.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
