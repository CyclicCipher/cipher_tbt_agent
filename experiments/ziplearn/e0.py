"""E0 — substrate check (DESIGN.md §11): does the substrate of §15 (PoPE with position-free channels; attention residuals)
break the gradient-trained comparison arm on the composition task, and what routes does it learn?

Four cells, one seed each, the same 3200-step budget the recorded RoPE/PoPE runs used (transformers/NOTES.md):
    {RoPE, PoPE + 2 zero channels} x {standard residual, attention residuals (one transformer block per AttnRes block)}
Each cell is one `h1_lid.py` run; its headline numbers and learned routes land in runs/e0/<cell>.json and the summary
in runs/e0/summary.json. Not a verdict on the design: 0/8 held-out is expected and fails nothing.

    python experiments/ziplearn/e0.py            (~2 min per cell on the GPU, run detached)
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
CELLS = [("rope_std", ["--pos", "rope", "--res", "std"]),
         ("pope0_std", ["--pos", "pope", "--n_zero", "2", "--res", "std"]),
         ("rope_attnres", ["--pos", "rope", "--res", "attnres"]),
         ("pope0_attnres", ["--pos", "pope", "--n_zero", "2", "--res", "attnres"]),
         # attribution cells, added after the first four ran: PoPE WITHOUT the zero channels, same seed
         ("pope_std", ["--pos", "pope", "--res", "std"]),
         ("pope_attnres", ["--pos", "pope", "--res", "attnres"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e0"))
    ap.add_argument("--only", default="", help="comma-separated cell names to run (default: all four)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else {c for c, _ in CELLS}
    results = {}
    for name, flags in CELLS:
        if name not in wanted:
            continue
        js = out / f"{name}.json"
        if js.exists():
            results[name] = json.load(open(js))
            print(f"{name}: reusing {js}", flush=True)
            continue
        t0 = time.time()
        cmd = [sys.executable, str(H1), "--steps", str(args.steps), "--seed", str(args.seed), "--json", str(js)] + flags
        with open(out / f"{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(H1.parent)).returncode
        if rc != 0 or not js.exists():
            print(f"{name}: FAILED rc={rc}; see {out / (name + '.log')}", flush=True)
            continue
        results[name] = json.load(open(js))
        r = results[name]
        print(f"{name}: {time.time() - t0:.0f}s | trained solved {r['trained_solved']}/{r['n_trained']} "
              f"(mean acc {r['trained_mean_acc']:.2f}) | held-out solved {r['held_solved']}/{r['n_held']} "
              f"(mean acc {r['held_mean_acc']:.2f}) | loss {r['train_loss']:.3f}", flush=True)
    json.dump(results, open(out / "summary.json", "w"), indent=1)
    print("\n| cell | trained solved | trained mean acc | held-out solved | held-out mean acc | final loss |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['trained_solved']}/{r['n_trained']} | {r['trained_mean_acc']:.2f} | "
              f"{r['held_solved']}/{r['n_held']} | {r['held_mean_acc']:.2f} | {r['train_loss']:.3f} |")
    for name, r in results.items():
        if r.get("routes"):
            print(f"\nroutes, {name} (mean weight per source; sources = embedding, block 1, block 2, ..., current partial sum):")
            for mixer, a in r["routes"].items():
                print(f"   {mixer:<20} " + " ".join(f"{x:.2f}" for x in a))
    print("done", flush=True)


if __name__ == "__main__":
    main()
