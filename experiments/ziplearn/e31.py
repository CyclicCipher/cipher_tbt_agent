"""E31 — continuous thoughts under attention residuals (DESIGN §18; pre-registered 2026-09-21). The gradient arm, for
insight only. Cells: c in {1, 2} thoughts per demonstration x {plain, attention residuals} x {raw feedback, the boundary
operator on the fed-back state}, all with Coconut's curriculum; plus the paper's negative control, no curriculum (c = 1,
attention residuals, operator). Measured: trained / held-out compositions solved and mean accuracy in thought mode
(E26's best held-out 0.20; E30's); the chain-of-thought ceiling; loss spikes after the stage switch; the probe of the
thought for the intermediate digits. Pass: held-out accuracy above 0.20 in some cell, and the operator removing the
instability where it appears. Refute: no cell above the looped model.

    python experiments/ziplearn/e31.py           (GPU; run detached) -> runs/e31/summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CC = HERE.parent / "transformers" / "coconut.py"
CELLS = [(f"c1_{res}_{fb}", ["--c", "1", "--res", res, "--feedback", fb]) for res in ("std", "attnres") for fb in ("raw", "operator")]
CELLS.append(("c1_attnres_operator_nocurriculum", ["--c", "1", "--res", "attnres", "--feedback", "operator", "--curriculum", "0"]))
CELLS += [(f"c2_{res}_{fb}", ["--c", "2", "--res", res, "--feedback", fb]) for res in ("std", "attnres") for fb in ("raw", "operator")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--only", default="", help="comma-separated cell names to run (default: all)")
    ap.add_argument("--out", default=str(HERE / "runs" / "e31"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    results = {}
    for name, flags in CELLS:
        if only and name not in only:
            continue
        js = out / f"{name}.json"
        if js.exists():
            results[name] = json.load(open(js))
            print(f"{name}: reusing", flush=True)
            continue
        t0 = time.time()
        cmd = [sys.executable, str(CC), "--steps", str(args.steps), "--json", str(js)] + flags
        with open(out / f"{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(CC.parent)).returncode
        if rc != 0 or not js.exists():
            print(f"{name}: FAILED rc={rc}", flush=True)
            continue
        r = json.load(open(js))
        results[name] = r
        th, cot = r["results"]["thought"], r["results"]["cot"]
        pr = r["probe"]
        print(f"{name}: {time.time() - t0:.0f}s | thought: trained {th['trained']['solved']}/{th['trained']['n']} (acc {th['trained']['mean_acc']:.2f}), "
              f"held-out {th['held']['solved']}/{th['held']['n']} (acc {th['held']['mean_acc']:.2f})"
              + (f" | cot: trained acc {cot['trained']['mean_acc']:.2f}, held-out acc {cot['held']['mean_acc']:.2f}" if cot else "")
              + f" | spikes {r['spikes']} | probe " + ", ".join(f"t{p['thought']}: held {p['held'][0]:.2f} (+{p['held'][1]:.1f})" for p in pr), flush=True)
    json.dump(results, open(out / "summary.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
