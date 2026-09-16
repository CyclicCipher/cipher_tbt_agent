"""The full matrix, resumable: every (arm, seed, size) run is a separate `train.py` process, then `evaluate.py`; runs
with a final checkpoint are skipped, runs with a latest checkpoint are resumed. Launch this detached and poll
`runs/matrix.log`:

    Windows:  start /b python experiments\\mdl_vs_ntp\\run_matrix.py > NUL
    POSIX:    nohup python experiments/mdl_vs_ntp/run_matrix.py &

    --tiers 1        (default 1,2,3)    --seeds 0,1,2    --budget_frac 0.333 (pilot)    --dry
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PY = sys.executable
TIERS = {
    1: [("ntp", "4M"), ("exit", "4M"), ("mdl", "4M")],
    2: [("curio", "4M"), ("mdl_nolib", "4M")],
    3: [("mdl_insample", "4M"), ("ntp", "1M"), ("mdl", "1M"), ("ntp", "10M"), ("mdl", "10M")],
}
TIER_SEEDS = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1]}
SIZE_SEEDS = [0]                                              # E6 size runs: seed 0 only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--seeds", default=None, help="override seeds for every tier, e.g. 0,1")
    ap.add_argument("--budget_frac", type=float, default=1.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    log = open(RUNS / "matrix.log", "a")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    jobs = []
    for t in [int(x) for x in args.tiers.split(",")]:
        for arm, size in TIERS[t]:
            seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (
                SIZE_SEEDS if size != "4M" else TIER_SEEDS[t])
            for s in seeds:
                jobs.append((arm, size, s))
    say(f"matrix: {len(jobs)} runs")
    for arm, size, seed in jobs:
        name = f"{arm}_s{seed}" + ("" if size == "4M" else f"_{size}")
        out = RUNS / name
        if (out / "ckpt_final.pt").exists() and (out / "eval.json").exists():
            say(f"skip {name} (done)")
            continue
        cmd = [PY, str(HERE / "train.py"), "--arm", arm, "--seed", str(seed), "--size", size, "--out", str(out),
               "--budget_frac", str(args.budget_frac)]
        if (out / "ckpt_latest.pt").exists() and not (out / "ckpt_final.pt").exists():
            cmd += ["--resume", str(out / "ckpt_latest.pt")]
        say(f"run  {name}: {' '.join(cmd)}")
        if args.dry:
            continue
        t0 = time.time()
        if not (out / "ckpt_final.pt").exists():
            with open(out.parent / f"{name}.log", "a") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
            if rc != 0:
                say(f"FAIL {name} rc={rc}")
                continue
        with open(out.parent / f"{name}.log", "a") as fh:
            rc = subprocess.call([PY, str(HERE / "evaluate.py"), "--run", str(out), "--curve"],
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
        say(f"done {name} rc={rc} in {(time.time() - t0) / 60:.1f} min")
    say("matrix complete")


if __name__ == "__main__":
    main()
