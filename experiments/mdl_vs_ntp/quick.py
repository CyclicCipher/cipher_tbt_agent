"""The 20-minute experiment: ntp vs exit vs mdl (miner-sourced library), one seed, matched compute, full evaluation.

    python experiments/mdl_vs_ntp/quick.py --minutes 20
    python experiments/mdl_vs_ntp/quick.py --minutes 3      (calibration: same pipeline, tiny budget)

What is kept from the plan: the environment and every eval set, matched TE across arms, the shortest-verified-solution
targets, the epiplexity curriculum, the buffer with spurious-solution dropping, ΔJ scoring on families the candidate
never came from, support/loo/usage pruning, slot init, and the pre-registered verdict rules. What is given up, and why:
seeds (task-level CIs only, so only large effects are detectable), the ablation arms curio / mdl_nolib / mdl_insample,
and the LEARNED proposer -- at this budget REINFORCE cannot bootstrap (the smoke accepted nothing in 3 rounds), so the
library is filled by the symbolic miner the plan already runs as a diagnostic. That tests the OBJECTIVE's library term
(the plan: "the claim under test is about the objective, not about symbolic libraries"), not invention by the model.

Budget: B is sized from the measured throughput so the three runs, their evaluations and the analysis fit `--minutes`
with a 10% margin (constants below). At 20 minutes: B ≈ 86M TE per arm, ~4 min of training each.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable
ARMS = [("ntp", []), ("exit", []), ("mdl", ["--library_source", "miner"])]
ROUND_PCT = 0.10                    # 10 rounds per run: half the plan's cadence, to halve the fixed round overhead
# Wall-clock model, seconds:  total = Σ_arm (fixed[arm] + slow[arm] · B / te_per_s) + evals + analysis, then a 10% margin.
# Measured 2026-09-20 at 12.54M TE with 9-10 rounds: ntp 39 s, exit 44 s, mdl(miner) 45 s -> per-run process overhead
# ~11 s, ~0.5-0.7 s per round, no measurable per-TE drag from buffer replay or library rewriting (slow = 1.0). Eval is
# 80 s per arm at worst case (every E3 task censored); analysis (10k-rep bootstrap) 47 s. `te_per_s` in config.yaml is
# the conservative 348k from the throughput probe; inside a run the observed rate was ~450k, so runs land early.
DEFAULT_SLOW = {"ntp": 1.0, "exit": 1.0, "mdl": 1.0}
DEFAULT_FIXED = {"ntp": 12.0, "exit": 18.0, "mdl": 20.0}
EVAL_S, ANALYSIS_S, MARGIN = 80.0, 50.0, 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--out", default=str(HERE / "runs" / "quick"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    te_per_s = float(cfg.get("te_per_s", 348000))
    slow = cfg.get("quick_slow", DEFAULT_SLOW)
    fixed = cfg.get("quick_fixed_s", DEFAULT_FIXED)
    fixed_total = sum(fixed[a] for a, _ in ARMS) + len(ARMS) * EVAL_S + ANALYSIS_S
    B = int(max(1, (args.minutes * 60 * MARGIN - fixed_total)) * te_per_s / sum(slow[a] for a, _ in ARMS))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "quick.log", "a")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"quick: {args.minutes:.0f} min budget -> B = {B:,} TE per arm ({B / te_per_s / 60:.1f} min of ntp), "
        f"rounds every {int(ROUND_PCT * 100)}%, fixed overhead model {fixed_total:.0f}s")
    t_all = time.time()
    timings = {}
    for arm, extra in ARMS:
        run_dir = out / f"{arm}_s{args.seed}"
        cmd = [PY, str(HERE / "train.py"), "--arm", arm, "--seed", str(args.seed), "--budget", str(B),
               "--budget_frac", "1.0", "--round_pct", str(ROUND_PCT), "--out", str(run_dir)] + extra
        if (run_dir / "ckpt_latest.pt").exists() and not (run_dir / "ckpt_final.pt").exists():
            cmd += ["--resume", str(run_dir / "ckpt_latest.pt")]
        t0 = time.time()
        if not (run_dir / "ckpt_final.pt").exists():
            with open(out / f"{arm}.log", "a") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
            if rc != 0:
                say(f"FAIL {arm} rc={rc}; see {arm}.log")
                return
        t_train = time.time() - t0
        with open(out / f"{arm}.log", "a") as fh:
            rc = subprocess.call([PY, str(HERE / "evaluate.py"), "--run", str(run_dir), "--curve"],
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
        t_eval = time.time() - t0 - t_train
        timings[arm] = dict(train=t_train, eval=t_eval)
        say(f"{arm}: train {t_train / 60:.1f} min, eval {t_eval / 60:.1f} min")
    t0 = time.time()
    with open(out / "analyze.log", "a") as fh:
        subprocess.call([PY, str(HERE / "analyze.py"), "--runs", str(out), "--out", str(out / "results")],
                        stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
    timings["analysis"] = time.time() - t0
    total = time.time() - t_all
    # Re-fit the wall-clock model from what actually happened, so the next `--minutes` is sized on measurements.
    train_s = {a: timings[a]["train"] for a, _ in ARMS}
    fit_fixed = {a: max(0.0, train_s[a] - (B / te_per_s) * slow[a]) for a in train_s}
    json.dump(dict(B=B, minutes=args.minutes, total_s=total, timings=timings, fitted_fixed_s=fit_fixed),
              open(out / "timing.json", "w"), indent=1)
    say(f"done in {total / 60:.1f} min (train {sum(train_s.values()) / 60:.1f}, "
        f"eval {sum(t['eval'] for a, t in timings.items() if a != 'analysis') / 60:.1f}, "
        f"analysis {timings['analysis'] / 60:.1f}) | fitted fixed overhead per arm {({a: round(v) for a, v in fit_fixed.items()})}")


if __name__ == "__main__":
    main()
