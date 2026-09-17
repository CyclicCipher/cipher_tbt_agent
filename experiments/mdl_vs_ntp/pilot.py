"""Phase 4 (pilot + gates) and Phase 5 (the matrix), as one detached, resumable process.

    Windows:  Start-Process python -ArgumentList experiments/mdl_vs_ntp/pilot.py -WindowStyle Hidden
    POSIX:    nohup python experiments/mdl_vs_ntp/pilot.py &
    poll:     runs/pilot/pilot.log, then runs/matrix.log

Pilot budget is B/3 with rounds every 5% of THAT budget (20 rounds, so "by round 4 / round 8" in the gates is
meaningful). LR is swept on ntp only and chosen by ID transduction EM; every other arm inherits it. Gates G1-G6 are
checked from the pilot's eval.json / logs; the plan's remedies are applied at most once each, then the full-matrix
wall-clock is projected from the pilot timings, tiers are reduced if it exceeds 16 h, and the matrix runs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PILOT = RUNS / "pilot"
PY = sys.executable
LRS = [3e-4, 1e-3, 3e-3]

sys.path.insert(0, str(HERE))
from run_matrix import TIER_SEEDS, build_jobs, run_jobs           # noqa: E402

PILOT.mkdir(parents=True, exist_ok=True)
_log = open(PILOT / "pilot.log", "a")


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    _log.write(line + "\n")
    _log.flush()


def cfg_get():
    with open(HERE / "config.yaml") as fh:
        return yaml.safe_load(fh)


def cfg_set(**kv):
    lines = open(HERE / "config.yaml").read().splitlines()
    for k, v in kv.items():
        for i, line in enumerate(lines):
            if line.startswith(f"{k}:"):
                comment = ("  #" + line.split("#", 1)[1]) if "#" in line else ""
                lines[i] = f"{k}: {v}{comment}"
                break
        else:
            lines.append(f"{k}: {v}")
    open(HERE / "config.yaml", "w").write("\n".join(lines) + "\n")
    say(f"config: {kv}")


def train_and_eval(arm, out, budget, lr, size="4M", seed=0):
    """A full run at `budget` (rounds every 5% of it), then evaluate; skipped if already done. Returns eval dict."""
    out = Path(out)
    if not ((out / "ckpt_final.pt").exists() and (out / "eval.json").exists()):
        cmd = [PY, str(HERE / "train.py"), "--arm", arm, "--seed", str(seed), "--size", size, "--lr", str(lr),
               "--budget", str(budget), "--budget_frac", "1.0", "--round_pct", "0.05", "--out", str(out)]
        if (out / "ckpt_latest.pt").exists():
            cmd += ["--resume", str(out / "ckpt_latest.pt")]
        say(f"run {out.name}: {' '.join(cmd[1:])}")
        t0 = time.time()
        with open(out.parent / f"{out.name}.log", "a") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
        if rc != 0:
            raise RuntimeError(f"{out.name} failed rc={rc}; see {out.name}.log")
        with open(out.parent / f"{out.name}.log", "a") as fh:
            rc = subprocess.call([PY, str(HERE / "evaluate.py"), "--run", str(out), "--curve"],
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
        if rc != 0:
            raise RuntimeError(f"evaluate {out.name} failed rc={rc}")
        say(f"done {out.name} in {(time.time() - t0) / 60:.1f} min")
    return json.load(open(out / "eval.json"))


def wall_seconds(out):
    last = None
    for line in open(Path(out) / "train.jsonl"):
        r = json.loads(line)
        if "wall" in r:
            last = r
    return last["wall"] if last else float("nan")


def loss_still_falling(out):
    rows = [json.loads(l) for l in open(Path(out) / "train.jsonl")]
    rows = [r for r in rows if "loss" in r]
    n = len(rows)
    if n < 20:
        return False
    late = np.mean([r["loss"] for r in rows[int(0.9 * n):]])
    mid = np.mean([r["loss"] for r in rows[int(0.7 * n):int(0.8 * n)]])
    return late < 0.97 * mid


def g(d, k):
    return d[str(k)] if str(k) in d else d[k]


def labeled_wake_rate_by_round(out, rnd):
    best = 0.0
    for line in open(Path(out) / "wake.jsonl"):
        r = json.loads(line)
        if r["round"] <= rnd:
            lab = [v for k, v in r["by_category"].items() if k.startswith("labeled")]
            if lab:
                best = max(best, float(np.mean(lab)))
    return best


def accepted_by_round(out, rnd):
    n, miner_pos, prop_pos = 0, False, False
    for line in open(Path(out) / "rounds.jsonl"):
        r = json.loads(line)
        if r["round"] <= rnd:
            n += sum(1 for e in r["events"] if e["event"] == "accept")
            miner_pos |= r["miner_best"] > 0
            prop_pos |= (r["best_proposed_dJ"] == r["best_proposed_dJ"]) and r["best_proposed_dJ"] > 0
    return n, miner_pos, prop_pos


def gates(ev, outs):
    """Returns {gate: (passed, detail)}."""
    res = {}
    ntp_em = ev["ntp"]["e1"]["e1_id"]["em"]
    falling = loss_still_falling(outs["ntp"])
    res["G1"] = (ntp_em >= 0.30 or falling, f"ntp ID EM {ntp_em:.3f}, loss falling {falling}")
    v16 = g(ev["ntp"]["e3"]["e3_id_labeled"]["verified_at"], 16)
    res["G1b"] = (v16 >= 0.20 or falling, f"ntp ID-labeled verified@16 {v16:.3f}")
    wr = labeled_wake_rate_by_round(outs["mdl"], 4)
    res["G2"] = (wr >= 0.05, f"mdl labeled wake verified rate by round 4: {wr:.3f}")
    n_acc, miner_pos, prop_pos = accepted_by_round(outs["mdl"], 8)
    res["G3"] = (n_acc >= 1, f"mdl accepted {n_acc} macros by round 8 (miner_best>0: {miner_pos}, any proposal>0: {prop_pos})")
    nd = {a: ev[a]["e2"]["e2_noisy"]["bits"] for a in ev}
    res["G4"] = (all(v >= 3.0 for v in nd.values()), f"noisy-digit NLL {({a: round(v, 2) for a, v in nd.items()})}")
    mdl_em, nolib_em = ev["mdl"]["e1"]["e1_id"]["em"], ev["mdl_nolib"]["e1"]["e1_id"]["em"]
    res["G5"] = (mdl_em >= nolib_em - 0.05, f"mdl ID EM {mdl_em:.3f} vs mdl_nolib {nolib_em:.3f}")
    res["G6"] = (nolib_em >= ntp_em - 0.05, f"mdl_nolib ID EM {nolib_em:.3f} vs ntp {ntp_em:.3f}")
    return res, dict(falling=falling, miner_pos=miner_pos, prop_pos=prop_pos)


def main():
    cfg = cfg_get()
    B = int(cfg["B"])
    remedies = {"double_B": 0, "10M": 0, "warm": 0, "beta": 0, "mix": 0}
    state_path = PILOT / "pilot.json"
    state = json.load(open(state_path)) if state_path.exists() else {}

    # ── LR sweep on ntp only ─────────────────────────────────────────────────────────────────────────────────────────
    pb = B // 3
    ems = {}
    for lr in LRS:
        ev = train_and_eval("ntp", PILOT / f"ntp_lr{lr:g}", pb, lr)
        ems[lr] = ev["e1"]["e1_id"]["em"]
        say(f"lr {lr:g}: ntp ID EM {ems[lr]:.3f}")
    lr = max(ems, key=ems.get)
    say(f"chosen LR {lr:g} (ID EM {ems[lr]:.3f})")
    cfg_set(lr=lr)
    state.update(lr=lr, lr_sweep=ems)
    json.dump(state, open(state_path, "w"))

    # ── the pilot arms and the gates, with each remedy at most once ──────────────────────────────────────────────────
    size = "4M"
    for attempt in range(6):
        pb = B // 3
        outs = {"ntp": PILOT / f"ntp_lr{lr:g}" if (attempt == 0 and size == "4M") else PILOT / f"ntp_B{B}_{size}"}
        ev = {"ntp": train_and_eval("ntp", outs["ntp"], pb, lr, size)}
        # a rerun is only forced on the arms a remedy actually touches: warm start / beta -> library arms, mix -> curriculum arms
        suffix = {"exit": "", "mdl": f"_w{remedies['warm']}b{remedies['beta']}m{remedies['mix']}",
                  "mdl_nolib": f"_m{remedies['mix']}"}
        for arm in ("exit", "mdl", "mdl_nolib"):
            outs[arm] = PILOT / f"{arm}_B{B}_{size}{suffix[arm]}"
            ev[arm] = train_and_eval(arm, outs[arm], pb, lr, size)
        res, info = gates(ev, outs)
        for k, (ok, detail) in res.items():
            say(f"{k}: {'PASS' if ok else 'FAIL'} — {detail}")
        state["gates"] = {k: dict(passed=ok, detail=d) for k, (ok, d) in res.items()}
        json.dump(state, open(state_path, "w"))
        failed = [k for k, (ok, _d) in res.items() if not ok]
        if not failed:
            break
        # remedies, in the plan's order
        if ("G1" in failed or "G1b" in failed):
            if info["falling"] and remedies["double_B"] < 1:
                remedies["double_B"] += 1
                B *= 2
                cfg_set(B=B)
                say("G1/G1b: loss still falling -> B doubled, rerunning the pilot")
                continue
            if remedies["10M"] < 1:
                remedies["10M"] += 1
                size = "10M"
                say("G1/G1b: flat and below threshold -> one attempt at the 10M config")
                continue
            say("STOP: G1 still fails after the 10M attempt")
            return
        if "G2" in failed:
            say("STOP: G2 failed — wake verified rate on labelled families < 5% by round 4; debug decoding/execution")
            return
        if "G4" in failed:
            say("STOP: G4 failed — noisy-digit NLL < 3.0 bits: the hash leaks; fix the generator and restart")
            return
        if "G3" in failed:
            if not info["miner_pos"]:
                say("STOP: G3 failed with miner_best <= 0 — a cost bug")
                return
            if remedies["warm"] < 1:
                remedies["warm"] += 1
                cfg_set(proposer_warm_start=200)
                say("G3: miner finds macros the proposer never scores > 0 -> proposer warm start (200 steps) for all "
                    "library arms; DEVIATION")
                continue
        if "G5" in failed and remedies["beta"] < 1:
            remedies["beta"] += 1
            cfg_set(beta=0.03)
            say("G5: mdl ID EM more than 5 pts below mdl_nolib -> beta = 0.03, rerunning mdl")
            continue
        if "G6" in failed and remedies["mix"] < 1:
            remedies["mix"] += 1
            cfg_set(curriculum_mix=0.2)
            say("G6: mdl_nolib ID EM more than 5 pts below ntp -> uniform mix 0.2 for curriculum arms, rerunning")
            continue
        say(f"gates still failing after remedies: {failed}; proceeding to the matrix as-is (reported as a deviation)")
        break

    # ── project the full-matrix wall-clock and reduce tiers if needed ────────────────────────────────────────────────
    per_run = {}
    for arm in ("ntp", "exit", "mdl", "mdl_nolib"):
        w = wall_seconds(outs[arm]) * 3 + 120                             # full B = 3x the pilot, plus eval
        per_run[arm] = w
    per_run["curio"] = per_run["mdl"]
    per_run["mdl_insample"] = per_run["mdl"]
    size_mult = {"4M": 1.0, "1M": 0.6, "10M": 2.6}
    tier_seeds = dict(TIER_SEEDS)
    for reduction in (None, "tier2", "tier3"):
        if reduction == "tier2":
            tier_seeds[2] = [0, 1]
        if reduction == "tier3":
            tier_seeds[3] = [0]
        jobs = build_jobs([1, 2, 3], None, tier_seeds)
        hours = sum(per_run[a] * size_mult[s] for a, s, _seed in jobs) / 3600
        say(f"projected matrix wall-clock {hours:.1f} h for {len(jobs)} runs" + (f" after {reduction}" if reduction else ""))
        if hours <= 16:
            break
    state.update(projected_hours=hours, tier_seeds={k: v for k, v in tier_seeds.items()}, B=B, size=size)
    json.dump(state, open(state_path, "w"))

    # ── the matrix ───────────────────────────────────────────────────────────────────────────────────────────────────
    say("launching the matrix")
    run_jobs(jobs, 1.0, False, say)
    say("running analysis")
    with open(RUNS / "analyze.log", "a") as fh:
        subprocess.call([PY, str(HERE / "analyze.py")], stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))
    say("pilot + matrix complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                                                  # noqa: BLE001
        say(f"ERROR: {type(e).__name__}: {e}")
        raise
