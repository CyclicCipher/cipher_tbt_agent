"""The whole experiment inside 20 minutes: calibrate, four arms, inspection, verdicts, a plain-English report.

    python experiments/inner_objective/run.py --minutes 20 --out experiments/inner_objective/runs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
ARMS = ["A", "B", "Ap", "Bp"]


def sh(cmd, log):
    with open(log, "a") as fh:
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent))


def last_eval(run_dir):
    rows = [json.loads(l) for l in open(Path(run_dir) / "eval.jsonl")]
    return rows[-1], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--out", default=str(HERE / "runs"))
    ap.add_argument("--max_steps", type=int, default=2500)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--task", default="compose", choices=["compose", "affine"])
    ap.add_argument("--rule", default="ftest", choices=["ftest", "warm"])
    ap.add_argument("--extra", default="", help="extra train.py arguments, e.g. '--keep_q 0.5 --mask_where update'")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a")
    t_all = time.time()

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # ── calibrate: ms/step of the more expensive arm, then size the run so four arms fit in ~45% of the budget ─────
    cal = out / "_cal"
    sh([PY, str(HERE / "train.py"), "--arm", "B", "--steps", "100", "--time_only", "80", "--d", str(args.d), "--task", args.task, "--rule", args.rule, *args.extra.split(),
        "--out", str(cal)], out / "calibrate.log")
    ms = None
    for line in open(out / "calibrate.log"):
        if "ms/step" in line:
            ms = float(line.split("->")[1].split("ms")[0])
    train_budget_s = args.minutes * 60 * 0.45
    steps = int(min(args.max_steps, (train_budget_s / 4) / (ms / 1000)))
    say(f"calibration: {ms:.0f} ms/step -> {steps} steps per arm ({4 * steps * ms / 1000 / 60:.1f} min of training)")

    # ── the four arms ─────────────────────────────────────────────────────────────────────────────────────────────
    for arm in ARMS:
        run_dir = out / arm
        if (run_dir / "trace.pt").exists():
            say(f"skip {arm} (done)")
            continue
        t0 = time.time()
        rc = sh([PY, str(HERE / "train.py"), "--arm", arm, "--steps", str(steps), "--d", str(args.d),
                 "--task", args.task, "--rule", args.rule, *args.extra.split(), "--out", str(run_dir)], out / f"{arm}.log")
        if rc != 0:
            say(f"FAIL {arm} rc={rc}; see {arm}.log")
            return
        ev, _ = last_eval(run_dir)
        say(f"{arm}: {time.time() - t0:.0f}s | train acc {ev['train_last_acc']:.3f} | held acc {ev['held_last_acc']:.3f} "
            f"| held solved {ev['held_solved']:.2f} | demos-to-criterion {ev['held_ttc_mean']:.2f}")

    # ── gate C0: the baseline must have learned the task at this size ────────────────────────────────────────────
    evA, _ = last_eval(out / "A")
    c0 = evA["held_solved"] >= 0.5
    say(f"gate C0 (baseline solves >= half the held-out compositions): {'PASS' if c0 else 'FAIL'} ({evA['held_solved']:.2f})")

    # ── inspection ────────────────────────────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    res_dir = out / "results"
    rc = sh([PY, str(HERE / "inspect_weights.py"), "--runs"] + [str(out / a) for a in ARMS] + ["--out", str(res_dir)],
            out / "inspect.log")
    say(f"inspection: {time.time() - t0:.0f}s rc={rc}")
    insp = {r["arm"]: r for r in json.load(open(res_dir / "inspect.json"))} if rc == 0 else {}
    sh([PY, str(HERE / "posthoc.py"), "--runs"] + [str(out / a) for a in ARMS], out / "posthoc.log")
    post = json.load(open(res_dir / "posthoc.json")) if (res_dir / "posthoc.json").exists() else {}

    # ── a second seed for A and B if there is slack ───────────────────────────────────────────────────────────────
    seed1 = {}
    if (time.time() - t_all) / 60 < args.minutes * 0.62:
        for arm in ("A", "B"):
            run_dir = out / f"{arm}_s1"
            t0 = time.time()
            if (run_dir / "trace.pt").exists():
                seed1[arm], _ = last_eval(run_dir)
                say(f"skip {arm} seed 1 (done)")
                continue
            rc = sh([PY, str(HERE / "train.py"), "--arm", arm, "--steps", str(steps), "--seed", "1", "--d", str(args.d),
                     "--task", args.task, "--rule", args.rule, *args.extra.split(), "--out", str(run_dir), "--snap_every", "100000"],
                    out / f"{arm}_s1.log")
            if rc == 0:
                seed1[arm], _ = last_eval(run_dir)
                say(f"{arm} seed 1: {time.time() - t0:.0f}s | held acc {seed1[arm]['held_last_acc']:.3f} | "
                    f"demos-to-criterion {seed1[arm]['held_ttc_mean']:.2f}")

    # ── verdicts ──────────────────────────────────────────────────────────────────────────────────────────────────
    ev = {a: last_eval(out / a)[0] for a in ARMS}
    v = {}
    d_ttc = ev["A"]["held_ttc_mean"] - ev["B"]["held_ttc_mean"]
    d_acc = ev["B"]["held_last_acc"] - ev["A"]["held_last_acc"]
    v["P1 competence"] = ("PASS" if (d_ttc >= 0.5 or d_acc >= 0.05) else "FAIL" if (d_ttc <= -0.5 or d_acc <= -0.05)
                          else "INCONCLUSIVE", f"demos-to-criterion A {ev['A']['held_ttc_mean']:.2f} vs B {ev['B']['held_ttc_mean']:.2f}; "
                          f"held-out accuracy A {ev['A']['held_last_acc']:.3f} vs B {ev['B']['held_last_acc']:.3f}")
    hurtA = ev["A"]["held_last_acc"] - ev["Ap"]["held_last_acc"]
    hurtB = ev["B"]["held_last_acc"] - ev["Bp"]["held_last_acc"]
    noise_trust = None
    if insp.get("Bp") and insp["Bp"]["trust_noise"]:
        tn = insp["Bp"]["trust_noise"]
        noise_trust = float(np.mean(tn[len(tn) // 4:]))
    v["P2 noise immunity"] = ("PASS" if (hurtA > 0.03 and hurtB < 0.03) else "FAIL" if hurtB >= hurtA else "INCONCLUSIVE",
                              f"noise families cost A {hurtA:+.3f} and B {hurtB:+.3f} of held-out accuracy; "
                              f"noise-family trust under B {noise_trust if noise_trust is None else round(noise_trust, 3)}")
    if insp:
        mf = [m for m in insp["B"]["masked_frac"] if m is not None]
        peak_mask = max(mf) if mf else 0.0
        nzB, nzA = insp["B"]["near_zero_frac"], insp["A"]["near_zero_frac"]
        v["P3 what the mask does"] = ("PASS" if (peak_mask > 0.2 and nzB > nzA + 0.05) else "FAIL" if peak_mask < 0.05
                                      else "INCONCLUSIVE", f"peak masked fraction {peak_mask:.2f}; near-zero weights A {nzA:.3f} vs B {nzB:.3f}")
        sA, sB = insp["A"]["mean_selectivity"], insp["B"]["mean_selectivity"]
        fA, fB = insp["A"]["mean_functional_selectivity"], insp["B"]["mean_functional_selectivity"]
        cB = insp["B"]["control_functional_selectivity"]
        v["P4 organisation"] = ("PASS" if (sB >= sA + 0.05 and fB > cB) else "FAIL" if sB <= sA - 0.05 else "INCONCLUSIVE",
                                f"primitive regions A {insp['A']['n_primitive_regions']} vs B {insp['B']['n_primitive_regions']}; "
                                f"mean selectivity A {sA:.3f} vs B {sB:.3f}; functional selectivity A {fA:.3f} vs B {fB:.3f} "
                                f"(random-set control {cB:.3f})")
    if post:
        v["P1b graded"] = ("PASS" if post["B"]["held_bits"] < post["A"]["held_bits"] - 0.05 else "FAIL"
                           if post["B"]["held_bits"] > post["A"]["held_bits"] + 0.05 else "INCONCLUSIVE",
                           f"held-out bits/digit A {post['A']['held_bits']:.3f} vs B {post['B']['held_bits']:.3f}; per-digit accuracy A "
                           f"{post['A']['held_digit_acc']:.3f} vs B {post['B']['held_digit_acc']:.3f}; steps to 20% training accuracy A "
                           f"{post['A']['steps_to']['0.2']} vs B {post['B']['steps_to']['0.2']}")
    trA, trB = ev["A"]["train_last_acc"], ev["B"]["train_last_acc"]
    v["P5 no harm on training"] = ("PASS" if trB >= trA - 0.05 else "FAIL", f"training accuracy A {trA:.3f} vs B {trB:.3f}")
    for k, (verdict, detail) in v.items():
        say(f"{verdict:<13} {k}: {detail}")

    # ── the report ────────────────────────────────────────────────────────────────────────────────────────────────
    total = (time.time() - t_all) / 60
    lines = ["# Inner objective — results", "",
             "**What was compared.** Two optimisers, same network (%s weights), same task, same examples, same number of steps. "
             "The baseline nudges every weight to make the current batch less wrong. The alternative computes one nudge per task family, "
             "trusts a family only if its two half-batches agree with each other, keeps a weight's nudge only if the families agree on it "
             "(their disagreement is no more than 3x the noise within a family), and shrinks toward zero any weight they cannot agree on. "
             "Each optimiser was also run with three 'noise' families mixed in whose answers are random." % f"{ev['A'].get('params', insp.get('A', {}).get('params', ''))}",
             "", "| arm | training accuracy | new-composition accuracy | demonstrations to solve a new composition | noise families cost |",
             "|---|---|---|---|---|"]
    for a, label in (("A", "baseline"), ("B", "inner objective"), ("Ap", "baseline + noise"), ("Bp", "inner objective + noise")):
        cost = "" if a in ("A", "B") else f"{(ev[a[0]]['held_last_acc'] - ev[a]['held_last_acc']):+.3f}"
        lines.append(f"| {label} | {ev[a]['train_last_acc']:.3f} | {ev[a]['held_last_acc']:.3f} | {ev[a]['held_ttc_mean']:.2f} of 8 | {cost} |")
    if seed1:
        lines += ["", "Second seed (A / B): held-out accuracy %.3f / %.3f, demonstrations to criterion %.2f / %.2f." %
                  (seed1["A"]["held_last_acc"], seed1["B"]["held_last_acc"], seed1["A"]["held_ttc_mean"], seed1["B"]["held_ttc_mean"])]
    lines += ["", "**Predictions, stated before running:**", ""]
    for k, (verdict, detail) in v.items():
        lines.append(f"- **{k}: {verdict}.** {detail}")
    if insp:
        lines += ["", "**Inside the weights.**"]
        for a in ("A", "B"):
            r = insp[a]
            regs = sorted(r["regions"], key=lambda x: -x["selectivity"])[:4]
            lines.append(f"- {('Baseline' if a == 'A' else 'Inner objective')}: {r['n_primitive_regions']} of {len(r['regions'])} weight clusters line up with a single "
                         f"primitive (selectivity >= 0.9); removing a cluster changes its own primitive's accuracy by {r['mean_functional_selectivity']:+.3f} more than the others' "
                         f"(a random cluster of the same size: {r['control_functional_selectivity']:+.3f}); {100 * r['near_zero_frac']:.1f}% of weights ended near zero. "
                         f"Sharpest clusters: " + "; ".join(f"R{x['id']} -> {x['primitive']} (sel {x['selectivity']:.2f}, ablation {x['ablation_primitive']}, formed by step {x['formed_50pct_step']}, mostly {list(x['anatomy'])[0]})" for x in regs))
        if insp["B"]["reads_from"]:
            rf = insp["B"]["reads_from"][:4]
            lines.append("- Wiring under the inner objective (attention head <- region it reads from, overlap of read/write subspaces): " +
                         "; ".join(f"{x['reader']} <- R{x['region']} ({x['primitive']}) {x['overlap']:.2f}" for x in rf))
    lines += ["", f"Gate C0 (baseline learned the task): {'PASS' if c0 else 'FAIL'}. Total wall-clock {total:.1f} min. "
              f"Figures: results/curves.png, results/participation_*.png, results/ablation_*.png."]
    lines.insert(1, f"*task: {args.task}, rule: {args.rule}{(' ' + args.extra) if args.extra else ''}, {steps} steps per arm*")
    (res_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    say(f"done in {total:.1f} min; report at {res_dir / 'report.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
