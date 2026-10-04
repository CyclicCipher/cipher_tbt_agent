"""Tables for the training-loop round (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §16) from train_loop.py outputs.
    python experiments/neural_turing_architecture/search_bench/summarize_train.py               (both label schemes)
    python experiments/neural_turing_architecture/search_bench/summarize_train.py runs/train_bellman_*.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["rloo", "bestofn", "bptt", "answer_opt", "pi_grad", "pi_mcts"]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def span(v):
    return f"{mean(v):.2f} ({min(v):.2f}–{max(v):.2f})" if len(v) > 1 else f"{mean(v):.2f}"


def main(paths):
    runs = []
    for p in paths:
        for q in sorted(glob.glob(str(p if Path(p).is_absolute() else HERE / p))):
            f = json.load(open(q))
            runs += [dict(r, vlabel=f["args"].get("vlabel", "path")) for r in f["runs"]]
    for lab in sorted({r["vlabel"] for r in runs}):
        report([r for r in runs if r["vlabel"] == lab], lab)


def report(runs, lab):
    by = defaultdict(list)
    for r in runs:
        by[(r["d"], r["arm"])].append(r)
    for d in sorted({r["d"] for r in runs}):
        arms = [a for a in ARMS if (d, a) in by]
        n = len(by[(d, arms[0])])
        print(f"\n## value labels: {lab}   d = {d}   ({n} seeds; mean, range over seeds)")
        for key, title in [("amortised", "held-out, AMORTISED policy (no search)"),
                           ("amortised_train", "TRAINING problems, amortised policy (fitting)"),
                           ("searched", "held-out, policy + value inside pi_mcts at the test budget"),
                           ("v_start", "V(start) on held-out problems (an optimal chain's discounted outcome: 0.70 / 0.49 / 0.34 / 0.24)")]:
            print(f"\n{title}, at the end:\n| arm | L=1 | L=2 | L=3 | L=4 |\n|---|---|---|---|---|")
            for a in arms:
                cells = [span([r["log"][-1]["eval"][L][key] for r in by[(d, a)]]) for L in ("1", "2", "3", "4")]
                print(f"| {a} | " + " | ".join(cells) + " |")
        its = [e["iter"] for e in by[(d, arms[0])][0]["log"]]
        ks = [e["thought_steps"] for e in by[(d, arms[0])][0]["log"]]
        print("\nlearning curve, held-out amortised success averaged over L = 1-4 (thought-steps spent: "
              + ", ".join(f"{k // 1000}k" for k in ks) + ")")
        print("| arm | " + " | ".join(f"it {i}" for i in its) + " |\n|---|" + "---|" * len(its))
        for a in arms:
            row = [mean([mean([r["log"][j]["eval"][L]["amortised"] for L in ("1", "2", "3", "4")]) for r in by[(d, a)]])
                   for j in range(len(its))]
            print(f"| {a} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
        print("\ntraining solve rate (last 80 problems before each evaluation)")
        print("| arm | " + " | ".join(f"it {i}" for i in its) + " |\n|---|" + "---|" * len(its))
        for a in arms:
            row = [mean([r["log"][j]["train_solved"] for r in by[(d, a)]]) for j in range(len(its))]
            print(f"| {a} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
        if "path_len" in by[(d, arms[0])][0]["log"][-1]:
            print("\nat the end: (thought, state) pairs distilled per solved problem (= the solution's length, except bestofn,"
                  " which distils every verified chain) | the policy's fit to its distilled targets (cosine)")
            for a in arms:
                last = [r["log"][-1] for r in by[(d, a)]]
                fits = [e["distil_fit"] for e in last if e["distil_fit"] is not None]
                print(f"| {a} | {mean([e['path_len'] for e in last]):.1f} | " + (f"{mean(fits):.2f}" if fits else "—") + " |")
        print("\nCPU seconds per run: " + ", ".join(f"{a} {mean([r['secs'] for r in by[(d, a)]]):.0f}" for a in arms))


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/train_d*_s*.json", "runs/train_bellman_*.json"])
