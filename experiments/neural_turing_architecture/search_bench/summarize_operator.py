"""Tables for the operator-code test (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §18.4) from run_operator.py outputs.
    python experiments/neural_turing_architecture/search_bench/summarize_operator.py runs/operator_*.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mean(v):
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")


def rng(v):
    return f"{min(v):.2f}–{max(v):.2f}" if max(v) - min(v) >= 0.005 else f"{min(v):.2f}"


def main(paths):
    worlds = []
    for p in paths:
        for q in sorted(glob.glob(str(p if Path(p).is_absolute() else HERE / p))):
            worlds += json.load(open(q))["worlds"]
    groups = defaultdict(list)
    for w in worlds:
        groups[(w["world_type"], w["d"])].append(w)
    for (wt, d), ws in sorted(groups.items()):
        print(f"\n## {wt}  d={d}  ({len(ws)} worlds, {mean([w['n_transitions'] for w in ws]):.0f} training transitions, "
              f"{mean([w['secs'] for w in ws]):.0f}s/world); planner model: "
              + ", ".join(f"{n} ×{c}" for n, c in Counter(w["planner_model"] for w in ws).items()))
        heading = "r2" in next(iter(ws[0]["models"].values()))
        print("| model | m | held-out unexplained | path integration k=1 / 4 / 8 (wall-free) | same, all walks |"
              + (" R² position | R² heading |" if heading else "") + " prototypes moving (main-edge share) |")
        print("|" + "---|" * (7 if heading else 5))
        for name in ws[0]["models"]:
            r = [w["models"][name] for w in ws]
            row = (f"| {name} | {r[0]['m']} | {rng([x['held_unexplained'] for x in r])} | "
                   + " / ".join(f"{mean([x['pi_clean'][k] for x in r]):.2f}" for k in (0, 3, 7)) + " | "
                   + " / ".join(f"{mean([x['pi'][k] for x in r]):.2f}" for k in (0, 3, 7)) + " |")
            if heading:
                row += f" {rng([x['r2']['position'] for x in r])} | {rng([x['r2']['heading'] for x in r])} |"
            if "prototypes" in r[0]:
                mv = [sum(p["moved"] > 0.5 for p in x["prototypes"]) for x in r]
                main_share = mean([p["main"] for x in r for p in x["prototypes"] if p["moved"] > 0.5])
                row += f" {mean(mv):.1f} of {len(r[0]['prototypes'])} ({main_share:.2f}) |"
            else:
                row += " — |"
            print(row)
        for mode in ws[0]["planners"]:
            names = list(ws[0]["planners"][mode])
            print(f"planners [{mode}]: " + ", ".join(f"{n} {mean([w['planners'][mode][n] for w in ws]):.2f}" for n in names))


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/operator_*.json"])
