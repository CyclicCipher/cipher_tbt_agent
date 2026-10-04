"""Tables for the tier test (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §17.7): Part A from run_tiers.py, Part B from
matrix_code.py.
    python experiments/neural_turing_architecture/search_bench/summarize_tiers.py runs/tiers_partA.json
    python experiments/neural_turing_architecture/search_bench/summarize_tiers.py runs/tiers_partB.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def span(v):
    return f"{min(v):.2f}" if max(v) - min(v) < 0.005 else f"{min(v):.2f}–{max(v):.2f}"


def part_b(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["world"], r["code"], r["m"], r["model"])].append(r)
    print("| world | code | m | model | unexplained change | path integration k=4 / k=8 (all) | same, wall-free |"
          " lookahead: model / true next codes / random | R² position, heading |")
    print("|" + "---|" * 9)
    for (w, c, m, k), rs in groups.items():
        r2 = (f"{span([r['r2']['position'] for r in rs])}, {span([r['r2']['heading'] for r in rs])}"
              if rs[0].get("r2") else "—")
        print(f"| {w} | {c} | {m} | {k} | {span([r['residual'] for r in rs])} | "
              f"{span([r['pi'][3] for r in rs])} / {span([r['pi'][7] for r in rs])} | "
              f"{span([r['pi_clean'][3] for r in rs])} / {span([r['pi_clean'][7] for r in rs])} | "
              f"{span([r['lookahead']['model'] for r in rs])} / {span([r['lookahead']['true_next'] for r in rs])} / "
              f"{span([r['lookahead']['random'] for r in rs])} | {r2} |")


def main(path):
    p = Path(path) if Path(path).is_absolute() else HERE / path
    data = json.load(open(p))
    if "rows" in data:
        return part_b(data["rows"])
    worlds = data["worlds"]
    groups = defaultdict(list)
    for w in worlds:
        groups[(w["world_type"], w["d"])].append(w)
    for (wt, d), ws in sorted(groups.items()):
        modes = [m for m in ("pos", "any") if m in next(iter(ws[0]["models"].values()))]
        print(f"\n## {wt}  d={d}  ({len(ws)} worlds, {mean([w['n_transitions'] for w in ws]):.0f} search transitions, "
              f"{mean([w['secs'] for w in ws]):.0f}s/world)")
        print("R² of the learned additive code: " + ", ".join(
            f"{k} {min(w['r2'][k] for w in ws):.2f}–{max(w['r2'][k] for w in ws):.2f}" for k in ws[0]["r2"]))
        hdr = "| inverse model | probe k=1 / k=3 |"
        for m in modes:
            hdr += f" progress [{m}] | cos [{m}] | planner 64 / 256 [{m}] |"
        print(hdr)
        print("|" + "---|" * (hdr.count("|") - 1))
        for name in ws[0]["models"]:
            r = [w["models"][name] for w in ws]
            row = f"| {name} | {mean([x['probe']['1'] for x in r]):.2f} / {mean([x['probe']['3'] for x in r]):.2f} |"
            for m in modes:
                row += (f" {mean([x[m]['progress'] for x in r]):.2f} | {mean([x[m]['cos'] for x in r]):.2f} | "
                        f"{mean([x[m]['planner']['64'] for x in r]):.2f} / {mean([x[m]['planner']['256'] for x in r]):.2f} |")
            print(row)
        for m in modes:
            b = defaultdict(list)
            for w in ws:
                for n, v in w["baselines"][m].items():
                    b[n].append(v)
            print(f"baselines [{m}]: " + ", ".join(f"{n} {mean(v):.2f}" for n, v in b.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/tiers_partA.json")
