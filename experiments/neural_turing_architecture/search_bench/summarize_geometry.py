"""Tables for the geometry test (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §15) from run_geometry.py outputs.
    python experiments/neural_turing_architecture/search_bench/summarize_geometry.py runs/geometry_grid_perm.json runs/geometry_grid2.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main(paths):
    worlds = []
    for p in paths:
        p = Path(p) if Path(p).is_absolute() else HERE / p
        run = json.load(open(p))
        worlds += [dict(w, m=run["args"]["m"], n_rand=run["args"]["n_rand"]) for w in run["worlds"]]
    groups = defaultdict(list)
    for w in worlds:
        groups[(w["world_type"], w["d"], w["m"], w["n_rand"])].append(w)
    for (wt, d, m_code, n_rand), ws in sorted(groups.items()):
        modes = [m for m in ("any", "one", "both") if m in ws[0]["codes"]["raw"]]
        budgets = list(ws[0]["codes"]["raw"][modes[0]]["planner"])
        print(f"\n## {wt}  d={d}  learned m={m_code}, {n_rand} random transitions  ({len(ws)} worlds, "
              f"{mean([w['n_transitions'] for w in ws]):.0f} search transitions, {mean([w['secs'] for w in ws]):.0f}s/world)")
        if "r2" in ws[0]:
            print("R² " + ", ".join(f"{k} {min(w['r2'][k] for w in ws):.2f}–{max(w['r2'][k] for w in ws):.2f}"
                                    for k in ws[0]["r2"]))
        ks = list(ws[0]["codes"]["raw"]["probe"])
        hdr = "| code | " + " | ".join(f"probe {k}" for k in ks)
        for m in modes:
            hdr += f" | progress [{m}] | cos [{m}] | " + " | ".join(f"plan@{B} [{m}]" for B in budgets)
        print(hdr + " |")
        print("|" + "---|" * (hdr.count("|")))
        for c in ws[0]["codes"]:
            row = [f"{mean([w['codes'][c]['probe'][k] for w in ws]):.2f}" for k in ks]
            for m in modes:
                r = [w["codes"][c][m] for w in ws]
                row += [f"{mean([x['progress'] for x in r]):.2f}", f"{mean([x['cos'] for x in r]):.2f}"]
                row += [f"{mean([x['planner'][B] for x in r]):.2f}" for B in budgets]
            print(f"| {c} | " + " | ".join(row) + " |")
        for m in modes:
            b = defaultdict(list)
            for w in ws:
                for name, v in w["baselines"][m].items():
                    b[name].append(v)
            print(f"baselines [{m}]: " + ", ".join(f"{n} {mean(v):.2f}" for n, v in b.items()))


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/geometry_grid_perm.json", "runs/geometry_grid2.json"])
