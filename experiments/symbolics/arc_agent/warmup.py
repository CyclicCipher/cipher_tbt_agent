"""Warm-up play (docs/AGENT.md part 22): the agent plays every public game offline with a long budget (warm-up actions
are never scored) and records the type of every WIN's cause. The draws become the goal-type prior shipped with the
agent (priors.json) and, per fold, the cross-validated priors of the evaluation (kaggle/harness.py --priors cv).
    venv312/Scripts/python.exe -m arc_agent.warmup      (from experiments/symbolics)
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kaggle"))

CAP, BUDGET, LIMIT = 20, 400, 150                       # 20 x the human baseline per level; 400 search nodes; 2.5 min a game


def main():
    import harness
    t0 = time.time()
    res = harness.main(None, "brain", None, fast=True, cap=CAP, budget=BUDGET, priors="even", time_limit=LIMIT)
    games = {r["game"]: dict(win_types=r["win_types"], kinds=r.get("kinds", []), corpus=r.get("corpus", []))
             for r in res}
    draws = [t for g in sorted(games) for t in games[g]["win_types"]]
    kinds = [k for g in sorted(games) for k in games[g]["kinds"]]
    library = harness.learn_library([t for g in sorted(games) for t in games[g]["corpus"]])
    out = dict(games=games, levels=sum(r["solved"] for r in res), counts=dict(Counter(draws)),
               kind_counts=dict(Counter(kinds)), games_with_kinds=sum(1 for g in games.values() if g["kinds"]),
               templates=len(library), seconds=round(time.time() - t0, 1))
    os.makedirs(HERE.parent / "runs", exist_ok=True)
    json.dump(out, open(HERE.parent / "runs" / "warmup.json", "w"), indent=1)
    json.dump(dict(draws=draws, kinds=kinds, library=library, source="warm-up on the 25 public games"),
              open(HERE / "priors.json", "w"), indent=1)
    print(f"warm-up: {out['levels']} levels won, goal types {out['counts']}, expert kinds {out['kind_counts']} "
          f"(from {out['games_with_kinds']} games), {len(library)} library templates, {out['seconds']} s")


if __name__ == "__main__":
    main()
