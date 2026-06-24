"""Score the ColumnAgent on the LockPath replica: levels completed + RHAE-proxy, per seed.

Run from the ProgramSynthesis directory:  python -m agent.column.column_score
The RHAE-proxy uses the BFS oracle's optimal action count as the (harsh) baseline — same as the symbolic
agent's score.py, so the two numbers are directly comparable.
"""

from __future__ import annotations

from statistics import mean

from arc_agi_3 import Environment
from arc_agi_3.games import LockPath

from ..wm.score import oracle_optimal, per_level_actions
from .agent import ColumnAgent


def run(seeds=range(12), max_actions: int = 6000):
    opt = oracle_optimal(LockPath)
    n = len(opt)
    rows = []
    for s in seeds:
        env = Environment(LockPath())
        per, completed = per_level_actions(env, ColumnAgent(seed=s), max_actions)
        lvl = [min(1.0, (opt[i] / per[i]) ** 2) if (i in completed and opt[i] and per.get(i)) else 0.0
               for i in range(n)]
        rows.append((s, len(completed), dict(sorted(per.items())), mean(lvl)))
    return opt, rows


if __name__ == "__main__":
    opt, rows = run()
    print(f"oracle-optimal actions/level: {opt}\n")
    for s, nc, per, sc in rows:
        print(f"  seed {s:2d}:  {nc}/{len(opt)} levels   actions/level {per}   RHAE {100 * sc:5.1f}%")
    print(f"\nmean levels completed: {mean(r[1] for r in rows):.2f}/{len(opt)}")
    print(f"mean RHAE-proxy:       {100 * mean(r[3] for r in rows):.1f}%   (symbolic agent ~37.9%)")
