"""Run ZipLearner's player on every ARC-AGI-3 replica game (src/tasks/games) and report, per level: solved or not,
actions used, the oracle's shortest solution (breadth-first search on the true game via snapshot/restore), and the
world model's quality (prediction accuracy on the transitions it saw, unknown-window rate) -- so that "could not learn
the mechanics" and "could not solve the level" are reported separately.

    python experiments/ziplearn/play_games.py --budget 150            (CPU; a few minutes; run detached)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from arcgames import play                                            # noqa: E402
from tasks.games import LockPath, MultiKey, Sokoban, CollectAll, Toggle, Tetris   # noqa: E402
from tasks.harness import Environment                                # noqa: E402
from tasks.oracle import solve_level                                 # noqa: E402

GAMES = {"LockPath": LockPath, "MultiKey": MultiKey, "Sokoban": Sokoban, "CollectAll": CollectAll, "Toggle": Toggle, "Tetris": Tetris}


def oracle_lengths(cls, n_levels):
    out = []
    g = cls()
    for lvl in range(n_levels):
        g.load_level(lvl)
        try:
            sol = solve_level(g, max_states=200_000)
        except Exception as e:                                       # a game without a usable oracle: report None
            sol = None
        out.append(None if sol is None else len(sol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150, help="actions per level")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--max_levels", type=int, default=None)
    ap.add_argument("--sleep", type=int, default=0, help="1: a sleep pass (context minimisation) after every level")
    ap.add_argument("--trace", type=int, default=0, help="1: write every step's frame, choice, prediction and outcome to <out>/trace_<game>.json")
    ap.add_argument("--out", default=str(HERE / "runs" / "e13"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    print(f"{'game':<11}{'level':>6}{'solved':>8}{'actions':>9}{'oracle':>8} | model: prediction acc, unknown cells, radius per action")
    for name in args.games.split(","):
        cls = GAMES[name]
        env = Environment(cls())
        n_levels = env.game.level_count if args.max_levels is None else min(args.max_levels, env.game.level_count)
        t0 = time.time()
        trace = [] if args.trace else None
        results, stats = play(env, budget_per_level=args.budget, max_levels=args.max_levels, sleep=bool(args.sleep), trace=trace)
        if trace is not None:
            json.dump(trace, open(out / f"trace_{name}.json", "w"))
        secs = time.time() - t0
        oracle = oracle_lengths(cls, n_levels)
        acc = stats["correct"] / max(1, stats["predictions"])
        unk = stats["unknown_cells"] / max(1, stats["cells"])
        for r in results:
            radii = ",".join(f"{a[-1]}:{m['radius']}" for a, m in r["model"].items())
            sl = ""
            if r.get("sleep"):
                b = sum(v["bits_before"] for a in r["sleep"].values() for v in a.values())
                af = sum(v["bits_after"] for a in r["sleep"].values() for v in a.values())
                cells = {a: {rr: v["cells"] for rr, v in rr_.items()} for a, rr_ in r["sleep"].items()}
                sl = f" | sleep: {b:.0f} -> {af:.0f} bits, cells kept {[cells[a].get('1') for a in cells]} of 9"
            print(f"{name:<11}{r['level']:>6}{str(r['solved']):>8}{r['actions']:>9}{str(oracle[r['level']]):>8} | "
                  f"{acc:.2f}, {unk:.3f}, {radii}{sl}")
        solved = sum(r["solved"] for r in results)
        print(f"{name:<11} {solved}/{n_levels} levels in {secs:.0f}s; plans: {stats['goal_plans']} toward a goal, "
              f"{stats['explore_plans']} exploratory; re-plans after a wrong prediction: {stats['replans']}")
        report[name] = dict(levels=results, oracle=oracle, stats=stats, seconds=secs, solved=solved, n_levels=n_levels)
    json.dump(report, open(out / "play.json", "w"), indent=1, default=str)
    print(f"\ntotal: {sum(r['solved'] for r in report.values())}/{sum(r['n_levels'] for r in report.values())} levels")


if __name__ == "__main__":
    main()
