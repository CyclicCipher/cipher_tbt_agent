"""E29 — learned search: thinking as acting (DESIGN §18). The imagination -- imagined paths enumerated by their
description length under a learned strategy, breadth-first until the strategy has learned anything -- against the
same enumeration with a strategy that never learns (= breadth-first search, E27's planner), at the same budget of
model calls per real step. The run logged in RESULTS.md was made against E27's `Player.search` itself (commit
78d2dbb's code plus this experiment); that BFS was then deleted, since the empty strategy reproduces it.

Two parts. (a) The four replica games whose goal is learnable on the first level (LockPath, MultiKey, CollectAll,
Toggle): real actions per level must not get worse. (b) Rooms: LockPath's level 0, then three empty rooms of side 12,
20 and 40 with the goal in the far corner; per level, the model calls to the first imagined success at the level's
first plan -- the strategy's directedness -- and whether the room of 40 (1,600 positions, beyond what 4,000 calls
can expand breadth-first) is solved. Pre-registered in DESIGN §11.

    python experiments/ziplearn/e29.py            (CPU, ~10 min; run detached) -> runs/e29/e29.json
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
from tasks.games import LockPath, MultiKey, CollectAll, Toggle       # noqa: E402
from tasks.games.lockpath import _LEVELS                             # noqa: E402
from tasks.harness import Environment                                # noqa: E402
from tasks.oracle import solve_level                                 # noqa: E402

GAMES = {"LockPath": LockPath, "MultiKey": MultiKey, "CollectAll": CollectAll, "Toggle": Toggle}


def room(n):
    """An empty room of side n, the agent in the top-left corner and the goal in the bottom-right, as level 0 has them."""
    rows = ["#" * n]
    for i in range(1, n - 1):
        row = ["."] * n
        row[0] = row[-1] = "#"
        if i == 1:
            row[1] = "A"
        if i == n - 2:
            row[n - 2] = "G"
        rows.append("".join(row))
    rows.append("#" * n)
    return rows


def oracle_lengths(game, n_levels):
    out = []
    for lvl in range(n_levels):
        game.load_level(lvl)
        sol = solve_level(game)
        out.append(len(sol) if sol is not None else None)
    return out


def summarise(results):
    """Per level: solved, actions, the first plan's calls to its first success, mean calls per plan, the strategy after sleep."""
    rows = []
    for r in results:
        th = r.get("think", [])
        firsts = [t["first"] for t in th if t["first"] is not None]
        rows.append(dict(level=r["level"], solved=r["solved"], actions=r["actions"], plans=len(th),
                         first_plan_calls=(th[0]["first"] if th else None), first_plan_shortest=(th[0]["shortest"] if th else None),
                         hits=len(firsts), mean_calls=(sum(t["calls"] for t in th) / len(th) if th else None),
                         strategy=r.get("sleep_policy")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--calls", type=int, default=4000)
    ap.add_argument("--out", default=str(HERE / "runs" / "e29"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"games": {}, "rooms": {}}
    # (a) the games
    print("E29 (a) the four goal-learnable games, learned imagination vs breadth-first search, sleep on, "
          f"{args.budget} actions/level, {args.calls} calls/step")
    totals = {}
    for planner in ("learned", "bfs"):
        report["games"][planner] = {}
        total = 0
        for name, cls in GAMES.items():
            t0 = time.time()
            results, stats = play(Environment(cls()), budget_per_level=args.budget, sleep=True, strategy=(planner == "learned"), max_nodes=args.calls)
            secs = time.time() - t0
            rows = summarise(results)
            total += sum(r["actions"] for r in results)
            report["games"][planner][name] = dict(levels=rows, seconds=secs, goal_plans=stats["goal_plans"], calls=stats["calls"])
            cells = []
            for r in rows:
                s = f"L{r['level']} {'won' if r['solved'] else 'LOST'} in {r['actions']}"
                if r["first_plan_calls"] is not None:
                    s += f" (first plan: success after {r['first_plan_calls']} calls)"
                cells.append(s)
            print(f"   {planner:<8}{name:<11}" + "  ".join(cells) + f"  [{secs:.0f}s, {stats['goal_plans']} goal plans]")
        totals[planner] = total
        print(f"   {planner}: {total} actions in total")
    ratio_a = totals["learned"] / max(1, totals["bfs"])
    # (b) the rooms
    print("\nE29 (b) rooms: LockPath level 0, then empty rooms of side 12, 20, 40 (goal in the far corner); depth unlimited")
    levels = [_LEVELS[0], room(12), room(20), room(40)]
    oracle = oracle_lengths(LockPath(levels=levels), 4)
    print(f"   oracle: {oracle}")
    for planner in ("learned", "bfs"):
        budgets = [args.budget] * 4 if planner == "learned" else [args.budget, args.budget, args.budget, 3]
        t0 = time.time()
        results, stats = play(Environment(LockPath(levels=levels)), budget_per_level=budgets, sleep=True, strategy=(planner == "learned"),
                              max_nodes=args.calls, max_depth=1000)
        secs = time.time() - t0
        rows = summarise(results)
        report["rooms"][planner] = dict(levels=rows, oracle=oracle, seconds=secs, calls=stats["calls"])
        for r in rows:
            s = r["strategy"]
            first = ("no success" if r["first_plan_calls"] is None
                     else f"success after {r['first_plan_calls']} calls, plan of {r['first_plan_shortest']}")
            mean = 0 if r["mean_calls"] is None else r["mean_calls"]
            line = (f"   {planner:<8}room level {r['level']} (oracle {oracle[r['level']]}): {'won' if r['solved'] else 'LOST'} in "
                    f"{r['actions']} actions; first plan: {first}; {r['plans']} plans, mean {mean:.0f} calls")
            if s:
                line += f"; strategy after sleep: {s['features']} features, {s['contexts']} contexts, {s['bits_after']:.0f} bits"
            print(line)
        print(f"   {planner}: {secs:.0f}s")
    # verdict
    L, B = report["rooms"]["learned"]["levels"], report["rooms"]["bfs"]["levels"]
    def calls_at(rows, lvl):
        return next((r["first_plan_calls"] for r in rows if r["level"] == lvl), None)
    room40 = next((r["solved"] for r in L if r["level"] == 3), False)
    c20, c40, b20 = calls_at(L, 2), calls_at(L, 3), calls_at(B, 2)
    b_ok = (c20 is not None and b20 is not None and c20 < b20 and c20 <= 3 * 4 * oracle[2]) and (c40 is not None and c40 <= 3 * 4 * oracle[3])
    if ratio_a > 1.25 or not room40:
        verdict = "REFUTED"
    elif ratio_a <= 1.1 and b_ok and room40:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\nE29 verdict: {verdict} — (a) actions learned/bfs = {ratio_a:.3f}; (b) calls to first success: room 20 learned {c20} vs bfs {b20}, "
          f"room 40 learned {c40} (bounds 3x4xoracle: {12 * oracle[2]}, {12 * oracle[3]}); (c) room 40 solved: {room40}")
    report["verdict"] = dict(verdict=verdict, ratio_a=ratio_a, c20=c20, c40=c40, b20=b20, room40=room40)
    json.dump(report, open(out / "e29.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
