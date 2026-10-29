"""Part 29 (docs/AGENT.md): does the arc agent plan its way to goals? The goal worlds of goal_worlds.py, played by the
whole ArcAgent (no priors from other games), CAP actions per level; a level not won by then ends the run. Arms switch
the executive's inverse model and viability off. Reports levels won, actions per won level, and (G1) the steps with the
box stuck before its level is won. Runs in parallel.
    python -m arc_agent.test_planning [--seeds 3] [--worlds G1,G2] [--arms full,no-inverse,no-viability] [--out f]
(from experiments/symbolics, with venv312)
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from arc_agent.goal_worlds import GOAL_WORLDS, LEVELS, GoalWorld

CAP, SEEDS, ARMS = 250, 3, ("full", "no-inverse", "no-viability")


def run(name, arm, seed):
    import arc_agent.executive as executive
    from arc_agent.agent import ArcAgent
    executive.MAX_NODES, executive.BUDGET = 400, 5.0    # the fast setting's node budget (results don't depend on load)
    executive.INVERSE, executive.VIABLE = arm != "no-inverse", arm != "no-viability"
    w = GoalWorld(name, seed)
    agent = ArcAgent(name, priors={})
    fd = w.frame_data()
    won, used, stuck, total, t0 = [], 0, 0, 0, time.time()
    while not w.won and used < CAP:
        a, data = agent.act(fd)
        level = w.level
        fd = w.step(a)
        used += 1
        total += 1
        stuck += w.stuck()
        if w.level > level:
            won.append(used)
            used = 0
    return dict(name=name, arm=arm, seed=seed, won=len(won), actions=won, stuck=stuck, total=total,
                notes=dict(agent.notes), seconds=round(time.time() - t0, 1))


def _job(args):
    return run(*args)


def main(argv):
    opts = {"seeds": str(SEEDS), "worlds": None, "arms": ",".join(ARMS), "out": None}
    for k in list(opts):
        if f"--{k}" in argv:
            opts[k] = argv[argv.index(f"--{k}") + 1]
    worlds = [w for w in GOAL_WORLDS if not opts["worlds"] or w.split()[0] in opts["worlds"].split(",")]
    jobs = [(w, a, s) for w in worlds for a in opts["arms"].split(",") for s in range(int(opts["seeds"]))]
    t0 = time.time()
    with ProcessPoolExecutor(8, max_tasks_per_child=1) as pool:
        res = list(pool.map(_job, jobs))
    for w in worlds:
        for arm in opts["arms"].split(","):
            rs = sorted([r for r in res if r["name"] == w and r["arm"] == arm], key=lambda r: r["seed"])
            print(f"{w:18s} {arm:13s} levels won {sum(r['won'] for r in rs)} of {LEVELS * len(rs)} | actions per won "
                  f"level {[r['actions'] for r in rs]} | box stuck {[r['stuck'] for r in rs]} of steps "
                  f"{[r['total'] for r in rs]} | plans {rs[0]['notes']}", flush=True)
    if opts["out"]:
        json.dump(res, open(opts["out"], "w"), indent=1, default=str)
    print(f"total {time.time() - t0:.1f} s wall")


if __name__ == "__main__":
    main(sys.argv[1:])
