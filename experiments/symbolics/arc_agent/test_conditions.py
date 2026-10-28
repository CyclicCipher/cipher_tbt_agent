"""Parts 20 and 27 (docs/AGENT.md): can the brain learn when an action's effect is stopped or changed, with no rule
saying so, including by state that is not on the screen?

The hand-built worlds of worlds.py. W1-W10: a 300-action random walk, accuracy over actions 101-300. W11-W14 (hidden
state): 1,000 actions, accuracy over actions 501-1,000. Before each action the brain predicts the avatar's next position
(W14: both blocks' next positions; a hit needs both right). Seeds and worlds run in parallel.
    python -m arc_agent.test_conditions [--seeds 3] [--worlds W1,W11] [--no-hidden] [--uncond] [--out runs/x.json]
(from experiments/symbolics, with venv312)
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from arc_agent.brain import Brain
from arc_agent.worlds import HIDDEN_WORLDS, WORLDS, World

STEPS, FROM, SEEDS = 300, 100, 3
LONG, LONG_FROM = 1000, 500                             # the hidden-state worlds


def avatar(brain):
    cols = [c for c in brain.planning() if c.frame.kind == "translation"]
    return max(cols, key=lambda c: c.uses) if cols else None


def targets(brain, name):
    """The columns whose next state is predicted: the avatar, or in W14 every block (planned with or not: a block
    whose moves no action predicts is driven by its clock, and still scored)."""
    if name.startswith("W14"):
        return [c for c in brain.columns if c.active and c.frame.kind == "translation"]
    c = avatar(brain)
    return [c] if c is not None else []


def run(name, conditional=True, seed=0, hidden=True):
    rng = np.random.default_rng(seed)
    w = World(name, rng)
    if "hidden" in Brain.__init__.__code__.co_varnames:
        brain = Brain(conditional=conditional, hidden=hidden)
    else:                                               # an agent from before part 27, as a control arm
        brain = Brain(conditional=conditional)
    others = brain.others if hasattr(brain, 'others') else (lambda: {o.cid: o.state for o in brain.planning()})
    g = w.frame()
    brain.start(g, True)
    long = name in HIDDEN_WORLDS
    steps, start = (LONG, LONG_FROM) if long else (STEPS, FROM)
    hits, total = 0, 0
    for t in range(steps):
        a = int(rng.choice(w.actions()))
        action = (a, None, None)
        preds = []
        for c in targets(brain, name):
            ctx = brain.context(c, c.state, g, others(), brain.v1.under_at)
            p, _ = c.dist(c.akey(action, c.state), ctx)
            preds.append((c.cid, c.frame.apply(c.state, max(p, key=p.get))))
        g2 = w.step(a)
        brain.observe(g, action, g2, ("key", a))
        if t >= start:
            total += 1
            hits += bool(preds) and all(any(o.cid == cid and o.active and o.state == s for o in brain.columns)
                                        for cid, s in preds)
        g = g2
    names = {c.cid: c.frame.kind for c in brain.columns}
    if getattr(brain, 'hidden', None) is not None:
        names.update({c.cid: f"delay {-10 - c.cid}" for c in brain.hidden.delays})
        names[brain.hidden.graph_col.cid] = "graph"
    attended, situation = {}, {}
    cols = targets(brain, name)
    if name.startswith("W14"):
        cols = [c for c in cols if 12 in getattr(c.frame, "colours", [])]
    for c in cols[:1]:
        for k in (1, 2, 3, 4):
            b = c.keys.get(("key", k))
            if b is None:
                continue
            f = b.attended()
            attended[k] = ("col", names.get(f[1], f[1])) if (f is not None and f[0] == "col") else f
            d = b.situation() if hasattr(b, 'situation') else None
            situation[k] = ("col", names.get(d[1], d[1])) if (d is not None and d[0] == "col") else d
    return dict(name=name, seed=seed, accuracy=hits / max(total, 1), attended=attended, situation=situation,
                columns=len([o for o in brain.columns if o.active]))


def side_ok(name, k, f):
    """C3: is the attended feature on the side key k moves toward (a 3 x 3 avatar, anchor at its top-left pixel)?"""
    if f is None:
        return False
    if name.startswith("W6"):
        return f[0] == "col"
    if name.startswith("W9"):
        return f[0] == "mod" or (k != 1 and f[0] == "at")
    if name.startswith("W10"):
        return f[0] == "num"
    if name.startswith("W8"):
        return f[0] in ("at", None) or f is None
    if f[0] != "at":
        return False
    dx, dy = f[1], f[2]
    return {1: dy < 0, 2: dy > 2, 3: dx < 0, 4: dx > 2}[k]


def _job(args):
    return run(*args)


def main(argv):
    opts = {"seeds": str(SEEDS), "worlds": None, "out": None}
    for k in list(opts):
        if f"--{k}" in argv:
            opts[k] = argv[argv.index(f"--{k}") + 1]
    hidden, conditional = "--no-hidden" not in argv, "--uncond" not in argv
    worlds = [w for w in WORLDS + HIDDEN_WORLDS if not opts["worlds"] or w.split()[0] in opts["worlds"].split(",")]
    jobs = [(w, conditional, s, hidden) for w in worlds for s in range(int(opts["seeds"]))]
    jobs.sort(key=lambda j: j[0] not in HIDDEN_WORLDS)  # the long runs first
    t0 = time.time()
    with ProcessPoolExecutor(8, max_tasks_per_child=1) as pool:
        res = list(pool.map(_job, jobs))
    for w in worlds:
        rs = sorted([r for r in res if r["name"] == w], key=lambda r: r["seed"])
        acc = [r["accuracy"] for r in rs]
        print(f"{w:18s} accuracy {[round(a, 3) for a in acc]} mean {np.mean(acc):.3f} | most weighted situation "
              f"{[r['situation'] for r in rs]} | attended (seed 0) {rs[0]['attended']}", flush=True)
    if opts["out"]:
        json.dump(res, open(opts["out"], "w"), indent=1, default=str)
    print(f"total {time.time() - t0:.1f} s wall (hidden area {'on' if hidden else 'off'})")


if __name__ == "__main__":
    main(sys.argv[1:])
