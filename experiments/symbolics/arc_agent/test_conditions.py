"""Part 20 (docs/AGENT.md): can the brain learn when an action's effect is stopped or changed, with no rule saying so?
Seven hand-built worlds (worlds.py), a 300-action random walk each, 3 seeds; two arms: ATTN (attention over context
experts) and UNCOND (the unconditional expert only). Before each action the brain predicts the avatar's next
position; accuracy is counted over actions 101-300.
    python -m arc_agent.test_conditions      (from experiments/symbolics, with venv312)
"""
import json
import time

import numpy as np

from arc_agent.brain import Brain
from arc_agent.worlds import WORLDS, World

STEPS, FROM, SEEDS = 300, 100, 3


def avatar(brain):
    cols = [c for c in brain.planning() if c.frame.kind == "translation"]
    return max(cols, key=lambda c: c.uses) if cols else None


def run(name, conditional, seed):
    rng = np.random.default_rng(seed)
    w = World(name, rng)
    brain = Brain(conditional=conditional)
    g = w.frame()
    brain.start(g, True)
    hits, total = 0, 0
    for t in range(STEPS):
        a = int(rng.choice(w.actions()))
        action = (a, None, None)
        c = avatar(brain)
        pred = None
        if c is not None:
            others = {o.cid: o.state for o in brain.planning()}
            ctx = brain.context(c, c.state, g, others, brain.v1.under_at)
            p, _ = c.dist(c.akey(action, c.state), ctx)
            pred = (c.cid, c.frame.apply(c.state, max(p, key=p.get)))
        g2 = w.step(a)
        brain.observe(g, action, g2, ("key", a))
        if t >= FROM:
            total += 1
            hits += pred is not None and any(o.cid == pred[0] and o.active and o.state == pred[1] for o in brain.columns)
        g = g2
    c = avatar(brain)
    attended = {}
    if c is not None:
        cols = {o.cid: o for o in brain.columns}
        for k in (1, 2, 3, 4):
            b = c.keys.get(("key", k))
            if b is not None:
                f = b.attended()
                if f is not None and f[0] == "col" and f[1] in cols:
                    f = ("col", cols[f[1]].frame.kind)
                attended[k] = f
    return dict(accuracy=hits / total, attended=attended, columns=len([o for o in brain.columns if o.active]))


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


def main():
    t0 = time.time()
    out = {}
    for name in WORLDS:
        for arm, conditional in (("ATTN", True), ("UNCOND", False)):
            runs = [run(name, conditional, s) for s in range(SEEDS)]
            out[f"{name} / {arm}"] = runs
            acc = [r["accuracy"] for r in runs]
            c3 = [np.mean([side_ok(name, k, f) for k, f in r["attended"].items()]) for r in runs if r["attended"]]
            print(f"{name:16s} {arm:6s} accuracy {[round(a, 3) for a in acc]} mean {np.mean(acc):.3f} | attended on the "
                  f"right side {np.mean(c3) if c3 else float('nan'):.2f} | e.g. {runs[0]['attended']} | "
                  f"{time.time() - t0:.0f} s", flush=True)
    json.dump(out, open("runs/test_conditions.json", "w"), indent=1, default=str)
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
