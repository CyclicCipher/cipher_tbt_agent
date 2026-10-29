"""A1: arithmetic with no operation given (docs/LAB.md part 3). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a1 [--out arc_agent_v2/runs/a1.json]   (the pre-registered runs)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a1.py -q                           (a fast check)

Each run:
1. builds a world (addition mod p, a random split of the equations) and a brain;
2. shows the brain every training equation once;
3. lets it sleep (consolidation), then scores its predictions after "=".

Arms:
- full;
- memorize-only: no consolidation, so the hippocampus alone.

The frame checks (M4) look inside the frame. They are the only part that uses the world's labels, to name places for
the report; the brain never sees them.
"""
import json
import os
import sys
import time
import numpy as np

from arc_agent_v2.worlds.arithmetic_world import ArithmeticWorld
from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import ARITHMETIC, BrainConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/


def run(p, frac, seed, arm, op="+"):
    t0 = time.time()
    world = ArithmeticWorld(p=p, ops=(op,), train_frac=frac, seed=seed)
    brain = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed),
                  consolidation=(arm == "full"))
    for ep in world.epoch(np.random.default_rng(seed)):
        brain.episode(ep)
    frame = brain.sleep()
    out = dict(p=p, frac=frac, seed=seed, arm=arm, op=op, train=len(world.train), held_out=len(world.held_out),
               held_out_acc=world.score(brain.predict_after), train_acc=world.score(brain.predict_after, held_out=False))
    out["seconds"] = round(time.time() - t0, 1)         # M5: learning, sleep and scoring (not the checks below)
    if frame is not None:
        out.update(inferred=frame.inferred, **frame_checks(world, brain, op))
    return out


def place_after(brain, prefix):
    brain.begin()
    for x in prefix:
        brain.step(x, learn=False)
    return brain.columns()[0].l6.place


def frame_checks(world, brain, op="+"):
    """M4: (a) the places after "a + b =" number p; (b) the displacements on the places after "a + b": how many are
    full permutations of them, and their orders; (c) "=" carries those places one-to-one onto the places after "=".
    The frame's own displacements are read; the labels only pick which prefixes to walk."""
    eqs = [(op, a, b) for a in range(world.p) for b in range(world.p)]
    C = {place_after(brain, world.episode(*eq)[:3]) for eq in eqs} - {None}
    D = {place_after(brain, world.episode(*eq)[:4]) for eq in eqs} - {None}
    eq_move = {}
    col = brain.columns()[0]
    for c in C:
        eq_move[c] = col.l5.move(c, world.sdr_of["="])
    orders, full = [], 0
    rename = {}                                         # part 4: coordinated places are named by their codes
    for orbit in brain.frame.coords:
        rename.update(orbit.codes)
    for m in brain.frame.displacements:
        m = {rename.get(s, s): rename.get(t, t) for s, t in m.items()}
        perm = {s: t for s, t in m.items() if s in C and t in C}
        if len(perm) == len(C) and len(C) > 0:
            full += 1
            s0 = next(iter(C))
            k, s = 1, perm[s0]
            while s != s0 and k <= len(C):
                s, k = perm[s], k + 1
            orders.append(k)
    return dict(places_after_eq=len(D), places_after_b=len(C), displacements=len(brain.frame.displacements),
                full_on_C=full, orders=sorted(set(orders)),
                eq_one_to_one=len(set(eq_move.values()) - {None}) == len(C) and None not in eq_move.values())


def positions(p, frac=0.5, seed=0, arm="full"):
    """The brain's prediction at every position of the held-out equations, not only after "=": the fraction right
    after k symbols, k = 0..4 (what comes next: a, "+", b, "=", c). Positions 0 and 2 are the world's free choices;
    1, 3 and 4 are determined by what came before."""
    world = ArithmeticWorld(p=p, train_frac=frac, seed=seed)
    brain = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed),
                  consolidation=(arm == "full"))
    for ep in world.epoch(np.random.default_rng(seed)):
        brain.episode(ep)
    brain.sleep()
    hits = np.zeros(5)
    for eq in world.held_out:
        ep, written = world.episode(*eq), world.written(*eq)
        for k in range(5):
            hits[k] += world.decode(brain.predict_after(ep[:k])) == written[k]
    return (hits / len(world.held_out)).round(3).tolist()


PLAN = ([(13, 0.5), (97, 0.5)], [(13, 0.3), (97, 0.1)])  # M1/M2; M3 (full only)
CONTROL = [(13, 0.5), (97, 0.5)]                            # the apparatus check: "?", a random table (full only)


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a1.json")
    jobs = [(p, f, s, arm) for p, f in PLAN[0] for s in range(3) for arm in ("full", "memorize")]
    jobs += [(p, f, s, "full") for p, f in PLAN[1] for s in range(3)]
    jobs += [(p, f, s, "full", "?") for p, f in CONTROL for s in range(3)]
    t0 = time.time()
    res = [run(*job) for job in jobs]                   # one process: each run is cheap
    for r in res:
        print({k: v for k, v in r.items()})
    print("\n%-4s %-5s %-9s %-26s %-26s" % ("p", "frac", "arm", "held-out (3 seeds)", "train"))
    for op, plan in (("+", PLAN[0] + PLAN[1]), ("?", CONTROL)):
        for p, f in plan:
            for arm in ("full", "memorize"):
                rs = [r for r in res if r["p"] == p and r["frac"] == f and r["arm"] == arm and r["op"] == op]
                if rs:
                    print("%-4d %-5.2f %-9s %-26s %-26s %s" % (p, f, arm + ("" if op == "+" else " ?"),
                                                              [round(r["held_out_acc"], 3) for r in rs],
                                                              [round(r["train_acc"], 3) for r in rs],
                                                              [r["seconds"] for r in rs]))
    print("total %.1f s wall" % (time.time() - t0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)


def test_a1_small():
    """A fast check on a small group: the full brain generalizes; memorizing alone does not."""
    full, mem = run(7, 0.6, 0, "full"), run(7, 0.6, 0, "memorize")
    assert full["train_acc"] == 1.0 and mem["train_acc"] == 1.0
    assert full["held_out_acc"] == 1.0 and mem["held_out_acc"] < 0.2


if __name__ == "__main__":
    main(sys.argv[1:])
