"""Part 5: one fact, two uses (docs/LAB.md part 5). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a5 [--out arc_agent_v2/runs/a5.json]   (the pre-registered runs)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a5.py -q                           (a fast check)

**The checks:**
- **F1:** one coordinate across the rings.
- **F2:** the digit's two uses.
- **F3:** answers never seen as answers.
- **F4:** relabelled answers, where naming must be refused.
- **F5:** parts 3 and 4 reproduce.
- **F6:** the cost.

The world's labels are used here only, to name inputs for the report and to choose the prefixes walked.
"""
import json
import os
import sys
import time

import numpy as np

from arc_agent_v2.worlds.arithmetic_world import ArithmeticWorld
from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import ARITHMETIC, BrainConfig
from arc_agent_v2.tests.test_a1 import run as run_a1
from arc_agent_v2.tests.test_a4 import EXCURSION, KINDS, a1 as a4_a1, shortcut, summarize, walk_checks

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
SEEDS = (0, 1, 2)


def trained(p, seed, factoring=True, **kw):
    world = ArithmeticWorld(p=p, train_frac=0.5, seed=seed, **kw)
    brain = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed), factoring=factoring)
    for ep in world.epoch(np.random.default_rng(seed)):
        brain.episode(ep)
    brain.sleep()
    return world, brain


def fac_after(brain, prefix):
    brain.episode(prefix, learn=False)
    return brain.columns()[0].l6.fac


def one_fact(p, seed):
    """F1 and F2 on standard A1."""
    world, brain = trained(p, seed)
    fac = brain.frame.factored
    out = dict(p=p, seed=seed, factored=fac is not None, why=brain.frame.unfactored)
    if fac is None:
        return out
    m = fac.modules
    label = {v: k for k, v in world.sdr_of.items()}
    single = {z: fac_after(brain, [world.sdr_of[str(z)]]) for z in range(p)}
    after = {a: fac_after(brain, [world.sdr_of[str(a)]]) for a in range(p)}
    same = shift_same = total = 0
    for a in range(p):
        for b in range(p):
            ab = fac_after(brain, world.episode("+", a, b)[:3])
            total += 1
            same += ab is not None and ab[1] == single[(a + b) % p][1]
            d_arg = m.unbind(ab[1], after[a][1]) if ab is not None else None   # b's shift as the second argument
            d_pos = m.unbind(single[b][1], fac.start[1])                    # b's shift from the start
            shift_same += d_arg == d_pos
    leaves = {q for _, _, q in brain.frame.moves} - {pl for pl, _, _ in brain.frame.moves}
    places = {pl for pl, _, _ in brain.frame.moves} | {q for _, _, q in brain.frame.moves}
    out.update(phases=fac.phases, same_coordinate=same / total, digit_two_uses=shift_same / total,
               identities=sorted(label.get(x, "?") for x in fac.identities),
               naming=[label.get(x, "?") for x in fac.naming],
               start_is_zero=fac.start[1] == single[0][1],
               coordinated=sum(1 for q in places - leaves if q in fac.coord) / max(1, len(places - leaves)))
    return out


def answers(p, seed, **kw):
    """F3 and F4: accuracy on the test equations, with step 3 and without (part 4's brain)."""
    out = dict(p=p, seed=seed, **kw)
    for name, fac in (("step3", True), ("part4", False)):
        world, brain = trained(p, seed, factoring=fac, **kw)
        out[name] = world.score(brain.predict_after)
        if fac:
            f = brain.frame.factored
            out["naming"] = [] if f is None else len(f.naming)
    return out


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a5.json")
    t0 = time.time()
    res = dict(one_fact=[one_fact(p, s) for p in (13, 97, 105) for s in SEEDS])
    res["withheld"] = [answers(p, s, withhold=True) for p in (13, 97) for s in SEEDS]
    res["relabelled"] = [answers(p, s, permute=True) for p in (13, 97) for s in SEEDS] + \
                        [answers(p, s, withhold=True, permute=True) for p in (13, 97) for s in SEEDS]
    res["seconds_part5"] = round(time.time() - t0, 1)
    t1 = time.time()                                    # F5: parts 3 and 4 reproduce
    base3 = {(r["p"], r["seed"]): r for r in json.load(open(os.path.join(HERE, "runs", "a1.json")))
             if r["arm"] == "full" and r["op"] == "+" and r["frac"] == 0.5}
    f3 = ("held_out_acc", "train_acc", "inferred", "places_after_eq", "places_after_b", "displacements",
          "full_on_C", "orders", "eq_one_to_one")
    res["part3"] = []
    for p in (13, 97):
        for s in SEEDS:
            r = run_a1(p, 0.5, s, "full")
            res["part3"].append(dict(p=p, seed=s, diff={k: (base3[(p, s)][k], r[k]) for k in f3 if r[k] != base3[(p, s)][k]}))
    base4 = json.load(open(os.path.join(HERE, "runs", "a4.json")))
    res["part4"] = dict(a1=[], walk=[], shortcut=[])
    for old in base4["a1"]:
        new = a4_a1(old["p"], old["seed"])
        res["part4"]["a1"].append(dict(p=old["p"], seed=old["seed"], diff={k: (old[k], new.get(k)) for k in old
                                                                           if old[k] != new.get(k)}))
    for old in base4["walk"]:
        new = walk_checks(old["kind"], old["seed"])
        res["part4"]["walk"].append(dict(kind=old["kind"], seed=old["seed"],
                                         diff={k: (old[k], new[k]) for k in old if old[k] != new[k]}))
    for kind in EXCURSION:
        trials = [t for s in SEEDS for t in shortcut(kind, s)["trials"]]
        old = [t for r in base4["shortcut"] if r["kind"] == kind for t in r["trials"]]
        res["part4"]["shortcut"].append(dict(kind=kind, same=trials == old,
                                             arms={a: summarize(trials, a) for a in "abcd"}))
    res["seconds_regression"] = round(time.time() - t1, 1)

    for r in res["one_fact"]:
        print("F1-F2 p=%-3d seed %d: factored %s, phases %s, same coordinate %s, digit's two uses %s, identities %s, "
              "naming %s, start = zero's position %s, places coordinated %s" % (
                  r["p"], r["seed"], r["factored"], r.get("phases"), r.get("same_coordinate"), r.get("digit_two_uses"),
                  r.get("identities"), r.get("naming"), r.get("start_is_zero"), r.get("coordinated")))
    for r in res["withheld"]:
        print("F3 withheld p=%-3d seed %d: step 3 %.3f, part 4's brain %.3f (naming %s)" % (
            r["p"], r["seed"], r["step3"], r["part4"], r["naming"]))
    for r in res["relabelled"]:
        print("F4 %-30s p=%-3d seed %d: step 3 %.3f, part 4's brain %.3f, namings accepted %s" % (
            "relabelled + withheld" if r.get("withhold") else "relabelled", r["p"], r["seed"], r["step3"], r["part4"],
            r["naming"]))
    for r in res["part3"]:
        print("F5 part 3 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "reproduced"))
    for r in res["part4"]["a1"]:
        print("F5 part 4 A1 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "reproduced"))
    for r in res["part4"]["walk"]:
        print("F5 part 4 %-9s seed %d: %s" % (r["kind"], r["seed"], r["diff"] or "reproduced"))
    for r in res["part4"]["shortcut"]:
        print("F5 part 4 shortcut %-9s: trials identical %s; %s" % (r["kind"], r["same"], r["arms"]))
    print("part 5's own runs %.1f s; the regression %.1f s" % (res["seconds_part5"], res["seconds_regression"]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1, default=str)


def test_a5_small():
    """A fast check: on p = 13 the rings share one coordinate, and withheld answers are given."""
    r = one_fact(13, 0)
    assert r["factored"] and r["phases"] == 6 and r["same_coordinate"] == 1.0 and r["digit_two_uses"] == 1.0
    assert sorted(r["identities"]) == ["+", "0", "="]
    a = answers(13, 0, withhold=True)
    assert a["step3"] == 1.0 and a["part4"] == 0.0


if __name__ == "__main__":
    main(sys.argv[1:])
