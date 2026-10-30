"""Part 7: chained expressions of one precedence level, zero-shot (docs/LAB.md part 7). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a7 [--out arc_agent_v2/runs/a7.json]   (the runs)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a7.py -q                          (a fast check)

**The checks:**
- **H1:** single equations.
- **H2:** chains of 2, 3 and 4 operations, never trained on.
- **H4:** the phases merged; every move of the frame is the world's.
- **H5:** the control, with the phase merge off.
- **H6:** parts 3, 4 (walks), 5 and 6 reproduce.
- **H7:** the cost.

The world's labels are used here only, to check and to report.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import ARITHMETIC, BrainConfig
from arc_agent_v2.tests.test_a6 import labels, truth
from arc_agent_v2.worlds.arithmetic_world import ArithmeticWorld

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
SEEDS = (0, 1, 2)
LENGTHS = (2, 3, 4)


def trained(p, seed, phase_merge=True):
    world = ArithmeticWorld(p=p, ops=("+", "-"), train_frac=0.5, seed=seed)
    brain = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed), phase_merge=phase_merge)
    t = time.time()
    for ep in world.epoch(np.random.default_rng(seed)):
        brain.episode(ep)
    brain.sleep()
    return world, brain, time.time() - t


def mechanism(world, brain):
    """H4: which states' phases merged, and whether every move of the frame is the world's."""
    f, fac = brain.frame, brain.frame.factored
    lab, conflicts = labels(world, f)
    name = {v: k for k, v in world.sdr_of.items()}
    wrong = sum(1 for u, x, v in f.moves if u in lab and truth(lab[u], name[x], world) != lab.get(v))
    out = dict(factored=fac is not None, wrong=wrong, conflicts=conflicts, why=f.unfactored)
    if fac is None:
        return out
    phases = defaultdict(set)                           # a kind of state -> the phases its places are in
    for u, s in lab.items():
        phases["B" + s[1] if s[0] == "B" else s[0]].add(fac.phase[u])
    one = lambda *kinds: len(set().union(*(phases[k] for k in kinds))) == 1
    out.update(phases=fac.phases, start_with_plus=one("S", "B+"), digit_with_result=one("A", "C"),
               minus_apart=not (phases["B-"] & (phases["S"] | phases["A"])))
    return out


def chains(p, seed):
    """H1, H2, H4, H5 and H7 for one world: the brain as built, and with the phase merge off."""
    out = dict(p=p, seed=seed)
    for arm, merge in (("merge", True), ("off", False)):
        world, brain, seconds = trained(p, seed, merge)
        out[arm + "_seconds"] = round(seconds, 1)
        for op in world.ops:
            qs = [q for q in world.queries() if q[2][0] == op]
            out["%s_single%s" % (arm, op)] = sum(world.decode(brain.predict_after(pre)) == world.written(*e)[-1]
                                                 for pre, _, e in qs) / len(qs)
        for n in LENGTHS:
            out["%s_chain%d" % (arm, n)] = world.score_chains(brain.predict_after, n, 1000, seed)
        if merge:
            out.update(mechanism(world, brain))
    return out


def regression():
    """H6: parts 3, 4 (walks), 5 and 6 against their recorded runs. Part 5's phase count is reported, not compared."""
    from arc_agent_v2.tests.test_a1 import run as run_a1
    from arc_agent_v2.tests.test_a4 import EXCURSION, shortcut, walk_checks
    from arc_agent_v2.tests.test_a5 import answers, one_fact
    from arc_agent_v2.tests.test_a6 import a2, reversal
    plain = lambda o: o.item() if isinstance(o, np.generic) else str(o)   # numpy scalars as numbers, not strings
    norm = lambda r: json.loads(json.dumps(r, default=plain))
    diff = lambda old, new, skip=(): {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
                                      if k not in skip and old.get(k) != new.get(k)}
    run = lambda name: json.load(open(os.path.join(HERE, "runs", name)))
    res = dict(part3=[], part4=dict(walk=[], shortcut=[]), part5=[], part6=[])
    f3 = ("held_out_acc", "train_acc", "inferred", "places_after_eq", "places_after_b", "displacements",
          "full_on_C", "orders", "eq_one_to_one")
    for old in run("a1.json"):
        if old["arm"] == "full" and old["op"] == "+" and old["frac"] == 0.5 and old["seed"] in SEEDS:
            new = run_a1(old["p"], 0.5, old["seed"], "full")
            res["part3"].append(dict(p=old["p"], seed=old["seed"], diff={k: (old[k], new[k]) for k in f3
                                                                         if old[k] != new[k]}))
    base4 = run("a4.json")
    for old in base4["walk"]:
        res["part4"]["walk"].append(dict(kind=old["kind"], seed=old["seed"],
                                         diff=diff(old, norm(walk_checks(old["kind"], old["seed"])))))
    for kind in EXCURSION:
        trials = [t for s in SEEDS for t in shortcut(kind, s)["trials"]]
        res["part4"]["shortcut"].append(dict(kind=kind, same=trials == [t for r in base4["shortcut"]
                                                                         if r["kind"] == kind for t in r["trials"]]))
    base5 = run("a5.json")
    for old in base5["one_fact"]:
        new = norm(one_fact(old["p"], old["seed"]))
        res["part5"].append(dict(check="F1-F2", p=old["p"], seed=old["seed"], diff=diff(old, new, ("phases",)),
                                 phases=(old.get("phases"), new.get("phases"))))
    for old in base5["withheld"]:
        res["part5"].append(dict(check="F3", p=old["p"], seed=old["seed"],
                                 diff=diff(old, norm(answers(old["p"], old["seed"], withhold=True)))))
    base6 = run("a6.json")
    for old in base6["a2"]:
        res["part6"].append(dict(check="A2", p=old["p"], seed=old["seed"],
                                 diff=diff(old, norm(a2(old["p"], old["seed"])), ("seconds",))))
    for old in base6["reversal"]:
        res["part6"].append(dict(check="reversal", p=old["p"], seed=old["seed"],
                                 diff=diff(old, norm(reversal(old["p"], old["seed"])),
                                           ("relation_seconds", "no_relation_seconds"))))
    return res


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a7.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save = lambda: json.dump(res, open(out_path, "w"), indent=1, default=str)
    say = lambda *a: print(*a, flush=True)
    t0 = time.time()
    res = dict(chains=[chains(p, s) for p in (13, 97) for s in SEEDS])
    res["seconds_part7"] = round(time.time() - t0, 1)
    save()                                              # part 7's own runs, kept whatever the regression does
    for r in res["chains"]:
        say("H1 p=%-3d seed %d: single + %.3f, - %.3f" % (r["p"], r["seed"], r["merge_single+"], r["merge_single-"]))
    for r in res["chains"]:
        say("H2 p=%-3d seed %d: chains of 2, 3, 4 operations %.3f, %.3f, %.3f" % (
            r["p"], r["seed"], r["merge_chain2"], r["merge_chain3"], r["merge_chain4"]))
    for r in res["chains"]:
        say("H4 p=%-3d seed %d: factored %s; phases %s; start with after-+ %s; after a digit with after a op b %s; "
            "after - apart %s; moves wrong %d; places with two states %d" % (
                r["p"], r["seed"], r["factored"], r.get("phases"), r.get("start_with_plus"),
                r.get("digit_with_result"), r.get("minus_apart"), r["wrong"], r["conflicts"]))
    for r in res["chains"]:
        say("H5 p=%-3d seed %d: phase merge off: chains %.3f, %.3f, %.3f; singles + %.3f, - %.3f" % (
            r["p"], r["seed"], r["off_chain2"], r["off_chain3"], r["off_chain4"], r["off_single+"], r["off_single-"]))
    for r in res["chains"]:
        say("H7 p=%-3d seed %d: training and sleep %.1f s (merge off %.1f s)" % (
            r["p"], r["seed"], r["merge_seconds"], r["off_seconds"]))
    say("part 7's own runs %.1f s" % res["seconds_part7"])
    t1 = time.time()
    res["regression"] = g = regression()
    res["seconds_regression"] = round(time.time() - t1, 1)
    save()
    for r in g["part3"]:
        say("H6 part 3 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["walk"]:
        say("H6 part 4 %-9s seed %d: %s" % (r["kind"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["shortcut"]:
        say("H6 part 4 shortcut %-9s: trials identical %s" % (r["kind"], r["same"]))
    for r in g["part5"]:
        say("H6 part 5 %s p=%-3d seed %d: %s%s" % (r["check"], r["p"], r["seed"], r["diff"] or "reproduced",
                                                  "; phases %s -> %s" % r["phases"] if "phases" in r else ""))
    for r in g["part6"]:
        say("H6 part 6 %-8s p=%-3d seed %d: %s" % (r["check"], r["p"], r["seed"], r["diff"] or "reproduced"))
    say("the regression %.1f s" % res["seconds_regression"])


def test_a7_small():
    """A fast check on p = 13: chains are answered zero-shot, and not without the phase merge."""
    r = chains(13, 0)
    assert r["merge_single+"] == 1.0 and r["merge_single-"] == 1.0
    assert min(r["merge_chain%d" % n] for n in LENGTHS) >= 0.99
    assert max(r["off_chain%d" % n] for n in LENGTHS) <= 0.05
    assert r["start_with_plus"] and r["digit_with_result"] and r["minus_apart"] and r["wrong"] == 0


if __name__ == "__main__":
    main(sys.argv[1:])
