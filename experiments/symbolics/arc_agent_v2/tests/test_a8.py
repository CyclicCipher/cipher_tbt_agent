"""Part 8: the hippocampus keyed by the cortex's location (docs/LAB.md part 8). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a8 [--out arc_agent_v2/runs/a8.json]   (the runs)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a8.py -q                          (a fast check)

**Day 1:** part 4's walk and sleep. **Day 2:** the same graph relabelled, one walk of 3n actions from node 0, learning
on; at each arrival the brain predicts the label before it is shown.

**The checks:**
- **I1–I3:** revisits and first visits, in three arms: (a) as built; (b) location keys off; (c) recency off.
- **I4:** parts 3, 4, 5 (F3), 6 and 7 reproduce.
- **I5:** the cost.

The world's nodes are used here only, to tell first visits from revisits and to score.
"""
import json
import os
import sys
import time

import numpy as np

from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import WALK
from arc_agent_v2.tests.test_a4 import STEPS
from arc_agent_v2.worlds.walk_world import KINDS, WalkWorld

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
SEEDS = (0, 1, 2)
ARMS = {"a": dict(), "b": dict(locations=False), "c": dict(today=False)}


def day2(kind, seed, **arm):
    """Day 1 as part 4's, then day 2 relabelled: counts of right predictions by kind of step."""
    world = WalkWorld(kind, seed=seed)
    brain = Brain(WALK, **arm)
    brain.episode(world.stream(0, np.random.default_rng((seed, 5)).integers(0, world.A, STEPS).tolist()))
    brain.sleep()
    world.relabel(np.random.default_rng((seed, 88)))
    acts = np.random.default_rng((seed, 89)).integers(0, world.A, 3 * world.n).tolist()
    nodes = world.nodes(0, acts)
    col = brain.columns()[0]
    out = {k: 0 for k in ("first", "first_right", "revisit", "revisit_right", "known", "known_right")}
    t = time.time()
    brain.begin()
    brain.step(world.labels[nodes[0]], "retina")
    seen = {nodes[0]}
    for a, v in zip(acts, nodes[1:]):
        brain.step(world.action[a], "motor")
        right = brain.predict("retina") == world.labels[v]
        kind_ = "revisit" if v in seen else "first"
        out[kind_] += 1
        out[kind_ + "_right"] += right
        if kind_ == "revisit" and not col.l6.lost():
            out["known"] += 1
            out["known_right"] += right
        brain.step(world.labels[v], "retina")
        seen.add(v)
    brain.finish()
    out["seconds"] = round(time.time() - t, 2)
    out["lost_at_end"] = col.l6.lost()
    return out


def walks(kind, seed):
    rec = dict(kind=kind, seed=seed)
    for name, arm in ARMS.items():
        rec[name] = day2(kind, seed, **arm)
    return rec


def regression():
    """I4: the predictions of parts 3, 5 (F3), 6 and 7, and part 4's walk checks and shortcut tests."""
    from arc_agent_v2.tests.test_a1 import run as run_a1
    from arc_agent_v2.tests.test_a4 import EXCURSION, shortcut, walk_checks
    from arc_agent_v2.tests.test_a5 import answers
    from arc_agent_v2.tests.test_a6 import a2, reversal
    from arc_agent_v2.tests.test_a7 import chains
    plain = lambda o: o.item() if isinstance(o, np.generic) else str(o)
    norm = lambda r: json.loads(json.dumps(r, default=plain))
    diff = lambda old, new, skip=(): {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
                                      if k not in skip and not k.endswith("seconds") and old.get(k) != new.get(k)}
    run = lambda name: json.load(open(os.path.join(HERE, "runs", name)))
    res = dict(part3=[], part4=dict(walk=[], shortcut=[]), part5=[], part6=[], part7=[])
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
    for old in run("a5.json")["withheld"]:
        res["part5"].append(dict(check="F3", p=old["p"], seed=old["seed"],
                                 diff=diff(old, norm(answers(old["p"], old["seed"], withhold=True)))))
    base6 = run("a6.json")
    for old in base6["a2"]:
        res["part6"].append(dict(check="A2", p=old["p"], seed=old["seed"], diff=diff(old, norm(a2(old["p"], old["seed"])))))
    for old in base6["reversal"]:
        res["part6"].append(dict(check="reversal", p=old["p"], seed=old["seed"],
                                 diff=diff(old, norm(reversal(old["p"], old["seed"])))))
    for old in run("a7.json")["chains"]:
        res["part7"].append(dict(p=old["p"], seed=old["seed"], diff=diff(old, norm(chains(old["p"], old["seed"])))))
    return res


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a8.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save = lambda: json.dump(res, open(out_path, "w"), indent=1, default=str)
    say = lambda *a: print(*a, flush=True)
    t0 = time.time()
    res = dict(walks=[walks(k, s) for k in KINDS for s in SEEDS])
    res["seconds_part8"] = round(time.time() - t0, 1)
    save()                                              # part 8's own runs, kept whatever the regression does
    frac = lambda r, k: "%d/%d" % (r[k + "_right"], r[k])
    for r in res["walks"]:
        say("I1-I3 %-9s seed %d: revisits (column knows its place) a %s, b %s, c %s | all revisits a %s, b %s, c %s"
            " | first visits a %s, b %s, c %s | a lost at the end %s" % (
                r["kind"], r["seed"], frac(r["a"], "known"), frac(r["b"], "known"), frac(r["c"], "known"),
                frac(r["a"], "revisit"), frac(r["b"], "revisit"), frac(r["c"], "revisit"),
                frac(r["a"], "first"), frac(r["b"], "first"), frac(r["c"], "first"), r["a"]["lost_at_end"]))
    for r in res["walks"]:
        say("I5 %-9s seed %d: day 2 walk %.2f s (arm a)" % (r["kind"], r["seed"], r["a"]["seconds"]))
    say("part 8's own runs %.1f s" % res["seconds_part8"])
    t1 = time.time()
    res["regression"] = g = regression()
    res["seconds_regression"] = round(time.time() - t1, 1)
    save()
    for r in g["part3"]:
        say("I4 part 3 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["walk"]:
        say("I4 part 4 %-9s seed %d: %s" % (r["kind"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["shortcut"]:
        say("I4 part 4 shortcut %-9s: trials identical %s" % (r["kind"], r["same"]))
    for name in ("part5", "part6"):
        for r in g[name]:
            say("I4 %s %-8s p=%-3d seed %d: %s" % (name.replace("part", "part "), r["check"], r["p"], r["seed"],
                                                  r["diff"] or "reproduced"))
    for r in g["part7"]:
        say("I4 part 7 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "reproduced"))
    say("the regression %.1f s" % res["seconds_regression"])


def test_a8_small():
    """A fast check on the torus: revisits are recalled by location on day 2, and not without the location keys."""
    r = walks("E1 torus", 0)
    assert r["a"]["known"] > 0 and r["a"]["known_right"] / r["a"]["known"] >= 0.95
    assert r["b"]["revisit_right"] == 0 and r["c"]["revisit_right"] == 0
    assert r["a"]["first_right"] == 0


if __name__ == "__main__":
    main(sys.argv[1:])
