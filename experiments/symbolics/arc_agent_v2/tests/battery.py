"""The regression battery: earlier parts' recorded predictions, re-checked (docs/LAB.md, the regression tiers). Run from
experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.battery            (tier 1)
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.battery --tier 2   (tier 2)

**The tiers:**
- **Tier 1, with every part's run:** parts 3, 5 (F1–F3), 6, 9 and 10 at p = 13 (3 seeds), and parts 4 and 8's walk
  checks. With the suite (which holds the fast fingerprints), it is the regression of a part.
- **Tier 2, when consolidation, the column or the thalamus changes:** the same parts at p = 97 (seed 0), with the full
  fingerprints (`test_fingerprints.py --full`). It is its own run, under 2 minutes.

Each entry compares with `runs/aN.json`; fields ending in "seconds" are not compared. Part 10 is checked last: its
day 2 changes the shared A2 brain (tests/trained.py).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
SEEDS = (0, 1, 2)


def run(tier=1):
    from arc_agent_v2.tests.test_a1 import run as run_a1
    from arc_agent_v2.tests.test_a4 import walk_checks
    from arc_agent_v2.tests.test_a5 import answers, one_fact
    from arc_agent_v2.tests.test_a6 import a2, reversal
    from arc_agent_v2.tests.test_a8 import walks
    from arc_agent_v2.tests.test_a9 import arm
    from arc_agent_v2.tests.test_a10 import run as run_a10
    plain = lambda o: o.item() if isinstance(o, np.generic) else str(o)
    norm = lambda r: json.loads(json.dumps(r, default=plain))
    diff = lambda old, new: {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
                             if not k.endswith("seconds") and old.get(k) != new.get(k)}
    load = lambda name: json.load(open(os.path.join(HERE, "runs", name)))
    cases = [(13, s) for s in SEEDS] if tier == 1 else [(97, 0)]
    g = dict(part3=[], part4=[], part5=[], part6=[], part8=[], part9=[], part10=[])
    f3 = ("held_out_acc", "train_acc", "inferred", "places_after_eq", "places_after_b", "displacements",
          "full_on_C", "orders", "eq_one_to_one")
    a1 = {(r["p"], r["seed"]): r for r in load("a1.json") if r["arm"] == "full" and r["op"] == "+" and r["frac"] == 0.5}
    a5, a6, a9, a10 = load("a5.json"), load("a6.json"), load("a9.json"), load("a10.json")
    pick = lambda rows, p, s: next(r for r in rows if r["p"] == p and r["seed"] == s)
    for p, s in cases:
        new = run_a1(p, 0.5, s, "full")
        g["part3"].append(dict(p=p, seed=s, diff={k: (a1[(p, s)][k], new[k]) for k in f3 if a1[(p, s)][k] != new[k]}))
        g["part5"].append(dict(check="F1-F2", p=p, seed=s, diff=diff(pick(a5["one_fact"], p, s), norm(one_fact(p, s)))))
        g["part5"].append(dict(check="F3", p=p, seed=s,
                               diff=diff(pick(a5["withheld"], p, s), norm(answers(p, s, withhold=True)))))
        g["part6"].append(dict(check="A2", p=p, seed=s, diff=diff(pick(a6["a2"], p, s), norm(a2(p, s)))))
        g["part6"].append(dict(check="reversal", p=p, seed=s, diff=diff(pick(a6["reversal"], p, s), norm(reversal(p, s)))))
        for w in ("+-", "+"):
            old = next(r for r in a9["runs"] if r["world"] == w and r["p"] == p and r["seed"] == s)
            new = dict(union=norm(arm(w, p, s, True)), control=norm(arm(w, p, s, False)))
            g["part9"].append(dict(world=w, p=p, seed=s, diff={a: diff(old[a], new[a]) for a in new if diff(old[a], new[a])}))
    if tier == 1:
        for old in load("a4.json")["walk"]:
            g["part4"].append(dict(kind=old["kind"], seed=old["seed"],
                                   diff=diff(old, norm(walk_checks(old["kind"], old["seed"])))))
        for old in load("a8.json")["walks"]:
            new = norm(walks(old["kind"], old["seed"]))
            g["part8"].append(dict(kind=old["kind"], seed=old["seed"],
                                   diff={a: diff(old[a], new[a]) for a in "abc" if diff(old[a], new[a])}))
    for p, s in cases:                                  # last: day 2 changes the shared A2 brain
        g["part10"].append(dict(p=p, seed=s, diff=diff(pick(a10["runs"], p, s), norm(run_a10(p, s)))))
    return g


# A difference already recorded: part 4's run predates part 5's generator correction, under which A5 declines in every
# seed (docs/LAB.md, part 5 results).
KNOWN = {("part4", "O1 A5"): {"orbits", "declined", "shifts_checked", "shifts_right"}}


def report(g, say=print):
    """Prints each entry; returns the number of unexpected differences."""
    bad = 0
    for name in ("part3", "part5", "part6", "part9", "part10", "part4", "part8"):
        for r in g[name]:
            label = " ".join(str(r[k]) for k in ("check", "world", "kind", "p", "seed") if k in r)
            known = KNOWN.get((name, r.get("kind")))
            if not r["diff"]:
                verdict = "reproduced"
            elif known is not None and set(r["diff"]) <= known:
                verdict = "reproduced (A5's known part-5 difference)"
            else:
                verdict, bad = r["diff"], bad + 1
            say("%s %s: %s" % (name.replace("part", "part "), label, verdict))
    return bad


if __name__ == "__main__":
    tier = 2 if "--tier" in sys.argv and sys.argv[sys.argv.index("--tier") + 1] == "2" else 1
    t = time.time()
    g = run(tier)
    bad = report(g)
    print("tier %d: %.1f s; unexpected differences: %d" % (tier, time.time() - t, bad))
