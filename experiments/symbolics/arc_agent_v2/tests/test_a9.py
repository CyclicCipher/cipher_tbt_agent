"""Part 9: the union-of-hypotheses column (docs/LAB.md part 9). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a9 [--out arc_agent_v2/runs/a9.json]   (the runs)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a9.py -q                          (a fast check)

**The checks:**
- **J1, J2:** singles and chains in the {+, −} world and the {+} world, with the cross-entropy of each answer
  (amendment 1, in bits: information.py).
- **J3:** the bursts per chain, and the union's size after each.
- **J4:** the control, `union=False`.
- **J5:**
  - parts 3, 4, 5, 6 and 8 reproduce;
  - part 7's accuracies are J1's own numbers in the same world (compared with `runs/a7.json`).
- **J6:** the cost.

The chains are the same as part 7's (the same random draws). The world's labels are used here only, to score.
"""
import json
import os
import sys
import time

import numpy as np

from arc_agent_v2.tests.test_a7 import LENGTHS, SEEDS, trained

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
WORLDS = {"+-": ("+", "-"), "+": ("+",)}


def query(brain, world, prefix, answer):
    """(right, bits, bursts, the union's sizes after each burst) for one prefix."""
    col = brain.columns()[0]
    col.bursts, col.burst_sizes = 0, []
    right = world.decode(brain.predict_after(prefix)) == answer
    return right, brain.bits(world.sdr_of[answer]), col.bursts, list(col.burst_sizes)


def arm(name, p, seed, union):
    world, brain, seconds = trained(p, seed, union, WORLDS[name])
    out = dict(seconds=round(seconds, 1))
    for op in world.ops:
        rs = [query(brain, world, pre, world.written(*e)[-1]) for pre, _, e in world.queries() if e[0] == op]
        out["single" + op] = sum(r[0] for r in rs) / len(rs)
        out["single%s_bits" % op] = float(np.mean([r[1] for r in rs]))
        out["single%s_bursts" % op] = sum(r[2] for r in rs)
    for n in LENGTHS:
        rng = np.random.default_rng((seed, 77, n))      # world.score_chains's draws: part 7's chains
        rs = []
        for _ in range(1000):
            ops = [world.ops[i] for i in rng.integers(0, len(world.ops), n)]
            syms = world.written_chain(ops, rng.integers(0, world.p, n + 1).tolist())
            rs.append(query(brain, world, [world.sdr_of[s] for s in syms[:-1]], syms[-1]))
        out["chain%d" % n] = sum(r[0] for r in rs) / len(rs)
        out["chain%d_bits" % n] = float(np.mean([r[1] for r in rs]))
        out["chain%d_bursts_as_predicted" % n] = sum(r[2] == n - 1 for r in rs) / len(rs)
        out["chain%d_one_hypothesis" % n] = sum(all(z == 1 for z in r[3]) for r in rs) / len(rs)
        out["chain%d_bursts" % n] = sorted({r[2] for r in rs})
    return out


def run(name, p, seed):
    return dict(world=name, p=p, seed=seed, union=arm(name, p, seed, True), control=arm(name, p, seed, False))


def regression(res):
    """J5: parts 3, 4, 5, 6 and 8 against their recorded runs; part 7 from J1's own numbers."""
    from arc_agent_v2.tests.test_a1 import run as run_a1
    from arc_agent_v2.tests.test_a4 import EXCURSION, shortcut, walk_checks
    from arc_agent_v2.tests.test_a5 import answers, one_fact
    from arc_agent_v2.tests.test_a6 import a2, reversal
    from arc_agent_v2.tests.test_a8 import walks
    plain = lambda o: o.item() if isinstance(o, np.generic) else str(o)
    norm = lambda r: json.loads(json.dumps(r, default=plain))
    diff = lambda old, new: {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
                             if not k.endswith("seconds") and old.get(k) != new.get(k)}
    load = lambda name: json.load(open(os.path.join(HERE, "runs", name)))
    g = dict(part3=[], part4=dict(walk=[], shortcut=[]), part5=[], part6=[], part7=[], part8=[])
    f3 = ("held_out_acc", "train_acc", "inferred", "places_after_eq", "places_after_b", "displacements",
          "full_on_C", "orders", "eq_one_to_one")
    for old in load("a1.json"):
        if old["arm"] == "full" and old["op"] == "+" and old["frac"] == 0.5 and old["seed"] in SEEDS:
            new = run_a1(old["p"], 0.5, old["seed"], "full")
            g["part3"].append(dict(p=old["p"], seed=old["seed"], diff={k: (old[k], new[k]) for k in f3
                                                                       if old[k] != new[k]}))
    base4 = load("a4.json")
    for old in base4["walk"]:
        g["part4"]["walk"].append(dict(kind=old["kind"], seed=old["seed"],
                                       diff=diff(old, norm(walk_checks(old["kind"], old["seed"])))))
    for kind in EXCURSION:
        trials = [t for s in SEEDS for t in shortcut(kind, s)["trials"]]
        g["part4"]["shortcut"].append(dict(kind=kind, same=trials == [t for r in base4["shortcut"]
                                                                       if r["kind"] == kind for t in r["trials"]]))
    base5 = load("a5.json")
    for old in base5["one_fact"]:
        g["part5"].append(dict(check="F1-F2", p=old["p"], seed=old["seed"],
                               diff=diff(old, norm(one_fact(old["p"], old["seed"])))))
    for old in base5["withheld"]:
        g["part5"].append(dict(check="F3", p=old["p"], seed=old["seed"],
                               diff=diff(old, norm(answers(old["p"], old["seed"], withhold=True)))))
    base6 = load("a6.json")
    for old in base6["a2"]:
        g["part6"].append(dict(check="A2", p=old["p"], seed=old["seed"], diff=diff(old, norm(a2(old["p"], old["seed"])))))
    for old in base6["reversal"]:
        g["part6"].append(dict(check="reversal", p=old["p"], seed=old["seed"],
                               diff=diff(old, norm(reversal(old["p"], old["seed"])))))
    mine = {(r["p"], r["seed"]): r for r in res["runs"] if r["world"] == "+-"}
    for old in load("a7.json")["chains"]:
        r = mine[(old["p"], old["seed"])]
        keys = ["single+", "single-"] + ["chain%d" % n for n in LENGTHS]
        g["part7"].append(dict(p=old["p"], seed=old["seed"],
                               diff={k: ((old["merge_" + k], old["off_" + k]), (r["union"][k], r["control"][k]))
                                     for k in keys if (old["merge_" + k], old["off_" + k]) != (r["union"][k], r["control"][k])}))
    for old in load("a8.json")["walks"]:
        new = norm(walks(old["kind"], old["seed"]))
        g["part8"].append(dict(kind=old["kind"], seed=old["seed"],
                               diff={a: diff(old[a], new[a]) for a in "abc" if diff(old[a], new[a])}))
    return g


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a9.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save = lambda: json.dump(res, open(out_path, "w"), indent=1, default=str)
    say = lambda *a: print(*a, flush=True)
    t0 = time.time()
    res = dict(runs=[run(w, p, s) for w in WORLDS for p in (13, 97) for s in SEEDS])
    res["seconds_part9"] = round(time.time() - t0, 1)
    save()                                              # part 9's own runs, kept whatever the regression does
    for r in res["runs"]:
        u, c = r["union"], r["control"]
        singles = ", ".join("%s %.3f (%.2g bits)" % (op, u["single" + op], u["single%s_bits" % op]) for op in WORLDS[r["world"]])
        chains = ", ".join("%.3f (%.2g bits)" % (u["chain%d" % n], u["chain%d_bits" % n]) for n in LENGTHS)
        say("J1-J2 {%s} p=%-3d seed %d: singles %s; chains of 2, 3, 4: %s" % (r["world"], r["p"], r["seed"], singles, chains))
    for r in res["runs"]:
        u = r["union"]
        say("J3 {%s} p=%-3d seed %d: chains with k-1 bursts %s; each burst one hypothesis %s; bursts seen %s; "
            "bursts in singles %s" % (
                r["world"], r["p"], r["seed"], [u["chain%d_bursts_as_predicted" % n] for n in LENGTHS],
                [u["chain%d_one_hypothesis" % n] for n in LENGTHS], [u["chain%d_bursts" % n] for n in LENGTHS],
                [u["single%s_bursts" % op] for op in WORLDS[r["world"]]]))
    for r in res["runs"]:
        c = r["control"]
        say("J4 {%s} p=%-3d seed %d: control chains %s; singles %s" % (
            r["world"], r["p"], r["seed"], [c["chain%d" % n] for n in LENGTHS],
            [c["single" + op] for op in WORLDS[r["world"]]]))
    for r in res["runs"]:
        say("J6 {%s} p=%-3d seed %d: training and sleep %.1f s" % (r["world"], r["p"], r["seed"], r["union"]["seconds"]))
    say("part 9's own runs %.1f s" % res["seconds_part9"])
    t1 = time.time()
    res["regression"] = g = regression(res)
    res["seconds_regression"] = round(time.time() - t1, 1)
    save()
    for r in g["part3"]:
        say("J5 part 3 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["walk"]:
        say("J5 part 4 %-9s seed %d: %s" % (r["kind"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["shortcut"]:
        say("J5 part 4 shortcut %-9s: trials identical %s" % (r["kind"], r["same"]))
    for name in ("part5", "part6"):
        for r in g[name]:
            say("J5 %s %-8s p=%-3d seed %d: %s" % (name.replace("part", "part "), r["check"], r["p"], r["seed"],
                                                  r["diff"] or "reproduced"))
    for r in g["part7"]:
        say("J5 part 7 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "accuracies reproduced"))
    for r in g["part8"]:
        say("J5 part 8 %-9s seed %d: %s" % (r["kind"], r["seed"], r["diff"] or "reproduced"))
    say("the regression %.1f s" % res["seconds_regression"])


def test_a9_small():
    """A fast check on p = 13: the {+} world chains through bursts, and its control doesn't."""
    u = arm("+", 13, 0, True)
    assert u["single+"] == 1.0 and min(u["chain%d" % n] for n in LENGTHS) >= 0.99
    assert u["chain3_bursts_as_predicted"] == 1.0 and u["single+_bursts"] == 0
    assert max(arm("+", 13, 0, False)["chain%d" % n] for n in LENGTHS) <= 0.05


if __name__ == "__main__":
    main(sys.argv[1:])
