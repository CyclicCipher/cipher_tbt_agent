"""Part 6: A2, three operations mixed, and subtraction from addition facts (docs/LAB.md part 6). Run from
experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a6 [--out arc_agent_v2/runs/a6.json]   (the pre-registered runs)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a6.py -q                           (a fast check)

**The checks:**
- **G1:** A2's held-out accuracy per operation.
- **G2:** P4 splits, and every move of the frame is the world's.
- **G3:** coordinates per family (the additive and the multiplicative orbit of the results).
- **G4:** factoring with shift classes, and the relation between them.
- **G5:** the reversal, with the relation and without.
- **G6:** parts 3, 4 and 5 reproduce.
- **G7:** the cost.

The world's labels are used here only: to name inputs and phases for the report, and to check the frame against the
world's tables.
"""
import json
import os
import sys
import time

import numpy as np

from arc_agent_v2.worlds.arithmetic_world import OPS, ArithmeticWorld
from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import ARITHMETIC, BrainConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
SEEDS = (0, 1, 2)
CHAIN = {13: (2, 3, 4), 97: (2, 3, 4, 8, 16, 32)}   # the multiplicative group's chain of quotients (G3)


def trained(p, seed, relation=True, **kw):
    world = ArithmeticWorld(p=p, train_frac=0.5, seed=seed, **kw)
    brain = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed), relation=relation)
    t = time.time()
    for ep in world.epoch(np.random.default_rng(seed)):
        brain.episode(ep)
    brain.sleep()
    return world, brain, time.time() - t


def labels(world, frame):
    """Each frame place reached by a training prefix, labelled with the world's state there; and the places given two
    labels (P3 merged different states). Only seen moves are walked, so the labels don't depend on what P4 inferred."""
    nxt = {(u, x): v for u, x, v in frame.moves}
    lab, conflicts = {frame.start: ("S",)}, 0
    for op, a, b in world.train:
        r = world.answer(op, a, b)
        u = frame.start
        for sym, state in zip(world.written(op, a, b), (("A", a), ("B", op, a), ("C", r), ("E", r), ("F",))):
            u = nxt[(u, world.sdr_of[sym])]
            if lab.setdefault(u, state) != state:
                conflicts += 1
    return lab, conflicts


def truth(state, sym, world):
    """Where the world's tables say a symbol leads from a state."""
    kind = state[0]
    if kind == "S" and sym.isdigit():
        return ("A", int(sym))
    if kind == "A" and sym in OPS:
        return ("B", sym, state[1])
    if kind == "B" and sym.isdigit():
        return ("C", world.answer(state[1], state[2], int(sym)))
    if kind == "C" and sym == "=":
        return ("E", state[1])
    if kind == "E" and sym == str(state[1]):
        return ("F",)
    return None


def a2(p, seed):
    """G1-G4 and G7 on A2."""
    world, brain, seconds = trained(p, seed, ops=("+", "-", "*"))
    f, fac = brain.frame, brain.frame.factored
    out = dict(p=p, seed=seed, seconds=round(seconds, 1))
    for op in world.ops:                                                         # G1
        qs = [q for q in world.queries() if q[2][0] == op]
        out["acc_" + op] = sum(world.decode(brain.predict_after(pre)) == world.written(*e)[-1] for pre, _, e in qs) / len(qs)
    lab, conflicts = labels(world, f)                                            # G2
    name = {v: k for k, v in world.sdr_of.items()}
    wrong = unlabelled = 0
    for u, x, v in f.moves:
        if u not in lab:
            unlabelled += 1
        elif truth(lab[u], name[x], world) != lab.get(v):
            wrong += 1
    out.update(split=f.split, inferred=f.inferred, moves=len(f.moves), wrong=wrong, unlabelled=unlabelled,
               conflicts=conflicts)
    C = {u: s[1] for u, s in lab.items() if s[0] == "C"}                         # G3
    orbits = [(o.size, o.periods, sorted(C.get(q, -1) for q in o.codes)) for o in f.coords]
    out["additive"] = any(n == p and per == (p,) and pl == list(range(p)) for n, per, pl in orbits)
    out["multiplicative"] = any(n == p - 1 and per == CHAIN.get(p) and pl == list(range(1, p)) for n, per, pl in orbits)
    out["orbits"] = [(n, per) for n, per, _ in orbits]
    out["factored"], out["why"] = fac is not None, f.unfactored                  # G4
    if fac is None:
        return out
    phase_of = {}                                       # each state's phase (a merged phase holds several: part 7)
    for u, s in lab.items():
        phase_of.setdefault("B" + s[1] if s[0] == "B" else s[0], fac.phase[u])
    cls = {n: fac.cls.get(ph) for n, ph in phase_of.items()}
    m, plus, minus = fac.modules, cls["B+"], cls["B-"]
    d = lambda c, b: fac.shift.get((c, world.sdr_of[str(b)]))
    alike = sum(1 for b in range(p) if d(plus, b) is not None and d(cls["S"], b) == d(plus, b))
    negated = sum(1 for b in range(p) if None not in (d(plus, b), d(minus, b)) and
                  m.bind(d(plus, b), d(minus, b)) == m.identity())
    k = fac.relation.get((min(plus, minus), max(plus, minus))) if None not in (plus, minus) else None
    out.update(classes=cls, left_out=sorted(n for n, ph in phase_of.items() if ph in fac.left_out), k=k,
               naming=sorted(name.get(x, "?") for x in fac.naming),
               identities=sorted(name.get(x, "?") for x in fac.identities),
               digits_alike=alike / p, digits_negated=negated / p, related=len(fac.related))
    return out


def reversal(p, seed):
    """G5: subtractions by the digits never subtracted, with the relation and without."""
    out = dict(p=p, seed=seed)
    for arm, relation in (("relation", True), ("no_relation", False)):
        world, brain, seconds = trained(p, seed, relation=relation, ops=("+", "-"), reverse=True)
        out[arm] = world.score(brain.predict_after)
        out[arm + "_seconds"] = round(seconds, 1)
        fac = brain.frame.factored
        out[arm + "_k"] = None if fac is None else sorted(fac.relation.values())
        out["never_subtracted"] = sorted(world.never_subtracted)
    return out


def regression():
    """G6: parts 3, 4 and 5 against their recorded runs."""
    from arc_agent_v2.tests.test_a1 import run as run_a1
    from arc_agent_v2.tests.test_a4 import EXCURSION, a1 as a4_a1, shortcut, walk_checks
    from arc_agent_v2.tests.test_a5 import answers, one_fact
    res = {}
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
        res["part4"]["shortcut"].append(dict(kind=kind, same=trials == old))
    base5 = json.load(open(os.path.join(HERE, "runs", "a5.json")))
    norm = lambda r: json.loads(json.dumps(r, default=str))
    res["part5"] = []
    for old in base5["one_fact"]:
        new = norm(one_fact(old["p"], old["seed"]))
        res["part5"].append(dict(check="F1-F2", p=old["p"], seed=old["seed"],
                                 diff={k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}))
    for old in base5["withheld"]:
        new = norm(answers(old["p"], old["seed"], withhold=True))
        res["part5"].append(dict(check="F3", p=old["p"], seed=old["seed"],
                                 diff={k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}))
    return res


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a6.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save = lambda: json.dump(res, open(out_path, "w"), indent=1, default=str)
    say = lambda *a: print(*a, flush=True)
    t0 = time.time()
    res = dict(a2=[a2(p, s) for p in (13, 97) for s in SEEDS])
    res["reversal"] = [reversal(p, s) for p in (13, 97) for s in SEEDS]
    res["seconds_part6"] = round(time.time() - t0, 1)
    save()                                              # part 6's own runs, kept whatever the regression does
    for r in res["a2"]:
        say("G1 p=%-3d seed %d: + %.3f, - %.3f, * %.3f" % (r["p"], r["seed"], r["acc_+"], r["acc_-"], r["acc_*"]))
    for r in res["a2"]:
        say("G2 p=%-3d seed %d: pairs split %d; %d moves (%d inferred), wrong %d, unlabelled %d, places with two "
            "states %d" % (r["p"], r["seed"], r["split"], r["moves"], r["inferred"], r["wrong"], r["unlabelled"],
                           r["conflicts"]))
    for r in res["a2"]:
        say("G3 p=%-3d seed %d: additive orbit of the results %s; multiplicative orbit of the nonzero results %s; "
            "orbits %s" % (r["p"], r["seed"], r["additive"], r["multiplicative"], r["orbits"]))
    for r in res["a2"]:
        say("G4 p=%-3d seed %d: factored %s; classes %s; left out %s; k %s; digits alike from the start and after + "
            "%.3f; negated after - %.3f; naming %s; identities %s" % (
                r["p"], r["seed"], r["factored"], r.get("classes"), r.get("left_out"), r.get("k"),
                r.get("digits_alike", 0), r.get("digits_negated", 0), r.get("naming"), r.get("identities")))
    for r in res["reversal"]:
        say("G5 p=%-3d seed %d: subtractions by the never-subtracted digits: relation %.3f (k %s), no relation %.3f" % (
            r["p"], r["seed"], r["relation"], r["relation_k"], r["no_relation"]))
    for r in res["a2"]:
        say("G7 p=%-3d seed %d: training and sleep %.1f s" % (r["p"], r["seed"], r["seconds"]))
    say("part 6's own runs %.1f s" % res["seconds_part6"])
    t1 = time.time()
    res["regression"] = g = regression()
    res["seconds_regression"] = round(time.time() - t1, 1)
    save()
    for r in g["part3"]:
        say("G6 part 3 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["a1"]:
        say("G6 part 4 A1 p=%-3d seed %d: %s" % (r["p"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["walk"]:
        say("G6 part 4 %-9s seed %d: %s" % (r["kind"], r["seed"], r["diff"] or "reproduced"))
    for r in g["part4"]["shortcut"]:
        say("G6 part 4 shortcut %-9s: trials identical %s" % (r["kind"], r["same"]))
    for r in g["part5"]:
        say("G6 part 5 %s p=%-3d seed %d: %s" % (r["check"], r["p"], r["seed"], r["diff"] or "reproduced"))
    say("the regression %.1f s" % res["seconds_regression"])


def test_a6_small():
    """A fast check on p = 13: A2 is answered and factored into the predicted classes, and the reversal needs the
    relation."""
    r = a2(13, 0)
    assert (r["acc_+"], r["acc_-"], r["acc_*"]) == (1.0, 1.0, 1.0)
    assert r["split"] > 0 and r["wrong"] == 0 and r["conflicts"] == 0 and r["additive"] and r["multiplicative"]
    c = r["classes"]
    assert c["S"] == c["B+"] != c["B-"] and c["B*"] is None and r["k"] == (12,) and r["naming"] == ["="]
    v = reversal(13, 0)
    assert v["relation"] == 1.0 and v["no_relation"] == 0.0


if __name__ == "__main__":
    main(sys.argv[1:])
