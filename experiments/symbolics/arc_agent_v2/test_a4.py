"""Part 4: coordinates learned from the frame (docs/LAB.md part 4, with its amendments). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.test_a4 [--out arc_agent_v2/runs/a4.json]   (the pre-registered runs)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/test_a4.py -q                           (a fast check)

**The checks:**
- **C1–C3:** A1's frames: the module chain on the ring after "a + b"; a move found from one instance; overlap as
  3-adic closeness.
- **C4:** part 3's A1 results unchanged.
- **C5–C6:** walk worlds inside and outside the hypothesis space.
- **C7–C8:** the shortcut test, four arms.
- **C9:** the design check and the cost.

The worlds' labels and graphs are used here only, to name places for the report and to score.
"""
import json
import os
import sys
import time

import numpy as np

from arc_agent_v2.arithmetic_world import ArithmeticWorld
from arc_agent_v2.brain import Brain
from arc_agent_v2.config import ARITHMETIC, WALK, BrainConfig
from arc_agent_v2.HDClibrary import overlap
from arc_agent_v2.test_a1 import place_after, run as run_a1
from arc_agent_v2.walk_world import EXCURSION, KINDS, WalkWorld

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
STEPS, TRIALS, EXPLORE = 1000, 20, 40


# ---- C1-C3: A1's frames ----------------------------------------------------------------------------------------------
def a1(p, seed):
    world = ArithmeticWorld(p=p, train_frac=0.5, seed=seed)
    brain = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed))
    for ep in world.epoch(np.random.default_rng(seed)):
        brain.episode(ep)
    frame = brain.sleep()
    ring = {z: place_after(brain, world.episode("+", 0, z)[:3]) for z in range(p)}   # the place after "0 + z"
    orbit = next((o for o in frame.coords if all(o.modules.is_code(c) for c in ring.values())), None)
    out = dict(p=p, seed=seed, periods=list(orbit.periods) if orbit else None,
               distinct=len(set(ring.values())) == p)
    if orbit is None:
        return out
    m, codes = orbit.modules, set(ring.values())
    maps = whole = pairs = matched = 0
    for d in frame.displacements:
        st = [(orbit.codes[s], orbit.codes[t]) for s, t in d.items()
              if s in orbit.codes and t in orbit.codes and orbit.codes[s] in codes]
        if not st:
            continue
        delta = m.unbind(st[0][1], st[0][0])            # the shift, from one instance
        ok = [m.bind(s, delta) == t for s, t in st]
        maps, whole, pairs, matched = maps + 1, whole + all(ok), pairs + len(ok), matched + sum(ok)
    out.update(maps=maps, maps_from_one=whole, instances=pairs, instances_matched=matched)
    if p == 81:                                         # C3: overlap is 3-adic closeness
        good = total = 0
        for z1 in range(p):
            for z2 in range(z1 + 1, p):
                k = sum(1 for e in range(1, 5) if (z1 - z2) % 3 ** e == 0)
                good += overlap(ring[z1], ring[z2]) == m.R * k
                total += 1
        out.update(adic_pairs=total, adic_exact=good)
    return out


# ---- C5-C6: walk worlds ----------------------------------------------------------------------------------------------
def episode1(kind, seed, coordinates=True, regularity=True):
    world = WalkWorld(kind, seed=seed)
    acts = np.random.default_rng((seed, 5)).integers(0, world.A, STEPS).tolist()
    brain = Brain(WALK, regularity=regularity, coordinates=coordinates)
    brain.episode(world.stream(0, acts))
    brain.sleep()
    return world, brain


def node_of(world, frame):
    """Each frame place's node, read from the label sensed there (unique labels)."""
    return {p: world.node_of[x] for p, s, x, _ in frame.content if s == "retina" and x in world.node_of}


def walk_checks(kind, seed):
    world, brain = episode1(kind, seed)
    f = brain.frame
    node = node_of(world, f)
    places = {p for p, _, _ in f.moves} | {q for _, _, q in f.moves}
    right = sum(world.T[node[p], world.index[a]] == node[q] for p, a, q in f.moves if p in node and q in node)
    out = dict(kind=kind, seed=seed, places=len(places), moves=len(f.moves), complete=len(f.moves) == len(places) * world.A,
               inferred=f.inferred, moves_right=int(right), wrong=len(f.moves) - int(right),
               orbits=[[o.size, list(o.periods)] for o in f.coords], declined=[list(d) for d in f.declined],
               faults=brain.bus.faults)
    shift_ok = shift_n = 0                              # every installed shift, against the true graph
    for o in f.coords:
        back = {c: p for p, c in o.codes.items()}
        for x, delta in o.shifts.items():
            for p, c in o.codes.items():
                q = back.get(o.modules.bind(c, delta))
                shift_n += 1
                shift_ok += q is not None and world.T[node[p], world.index[x]] == node[q]
    out.update(shifts_checked=shift_n, shifts_right=shift_ok)
    return out


# ---- C7-C8: the shortcut test -------------------------------------------------------------------------------------------
def route_of(brain, stream):
    brain.episode(stream, learn=False)
    return brain.columns()[0].route()


def shortcut(kind, seed):
    world, carried = episode1(kind, seed, coordinates=True)
    _, table = episode1(kind, seed, coordinates=False)
    rng = np.random.default_rng((seed, 77))
    K, M = EXCURSION[kind]
    trials = []
    for _ in range(TRIALS):
        world.relabel(rng)
        start = int(rng.integers(world.n))
        acts = rng.integers(0, world.A, EXPLORE).tolist() + [0] * int(rng.integers(1, K + 1)) + \
            [2] * int(rng.integers(1, M + 1))
        stream, here = world.stream(start, acts), world.nodes(start, acts)[-1]
        true, plane = world.distance(here, start), world.plane(acts)
        fresh = Brain(WALK)
        fresh.episode(stream)
        fresh.sleep()
        graph = Brain(WALK, regularity=False, coordinates=False)
        graph.episode(stream)
        graph.sleep()
        rec = dict(true=true, plane=plane, shortcut=true < plane)
        for arm, brain in (("a", carried), ("b", table), ("c", fresh), ("d", graph)):
            route = route_of(brain, stream)
            if route is None:
                rec[arm] = [False, None]
                continue
            end = world.nodes(here, [world.index[x] for x in route])[-1]
            rec[arm] = [end == start, len(route)]
        trials.append(rec)
    return dict(kind=kind, seed=seed, trials=trials)


def summarize(trials, arm):
    ok = [t for t in trials if t[arm][0]]
    sc = [t for t in trials if t["shortcut"]]
    ratio = [t[arm][1] / t["true"] for t in ok if t["true"] > 0]
    return dict(success=round(len(ok) / len(trials), 3), ratio=round(float(np.mean(ratio)), 3) if ratio else None,
                shortcut_taken=round(sum(1 for t in sc if t[arm][0] and t[arm][1] == t["true"]) / len(sc), 3)
                if sc else None)


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a4.json")
    t0 = time.time()
    res = dict(a1=[a1(p, s) for p in (13, 97, 105, 81) for s in SEEDS])
    base = {(r["p"], r["seed"]): r for r in json.load(open(os.path.join(HERE, "runs", "a1.json")))
            if r["arm"] == "full" and r["op"] == "+" and r["frac"] == 0.5}
    fields = ("held_out_acc", "train_acc", "inferred", "places_after_eq", "places_after_b", "displacements",
              "full_on_C", "orders", "eq_one_to_one")
    res["c4"] = []
    for p in (13, 97):
        for s in SEEDS:
            r = run_a1(p, 0.5, s, "full")
            res["c4"].append(dict(p=p, seed=s, same=all(r[k] == base[(p, s)][k] for k in fields),
                                  diff={k: (base[(p, s)][k], r[k]) for k in fields if r[k] != base[(p, s)][k]}))
    res["walk"] = [walk_checks(k, s) for k in KINDS for s in SEEDS]
    res["shortcut"] = [shortcut(k, s) for k in EXCURSION for s in SEEDS]
    res["seconds"] = round(time.time() - t0, 1)

    for r in res["a1"]:
        print("C1-C3 A1 p=%-3d seed %d: periods %s, distinct %s, maps reproduced from one instance %s/%s, instances %s/%s%s"
              % (r["p"], r["seed"], r["periods"], r["distinct"], r.get("maps_from_one"), r.get("maps"),
                 r.get("instances_matched"), r.get("instances"),
                 ", 3-adic overlap exact %d/%d" % (r["adic_exact"], r["adic_pairs"]) if "adic_pairs" in r else ""))
    for r in res["c4"]:
        print("C4 A1 p=%-3d seed %d: part 3's results reproduced: %s %s" % (r["p"], r["seed"], r["same"], r["diff"] or ""))
    for r in res["walk"]:
        print("C5-C6 %-9s seed %d: places %d, complete %s, inferred %d, wrong moves %d, orbits %s, declined %s, "
              "installed shifts right %d/%d, bus faults %d" % (r["kind"], r["seed"], r["places"], r["complete"],
                                                             r["inferred"], r["wrong"], r["orbits"], r["declined"],
                                                             r["shifts_right"], r["shifts_checked"], r["faults"]))
    for kind in EXCURSION:
        trials = [t for r in res["shortcut"] if r["kind"] == kind for t in r["trials"]]
        same = sum(t["a"] == t["b"] for t in trials)
        print("C7-C9 %-9s: %d trials, shortcut trials %.2f; (a) == (b) in %d" % (kind, len(trials),
                                                                                np.mean([t["shortcut"] for t in trials]), same))
        for arm, name in (("a", "carried coordinates"), ("b", "carried table"), ("c", "relearning"), ("d", "graph memory")):
            print("    (%s) %-19s %s" % (arm, name, summarize(trials, arm)))
    print("total %.1f s" % res["seconds"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def test_a4_small():
    """A fast check: E1's episode gives Z/6 x Z/10's chain, and carried coordinates find the way home."""
    r = walk_checks("E1 torus", 0)
    assert r["places"] == 60 and r["orbits"] == [[60, [2, 2, 3, 5]]] and r["faults"] == 0
    assert r["shifts_right"] == r["shifts_checked"] > 0


if __name__ == "__main__":
    main(sys.argv[1:])
