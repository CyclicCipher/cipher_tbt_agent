"""Part 10: precedence, with working memory in the thalamus (docs/LAB.md part 10). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a10 [--out arc_agent_v2/runs/a10.json]   (the run)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a10.py -q                           (a fast check)

**Day 1:** A2's world (half of the single equations), then sleep. **Day 2:** 300 random expressions of 2 operations
with standard precedence, learning on.

**The checks:**
- **K1, K2:** held-out 2-operation expressions and never-shown 3-operation ones, with the precedence-sensitive ones
  apart.
- **K3:** the gate's learned P(yes) per context.
- **K4:** the control (the gate off).
- **K5:** the battery (amendment 7).
- **K6:** the cost.

The battery runs first: it uses the shared trained brains (tests/trained.py) before day 2 changes A2's. The world's
labels are used here only, to score and to name contexts.
"""
import json
import os
import sys
import time

import numpy as np

from arc_agent_v2.tests.trained import trained as shared

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
SEEDS = (0, 1, 2)
DAY2, TESTS = 300, 1000
OPS = ("+", "-", "*")


def days(p, seed):
    """Day 1 (a shared trained brain) and day 2. Returns (world, brain, day 2's expressions, day 2's seconds)."""
    world, brain, _ = shared(p, seed, dict(ops=OPS, train_frac=0.5), {})
    rng = np.random.default_rng((seed, 10))
    day2 = [world.random_expr(2, rng) for _ in range(DAY2)]
    t = time.time()
    for ops, xs in day2:
        brain.episode([world.sdr_of[s] for s in world.written_expr(ops, xs)])
    return world, brain, set(day2), time.time() - t


def score(world, brain, n_ops, seed, exclude, gate):
    """(accuracy, accuracy on the precedence-sensitive ones, how many of those) on TESTS expressions never shown."""
    brain.thalamus.enabled = gate
    rng = np.random.default_rng((seed, 11, n_ops))
    right = sens = sens_right = n = 0
    while n < TESTS:
        ops, xs = world.random_expr(n_ops, rng)
        if (ops, xs) in exclude:
            continue
        n += 1
        syms = world.written_expr(ops, xs)
        ok = world.decode(brain.predict_after([world.sdr_of[s] for s in syms[:-1]])) == syms[-1]
        right += ok
        if world.written_chain(list(ops), list(xs))[-1] != syms[-1]:
            sens += 1
            sens_right += ok
    brain.thalamus.enabled = True
    return right / n, sens_right / max(1, sens), sens


def run(p, seed):
    t0 = time.time()
    world, brain, day2, sec2 = days(p, seed)
    out = dict(p=p, seed=seed, day2_seconds=round(sec2, 1))
    for n_ops in (2, 3):
        for name, gate in (("gate", True), ("off", False)):
            acc, sens, n = score(world, brain, n_ops, seed, day2 if n_ops == 2 else set(), gate)
            out["%s%d" % (name, n_ops)], out["%s%d_sensitive" % (name, n_ops)], out["sensitive%d" % n_ops] = acc, sens, n
    S, th = world.sdr_of, brain.thalamus
    out["hold"] = {a + b: th.p_yes("hold", (S[a], S[b])) for a in OPS for b in OPS}
    out["release"] = {a + b: th.p_yes("release", (S[a], S[b])) for a in OPS for b in OPS + ("=",)}
    others = [v[1] / sum(v) for (kind, (a, b)), v in th.counts.items()
              if kind == "hold" and not (b == S["*"] and a in (S["+"], S["-"], S["*"]))]
    out["hold_others_max"] = max(others, default=None)
    out["contexts_learned"] = len(th.counts)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def battery():
    """K5 (amendment 7): parts 3, 5, 6 and 9 at p = 13 (3 seeds) and p = 97 (seed 0); parts 4 and 8's walk checks."""
    from arc_agent_v2.tests.test_a1 import run as run_a1
    from arc_agent_v2.tests.test_a4 import walk_checks
    from arc_agent_v2.tests.test_a5 import answers, one_fact
    from arc_agent_v2.tests.test_a6 import a2, reversal
    from arc_agent_v2.tests.test_a8 import walks
    from arc_agent_v2.tests.test_a9 import arm
    plain = lambda o: o.item() if isinstance(o, np.generic) else str(o)
    norm = lambda r: json.loads(json.dumps(r, default=plain))
    diff = lambda old, new: {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
                             if not k.endswith("seconds") and old.get(k) != new.get(k)}
    load = lambda name: json.load(open(os.path.join(HERE, "runs", name)))
    cases = [(13, s) for s in SEEDS] + [(97, 0)]
    g = dict(part3=[], part4=[], part5=[], part6=[], part8=[], part9=[])
    f3 = ("held_out_acc", "train_acc", "inferred", "places_after_eq", "places_after_b", "displacements",
          "full_on_C", "orders", "eq_one_to_one")
    a1 = {(r["p"], r["seed"]): r for r in load("a1.json") if r["arm"] == "full" and r["op"] == "+" and r["frac"] == 0.5}
    a5, a6, a9 = load("a5.json"), load("a6.json"), load("a9.json")
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
    for old in load("a4.json")["walk"]:
        g["part4"].append(dict(kind=old["kind"], seed=old["seed"], diff=diff(old, norm(walk_checks(old["kind"], old["seed"])))))
    for old in load("a8.json")["walks"]:
        new = norm(walks(old["kind"], old["seed"]))
        g["part8"].append(dict(kind=old["kind"], seed=old["seed"], diff={a: diff(old[a], new[a]) for a in "abc"
                                                                         if diff(old[a], new[a])}))
    return g


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a10.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    say = lambda *a: print(*a, flush=True)
    t0 = time.time()
    res = dict(battery=battery())
    res["seconds_battery"] = round(time.time() - t0, 1)
    t1 = time.time()
    res["runs"] = [run(p, s) for p in (13, 97) for s in SEEDS]
    res["seconds_part10"] = round(time.time() - t1, 1)
    res["seconds_total"] = round(time.time() - t0, 1)
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    for r in res["runs"]:
        say("K1-K2 p=%-3d seed %d: 2 ops %.3f (sensitive %.3f of %d); 3 ops %.3f (sensitive %.3f of %d)" % (
            r["p"], r["seed"], r["gate2"], r["gate2_sensitive"], r["sensitive2"], r["gate3"], r["gate3_sensitive"],
            r["sensitive3"]))
    fmt = lambda d: " ".join("%s %s" % (k, "-" if v is None else "%.2f" % v) for k, v in d.items())
    for r in res["runs"]:
        say("K3 p=%-3d seed %d: hold %s | release %s | other hold contexts at most %s (%d contexts learned)" % (
            r["p"], r["seed"], fmt(r["hold"]), fmt(r["release"]),
            "-" if r["hold_others_max"] is None else "%.2f" % r["hold_others_max"], r["contexts_learned"]))
    for r in res["runs"]:
        say("K4 p=%-3d seed %d: gate off: 2 ops %.3f (sensitive %.3f); 3 ops %.3f (sensitive %.3f)" % (
            r["p"], r["seed"], r["off2"], r["off2_sensitive"], r["off3"], r["off3_sensitive"]))
    g = res["battery"]
    for name in ("part3", "part5", "part6", "part9", "part4", "part8"):
        for r in g[name]:
            label = " ".join(str(r[k]) for k in ("check", "world", "kind", "p", "seed") if k in r)
            say("K5 %s %s: %s" % (name.replace("part", "part "), label, r["diff"] or "reproduced"))
    for r in res["runs"]:
        say("K6 p=%-3d seed %d: day 2 %.1f s, the run %.1f s" % (r["p"], r["seed"], r["day2_seconds"], r["seconds"]))
    say("battery %.1f s; part 10's runs %.1f s; total %.1f s" % (res["seconds_battery"], res["seconds_part10"],
                                                               res["seconds_total"]))


def test_a10_small():
    """A fast check on p = 13: precedence is learned from 300 expressions, and the gate is what answers it."""
    r = run(13, 1)
    assert r["gate2"] >= 0.95 and r["gate2_sensitive"] >= 0.95 and r["off2_sensitive"] <= 0.10
    assert r["hold"]["+*"] > 0.9 and r["hold"]["-*"] > 0.9


if __name__ == "__main__":
    main(sys.argv[1:])
