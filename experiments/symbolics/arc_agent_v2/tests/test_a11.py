"""Part 11: A3, step 1 — two-digit numerals, measured (docs/LAB.md part 11; docs/A3.md). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a11 [--out arc_agent_v2/runs/a11.json]   (the run)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a11.py -q                           (a fast check)

**The checks:**
- **N1, N4:** exact answers, and per digit, on (a) standard and (b) withheld.
- **N2:** the coordinates and their chains.
- **N3:** the digit shift classes, by position, against ×10, the relation, and the naming.
- **N5:** the cost.

The world's labels are used here only, to score and to name phases.
"""
import json
import os
import sys
import time

import numpy as np

from arc_agent_v2.tests.trained import trained as shared

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
SEEDS = (0, 1, 2)
P, BASE = 100, 10


def phases(world, brain):
    """Each kind of state's phase, from training prefixes over seen moves: S, T (after the first tens digit), N (after
    the first number), B (after "+"), BB (after the second tens digit), C (after "a + b"), E (after "="), E2."""
    f, fac = brain.frame, brain.frame.factored
    nxt = {(u, x): v for u, x, v in f.moves}
    kinds = ("T", "N", "B", "BB", "C", "E", "E2")
    out = {"S": {fac.phase[f.start]}} if fac is not None and f.start in fac.phase else {"S": set()}
    for eq in world.train[:2000]:
        u = f.start
        for sym, kind in zip(world.written(*eq), kinds):
            u = nxt.get((u, world.sdr_of[sym]))
            if u is None:
                break
            if fac is not None and u in fac.phase:
                out.setdefault(kind, set()).add(fac.phase[u])
    return out


def times(m, code, k):
    """code bound to itself k times (k * the shift), in the Modules m."""
    out = m.identity()
    for _ in range(k):
        out = m.bind(out, code)
    return out


def run(seed, withhold):
    t = time.time()
    world, brain, sec = shared(P, seed, dict(ops=("+",), base=BASE, train_frac=0.5, withhold=withhold), {})
    out = dict(seed=seed, withhold=withhold, train_seconds=round(sec, 1))
    eqs = world.held_out
    digits = [world.score_digits_of(brain.predict_after, eq) for eq in eqs]
    out["exact"] = float(np.mean([all(d) for d in digits]))
    out["digit"] = [float(np.mean([d[i] for d in digits])) for i in range(world.k)]
    f = brain.frame
    out["orbits"] = [[o.size, list(o.periods)] for o in f.coords]
    fac = f.factored
    out["factored"], out["why"] = fac is not None, f.unfactored
    if fac is not None:
        ph = phases(world, brain)
        out["classes"] = {k: sorted({fac.cls.get(p) for p in v}, key=str) for k, v in ph.items()}
        m, S = fac.modules, world.sdr_of
        tens = [fac.cls.get(p) for p in ph.get("B", ())] + [fac.cls.get(p) for p in ph.get("S", ())]
        units = [fac.cls.get(p) for p in ph.get("BB", ())] + [fac.cls.get(p) for p in ph.get("T", ())]
        tens = next((c for c in tens if c is not None), None)
        units = next((c for c in units if c is not None), None)
        out["tens_class"], out["units_class"] = tens, units
        if tens is not None and units is not None:
            ok = n = 0
            for d in range(BASE):
                dt, du = fac.shift.get((tens, S[str(d)])), fac.shift.get((units, S[str(d)]))
                if dt is not None and du is not None:
                    n += 1
                    ok += dt == times(m, du, BASE)
            out["tens_is_10x_units"] = "%d/%d" % (ok, n)
        out["relation"] = {str(k): v for k, v in fac.relation.items()}
        label = {v: k for k, v in S.items()}
        out["naming"] = [label.get(x, "?") for x in fac.naming]
        out["phases"] = fac.phases
    out["seconds"] = round(time.time() - t, 1)
    return out


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a11.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    say = lambda *a: print(*a, flush=True)
    t0 = time.time()
    res = dict(runs=[run(s, w) for w in (False, True) for s in SEEDS])
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    for r in res["runs"]:
        say("N1/N4 %s seed %d: exact %.3f; per digit %s" % ("withheld" if r["withhold"] else "standard", r["seed"],
                                                          r["exact"], ["%.3f" % d for d in r["digit"]]))
    for r in res["runs"]:
        say("N2 %s seed %d: orbits %s" % ("withheld" if r["withhold"] else "standard", r["seed"], r["orbits"]))
    for r in res["runs"]:
        say("N3 %s seed %d: factored %s (%s); classes %s; tens class %s, units class %s; tens shift = 10 x units %s; "
            "relation %s; naming %s" % ("withheld" if r["withhold"] else "standard", r["seed"], r["factored"], r["why"],
                                        r.get("classes"), r.get("tens_class"), r.get("units_class"),
                                        r.get("tens_is_10x_units"), r.get("relation"), r.get("naming")))
    say("N5: the run %.1f s (training and sleep %s)" % (res["seconds"], [r["train_seconds"] for r in res["runs"]]))


def test_a11_small():
    """A fast check: two-digit numerals mod 100 are answered on held-out equations (seed 1)."""
    r = run(1, False)
    assert r["exact"] >= 0.99


if __name__ == "__main__":
    main(sys.argv[1:])
