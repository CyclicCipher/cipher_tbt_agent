"""Fingerprints of consolidated frames: a check that a change leaves what consolidation learns identical (the
performance step, 2026-10-29). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_fingerprints check [--full]   (compare with the baseline)
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_fingerprints save             (a new baseline: only on a
                                                                                          recorded, intended change)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_fingerprints.py -q     (the fast tier, in the suite)

**What a fingerprint covers:** for each world, digests of the frame's moves, contents, displacements, coordinates and
factoring (phases, transitions, shifts, coordinates, naming, classes, relations, counts).

**The tiers:**
- **"fast"** (p = 13, and the walk worlds): about 2 s, part of the suite.
- **"full"** (p = 97 too): about 25 s, run when consolidation, the column or the thalamus changes (docs/LAB.md, the
  regression tiers).

**The baseline:** `fingerprints.json` beside this file. It changes only with a recorded reason.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import ARITHMETIC, WALK, BrainConfig
from arc_agent_v2.worlds.arithmetic_world import ArithmeticWorld
from arc_agent_v2.worlds.walk_world import WalkWorld

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fingerprints.json")
ARITH = [("A1", 13, 0, {}), ("A1", 13, 1, {}), ("A2", 13, 0, dict(ops=("+", "-", "*"))),
         ("rev", 13, 2, dict(ops=("+", "-"), reverse=True)), ("withheld", 13, 0, dict(withhold=True)),
         ("permute", 13, 1, dict(permute=True)),
         ("A1", 97, 0, {}), ("A2", 97, 0, dict(ops=("+", "-", "*"))), ("pm", 97, 1, dict(ops=("+", "-"))),
         ("rev", 97, 0, dict(ops=("+", "-"), reverse=True))]
WALKS = ("E1 torus", "E2 cycle", "O1 A5", "O2 box")

_b = lambda s: s.idx.tobytes() if s is not None else b"-"


def _digest(items):
    h = hashlib.sha256()
    for it in sorted(items):
        h.update(it)
        h.update(b"|")
    return h.hexdigest()[:16]


def fingerprint(frame):
    b, d = _b, _digest
    out = dict(inferred=frame.inferred, split=frame.split, moves=d(b(p) + b(x) + b(q) for p, x, q in frame.moves),
               content=d(b(p) + s.encode() + b(x) + str(n).encode() for p, s, x, n in frame.content),
               disp=d(d(b(a) + b(c) for a, c in m.items()).encode() for m in frame.displacements),
               coords=[[o.size, list(o.periods)] for o in frame.coords],
               declined=[list(map(str, x)) for x in frame.declined], unfactored=frame.unfactored)
    f = frame.factored
    if f is not None:
        out.update(phases=f.phases, phase=d(b(p) + b(q) for p, q in f.phase.items()),
                   trans=d(b(a) + b(x) + b(c) for (a, x), c in f.trans.items()),
                   shift=d(str(k).encode() + b(x) + b(c) for (k, x), c in f.shift.items()),
                   coord=d(b(p) + b(c) for p, c in f.coord.items()), named=d(b(c) + b(x) for c, x in f.named.items()),
                   cls=d(b(p) + str(k).encode() for p, k in f.cls.items()),
                   relation=sorted(map(str, f.relation.items())),
                   counts=d(b(p) + s.encode() + b(x) + str(n).encode() for (p, s), c in f.counts.items()
                            for x, n in c.items()),
                   cells=len(f.cells), start=b(f.start[0]).hex()[:16] + b(f.start[1]).hex()[:16])
    return out


def _arith(p, seed, **kw):
    world = ArithmeticWorld(p=p, train_frac=0.5, seed=seed, **kw)
    brain = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed))
    for ep in world.epoch(np.random.default_rng(seed)):
        brain.episode(ep)
    return brain.sleep()


def _walk(kind, seed):
    world = WalkWorld(kind, seed=seed)
    brain = Brain(WALK)
    brain.episode(world.stream(0, np.random.default_rng((seed, 5)).integers(0, world.A, 1000).tolist()))
    return brain.sleep()


def compute(full=False):
    """{case: fingerprint}, and {case: seconds}."""
    res, secs = {}, {}
    for name, p, s, kw in ARITH:
        if p > 13 and not full:
            continue
        t = time.time()
        res["%s %d %d" % (name, p, s)] = fingerprint(_arith(p, s, **kw))
        secs["%s %d %d" % (name, p, s)] = round(time.time() - t, 2)
    for kind in WALKS:
        t = time.time()
        res[kind] = fingerprint(_walk(kind, 0))
        secs[kind] = round(time.time() - t, 2)
    return json.loads(json.dumps(res)), secs


def differences(res):
    base = json.load(open(BASELINE))
    return {k: {f: (base[k].get(f), v.get(f)) for f in set(base[k]) | set(v) if base[k].get(f) != v.get(f)}
            for k, v in res.items() if v != base.get(k)}


def test_fingerprints_fast():
    """What consolidation learns at p = 13 and in the walk worlds is unchanged from the baseline."""
    res, _ = compute(full=False)
    assert not differences(res)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    res, secs = compute(full=(mode == "save" or "--full" in sys.argv))
    print("seconds:", secs, "total %.1f" % sum(secs.values()))
    if mode == "save":
        json.dump(res, open(BASELINE, "w"), indent=1)
        print("saved", len(res), "fingerprints to", BASELINE)
    else:
        diff = differences(res)
        print("identical to the baseline (%d cases)" % len(res) if not diff else diff)
