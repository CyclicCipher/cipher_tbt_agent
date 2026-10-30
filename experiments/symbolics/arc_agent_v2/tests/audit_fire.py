"""The audit (docs/AUDIT.md section 1): where each mechanism fires, by world. Counters are wrapped around v2's functions
from outside, with no change to architecture/. Run from experiments/symbolics: python -m arc_agent_v2.tests.audit_fire.

It measures the brain as it was at the audit (commit d660898): part 12 replaced the day memory's override and the
thalamus's best place, which the walk counters read, so check out that commit to reproduce runs/audit_fire.json."""
import json
import os
from collections import Counter

import numpy as np

from arc_agent_v2.architecture import column as colmod
from arc_agent_v2.architecture import frameLearning as fl
from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import ARITHMETIC, WALK, BrainConfig
from arc_agent_v2.worlds.arithmetic_world import ArithmeticWorld
from arc_agent_v2.worlds.walk_world import WalkWorld

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
C = Counter()


def wrap(mod, name, count):
    f = getattr(mod, name)

    def g(*a, **k):
        r = f(*a, **k)
        count(a, k, r)
        return r
    setattr(mod, name, g)


wrap(fl, "constant", lambda a, k, r: C.update({"constant moves filled": r}))
wrap(fl, "constants", lambda a, k, r: C.update({"(context, input) pairs constant": int(r.sum())}))
orig_pool = fl.Pool.pool


def pool(self, a, b):
    a2, b2 = self.find(a), self.find(b)
    before = (int(self.to[b2]) if a2 != b2 else None, len(self.refused))
    orig_pool(self, a, b)
    if a2 == b2 or self.bad[a2] or self.bad[b2]:
        return
    if self.to[b2] == a2:
        C["poolings made"] += 1
    elif len(self.refused) > before[1]:
        C["poolings refused (conflict)"] += 1
    elif (min(a2, b2), max(a2, b2)) not in self.refused:
        C["poolings refused (fewer than 2 agreements across contexts)"] += 1


fl.Pool.pool = pool
orig_seed = fl.Pool._seed


def _seed(self, pairs, S, T, k):
    C["fixed-point instances (not identity evidence)"] += int((S == T).sum())
    return orig_seed(self, pairs, S, T, k)


fl.Pool._seed = _seed
orig_moved = colmod.Column._moved


def _moved(self, hyps, x):
    f = self.fac
    trans_no_shift = sum(1 for (ph, c) in hyps if f.trans.get((ph, x)) is not None and
                         f.shift.get((f.cls.get(ph), x)) is None)
    out, burst = orig_moved(self, hyps, x)
    C["union: hypotheses dropped, transition without shift"] += trans_no_shift
    C["union: bursts"] += burst is not None
    return out, burst


colmod.Column._moved = _moved
orig_seeded = colmod.Column._seeded
colmod.Column._seeded = lambda self, place: (C.update({"union: re-seeded from the place": 1}) or orig_seeded(self, place))


def arith(name, p, seed, queries=True, **kw):
    C.clear()
    world = ArithmeticWorld(p=p, train_frac=0.5, seed=seed, **kw)
    brain = Brain(BrainConfig(ARITHMETIC.sdr, ARITHMETIC.sensors, ARITHMETIC.regions, seed))
    for ep in world.epoch(np.random.default_rng(seed)):
        brain.episode(ep)
    f = brain.sleep()
    if queries:
        for pre, _, _ in world.queries()[:500]:
            brain.predict_after(pre)
    fac = f.factored
    out = dict(C)
    out.update({"pairs split by context": f.split, "orbits": len(f.coords), "factored": fac is not None,
                "naming accepted": len(fac.naming) if fac else 0,
                "shift classes": len(set(fac.cls.values())) if fac else 0,
                "phases left out": len(fac.left_out) if fac else 0,
                "relations found": len(fac.relation) if fac else 0,
                "shifts given by a relation": len(fac.related) if fac else 0})
    return name, out


def walk(kind):
    C.clear()
    world = WalkWorld(kind, seed=0)
    brain = Brain(WALK)
    brain.episode(world.stream(0, np.random.default_rng((0, 5)).integers(0, world.A, 1000).tolist()))
    f = brain.sleep()
    world.relabel(np.random.default_rng((0, 88)))
    acts = np.random.default_rng((0, 89)).integers(0, world.A, 180).tolist()
    brain.begin()
    nodes = world.nodes(0, acts)
    brain.step(world.labels[nodes[0]], "retina")
    overrides = 0
    for a, v in zip(acts, nodes[1:]):
        brain.step(world.action[a], "motor")
        where = brain.where("retina")
        overrides += where is not None and brain.hippocampus.recall_today(where, "retina") is not None
        brain.predict("retina")
        brain.step(world.labels[v], "retina")
    fac = f.factored
    out = dict(C)
    out.update({"pairs split by context": f.split, "orbits": len(f.coords), "factored": fac is not None,
                "naming accepted": len(fac.naming) if fac else 0,
                "shift classes": len(set(fac.cls.values())) if fac else 0,
                "relations found": len(fac.relation) if fac else 0,
                "day 2: today's memory overrides": overrides})
    return kind, out


def precedence():
    from arc_agent_v2.tests.test_a10 import days
    C.clear()
    world, brain, _, _ = days(13, 7)
    th = brain.thalamus
    held = sum(1 for (k, c), v in th.counts.items() if k == "hold" and v[1] / sum(v) > 0.0909)
    rel = sum(1 for (k, c), v in th.counts.items() if k == "release" and v[1] / sum(v) > 0.0909)
    out = dict(C)
    out.update({"thalamus: hold contexts above the prior": held, "thalamus: release contexts above the prior": rel})
    return "precedence p=13 (part 10)", out


rows = [arith("A1 p=13", 13, 7), arith("A2 p=13", 13, 7, ops=("+", "-", "*")),
        arith("reversal p=13", 13, 7, ops=("+", "-"), reverse=True), arith("withheld p=13", 13, 7, withhold=True),
        arith("numerals p=100", 100, 7, base=10)]
rows += [walk(k) for k in ("E1 torus", "E2 cycle", "O1 A5", "O2 box")]
rows.append(precedence())
json.dump(rows, open(os.path.join(RUNS, "audit_fire.json"), "w"), indent=1)
keys = sorted({k for _, r in rows for k in r})
for k in keys:
    print("%-58s %s" % (k, " | ".join("%s: %s" % (n.split()[0], r.get(k, 0)) for n, r in rows)))
