"""Part 12: arithmetic and MNIST in one brain (docs/LAB.md part 12). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.test_a12 [--out arc_agent_v2/runs/a12.json]   (the run)
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_a12.py -q                           (a fast check)

**One brain** (`config.JOINT`): region S (symbols, a learned frame) and region V (retina and gaze, an innate
retinotopic frame). One stream: A2's training equations at p = 13 and the MNIST training images, interleaved; one day,
one sleep.

**The checks:**
- **U1–U3:** the k-cluster scores (k = 100, 10) and V's object count.
- **U4:** the encoder's 1-NN.
- **U5:** S's objects, in the joint brain and in the battery's brains; the walks' day 1.
- **U6:** A2's held-out accuracy in the joint brain.
- **U7:** non-interference (seed 0): the same stream without images, and without equations.
- **U8:** part 8's relabelled walks.
- **U10:** part 9's held-out answers' bits.
- **U11:** the cost. (U9 is the battery, run apart: tests/battery.py, tiers 1 and 2.)

Labels are used here only, to score.
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from arc_agent_v2.architecture.brain import Brain
from arc_agent_v2.config.config import JOINT, WALK, BrainConfig, retinotopy
from arc_agent_v2.worlds.arithmetic_world import ArithmeticWorld
from arc_agent_v2.worlds.mnist_world import MnistWorld

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
SEEDS = (0, 1, 2)
N_TRAIN, N_TEST = 3000, 1000
OPS = ("+", "-", "*")


def joint(seed, images=True, equations=True, n_train=N_TRAIN, n_test=N_TEST):
    """Train the joint brain on the stream (or the part of it the arm keeps) and score it."""
    cfg = BrainConfig(JOINT.sdr, JOINT.sensors, JOINT.regions, seed, JOINT.efference)
    t0 = time.time()
    mn = MnistWorld(retinotopy(cfg, "V"), seed=seed, n_train=n_train, n_test=n_test)
    ar = ArithmeticWorld(p=13, ops=OPS, train_frac=0.5, seed=seed)
    items = [("A", ep) for ep in ar.epoch(np.random.default_rng(seed))] + [("V", k) for k in range(n_train)]
    order = np.random.default_rng((seed, 12)).permutation(len(items))
    keep = {"A": equations, "V": images}
    brain, which = Brain(cfg), {}
    t1 = time.time()
    for i in order:
        kind, v = items[i]
        if not keep[kind]:
            continue
        which[len(brain.hippocampus.episodes)] = (kind, v)
        brain.episode([("symbols", x) for x in v] if kind == "A" else mn.episode("train", v))
    t2 = time.time()
    brain.sleep()
    t3 = time.time()
    out = dict(seed=seed, images=images, equations=equations, seconds_worlds=round(t1 - t0, 1),
               seconds_day=round(t2 - t1, 1), seconds_sleep=round(t3 - t2, 1))
    if equations:
        S = brain.frames["S"]
        out["S_objects"] = len(S.objects)
        qs = ar.queries()
        answers = []
        for op in OPS:
            q = [x for x in qs if x[2][0] == op]
            preds = [ar.decode(brain.predict_after(pre)) for pre, _, _ in q]
            answers += preds
            out["acc_" + op] = sum(p == ar.written(*e)[-1] for p, (_, _, e) in zip(preds, q)) / len(q)
        from arc_agent_v2.tests.test_fingerprints import fingerprint
        out["S_fingerprint"] = fingerprint(S)
        out["S_answers"] = answers
    if images:
        objs = brain.frames["V"].objects
        V = brain.regions["V"][0]
        t4 = time.time()
        post = []
        for k in range(n_test):
            brain.episode(mn.episode("test", k), learn=False)
            post.append(V.objects_posterior())
        out["seconds_test_images"] = round(time.time() - t4, 1)
        image_of = lambda e: which[e][1]
        label_of = {o.sdr: Counter(int(mn.train_labels[image_of(e)]) for e in o.members).most_common(1)[0][0]
                    for o in objs}
        out["V_objects"] = len(objs)
        out["V_sizes"] = sorted((o.n for o in objs), reverse=True)
        out["V_members"] = sorted(sorted(image_of(e) for e in o.members) for o in objs)
        ranked = [o.sdr for o in sorted(objs, key=lambda o: (-o.n, min(o.members)))]
        for k in (100, 10):
            top = ranked[:k]
            pred = [max(top, key=lambda o: p.get(o, 0.0)) for p in post]
            if k == 100:
                acc = float(np.mean([label_of[o] == y for o, y in zip(pred, mn.test_labels)]))
            else:
                M = np.zeros((len(top), 10))
                for o, y in zip(pred, mn.test_labels):
                    M[top.index(o), y] += 1
                r, c = linear_sum_assignment(-M)
                acc = float(M[r, c].sum() / n_test)
            out["k%d" % k] = acc
        t5 = time.time()
        out["nn"] = mn.nearest_neighbour()
        out["seconds_nn"] = round(time.time() - t5, 1)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def objects_elsewhere():
    """U5: the objects of the battery's arithmetic brains, and of the walks' day 1."""
    from arc_agent_v2.tests.trained import trained
    from arc_agent_v2.tests.test_a4 import STEPS
    from arc_agent_v2.worlds.walk_world import KINDS, WalkWorld
    out = {}
    for name, kw in (("A1", {}), ("A2", dict(ops=OPS)), ("reversal", dict(ops=("+", "-"), reverse=True))):
        out[name] = [len(trained(13, s, dict(train_frac=0.5, **kw))[1].frame.objects) for s in SEEDS]
    for kind in KINDS:
        n = []
        for s in SEEDS:
            world = WalkWorld(kind, seed=s)
            brain = Brain(WALK)
            brain.episode(world.stream(0, np.random.default_rng((s, 5)).integers(0, world.A, STEPS).tolist()))
            n.append(len(brain.sleep().objects))
        out[kind] = n
    return out


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "runs", "a12.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    say = lambda *a: print(*a, flush=True)
    t0 = time.time()
    res = dict(runs=[joint(s) for s in SEEDS])
    res["controls"] = [joint(0, images=False), joint(0, equations=False)]
    t1 = time.time()
    res["objects_elsewhere"] = objects_elsewhere()
    from arc_agent_v2.tests.test_a8 import walks
    from arc_agent_v2.worlds.walk_world import KINDS
    res["part8"] = [walks(k, s) for k in KINDS for s in SEEDS]
    from arc_agent_v2.tests.test_a9 import arm
    res["part9"] = [dict(world=w, seed=s, **{k: v for k, v in arm(w, 13, s, True).items() if k.endswith("bits")})
                    for w in ("+-", "+") for s in SEEDS]
    res["seconds_checks"] = round(time.time() - t1, 1)
    res["seconds_total"] = round(time.time() - t0, 1)
    json.dump(res, open(out_path, "w"), indent=1, default=str)

    for r in res["runs"]:
        say("U1-U3 seed %d: V %d objects (largest %s); 100 clusters %.3f; 10 clusters %.3f" % (
            r["seed"], r["V_objects"], r["V_sizes"][:8], r["k100"], r["k10"]))
    for r in res["runs"]:
        say("U4 seed %d: the encoder's 1-NN %.3f" % (r["seed"], r["nn"]))
    say("U5 S's objects in the joint brain: %s; elsewhere: %s" % ([r["S_objects"] for r in res["runs"]],
                                                                  res["objects_elsewhere"]))
    for r in res["runs"]:
        say("U6 seed %d: A2 held out %s" % (r["seed"], " ".join("%s %.3f" % (op, r["acc_" + op]) for op in OPS)))
    full, no_img, no_eq = res["runs"][0], res["controls"][0], res["controls"][1]
    say("U7 S without images: frame %s, answers %s; V without equations: members %s, scores %s" % (
        "identical" if full["S_fingerprint"] == no_img["S_fingerprint"] else "DIFFERENT",
        "identical" if full["S_answers"] == no_img["S_answers"] else "DIFFERENT",
        "identical" if full["V_members"] == no_eq["V_members"] else "DIFFERENT",
        "identical" if (full["k100"], full["k10"]) == (no_eq["k100"], no_eq["k10"]) else "DIFFERENT"))
    old8 = {(w["kind"], w["seed"]): w for w in json.load(open(os.path.join(HERE, "runs", "a8.json")))["walks"]}
    for r in res["part8"]:
        o = old8[(r["kind"], r["seed"])]
        cells = []
        for a in "abc":
            cells.append("%s %d/%d (part 8: %d/%d)" % (a, r[a]["known_right"], r[a]["known"], o[a]["known_right"],
                                                      o[a]["known"]))
        say("U8 %s seed %d: revisits known right: %s" % (r["kind"], r["seed"], "; ".join(cells)))
    for r in res["part9"]:
        say("U10 {%s} seed %d: %s" % (r["world"], r["seed"], " ".join("%s %.4f" % (k[:-5], v) for k, v in r.items()
                                                                        if k.endswith("bits"))))
    for r in res["runs"] + res["controls"]:
        say("U11 seed %d%s: %.1f s (day %.1f, sleep %.1f, test images %s, 1-NN %s)" % (
            r["seed"], "" if r["images"] and r["equations"] else (" (no images)" if not r["images"] else
                                                                   " (no equations)"),
            r["seconds"], r["seconds_day"], r["seconds_sleep"], r.get("seconds_test_images", "-"),
            r.get("seconds_nn", "-")))
    say("checks %.1f s; total %.1f s" % (res["seconds_checks"], res["seconds_total"]))


def test_a12_small():
    """A fast check: one brain learns both; S keeps one object and A2; V's objects explain its images."""
    r = joint(3, n_train=300, n_test=50)
    assert r["S_objects"] == 1 and min(r["acc_" + op] for op in OPS) == 1.0
    assert r["V_objects"] >= 1 and sum(r["V_sizes"]) == 300


if __name__ == "__main__":
    main(sys.argv[1:])
