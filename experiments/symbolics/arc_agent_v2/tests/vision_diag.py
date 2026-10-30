"""Vision diagnostics (docs/VISION.md; docs/LAB.md part 12, D1). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m arc_agent_v2.tests.vision_diag [d1|d2]

**D1:** does the retina's encoder keep similarity? Pixel overlap against SDR overlap, for a patch and the same patch
after the image moves a pixel, and for same-digit and different-digit pairs at one position.

**D2** (docs/VISION.md §6): do local features at object-relative locations carry the digit?
- **The measure:** the content code length of the images under the true digit partition, against one object.
  - **How:** leave-one-out, information.py's content model, through objects.py's `_Data`.
  - **Reported:** the gain per image, and the net after the partition's own cost.
- **The representations:** whole tiles, small tiles, a silhouette, and a hand-computed local orientation (the
  structure tensor), at retinal or bounding-box locations.

These are probes of what a representation carries. Labels are used only to form the partition being measured.
Nothing here is in the brain.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from arc_agent_v2.architecture import objects as O
from arc_agent_v2.architecture.HDClibrary import SDR
from arc_agent_v2.worlds.mnist_world import MnistWorld, _idx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # arc_agent_v2/
D, W = 2048, 40


def d1():
    rng = np.random.default_rng(0)
    imgs = _idx("train-images-idx3-ubyte")[:3000] > 127
    labels = _idx("train-labels-idx1-ubyte")[:3000]
    R = np.random.default_rng((0, 4949)).standard_normal((D, 49))
    sdr = lambda p: set(np.argpartition(-(R @ np.where(p.ravel(), 1.0, -1.0)), W)[:W].tolist())
    tile = lambda img, r, c: img[7 * r:7 * r + 7, 7 * c:7 * c + 7]

    def jacc(a, b):
        a, b = a.ravel(), b.ravel()
        u = (a | b).sum()
        return (a & b).sum() / u if u else 1.0

    out = {}
    for name, shift in (("right", (0, 1)), ("down", (1, 0))):
        pj, so = [], []
        for k in range(1000):
            moved = np.roll(imgs[k], shift, axis=(0, 1))
            for r, c in ((1, 1), (1, 2), (2, 1), (2, 2)):
                a, b = tile(imgs[k], r, c), tile(moved, r, c)
                if a.sum() >= 5 and b.sum() >= 5:
                    pj.append(jacc(a, b))
                    so.append(len(sdr(a) & sdr(b)) / W)
        out["shift_" + name] = dict(pixels=float(np.mean(pj)), sdr=float(np.mean(so)))
    idx = [k for k in range(3000) if tile(imgs[k], 1, 1).sum() >= 5]
    codes = {k: sdr(tile(imgs[k], 1, 1)) for k in idx}
    acc = {"same": ([], []), "different": ([], [])}
    for _ in range(20000):
        i, j = rng.choice(idx, 2, replace=False)
        key = "same" if labels[i] == labels[j] else "different"
        acc[key][0].append(jacc(tile(imgs[i], 1, 1), tile(imgs[j], 1, 1)))
        acc[key][1].append(len(codes[i] & codes[j]) / W)
    for key, (p, s) in acc.items():
        out[key] = dict(pixels=float(np.mean(p)), sdr=float(np.mean(s)))
    return out


# ---- D2 -----------------------------------------------------------------------------------------------------------
def _images(seed, n):
    """The same training images as MnistWorld(seed=seed, n_train=n), with their labels and part 12's tile SDRs."""
    rng = np.random.default_rng((seed, 28))
    tr = _idx("train-images-idx3-ubyte")
    tri = np.sort(rng.choice(len(tr), n, replace=False))
    from arc_agent_v2.config.config import JOINT, BrainConfig, retinotopy
    cfg = BrainConfig(JOINT.sdr, JOINT.sensors, JOINT.regions, seed, JOINT.efference)
    world = MnistWorld(retinotopy(cfg, "V"), seed=seed, n_train=n, n_test=1)
    assert (world.train_labels == _idx("train-labels-idx1-ubyte")[tri]).all()
    return tr[tri], world.train_labels, world.train


def _symbols(names, seed):
    rng = np.random.default_rng((seed, 2222))
    return {s: SDR.random(D, W, rng) for s in names}


def _orientation(img):
    """(per pixel: the structure tensor's dominant orientation bin of 8, its energy; the smoothed ink)."""
    I = gaussian_filter(img / 255.0, 1.0)
    gx, gy = sobel(I, axis=1), sobel(I, axis=0)
    Jxx, Jyy, Jxy = (gaussian_filter(a, 1.5) for a in (gx * gx, gy * gy, gx * gy))
    theta = (0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)) % np.pi
    return np.floor(theta / np.pi * 8).astype(int) % 8, Jxx + Jyy, I


def _points(n_side, box=None):
    """n_side x n_side points: on a retinal grid (2, 6, ... for 7; the tiles' centres for 4), or spread over a box."""
    if box is None:
        if n_side == 4:
            return [(r, c) for r in (3, 10, 17, 24) for c in (3, 10, 17, 24)]
        return [(r, c) for r in range(2, 28, 4) for c in range(2, 28, 4)]
    r0, r1, c0, c1 = box
    rr = np.rint(np.linspace(r0, r1, n_side)).astype(int)
    cc = np.rint(np.linspace(c0, c1, n_side)).astype(int)
    return [(r, c) for r in rr for c in cc]


def representations(images, tiles16, seed):
    """{name: [per image: [(cell, sensor, SDR)]]}."""
    n = len(images)
    out = {"tiles16": [[(p, "r", s) for p, s in enumerate(t)] for t in tiles16]}
    R16 = np.random.default_rng((seed, 1616)).standard_normal((D, 16))
    blocks = images.reshape(n, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(n, 49, 16)
    v = np.where(blocks > 127, 1.0, -1.0) @ R16.T
    top = np.argpartition(-v, W, axis=2)[:, :, :W]
    out["tiles49"] = [[(p, "r", SDR._sorted(D, np.sort(top[i, p]).astype(np.int32))) for p in range(49)]
                      for i in range(n)]
    sym = _symbols(["blank", "ink"] + ["o%d" % b for b in range(8)], seed)
    ink = (blocks > 127).any(2)
    out["ink49"] = [[(p, "r", sym["ink" if ink[i, p] else "blank"]) for p in range(49)] for i in range(n)]
    feats = [_orientation(img.astype(float)) for img in images]
    E = np.concatenate([e.ravel() for _, e, _ in feats])
    thr = 0.1 * np.percentile(E, 95)                    # an edge: energy above a tenth of the 95th percentile
    for name, side, boxed in (("orient16", 4, False), ("orient49", 7, False), ("orient49box", 7, True)):
        reps = []
        for img, (b, e, I) in zip(images, feats):
            box = None
            if boxed:
                rows, cols = np.flatnonzero((img > 127).any(1)), np.flatnonzero((img > 127).any(0))
                box = (rows[0], rows[-1], cols[0], cols[-1])
            obs = []
            for p, (r, c) in enumerate(_points(side, box)):
                s = "o%d" % b[r, c] if e[r, c] > thr else ("ink" if I[r, c] > 0.2 else "blank")
                obs.append((p, "r", sym[s]))
            reps.append(obs)
        out[name] = reps
    return out, float(thr)


def code_length(obs, labels):
    """(content gain per image of the true partition over one object, the partition's extra cost per image)."""
    eps = list(enumerate(obs))
    data = O._Data(eps, D, W)
    rows = np.arange(data.ep.size)

    def content(groups):
        total = 0.0
        for g in np.unique(groups):
            mem = groups[data.ep] == g
            C, N = data.stats(rows[mem])
            total += data.loglik(rows[mem], C, N, np.ones(mem.sum(), bool)).sum()
        return total

    def crp(groups):
        return sum(math.lgamma((groups == g).sum()) for g in np.unique(groups)) - math.lgamma(len(groups) + 1.0)

    one = np.zeros(len(eps), int)
    n = len(eps)
    return (content(labels) - content(one)) / n, (crp(one) - crp(labels)) / n


def d2(seed=5, n=3000):
    images, labels, tiles16 = _images(seed, n)
    reps, thr = representations(images, tiles16, seed)
    out = dict(seed=seed, n=n, edge_threshold=thr)
    for name, obs in reps.items():
        gain, cost = code_length(obs, labels)
        out[name] = dict(gain=gain, cost=cost, net=gain - cost, locations=len(obs[0]))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "d2"
    res = d1() if which == "d1" else d2()
    json.dump(res, open(os.path.join(HERE, "runs", "vision_%s.json" % which), "w"), indent=1)
    if which == "d1":
        for k, v in res.items():
            print("%-14s pixels %.3f  SDR %.3f" % (k, v["pixels"], v["sdr"]))
    else:
        print("edge threshold %.4g" % res["edge_threshold"])
        for k, v in res.items():
            if isinstance(v, dict):
                print("%-12s %2d locations: gain %.2f nats/image, the partition's cost %.2f, net %+.2f" % (
                    k, v["locations"], v["gain"], v["cost"], v["net"]))
