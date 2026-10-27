"""Slips with repeated colours, structure known (docs/COORDINATES.md, part 9): an action sometimes does nothing, and
the agent isn't told. Dead reckoning, a particle filter over slip histories (each with its own map), a soft filter
(one belief, one shared map of colour counts) and edge memory predict the next colour; the oracle sees the slips.
    python -m structure.claude.structure_slip      (from experiments/symbolics)
"""
import json
import time

import numpy as np

from structure.claude.structure_transfer import torus

STEPS, EPS, SEEDS, K, ALPHA = 600, 0.01, 3, 128, 0.01
COLOURS, SLIPS = (3, 6, 60), (0.0, 0.05, 0.2)


def cell_dist(stored, C):
    """Predicted colour distribution from a cell memory entry (-1 = never written)."""
    p = np.full(C, 1 / C)
    if stored >= 0:
        p = np.full(C, EPS / C)
        p[stored] += 1 - EPS
    return p


def score(p, o):
    top = np.flatnonzero(p >= p.max() - 1e-12)
    return float(np.log(p[o])), (o in top) / len(top)


def run(T, colour, C, p, rng, beam=K):
    n, A = T.shape
    x = int(rng.integers(n))
    acts, slips = rng.integers(A, size=STEPS), rng.random(STEPS) < p
    res = {k: dict(logp=0.0, acc=0.0, loc=0) for k in ("oracle", "oracle, past slips only", "dead reckoning", "particle filter",
                                                     "soft filter", "edge memory")}
    o = colour[x]
    oracle_map, dr_map, dr = {x: o}, {x: o}, x
    pos, maps, logw = np.array([x]), np.full((1, n), -1), np.zeros(1)
    maps[0, x] = o
    b, counts = np.zeros(n), np.zeros((n, C))
    b[x] = 1.0
    counts[x, o] += 1
    edges, prev_o = {}, o
    with np.errstate(divide="ignore"):
        lstay, lmove = np.log(p), np.log(1 - p)
    for t in range(STEPS):
        a = int(acts[t])
        x_prev = x
        x = x if slips[t] else int(T[x, a])
        o = int(colour[x])
        preds = {"oracle": cell_dist(oracle_map.get(x, -1), C),          # knows whether this step slipped
                 "oracle, past slips only": (1 - p) * cell_dist(oracle_map.get(int(T[x_prev, a]), -1), C)
                 + p * cell_dist(oracle_map.get(x_prev, -1), C)}   # exploratory: the fair bound
        dr = int(T[dr, a])
        preds["dead reckoning"] = cell_dist(dr_map.get(dr, -1), C)
        # particle filter: every slip history branches into moved / slipped
        cp = np.concatenate([T[pos, a], pos])
        cm = np.concatenate([maps, maps])
        cw = np.concatenate([logw + lmove, logw + lstay])
        live = np.isfinite(cw)
        cp, cm, cw = cp[live], cm[live], cw[live]
        stored = cm[np.arange(len(cp)), cp]
        P = np.where(stored[:, None] >= 0, EPS / C, 1 / C) + np.where(stored[:, None] >= 0, (1 - EPS) * (np.arange(C) == stored[:, None]), 0)
        w = np.exp(cw - cw.max())
        w /= w.sum()
        preds["particle filter"] = w @ P
        res["particle filter"]["loc"] += int(cp[np.argmax(cw)] == x)
        cw = cw + np.log(P[:, o])
        cm[np.arange(len(cp)), cp] = o
        keep = np.argsort(-cw)[:beam]
        pos, maps, logw = cp[keep], cm[keep], cw[keep] - cw[keep].max()
        # soft filter: one belief, one shared map of colour counts
        bm = np.zeros(n)
        bm[T[:, a]] = b
        bp = (1 - p) * bm + p * b
        Pc = (counts + ALPHA) / (counts.sum(1, keepdims=True) + ALPHA * C)
        preds["soft filter"] = bp @ Pc
        res["soft filter"]["loc"] += int(np.argmax(bp) == x)
        b = bp * Pc[:, o]
        b /= b.sum()
        counts[:, o] += b
        preds["edge memory"] = cell_dist(edges.get((prev_o, a), -1), C)
        for k, q in preds.items():
            lp, acc = score(q, o)
            res[k]["logp"] += lp
            res[k]["acc"] += acc
        oracle_map[x], dr_map[dr], edges[(prev_o, a)], prev_o = o, o, o, o
    return {k: dict(log_loss=-v["logp"] / STEPS, accuracy=v["acc"] / STEPS, localized=v["loc"] / STEPS) for k, v in res.items()}


def main():
    t0 = time.time()
    T, out = torus(), {}
    for C in COLOURS:
        for p in SLIPS:
            runs = []
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                colour = rng.permutation(60) if C == 60 else rng.integers(0, C, 60)
                runs.append(run(T, colour, C, p, rng))
            mean = {k: {m: float(np.mean([r[k][m] for r in runs])) for m in runs[0][k]} for k in runs[0]}
            out[f"C {C}, p {p}"] = dict(mean=mean, runs=runs)
            print(f"C {C:2d}, slip {p:4.2f}: " + " | ".join(
                f"{k} acc {v['accuracy']:.3f} loss {v['log_loss']:.2f}" + (f" loc {v['localized']:.2f}" if "filter" in k else "")
                for k, v in mean.items()), flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/structure_slip.json", "w"), indent=1)


def exploratory_k():
    """Exploratory (after A3 failed): is the particle filter's gap the pruning to 128 histories, or the data?"""
    t0, T, out = time.time(), torus(), {}
    for C in COLOURS:
        for p in (0.05, 0.2):
            for k in (128, 1024, 8192):
                accs = []
                for seed in range(SEEDS):
                    rng = np.random.default_rng(seed)
                    colour = rng.permutation(60) if C == 60 else rng.integers(0, C, 60)
                    accs.append(run(T, colour, C, p, rng, k)["particle filter"]["accuracy"])
                out[f"C {C}, p {p}, K {k}"] = float(np.mean(accs))
                print(f"C {C:2d}, slip {p:4.2f}, K {k:5d}: particle filter accuracy {np.mean(accs):.3f}  ({time.time() - t0:.0f} s)", flush=True)
    json.dump(out, open("runs/structure_slip_k.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    exploratory_k() if "--k" in sys.argv else main()
