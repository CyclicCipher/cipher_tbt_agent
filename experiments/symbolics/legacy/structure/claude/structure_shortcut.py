"""The shortcut test (docs/COORDINATES.md, part 11): after exploring and an excursion, go back to the start. Carried
geometry knows the torus wraps before it has ever wrapped; a Bayesian planner over part 8's tori only knows what this
trial's colours showed; graph memory can only reuse transitions it has seen.
    python -m structure.claude.structure_shortcut      (from experiments/symbolics)
"""
import json
import time
from collections import deque

import numpy as np

from structure.claude.structure_aliased import PAIRINGS, PMAX, models
from structure.claude.structure_miniature import walk
from structure.claude.structure_transfer import torus

SEEDS, TRIALS, EXPLORE, STEPS1 = 3, 20, 40, 300
NT = len(PAIRINGS) * PMAX * PMAX
K = np.arange(NT)
PAIR, H, W = K // (PMAX * PMAX), (K % (PMAX * PMAX)) // PMAX + 1, K % PMAX + 1
P = np.array(PAIRINGS)                                    # P[pairing] = ((a1, a2), (a3, a4))


def displacement(counts):
    """Each torus hypothesis's unwrapped displacement (x, y) for action counts."""
    a1, a2, a3, a4 = P[PAIR, 0, 0], P[PAIR, 0, 1], P[PAIR, 1, 0], P[PAIR, 1, 1]
    return counts[..., a1] - counts[..., a2], counts[..., a3] - counts[..., a4]


def home_plans(hist):
    """Each hypothesis's shortest way home from its own position, as action counts."""
    x, y = displacement(hist)
    x, y = x % H, y % W
    plans = np.zeros((NT, 4), int)
    a1, a2, a3, a4 = P[PAIR, 0, 0], P[PAIR, 0, 1], P[PAIR, 1, 0], P[PAIR, 1, 1]
    up, right = H - x < x, W - y < y                      # wrap forward when that is shorter
    plans[K, np.where(up, a1, a2)] += np.where(up, H - x, x)
    plans[K, np.where(right, a3, a4)] += np.where(right, W - y, y)
    return plans


def succeeds(plans, hist):
    """success[i, k]: plan i brings hypothesis k home."""
    x, y = displacement(hist[None, :] + plans)
    return (x % H == 0) & (y % W == 0)


def plane_plans(hist):
    """Exploratory (after S2 failed): undo the displacement in the plane, in each pairing's frame. It brings every
    hypothesis with that pairing home, whatever its periods, but no torus with periods <= 15 proposes it once the
    displacement exceeds half the period range."""
    plans = []
    for (a1, a2), (a3, a4) in PAIRINGS:
        x, y = hist[a1] - hist[a2], hist[a3] - hist[a4]
        c = np.zeros(4, int)
        c[a2 if x > 0 else a1] += abs(x)
        c[a4 if y > 0 else a3] += abs(y)
        plans.append(c)
    return np.array(plans)


def bfs(edges, src, dst):
    seen, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            return seen[u]
        for v in edges.get(u, ()):
            if v not in seen:
                seen[v] = seen[u] + 1
                q.append(v)
    return None


def main():
    t0 = time.time()
    out = {}
    for env, (h, w) in (("E1 torus 6x10", (6, 10)), ("E4 torus 4x15", (4, 15))):
        T, true_k = torus(h, w), (h - 1) * PMAX + (w - 1)     # pairing 0 is N/S with E/W
        for C in (6, 60):
            recs = []
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                colour_of = lambda: rng.permutation(60) if C == 60 else rng.integers(0, C, 60)
                a1, x1 = walk(T, STEPS1, rng)
                _, cum = models(a1, colour_of()[x1], C)
                learned = int(np.argmax(cum[:, -1]))
                for _ in range(TRIALS):
                    colour = colour_of()
                    acts = list(rng.integers(4, size=EXPLORE))
                    k, m = int(rng.integers(1, h)), int(rng.integers(1, w))
                    acts += [0] * k + [2] * m
                    acts = np.array(acts)
                    x = [int(rng.integers(60))]
                    for a in acts:
                        x.append(int(T[x[-1], a]))
                    x = np.array(x)
                    col = colour[x]
                    hist = np.bincount(acts, minlength=4)
                    dx, dy = displacement(hist)
                    plane = int(abs(dx[true_k]) + abs(dy[true_k]))
                    optimal = int(min(dx[true_k] % h, -dx[true_k] % h) + min(dy[true_k] % w, -dy[true_k] % w))
                    plans = home_plans(hist)
                    ok_true = succeeds(plans, hist)[:, true_k]
                    rec = dict(optimal=optimal, plane=plane, shortcut=optimal < plane)
                    # carried and true geometry: their own shortest plans
                    for name, kk in (("carried geometry", learned), ("true geometry", true_k)):
                        if kk < NT:
                            rec[name] = dict(success=bool(ok_true[kk]), length=int(plans[kk].sum()))
                        else:
                            rec[name] = dict(success=False, length=None)
                    # relearning: the plan most likely to reach home under this trial's posterior
                    _, cum2 = models(acts, col, C)
                    lw = cum2[:NT, -1]
                    post = np.exp(lw - lw.max())
                    post /= post.sum()
                    for name, cands in (("relearning", plans), ("relearning + plane route", np.vstack([plans, plane_plans(hist)]))):
                        uniq = np.unique(cands, axis=0)
                        score = succeeds(uniq, hist).astype(float) @ post
                        best = np.flatnonzero(score >= score.max() - 1e-9)
                        pick = uniq[best[np.argmin(uniq[best].sum(1))]]
                        x_, y_ = displacement(hist + pick)
                        rec[name] = dict(success=bool(x_[true_k] % h == 0 and y_[true_k] % w == 0), length=int(pick.sum()),
                                         p_success=float(score.max()))
                    if C == 60:
                        edges = {}
                        for t, a in enumerate(acts):
                            edges.setdefault(col[t], set()).add(col[t + 1])
                        d = bfs(edges, col[-1], col[0])
                        rec["graph memory"] = dict(success=d is not None, length=d)
                    recs.append(rec)
            agents = [a for a in ("carried geometry", "relearning", "relearning + plane route", "graph memory", "true geometry") if a in recs[0]]
            summ = {}
            for a in agents:
                succ = [r[a]["success"] for r in recs]
                ratio = [r[a]["length"] / r["optimal"] for r in recs if r[a]["success"] and r["optimal"] > 0]
                sc = [r for r in recs if r["shortcut"]]
                took = [r[a]["success"] and r[a]["length"] == r["optimal"] for r in sc]
                ratio_sc = [r[a]["length"] / r["optimal"] for r in sc if r[a]["success"] and r["optimal"] > 0]
                summ[a] = dict(success=float(np.mean(succ)), length_ratio=float(np.mean(ratio)) if ratio else None,
                               shortcut_taken=float(np.mean(took)) if took else None,
                               length_ratio_on_shortcut_trials=float(np.mean(ratio_sc)) if ratio_sc else None)
            frac_sc = float(np.mean([r["shortcut"] for r in recs]))
            out[f"{env}, C {C}"] = dict(summary=summ, shortcut_fraction=frac_sc, trials=recs)
            print(f"{env}, {C} colours: shortcut trials {frac_sc:.2f}", flush=True)
            for a, s in summ.items():
                print(f"   {a:24s} success {s['success']:.2f}  length ratio {s['length_ratio'] if s['length_ratio'] is None else round(s['length_ratio'], 3)}  "
                      f"shortcut taken {s['shortcut_taken'] if s['shortcut_taken'] is None else round(s['shortcut_taken'], 2)}  "
                      f"ratio on shortcut trials {s['length_ratio_on_shortcut_trials'] if s['length_ratio_on_shortcut_trials'] is None else round(s['length_ratio_on_shortcut_trials'], 3)}",
                      flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/structure_shortcut.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
