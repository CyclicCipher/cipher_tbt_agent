"""Prequential marginal likelihood for the structural hypotheses (docs/COORDINATES.md, parts 3 and 4): each observed
entry is coded from the completion of the entries before it, so an axiom earns credit for restricting the allowed
values as well as for filling them. Part 4 adds the hypothesis 'chain' (every a + b is a or b) and reads off ranks.
    python -m structure.claude.structure_prequential      (from experiments/symbolics)
"""
import json
import math
import time

import numpy as np

from structure.claude.structure_hypotheses import EPS, HYPOTHESES, N, complete, tasks

SPACE = dict(HYPOTHESES, chain={"comm", "assoc", "idem", "total"})
TRUTH = {"T1 cyclic": "abelian group", "T2 Z6 x Z10": "abelian group", "T3 A5": "group",
         "T4 chain max": "semilattice", "T5 subtraction": "quasigroup", "T6 random": "none"}
ORDERS = 3


def checkpoints(m, start=20, growth=1.15):
    cps, b = [0], min(start, m)
    while True:
        cps.append(b)
        if b >= m:
            return cps
        b = min(m, math.ceil(b * growth))


def log_prob(full, axioms, r, c, v):
    """log P(v) for entries (r, c) given the completion `full` of the entries before them."""
    pred = full[r, c]
    det = pred >= 0
    q = (pred == v).astype(float)
    u = ~det
    if u.any():
        ru, cu, vu, k = r[u], c[u], v[u], np.arange(u.sum())
        size, ok = np.full(len(ru), float(N)), np.ones(len(ru), bool)
        if "latin" in axioms:
            kr, kc = np.nonzero(full >= 0)
            R = np.zeros((N, N), bool)
            C = np.zeros((N, N), bool)
            R[kr, full[kr, kc]] = True                    # value already in the row
            C[kc, full[kr, kc]] = True                    # value already in the column
            allowed = ~R[ru] & ~C[cu]
            size, ok = allowed.sum(1).astype(float), allowed[k, vu]
        if "total" in axioms:
            a1 = allowed[k, ru] if "latin" in axioms else np.ones(len(ru), bool)
            a2 = (allowed[k, cu] if "latin" in axioms else np.ones(len(ru), bool)) & (cu != ru)
            size, ok = a1 + a2.astype(float), ((vu == ru) & a1) | ((vu == cu) & a2)
        q[u] = np.where(size > 0, ok / np.maximum(size, 1), 0.0)
    return np.log((1 - EPS) * q + EPS / N)


def prequential(T_obs, axioms, rng, rules=()):
    obs = np.argwhere(T_obs >= 0)
    obs = obs[rng.permutation(len(obs))]
    fit, total = -np.ones_like(T_obs), 0.0
    cps = checkpoints(len(obs))
    for s, e in zip(cps[:-1], cps[1:]):
        full, _ = complete(fit, axioms, rules)
        r, c = obs[s:e, 0], obs[s:e, 1]
        total += float(log_prob(full, axioms, r, c, T_obs[r, c]).sum())
        fit[r, c] = T_obs[r, c]
    return total


def ranks(T_obs, pi):
    full, _ = complete(T_obs, SPACE["chain"])
    off = ~np.eye(N, dtype=bool)
    above = ((full == np.arange(N)[:, None]) & off).sum(1)   # a + b = a: b is below a
    below = ((full == np.arange(N)[None, :]) & off).sum(1)   # a + b = b: a is below b
    est = np.empty(N, int)
    est[np.argsort(above - below, kind="stable")] = np.arange(N)
    pos = np.empty(N, int)
    pos[pi] = np.arange(N)
    return {"spearman": float(np.corrcoef(est, pos)[0, 1]), "largest rank error": int(np.max(np.abs(est - pos))),
            "elements misplaced": int(np.sum(est != pos))}


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)                        # part 2's masks, drawn in the same order
    out = {}
    for frac in (0.5, 0.25):
        print(f"\nobserved fraction {frac}", flush=True)
        for name, (T, pi) in tasks(np.random.default_rng(7)).items():
            mask = rng.random((N, N)) < frac
            T_obs = np.where(mask, T, -1)
            per = {h: [prequential(T_obs, ax, np.random.default_rng(100 + o)) for o in range(ORDERS)]
                   for h, ax in SPACE.items()}
            mean = {h: float(np.mean(v)) for h, v in per.items()}
            rec = {"log marginal likelihood (mean of orders)": mean, "per order": per}
            for part, space in (("part 3", HYPOTHESES), ("part 4", SPACE)):
                ranked = sorted(space, key=lambda h: -mean[h])
                win, second = ranked[0], ranked[1]
                margins = [per[win][o] - max(per[h][o] for h in space if h != win) for o in range(ORDERS)]
                rec[part] = dict(winner=win, runner_up=second, margin=mean[win] - mean[second],
                                 smallest_margin_over_orders=min(margins))
            if name == "T4 chain max":
                rec["part 4 ranks"] = ranks(T_obs, pi)
            out[f"{name} @ {frac}"] = rec
            p3, p4 = rec["part 3"], rec["part 4"]
            print(f"   {name:16s} part 3: {p3['winner']:15s} {'right' if p3['winner'] == TRUTH[name] else 'WRONG':5s} "
                  f"margin {p3['margin']:8.1f} (min over orders {p3['smallest_margin_over_orders']:8.1f}) over {p3['runner_up']}"
                  f" | part 4: {p4['winner']} by {p4['margin']:.1f} over {p4['runner_up']}", flush=True)
            print("      " + ", ".join(f"{h} {mean[h]:.0f}" for h in sorted(SPACE, key=lambda h: -mean[h])), flush=True)
            if "part 4 ranks" in rec:
                print(f"      ranks: {rec['part 4 ranks']}", flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/structure_prequential.json", "w"), indent=1)


if __name__ == "__main__":
    main()
