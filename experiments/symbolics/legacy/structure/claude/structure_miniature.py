"""The thesis in miniature (docs/COORDINATES.md, part 10; arc_agent/claude/DESIGN.md): learn the geometry from
repeated colours, compute its coordinates, then reuse one content model over new episodes and new colourings. The
content hypotheses are the geometry's quotients: colour constant on the cosets of a subgroup.
    python -m structure.claude.structure_miniature      (from experiments/symbolics)
"""
import json
import time

import numpy as np

from structure.claude.structure_aliased import EPS, PAIRINGS, PMAX, cell_pred, model_codes, models
from structure.claude.structure_irreps import choose, irreps, matrix_model, multiplication, prune
from structure.claude.structure_slip import score
from structure.claude.structure_transfer import torus

C, STEPS1, STEPS2, SEEDS, EPISODES = 6, 300, 200, 3, 3   # EPISODES transfer episodes of each kind
PREDICTORS = ("cell memory", "content model", "relearning", "edge memory")


def walk(T, steps, rng):
    actions = rng.integers(T.shape[1], size=steps)
    x = [int(rng.integers(len(T)))]
    for a in actions:
        x.append(int(T[x[-1], a]))
    return actions, np.array(x)


def closure(M, elems):
    s = set(int(e) for e in elems) | {0}
    while True:
        e = np.array(sorted(s))
        new = s | set(np.unique(M[np.ix_(e, e)]).tolist())
        if new == s:
            return frozenset(s)
        s = new


def subgroups(M):
    """All subgroups, as joins of cyclic ones, from the multiplication table alone."""
    found = {closure(M, [g]) for g in range(len(M))}
    while True:
        new = found | {closure(M, a | b) for a in found for b in found}
        if new == found:
            return sorted(found, key=len)
        found = new


def cosets(M, H):
    return M[:, np.array(sorted(H))].min(axis=1)      # each element's coset gH, named by its smallest member


def torus_table(k):
    """Transition table of torus hypothesis k (part 8's order: pairing, then h, then w)."""
    (a1, a2), (a3, a4) = PAIRINGS[k // (PMAX * PMAX)]
    h, w = (k % (PMAX * PMAX)) // PMAX + 1, k % PMAX + 1
    cells = np.arange(h * w)
    x, y = cells // w, cells % w
    TT = np.zeros((h * w, 4), int)
    TT[:, a1], TT[:, a2] = ((x + 1) % h) * w + y, ((x - 1) % h) * w + y
    TT[:, a3], TT[:, a4] = x * w + (y + 1) % w, x * w + (y - 1) % w
    return TT


def dist(pred):
    p = np.full(C, 1 / C)
    if pred >= 0:
        p = np.full(C, EPS / C)
        p[pred] += 1 - EPS
    return p


def episode(T, learned, colour, rng):
    actions, x = walk(T, STEPS2, rng)
    col = colour[x]
    model, keymap, cos = learned["model"], learned["keymap"], learned["cosets"]
    K = len(cos)
    addr = model["start"]
    cell = keymap[model["key"](addr)]
    cellmem = {cell: col[0]}
    mem, logw = -np.ones((K, cos.shape[1]), int), np.zeros(K)
    mem[np.arange(K), cos[:, cell]] = col[0]
    names, codes = model_codes(actions, col)                  # relearning: part 8's models, fresh in this episode
    preds = np.array([cell_pred(code, col) for code in codes])
    rw = np.zeros(len(codes))
    res = {p: dict(logp=0.0, acc=0.0, first_acc=0.0, curve=[]) for p in PREDICTORS}
    first = 0
    for t in range(1, STEPS2 + 1):
        o, a = col[t], int(actions[t - 1])
        addr = model["move"](addr, a)
        cell = keymap.get(model["key"](addr), -1)
        q = {"cell memory": dist(cellmem.get(cell, -1))}
        stored = mem[np.arange(K), cos[:, cell]]
        Pk = np.where(stored[:, None] >= 0, EPS / C, 1 / C) + np.where(stored[:, None] >= 0, (1 - EPS) * (np.arange(C) == stored[:, None]), 0)
        w = np.exp(logw - logw.max())
        q["content model"] = (w / w.sum()) @ Pk
        pr = preds[:, t]
        Pm = np.where(pr[:, None] >= 0, EPS / C, 1 / C) + np.where(pr[:, None] >= 0, (1 - EPS) * (np.arange(C) == pr[:, None]), 0)
        v = np.exp(rw - rw.max())
        q["relearning"] = (v / v.sum()) @ Pm
        q["edge memory"] = dist(pr[-1])
        is_first = x[t] not in x[:t]
        first += is_first
        for p, d in q.items():
            lp, acc = score(d, o)
            res[p]["logp"] += lp
            res[p]["acc"] += acc
            res[p]["first_acc"] += acc * is_first
            res[p]["curve"].append(acc)                          # exploratory: per-step accuracy
        logw += np.log(Pk[:, o])
        rw += np.log(Pm[:, o])
        mem[np.arange(K), cos[:, cell]] = o
        cellmem[cell] = o
    return {p: dict(accuracy=v["acc"] / STEPS2, first_visit_accuracy=v["first_acc"] / max(first, 1),
                    log_loss=-v["logp"] / STEPS2, curve=v["curve"]) for p, v in res.items()}, int(first)


def main():
    t0 = time.time()
    out = {}
    for env, T in (("E1 torus 6x10", torus(6, 10)), ("E4 torus 4x15", torus(4, 15))):
        true_subs = subgroups(multiplication(T))
        striped = [H for H in true_subs if 2 <= len(T) // len(H) <= 20]
        out[env] = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            actions, x = walk(T, STEPS1, rng)
            names, cum = models(actions, rng.integers(0, C, len(T))[x], C)
            k = int(np.argmax(cum[:, -1]))
            rec = dict(learned=names[k], true_subgroups=len(true_subs))
            if k >= 3 * PMAX * PMAX:
                rec["note"] = "no torus learned"
                out[env].append(rec)
                continue
            TT = torus_table(k)
            M = multiplication(TT)
            subs = subgroups(M)
            reps, classes = irreps(TT, np.random.default_rng(1000 + seed))
            chosen, distinct = choose(classes, len(TT))
            chosen = prune(chosen, len(TT))
            model = matrix_model(chosen, TT.shape[1])
            keymap = {model["key"]([r["mats"][c] for r in chosen]): c for c in range(len(TT))}
            learned = dict(model=model, keymap=keymap, cosets=np.array([cosets(M, H) for H in subs]))
            rec.update(learned_subgroups=len(subs), distinct_addresses=distinct, address_dims=[r["d"] for r in chosen])
            for kind in ("random", "striped"):
                eps_ = []
                for _ in range(EPISODES):
                    if kind == "random":
                        colour = rng.integers(0, C, len(T))
                        H = None
                    else:
                        H = striped[int(rng.integers(len(striped)))]
                        colour = rng.integers(0, C, len(T))[cosets(multiplication(T), H)]
                    r, nfirst = episode(T, learned, colour, rng)
                    eps_.append(dict(result=r, first_visits=nfirst, cosets=None if H is None else len(T) // len(H)))
                rec[kind] = eps_
            out[env].append(rec)
            print(f"{env} seed {seed}: learned {rec['learned']} | subgroups {len(subs)} (true {len(true_subs)}) | "
                  f"distinct addresses {distinct}, address blocks {rec['address_dims']}", flush=True)
    print()
    summary = {}
    for kind in ("random", "striped"):
        for p in PREDICTORS:
            vals = [e["result"][p] for env in out.values() for rec in env if kind in rec for e in rec[kind]]
            summary[(kind, p)] = {m: float(np.mean([v[m] for v in vals])) for m in vals[0] if m != "curve"}
            s = summary[(kind, p)]
            print(f"{kind:8s} {p:14s} accuracy {s['accuracy']:.3f}  first-visit {s['first_visit_accuracy']:.3f}  "
                  f"log loss {s['log_loss']:.2f}", flush=True)
    print("exploratory: accuracy by step window (1-25, 26-50, 51-100, 101-200)")
    for kind in ("random", "striped"):
        for p in PREDICTORS:
            curves = np.array([e["result"][p]["curve"] for env in out.values() for rec in env if kind in rec for e in rec[kind]])
            m = curves.mean(0)
            print(f"   {kind:8s} {p:14s} " + "  ".join(f"{m[a:b].mean():.3f}" for a, b in ((0, 25), (25, 50), (50, 100), (100, 200))), flush=True)
    cos_counts = [e["cosets"] for env in out.values() for rec in env if "striped" in rec for e in rec["striped"]]
    print(f"striped episodes' coset counts: {cos_counts}")
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(dict(runs=out, summary={f"{k[0]} / {k[1]}": v for k, v in summary.items()}),
              open("runs/structure_miniature.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
