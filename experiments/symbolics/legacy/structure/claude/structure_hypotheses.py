"""Competing structural hypotheses for a partly observed operation table (docs/COORDINATES.md, part 2).
Each hypothesis is a set of axioms that completes the table and can be contradicted; a predictive likelihood on
held-out observed entries picks one; the winner's coordinates are computed.
    python -m structure.claude.structure_hypotheses      (from experiments/symbolics)
"""
import itertools
import json
import math
import time

import numpy as np

N, EPS = 60, 0.01
HYPOTHESES = {"none": set(), "commutative": {"comm"}, "commutative semigroup": {"comm", "assoc"},
              "semilattice": {"comm", "assoc", "idem"}, "quasigroup": {"latin"}, "group": {"assoc", "latin"},
              "abelian group": {"comm", "assoc", "latin"}}


# ---- tasks ----------------------------------------------------------------------------------------------------------
def a5():
    perms = [p for p in itertools.permutations(range(5))
             if sum(1 for i in range(5) for j in range(i + 1, 5) if p[i] > p[j]) % 2 == 0]
    index = {p: k for k, p in enumerate(perms)}
    return np.array([[index[tuple(p[q[i]] for i in range(5))] for q in perms] for p in perms])


def tasks(rng):
    i = np.arange(N)
    raw = {"T1 cyclic": (i[:, None] + i[None, :]) % N,
           "T2 Z6 x Z10": ((i[:, None] // 10 + i[None, :] // 10) % 6) * 10 + (i[:, None] % 10 + i[None, :] % 10) % 10,
           "T3 A5": a5(),
           "T4 chain max": np.maximum(i[:, None], i[None, :]),
           "T5 subtraction": (i[:, None] - i[None, :]) % N,
           "T6 random": rng.integers(0, N, (N, N))}
    out = {}
    for name, T in raw.items():                           # relabel by a random permutation
        pi = rng.permutation(N)
        R = np.empty_like(T)
        R[np.ix_(pi, pi)] = pi[T]
        out[name] = (R, pi)
    return out


# ---- completion ------------------------------------------------------------------------------------------------------
def _put(T, r, c, v):
    """Write proposals into unknown cells (first proposal per cell wins); return how many conflict with known cells."""
    if not len(r):
        return 0
    known = T[r, c] >= 0
    bad = int(np.sum(known & (T[r, c] != v)))
    r, c, v = r[~known], c[~known], v[~known]
    if len(r):
        _, first = np.unique(r * len(T) + c, return_index=True)
        T[r[first], c[first]] = v[first]
    return bad


def _assoc_more(T, latin):
    """Part 7: the other three patterns of (a + b) + d = a + (b + d) among a + b = c, b + d = f, c + d = e, a + f = e.
    The reverse pattern needs associativity alone; the two that recover a + b or b + d need cancellation (Latin)."""
    n, contra = len(T), 0
    if latin:
        Linv, Rinv = -np.ones((n, n), int), -np.ones((n, n), int)
        kr, kc = np.nonzero(T >= 0)
        Linv[kr, T[kr, kc]] = kc                          # a + f = e  ->  Linv[a, e] = f
        Rinv[kc, T[kr, kc]] = kr                          # x + d = e  ->  Rinv[d, e] = x
    for a in range(n):
        B = np.nonzero(T[a] >= 0)[0]
        if not len(B):
            continue
        C = T[a, B]
        Fm = T[B]                                         # reverse: a+b=c, b+d=f, a+f=e  ->  c+d=e
        Ea = np.full(Fm.shape, -1)
        Ea[Fm >= 0] = T[a, Fm[Fm >= 0]]
        i, d = np.nonzero(Ea >= 0)
        contra += _put(T, C[i], d, Ea[i, d])
        if latin:
            Em = T[C]                                     # a+b=c, c+d=e, a+f=e  ->  b+d=f
            Fa = np.full(Em.shape, -1)
            Fa[Em >= 0] = Linv[a, Em[Em >= 0]]
            i, d = np.nonzero(Fa >= 0)
            contra += _put(T, B[i], d, Fa[i, d])
            Rb, Rc = Rinv[:, B], Rinv[:, C]               # b+d=f, a+f=e, c+d=e  ->  a+b=c  (here B are the f, C the e)
            ok = (Rb >= 0) & (Rc >= 0)
            contra += _put(T, np.full(int(ok.sum()), a), Rb[ok], Rc[ok])
    return contra


def _sudoku(T):
    """Part 7: Latin propagation. Naked single: a cell with one allowed value gets it. Hidden single: a value missing
    from a row (column) that fits only one of its cells goes there. No option left is a contradiction."""
    n = len(T)
    kr, kc = np.nonzero(T >= 0)
    R, C = np.zeros((n, n), bool), np.zeros((n, n), bool)
    R[kr, T[kr, kc]] = True
    C[kc, T[kr, kc]] = True
    U = T < 0
    nR, nC, Ui = (~R).astype(np.int32), (~C).astype(np.int32), U.astype(np.int32)
    count = nR @ nC.T                                     # allowed values per cell
    contra = int(np.sum(U & (count == 0)))
    r, c = np.nonzero(U & (count == 1))
    if len(r):
        contra += _put(T, r, c, np.argmax(~R[r] & ~C[c], axis=1))
    rows = Ui @ nC                                        # rows[r, v]: unknown cells of row r whose column lacks v
    contra += int(np.sum(~R & (rows == 0)))
    r, v = np.nonzero(~R & (rows == 1))
    if len(r):
        contra += _put(T, r, np.argmax(U[r] & ~C[:, v].T, axis=1), v)
    cols = Ui.T @ nR                                      # cols[c, v]: unknown cells of column c whose row lacks v
    contra += int(np.sum(~C & (cols == 0)))
    c, v = np.nonzero(~C & (cols == 1))
    if len(c):
        contra += _put(T, np.argmax(U[:, c].T & ~R[:, v].T, axis=1), c, v)
    return contra


def complete(T, axioms, rules=()):
    """Close a partial table under the axioms. `rules` adds part 7's stronger inference: "assoc4", "sudoku"."""
    T, contra = T.copy(), 0
    n = len(T)
    for _ in range(50):
        before = (T >= 0).sum()
        if "comm" in axioms:
            both = (T >= 0) & (T.T >= 0)
            contra += int(np.sum(both & (T != T.T)) // 2)
            T = np.where((T < 0) & (T.T >= 0), T.T, T)
        if "idem" in axioms:
            d = np.arange(n)
            known = T[d, d] >= 0
            contra += int(np.sum(known & (T[d, d] != d)))
            T[d[~known], d[~known]] = d[~known]
        if "assoc" in axioms:
            for a in range(n):
                B = np.nonzero(T[a] >= 0)[0]
                if not len(B):
                    continue
                C = T[a, B]
                Fm, Em = T[B], T[C]
                ok = (Fm >= 0) & (Em >= 0)
                f, e = Fm[ok], Em[ok]
                if not len(f):
                    continue
                kn = T[a, f] >= 0
                contra += int(np.sum(kn & (T[a, f] != e)))
                if (~kn).any():
                    fu, first = np.unique(f[~kn], return_index=True)
                    T[a, fu] = e[~kn][first]
        if "assoc" in axioms and "assoc4" in rules:
            contra += _assoc_more(T, "latin" in axioms)
        if "total" in axioms:                             # every a + b is a or b (part 4); restricts, never fills
            r, c = np.nonzero(T >= 0)
            contra += int(np.sum((T[r, c] != r) & (T[r, c] != c)))
        if "latin" in axioms:
            for M in (T, T.T):                            # rows, then columns (T.T is a view: writes go through)
                for r in range(n):
                    row = M[r]
                    vals = row[row >= 0]
                    contra += len(vals) - len(np.unique(vals))
                    miss = np.nonzero(row < 0)[0]
                    if len(miss) == 1:
                        left = np.setdiff1d(np.arange(n), vals)
                        if len(left) == 1:
                            row[miss[0]] = left[0]
        if "latin" in axioms and "sudoku" in rules:
            contra += _sudoku(T)
        if (T >= 0).sum() == before:
            break
    return T, contra


def score(T_obs, axioms, rng, splits=3):
    obs = np.argwhere(T_obs >= 0)
    total = 0.0
    for _ in range(splits):
        perm = rng.permutation(len(obs))
        check = obs[perm[: len(obs) // 5]]
        fit = T_obs.copy()
        fit[check[:, 0], check[:, 1]] = -1
        full, contra = complete(fit, axioms)
        pred = full[check[:, 0], check[:, 1]]
        true = T_obs[check[:, 0], check[:, 1]]
        det = pred >= 0
        total += np.sum(det & (pred == true)) * math.log(1 - EPS) + np.sum(det & (pred != true)) * math.log(EPS)
        total += np.sum(~det) * math.log(1 / N) + contra * math.log(EPS)
    return float(total)


# ---- coordinates of the winner ---------------------------------------------------------------------------------------
def coordinates(name, full, pi, true_T):
    if (full < 0).any():
        return {"note": f"table not fully completed ({int((full < 0).sum())} entries unknown)"}
    if name == "abelian group":
        ident = int(np.nonzero([np.all(full[e] == np.arange(N)) for e in range(N)])[0][0])
        S = np.zeros((N, N))
        for a, w in enumerate(np.random.default_rng(1).normal(size=N)):
            S[full[a], np.arange(N)] += w
        _, vecs = np.linalg.eig(S.T)
        chis = (vecs / vecs[ident]).T
        orders = []
        for c in chis:
            orders.append(next((k for k in range(1, N + 1) if np.max(np.abs(c ** k - 1)) < 1e-6), None))
        return {"largest character order": max(o for o in orders if o), "cyclic": N in orders,
                "orders": sorted(set(o for o in orders if o))}
    if name == "group":
        centre = [a for a in range(N) if np.all(full[a] == full[:, a])]
        return {"centre size": len(centre)}
    if name == "semilattice":
        rank = np.array([int(np.sum(full[a] == a)) for a in range(N)])
        true_rank = np.empty(N, int)
        true_rank[pi] = np.arange(N) + 1                  # element pi[k] is the k-th in the chain
        return {"ranks equal the true order": bool(np.all(rank == true_rank) or np.all(rank == N + 1 - true_rank))}
    return {}


# ---- exploratory (after C3 could not be read off T4's incomplete table) --------------------------------------------
def partial_order(full, pi):
    """The order the semilattice completion determines, a <= b iff a + b = b, read from the known entries only,
    against the true chain order: pairs ordered correctly, wrongly, and left undetermined."""
    pos = np.empty(N, int)
    pos[pi] = np.arange(N)
    a, b = np.triu_indices(N, 1)
    known = full[a, b] >= 0
    hi = np.where(pos[a] > pos[b], a, b)
    return {"pairs": len(a), "ordered correctly": int(np.sum(known & (full[a, b] == hi))),
            "ordered wrongly": int(np.sum(known & (full[a, b] != hi))), "undetermined": int(np.sum(~known))}


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    out = {}
    truth = {"T1 cyclic": "abelian group", "T2 Z6 x Z10": "abelian group", "T3 A5": "group",
             "T4 chain max": "semilattice", "T5 subtraction": "quasigroup", "T6 random": "none"}
    for frac in (0.5, 0.25):
        print(f"\nobserved fraction {frac}")
        for name, (T, pi) in tasks(np.random.default_rng(7)).items():
            mask = rng.random((N, N)) < frac
            T_obs = np.where(mask, T, -1)
            scores = {h: score(T_obs, ax, np.random.default_rng(3)) for h, ax in HYPOTHESES.items()}
            top = max(scores.values())
            tied = [h for h, s in scores.items() if s >= top - 1e-6]
            win = max(tied, key=lambda h: len(HYPOTHESES[h]))
            full, contra = complete(T_obs, HYPOTHESES[win])
            coords = coordinates(win, full, pi, T)
            acc = float(np.mean(full[~mask] == T[~mask])) if (full[~mask] >= 0).all() else None
            det = (full >= 0) & ~mask
            det_acc = float(np.mean(full[det] == T[det])) if det.any() else None
            if win == "semilattice":
                coords["exploratory: partial order"] = partial_order(full, pi)
            out[f"{name} @ {frac}"] = dict(winner=win, right=win == truth[name], scores=scores, coords=coords,
                                           unseen_accuracy=acc,
                                           unseen_determined=int(det.sum()), unseen_total=int((~mask).sum()),
                                           determined_accuracy=det_acc)
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            print(f"   {name:16s} winner {win:22s} {'right' if win == truth[name] else 'WRONG (truth: ' + truth[name] + ')':28s} "
                  f"unseen entries determined {int(det.sum())}/{int((~mask).sum())}, of those correct "
                  f"{det_acc if det_acc is None else round(det_acc, 4)}   {coords}")
            print(f"      scores: " + ", ".join(f"{h} {s:.0f}" for h, s in ranked[:4]))
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/structure_hypotheses.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
