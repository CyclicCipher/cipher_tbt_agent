"""The tier test, Part B -- tier 2: codes in which actions are MATRICES (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §17.6
pre-registration, results §17.7). Discrete actions, given as labels (as TEM gets them); no thoughts.

Worlds (graphs from `structured_task`): grid (8 × 8, 4 moves), heading (8 × 8 × 4, six egocentric actions), perm (S₅,
4 generators). Codes, all centred and whitened (EᵀE = N·I), m ∈ {2, 4, 8} (m = 2 added after a smoke run showed that surplus
dimensions alone break an additive code's decoding, as in §15):
  learned add   E[dest(i, a)] ≈ E[i] + b_a              GCML's additive form (eq 11), discrete actions
                (models fitted and codes learned on the transitions that move -- see `moves`)
  learned mat   E[dest(i, a)] ≈ M_a·E[i] + b_a          actions as (affine) matrices; same optimiser, best of 3 restarts
  constructed   the exact codes, with both models fitted by least squares: grid (x, y); heading (x, y, heading vector)
                -- the factored tier-1 code, itself a matrix code (forward adds the heading vector to the position, a
                turn rotates the heading vector); S₅ the arrangement itself (each generator permutes its entries)
Measured:
  path integration   500 random starts × 8 random actions: compose the model's predictions from the start's code and
                     decode the nearest node's code after each step; over all sequences, and over those that never hit
                     a wall (on the grid and heading world a move into a wall stays put, which no linear model can do)
  lookahead          one-step lookahead planning in the code: at each step take the action whose PREDICTED code is
                     nearest the goal's (from the current node's code); 300 problems at distance 4, horizon 8. Also with
                     the TRUE next codes (the code's metric alone, no model error) and with random actions
    python experiments/neural_turing_architecture/search_bench/matrix_code.py --out runs/tiers_partB.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from mock_task import TaskConfig  # noqa: E402
from structured_task import StructuredWalk  # noqa: E402

SHAPE = {"grid": (64, 4), "heading": (256, 6), "perm": (120, 4)}


def world(name, seed=0):
    N, k = SHAPE[name]
    return StructuredWalk(TaskConfig(N=N, k=k, d=64), seed, 0, graph=name, keys="global")


def whiten(E):
    E = E - E.mean(0)
    return torch.linalg.qr(E)[0] * math.sqrt(E.shape[0])


def moves(dest):
    """The transitions that change the state: a move into a wall stays put, which no linear model can represent, so
    models are fitted (and codes learned) on the rest; path integration is scored with and without walls."""
    return dest != torch.arange(dest.shape[0])[:, None]


def fit_model(E, dest, kind):
    """Least-squares (affine) model per action: next ≈ E·M_aᵀ + b_a (matrix) or E + b_a (additive)."""
    N, k = dest.shape
    ok = moves(dest)
    M, b = [], []
    for a in range(k):
        cur, nxt = E[ok[:, a]], E[dest[ok[:, a], a]]
        if kind == "add":
            M.append(torch.eye(E.shape[1]))
            b.append((nxt - cur).mean(0))
        else:
            A = torch.cat([cur, torch.ones(cur.shape[0], 1)], 1)
            sol = torch.linalg.lstsq(A, nxt).solution                                # (m + 1, m)
            M.append(sol[:-1].T)
            b.append(sol[-1])
    return torch.stack(M), torch.stack(b)


def predict(E, M, b, a):
    """E (n, m), actions a (n,) -> predicted next codes."""
    return torch.einsum("nij,nj->ni", M[a], E) + b[a]


def residual(E, dest, M, b):
    """Unexplained share of the code's change over the transitions that move."""
    ok = moves(dest).T[..., None].float()                                          # (k, N, 1)
    nxt = torch.stack([E[dest[:, a]] for a in range(dest.shape[1])])               # (k, N, m)
    pred = torch.stack([E @ M[a].T + b[a] for a in range(dest.shape[1])])
    return float((ok * (nxt - pred) ** 2).sum() / (ok * (nxt - E) ** 2).sum())


def learn(dest, m, kind, seed, steps=3000, lr=0.02, restarts=3):
    N, k = dest.shape
    ok = moves(dest).T[..., None].float()                                          # (k, N, 1)
    best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(1000 * seed + r)
        E = torch.randn(N, m, generator=g).requires_grad_(True)
        M = (torch.eye(m).repeat(k, 1, 1) + 0.1 * torch.randn(k, m, m, generator=g)).requires_grad_(kind == "mat")
        b = torch.zeros(k, m, requires_grad=True)
        opt = torch.optim.Adam([E, b] + ([M] if kind == "mat" else []), lr=lr)
        for _ in range(steps):
            W = whiten(E)
            Mk = M if kind == "mat" else torch.eye(m).expand(k, m, m)
            pred = torch.einsum("aij,nj->ani", Mk, W) + b[:, None]
            loss = (ok * (W[dest.T] - pred) ** 2).sum() / ok.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        W = whiten(E.detach())
        Mf, bf = fit_model(W, dest, kind)
        res = residual(W, dest, Mf, bf)
        if best is None or res < best[0]:
            best = (res, W, Mf, bf)
    return best


def constructed(name, t):
    if name == "grid":
        return t.xy
    if name == "heading":
        return t.codes["allo"] * torch.tensor([7.0, 7.0, 1.0, 1.0])                 # position in cells, unit heading
    perms = list(itertools.permutations(range(5)))
    return torch.tensor(perms, dtype=torch.float)[:, :4]                            # the 5th entry is affine in these


def path_integration(E, M, b, dest, gen, n=500, K=8):
    N, k = dest.shape
    true = torch.randint(N, (n,), generator=gen)
    acts = torch.randint(k, (n, K), generator=gen)
    pred, clean = E[true], torch.ones(n, dtype=torch.bool)
    acc, acc_clean = [], []
    for s in range(K):
        nxt = dest[true, acts[:, s]]
        clean &= nxt != true                                                        # never stayed put (a wall)
        pred = predict(pred, M, b, acts[:, s])
        hit = torch.cdist(pred, E).argmin(1) == nxt
        acc.append(float(hit.float().mean()))
        acc_clean.append(float(hit[clean].float().mean()) if bool(clean.any()) else float("nan"))
        true = nxt
    return acc, acc_clean, float(clean.float().mean())


def lookahead(E, M, b, dest, dist, gen, n=300, L=4, T=8):
    N, k = dest.shape
    pairs = (dist == L).nonzero()
    pairs = pairs[torch.randperm(pairs.shape[0], generator=gen)[:n]]
    out = {"model": 0, "true_next": 0, "random": 0}
    for s, tg in pairs.tolist():
        for mode in out:
            cur = s
            for _ in range(T):
                if mode == "model":
                    a = int(((predict(E[cur].expand(k, -1), M, b, torch.arange(k)) - E[tg]) ** 2).sum(1).argmin())
                elif mode == "true_next":
                    a = int(((E[dest[cur]] - E[tg]) ** 2).sum(1).argmin())
                else:
                    a = int(torch.randint(k, (1,), generator=gen))
                cur = int(dest[cur, a])
                if cur == tg:
                    out[mode] += 1
                    break
    return {m: v / pairs.shape[0] for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["grid", "heading", "perm"])
    ap.add_argument("--m", type=int, nargs="+", default=[2, 4, 8])          # m = 2: added after the smoke run (§17.7)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--kinds", nargs="+", default=["add", "mat"])
    ap.add_argument("--no_constructed", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    rows, t0 = [], time.time()
    for name in a.worlds:
        t = world(name)
        dest, dist = t.dest, t.dist_all
        cands = [] if a.no_constructed else [("constructed", None, s) for s in range(a.seeds)]
        cands += [("learned", m, s) for m in a.m for s in range(a.seeds)]
        for src, m, seed in cands:
            for kind in a.kinds:
                if src == "constructed":
                    E = whiten(constructed(name, t))
                    M, b = fit_model(E, dest, kind)
                    res = residual(E, dest, M, b)
                else:
                    res, E, M, b = learn(dest, m, kind, seed, steps=a.steps, restarts=a.restarts)
                gen = torch.Generator().manual_seed(seed)
                acc, acc_clean, frac_clean = path_integration(E, M, b, dest, gen)
                look = lookahead(E, M, b, dest, dist, gen)
                fit = {}
                if name == "heading":                                       # is a learned code the factored one?
                    A = torch.cat([E, torch.ones(E.shape[0], 1)], 1)
                    for part, Y in (("position", t.xy), ("heading", t.codes["allo"][:, 2:])):
                        r = Y - A @ torch.linalg.lstsq(A, Y).solution
                        fit[part] = float(1 - (r ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
                rows.append(dict(world=name, code=src, m=E.shape[1], model=kind, seed=seed, residual=res, pi=acc,
                                 pi_clean=acc_clean, frac_clean=frac_clean, lookahead=look, r2=fit))
                print(f"{name:8s} {src:11s} m={E.shape[1]} {kind} seed {seed}: unexplained {res:.3f}  "
                      f"PI k=1/4/8 {acc[0]:.2f}/{acc[3]:.2f}/{acc[7]:.2f}  clean {acc_clean[3]:.2f}/{acc_clean[7]:.2f}  "
                      f"lookahead {look['model']:.2f} (true-next {look['true_next']:.2f}, random {look['random']:.2f})",
                      flush=True)
    print(f"{time.time() - t0:.0f}s")
    if a.out:
        p = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
        json.dump(dict(args=vars(a), rows=rows), open(p, "w"), indent=0)
        print(f"-> {p}")


if __name__ == "__main__":
    main()
