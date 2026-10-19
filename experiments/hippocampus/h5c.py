"""Experiment H5c: a random conjunctive (dentate-gyrus-like) layer for the neural store (EXPERIMENT_H5C.md).

H5b showed that a pairwise Hebbian matrix cannot store a conjunction (a AND r -> b). Here the store gets conjunctive
units: every pair of active bits drawn from two different slots of an episode, (bit i of slot x, bit j of slot y),
activates one hashed "pair unit" out of N_PAIR (a random, fixed expansion, like the dentate gyrus). For each slot pair
(x, y) a clipped Hebbian map W_xy[unit] -> bits of the third slot is learned in one shot. Recall from a cue of two
slots: the cue's k*k pair units each vote for third-slot bits; output the bits all of them vote for (the Willshaw
threshold); decode the entity by overlap / k. The cue's two slots come from the learned slot map (H5b).
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from h1 import N_ENT, rel
from h5 import D, codes, facts_union, learn_slots

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent / "runs" / "h5"
N_PAIR = 65536
PAIRS = {(0, 1): 2, (1, 2): 0, (0, 2): 1}                            # cue slots -> the slot they recall


def bits(C, toks):
    return C[toks].nonzero()[:, 1].view(len(toks), -1)               # (n, k) active bit indices per token


def pair_units(bx, by, x, y):
    """(n, k) and (n, k) bit indices of slots x and y -> (n, k*k) hashed unit indices, distinct per slot pair."""
    i = bx[:, :, None].long()
    j = by[:, None, :].long()
    h = (i * 1000003 + j * 999983 + (x * 3 + y) * 7919) * 2654435761
    return (h % N_PAIR).reshape(len(bx), -1)


@torch.no_grad()
def run(k, sizes, n_probe=1000):
    t0 = time.time()
    C = codes(k)
    ent_slot, ans_slot = learn_slots()
    facts = facts_union(4)
    W = {xy: torch.zeros(N_PAIR, D, device=DEV, dtype=torch.bool) for xy in PAIRS}
    fwd, rev, stored, out = {}, {}, 0, []
    rng = random.Random(1)
    for n in sizes:
        new = facts[stored:n]
        for a, r, b in new:
            fwd.setdefault((a, r), set()).add(b)
            rev.setdefault((r, b), set()).add(a)
        if new:
            toks = torch.tensor([(a, rel(r), b) for a, r, b in new], device=DEV)
            B = [bits(C, toks[:, s]) for s in range(3)]               # per slot (m, k)
            for (x, y), z in PAIRS.items():
                U = pair_units(B[x], B[y], x, y)                      # (m, k*k)
                target = C[toks[:, z]].bool()                         # (m, D)
                for u, t in zip(U, target):                           # one-shot clipped Hebbian write
                    W[(x, y)][u] |= t
        stored = n
        probe = rng.sample(facts[:n], min(n_probe, n))
        res = {}
        for kind in ("fwd", "rev"):
            if kind == "fwd":
                q = [(a, rel(r), rel(r)) for a, r, b in probe]          # (entity, question type, relation)
                correct = [fwd[(a, r)] for a, r, b in probe]
            else:
                q = [(b, rel(r) + 6, rel(r)) for a, r, b in probe]      # converse token = relation + N_REL
                correct = [rev[(r, b)] for a, r, b in probe]
            es = ent_slot[q[0][1]]                                     # the entity's learned slot (same per kind)
            ans = ans_slot[q[0][1]]
            x, y = sorted((es, 1))
            ent = torch.tensor([e for e, _, _ in q], device=DEV)
            relt = torch.tensor([t for _, _, t in q], device=DEV)
            bx = bits(C, ent if x == es else relt)
            by = bits(C, relt if y == 1 else ent)
            U = pair_units(bx, by, x, y)                              # (n, k*k)
            votes = W[(x, y)][U].float().sum(1)                       # (n, D)
            recalled = (votes >= U.shape[1] - 0.5).float()
            score = recalled @ C[:N_ENT].T / k
            score[torch.arange(len(q)), ent] = -1
            top = score.topk(12, dim=1).indices.tolist()
            res[kind] = round(sum(t[0] in ok for t, ok in zip(top, correct)) / len(q), 4)
            res[kind + "@12"] = round(sum(any(v in ok for v in t) for t, ok in zip(top, correct)) / len(q), 4)
        dens = round(sum(w.float().mean().item() for w in W.values()) / 3, 4)
        out.append(dict(stored=n, density=dens, top1_fwd=res["fwd"], top1_rev=res["rev"], at12_fwd=res["fwd@12"],
                        at12_rev=res["rev@12"]))
        print(f"k={k} stored={n} density={dens} top1 fwd={res['fwd']} rev={res['rev']}  "
              f"@12 fwd={res['fwd@12']} rev={res['rev@12']}", flush=True)
    return dict(k=k, D=D, n_pair=N_PAIR, memory_bytes=3 * N_PAIR * D // 8, sizes=sizes, curve=out,
                secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    a = ap.parse_args()
    p = OUT / f"conjunctive_k{a.k}.json"
    if p.exists():
        print(f"{p} exists: not re-run")
        raise SystemExit
    r = run(a.k, [250, 500, 1000, 2000, 4000, 8000, 11930])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
