"""Experiment H5: a neural substrate for the store, a Willshaw autoassociative network (CA3 after Marr), against the
table (EXPERIMENT_H5.md).

Every token has a fixed random sparse binary code (D bits, k active). An episode (a, r, b) is the union of its three
codes. STORE: clipped Hebbian learning, W[i, j] = 1 if bits i and j were ever co-active in a stored episode (symmetric:
the network stores the joint pattern, so it can complete from any part). RECALL from a cue (the question's tokens plus
the learned expansion, as the table's query): dendritic sums s = W . cue; output the bits whose sum reaches the cue's
size (the Willshaw threshold), keeping the cue. DECODE: entities ranked by overlap with the recalled pattern / k,
excluding the cue's entity. Two scores: top-1 (the decoded answer) and @12 (the answer among the 12 best, the analogue
of the table's 12-episode read set). The union code carries no roles, so a cue {x, r} also completes episodes where x
has the other role; top-1 pays for that, @12 mostly does not.

The table (H1c with expansion) retrieves the answer episode at rank 1 with recall 1.00 at any size; this measures what
the network loses as it fills, for codes of different sparsity.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from h1 import N_ENT, N_REL, TEST_WORLD, conv, make_facts, rel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).parent
OUT = HERE / "runs" / "h5"
D = 2048
VOCAB = N_ENT + 2 * N_REL


def codes(k, seed=12345):
    g = torch.Generator().manual_seed(seed)
    C = torch.zeros(VOCAB, D)
    for v in range(VOCAB):
        C[v, torch.randperm(D, generator=g)[:k]] = 1.0
    return C.to(DEV)


def facts_union(n_worlds):
    facts = set()
    for w in range(100, 100 + n_worlds):
        facts |= set(make_facts(seed=w).facts)
    facts = sorted(facts)
    random.Random(0).shuffle(facts)
    return facts


@torch.no_grad()
def run(k, sizes, n_probe=1000):
    t0 = time.time()
    C = codes(k)
    facts = facts_union(4)                                           # up to ~11,930 episodes
    fwd, rev = {}, {}
    W = torch.zeros(D, D, device=DEV, dtype=torch.bool)
    stored, out = 0, []
    rng = random.Random(1)
    for n in sizes:
        for a, r, b in facts[stored:n]:                              # clipped Hebbian write, one episode at a time
            fwd.setdefault((a, r), set()).add(b)
            rev.setdefault((r, b), set()).add(a)
            x = (C[a] + C[rel(r)] + C[b]).clamp(max=1).bool()
            W |= x[:, None] & x[None, :]
        stored = n
        probe = rng.sample(facts[:n], min(n_probe, n))
        res = {}
        for kind in ("fwd", "rev"):
            if kind == "fwd":
                cue_toks = [(a, rel(r)) for a, r, b in probe]
                correct = [fwd[(a, r)] for a, r, b in probe]
            else:
                # the converse token r° never occurs in a stored episode; the learned expansion maps it to r, so the
                # cue is the stored tokens the question points at: b and r (the Willshaw threshold is the cue's size)
                cue_toks = [(b, rel(r)) for a, r, b in probe]
                correct = [rev[(r, b)] for a, r, b in probe]
            Wf = W.float()
            cue = torch.stack([C[list(t)].amax(0) for t in cue_toks])            # (n, D)
            s = cue @ Wf                                                         # dendritic sums (W symmetric)
            recalled = ((s >= cue.sum(1, keepdim=True) - 0.5) | cue.bool()).float()
            score = recalled @ C[:N_ENT].T / k                                   # (n, entities)
            for i, toks in enumerate(cue_toks):
                for t in toks:
                    if t < N_ENT:
                        score[i, t] = -1
            top = score.topk(12, dim=1).indices.tolist()
            res[kind] = round(sum(t[0] in ok for t, ok in zip(top, correct)) / len(probe), 4)
            res[kind + "@12"] = round(sum(any(x in ok for x in t) for t, ok in zip(top, correct)) / len(probe), 4)
        out.append(dict(stored=n, density=round(W.float().mean().item(), 4), top1_fwd=res["fwd"],
                        top1_rev=res["rev"], at12_fwd=res["fwd@12"], at12_rev=res["rev@12"]))
        print(f"k={k} stored={n} density={out[-1]['density']} top1 fwd={res['fwd']} rev={res['rev']}  "
              f"@12 fwd={res['fwd@12']} rev={res['rev@12']}", flush=True)
    return dict(k=k, D=D, sizes=sizes, curve=out, secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    a = ap.parse_args()
    p = OUT / f"willshaw_k{a.k}.json"
    if p.exists():
        print(f"{p} exists: not re-run")
        raise SystemExit
    r = run(a.k, [250, 500, 1000, 2000, 4000, 8000, 11930])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
