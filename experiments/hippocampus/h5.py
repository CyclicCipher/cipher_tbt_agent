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
from collections import Counter

from h1 import N_ENT, N_REL, TEST_WORLD, TRAIN_WORLDS, conv, make_facts, rel

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


def learn_slots():
    """H5b: from the training worlds, for each question type (its relation-like token), which slot of the answer
    episode holds the question's entity, and which slot holds the answer. Counted, not assumed."""
    ent_slot, ans_slot = {}, {}
    for w in TRAIN_WORLDS:
        for a, r, b in make_facts(seed=w).facts:
            ep = (a, rel(r), b)
            for (e, t), ans in (((a, rel(r)), b), ((b, conv(r)), a)):
                ent_slot.setdefault(t, Counter())[ep.index(e)] += 1
                ans_slot.setdefault(t, Counter())[ep.index(ans)] += 1
    return ({t: c.most_common(1)[0][0] for t, c in ent_slot.items()},
            {t: c.most_common(1)[0][0] for t, c in ans_slot.items()})


def facts_union(n_worlds):
    facts = set()
    for w in range(100, 100 + n_worlds):
        facts |= set(make_facts(seed=w).facts)
    facts = sorted(facts)
    random.Random(0).shuffle(facts)
    return facts


@torch.no_grad()
def run(k, sizes, n_probe=1000, slots=False):
    """slots=False: H5 (an episode is the union of its tokens' codes). slots=True: H5b (each token's code is permuted
    by a fixed random permutation per slot within the episode: subject, relation, object)."""
    t0 = time.time()
    C = codes(k)
    g = torch.Generator().manual_seed(777)
    perms = [torch.randperm(D, generator=g).to(DEV) for _ in range(3)] if slots else [torch.arange(D, device=DEV)] * 3
    S = [C[:, pm] for pm in perms]                                   # S[j][t]: token t's code bound to slot j
    ent_slot, ans_slot = learn_slots() if slots else (None, None)
    facts = facts_union(4)                                           # up to ~11,930 episodes
    fwd, rev = {}, {}
    W = torch.zeros(D, D, device=DEV, dtype=torch.bool)
    stored, out = 0, []
    rng = random.Random(1)
    for n in sizes:
        for a, r, b in facts[stored:n]:                              # clipped Hebbian write, one episode at a time
            fwd.setdefault((a, r), set()).add(b)
            rev.setdefault((r, b), set()).add(a)
            x = (S[0][a] + S[1][rel(r)] + S[2][b]).clamp(max=1).bool()
            W |= x[:, None] & x[None, :]
        stored = n
        probe = rng.sample(facts[:n], min(n_probe, n))
        res = {}
        for kind in ("fwd", "rev"):
            if kind == "fwd":
                cue_toks = [(a, rel(r)) for a, r, b in probe]
                correct = [fwd[(a, r)] for a, r, b in probe]
                qtype = [rel(r) for a, r, b in probe]
            else:
                # the converse token r° never occurs in a stored episode; the learned expansion maps it to r, so the
                # cue is the stored tokens the question points at: b and r (the Willshaw threshold is the cue's size)
                cue_toks = [(b, rel(r)) for a, r, b in probe]
                correct = [rev[(r, b)] for a, r, b in probe]
                qtype = [conv(r) for a, r, b in probe]
            Wf = W.float()
            if slots:                                                            # entity at its learned slot, relation at 1
                cue = torch.stack([torch.maximum(S[ent_slot[q]][e], S[1][t]) for (e, t), q in zip(cue_toks, qtype)])
            else:
                cue = torch.stack([C[list(t)].amax(0) for t in cue_toks])        # (n, D)
            s = cue @ Wf                                                         # dendritic sums (W symmetric)
            recalled = ((s >= cue.sum(1, keepdim=True) - 0.5) | cue.bool()).float()
            if slots:                                                            # decode at the learned answer slot
                score = torch.stack([recalled[i] @ S[ans_slot[q]][:N_ENT].T for i, q in enumerate(qtype)]) / k
            else:
                score = recalled @ C[:N_ENT].T / k                               # (n, entities)
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
    return dict(k=k, D=D, slots=slots, slot_map=(None if not slots else
                {"ent": {int(t): v for t, v in ent_slot.items()}, "ans": {int(t): v for t, v in ans_slot.items()}}),
                sizes=sizes, curve=out, secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--slots", action="store_true")
    a = ap.parse_args()
    p = OUT / f"willshaw{'_slots' if a.slots else ''}_k{a.k}.json"
    if p.exists():
        print(f"{p} exists: not re-run")
        raise SystemExit
    r = run(a.k, [250, 500, 1000, 2000, 4000, 8000, 11930], slots=a.slots)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(p, "w"), indent=1)
