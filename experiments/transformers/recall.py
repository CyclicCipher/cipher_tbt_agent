"""CONTEXT-CUED MEMORY SAMPLING: retrieve candidate programs from solved cases instead of constructing them.

THE PRINCIPLE (`reference_hypothesis_generation`, Dasgupta & Gershman): hypotheses are not enumerated and not derived —
they come from a SMALL, context-cued sample drawn from memory, and are then evaluated. The sample is stochastic and
biased by similarity to past experience, which is what makes it a sampler rather than a lookup.

WHY THIS IS STRUCTURALLY DIFFERENT FROM THE OTHER TWO, and the reason it is worth trying after both of them failed in
their own ways. `propose.py` built the program step by step from the difference and was at chance on the first step;
`localise.py` searched with a meet criterion and only halved the exponent. Both had to identify a step in isolation.
**Memory sampling proposes WHOLE PROGRAMS**, so it never has to solve the step-1-in-isolation problem that is provably
at chance here. That is a real structural advantage — if similar-looking tasks have similar programs.

THE CRUX, AND IT IS A DIAGNOSTIC RATHER THAN AN ASSUMPTION. The held-out task's program is by construction NOT in memory,
so pure retrieval must fail; the mechanism has to propose retrieved programs PLUS their edit-neighbourhoods ("this looks
like a task whose program was (a,b,c) — try programs near (a,b,c)"). That works only if **behavioural similarity tracks
PROGRAM distance**. In a group there is no reason it should: two elements one generator apart can act very differently on
a probe, and two that act similarly can be far apart in the Cayley graph. So two diagnostics run BEFORE the mechanism,
either of which can kill it:

  1. REACHABILITY — the distribution of minimum program-edit distance from each held-out program to any program in
     memory. If held-out programs are mostly >r edits away, an r-edit neighbourhood cannot reach them however good
     retrieval is, and nothing else matters.
  2. CUE VALIDITY — the rank correlation between behavioural similarity and program-edit distance across the universe.
     This is what retrieval is betting on. At zero correlation, retrieval is random sampling with extra steps.

(Spearman here uses AVERAGE RANKS. Without tie handling this project has already once reported rho = +1.000 on
information-free data — see `feedback_verify_the_apparatus_not_the_number` — and program-edit distances are heavily tied
by construction, so untied ranks would be actively misleading rather than merely imprecise.)

THE CUE IS COMPUTED FROM DEMONSTRATIONS ONLY, which keeps this honest. For a query we have `n_demos` (input, output)
pairs. For each program in memory we can EXECUTE it on those same inputs — we know its program, and execution is the
thing this model does well (0.997). Similarity is then agreement between the memory program's outputs and the observed
outputs on identical inputs. Nothing about the query's decomposition is used, and the memory holds only tasks whose
programs were already known.

Sampling is softmax over similarity at temperature `--temp` (Dasgupta/Gershman's stochastic sample); `--temp 0` is the
top-k limit, reported alongside so the contribution of stochasticity is visible rather than assumed.

Usage:  python experiments/transformers/recall.py
"""
from __future__ import annotations

import argparse
import itertools

import torch

from arity import build_splits
from chaining import apply_task
from diversity import NAMES
from h1_lid import L, V

MAXM = 3
PRIMS = "rot1,reverse,swap_pairs,add1,negate"


def edit_distance(a, b):
    """Levenshtein over primitive sequences — the program-space metric the edit-neighbourhood moves in."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[len(a)][len(b)]


def neighbourhood(prog, prims, radius):
    """Every program within `radius` edits, capped to length 1..MAXM. radius=0 is the program itself."""
    out = {tuple(prog)}
    for _ in range(radius):
        cur = set(out)
        for p in cur:
            for i in range(len(p)):
                for q in prims:
                    out.add(p[:i] + (q,) + p[i + 1:])                        # substitute
                if len(p) > 1:
                    out.add(p[:i] + p[i + 1:])                               # delete
            for i in range(len(p) + 1):
                if len(p) < MAXM:
                    for q in prims:
                        out.add(p[:i] + (q,) + p[i:])                        # insert
    return [p for p in out if 1 <= len(p) <= MAXM]


def spearman(a, b):
    """Rank correlation with AVERAGE ranks for ties. Edit distances take few distinct values, so ties dominate and
    untied ranks would manufacture structure that is not there."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def cue_similarity(x, y, mem, dev):
    """Agreement between each memory program's outputs on the SAME inputs and the observed outputs. Computed from
    demonstrations alone — the query's decomposition is never touched."""
    return torch.stack([(apply_task(x, p) == y).float().mean() for p in mem])


def run(task, mem, prims, dev, k, radius, temp, n_demos=8, seed=0, randomise=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randint(0, V, (n_demos, L), generator=g, device=dev)
    y = apply_task(x, task)
    probe = torch.randint(0, V, (24, L), generator=g, device=dev)
    want = tuple(apply_task(probe, task).flatten().tolist())

    if randomise:
        pick = torch.randperm(len(mem), generator=g, device=dev)[:k].tolist()
    else:
        sim = cue_similarity(x, y, mem, dev)
        if temp <= 0:
            pick = torch.topk(sim, min(k, len(mem))).indices.tolist()
        else:
            pick = torch.multinomial((sim / temp).softmax(0), min(k, len(mem)),
                                     replacement=False, generator=g).tolist()

    cands, seen = [], set()
    for i in pick:
        for c in neighbourhood(mem[i], prims, radius):
            if c not in seen:
                seen.add(c)
                cands.append(c)
    for c in cands:
        if torch.equal(apply_task(x, c), y) and tuple(apply_task(probe, c).flatten().tolist()) == want:
            return True, len(cands)
    return False, len(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=0.02, help="softmax temperature; 0 = top-k")
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    prims = [NAMES.index(nm) for nm in PRIMS.split(",")]
    sup, held = build_splits(args.seed, prims, max_len=MAXM)
    mem = sup[1] + sup[2] + sup[3]                                   # memory = the tasks whose programs are known
    print(f"device {dev} | memory holds {len(mem)} solved programs | held-out {len(held[2])} pairs + "
          f"{len(held[3])} triples\n")

    # DIAGNOSTIC 1 -- REACHABILITY. Can an r-edit neighbourhood of memory even contain the held-out programs?
    print("(1) REACHABILITY: minimum program-edit distance from each held-out program to ANY program in memory")
    for pn, tasks in (("pairs", held[2]), ("triples", held[3])):
        ds = [min(edit_distance(t, m) for m in mem) for t in tasks]
        hist = {d: ds.count(d) for d in sorted(set(ds))}
        print(f"    {pn:<8} {hist}   (an r-edit neighbourhood reaches only distances <= r)")
    # AND HOW MUCH OF THE WHOLE SPACE those neighbourhoods cover, which decides whether "reachable" is structure or an
    # artifact of a small universe. If memory's r-edit neighbourhoods already cover nearly every program, then cued
    # retrieval is a REORDERING of enumeration rather than a reduction of it, and will not survive a larger space.
    n_all = sum(len(prims) ** m for m in range(1, MAXM + 1))
    for radius in (1, 2):
        cov = set()
        for m in mem:
            cov.update(neighbourhood(m, prims, radius))
        print(f"    radius {radius}: memory's neighbourhoods cover {len(cov)}/{n_all} = {len(cov) / n_all:.1%} "
              f"of all programs")

    # DIAGNOSTIC 2 -- CUE VALIDITY. Does behavioural similarity track program distance at all?
    g = torch.Generator(device=dev).manual_seed(3)
    x = torch.randint(0, V, (args.demos, L), generator=g, device=dev)
    allt = mem + held[2] + held[3]
    sims, dists = [], []
    for a, b in itertools.combinations(range(len(allt)), 2):
        sims.append(float((apply_task(x, allt[a]) == apply_task(x, allt[b])).float().mean()))
        dists.append(edit_distance(allt[a], allt[b]))
    rho = spearman(sims, dists)
    print(f"\n(2) CUE VALIDITY: Spearman(behavioural similarity, program-edit distance) = {rho:+.3f} "
          f"over {len(sims)} task pairs")
    print("    Retrieval bets on this being strongly NEGATIVE (similar behaviour => nearby program).")

    print(f"\n(3) MECHANISM   temp={args.temp} (softmax sample) and temp=0 (top-k), against random retrieval")
    print(f"    {'retrieval':<16}{'k':>3}{'radius':>7}{'cands':>7}{'HELD pairs':>12}{'HELD triples':>14}")
    for label, temp, rnd in (("cued (sampled)", args.temp, False), ("cued (top-k)", 0.0, False),
                             ("random", 0.0, True)):
        for k in (1, 3, 5):
            for radius in (1, 2):
                cells, cands = [], []
                for tasks in (held[2], held[3]):
                    rs = [run(t, mem, prims, dev, k, radius, temp, args.demos, args.seed, rnd) for t in tasks]
                    cells.append(sum(r[0] for r in rs) / len(rs))
                    cands.append(sum(r[1] for r in rs) / len(rs))
                print(f"    {label:<16}{k:>3}{radius:>7}{sum(cands) / 2:>7.0f}"
                      f"{cells[0]:>12.3f}{cells[1]:>14.3f}")

    print("\nReference points on held-out triples: enumeration 1.000 at ~70 expansions; meet-in-the-middle 1.000 at")
    print("29.3; difference-scoring 0.389 at 13. A retrieval method has to beat those on the accuracy/cost trade-off.")
    print("If diagnostic (1) shows held-out programs are mostly further than the radius, or (2) is near zero, the")
    print("mechanism is dead for a structural reason and the table below it merely confirms it.")


if __name__ == "__main__":
    main()
