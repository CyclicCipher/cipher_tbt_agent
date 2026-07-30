"""THE PROPOSAL MECHANISM: GCML difference-scoring instead of enumeration.

WHERE THIS SITS. `search.py` solved held-out composition (0.048 -> 0.778) by hypothesise-and-test, but it ENUMERATED all
155 candidate programs, and the space is `|prims|^depth`. `decompose.py` then showed the alternative — learning an
amortised identifier — fails at every coverage level tested (held-out 0.000 at 10, 40 and 134 programs). So the open
problem is the PROPOSAL DISTRIBUTION: sample a handful of candidate decompositions rather than enumerate them.

THE MECHANISM, from Lin/Yang/Zhao/Pezzulo/Maass, Nat Mach Intell 2026 (notes in
`src/tbt/notes/gcml_neural_sampling_cognitive_maps.md`). A learned INVERSE model `W` maps a state DIFFERENCE to the action
that reduces it — `u = W(s* - s)` (eq 15) — trained by one local Hebbian rule from single-step transitions,
`dW = eta * a_t (s_{t+1} - s_t)^T` (eq 14), with a saturating binary variant (eq 21) under which `W` becomes an OVERLAP
DETECTOR between a difference and each building block. Selection is winner-take-all over `u` plus noise (eq 19), the
noise being what makes it a SAMPLER rather than a deterministic policy.

WHY THIS IS NOT A RE-RUN OF `decompose.py`, which is the point of trying it. That file trained a classifier on
(demonstrations of a composition -> program) pairs and memorised them. **`W` here is trained ONLY on single-primitive
transitions** — never on a composition, never on a program label — and the proposal for a composite difference falls out of
the same linear map. It is a proposal mechanism, not a classifier, so it is not obviously subject to what failed there.

THE STRUCTURAL WORRY, stated before running because it decides how to read the result. GCML's overlap detector works
because in their task composition is ADDITIVE in the difference space: adding building blocks to an image makes the total
difference the union of the parts, so "which BB is contained in the remaining difference" is well posed. Our composition
is FUNCTIONAL composition of group elements, and the difference of `phi_c . phi_b . phi_a` is the signature of the
COMPOSED map, not the union of the three parts. So there is a real possibility that difference-scoring cannot see the
parts at all. That is the hypothesis under test, not a caveat.

A REPRESENTATION PREDICTION, AND ITS CORRECTION BY THE DIAGNOSTIC BELOW. From `reference_l5_operator_kinds` (an operator
is a position-invariant DELTA in whatever dimension the action changes) I predicted that PERMUTATIONS specifically would
be invisible in a one-hot (position, value) difference, since a permutation leaves those marginals untouched, while value
maps would survive. **Measured, the effect is more general and the prediction was too narrow:** the mean one-hot
difference is ~0.04–0.07 (i.e. sampling noise) for EVERY primitive including `add1` and `negate`, because *every*
primitive here is a bijection and a bijection preserves the uniform marginal. So a Hebbian `W` over state-difference
features has no signal at all to learn from, for any of them — which is exactly why the `onehot` arm sits flat at 0.056.
The relational signature is 2.7–3.5 for all five. Hence two representations:

    onehot      Psi(s)[i,v] = 1 if s[i] == v.  Difference D = Psi(y) - Psi(x). W row = E_x[Psi(phi_p x) - Psi(x)].
                PREDICTED: works for value maps, blind to permutations.
    relational  Phi(x,y)[i,j,d] = 1 if y[j] == (x[i]+d) mod V -- the state-INDEPENDENT transformation signature. For a
                primitive that permutes by pi and shifts by delta it has mass exactly at (i, pi(i), delta).
                PREDICTED: identifies a single primitive well; whether it exposes the PARTS of a composite is the open
                question above.

BASELINES, both necessary. `greedy-forward` scores each primitive by how much APPLYING it reduces Hamming distance to the
goal — the same idea as `W` but computed with the forward model instead of a learned linear map, so it separates "the idea
does not work" from "the linear approximation does not work". `random` is the chance floor.

WHAT IS MEASURED. Beam search over programs, extending a beam of width B by every primitive at each depth and keeping the
best B by proposal score, with exact verification of every beam entry against all demonstrations. Reported: whether the
beam contains a program FUNCTIONALLY equal to the task (writing-agnostic, since several writings can be equivalent), and
the number of candidates evaluated -- `B * |prims| * depth`, LINEAR in depth, against enumeration's `|prims|^depth`. That
scaling difference is the entire point; the toy numbers are close only because the toy is small.

Verification uses the TRUE primitives, deliberately: the question here is PROPOSAL quality, and `search.py` already
established that a learned executor can stand in for verification (its answer accuracy equalled its found-function rate to
three decimals, at 0.86 fidelity). So multiply these numbers by ~0.86 to predict the end-to-end figure.

Usage:  python experiments/transformers/propose.py
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


def onehot(s):
    """Psi(s)[i, v] = 1 if s[i] == v, flattened. `s` is (..., L)."""
    return torch.zeros(*s.shape[:-1], L, V, device=s.device).scatter_(
        -1, s.unsqueeze(-1), 1.0).reshape(*s.shape[:-1], L * V)


def relational(x, y):
    """Phi(x,y)[i, j, d] = 1 if y[j] == (x[i] + d) mod V. The state-INDEPENDENT signature of the transformation carrying
    x to y: for a primitive permuting by pi and shifting by delta, mass sits at (i, pi(i), delta) whatever x is."""
    d = (y.unsqueeze(-2) - x.unsqueeze(-1)) % V                      # (..., L_src, L_tgt)
    return torch.zeros(*d.shape, V, device=x.device).scatter_(
        -1, d.unsqueeze(-1), 1.0).reshape(*x.shape[:-1], L * L * V)


def learn_W(prims, kind, dev, n=4096, sat=False, seed=0):
    """The Hebbian inverse model, trained on SINGLE-PRIMITIVE transitions ONLY -- no composition ever, no program label
    ever. `sat` is GCML eq 21's saturating binary variant, under which W becomes an overlap detector."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rows = []
    for p in prims:
        x = torch.randint(0, V, (n, L), generator=g, device=dev)
        y = apply_task(x, (p,))
        f = (onehot(y) - onehot(x)) if kind == "onehot" else relational(x, y)
        w = f.mean(0)
        rows.append(torch.clamp(w * n, max=1.0) if sat else w)
    W = torch.stack(rows)
    return W / W.norm(dim=1, keepdim=True).clamp(min=1e-9)           # cosine scoring, so signature sparsity does not
    #                                                                  decide the ranking by magnitude alone


def score(W, kind, cur, goal, prims, dev):
    """Proposal scores over primitives for the difference (cur -> goal). `cur`/`goal` are (n_demos, L)."""
    f = (onehot(goal) - onehot(cur)) if kind == "onehot" else relational(cur, goal)
    return (f @ W.T).mean(0)                                          # aggregate the evidence across demonstrations


def score_forward(cur, goal, prims):
    """`greedy-forward`: how much does APPLYING each primitive reduce Hamming distance to the goal? The same idea as W,
    computed with the forward model rather than a learned linear map."""
    base = (cur != goal).float().sum(-1).mean()
    out = []
    for p in prims:
        out.append(base - (apply_task(cur, (p,)) != goal).float().sum(-1).mean())
    return torch.stack(out)


def beam_search(task, prims, W, kind, beam, dev, n_demos=8, noise=0.0, seed=0):
    """Extend a beam of partial programs by every primitive, keep the best `beam` by proposal score, verify every entry
    against all demonstrations. Returns (found the function, candidates evaluated, true-first-primitive in the level-1
    shortlist)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randint(0, V, (n_demos, L), generator=g, device=dev)
    goal = apply_task(x, task)
    probe = torch.randint(0, V, (24, L), generator=g, device=dev)
    want = tuple(apply_task(probe, task).flatten().tolist())

    live = [((), x)]                                                  # (program so far, current state of every demo)
    evaluated, found, first_ok = 0, False, None
    for depth in range(MAXM):
        scored = []
        for prog, cur in live:
            s = (score(W, kind, cur, goal, prims, dev) if W is not None
                 else score_forward(cur, goal, prims))
            if noise:
                s = s + noise * torch.randn(len(prims), generator=g, device=dev)
            for k, p in enumerate(prims):
                scored.append((float(s[k]), prog + (p,), apply_task(cur, (p,))))
        scored.sort(key=lambda r: -r[0])
        if depth == 0:
            top = [r[1][-1] for r in scored[:beam]]
            first_ok = task[0] in top
        live = []
        for _sc, prog, cur in scored[:beam]:
            evaluated += 1
            if torch.equal(cur, goal) and tuple(apply_task(probe, prog).flatten().tolist()) == want:
                found = True
            live.append((prog, cur))
        if found:
            break
    return found, evaluated, first_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", default="1,2,3,5")
    ap.add_argument("--noise", type=float, default=0.0, help="GCML's epsilon: makes WTA a sampler (eq 19)")
    ap.add_argument("--sat", type=int, default=0, help="1 = GCML eq 21's saturating binary W (overlap detector)")
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    prims = [NAMES.index(nm) for nm in PRIMS.split(",")]
    sup, held = build_splits(args.seed, prims, max_len=MAXM)
    n_enum = sum(len(prims) ** m for m in range(1, MAXM + 1))
    print(f"device {dev} | {len(prims)} primitives {PRIMS} | held-out {len(held[2])} pairs + {len(held[3])} triples")
    print(f"W is trained on SINGLE-PRIMITIVE transitions only -- never a composition, never a program label."
          f"{' Saturating (eq 21).' if args.sat else ''}")
    print(f"Enumeration evaluates {n_enum} candidates and reaches 0.778 on held-out triples (search.py, learned executor).\n")

    # DIAGNOSTIC for the representation prediction, run BEFORE the row normalisation hides it: a permutation leaves the
    # (position, value) marginals untouched, so its mean one-hot difference should be ~0 and a Hebbian W on that
    # representation must be blind to it. The relational signature should be large for both kinds.
    print(f"{'primitive':<12}{'|W| onehot':>12}{'|W| relational':>16}   (predicted: permutations ~0 under onehot)")
    for p in prims:
        g0 = torch.Generator(device=dev).manual_seed(1)
        x = torch.randint(0, V, (4096, L), generator=g0, device=dev)
        y = apply_task(x, (p,))
        print(f"{NAMES[p]:<12}{(onehot(y) - onehot(x)).mean(0).norm():>12.4f}"
              f"{relational(x, y).mean(0).norm():>16.4f}")
    print()

    scorers = [("hebb-onehot", learn_W(prims, "onehot", dev, sat=bool(args.sat)), "onehot"),
               ("hebb-relational", learn_W(prims, "relational", dev, sat=bool(args.sat)), "relational"),
               ("greedy-forward", None, None)]
    pools = [("HELD-OUT pairs", held[2]), ("HELD-OUT triples", held[3]), ("supervised triples", sup[3])]

    print(f"{'scorer':<17}{'beam':>5}{'cands':>7}" + "".join(f"{p[0][:16]:>19}" for p in pools) + f"{'first@beam':>12}")
    for name, W, kind in scorers:
        for beam in [int(b) for b in args.beams.split(",")]:
            cells, firsts, cands = [], [], []
            for _pn, tasks in pools:
                rs = [beam_search(t, prims, W, kind, beam, dev, args.demos, args.noise, args.seed) for t in tasks]
                cells.append(sum(r[0] for r in rs) / len(rs))
                cands.append(sum(r[1] for r in rs) / len(rs))
                firsts += [r[2] for r in rs if r[2] is not None]
            print(f"{name:<17}{beam:>5}{sum(cands) / len(cands):>7.0f}"
                  + "".join(f"{c:>19.3f}" for c in cells)
                  + f"{sum(firsts) / len(firsts):>12.3f}")

    print(f"\n'cands' = candidates actually evaluated (beam x |prims| x depth, LINEAR in depth) against enumeration's")
    print(f"{n_enum} (= |prims|^depth summed). 'first@beam' = the task's canonical first primitive is in the level-1")
    print(f"shortlist; chance at beam 1 is {1 / len(prims):.3f}. Columns are 'found a program that IS the function',")
    print("verified exactly against all demonstrations, so equivalent writings count.")
    print("Multiply by ~0.86 (search.py's executor fidelity) to predict the end-to-end figure with a learned executor.")
    print("\nIF difference-scoring cannot rank the parts of a COMPOSED map, these sit at the random floor and the")
    print("GCML overlap detector is confirmed to depend on its task's composition being ADDITIVE in the difference space.")


if __name__ == "__main__":
    main()
