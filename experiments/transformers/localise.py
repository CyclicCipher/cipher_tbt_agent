"""PREDICTION-ERROR LOCALISATION: find the decomposition point where two predictions MEET.

THE PRINCIPLE (`reference_tbt_segmentation_and_grouping`): an object is a RECOGNITION construct and its boundaries come
from prediction MISMATCH. Applied to decomposition, the "boundary" between the parts of a composition is the point where
prediction from one side stops agreeing with prediction from the other.

WHAT THE PREVIOUS TWO ATTEMPTS ESTABLISHED, because this design is built out of their failures rather than around them.
`propose.py` measured that difference-scoring gives **chance-level** first-step identification (0.136/0.136/0.182 against
0.200) for a Hebbian W on two representations AND for a forward-model baseline — so it is the idea of greedy
difference-guided selection that fails, not its approximation. But the same file also measured that the mechanism is an
**excellent LAST-step recogniser**: held-out pairs reach 1.000 once the state is one step from the goal, because the
remaining difference genuinely IS a single primitive's signature. So error is informative NEAR a boundary and
uninformative far from one. That is exactly the asymmetry a bidirectional method exploits.

THE MECHANISM. Grow a frontier FORWARD from the input and a frontier BACKWARD from the observed output, the latter using
primitive INVERSES (every primitive here is a bijection, so they exist and are exact). The decomposition point is where a
forward state and a backward state COINCIDE — zero prediction error at the join, evaluated across every demonstration at
once so a coincidence on one demonstration cannot fake a meet. Iterative deepening over total depth handles the unknown
arity. Cost is `|prims|^ceil(d/2) + |prims|^floor(d/2)` instead of `|prims|^d`.

WHAT THIS IS AND IS NOT, stated plainly because it does not dodge the standing objection. This is **still search**. It
halves the exponent; it does not remove it. `|prims|^(d/2)` beats `|prims|^d` and is still exponential, so the wall moves
rather than disappearing. What it does establish is that the prediction-error signal is exact and free at the JOIN even
though it is worthless in the interior — which is a real structural fact about this problem and the reason a meet-based
method works where a greedy one cannot.

AND THE REFRAMING THIS LINE OF WORK FORCED, which may matter more than the mechanism. Our primitives factor into a
position permutation and a value map, and both are group elements. Two consequences:
  * The COMPOSED map is EXACTLY recoverable from demonstrations — intersect the relational signature `Phi(x,y)` across
    demonstrations and the accidental value coincidences cancel, leaving `(pi_f, delta_f)` precisely. Perception of the
    whole is not the hard part and never was.
  * What remains is FACTORING a known group element into generators. That is the word problem on a Cayley graph, and its
    difficulty is combinatorial and intrinsic — not statistical.
**So identification failing to improve with data (`decompose.py`: held-out 0.000 at 10, 40 and 134 programs) is exactly
what factorisation predicts.** It is not a pattern-recognition problem, so no amount of coverage could have helped, and
the flat sweep was the right answer to the wrong question. Search is not a workaround for a missing learned mechanism; for
this problem class it is what the problem *is*. The open question becomes which structure of the group a proposal
mechanism could exploit — meet-in-the-middle being the generic one that needs no structure at all.

Usage:  python experiments/transformers/localise.py
"""
from __future__ import annotations

import argparse
import itertools

import torch

from arity import build_splits
from chaining import apply_task
from diversity import NAMES
from h1_lid import L, V
from propose import relational

MAXM = 3
PRIMS = "rot1,reverse,swap_pairs,add1,negate"


def gather_tables(p, dev):
    """A primitive as `out = vmp[x[idx]]`, and its exact INVERSE as the argsort of each table. Asserted below rather
    than trusted: an inverse that is silently wrong would make every backward frontier state garbage and the meet would
    simply never fire, which reads as a null result rather than as a bug."""
    ar_L, ar_V = torch.arange(L, device=dev), torch.arange(V, device=dev)
    f = lambda s: apply_task(s, (p,))
    positional = all(bool((f(torch.full((1, L), c, dtype=torch.long, device=dev)) == c).all()) for c in range(V))
    idx = f(ar_L.unsqueeze(0)).squeeze(0) if positional else ar_L.clone()
    vmp = ar_V.clone() if positional else f(ar_V.unsqueeze(0)).squeeze(0)
    return (idx, vmp), (torch.argsort(idx), torch.argsort(vmp))


def apply_tab(s, tab):
    idx, vmp = tab
    return vmp[s.gather(-1, idx.expand(*s.shape[:-1], L))]


def recover_map(x, y, affine=True):
    """The COMPOSED transformation, recovered by intersecting a relational signature across demonstrations: keep every
    hypothesis consistent with EVERY demonstration, so accidental coincidences cancel.

    THE FEATURE SPACE HAS TO SPAN THE ACTUAL GROUP, and getting this wrong is instructive. `affine=False` tests only
    `y[j] == x[i] + d`, i.e. the TRANSLATION subgroup of the value maps. But `negate` is `x -> -x`, a reflection, so the
    value group here is the affine group `{x -> ±x + c}`. Under the translation-only feature the intersection comes back
    EMPTY for any composition containing an odd number of negations — and it does so no matter how many demonstrations
    are supplied (measured flat from 4 to 64). **A failure that is flat in the amount of data is a wrong hypothesis
    space, not insufficient evidence** — the same signature as `decompose.py`'s flat coverage sweep, from a completely
    different mechanism."""
    n = x.shape[0]
    signs = (1, -1) if affine else (1,)
    cons = []
    for s in signs:
        d = (y.unsqueeze(-2) - s * x.unsqueeze(-1)) % V               # (n, L_src, L_tgt)
        hit = torch.zeros(n, L, L, V, device=x.device).scatter_(-1, d.unsqueeze(-1), 1.0)
        cons.append(hit.min(0).values > 0)                            # consistent across every demonstration
    return torch.stack(cons, dim=-2)                                 # (L_src, L_tgt, |signs|, V)


def bidirectional(task, prims, fwd, inv, dev, n_demos=8, max_depth=MAXM):
    """Meet-in-the-middle with exact error localisation at the join. Returns (found, candidates evaluated, depth used)."""
    g = torch.Generator(device=dev).manual_seed(1234)
    x = torch.randint(0, V, (n_demos, L), generator=g, device=dev)
    y = apply_task(x, task)
    probe = torch.randint(0, V, (24, L), generator=g, device=dev)
    want = tuple(apply_task(probe, task).flatten().tolist())

    evaluated = 0
    for depth in range(1, max_depth + 1):
        nb = depth // 2                                              # steps peeled BACKWARD from the observation
        nf = depth - nb                                              # steps grown FORWARD from the input
        back = {}
        for prog in itertools.product(range(len(prims)), repeat=nb):
            s = y
            for p in reversed(prog):                                 # undo the suffix, last primitive first
                s = apply_tab(s, inv[p])
            evaluated += 1
            back.setdefault(tuple(s.flatten().tolist()), []).append(prog)
        for prog in itertools.product(range(len(prims)), repeat=nf):
            s = x
            for p in prog:
                s = apply_tab(s, fwd[p])
            evaluated += 1
            key = tuple(s.flatten().tolist())                        # the JOIN: zero prediction error across every demo
            for suffix in back.get(key, []):
                full = tuple(prims[i] for i in prog + suffix)
                if tuple(apply_task(probe, full).flatten().tolist()) == want:
                    return True, evaluated, depth
    return False, evaluated, None


def forward_only(task, prims, fwd, dev, n_demos=8, max_depth=MAXM):
    """Exhaustive forward enumeration with the same verification, as the cost baseline."""
    g = torch.Generator(device=dev).manual_seed(1234)
    x = torch.randint(0, V, (n_demos, L), generator=g, device=dev)
    y = apply_task(x, task)
    probe = torch.randint(0, V, (24, L), generator=g, device=dev)
    want = tuple(apply_task(probe, task).flatten().tolist())
    evaluated = 0
    for depth in range(1, max_depth + 1):
        for prog in itertools.product(range(len(prims)), repeat=depth):
            s = x
            for p in prog:
                s = apply_tab(s, fwd[p])
            evaluated += 1
            if torch.equal(s, y):
                full = tuple(prims[i] for i in prog)
                if tuple(apply_task(probe, full).flatten().tolist()) == want:
                    return True, evaluated, depth
    return False, evaluated, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    prims = [NAMES.index(nm) for nm in PRIMS.split(",")]
    sup, held = build_splits(args.seed, prims, max_len=MAXM)
    fwd, inv = {}, {}
    for k, p in enumerate(prims):
        fwd[k], inv[k] = gather_tables(p, dev)
    # The inverses are ASSERTED, not assumed.
    gchk = torch.Generator(device=dev).manual_seed(5)
    s = torch.randint(0, V, (64, L), generator=gchk, device=dev)
    for k in range(len(prims)):
        assert torch.equal(apply_tab(s, fwd[k]), apply_task(s, (prims[k],))), f"forward table wrong for {NAMES[prims[k]]}"
        assert torch.equal(apply_tab(apply_tab(s, fwd[k]), inv[k]), s), f"inverse wrong for {NAMES[prims[k]]}"
    print(f"device {dev} | {len(prims)} primitives {PRIMS} | forward and INVERSE tables verified exactly")

    # Is the composed map really EXACTLY recoverable from demonstrations? Measured across demonstration counts rather
    # than asserted -- at 8 demonstrations it is only partly so, and the claim needs the number attached to it.
    print("composed-map recovery on held-out triples (exact = one surviving hypothesis per source position):")
    for label, aff in (("translation-only", False), ("AFFINE (correct group)", True)):
        out = []
        for nd in (4, 8, 16, 32, 64):
            gchk = torch.Generator(device=dev).manual_seed(11)
            exact = res = 0
            for t in held[3]:
                x = torch.randint(0, V, (nd, L), generator=gchk, device=dev)
                m = recover_map(x, apply_task(x, t), affine=aff)
                exact += int(int(m.sum()) == L)
                res += int(m.sum()) / L                              # mean surviving hypotheses per source position
            out.append(f"{nd}d {exact:>2}/{len(held[3])} ({res / len(held[3]):.2f})")
        print(f"   {label:<24}" + "  ".join(out))
    print("   (n_demos exact/total (mean surviving hypotheses per position). A rate FLAT in n_demos means the")
    print("    hypothesis space is wrong, not the evidence insufficient.)\n")

    print(f"{'method':<22}{'pool':<20}{'n':>4}{'found':>8}{'cands':>8}")
    rows = [("meet-in-the-middle", bidirectional), ("forward enumeration", forward_only)]
    for name, fn in rows:
        for pn, tasks in (("HELD-OUT pairs", held[2]), ("HELD-OUT triples", held[3]),
                          ("supervised triples", sup[3])):
            rs = [fn(t, prims, fwd, inv, dev, args.demos) if fn is bidirectional
                  else fn(t, prims, fwd, dev, args.demos) for t in tasks]
            print(f"{name:<22}{pn:<20}{len(tasks):>4}{sum(r[0] for r in rs) / len(rs):>8.3f}"
                  f"{sum(r[1] for r in rs) / len(rs):>8.1f}")

    print("\n'cands' = states actually expanded. Meet-in-the-middle costs |prims|^ceil(d/2) + |prims|^floor(d/2) against")
    print("enumeration's |prims|^d, so the exponent HALVES -- and it is still exponential. Verification is exact here")
    print("(true primitives), so this isolates SEARCH cost; multiply by ~0.86 for search.py's learned-executor fidelity.")
    print("Reference points: propose.py's best difference-scoring reached 0.389 on held-out triples at 13 candidates;")
    print("enumeration reaches 1.000 here (exact verification) and reached 0.778 with a learned executor at 155.")


if __name__ == "__main__":
    main()
