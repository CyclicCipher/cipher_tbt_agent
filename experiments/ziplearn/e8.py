"""E8 — structures of structures (DESIGN.md §7 one level up, OPEN-5; pre-registered from the E7 anatomy numbers).

One new kind of description whose parameters are OTHER descriptions, in two places:
  A. the library re-describes itself: a generating set of items chosen by price, every other item a WORD in them
     (`WordLibrary`). Pass: the chosen description costs no more than the 138.1 bits the anatomy pass found by hand
     (flat: 161.4). Refute: it never beats flat.
  B. generalisation one level up: tasks whose position permutation is in the GROUP the library generates but was
     never seen (19 of the 36). Three learners on the same tasks: from scratch (`TwoLayer`), with the named items
     (`TwoLayerWithLibrary` -- none matches), and with words (`TwoLayerWithWords`, every group element a candidate
     at its word price). Pass: with words, exact after <= 2 demonstrations on average and fewer bits than from scratch;
     refute: no better than from scratch.
  C. E6's blocks consolidate into templates (`consolidate`): same-kind blocks share one structure name. Pass: the two
     shift blocks cost <= 17.5 bits as a template (22.2 separate), and retention is unchanged. Refute: no saving.

    python experiments/ziplearn/e8.py            (seconds, CPU) -> runs/e8/e8.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inner_objective"))
sys.path.insert(0, str(HERE.parent / "transformers"))
from ziplearner import (WordLibrary, TwoLayer, TwoLayerWithLibrary, TwoLayerWithWords, PermLibrary, ContinualLayer,
                        consolidate, perm_then)                                     # noqa: E402
import h1_lid as H                                                                  # noqa: E402
import e6                                                                           # noqa: E402

L, V = 6, 5


def apply_task(x, pi, a, b):
    """y[j] = a * x[pi[j]] + b mod V."""
    return (a * x[..., list(pi)] + b) % V


def run_learner(make, tasks, K, n, rng):
    ttc, bits = [], []
    for pi, (a, b) in tasks:
        for _ in range(n):
            x = rng.integers(0, V, (K + 1, L))
            y = apply_task(x, pi, a, b)
            learner = make()
            first = None
            for k in range(K):
                learner.observe_demo(x[k], y[k])
                pred = learner.predict_seq(x[K])
                if first is None and all(p is not None and p == int(t) for p, t in zip(pred, y[K])):
                    first = k + 1
            ttc.append(first if first is not None else K + 1)
            _n, _ab, pos = learner.best()
            bits.append(pos.cost + learner.bits_pos_name)
    return float(np.mean(ttc)), float(np.mean(bits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e8"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    named = [tuple(p) for p in json.load(open(HERE / "runs" / "e3" / "e3.json"))["named"]]
    prims = {n: tuple(H.PRIMS[n](__import__("torch").arange(L)).tolist()) for n in H.NAMES[:5]}

    # ── A. the library re-describes itself ─────────────────────────────────────────────────────────────────────────
    lib = WordLibrary(named, L, max_generators=4)
    gens = [[n for n, p in prims.items() if p == g] or [f"{a}∘{b}" for a in H.NAMES[:5] for b in H.NAMES[:5]
            if perm_then(prims[a], prims[b]) == g][:1] for g in lib.generators]
    group = lib.group()
    print(f"A. library: {len(named)} items; flat {lib.price_flat():.1f} bits; chosen description {lib.price:.1f} bits with "
          f"{len(lib.generators)} generators {gens}; group reached: {len(group)} permutations; "
          f"longest word {max(len(w) for w in lib.words.values())}")
    a_pass = lib.price <= 138.1

    # ── B. generalisation one level up: unseen elements of the group ───────────────────────────────────────────────
    unseen = [pi for pi in group if pi not in named and any(pi[j] != j for j in range(L))]
    value_maps = [(1, 0), (1, 1), (4, 0)]                                 # identity, inc, negate
    tasks = [(pi, value_maps[i % 3]) for i, pi in enumerate(unseen)]
    perm_lib = PermLibrary()
    perm_lib.named = list(named)
    learners = {"from scratch": lambda: TwoLayer(L, V),
                "named items only": lambda: TwoLayerWithLibrary(L, V, perm_lib),
                "words (the group)": lambda: TwoLayerWithWords(L, V, lib)}
    b_res = {}
    print(f"\nB. {len(unseen)} permutations in the group that were never named, each with a value map, {args.n} instances:")
    for name, make in learners.items():
        ttc, bits = run_learner(make, tasks, args.k, args.n, rng)
        b_res[name] = dict(demos_to_exact=ttc, position_bits=bits)
        print(f"   {name:<20} demonstrations to exact {ttc:.2f} | position layer {bits:.1f} bits")
    b_pass = (b_res["words (the group)"]["demos_to_exact"] <= 2.0 and
              b_res["words (the group)"]["position_bits"] < b_res["from scratch"]["position_bits"])

    # ── C. E6's blocks consolidate into templates ──────────────────────────────────────────────────────────────────
    V11 = 11
    layer = ContinualLayer(V11)
    for name in ["A", "B", "C", "A"]:
        for _ in range(40):
            x = int(rng.integers(V11))
            layer.observe(x, e6.apply(e6.RULES[name], x, V11))
    before, after, templates = consolidate(layer)
    ret_before = e6.retention(layer, V11, 3, rng)
    shifts = [t for t in templates if t["kind"] == "shift"]
    two_shift_before = sum(b["layer"].price(b["layer"].keep()) for b in layer.blocks if b["layer"].keep().name == "shift") + 2 * math.log2(len(layer.blocks))
    n_kinds = len(layer.blocks[0]["layer"].structs)
    two_shift_after = (math.log2(len(templates)) + math.log2(n_kinds) +
                       sum(b["layer"].price(b["layer"].keep()) - math.log2(n_kinds) for b in layer.blocks if b["layer"].keep().name == "shift"))
    print(f"\nC. E6's blocks: {[ (b['layer'].keep().name, b['n']) for b in layer.blocks ]} -> templates {[(t['kind'], t['contexts']) for t in templates]}")
    print(f"   all blocks: {before:.1f} bits separate -> {after:.1f} as templates; the two shift blocks: {two_shift_before:.1f} -> {two_shift_after:.1f}")
    print(f"   retention with the template description (routing unchanged by construction): {ret_before}")
    c_pass = two_shift_after <= 17.5 and all(v >= 0.999 for v in ret_before.values())

    verdict = "PASS" if (a_pass and b_pass and c_pass) else "REFUTED"
    print(f"\nE8 verdict: {verdict} (A {a_pass}, B {b_pass}, C {c_pass})")
    json.dump(dict(A=dict(flat=lib.price_flat(), chosen=lib.price, generators=[list(g) for g in lib.generators], generators_named=gens,
                          group_size=len(group), words={str(k): v for k, v in lib.words.items()}),
                   B=dict(n_unseen=len(unseen), results=b_res), C=dict(before=before, after=after, templates=templates,
                   two_shift_before=two_shift_before, two_shift_after=two_shift_after, retention=ret_before),
                   verdict=verdict), open(out / "e8.json", "w"), indent=1)


if __name__ == "__main__":
    main()
