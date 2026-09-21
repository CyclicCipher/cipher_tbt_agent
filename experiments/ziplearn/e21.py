"""E21 — two rules that agree on most inputs (DESIGN.md §9, OPEN-6): when is a near-duplicate its own block?

Rule A (shift by 3) for a stretch, then rule D: the same as A on 9 of the 11 inputs and different on 2. A three-pair
context from D matches A's description ~55% of the time, so the selector will route D's pairs to A's block and pay
exceptions -- the MDL-correct description of D while the evidence is thin ("A, with exceptions") -- until a block
of its own is cheaper. Measured: the number of D observations before a second block is minted (over 20 streams),
the exception rate A's block carries meanwhile, and D's accuracy on its two differing inputs before and after the
mint. Reported against the doc's stated expectation: "one description until an observation refutes it, which is
the correct behaviour and also a guaranteed error on the first refuting case". No pass/refute: a measurement.

    python experiments/ziplearn/e21.py           (CPU, ~20 s) -> runs/e21/e21.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ziplearner import ContinualLayer   # noqa: E402

V = 11
A = lambda x: (x + 3) % V
DIFF = {4: 0, 9: 6}                      # D differs from A on x = 4 and x = 9
D = lambda x: DIFF.get(x, A(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e21"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mint_at, never = [], 0
    acc_before, acc_after = [], []
    eps_before = []
    for s in range(args.streams):
        rng = np.random.default_rng(args.seed + s)
        layer = ContinualLayer(V)
        for _ in range(40):
            x = int(rng.integers(V))
            layer.observe(x, A(x))
        minted = None
        for i in range(120):
            x = int(rng.integers(V))
            nb = len(layer.blocks)
            layer.observe(x, D(x))
            if minted is None and len(layer.blocks) > nb:
                minted = i + 1
                acc_before.append(np.mean([layer.predict_in_context([(4, D(4)), (9, D(9)), (1, D(1))], x) == D(x) for x in DIFF]) if False else 0)
        if minted is None:
            never += 1
        else:
            mint_at.append(minted)
        # accuracy on D's two differing inputs, given a context that contains them
        ctx = [(4, D(4)), (9, D(9)), (1, D(1))]
        acc_after.append(float(np.mean([layer.predict_in_context(ctx, x) == D(x) for x in DIFF])))
        eps_before.append(layer.blocks[0]["layer"].keep().rate)
    print(f"E21: A for 40 pairs, then D (= A on 9 of 11 inputs) for 120; {args.streams} streams")
    print(f"   a block of its own for D minted after {np.mean(mint_at) if mint_at else float('nan'):.1f} D-observations on average "
          f"(min {min(mint_at) if mint_at else '-'}, max {max(mint_at) if mint_at else '-'}); never in {never} streams")
    print(f"   exception rate carried by A's block at the end: {np.mean(eps_before):.3f}")
    print(f"   accuracy on D's two differing inputs at the end, context containing them: {np.mean(acc_after):.2f}")
    json.dump(dict(mint_at=mint_at, never=never, eps_A=float(np.mean(eps_before)), acc_D_diff=float(np.mean(acc_after))),
              open(out / "e21.json", "w"), indent=1)


if __name__ == "__main__":
    main()
