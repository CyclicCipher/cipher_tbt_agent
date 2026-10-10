"""The headline measure: forward-equivalents (and steps) each run needs to first reach a validation loss.
Usage: python reach.py 2.37 runs/e2/m2/*.json runs/e1/m2/base_s*.json"""
from __future__ import annotations

import json
import sys

target = float(sys.argv[1])
print(f"first evaluation at or below val {target}")
for f in sys.argv[2:]:
    r = json.load(open(f))
    hit = next((p for p in r["log"] if p["val"] <= target), None)
    name = f.replace("\\", "/").split("runs/")[-1]
    if hit:
        print(f"  {name:40s} step {hit['step']:4d}  cost {hit['cost']:7d} fwd  t {hit['t']:6.1f}s")
    else:
        print(f"  {name:40s} not reached (final {r['log'][-1]['val']:.4f}, eval every {r['args']['eval_every']} steps)")
