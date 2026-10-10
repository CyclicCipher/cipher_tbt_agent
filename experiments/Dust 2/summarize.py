"""Print the training (M2) table of an experiment: validation loss per arm and seed, mean, difference from the baseline,
wall-clock and cost. Usage: python summarize.py runs/e1/m2 [base_arm]"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
base = sys.argv[2] if len(sys.argv) > 2 else "base"
runs = defaultdict(dict)
for f in sorted(d.glob("*_s*.json")):
    arm, seed = f.stem.rsplit("_s", 1)
    runs[arm][int(seed)] = json.load(open(f))
bm = sum(r["val"] for r in runs[base].values()) / len(runs[base]) if base in runs else None
print(f"{'arm':16s} {'val per seed':22s} {'mean':>7s} {'vs base':>8s} {'secs':>6s} {'cost':>8s}")
for arm, rs in runs.items():
    vals = [rs[s]["val"] for s in sorted(rs)]
    m = sum(vals) / len(vals)
    secs = sum(rs[s]["secs"] for s in rs) / len(rs)
    cost = rs[min(rs)]["cost"]
    diff = f"{m - bm:+.4f}" if bm is not None else ""
    print(f"{arm:16s} {' / '.join(f'{v:.4f}' for v in vals):22s} {m:7.4f} {diff:>8s} {secs:6.0f} {cost:8d}")
