"""Composition gap = (compose direct) - (parts -> compose transfer), per arm.

Reads the two run files written by train_bc and reports the gap with error bars
over seeds. The gap is the program-centric signal: ~0 means the arm *composes*
the whole from separately-learned parts; large means it can only *memorize* the
whole when trained on it directly.

Prereqs (run both with matching --seeds, e.g. 5):
    python -m agent.train_bc --train-mechanics compose --seeds 5 --compile
    python -m agent.train_bc --train-mechanics nav,key_door,block_pad \
        --test-mechanic compose --seeds 5 --compile
Then:
    python -m agent.gap_report
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Dict, List


def _finals(path: str) -> Dict[str, List[float]]:
    with open(path) as f:
        d = json.load(f)
    if d.get("seed_finals"):
        return d["seed_finals"]
    return {b: [h[-1]["test_masked"]] for b, h in d["results"].items()}


def _ms(v: List[float]):
    return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)


def main() -> None:
    runs = os.path.join(os.path.dirname(__file__), "runs")
    direct_path = os.path.join(runs, "bc_compose_to_compose.json")
    transfer_path = os.path.join(runs, "bc_nav+key_door+block_pad_to_compose.json")
    for p in (direct_path, transfer_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run both compose-direct and parts->compose sweeps")

    direct, transfer = _finals(direct_path), _finals(transfer_path)
    print(f"{'arm':9s} {'direct':>16s} {'transfer':>16s} {'gap (d - t)':>16s}")
    for b in direct:
        d, t = direct[b], transfer.get(b, [])
        n = min(len(d), len(t))
        gaps = [d[i] - t[i] for i in range(n)]   # paired by seed index
        dm, dsd = _ms(d)
        tm, tsd = _ms(t)
        gm, gsd = _ms(gaps) if gaps else (float("nan"), 0.0)
        print(f"{b:9s} {dm:6.3f} +/- {dsd:.3f} {tm:6.3f} +/- {tsd:.3f} {gm:+6.3f} +/- {gsd:.3f}")
    print("\nlower gap = composes the whole from its parts (program-centric);"
          "\nlarge gap = only memorizes the whole (value-centric).")


if __name__ == "__main__":
    main()
