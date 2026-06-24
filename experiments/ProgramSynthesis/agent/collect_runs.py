"""Collect all BC sweep runs in agent/runs/ into one compact JSON for plotting.

Each run file (written by train_bc.py) holds a per-step history per binding arm.
This flattens them to held-out-accuracy curves keyed by run, so a small-multiples
widget can show every mechanic's arm ordering side by side.

    python -m agent.collect_runs            # prints combined JSON to stdout
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict


def collect(runs_dir: str) -> Dict:
    out: Dict[str, Dict] = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        train = d.get("train_mechanics", [])
        test = d.get("test_mechanic", "")
        transfer = train != [test]
        title = (f"{'+'.join(train)} -> {test}") if transfer else test
        arms = {}
        finals = {}
        for binding, hist in d["results"].items():
            arms[binding] = {
                "step": [h["step"] for h in hist],
                "test": [round(h["test_masked"], 4) for h in hist],
            }
            finals[binding] = round(hist[-1]["test_masked"], 4)
        out[os.path.basename(path)[:-5]] = {
            "title": title, "transfer": transfer, "arms": arms, "final": finals,
        }
    return out


if __name__ == "__main__":
    import sys
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "runs"
    )
    print(json.dumps(collect(runs_dir)))
