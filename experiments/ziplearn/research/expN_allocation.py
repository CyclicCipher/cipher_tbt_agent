"""expN (S1, 2026-09-27) -- is B0's failing rope2d cell a real interference or an allocation artefact?

DESIGN.md S22.3 asserted it was a register-allocation problem solvable by graph colouring. That was a guess. This
script checks it against the compiler's own arithmetic instead, and reports the table rather than an opinion.

The failure: `layer 1: 159 position-free pairs do not fit beside the N column position pairs of a 465-wide head`.
brainbuilder.py line ~504 computes the room for a layer's position-free (content) tail as

    room = P - (P // 2 + col_pairs)        # P = hd // 2 rotary pairs

i.e. it reserves the ENTIRE row half of every head unconditionally, because `rope2d` rotates the first half of the
pairs by the row coordinate and the second half by the column coordinate. The question this answers: in the layer
that fails, does any head actually USE the row axis? If none does, the reservation is dead space and the fix is
liveness, not width.

Prints JSON to runs/research/expN.json. Compiles a blueprint into tensors; runs no forward pass and trains nothing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import brainbuilder as bb                                              # noqa: E402
from ziplib.blueprint import Arch, Blueprint                           # noqa: E402


def main():
    t0 = time.time()
    bp_path = HERE.parent / "blueprints" / "gridworld.json"
    out = {"blueprint": str(bp_path.name), "cells": {}}

    for r, pos in ((1, "onehot"), (1, "rope2d"), (4, "onehot"), (4, "rope2d")):
        key = f"r{r}_{pos}"
        bp = Blueprint.load(bp_path)
        try:
            bp.field["r"] = r
        except Exception:
            pass
        arch = Arch(d_model="auto", n_head="auto", head_dim="auto", n_layer="auto", pos=pos,
                    max_len="auto", p_star=0.99, dims=dict(H=8, W=11, C=17, V=16, nA=4))
        rec = {"r": r, "pos": pos}
        try:
            brain = bb.compile(bp, arch)
            plan = getattr(brain, "plan", {}) or {}
            rec["ok"] = True
            rec["d_model"] = int(getattr(brain, "d", plan.get("d", 0)) or plan.get("d", 0))
            rec["n_head"] = plan.get("n_head")
            rec["layers"] = plan.get("layers")
            rec["notes"] = plan.get("notes")
        except Exception as e:                                         # the failing cell: capture, do not raise
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        out["cells"][key] = rec
        print(key, "OK" if rec.get("ok") else rec.get("error"), flush=True)

    # ---- the arithmetic behind the failing cell, recomputed by hand from the blueprint ----
    bp = Blueprint.load(bp_path)
    bp.field["r"] = 4
    ex = bp.expand()
    per_layer = {}
    for c in ex.circuits:
        per_layer.setdefault(c.get("place", {}).get("layer", "auto"), []).append((c["name"], c["instr"]))
    out["circuits_by_declared_layer"] = {str(k): v for k, v in per_layer.items()}

    # which instruction kinds use which coordinate axis: Gather is the only positional reader (S21.3 items 1-2)
    kinds = {}
    for c in ex.circuits:
        kinds.setdefault(c["instr"], 0)
        kinds[c["instr"]] += 1
    out["instruction_counts_r4"] = kinds
    out["axis_users"] = {
        "row_axis": sorted({c["instr"] for c in ex.circuits if c["instr"] == "Gather"}),
        "column_axis": sorted({c["instr"] for c in ex.circuits if c["instr"] == "Gather"}),
        "position_free_content": sorted({c["instr"] for c in ex.circuits if c["instr"] in ("Match",)}),
    }
    out["reading"] = (
        "Gather is the only instruction that reads a coordinate axis; Match compares CONTENT and needs "
        "position-free pairs. If the failing layer holds no Gather, the row half it reserves is dead space."
    )
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expN.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
