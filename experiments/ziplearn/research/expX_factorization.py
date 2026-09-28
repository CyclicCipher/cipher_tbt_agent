"""expX (E44) -- is the evidence collapse a FACTORIZATION issue? (the user's question, 2026-09-27)

E43 reported observations per distinct window falling ~4x per radius step and read it as the n-gram trap in two
dimensions. That reading used `LocalRule.full` -- the WHOLE-window evidence -- because the in-context arms feed
raw demonstrations. But `full` is the UNFACTORIZED representation: it treats a (2r+1)^2 window as one atomic key,
so every irrelevant cell that varies mints a new key for the same rule. The sleep pass already fixes exactly that
by dropping cells the price cannot justify (E24: rules keep 2-4 of 9, ~10x fewer bits), which is subset selection
-- the crudest possible factorization.

So the question is whether the collapse survives factorization. Measured per game and radius:
  - distinct keys and observations per key, for `full` (unfactorized) against `table` (the masked, factored form);
  - how many of the (2r+1)^2 cells the mask keeps, against how many exist;
  - and a second axis the mask cannot express: how many distinct BEHAVIOURS the centre colour has, against how
    many colours occur -- i.e. how far a colour-equivalence merge would go, which is value-axis factorization
    rather than position-axis.

No network trained or fitted.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))

import arcgames as AG                                                  # noqa: E402
from tasks.games import CollectAll, LockPath, MultiKey, Sokoban, Toggle  # noqa: E402
from tasks.harness import Environment                                  # noqa: E402

RADII = (1, 2, 3)
GAMES = {"LockPath": LockPath, "CollectAll": CollectAll, "MultiKey": MultiKey,
         "Sokoban": Sokoban, "Toggle": Toggle}


class WideActionModel(AG.ActionModel):
    def __init__(self, radii=RADII):
        super().__init__(radii=radii)


def obs_of(tab):
    return sum(sum(c.values()) for c in tab.values())


def colour_behaviours(rule, r):
    """Value-axis factorization: group the CENTRE colour by the distribution of outcomes it leads to. If many
    colours behave identically (every wall is a wall), a merge collapses the key space without dropping any cell."""
    side = 2 * r + 1
    mid = (side * side) // 2
    beh = defaultdict(lambda: defaultdict(int))
    for key, counts in rule.full.items():
        cols = np.frombuffer(key, dtype=np.int16)
        c = int(cols[mid])
        for out, n in counts.items():
            beh[c][int(out) - c if out != AG.BORDER and c != AG.BORDER else 9999] += n
    sigs = {}
    for c, d in beh.items():
        tot = sum(d.values())
        sigs[c] = tuple(sorted((k, round(v / tot, 2)) for k, v in d.items() if v / tot > 0.02))
    classes = len(set(sigs.values()))
    return {"colours_seen": len(sigs), "distinct_behaviours": classes,
            "merge_factor": len(sigs) / max(1, classes)}


def main():
    t0 = time.time()
    seed, budget, levels = 0, 400, 3
    AG.ActionModel = WideActionModel
    out = {"experiment": "E44", "seed": seed, "budget": budget, "levels": levels, "games": {}}

    for name, cls in GAMES.items():
        try:
            env = Environment(cls())
            _r, _s, player = AG.play(env, budget_per_level=budget, max_levels=levels, seed=seed,
                                     sleep=True, return_player=True)
        except Exception as e:
            print(f"{name}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            continue
        acts = [a for a in player.actions if player.models[a].n_obs]
        rec = {}
        for r in RADII:
            rules = [player.models[a].rules[r] for a in acts]
            n_cells = (2 * r + 1) ** 2
            full_keys = sum(len(rl.full) for rl in rules)
            full_obs = sum(obs_of(rl.full) for rl in rules)
            tab_keys = sum(len(rl.table) for rl in rules)
            tab_obs = sum(obs_of(rl.table) for rl in rules)
            kept = [int(rl.mask.sum()) for rl in rules]
            cb = [colour_behaviours(rl, r) for rl in rules]
            rec[r] = {
                "cells": n_cells, "mask_cells_kept": kept, "mean_kept": float(np.mean(kept)),
                "full_keys": full_keys, "full_obs_per_key": full_obs / max(1, full_keys),
                "table_keys": tab_keys, "table_obs_per_key": tab_obs / max(1, tab_keys),
                "factor_gain": (tab_obs / max(1, tab_keys)) / max(1e-9, full_obs / max(1, full_keys)),
                "colour_merge_factor": float(np.mean([c["merge_factor"] for c in cb])),
                "colours_seen": float(np.mean([c["colours_seen"] for c in cb])),
                "distinct_behaviours": float(np.mean([c["distinct_behaviours"] for c in cb])),
            }
            v = rec[r]
            print(f"  {name:<11} r={r}  cells {n_cells:>2} kept {v['mean_kept']:4.1f}  "
                  f"UNFACTORED {v['full_keys']:>6} keys, {v['full_obs_per_key']:6.2f} obs/key   "
                  f"MASKED {v['table_keys']:>6} keys, {v['table_obs_per_key']:8.2f} obs/key  "
                  f"(x{v['factor_gain']:.1f})   colours {v['colours_seen']:.1f}->"
                  f"{v['distinct_behaviours']:.1f} behaviours (x{v['colour_merge_factor']:.1f})", flush=True)
        out["games"][name] = rec

    out["summary"] = {
        g: {"unfactored_obs_per_key": {r: round(v["full_obs_per_key"], 2) for r, v in rec.items()},
            "masked_obs_per_key": {r: round(v["table_obs_per_key"], 2) for r, v in rec.items()},
            "mean_cells_kept": {r: round(v["mean_kept"], 1) for r, v in rec.items()},
            "colour_merge_factor": {r: round(v["colour_merge_factor"], 2) for r, v in rec.items()}}
        for g, rec in out["games"].items()}
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expX.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\n" + json.dumps(out["summary"], indent=1), f"\n{out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
