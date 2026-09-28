"""expW (E43) -- how much of the board should the rule see? (the user's question, 2026-09-27)

The frame is ALREADY fully visible: every cell is a token in the block's sequence. What is local is the RULE --
a `LocalRule` maps a (2r+1)x(2r+1) window to the centre's new colour, and `ActionModel` only ever offers
r in (1, 2), picking between them by price. Toggle's 0.227 in E42 is the visible cost: a radius-1 window cannot
express a rule that changes a distant cell.

So this measures the trade directly, per game and per radius r in (1, 2, 3):
  - accuracy on windows where the centre cell CHANGES (E42's standing rule: the aggregate is worthless, since a
    constant "unchanged" predictor scores ~0.92);
  - the EVIDENCE COLLAPSE -- observations per distinct window -- because a bigger field is E40's trap in two
    dimensions: more context, exponentially less evidence per context;
  - which radius the PRICE picks, against which one actually predicts best. E39 showed the price undervalues
    what generalises, so these may disagree, and that disagreement is the point.

Sample size is the other fix E42 demanded: it had 21-34 changed windows per game, where one item moved the score
3-5 points. This plays longer and reports the counts it actually achieved rather than assuming they suffice.
No network trained or fitted anywhere.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))
sys.path.insert(0, str(HERE))

import arcgames as AG                                                  # noqa: E402
import e28 as E28                                                      # noqa: E402
from expU_icl_block import raw_rule, tokens_from                       # noqa: E402
from tasks.games import CollectAll, LockPath, MultiKey, Sokoban, Toggle  # noqa: E402
from tasks.harness import Environment                                  # noqa: E402

RADII = (1, 2, 3)
GAMES = {"LockPath": LockPath, "CollectAll": CollectAll, "MultiKey": MultiKey,
         "Sokoban": Sokoban, "Toggle": Toggle}
K = 512                                                                # demonstrations per action, in context


class WideActionModel(AG.ActionModel):
    def __init__(self, radii=RADII):
        super().__init__(radii=radii)


def scaffold_rule(r):
    from types import SimpleNamespace
    n = (2 * r + 1) ** 2
    return SimpleNamespace(r=r, mask=np.ones(n, dtype=bool), table={})


def evaluate(sim, test_by_slot, r):
    right = wrong = ch_r = ch_n = un_r = un_n = 0
    ent = sim.memory[2:]
    side = 2 * r + 1
    for slot, items in test_by_slot.items():
        for key, counts in items:
            truth = int(max(counts, key=counts.get))
            cols = np.frombuffer(key, dtype=np.int16)
            centre = cols[len(cols) // 2]
            cen = E28.BORDER if centre == E28.BORDER else int(centre)
            changed = cen != truth
            ch_n += changed; un_n += (not changed)
            if len(ent) == 0:
                wrong += 1; continue
            q = torch.zeros(sim.memory.shape[1])
            for idx, col in enumerate(cols):
                o = (idx // side - r, idx % side - r)
                c = sim.BORDER if col == E28.BORDER else int(col)
                if o == (0, 0):
                    q[sim.T.start + c] = 1.0
                elif o in sim.offs:
                    q[sim.N[sim.offs.index(o)].start + c] = 1.0
            q[sim.A.start + slot] = float(len(sim.offs) + 2)
            got = int(torch.argmax(ent[int(torch.argmax(ent @ q))][sim.OUT]).item())
            hit = got == truth
            right += hit; wrong += not hit
            ch_r += hit and changed; un_r += hit and not changed
    n = right + wrong
    return {"n": n, "accuracy": right / max(1, n),
            "unchanged_baseline": un_n / max(1, n),
            "changed": {"n": int(ch_n), "accuracy": ch_r / max(1, ch_n)},
            "balanced": 0.5 * (ch_r / max(1, ch_n) + un_r / max(1, un_n))}


def main():
    t0 = time.time()
    seed, budget, levels = 0, 400, 3
    AG.ActionModel = WideActionModel                                   # radii 1,2,3 compete instead of 1,2
    out = {"experiment": "E43", "seed": seed, "budget": budget, "levels": levels,
           "radii": list(RADII), "k_per_action": K, "games": {}}
    print(f"radii {RADII}, budget {budget}, levels {levels}, K {K}", flush=True)

    for name, cls in GAMES.items():
        try:
            env = Environment(cls())
            _r, _s, player = AG.play(env, budget_per_level=budget, max_levels=levels, seed=seed,
                                     sleep=True, return_player=True)
        except Exception as e:
            print(f"{name}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            continue
        acts = [a for a in player.actions if player.models[a].n_obs]
        priced = {r: 0 for r in RADII}
        for a in acts:
            priced[player.models[a].best().r] += 1
        rec = {"n_actions": len(acts), "price_picks": priced, "by_radius": {}}
        for r in RADII:
            rules = {i: player.models[a].rules[r] for i, a in enumerate(acts)}
            n_obs = sum(sum(sum(c.values()) for c in rl.full.values()) for rl in rules.values())
            n_uniq = sum(len(rl.full) for rl in rules.values())
            rng = np.random.default_rng(seed)
            tr, te = {}, {}
            for i, rl in rules.items():
                items = sorted(rl.full.items()); rng.shuffle(items)
                cut = max(1, int(0.7 * len(items)))
                tr[i], te[i] = items[:cut], items[cut:]
            sim = E28.WrittenSim({i: scaffold_rule(r) for i in range(len(acts))},
                                 list(range(len(acts))), 8, 11)
            tokens_from(sim, {}, list(range(len(acts))))
            empty = evaluate(sim, te, r)
            n_tok = tokens_from(sim, {i: raw_rule(v[:K], r) for i, v in tr.items()}, list(range(len(acts))))
            got = evaluate(sim, te, r)
            rec["by_radius"][r] = {
                "cells": (2 * r + 1) ** 2, "offsets": len(sim.offs), "d": int(sim.memory.shape[1]),
                "observations": int(n_obs), "distinct_windows": int(n_uniq),
                "obs_per_window": n_obs / max(1, n_uniq), "tokens": n_tok,
                "empty_changed": empty["changed"]["accuracy"], **got}
            v = rec["by_radius"][r]
            print(f"  {name:<11} r={r}  {v['cells']:>2} cells  obs/window {v['obs_per_window']:5.2f}  "
                  f"CHANGED {v['changed']['accuracy']:.3f} on {v['changed']['n']:>5}  "
                  f"all {v['accuracy']:.3f} (unch {v['unchanged_baseline']:.3f})  balanced {v['balanced']:.3f}",
                  flush=True)
        best_r = max(RADII, key=lambda r: rec["by_radius"][r]["changed"]["accuracy"])
        rec["best_radius_by_accuracy"] = best_r
        rec["price_majority_radius"] = max(priced, key=priced.get)
        rec["price_agrees"] = best_r == rec["price_majority_radius"]
        out["games"][name] = rec
        print(f"  {name:<11} price picks r={rec['price_majority_radius']} "
              f"({priced}); best on changed windows r={best_r} -> "
              f"{'AGREE' if rec['price_agrees'] else 'DISAGREE'}", flush=True)

    out["summary"] = {g: {"price": r["price_majority_radius"], "best": r["best_radius_by_accuracy"],
                          "agrees": r["price_agrees"],
                          "changed_n": {rr: r["by_radius"][rr]["changed"]["n"] for rr in RADII},
                          "changed_acc": {rr: round(r["by_radius"][rr]["changed"]["accuracy"], 3) for rr in RADII}}
                      for g, r in out["games"].items()}
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expW.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\n" + json.dumps(out["summary"], indent=1), f"\n{out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
