"""expV (E42, DESIGN §18.1) -- does the written interpreter transfer? ONE set of weights, five games, no training.

E41 showed a written block learning a world model from demonstrations in its context, with byte-identical weights
and 0.000 accuracy on an empty context. But its demonstrations came from the same levels as its test windows, so
what it showed was generalisation across WINDOWS. The claim worth having is stronger and is what ARC-AGI-3 asks
for: one written interpreter, never retrained, consuming ANY game's demonstrations as context.

One `WrittenSim` is compiled ONCE over the full radius-1 neighbourhood with six generic action slots; every arm
uses those same weights. What varies is only the context:
  empty        no entry tokens                                  -- the floor
  native       game G's own demonstrations, tested on G         -- does the interpreter work on a game it was
                                                                   not compiled for?
  cross-level  demonstrations from G's levels 0-1, tested on G's later level
  cross-game   LockPath's demonstrations, tested on G           -- does anything transfer between games?
and every arm is scored against UNCHANGED (predict the centre cell keeps its colour), which is the baseline E41's
substitution arm lacked and without which a high number means nothing: most cells do not change in most games.

Pass: native >= 0.7 on every game (the interpreter is not LockPath-specific) AND cross-level >= 0.9 x native.
Refute: native < 0.5 on any game, or cross-level < 0.5 x native. `cross-game` is REPORTED against the unchanged
baseline with no directional prediction -- it is a measurement, not a claim.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))
sys.path.insert(0, str(HERE))

import e28 as E28                                                      # noqa: E402
from arcgames import play                                              # noqa: E402
from expU_icl_block import all_ones_rule, raw_rule, tokens_from        # noqa: E402
from tasks.games import CollectAll, LockPath, MultiKey, Sokoban, Toggle  # noqa: E402
from tasks.harness import Environment                                  # noqa: E402

SLOTS = 6                                                              # generic action slots; any game maps into them
GAMES = {"LockPath": LockPath, "CollectAll": CollectAll, "MultiKey": MultiKey,
         "Sokoban": Sokoban, "Toggle": Toggle}


def play_game(cls, levels, budget, seed, sleep=True):
    env = Environment(cls())
    _res, _st, player = play(env, budget_per_level=budget, max_levels=levels, seed=seed,
                             sleep=sleep, return_player=True)
    acts = [a for a in player.actions if player.models[a].n_obs]
    return {i: player.models[a].best() for i, a in enumerate(acts)}, len(acts)


def evaluate(sim, test_by_slot, g):
    """Held-out whole-window observations: does the block's lookup return the observed colour? Scored beside
    UNCHANGED (the centre keeps its colour), which is the only honest baseline on grids where little moves."""
    right = wrong = unch_right = 0
    ch_right = ch_n = un_right = un_n = 0                              # split by whether the centre ACTUALLY changes
    ent = sim.memory[2:]
    for slot, items in test_by_slot.items():
        if slot >= SLOTS:
            continue
        for key, counts in items:
            truth = int(max(counts, key=counts.get))
            cols = np.frombuffer(key, dtype=np.int16)
            side = int(round(len(cols) ** 0.5))
            centre = cols[len(cols) // 2]
            cen = E28.BORDER if centre == E28.BORDER else int(centre)
            changed = cen != truth
            unch_right += int(not changed)
            ch_n += changed; un_n += (not changed)
            if len(ent) == 0:
                wrong += 1
                continue
            q = torch.zeros(sim.memory.shape[1])
            for idx, col in enumerate(cols):
                o = (idx // side - side // 2, idx % side - side // 2)
                c = sim.BORDER if col == E28.BORDER else int(col)
                if o == (0, 0):
                    q[sim.T.start + c] = 1.0
                elif o in sim.offs:
                    q[sim.N[sim.offs.index(o)].start + c] = 1.0
            q[sim.A.start + slot] = float(len(sim.offs) + 2)
            score = ent @ q
            got = int(torch.argmax(ent[int(torch.argmax(score))][sim.OUT]).item())
            hit = got == truth
            right += hit
            wrong += not hit
            if changed:
                ch_right += hit
            else:
                un_right += hit
    n = right + wrong
    # The aggregate is dominated by windows where nothing happens -- on these grids ~90% of them -- so a constant
    # "unchanged" predictor scores ~0.92 while being useless to a planner (E25: the failures ARE the missed
    # changes). `changed` is the number that decides whether the model is worth anything.
    return {"n": n, "accuracy": right / max(1, n), "unchanged_baseline": unch_right / max(1, n),
            "changed": {"n": int(ch_n), "accuracy": ch_right / max(1, ch_n), "share": ch_n / max(1, n)},
            "unchanged": {"n": int(un_n), "accuracy": un_right / max(1, un_n)},
            "balanced": 0.5 * (ch_right / max(1, ch_n) + un_right / max(1, un_n))}


def split(rules, seed):
    rng = np.random.default_rng(seed)
    tr, te = {}, {}
    for slot, rule in rules.items():
        items = sorted(rule.full.items())
        rng.shuffle(items)
        cut = max(1, int(0.7 * len(items)))
        tr[slot], te[slot] = items[:cut], items[cut:]
    return tr, te


def main():
    t0 = time.time()
    seed, budget, K = 0, 150, 128
    out = {"experiment": "E42", "seed": seed, "slots": SLOTS, "k_per_slot": K, "games": {}}

    # ---- one block, compiled once; these weights never change again ----
    fake_actions = list(range(SLOTS))
    sim = E28.WrittenSim({a: all_ones_rule(1) for a in fake_actions}, fake_actions, 8, 11)
    w0 = {k: v.clone() for k, v in sim.state_dict().items() if "memory" not in k}
    out["d"] = int(sim.memory.shape[1])
    print(f"one block: d={out['d']}, {len(sim.offs)} offsets, {SLOTS} action slots", flush=True)

    played = {}
    for name, cls in GAMES.items():
        try:
            early, n_a = play_game(cls, 2, budget, seed)
            late, _ = play_game(cls, 3, budget, seed)
            played[name] = {"early": early, "late": late, "n_actions": n_a}
            print(f"  played {name}: {n_a} actions, "
                  f"{sum(len(r.full) for r in early.values())} early windows, "
                  f"{sum(len(r.full) for r in late.values())} late windows", flush=True)
        except Exception as e:
            print(f"  {name}: SKIPPED ({type(e).__name__}: {e})", flush=True)
    out["played"] = {k: {"n_actions": v["n_actions"]} for k, v in played.items()}

    lock_tr, _ = split(played["LockPath"]["early"], seed) if "LockPath" in played else ({}, {})

    for name, p in played.items():
        tr, te = split(p["early"], seed)
        rec = {}
        tokens_from(sim, {}, fake_actions)
        rec["empty"] = evaluate(sim, te, name)
        n = tokens_from(sim, {s: raw_rule(v[:K]) for s, v in tr.items()}, fake_actions)
        rec["native"] = {"tokens": n, **evaluate(sim, te, name)}
        # cross-level: the later play saw one more level; test on the windows it adds
        late_tr, late_te = split(p["late"], seed + 1)
        seen_keys = {s: {k for k, _ in te.get(s, [])} for s in late_te}   # items carry a dict; key on the window
        extra = {s: [it for it in late_te.get(s, []) if it[0] not in seen_keys[s]] for s in late_te}
        if sum(len(v) for v in extra.values()) >= 20:
            rec["cross_level"] = {"tokens": n, **evaluate(sim, extra, name)}
        if lock_tr and name != "LockPath":
            n2 = tokens_from(sim, {s: raw_rule(v[:K]) for s, v in lock_tr.items()}, fake_actions)
            rec["cross_game_from_LockPath"] = {"tokens": n2, **evaluate(sim, te, name)}
        out["games"][name] = rec
        for k, v in rec.items():
            if isinstance(v, dict) and "accuracy" in v:
                print(f"  {name:<11} {k:<26} all {v['accuracy']:.3f} (unch-baseline {v['unchanged_baseline']:.3f})"
                      f"  CHANGED {v['changed']['accuracy']:.3f} on {v['changed']['n']:>4} "
                      f"({100*v['changed']['share']:.1f}%)  balanced {v['balanced']:.3f}", flush=True)

    now = {k: v for k, v in sim.state_dict().items() if "memory" not in k}
    out["weights_unchanged"] = all(torch.equal(w0[k], now[k]) for k in w0)
    nat = {g: r["native"]["accuracy"] for g, r in out["games"].items() if "native" in r}
    cl = {g: r["cross_level"]["accuracy"] for g, r in out["games"].items() if "cross_level" in r}
    out["verdict"] = {
        "weights_unchanged": bool(out["weights_unchanged"]),
        "native_min": min(nat.values()) if nat else None,
        "native_all_ge_0.7": bool(nat and min(nat.values()) >= 0.7),
        "native_beats_unchanged_everywhere": bool(all(
            out["games"][g]["native"]["accuracy"] > out["games"][g]["native"]["unchanged_baseline"] for g in nat)),
        "native_changed_accuracy": {g: out["games"][g]["native"]["changed"]["accuracy"] for g in nat},
        "native_balanced": {g: out["games"][g]["native"]["balanced"] for g in nat},
        "cross_level_ratio": {g: cl[g] / nat[g] for g in cl if nat.get(g)},
    }
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expV.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\nverdict:", json.dumps(out["verdict"], default=str), f"{out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
