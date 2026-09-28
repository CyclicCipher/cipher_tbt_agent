"""expU (E41, DESIGN §18.1) -- in-context learning in a WRITTEN block. No network is trained; none is fitted.

§18's frame says the block is an interpreter and knowledge is a program in the context. E28 already built that and
nobody framed the consequence: its rules live in MEMORY TOKENS, so changing the tokens changes the world model
while every weight stays where it was. This measures how far that goes.

One `WrittenSim` is compiled ONCE over the full radius-1 neighbourhood, so the layout and all weights are fixed;
the arms differ only in what is in the context:
  store        the counted, sleep-swept LocalRule table (E28's baseline; dropped cells become wildcards)
  substitution another action's counted rules in the same weights -- the weights are not the world model
  icl(k)       k RAW observed transitions: whole windows, no counting, no majority vote, no sleep, no store
  empty        no entry tokens at all
Pass: icl rises monotonically in k and reaches >= 0.9 of `store` at some k <= the store's entry count, and
substitution is exact. Refute: icl flat in k, or substitution wrong.

Reported separately, because it is where the generalising happens: accuracy on held-out windows that appear
VERBATIM among the demonstrations, against those that do not (the nearest-key default -- §21.10's first
NOT-DESIGNED item, and E28's "a free generalisation the price does not account for").
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))

import brainbuilder as bb                                              # noqa: E402
import e28 as E28                                                      # noqa: E402


def all_ones_rule(r=1):
    n = (2 * r + 1) ** 2
    return SimpleNamespace(r=r, mask=np.ones(n, dtype=bool), table={})


def tokens_from(sim, rules, actions):
    """Rebuild `sim.memory` from an arbitrary {action: rule-like} -- the same construction `WrittenSim.__init__`
    uses, so the tokens are comparable and every weight is untouched."""
    d = sim.memory.shape[1]
    mem = []
    for ai, a in enumerate(actions):
        r = rules.get(a)
        if r is None:
            continue
        side = 2 * r.r + 1
        cells = [(idx // side - r.r, idx % side - r.r) for idx in np.flatnonzero(r.mask)]
        for key, counts in r.table.items():
            colours = np.frombuffer(key, dtype=np.int16)
            new = max(counts, key=counts.get)
            v = torch.zeros(d)
            ok = True
            for o, col in zip(cells, colours):
                col = sim.BORDER if col == E28.BORDER else int(col)
                if o == (0, 0):
                    sl = sim.T
                elif o in sim.offs:
                    sl = sim.N[sim.offs.index(o)]
                else:
                    ok = False
                    break
                v[sl.start + col] = 1.0
            if not ok:
                continue
            v[sim.A.start + ai] = 1.0
            v[sim.F] = 1.0
            v[sim.OUT.start + int(new)] = 1.0
            mem.append(v)
    border = torch.zeros(d); border[sim.S] = 1.0; border[sim.T.start + sim.BORDER] = 1.0
    zero = torch.zeros(d); zero[sim.Z] = 1.0
    sim.memory = torch.stack([border, zero] + mem) if mem else torch.stack([border, zero])
    sim.n_mem = sim.memory.shape[0]
    return len(mem)


def raw_rule(full_items, r=1):
    """k raw observations as a rule-like: whole windows, one entry each, no counting or majority."""
    n = (2 * r + 1) ** 2
    tab = {}
    for key, counts in full_items:
        tab[key] = dict(counts)
    return SimpleNamespace(r=r, mask=np.ones(n, dtype=bool), table=tab)


def main():
    t0 = time.time()
    seed = 0
    rules, actions, results, AG = bb.play_rules(seed=seed, play_levels=2, budget=150, sleep=True)
    H, W = 8, 11
    out = {"experiment": "E41", "seed": seed, "play": [(r["level"], r["actions"]) for r in results],
           "actions": [a.name for a in actions], "arms": {}}
    print(json.dumps({k: out[k] for k in ("seed", "play", "actions")}), flush=True)

    # one block, full r=1 neighbourhood, so the layout and every weight are fixed across arms
    scaffold = {a: all_ones_rule(1) for a in actions}
    sim = E28.WrittenSim(scaffold, actions, H, W)
    w0 = {k: v.clone() for k, v in sim.state_dict().items() if "memory" not in k}
    out["offsets"] = [list(o) for o in sim.offs]
    out["d"] = int(sim.memory.shape[1])

    # held-out transitions: whole-window evidence the rules collected, split per action
    rng = np.random.default_rng(seed)
    train_items, test_items = {}, {}
    for a in actions:
        items = sorted(rules[a].full.items())
        rng.shuffle(items)
        cut = max(1, int(0.7 * len(items)))
        train_items[a], test_items[a] = items[:cut], items[cut:]
    out["evidence"] = {a.name: {"train": len(train_items[a]), "test": len(test_items[a])} for a in actions}
    print("evidence:", json.dumps(out["evidence"]), flush=True)

    def evaluate(tag):
        """For every held-out whole-window observation, does the block's lookup return the observed colour?"""
        right = wrong = 0
        verbatim_r = verbatim_w = novel_r = novel_w = 0
        keys_in_ctx = {a: set(k for k, _ in (ctx_items.get(a) or [])) for a in actions}
        for ai, a in enumerate(actions):
            for key, counts in test_items[a]:
                truth = max(counts, key=counts.get)
                cols = np.frombuffer(key, dtype=np.int16)
                q = torch.zeros(sim.memory.shape[1])
                side = 3
                for idx, col in enumerate(cols):
                    o = (idx // side - 1, idx % side - 1)
                    c = sim.BORDER if col == E28.BORDER else int(col)
                    if o == (0, 0):
                        q[sim.T.start + c] = 1.0
                    elif o in sim.offs:
                        q[sim.N[sim.offs.index(o)].start + c] = 1.0
                # the block's own scoring: the action dim carries weight g = nO + 2, so ONE action match
                # outweighs a full match on every cell (e28.py's `g`). A plain dot product would let an entry
                # for the wrong action win on cell agreement alone.
                q[sim.A.start + ai] = float(len(sim.offs) + 2)
                ent = sim.memory[2:]
                if len(ent) == 0:
                    wrong += 1; novel_w += 1; continue
                score = ent @ q
                got = int(torch.argmax(ent[int(torch.argmax(score))][sim.OUT]).item())
                hit = (got == int(truth))
                right += hit; wrong += (not hit)
                if key in keys_in_ctx[a]:
                    verbatim_r += hit; verbatim_w += (not hit)
                else:
                    novel_r += hit; novel_w += (not hit)
        n = right + wrong
        return {"n": n, "accuracy": right / max(1, n),
                "verbatim": {"n": verbatim_r + verbatim_w,
                             "accuracy": verbatim_r / max(1, verbatim_r + verbatim_w)},
                "novel": {"n": novel_r + novel_w, "accuracy": novel_r / max(1, novel_r + novel_w)}}

    # ---- arm: store (E28's counted, swept table) ----
    ctx_items = {a: sorted(rules[a].table.items()) for a in actions}
    n_tok = tokens_from(sim, rules, actions)
    out["arms"]["store"] = {"tokens": n_tok, **evaluate("store")}
    print(f"  store        tokens {n_tok:>5}  acc {out['arms']['store']['accuracy']:.3f}", flush=True)
    store_acc = out["arms"]["store"]["accuracy"]
    store_entries = n_tok

    # ---- arm: empty ----
    ctx_items = {a: [] for a in actions}
    tokens_from(sim, {}, actions)
    out["arms"]["empty"] = {"tokens": 0, **evaluate("empty")}
    print(f"  empty        tokens     0  acc {out['arms']['empty']['accuracy']:.3f}", flush=True)

    # ---- arm: substitution (another action's rules in the same weights) ----
    swapped = {actions[i]: rules[actions[(i + 1) % len(actions)]] for i in range(len(actions))}
    ctx_items = {a: sorted(swapped[a].table.items()) for a in actions}
    tokens_from(sim, swapped, actions)
    out["arms"]["substitution"] = {"tokens": int(sim.n_mem - 2), **evaluate("substitution")}
    print(f"  substitution tokens {sim.n_mem-2:>5}  acc {out['arms']['substitution']['accuracy']:.3f} "
          f"(expected LOW: the tokens are another action's)", flush=True)

    # ---- arm: icl(k) -- raw demonstrations, no counting anywhere ----
    out["arms"]["icl"] = []
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for k in ks:
        demos = {a: train_items[a][:k] for a in actions}
        ctx_items = demos
        rl = {a: raw_rule(demos[a]) for a in actions}
        n_tok = tokens_from(sim, rl, actions)
        rec = {"k_per_action": k, "tokens": n_tok, **evaluate(f"icl{k}")}
        out["arms"]["icl"].append(rec)
        print(f"  icl k={k:<4}    tokens {n_tok:>5}  acc {rec['accuracy']:.3f}  "
              f"(verbatim {rec['verbatim']['accuracy']:.3f} on {rec['verbatim']['n']}, "
              f"novel {rec['novel']['accuracy']:.3f} on {rec['novel']['n']})", flush=True)
        if n_tok >= sum(len(v) for v in train_items.values()):
            break

    # weights untouched throughout?
    now = {k: v for k, v in sim.state_dict().items() if "memory" not in k}
    out["weights_unchanged"] = all(torch.equal(w0[k], now[k]) for k in w0)
    best = max(out["arms"]["icl"], key=lambda r: r["accuracy"])
    out["verdict"] = {
        "weights_unchanged": bool(out["weights_unchanged"]),
        "store_accuracy": store_acc, "best_icl": best["accuracy"],
        "icl_reaches_90pct_of_store": bool(best["accuracy"] >= 0.9 * store_acc),
        "k_at_90pct": next((r["k_per_action"] for r in out["arms"]["icl"]
                            if r["accuracy"] >= 0.9 * store_acc), None),
        "store_entries": store_entries,
        "monotone_in_k": all(b["accuracy"] >= a["accuracy"] - 1e-9
                             for a, b in zip(out["arms"]["icl"], out["arms"]["icl"][1:])),
    }
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expU.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\nverdict:", json.dumps(out["verdict"], default=str), f"{out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
