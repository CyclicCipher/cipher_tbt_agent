"""A stronger completion engine (docs/COORDINATES.md, part 7): associativity in all four patterns plus Sudoku-style
Latin propagation. Arm A checks that no verdict changes; arm B measures the code of the true hypothesis against the
information-theoretic minimum, with checkpoints every 2% and an ablation of the two rule families.
    python -m structure.claude.structure_engine      (from experiments/symbolics)
"""
import json
import time

import numpy as np

from structure.claude.structure_hypotheses import HYPOTHESES, N, complete, tasks
from structure.claude.structure_prequential import SPACE, TRUTH, checkpoints, log_prob, prequential

FULL = ("assoc4", "sudoku")
ARM_B = {"T1 cyclic": "abelian group", "T2 Z6 x Z10": "abelian group", "T3 A5": "group"}
ABLATION = {"base": (), "base + associativity": ("assoc4",), "base + Sudoku": ("sudoku",)}


def prequential_fine(T_obs, axioms, rng, rules, growth=1.02):
    """Prequential code with incremental completion (exact for a hypothesis that is never contradicted)."""
    obs = np.argwhere(T_obs >= 0)
    obs = obs[rng.permutation(len(obs))]
    full, _ = complete(-np.ones_like(T_obs), axioms, rules)
    total, contra, wrong, undetermined, crystal = 0.0, 0, 0, 0, None
    for s, e in zip(*(lambda c: (c[:-1], c[1:]))(checkpoints(len(obs), growth=growth))):
        r, c = obs[s:e, 0], obs[s:e, 1]
        v = T_obs[r, c]
        pred = full[r, c]
        wrong += int(np.sum((pred >= 0) & (pred != v)))
        undetermined += int(np.sum(pred < 0))
        total += float(log_prob(full, axioms, r, c, v).sum())
        full[r, c] = v
        if (full < 0).any():                              # a fully known table can't change: skip re-completion
            full, k = complete(full, axioms, rules)
            contra += k
        if crystal is None and not (full < 0).any():
            crystal = e
    return dict(code=-total, contradictions=contra, determined_wrong=wrong, undetermined_entries=undetermined,
                table_known_after=crystal)


def main():
    t0 = time.time()
    part3 = json.load(open("runs/structure_prequential.json"))
    obs_tables, rng = {}, np.random.default_rng(0)        # part 2's masks, drawn in the same order
    for frac in (0.5, 0.25):
        for name, (T, pi) in tasks(np.random.default_rng(7)).items():
            obs_tables[(name, frac)] = np.where(rng.random((N, N)) < frac, T, -1)
    out = {"arm A": {}, "arm B": {}}

    print("arm A: full engine, part 3's checkpoints, order seed 100", flush=True)
    for (name, frac), T_obs in obs_tables.items():
        new = {h: prequential(T_obs, ax, np.random.default_rng(100), FULL) for h, ax in SPACE.items()}
        old = {h: part3[f"{name} @ {frac}"]["per order"][h][0] for h in SPACE}
        rec = {"full engine": new, "part 3 engine": old}
        for part, space in (("part 3", HYPOTHESES), ("part 4", SPACE)):
            ranked = sorted(space, key=lambda h: -new[h])
            rec[part] = dict(winner=ranked[0], margin=new[ranked[0]] - new[ranked[1]], runner_up=ranked[1],
                             part3_engine_winner=max(space, key=lambda h: old[h]))
        out["arm A"][f"{name} @ {frac}"] = rec
        w3, w4 = rec["part 3"], rec["part 4"]
        print(f"   {name:16s} @ {frac}: part 3 {w3['winner']:14s} ({'unchanged' if w3['winner'] == w3['part3_engine_winner'] else 'CHANGED'}, "
              f"{'right' if w3['winner'] == TRUTH[name] else 'WRONG'}) margin {w3['margin']:7.1f} | part 4 {w4['winner']:14s} "
              f"({'unchanged' if w4['winner'] == w4['part3_engine_winner'] else 'CHANGED'}) | winner's code {-new[w3['winner']]:.0f} "
              f"(part 3 engine {-old[w3['winner']]:.0f})", flush=True)
    print(f"   arm A {time.time() - t0:.0f} s", flush=True)

    t1 = time.time()
    print("\narm B: true hypothesis, checkpoints every 2%, incremental completion", flush=True)
    for (name, frac), T_obs in obs_tables.items():
        if name not in ARM_B:
            continue
        ax = HYPOTHESES[ARM_B[name]]
        runs = {"full": [prequential_fine(T_obs, ax, np.random.default_rng(100 + o), FULL) for o in range(3)]}
        for label, rules in ABLATION.items():
            runs[label] = [prequential_fine(T_obs, ax, np.random.default_rng(100), rules)]
        out["arm B"][f"{name} @ {frac}"] = runs
        f = runs["full"]
        codes = ", ".join("%.1f" % r["code"] for r in f)
        print(f"   {name:16s} @ {frac}: full engine code {np.mean([r['code'] for r in f]):6.1f} nats "
              f"(orders {codes}), contradictions "
              f"{sum(r['contradictions'] for r in f)}, determined-wrong {sum(r['determined_wrong'] for r in f)}, "
              f"table known after {[r['table_known_after'] for r in f]} entries, undetermined {[r['undetermined_entries'] for r in f]}", flush=True)
        print("      order 100: " + ", ".join(f"{k} {runs[k][0]['code']:.1f} (known after {runs[k][0]['table_known_after']})"
                                          for k in ("base", "base + associativity", "base + Sudoku", "full")), flush=True)
    print(f"   arm B {time.time() - t1:.0f} s\n\ntotal {time.time() - t0:.0f} s")
    json.dump(out, open("runs/structure_engine.json", "w"), indent=1)


if __name__ == "__main__":
    main()
