"""Learning the structure from repeated colours (docs/COORDINATES.md, part 8): torus hypotheses (pairing of the
actions into inverse pairs, and the two periods) compete with structure-free models by the prequential likelihood of
the colour sequence. A hypothesis predicts: same cell, same colour.
    python -m structure.claude.structure_aliased      (from experiments/symbolics)
"""
import json
import time

import numpy as np

from structure.claude.structure_transfer import cycle, torus

L, EPS, SEEDS, COLOURS, PMAX = 3000, 0.01, 3, (2, 3, 6, 60), 15
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def previous(code):
    """Index of the previous step with the same code (-1 if none)."""
    order = np.lexsort((np.arange(len(code)), code))
    same = code[order[1:]] == code[order[:-1]]
    prev = -np.ones(len(code), int)
    prev[order[1:][same]] = order[:-1][same]
    return prev


def cell_logp(code, col, C):
    prev = previous(code)
    hit = np.where(prev >= 0, col == col[np.maximum(prev, 0)], False)
    return np.where(prev < 0, np.log(1 / C), np.log((1 - EPS) * hit + EPS / C))


def models(actions, col, C):
    names, rows = [], []
    for k, ((a1, a2), (a3, a4)) in enumerate(PAIRINGS):
        dx = np.concatenate([[0], np.cumsum((actions == a1).astype(int) - (actions == a2))])
        dy = np.concatenate([[0], np.cumsum((actions == a3).astype(int) - (actions == a4))])
        for h in range(1, PMAX + 1):
            for w in range(1, PMAX + 1):
                names.append(f"torus {h}x{w}, pairs {PAIRINGS[k]}")
                rows.append(cell_logp((dx % h) * w + dy % w, col, C))
    names.append("null")
    rows.append(np.full(len(col), np.log(1 / C)))
    names.append("edge memory")
    key = np.concatenate([[-1], col[:-1] * 4 + actions])     # (previous colour, action) -> this colour
    key[0] = -1 - 10 ** 6                                     # the first observation has no key
    rows.append(cell_logp(key, col, C))
    return names, np.cumsum(np.array(rows), axis=1)


def main():
    t0 = time.time()
    out = {}
    for env, T, true in (("E1 torus 6x10", torus(), f"torus 6x10, pairs {PAIRINGS[0]}"),
                         ("E2 cycle 60, +-1 +-7", cycle(), None)):
        out[env] = {}
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            actions = rng.integers(4, size=L)
            x = [int(rng.integers(60))]
            for a in actions:
                x.append(int(T[x[-1], a]))
            x = np.array(x)
            for C in COLOURS:
                colour = np.random.default_rng(100 * seed + C).permutation(60) if C == 60 else \
                    np.random.default_rng(100 * seed + C).integers(0, C, 60)
                names, cum = models(actions, colour[x], C)
                final = cum[:, -1]
                order = np.argsort(-final)
                rec = dict(winner=names[order[0]], runner_up=names[order[1]], margin=float(final[order[0]] - final[order[1]]))
                if true:
                    ti = names.index(true)
                    others = np.delete(cum, ti, axis=0)
                    best = others.max(0)
                    wins = cum[ti] > best
                    lost = np.nonzero(~wins)[0]
                    rec["T*"] = int(lost[-1] + 1) if wins[-1] else None
                    rec["strongest rival at T*-1"] = names[np.delete(np.arange(len(names)), ti)[np.argmax(others[:, max(rec['T*'] - 1, 0)])]] if rec["T*"] else None
                    rec["true wins at end"] = bool(wins[-1])
                out[env][f"seed {seed}, C {C}"] = rec
                extra = f" | T* {rec['T*']} (last rival {rec['strongest rival at T*-1']})" if true else ""
                print(f"{env:22s} seed {seed} C {C:2d}: winner {rec['winner']:34s} by {rec['margin']:8.1f} over {rec['runner_up']}{extra}",
                      flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/structure_aliased.json", "w"), indent=1)


if __name__ == "__main__":
    main()
