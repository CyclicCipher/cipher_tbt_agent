"""Computed coordinates as memory addresses (docs/COORDINATES.md, part 5), in the Tolman-Eichenbaum machine's
setting: walk a graph whose actions keep their meaning while the node labels are reshuffled each episode. Episode 1
computes the structure (commutator test, then characters or the regular representation); episode 2 path-integrates
an address and stores labels at it (a Spotlight-style cell memory), against an edge memory with no structure.
    python -m structure.claude.structure_transfer      (from experiments/symbolics)
"""
import itertools
import json
import time

import numpy as np

STEPS2, SEEDS, ALPHABET = 300, 3, 1000


# ---- environments: transition tables T[node, action] ----------------------------------------------------------------
def torus(h=6, w=10):
    x = np.arange(h * w)
    i, j = x // w, x % w
    return np.stack([((i + 1) % h) * w + j, ((i - 1) % h) * w + j, i * w + (j + 1) % w, i * w + (j - 1) % w], 1)


def cycle(n=60):
    x = np.arange(n)
    return np.stack([(x + 1) % n, (x - 1) % n, (x + 7) % n, (x - 7) % n], 1)


def a5_cayley():
    compose = lambda p, q: tuple(p[q[i]] for i in range(5))
    a, b = (1, 0, 3, 2, 4), (2, 1, 4, 3, 0)               # (1 2)(3 4) and (1 3 5), 0-indexed
    binv = tuple(np.argsort(b))
    gens, elems, frontier = [a, b, binv], [tuple(range(5))], [tuple(range(5))]
    while frontier:                                       # close the identity under the generators
        new = [compose(g, s) for g in frontier for s in gens]
        frontier = [g for g in dict.fromkeys(new) if g not in elems]
        elems += frontier
    assert len(elems) == 60
    index = {g: k for k, g in enumerate(elems)}
    return np.array([[index[compose(g, s)] for s in gens] for g in elems])


# ---- episode 1: the structure, from the learner's view (labels only) ------------------------------------------------
def episode1(T, rng):
    n, A = T.shape
    labels = rng.choice(ALPHABET, n, replace=False)
    x, seen, ids, TT = int(rng.integers(n)), np.zeros(T.shape, bool), {}, -np.ones(T.shape, int)
    ids[labels[x]] = 0
    steps = 0
    while not seen.all():
        a = int(rng.integers(A))
        y = T[x, a]
        ids.setdefault(labels[y], len(ids))
        TT[ids[labels[x]], a] = ids[labels[y]]
        seen[x, a], x, steps = True, y, steps + 1
    return TT, steps


def structure(TT, rng):
    n, A = TT.shape
    noncommuting = sum(int(np.sum(TT[TT[:, a], b] != TT[TT[:, b], a])) for a in range(A) for b in range(a + 1, A))
    S = np.zeros((n, n))
    for a, w in enumerate(rng.normal(size=A)):
        S[np.arange(n), TT[:, a]] += w                    # (Q_a f)(x) = f(TT[x, a])
    _, vecs = np.linalg.eig(S)
    chars = []
    for f in vecs.T:
        if abs(f[0]) < 1e-9:
            continue
        f = f / f[0]                                      # normalized at the start node (id 0)
        lam = [f[TT[:, a]] / f for a in range(A)]
        if np.max(np.abs(np.abs(f) - 1)) > 1e-6 or max(np.ptp(np.abs(l - l[0])) + np.max(np.abs(l - l[0])) for l in lam) > 1e-6:
            continue                                      # not a character: not a common eigenvector of the actions
        order = next(k for k in range(1, n + 1) if np.max(np.abs(f ** k - 1)) < 1e-6)
        phase = np.round(np.angle(f) * order / (2 * np.pi)).astype(int) % order
        step = [int(np.round(np.angle(l[0]) * order / (2 * np.pi))) % order for l in lam]
        chars.append((order, step, phase))
    chosen, addr = [], [()] * n
    for order, step, phase in sorted(chars, key=lambda c: -c[0]):
        new = [addr[x] + (phase[x],) for x in range(n)]
        if len(set(new)) > len(set(addr)):
            chosen.append((order, step))
            addr = new
        if len(set(addr)) == n:
            break
    return dict(noncommuting_pairs=noncommuting, characters=len(chars), chosen=chosen,
                distinct_addresses=len(set(addr)), injective=len(set(addr)) == n)


# ---- episode 2: coordinate memory against edge memory ---------------------------------------------------------------
def episode2(T, TT, st, rng):
    n, A = T.shape
    labels = rng.choice(ALPHABET, n, replace=False)
    use_chars = st["noncommuting_pairs"] == 0 and st["injective"]
    orders = np.array([o for o, _ in st["chosen"]]) if use_chars else None
    steps = np.array([s for _, s in st["chosen"]]).reshape(len(st["chosen"]), A) if use_chars else None
    x = int(rng.integers(n))
    addr = tuple(np.zeros(len(orders), int)) if use_chars else 0
    cells, edges, visited, traversed = {addr: labels[x]}, {}, {x}, set()
    rec = dict(coord=0, edge=0, predictable=0, coord_on_predictable=0, new_edge_old_node=0, edge_on_new_edge=0)
    for _ in range(STEPS2):
        a = int(rng.integers(A))
        y = T[x, a]
        addr = tuple((np.array(addr) + steps[:, a]) % orders) if use_chars else int(TT[addr, a])
        p_coord, p_edge = cells.get(addr), edges.get((labels[x], a))
        rec["coord"] += p_coord == labels[y]
        rec["edge"] += p_edge == labels[y]
        if y in visited:
            rec["predictable"] += 1
            rec["coord_on_predictable"] += p_coord == labels[y]
            if (x, a) not in traversed:
                rec["new_edge_old_node"] += 1
                rec["edge_on_new_edge"] += p_edge == labels[y]
        cells[addr], edges[(labels[x], a)] = labels[y], labels[y]
        visited.add(y)
        traversed.add((x, a))
        x = y
    rec = {k: int(v) for k, v in rec.items()}
    rec.update(coord_accuracy=rec["coord"] / STEPS2, edge_accuracy=rec["edge"] / STEPS2,
               coord_accuracy_on_predictable=rec["coord_on_predictable"] / max(rec["predictable"], 1),
               used="characters" if use_chars else "regular representation")
    return rec


def main():
    t0 = time.time()
    out = {}
    for name, T in (("E1 torus 6x10", torus()), ("E2 cycle 60, +-1 +-7", cycle()), ("E3 A5 Cayley", a5_cayley())):
        out[name] = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            TT, steps1 = episode1(T, rng)
            st = structure(TT, rng)
            ep2 = episode2(T, TT, st, rng)
            size = sum(len(s) for _, s in st["chosen"]) if ep2["used"] == "characters" else TT.size
            out[name].append(dict(episode1_steps=steps1, structure={k: v for k, v in st.items()},
                                  transferred_integers=size, episode2=ep2))
            print(f"{name:22s} seed {seed}: ep1 {steps1:5d} steps | non-commuting (node, pair) {st['noncommuting_pairs']:4d} | "
                  f"characters {st['characters']:2d}, chosen orders {[o for o, _ in st['chosen']]}, "
                  f"distinct addresses {st['distinct_addresses']:2d} | uses {ep2['used']}, {size} integers | "
                  f"ep2 coord {ep2['coord_accuracy']:.3f} edge {ep2['edge_accuracy']:.3f}, coord on revisits "
                  f"{ep2['coord_accuracy_on_predictable']:.3f} ({ep2['predictable']}), edge on new-edge revisits "
                  f"{ep2['edge_on_new_edge']}/{ep2['new_edge_old_node']}", flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/structure_transfer.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
