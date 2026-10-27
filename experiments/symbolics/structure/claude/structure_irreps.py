"""Irreducible representations from data (docs/COORDINATES.md, part 6): decompose the regular representation of the
actions' group, computed from the episode-1 transition table alone, and use the smallest faithful block sum as a
matrix-valued address. One method for commuting and non-commuting actions.
    python -m structure.claude.structure_irreps      (from experiments/symbolics)
"""
import json
import time

import numpy as np

from structure.claude.structure_transfer import SEEDS, a5_cayley, cycle, episode1, episode2, structure, torus


def bfs(TT):
    """Each node's parent and action on a breadth-first tree from the start node (id 0, the identity)."""
    parent, order = {0: None}, [0]
    for x in order:
        for a in range(TT.shape[1]):
            y = int(TT[x, a])
            if y not in parent:
                parent[y] = (x, a)
                order.append(y)
    return parent, order


def irreps(TT, rng):
    n, A = TT.shape
    parent, order = bfs(TT)
    words = {0: []}
    for y in order[1:]:
        x, a = parent[y]
        words[y] = words[x] + [a]
    M = np.empty((n, n), int)                             # M[x, y] = x.y: follow y's word from x
    for y in range(n):
        col = np.arange(n)
        for a in words[y]:
            col = TT[col, a]
        M[:, y] = col
    Z = np.zeros((n, n))
    for h, z in enumerate(rng.normal(size=n)):            # left translations commute with the (right) actions
        L = np.zeros((n, n))
        L[np.arange(n), M[h]] = 1                         # (L_h f)(y) = f(h.y)
        Z += z * (L + L.T)
    vals, vecs = np.linalg.eigh(Z)
    cuts = np.nonzero(np.diff(vals) > 1e-6 * max(1.0, np.abs(vals).max()))[0] + 1
    R = []
    for a in range(A):
        P = np.zeros((n, n))
        P[np.arange(n), TT[:, a]] = 1                     # (R_a f)(y) = f(y.a)
        R.append(P)
    reps = []
    for idx in np.split(np.arange(n), cuts):
        U = vecs[:, idx]
        As = [U.T @ P @ U for P in R]
        resid = max(float(np.linalg.norm(P @ U - U @ Aa)) for P, Aa in zip(R, As))
        mats = np.empty((n, len(idx), len(idx)))
        mats[0] = np.eye(len(idx))
        for y in order[1:]:
            x, a = parent[y]
            mats[y] = mats[x] @ As[a]
        chi = np.trace(mats, axis1=1, axis2=2)
        keys = {tuple(np.round(m * 1e4).astype(int).ravel()) for m in mats}
        reps.append(dict(d=len(idx), As=As, mats=mats, chi=chi, norm=float(np.mean(chi ** 2)), resid=resid,
                         distinct=len(keys)))
    classes = {}
    for r in reps:                                        # equivalent representations share a character
        classes.setdefault(tuple(np.round(r["chi"], 5)), []).append(r)
    return reps, list(classes.values())


def choose(classes, n):
    """Greedy block sum: smallest dimension first, then most distinct addresses, until every node is distinct."""
    chosen, addr = [], [()] * n
    for cls in sorted(classes, key=lambda c: (c[0]["d"], -c[0]["distinct"])):
        r = cls[0]
        new = [addr[x] + tuple(np.round(r["mats"][x] * 1e4).astype(int).ravel()) for x in range(n)]
        if len(set(new)) > len(set(addr)):
            chosen.append(r)
            addr = new
        if len(set(addr)) == n:
            break
    return chosen, len(set(addr))


def prune(chosen, n):
    """Exploratory (after R2 failed): drop any chosen block, earliest first, whose removal keeps addresses distinct."""
    keep = list(chosen)
    for r in list(chosen):
        rest = [q for q in keep if q is not r]
        keys = {tuple(np.concatenate([np.round(q["mats"][x] * 1e4).astype(int).ravel() for q in rest])) for x in range(n)} if rest else {()}
        if len(keys) == n:
            keep = rest
    return keep


def rotation_angle(m):
    return float(np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1))))


def matrix_model(chosen, A):
    key = lambda ad: tuple(np.concatenate([np.round(B * 1e4).astype(int).ravel() for B in ad]))
    return dict(start=[np.eye(r["d"]) for r in chosen], key=key, used="irreducible representations",
                size=sum(r["d"] ** 2 for r in chosen) * A,
                move=lambda ad, a: [B @ r["As"][a] for B, r in zip(ad, chosen)])


def main():
    t0 = time.time()
    out = {}
    for name, T in (("E1 torus 6x10", torus()), ("E2 cycle 60, +-1 +-7", cycle()), ("E3 A5 Cayley", a5_cayley())):
        out[name] = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            TT, _ = episode1(T, rng)
            st = structure(TT, rng)                       # part 5's draws, so episode 2 repeats part 5's walk
            reps, classes = irreps(TT, np.random.default_rng(1000 + seed))
            n, A = TT.shape
            chosen, distinct = choose(classes, n)
            abelian = all((c[0]["d"], round(c[0]["norm"])) in ((1, 1), (2, 2)) for c in classes)
            rec = dict(subspace_dims=sorted(r["d"] for r in reps), classes=len(classes),
                       class_dims_norms=sorted((c[0]["d"], round(c[0]["norm"], 6), len(c)) for c in classes),
                       max_residual=max(r["resid"] for r in reps), chosen_dims=[r["d"] for r in chosen],
                       distinct_addresses=distinct, abelian_readout=abelian,
                       commutator_test_abelian=st["noncommuting_pairs"] == 0)
            if name.startswith("E3"):
                rot = {}
                for k, c in enumerate(cl for cl in classes if cl[0]["d"] == 3):
                    Aa, Ab = c[0]["As"][0], c[0]["As"][1]
                    rot[f"3-dim irreducible {k + 1}"] = dict(det_a=float(np.linalg.det(Aa)), det_b=float(np.linalg.det(Ab)),
                                                             angle_a=rotation_angle(Aa), angle_b=rotation_angle(Ab),
                                                             angle_ab=rotation_angle(Aa @ Ab))
                rec["rotations"] = rot
            pruned = prune(chosen, n) if distinct == n else chosen
            rec["exploratory: pruned dims"] = [r["d"] for r in pruned]
            rec["exploratory: pruned numbers"] = sum(r["d"] ** 2 for r in pruned) * A
            if distinct == n:
                ep2 = episode2(T, matrix_model(chosen, A), rng)
                rec["episode2"] = ep2
                rec["transferred_numbers"] = sum(r["d"] ** 2 for r in chosen) * A
            out[name].append(rec)
            dims = rec["class_dims_norms"]
            summary = {}
            for d, nm, mult in dims:
                summary[(d, nm, mult)] = summary.get((d, nm, mult), 0) + 1
            print(f"{name:22s} seed {seed}: subspaces {len(reps)}, irreducibles {len(classes)} "
                  f"[(dim, norm, copies): count] {dict(summary)}, residual {rec['max_residual']:.1e} | chosen dims "
                  f"{rec['chosen_dims']}, distinct {distinct} (exploratory pruned: {rec['exploratory: pruned dims']}, "
                  f"{rec['exploratory: pruned numbers']} numbers) | abelian read-out {abelian} (commutator test "
                  f"{rec['commutator_test_abelian']})", flush=True)
            if "rotations" in rec:
                for k, v in rec["rotations"].items():
                    print(f"      {k}: det {v['det_a']:+.3f} {v['det_b']:+.3f}; a {v['angle_a']:.1f} deg, b {v['angle_b']:.1f} deg, "
                          f"a.b {v['angle_ab']:.1f} deg", flush=True)
            if "episode2" in rec:
                e = rec["episode2"]
                print(f"      episode 2: {rec['transferred_numbers']} numbers transferred; coordinate memory {e['coord_accuracy']:.3f} "
                      f"(revisits {e['coord_accuracy_on_predictable']:.3f} of {e['predictable']}), edge memory {e['edge_accuracy']:.3f}",
                      flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/structure_irreps.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
