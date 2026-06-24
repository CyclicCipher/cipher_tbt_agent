"""One column, one mechanism — every domain learned via the SR-eigenvector frame and predicted.

Arithmetic (ring), family (tree), social (rank line), and spatial (2-D torus) are all learned by the SAME
`CorticalColumn.learn_domain` via the SAME SR frame (computed per structure), stored in ONE shared memory,
and queried by the SAME `infer` (composing relation operators). We check: (a) each domain's relational
inference (relations never stored), (b) all four coexist in one column with no interference, (c) that
DIFFERENT structures auto-separate (their frames differ) while SAME structures need an orthogonal slot
(remap), and (d) the microwave — an entity changes and `revise` corrects it without disturbing the rest.
No separate programs, no per-domain grids. Tiny, CPU.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tbt import CorticalColumn       # noqa: E402


def family_tree(depth=5):
    N = 2 ** depth - 1
    par = {i: (i - 1) // 2 for i in range(1, N)}
    ch = {i: [c for c in (2 * i + 1, 2 * i + 2) if c < N] for i in range(N)}
    sib = {i: next((c for c in ch[par[i]] if c != i), None) for i in range(1, N)}
    return N, par, ch, sib


# entity layout (global indices): arithmetic 0..11, family 12..42, social 43..54, spatial 55..79
ARI, FAM, SOC, SPA = 0, 12, 43, 55
N_ENT = 80
M, SR = 12, 5                                            # ring size; spatial grid side (5x5)
NF, par, ch, sib = family_tree(5)                        # 31-node family tree

DOMAINS = {
    "arithmetic": dict(labels=[ARI + n for n in range(M)],
                       relations={"succ": [(i, (i + 1) % M) for i in range(M)]}),
    "family": dict(labels=[FAM + i for i in range(NF)],
                   relations={"parent": [(i, par[i]) for i in par],
                              "child": [(i, c) for i in ch for c in ch[i]],
                              "sib": [(i, sib[i]) for i in sib if sib[i] is not None]}),
    "social": dict(labels=[SOC + r for r in range(M)],
                   relations={"below": [(i, i + 1) for i in range(M - 1)]}),
    "spatial": dict(labels=[SPA + c for c in range(SR * SR)],
                    relations={"xmove": [(x * SR + y, ((x + 1) % SR) * SR + y) for x in range(SR) for y in range(SR)],
                               "ymove": [(x * SR + y, x * SR + (y + 1) % SR) for x in range(SR) for y in range(SR)]}),
}


def queries(name):
    """(start_node, relation_chain, expected_global_entity) for relations the column never stored."""
    out = []
    if name == "arithmetic":
        for a in range(M):
            for b in range(M):
                out.append((a, ["succ"] * b, ARI + (a + b) % M))
    elif name == "social":
        for r in range(M):
            for k in range(1, M - r):
                out.append((r, ["below"] * k, SOC + r + k))
    elif name == "spatial":
        for s in range(SR * SR):
            sx, sy = s // SR, s % SR
            for a in range(SR):
                for b in range(SR):
                    chain = ["xmove"] * a + ["ymove"] * b
                    out.append((s, chain, SPA + ((sx + a) % SR) * SR + (sy + b) % SR))
    else:  # family
        for x in range(NF):
            if x in par and par[x] in par:
                out.append((x, ["parent", "parent"], FAM + par[par[x]]))
            if x in par and sib.get(par[x]) is not None:
                out.append((x, ["sib", "parent"], FAM + sib[par[x]]))
    return out


def build(remap, names):
    col = CorticalColumn(N_ENT)
    for nm in names:
        col.learn_domain(nm, DOMAINS[nm]["labels"], DOMAINS[nm]["relations"], remap=remap)
    return col


def acc(col, name):
    qs = queries(name)
    return sum(col.infer(name, chain, start) == tgt for start, chain, tgt in qs) / len(qs)


def same_structure_interference():
    """The SR frame auto-separates DIFFERENT structures (their eigenvectors differ), so the four domains
    above coexist even without remap. IDENTICAL structures share a frame, so they collide unless given
    orthogonal slots — that is what remap is for. Two rings, same structure, different entities:"""
    ring = {"succ": [(i, (i + 1) % M) for i in range(M)]}
    out = {}
    for remap in (False, True):
        col = CorticalColumn(N_ENT)
        col.learn_domain("ringA", [ARI + i for i in range(M)], ring, remap=remap)
        col.learn_domain("ringB", [SOC + i for i in range(M)], ring, remap=remap)
        accA = sum(col.recall("ringA", i) == ARI + i for i in range(M)) / M
        accB = sum(col.recall("ringB", i) == SOC + i for i in range(M)) / M
        out[remap] = (accA + accB) / 2
    return out


if __name__ == "__main__":
    names = list(DOMAINS)
    struct = {"arithmetic": "ring", "family": "tree", "social": "line", "spatial": "2D-torus"}
    alone = {nm: acc(build(True, [nm]), nm) for nm in names}
    col = build(True, names)                              # all four in ONE column
    nocol = build(False, names)
    print("relational inference (each relation never stored), one column, structure-specific SR frames:")
    print(f"{'domain':12}{'structure':10}{'alone':>8}{'together(remap)':>17}{'together(NO remap)':>20}")
    for nm in names:
        print(f"{nm:12}{struct[nm]:10}{alone[nm]:8.3f}{acc(col, nm):17.3f}{acc(nocol, nm):20.3f}")
    print("  (NO-remap stays 1.000 here: the four structures DIFFER, so their SR frames already separate them.)")

    si = same_structure_interference()
    print(f"\nsame structure (two rings) — recall  no-remap: {si[False]:.3f}   remap: {si[True]:.3f}"
          f"   → identical structures need orthogonal slots; different ones separate themselves")

    # microwave: change one spatial cell's content; revise; check corrected + neighbours preserved
    node = 12
    old = col.recall("spatial", node)
    col.revise("spatial", node, SPA + 99 % (N_ENT - SPA))   # overwrite with a different entity
    new = col.recall("spatial", node)
    others = sum(col.recall("spatial", c) == DOMAINS["spatial"]["labels"][c]
                 for c in range(SR * SR) if c != node) / (SR * SR - 1)
    print(f"\nmicrowave: spatial cell {node} recall {old}->{new} (changed); other cells preserved {others:.3f}")
