"""Type A: the human layer 2/3 pyramidal cell (LAB.md E0b), reduced by rule from Gidon et al. 2020's reconstruction.

One concept: building this cell type.
- **The source:** cell 171122005 (ModelDB 254217, `data/gidon2020_171122005.hoc`), with Gidon's passive constants
  (`biophys.hoc`) and dCaAP (`dCaAP.mod`, `xor.hoc`).
- **The rule:** model the trunk (the soma to Gidon's dCaAP site, apic[60](0.99)), the main paths of the site's two
  child subtrees (the tuft), and the main paths of the two largest basal trees. Every other subtree becomes an
  equivalent cylinder attached where it attaches (`morphology.compartments`; LAB.md E0b, amendment 2).
- **The soma and synapses:** E0's AdEx and AMPA/NMDA contacts.

**Sites** (`cell.groups`), each defined by a point on a path, not by a compartment's index, so a finer cut (check 6)
keeps them in place:
- "b1.d", "b2.d": the compartment that holds basal 1's (2's) tip; "b1.m": the one 60 µm nearer the soma than b1.d's tip;
- "zone": the dCaAP's compartment;
- "n4": 9 points evenly spaced along basal 1 (the centres of 9 equal pieces), distal first;
- "basal1", "basal2", "trunk", "tuft1", "tuft2", "basal", "tuft": every compartment of those paths.
"""
from functools import lru_cache
import os

import numpy as np

from ..cell import AdEx, DCaAP, passive
from ..morphology import compartments, main_path, read_hoc

HOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gidon2020_171122005.hoc")
SITE = ("apic[60]", 0.99)          # Gidon's coupled dCaAP site (xor.hoc)
CM, RM, RA, EL = 0.45, 37000.0, 100.0, -74.0
W = 2970.0                         # pA: Gidon's 4 mA/cm² over their site's segment (74 µm²)


@lru_cache(maxsize=1)
def tree():
    return read_hoc(HOC)


def paths(t):
    """The modelled paths, by the rule (each a list of sections from the soma outward)."""
    trunk, s = [], SITE[0]
    while s != "soma":
        trunk.append(s)
        s = t.parent[s][0]
    tufts = [main_path(t, c) for c in sorted(t.children[SITE[0]], key=t.subtree_area, reverse=True)]
    roots = sorted((c for c in t.children["soma"] if c.startswith("dend")), key=t.subtree_area, reverse=True)
    basals = [main_path(t, r) for r in roots[:2]]
    return dict(trunk=trunk[::-1], tuft1=tufts[0], tuft2=tufts[1], basal1=basals[0], basal2=basals[1])


def _point(t, path, s_um):
    """(section, x) of the point s_um along a path from its start."""
    for sec in path:
        L = t.length(sec)
        if s_um <= L + 1e-9:
            return sec, max(0.0, s_um) / L
        s_um -= L
    return path[-1], 1.0


def l23(max_len=30.0, full=False):
    """The type A cell. full=True: every section modelled (the reference for check 2), passive only."""
    t = tree()
    P = paths(t)
    modelled = set(t.points) - {"soma"} if full else {s for p in P.values() for s in p}
    geo = compartments(t, modelled, RA, RM, CM, max_len)
    cell = passive(geo, CM, RM, EL, "L2/3 (type A)" + (", full" if full else ""))
    g = {name: [c for s in secs for c, _, _ in geo.where[s]] for name, secs in P.items()}
    g["basal"], g["tuft"] = g["basal1"] + g["basal2"], g["tuft1"] + g["tuft2"]
    L1, L2 = (sum(t.length(s) for s in P[b]) for b in ("basal1", "basal2"))
    g["b1.d"] = [geo.at(*_point(t, P["basal1"], L1))]
    g["b1.m"] = [geo.at(*_point(t, P["basal1"], L1 - 60.0))]
    g["b2.d"] = [geo.at(*_point(t, P["basal2"], L2))]
    g["n4"] = [geo.at(*_point(t, P["basal1"], L1 * (8.5 - i) / 9)) for i in range(9)]
    g["zone"] = [geo.at(*SITE)]
    g["soma"] = [0]
    g["lumped"] = [c for s, w in geo.where.items() if s.endswith("~eq") for c, _, _ in w]
    cell.groups = g
    cell.rest = {"V": np.full(cell.N, EL)}
    if not full:
        cell.adex = AdEx()
        cell.dcaap = DCaAP(zone=g["zone"][0], w=W)
    return cell
