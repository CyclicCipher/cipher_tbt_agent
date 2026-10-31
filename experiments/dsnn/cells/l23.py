"""Type A: the human layer 2/3 pyramidal cell, as Gidon et al. 2020 built it (LAB.md E0e; ModelDB 254217).

One concept: building this cell type. Their files: `_morph/171122005.hoc` (with its `geom()`), `biophys.hoc`,
`xor.hoc`, `dCaAP.mod`, `Traub.mod`, `syns.hoc`.
- **Morphology:** reconstruction 171122005, every section, every length and diameter × 1.25 (their `geom()`, for the
  imaging's 20% shrinkage), and their axon: 1000 µm long, 1 µm thick, at soma(1).
- **The cut, NEURON's:** apical sections into 1 + 2·int(L/30) segments, all others into 1 + 2·int(L/60) (`geom_nseg`,
  then `biophys`). Each segment is a compartment.
- **Passive:** Rm 37,000 Ω·cm², Cm 0.45 µF/cm², Ra 100 Ω·cm, E_L −74 mV everywhere.
- **Soma and axon:** Traub's Na⁺ (0.1 S/cm²) and K⁺ (0.045 S/cm²) (`traub.py`); their leak is the cell's leak.
- **The dCaAP** at one site, `dCaAP.mod` exactly (`cell.DCaAP`), w = their density × the site's segment area:
  - "uncoupled" (Fig. 3, the XOR): apic[76](0.214), 1.6 mA/cm²;
  - "coupled" (Fig. S9, the soma's output): apic[60](0.99), 4 mA/cm².
- **Synapses:** `cell.GIDON`.
- **Section lists** (where synapses go, `protocols.place`): "all_no_axon", "subdomain" (apic[60]'s subtree without
  apic[60]), "basal", "apical".
- **Rest:** V = −75 mV everywhere (their v_init), the gates at their steady state there.

**Sites** (`cell.groups`), each a point on a path, so a different cut keeps them in place: "zone" (the dCaAP's
compartment), "soma"; and E0's: "b1.d", "b2.d" (the compartments holding basal 1's and 2's tips), "b1.m" (60 µm
nearer the soma than b1.d's tip), "n4" (9 points along basal 1, distal first), "basal1", "basal2", "trunk", "tuft1",
"tuft2", "basal", "tuft" (the main paths: `paths`).

`reduced=True` gives the earlier reduction (LAB.md E0b: the five paths, every other subtree an equivalent cylinder),
at the corrected scale. It isn't yet re-validated against the full cell.
"""
from functools import lru_cache
import os

import numpy as np

from ..cell import GIDON, DCaAP, passive
from ..morphology import compartments, main_path, read_hoc
from ..traub import Traub

HOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gidon2020_171122005.hoc")
SCALE = 1.25                       # their geom()
SITES = dict(uncoupled=(("apic[76]", 0.214), 1.6), coupled=(("apic[60]", 0.99), 4.0))     # (point, mA/cm²)
CM, RM, RA, EL, V_INIT = 0.45, 37000.0, 100.0, -74.0, -75.0
TRAUB_NA, TRAUB_K = 0.1, 0.045     # S/cm²


@lru_cache(maxsize=1)
def tree():
    """The reconstruction at their scale, with their axon."""
    return read_hoc(HOC).scaled(SCALE).with_cylinder("axon", "soma", 1.0, 1000.0, 1.0)


def paths(t):
    """The main paths (each a list of sections from the soma outward): the trunk to apic[60], its two child subtrees'
    main paths, and the two largest basal trees' main paths."""
    trunk, s = [], "apic[60]"
    while s != "soma":
        trunk.append(s)
        s = t.parent[s][0]
    tufts = [main_path(t, c) for c in sorted(t.children["apic[60]"], key=t.subtree_area, reverse=True)]
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


def nseg(t, s):
    """NEURON's segment count in their model."""
    return 1 + 2 * int(t.length(s) / (30.0 if s.startswith("apic") else 60.0))


def l23(site="uncoupled", reduced=False, max_len=30.0):
    """The type A cell, its dCaAP at `site` ("uncoupled", "coupled", or None for none)."""
    t = tree()
    P = paths(t)
    if reduced:
        modelled = {s for p in P.values() for s in p}
        geo = compartments(t, modelled, RA, RM, CM, max_len)
    else:
        geo = compartments(t, set(t.points) - {"soma"}, RA, RM, CM, nseg_of=lambda s: nseg(t, s))
    cell = passive(geo, CM, RM, EL, "L2/3 (type A)" + (", reduced" if reduced else ""))
    secs = {name: [c for s in ss for c, _, _ in geo.where[s]] for name, ss in P.items()}
    g = dict(secs, basal=secs["basal1"] + secs["basal2"], tuft=secs["tuft1"] + secs["tuft2"], soma=[0])
    L1, L2 = (sum(t.length(s) for s in P[b]) for b in ("basal1", "basal2"))
    g["b1.d"] = [geo.at(*_point(t, P["basal1"], L1))]
    g["b1.m"] = [geo.at(*_point(t, P["basal1"], L1 - 60.0))]
    g["b2.d"] = [geo.at(*_point(t, P["basal2"], L2))]
    g["n4"] = [geo.at(*_point(t, P["basal1"], L1 * (8.5 - i) / 9)) for i in range(9)]
    g["axon"] = [c for c, _, _ in geo.where["axon"]]
    cell.groups = g
    cell.syn = GIDON
    na, k = np.zeros(cell.N), np.zeros(cell.N)
    for c in [0] + g["axon"]:
        na[c], k[c] = TRAUB_NA, TRAUB_K
    cell.traub = Traub(na, k)
    if site:
        (sec, x), dens = SITES[site]
        z = geo.at(sec, x)
        g["zone"] = [z]
        cell.dcaap = DCaAP(zone=z, w=dens * cell.area[z] * 10.0)          # mA/cm² × µm² × 10 = pA
    cell.rest = {"V": np.full(cell.N, V_INIT)}
    cell.tree, cell.where = t, geo.where
    sections = [s for s in t.points if s != "axon"]
    sub = [s for s in t.subtree("apic[60]") if s != "apic[60]"]
    cell.seclists = dict(all_no_axon=sections, subdomain=sub, basal=[s for s in t.points if s.startswith("dend")],
                         apical=[s for s in t.points if s.startswith("apic")])
    return cell
