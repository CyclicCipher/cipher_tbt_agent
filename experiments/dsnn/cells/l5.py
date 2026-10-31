"""Type B: the layer 5-like cell (LAB.md E0b): Bahl et al. 2012's reduced model 2 (ModelDB 146026), ported as published.

One concept: building this cell type. The source files: `reduced_geometry_20com.hoc` (the geometry),
`reduced_model.hoc` (which channels go where, the distance rules) and `init_models_with_ca/init_model2.hoc` (model 2's
fitted values). Channels: `channels.py`.

**Compartments:** NEURON's: each section is cut into nseg equal segments (the source's nseg, or finer with `max_len`).
A segment is a cylinder with the section's diameter at its centre. Between two nodes the axial resistance is the sum of
the two half-segments' resistances. A child attached at its parent's 0 or 1 end also crosses the parent's end
half-segment.

**Distances** (for the channel gradients) are NEURON's `distance()` from soma(0), as the source sets them.

**The rest state** is not v_init: the source starts at v_init = −65 mV. Here the cell is run silent for 20 s from
there, and its final state is the rest state every protocol starts from (`cell.rest`). Check 3 starts at −65 mV, as
the source's own example does (`at_voltage`).
"""
from dataclasses import replace
from functools import lru_cache
import math

import numpy as np

from .. import channels as ch
from ..cell import passive
from ..morphology import Geometry

SOMA_AREA, BASAL_AREA, APICAL_AREA, TUFT_AREA = 1682.96028429, 7060.90626796, 9312.38528764, 9434.24861189
D_SOMA = math.sqrt(SOMA_AREA / math.pi)
# (name, parent, x on parent, L, d at 0, d at 1, nseg, Ra, apical tree?)
SECTIONS = (
    ("soma", None, 0.0, D_SOMA, D_SOMA, D_SOMA, 1, 82.0, False),
    ("basal", "soma", 0.5, 257.0, BASAL_AREA / math.pi / 257.0, BASAL_AREA / math.pi / 257.0, 1, 734.0, False),
    ("apical", "soma", 1.0, 500.0, APICAL_AREA / math.pi / 500.0, APICAL_AREA / math.pi / 500.0, 5, 382.22414867,
     True),
    ("tuft", "apical", 1.0, 499.0, TUFT_AREA / math.pi / 499.0, TUFT_AREA / math.pi / 499.0, 2, 527.0, True),
    ("hillock", "soma", 0.0, 20.0, 3.5, 2.0, 5, 82.0, False),
    ("iseg", "hillock", 1.0, 25.0, 2.0, 1.5, 5, 82.0, False),
    ("axon", "iseg", 1.0, 500.0, 1.5, 1.5, 1, 82.0, False),
)
E_PAS = -80.398657
RM_AXOSOMATIC, CM_AXOSOMATIC, SPINEFACTOR = 20587.734598, 2.230408, 0.784797
SOMA_G = dict(nat=236.616175, kfast=67.197508, kslow=475.820646, nap=1.443953, km=10.459916)
TUFT_G = dict(ih=16.194815, nat=47.817841, sca=0.45423528, kca=6.15058501)
BASAL_IH, HILLOCK_NAT, ISEG_NAT, ISEG_VSHIFT2 = 11.039583, 9512.289205, 13326.766938, -10.612583
DECAY_KFAST, DECAY_KSLOW = 20.075497, 37.711817
VSHIFT_SCA = 7.19485311


def _geometry(max_len):
    names, parent, R, area, dist, length, sec_of, where = [], [], [], [], [], [], [], {}
    for name, par, px, L, d0, d1, nseg, Ra, _ in SECTIONS:
        n = max(nseg, math.ceil(L / max_len - 1e-9)) if max_len else nseg
        d = [d0 + (d1 - d0) * (j + 0.5) / n for j in range(n)]
        half = [4.0 * Ra * (L / n / 2) / (math.pi * dj * dj) * 0.01 for dj in d]      # MΩ
        if par is None:
            start, r_up = 0.0, 0.0
        else:
            segs = where[par]
            pn = len(segs)
            if px in (0.0, 1.0):                         # the parent's end node: cross its end half-segment
                k = 0 if px == 0.0 else pn - 1
                p_comp, r_up = segs[k][0], segs[k][1]
            else:
                k = min(int(px * pn), pn - 1)
                p_comp, r_up = segs[k][0], 0.0
            start = _dist0(par, px)
        where[name] = []
        for j in range(n):
            c = len(names)
            names.append("%s.%d" % (name, j))
            parent.append(-1 if par is None else (p_comp if j == 0 else c - 1))
            R.append(0.0 if par is None and j == 0 else (r_up + half[0] if j == 0 else half[j - 1] + half[j]))
            area.append(math.pi * d[j] * L / n)
            dist.append(start + (j + 0.5) / n * L)
            length.append(0.0 if name == "soma" else L / n)
            sec_of.append(name)
            where[name].append((c, half[j]))
    geo = Geometry(names, np.array(parent), np.array(R), np.array(area), np.array(dist),
                   np.array(length), {s: [(c, 0.0, 0.0) for c, _ in w] for s, w in where.items()})
    return geo, np.array(sec_of)


def _dist0(sec, x):
    """NEURON's distance() from soma(0) to point x of a section."""
    for name, par, px, L, *_ in SECTIONS:
        if name == sec:
            return x * L if par is None else _dist0(par, px) + x * L
    raise KeyError(sec)


@lru_cache(maxsize=4)
def l5(max_len=None):
    """The type B cell, in its rest state. max_len: cut each section into at least ceil(L / max_len) segments."""
    geo, sec = _geometry(max_len)
    apical_tree = np.isin(sec, ("apical", "tuft"))
    cell = passive(geo, None, None, E_PAS, "L5-like (type B)",
                   Cm_of=lambda i: CM_AXOSOMATIC * (SPINEFACTOR if apical_tree[i] else 1.0),
                   Rm_of=lambda i: RM_AXOSOMATIC / (SPINEFACTOR if apical_tree[i] else 1.0))
    N, dist = cell.N, cell.dist
    dens = {k: np.zeros(N) for k in ch.NAMES}
    vshift2 = np.zeros(N)
    soma, basal, apical, tuft = (sec == s for s in ("soma", "basal", "apical", "tuft"))
    for k, v in SOMA_G.items():
        dens[k][soma] = v
    dens["ih"][basal] = BASAL_IH
    for k, v in TUFT_G.items():
        dens[k][tuft] = v
    d_tuft0 = _dist0("tuft", 0.0)
    mnat = (TUFT_G["nat"] - SOMA_G["nat"]) / d_tuft0
    mih = TUFT_G["ih"] / d_tuft0
    dens["nat"][apical] = mnat * dist[apical] + SOMA_G["nat"]
    dens["ih"][apical] = mih * dist[apical]
    dens["kfast"][apical_tree] = SOMA_G["kfast"] * np.exp(-dist[apical_tree] / DECAY_KFAST)
    dens["kslow"][apical_tree] = SOMA_G["kslow"] * np.exp(-dist[apical_tree] / DECAY_KSLOW)
    dens["nat"][sec == "hillock"] = HILLOCK_NAT
    dens["nat"][sec == "iseg"] = ISEG_NAT
    vshift2[sec == "iseg"] = ISEG_VSHIFT2
    cell.hh = ch.HH(dens=dens, vshift2_nat=vshift2, vshift_sca=VSHIFT_SCA)
    idx = lambda s: [i for i in range(N) if sec[i] == s]
    cell.groups = {s: idx(s) for s in ("soma", "basal", "apical", "tuft", "hillock", "iseg", "axon")}
    node = lambda s, x: idx(s)[min(int(x * len(idx(s))), len(idx(s)) - 1)]       # NEURON's node for sec(x)
    cell.groups["apical(0.5)"] = [node("apical", 0.5)]
    cell.groups["tuft(0.5)"] = [node("tuft", 0.5)]
    cell.rest = at_voltage(cell, -65.0)
    cell.rest = _settle(cell, 20000.0)
    return cell


def at_voltage(cell, v):
    """NEURON's finitialize(v): every compartment at v, every gate at its steady state there."""
    X = np.zeros((13, cell.N))
    P = cell.hh.params()
    for i in range(cell.N):
        ch.init(X, i, v, cell.hh.vshift2_nat[i], P)
    return {"V": np.full(cell.N, float(v)), "X": X}


def _settle(cell, T):
    from ..sim import Stim, simulate
    r = simulate([Stim()], T, cell, final=True)
    if r.spikes[0][0]:
        raise RuntimeError("type B fires at rest")
    return r.final[0][0]


def from_v_init(cell, v=-65.0):
    """The cell started as the source's examples start it."""
    return replace(cell, rest=at_voltage(cell, v))
