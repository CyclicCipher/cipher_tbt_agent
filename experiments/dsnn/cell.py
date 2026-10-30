"""The reduced human layer 2/3 pyramidal neuron, v0 (LAB.md E0): its compartments, constants and mechanisms.

One concept: what the cell IS. Stepping it through time is `sim.py`'s job.

**Units:** mV, ms, pF, nS, pA (so pA / pF = mV / ms, and nS · mV = pA).
"""
from dataclasses import dataclass, fields, replace
import math

import numpy as np

# (name, parent, length in um, diameter in um, kind). The soma is lumped (its area is a parameter).
MORPHOLOGY = (
    ("soma", None, None, None, "soma"),
    ("b1.p", "soma", 60.0, 1.0, "branch"),
    ("b1.m", "b1.p", 60.0, 0.9, "branch"),
    ("b1.d", "b1.m", 60.0, 0.8, "branch"),
    ("b2.p", "soma", 60.0, 1.0, "branch"),
    ("b2.m", "b2.p", 60.0, 0.9, "branch"),
    ("b2.d", "b2.m", 60.0, 0.8, "branch"),
    ("trunk", "soma", 150.0, 2.5, "trunk"),
    ("zone", "trunk", 100.0, 2.0, "zone"),
    ("t1.p", "zone", 60.0, 0.9, "branch"),
    ("t1.m", "t1.p", 60.0, 0.8, "branch"),
    ("t1.d", "t1.m", 60.0, 0.7, "branch"),
    ("t2.p", "zone", 60.0, 0.9, "branch"),
    ("t2.m", "t2.p", 60.0, 0.8, "branch"),
    ("t2.d", "t2.m", 60.0, 0.7, "branch"),
)
NAMES = tuple(m[0] for m in MORPHOLOGY)
INDEX = {n: i for i, n in enumerate(NAMES)}
SOMA, ZONE = INDEX["soma"], INDEX["zone"]
BASAL = tuple(INDEX[n] for n in NAMES if n.startswith("b"))
TUFT = tuple(INDEX[n] for n in NAMES if n.startswith("t") and n != "trunk")


@dataclass(frozen=True)
class Params:
    """Every number of the cell. Defaults are LAB.md E0's; `dca_w` is set by its rule (`protocols.calibrate`)."""
    # passive (Cm: human L2/3, Eyal et al. 2016; the rest typical cortical values)
    Cm: float = 0.5            # uF/cm^2
    Rm: float = 20000.0        # Ohm cm^2
    Ra: float = 150.0          # Ohm cm
    EL: float = -70.0          # mV
    soma_area: float = 25000.0  # um^2: the soma plus the dendrites not modelled
    # synapses, per contact (conductances: human L2/3, Eyal et al. 2018)
    g_ampa: float = 0.88       # nS
    ampa_rise: float = 0.3     # ms
    ampa_decay: float = 1.8
    g_nmda: float = 1.31       # nS
    nmda_rise: float = 2.0
    nmda_decay: float = 70.0
    mg: float = 1.0            # mM (Jahr & Stevens 1990)
    mg_gamma: float = 0.062    # 1/mV
    # the soma: AdEx (Brette & Gerstner 2005), "adapting" (Naud et al. 2008)
    VT: float = -50.0
    DT: float = 2.0
    a: float = 2.0             # nS
    tau_w: float = 300.0       # ms
    b: float = 60.0            # pA
    Vr: float = -58.0
    V_detect: float = 0.0      # a spike is a crossing of this
    V_spike: float = 30.0      # the soma is held here for t_spike: the spike's waveform, and so the bAP
    t_spike: float = 1.0
    # the dCaAP (Gidon et al. 2020's model, as in Hashemi et al. 2025)
    dca_theta: float = -36.0
    dca_tauA: float = 3.0
    dca_tauB: float = 0.4
    dca_delta: float = 21.0
    dca_ref: float = 200.0
    dca_tauK: float = 0.3
    dca_w: float = 0.0         # pA; 0 = not yet set by its rule
    dca_Ith: float = 0.0       # pA: the zone's threshold current for a long step (amendment 1), set by its rule

    def with_(self, **kw):
        return replace(self, **{k: float(v) for k, v in kw.items()})

    @staticmethod
    def names():
        return [f.name for f in fields(Params)]


def _peak_norm(rise, decay):
    """The factor that makes exp(-t/decay) - exp(-t/rise) peak at 1."""
    tp = rise * decay / (decay - rise) * math.log(decay / rise)
    return 1.0 / (math.exp(-tp / decay) - math.exp(-tp / rise))


class Cell:
    """The cell's electrical structure, built from Params: per compartment C and g_L, the coupling, the step matrix."""

    def __init__(self, p: Params):
        self.p = p
        n = len(MORPHOLOGY)
        self.N = n
        area = np.zeros(n)
        R = np.zeros(n)                    # each compartment's axial resistance (MOhm); 0 for the lumped soma
        for i, (name, parent, L, d, kind) in enumerate(MORPHOLOGY):
            if kind == "soma":
                area[i] = p.soma_area
            else:
                area[i] = math.pi * d * L
                R[i] = 4.0 * p.Ra * L * 0.01 / (math.pi * d * d)
        self.area = area
        self.C = p.Cm * area * 1e-2                   # pF
        self.gL = area * 10.0 / p.Rm                  # nS
        self.parent = np.array([-1 if m[1] is None else INDEX[m[1]] for m in MORPHOLOGY])
        self.G = np.zeros((n, n))                     # coupling conductances (nS), symmetric
        for i in range(n):
            j = self.parent[i]
            if j >= 0:
                g = 1000.0 / (R[i] / 2 + R[j] / 2)
                self.G[i, j] = self.G[j, i] = g
        self.ampa_norm = p.g_ampa * _peak_norm(p.ampa_rise, p.ampa_decay)
        self.nmda_norm = p.g_nmda * _peak_norm(p.nmda_rise, p.nmda_decay)

    def step_matrix(self, dt):
        """M0 for backward Euler: (C/dt + g_L + sum of couplings) on the diagonal, minus the couplings off it."""
        M = -self.G.copy()
        M[np.diag_indices(self.N)] = self.C / dt + self.gL + self.G.sum(1)
        return M

    def input_resistance(self):
        """The soma's input resistance (MOhm), from the passive structure."""
        M = -self.G.copy()
        M[np.diag_indices(self.N)] = self.gL + self.G.sum(1)
        return 1000.0 * np.linalg.inv(M)[SOMA, SOMA]
