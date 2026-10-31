"""What a cell IS (LAB.md E0b): a tree of compartments, its passive constants, and its mechanisms.

One concept: the cell. Building a particular cell type is `cells/`; stepping cells through time is `sim.py`.

**Units:** mV, ms, pF, nS, pA (pA / pF = mV / ms, and nS · mV = pA); lengths in µm.

**A cell has:**
- compartments 0..N-1, 0 the soma, each with its parent (-1 for the soma), axial conductance to the parent, membrane
  area, capacitance, leak and leak reversal;
- named groups of compartments (e.g. "basal1", "tuft"), the sites the protocols use;
- synapses (the same AMPA + NMDA contact everywhere);
- one way to spike: an AdEx soma (`adex`), or Hodgkin–Huxley channels (`hh`, `channels.py`);
- optionally a dCaAP: Gidon's phenomenological one (`dcaap`), or a channel (`cach`).
"""
from dataclasses import dataclass, field, replace
import math

import numpy as np


@dataclass(frozen=True)
class Synapses:
    """One contact = one AMPA + one NMDA conductance (Eyal et al. 2018's human L2/3 values; LAB.md E0)."""
    g_ampa: float = 0.88       # nS
    ampa_rise: float = 0.3     # ms
    ampa_decay: float = 1.8
    g_nmda: float = 1.31       # nS
    nmda_rise: float = 2.0
    nmda_decay: float = 70.0
    mg: float = 1.0            # mM (Jahr & Stevens 1990)
    mg_gamma: float = 0.062    # 1/mV

    def norms(self):
        """Each conductance's peak per contact (nS), for a difference of two exponentials."""
        return self.g_ampa * _peak_norm(self.ampa_rise, self.ampa_decay), self.g_nmda * _peak_norm(self.nmda_rise,
                                                                                                   self.nmda_decay)


@dataclass(frozen=True)
class AdEx:
    """The AdEx soma (Brette & Gerstner 2005), "adapting" (Naud et al. 2008), with E0's spike waveform and refractory
    hold (amendments 3 and 6)."""
    VT: float = -50.0
    DT: float = 2.0
    a: float = 2.0             # nS
    tau_w: float = 300.0       # ms
    b: float = 60.0            # pA
    Vr: float = -58.0
    V_detect: float = 0.0
    V_spike: float = 30.0
    t_spike: float = 1.0
    t_ref: float = 2.0


@dataclass(frozen=True)
class DCaAP:
    """Gidon et al. 2020's dCaAP (`dCaAP.mod`, `xor.hoc`), with K in its step-free form (LAB.md E0b)."""
    zone: int                  # the compartment
    w: float = 2970.0          # pA
    tauK: float = 0.0          # ms; fitted by `protocols.fit_tauK`
    theta: float = -36.0
    D: float = 0.3
    V_rest: float = -75.0
    tauA: float = 3.0
    tauB: float = 0.4
    delta: float = 21.0
    ref: float = 200.0


@dataclass(frozen=True)
class CaChannel:
    """The channel dCaAP (LAB.md E0c-A): g·m·h·(V − E_Ca) at the zone, E_Ca = 140 mV; m and h first order, with
    m∞ = 1/(1 + exp(−(V − Vm)/km)), h∞ = 1/(1 + exp((V − Vh)/kh)), and time constants independent of voltage."""
    zone: int
    g: float                   # nS
    Vm: float                  # mV
    km: float
    taum: float                # ms
    Vh: float
    kh: float
    tauh: float
    i_ref: float = 0.0         # pA: the peak of its current in the I_th step (an "event" reaches half of it)

    def minf(self, v):
        return 1.0 / (1.0 + math.exp(-(v - self.Vm) / self.km))

    def hinf(self, v):
        return 1.0 / (1.0 + math.exp((v - self.Vh) / self.kh))


@dataclass
class Cell:
    name: str
    names: list
    parent: np.ndarray
    gax: np.ndarray            # nS, to the parent (0 for the soma)
    area: np.ndarray           # µm²
    C: np.ndarray              # pF
    gL: np.ndarray             # nS
    EL: np.ndarray             # mV
    dist: np.ndarray           # µm from the soma, path distance to the centre
    length: np.ndarray         # µm, each compartment's own length (0 for the soma); contacts are spread by length
    groups: dict = field(default_factory=dict)
    syn: Synapses = Synapses()
    adex: AdEx = None
    dcaap: DCaAP = None
    hh: object = None          # channels.HH
    rest: dict = None          # the initial state: {"V": (N,), and for hh cells the gates}
    cach: CaChannel = None     # the channel dCaAP (E0c-A), in place of `dcaap`

    @property
    def N(self):
        return len(self.names)

    def index(self, name):
        return self.names.index(name)

    def site(self, name):
        """A named site's one compartment (a group of one, e.g. "zone", "b1.d")."""
        (c,) = self.groups[name]
        return c

    def with_dcaap(self, **kw):
        return replace(self, dcaap=replace(self.dcaap, **kw))

    def passive_matrix(self):
        """The conductance matrix of the passive cell (nS): leak on the diagonal, axial coupling off it."""
        G = np.diag(self.gL.astype(float))
        for i in range(1, self.N):
            p, g = self.parent[i], self.gax[i]
            G[i, i] += g
            G[p, p] += g
            G[i, p] -= g
            G[p, i] -= g
        return G

    def impedance(self):
        """Steady-state transfer resistances (MΩ): Z[i, j] = V_i / I_j."""
        return 1000.0 * np.linalg.inv(self.passive_matrix())

    def input_resistance(self, i=0):
        return float(self.impedance()[i, i])


def passive(geo, Cm, Rm, EL, name, Cm_of=None, Rm_of=None):
    """Arrays from a geometry: C = Cm·area, gL = area / Rm, gax = 1 / R. Cm_of / Rm_of: per-compartment overrides."""
    area = np.asarray(geo.area, float)
    cm = np.array([Cm_of(i) if Cm_of else Cm for i in range(len(area))])
    rm = np.array([Rm_of(i) if Rm_of else Rm for i in range(len(area))])
    gax = np.where(np.asarray(geo.R) > 0, 1000.0 / np.where(np.asarray(geo.R) > 0, geo.R, 1.0), 0.0)
    return Cell(name=name, names=list(geo.names), parent=np.asarray(geo.parent, int), gax=gax, area=area,
                C=cm * area * 1e-2, gL=area * 10.0 / rm, EL=np.full(len(area), float(EL)), dist=np.asarray(geo.dist),
                length=np.asarray(geo.length, float))


def _peak_norm(rise, decay):
    """The factor that makes exp(-t/decay) - exp(-t/rise) peak at 1."""
    tp = rise * decay / (decay - rise) * math.log(decay / rise)
    return 1.0 / (math.exp(-tp / decay) - math.exp(-tp / rise))
