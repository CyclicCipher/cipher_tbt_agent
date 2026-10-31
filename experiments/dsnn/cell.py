"""What a cell IS (LAB.md E0b): a tree of compartments, its passive constants, and its mechanisms.

One concept: the cell. Building a particular cell type is `cells/`; stepping cells through time is `sim.py`.

**Units:** mV, ms, pF, nS, pA (pA / pF = mV / ms, and nS · mV = pA); lengths in µm.

**A cell has:**
- compartments 0..N-1, 0 the soma, each with its parent (-1 for the soma), axial conductance to the parent, membrane
  area, capacitance, leak and leak reversal;
- named groups of compartments (e.g. "basal1", "tuft"), the sites the protocols use;
- its synapses' kinetics (`syn`: AMPA, NMDA, two GABA_A; `sim.py` says which an event opens);
- one way to spike: an AdEx soma (`adex`), or Hodgkin–Huxley channels: Bahl's (`hh`, `channels.py`) or Traub's
  (`traub`, `traub.py`);
- optionally a dCaAP: Gidon's phenomenological one (`dcaap`), or a channel (`cach`);
- optionally its morphology (`tree`, the sections; `where`, section → compartments; `seclists`, named section lists),
  for placing synapses by position (`protocols.place`).
"""
from dataclasses import dataclass, field, replace
import math

import numpy as np


@dataclass(frozen=True)
class Synapses:
    """A cell's synaptic kinetics: AMPA, NMDA (Mg²⁺ block 1/(1 + mg/3.57·e^(−γV))), and two GABA_A (slow, "martinotti";
    fast, "basket"). Each a difference of two exponentials peaking at its conductance per contact.
    Defaults: E0's AMPA/NMDA (type B uses them); `GIDON` is type A's (LAB.md E0e)."""
    g_ampa: float = 0.88       # nS
    ampa_rise: float = 0.3     # ms
    ampa_decay: float = 1.8
    g_nmda: float = 1.31       # nS
    nmda_rise: float = 2.0
    nmda_decay: float = 70.0
    mg: float = 1.0            # mM (Jahr & Stevens 1990)
    mg_gamma: float = 0.062    # 1/mV
    g_gaba_s: float = 0.5      # nS (Gidon et al.'s "martinotti")
    gaba_s_rise: float = 2.0
    gaba_s_decay: float = 23.0
    g_gaba_f: float = 0.5      # nS ("basket")
    gaba_f_rise: float = 0.5
    gaba_f_decay: float = 5.0
    e_gaba: float = -75.0      # mV

    def norms(self):
        """Each conductance's peak-normalized amplitude per contact (nS): AMPA, NMDA, GABA_S, GABA_F."""
        return tuple(g * _peak_norm(a, b) for g, a, b in zip(*self._kinetics()))

    def table(self):
        """(norms, rises, decays, reversals), each by receptor (AMPA, NMDA, GABA_S, GABA_F)."""
        g, rise, decay = self._kinetics()
        return self.norms(), rise, decay, (0.0, 0.0, self.e_gaba, self.e_gaba)

    def _kinetics(self):
        return ((self.g_ampa, self.g_nmda, self.g_gaba_s, self.g_gaba_f),
                (self.ampa_rise, self.nmda_rise, self.gaba_s_rise, self.gaba_f_rise),
                (self.ampa_decay, self.nmda_decay, self.gaba_s_decay, self.gaba_f_decay))


GIDON = Synapses(g_ampa=0.7, ampa_rise=0.3, ampa_decay=1.8, g_nmda=1.3, nmda_rise=8.0, nmda_decay=35.0, mg=1.0,
                 mg_gamma=0.077)   # their syns.hoc ("as in Eyal et al. 2018"), with Rhodes 2006's γ


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
    """Gidon et al. 2020's dCaAP, exactly as `dCaAP.mod` (LAB.md E0e): onset when the site starts a step above θ outside
    the refractory period; K = exp(−(V − θ)/((θ − V_rest)·D)), capped at 1."""
    zone: int                  # the compartment
    w: float = 2970.0          # pA: their density (mA/cm²) × the site's segment area
    theta: float = -36.0
    D: float = 0.3
    V_rest: float = -75.0
    tauA: float = 3.0
    tauB: float = 0.4
    delta: float = 21.0
    ref: float = 200.0


@dataclass(frozen=True)
class ChannelDCaAP:
    """The channel dCaAP (LAB.md E0d): Gidon et al.'s Ca_HVA + BK + Ca²⁺ shell (`dcaap_channels.py`) at one
    compartment, as total conductances, the shell's calcium computed over `area` (their compartment's membrane)."""
    zone: int
    g_ca: float                # nS, Ca_HVA's maximal total
    g_bk: float                # nS, BK's maximal total
    area: float = 31101.767    # µm², the pool's membrane (their compartment: L 100, diam 99)
    ma: float = 1.0
    mb: float = 1.0
    ha: float = 1.0
    hb: float = 1.0
    mtf: float = 0.2           # BK's mtau_factor
    htf: float = 0.02          # BK's htau_factor
    zhalf: float = 0.01        # mM
    taur: float = 60.0         # ms
    i_ref: float = 0.0         # pA: Ca_HVA's peak in the first event of the I_th step (an "event" reaches half)

    def q(self):
        """The constant rows of `dcaap_channels` (Q_*)."""
        return (self.ma, self.mb, self.ha, self.hb, self.mtf, self.htf, self.zhalf, self.taur, self.area)


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
    rest: dict = None          # the initial state: {"V": (N,), and the gates: "X" (hh), "T" (traub), "Q" (cach)}
    cach: ChannelDCaAP = None  # the channel dCaAP (E0d), in place of `dcaap`
    traub: object = None       # traub.Traub
    tree: object = None        # morphology.Tree
    where: dict = None         # section -> [(compartment, arc start, arc end)]
    seclists: dict = None      # name -> [sections]

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
