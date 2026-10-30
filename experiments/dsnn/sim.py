"""Stepping cells through time (LAB.md E0, "Numerics"): one simulator for every experiment and for the viz.

**What runs:** a batch of runs (`Stim`s), each a copy of a `Wiring` of cells, all with one `Params`.

**The step:**
- **Backward Euler** for the leak, the coupling and the synaptic conductances (stable for thin branches).
- **From the previous step:** the NMDA Mg2+ block, the AdEx exponential, the adaptation and the dCaAP current.
- **A spike:** the soma crosses V_detect. It is then held at V_spike for t_spike (its waveform, which is the bAP
  through the coupling), then at Vr for t_ref (refractory; LAB.md E0, amendment 3).
- **A dCaAP:** the zone crosses dca_theta upward outside its refractory period. Its strength
  K = exp(-(I_in - I_th) / (tauK I_th)), capped at 1. I_in is the injected and synaptic current into the zone and the
  tuft at the crossing (amendment 2); I_th is the zone's threshold current for a long step (amendment 1).
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .cell import Cell, Params, SOMA, TUFT, ZONE

DT = 0.05            # ms
REC_EVERY = 10       # record every 0.5 ms


@dataclass
class Stim:
    """One run's interventions. Times in ms; `neuron` indexes the wiring's cells."""
    label: str = ""
    spiking: bool = True                                # False: the soma's spike mechanism off (as with TTX)
    syn: list = field(default_factory=list)            # (t, neuron, comp, contacts)
    steps: list = field(default_factory=list)          # (t0, t1, neuron, comp, pA)
    poisson: list = field(default_factory=list)        # (t0, t1, neuron, comp, contacts, rate_hz, seed)

    def contacts(self, t, comp, n, neuron=0):
        if n:
            self.syn.append((float(t), int(neuron), int(comp), float(n)))
        return self

    def current(self, t0, t1, comp, pA, neuron=0):
        if pA:
            self.steps.append((float(t0), float(t1), int(neuron), int(comp), float(pA)))
        return self

    def train(self, t0, t1, comp, contacts, rate, seed, neuron=0):
        if contacts and rate:
            self.poisson.append((float(t0), float(t1), int(neuron), int(comp), int(contacts), float(rate), seed))
        return self


@dataclass
class Wiring:
    """How many cells, and the synapses between them: (src, dst, dst_comp, contacts, delay_ms)."""
    n: int = 1
    edges: list = field(default_factory=list)


@dataclass
class Result:
    t: np.ndarray            # recorded times (ms)
    V: np.ndarray            # (runs, cells, compartments, times), float32
    spikes: list             # [run][cell] -> list of spike times
    dcaap: list              # [run][cell] -> list of (onset time, K)
    syn: list                # [run] -> list of (t, cell, comp, contacts): every activation delivered
    seconds: float           # wall-clock
    steps: int


def simulate(stims, T, p: Params, wiring: Wiring = None, dt=DT, rec_every=REC_EVERY, keep_syn=False):
    import time
    t_start = time.perf_counter()
    wiring = wiring or Wiring()
    cell = Cell(p)
    nn, N, B = wiring.n, cell.N, len(stims)
    S = B * nn
    K = int(round(T / dt))
    sid = lambda b, nrn: b * nn + nrn

    # ---- the drive, built CHUNK steps at a time (bounded memory): contacts activated and current injected ----
    CHUNK = 1000
    syn_log = [[] for _ in range(B)]
    events = {}                                   # step -> [(system, comp, contacts)]
    currents = []                                 # (k0, k1, system, comp, pA)
    trains = []                                   # (k0, k1, system, comp, contacts x rate per step, rng, run, cell)
    for b, st in enumerate(stims):
        for t, nrn, c, n in st.syn:
            k = int(round(t / dt))
            if 0 <= k < K:
                events.setdefault(k, []).append((sid(b, nrn), c, n))
                if keep_syn:
                    syn_log[b].append((t, nrn, c, n))
        for t0, t1, nrn, c, amp in st.steps:
            currents.append((int(round(t0 / dt)), int(round(t1 / dt)), sid(b, nrn), c, amp))
        for t0, t1, nrn, c, n, rate, seed in st.poisson:
            trains.append((int(round(t0 / dt)), int(round(t1 / dt)), sid(b, nrn), c, n * rate * dt / 1000.0,
                           np.random.default_rng(seed), b, nrn))
    spiking = np.repeat(np.array([st.spiking for st in stims]), nn)
    edges = [(s, d, c, n, int(round(dl / dt))) for s, d, c, n, dl in wiring.edges]
    out_edges = [[e for e in edges if e[0] == i] for i in range(nn)]
    ring = np.zeros((max([e[4] for e in edges], default=0) + 1, S, N))   # contacts delivered by the wiring
    inc = np.zeros((CHUNK, S, N))
    inj = np.zeros((CHUNK, S, N))

    def build(k0):
        """The drive for steps k0 .. k0 + CHUNK."""
        inc[:] = 0.0
        inj[:] = 0.0
        k1 = k0 + CHUNK
        for k in range(k0, min(k1, K)):
            for s, c, n in events.get(k, ()):
                inc[k - k0, s, c] += n
        for a, z, s, c, amp in currents:
            lo, hi = max(a, k0), min(z, k1)
            if lo < hi:
                inj[lo - k0:hi - k0, s, c] += amp
        for a, z, s, c, lam, rng, b, nrn in trains:
            lo, hi = max(a, k0), min(z, k1, K)
            if lo < hi:
                counts = rng.poisson(lam, hi - lo)
                inc[lo - k0:hi - k0, s, c] += counts
                if keep_syn:
                    for j in np.flatnonzero(counts):
                        syn_log[b].append(((lo + j) * dt, nrn, c, float(counts[j])))

    # ---- constants ----
    M0 = cell.step_matrix(dt)
    Cdt = cell.C / dt
    rest = cell.gL * p.EL
    dAd, dAr = math.exp(-dt / p.ampa_decay), math.exp(-dt / p.ampa_rise)
    dNd, dNr = math.exp(-dt / p.nmda_decay), math.exp(-dt / p.nmda_rise)
    ii = np.arange(N)
    gL0 = cell.gL[SOMA]
    APICAL = [ZONE] + list(TUFT)
    if p.dca_w > 0 and p.dca_Ith <= 0:
        raise ValueError("the dCaAP needs dca_Ith (its rule: protocols.zone_rheobase)")

    # ---- state ----
    V = np.full((S, N), p.EL)
    aA = np.zeros((S, N)); bA = np.zeros((S, N)); aN = np.zeros((S, N)); bN = np.zeros((S, N))
    w = np.zeros(S)
    clamp = np.zeros(S)                       # ms of spike hold left
    d_on = np.full(S, -1.0)                   # ms since the dCaAP's onset; -1 when none
    d_K = np.zeros(S)
    d_ref = np.zeros(S)
    R = K // rec_every
    rec = np.zeros((S, N, R), np.float32)
    spikes = [[] for _ in range(S)]
    dca = [[] for _ in range(S)]
    A = np.empty((S, N, N))
    sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    for k in range(K):
        t = k * dt
        if k % CHUNK == 0:
            build(k)
        slot = k % len(ring)
        x = inc[k % CHUNK] + ring[slot]
        ring[slot] = 0.0
        if x.any():
            aA += x * cell.ampa_norm; bA += x * cell.ampa_norm
            aN += x * cell.nmda_norm; bN += x * cell.nmda_norm
        gs = (aA - bA) + (aN - bN) / (1.0 + p.mg / 3.57 * np.exp(-p.mg_gamma * V))   # E_syn = 0
        rhs = Cdt * V + rest + inj[k % CHUNK]
        # the soma: AdEx's exponential and adaptation, where spiking is on
        expo = gL0 * p.DT * np.exp(np.minimum((V[:, SOMA] - p.VT) / p.DT, 20.0))
        rhs[:, SOMA] += np.where(spiking, expo - w, 0.0)
        # the dCaAP's current
        on = d_on >= 0
        if on.any():
            I_d = p.dca_w * d_K * (sig(d_on / p.dca_tauA) - sig((d_on - p.dca_delta) / p.dca_tauB))
            rhs[:, ZONE] += np.where(on, I_d, 0.0)
        A[:] = M0
        A[:, ii, ii] += gs
        held = clamp > 0                          # the spike's hold at V_spike, then the refractory hold at Vr
        if held.any():
            A[held, SOMA, :] = 0.0
            A[held, SOMA, SOMA] = 1.0
            rhs[held, SOMA] = np.where(clamp[held] > p.t_ref + 1e-9, p.V_spike, p.Vr)
        Vn = np.linalg.solve(A, rhs[..., None])[..., 0]

        # spikes: count down the hold, detect new crossings
        if held.any():
            clamp[held] -= dt
        new = spiking & ~held & (Vn[:, SOMA] >= p.V_detect)
        if new.any():
            for s in np.flatnonzero(new):
                spikes[s].append(t)
                b, src = divmod(s, nn)
                for _, dst, c, n, dl in out_edges[src]:
                    if k + dl < K:
                        ring[(k + dl) % len(ring), sid(b, dst), c] += n
                        if keep_syn:
                            syn_log[b].append(((k + dl) * dt, dst, c, n))
            clamp[new] = p.t_spike + p.t_ref
            w[new] += p.b
            Vn[new, SOMA] = p.V_spike
        w += np.where(spiking, dt / p.tau_w * (p.a * (V[:, SOMA] - p.EL) - w), 0.0)

        # the dCaAP: onset, its strength K, its time course
        if p.dca_w > 0:
            cross = (V[:, ZONE] < p.dca_theta) & (Vn[:, ZONE] >= p.dca_theta) & (d_ref <= 0)
            if cross.any():                       # the input: injected and synaptic current into the zone and tuft
                I_in = (inj[k % CHUNK][:, APICAL] - gs[:, APICAL] * Vn[:, APICAL]).sum(1)
                Kv = np.minimum(1.0, np.exp(-np.maximum(0.0, I_in - p.dca_Ith) / (p.dca_tauK * p.dca_Ith)))
                for s in np.flatnonzero(cross):
                    dca[s].append((t, float(Kv[s])))
                d_on[cross] = 0.0
                d_K[cross] = Kv[cross]
                d_ref[cross] = p.dca_ref
            d_on = np.where(d_on >= 0, d_on + dt, d_on)
            d_on[d_on > p.dca_delta + 10 * p.dca_tauB] = -1.0
            d_ref -= dt

        aA *= dAd; bA *= dAr; aN *= dNd; bN *= dNr
        V = Vn
        if k % rec_every == rec_every - 1:
            rec[:, :, k // rec_every] = V
    t_rec = (np.arange(R) * rec_every + rec_every - 1) * dt
    reshape = lambda L: [[L[sid(b, i)] for i in range(nn)] for b in range(B)]
    return Result(t_rec, rec.reshape(B, nn, N, R), reshape(spikes), reshape(dca), syn_log,
                  time.perf_counter() - t_start, K)
