"""Stepping cells through time (LAB.md E0b, "The simulator"; E0e): one simulator for every experiment and for the viz.

**What runs:** a batch of runs (`Stim`s), each a copy of a `Wiring` of cells. Every compartment of every cell of
every run is one node of a forest (each cell a tree, its soma the root), stepped together by one compiled loop.

**Synapses** (E0e): four receptor types, `AMPA`, `NMDA` (with its Mg²⁺ block), `GABA_S` and `GABA_F` (two GABA_A
kinetics), each a difference of two exponentials peaking at its conductance per contact. An event carries a mask of
the types it opens: `CONTACT` = AMPA + NMDA (E0's contact; the default), `AMPA` alone, `GABA_S`, `GABA_F`. Each cell
brings its own kinetics (`cell.syn`).

**The step** (dt, from t to t + dt):
1. **Inputs** for this step: synaptic events, current steps, voltage clamps of the soma, and contacts delivered by
   the wiring after its delays.
2. **The dCaAP** (Gidon et al. 2020's `dCaAP.mod`, exactly; LAB.md E0e): at the start of the step the site's V is
   above θ, outside the refractory period. Then K = exp(−(V − θ)/((θ − V_rest)·D)), capped at 1: about 1 at a
   crossing, smaller when the refractory period ends with the site held above θ. Its current from this step:
   w·K·(A − B), A and B logistic from 0.001, B from Δ after the onset.
   **Or the channel dCaAP** (`cell.ChannelDCaAP`, LAB.md E0d): Ca_HVA and BK at the zone, conductances like any other.
3. **Backward Euler** for the leak, the coupling and every conductance (synaptic and channel), each conductance taken
   at the start of the step (NEURON's linearization): the NMDA Mg²⁺ block, the AdEx exponential and adaptation, the
   channel gates.
4. **The solve:** Hines elimination (children into parents, then back out), O(compartments). A held soma (a spike's
   hold, or a clamp) is set, not solved.
5. **Spikes:**
   - an AdEx soma crosses V_detect: it is held at V_spike for t_spike, then at Vr for t_ref (LAB.md E0, amendments 3
     and 6);
   - a Hodgkin–Huxley soma (Bahl's channels, or Traub's) crosses 0 mV upward.
6. **Gates** advance by exponential Euler at the new voltage (`channels`, `traub`, `dcaap_channels`); synapses decay.

**Ca²⁺ current** (`record_ca`): each recorded compartment's Ca²⁺ current (pA, inward negative): the HH sca current and
the channel dCaAP's at the step's new voltage, and the phenomenological dCaAP's current, which Gidon et al. identify
as Ca²⁺.

**"Spiking off"** (as with TTX): an AdEx soma loses its exponential and its spikes; a Hodgkin–Huxley cell loses its
Na⁺ conductances.
"""
from dataclasses import dataclass, field
import math
import time

import numpy as np
from numba import njit

from . import channels as ch
from . import dcaap_channels as dch
from . import traub as tr
from .channels import P_ECA, P_EHD, P_EK, P_ENA

DT = 0.05            # ms
REC_EVERY = 10       # record every 0.5 ms
NA, NI = 20, 4       # per-cell parameters: floats, ints (see `_cell_arrays`)
AMPA, NMDA, GABA_S, GABA_F = 1, 2, 4, 8          # event masks; receptor r is bit r
CONTACT = AMPA | NMDA
NR = 4


@dataclass
class Stim:
    """One run's interventions. Times in ms; `neuron` indexes the wiring's cells; `comp` is a compartment index;
    `kind` is an event mask (`CONTACT`, `AMPA`, `GABA_S`, `GABA_F`)."""
    label: str = ""
    spiking: bool = True                                # False: the spike mechanism off (as with TTX)
    syn: list = field(default_factory=list)            # (t, neuron, comp, contacts, kind)
    steps: list = field(default_factory=list)          # (t0, t1, neuron, comp, pA)
    poisson: list = field(default_factory=list)        # (t0, t1, neuron, comp, contacts, rate_hz, seed, kind)
    clamps: list = field(default_factory=list)         # (t0, t1, neuron, mV): the soma held at mV

    def contacts(self, t, comp, n, neuron=0, kind=CONTACT):
        if n:
            self.syn.append((float(t), int(neuron), int(comp), float(n), int(kind)))
        return self

    def current(self, t0, t1, comp, pA, neuron=0):
        if pA:
            self.steps.append((float(t0), float(t1), int(neuron), int(comp), float(pA)))
        return self

    def train(self, t0, t1, comp, contacts, rate, seed, neuron=0, kind=CONTACT):
        """`contacts` independent Poisson trains at `rate` Hz (as one Poisson train of contacts × rate)."""
        if contacts and rate:
            self.poisson.append((float(t0), float(t1), int(neuron), int(comp), float(contacts), float(rate), seed,
                                 int(kind)))
        return self

    def clamp(self, t0, t1, mV, neuron=0):
        self.clamps.append((float(t0), float(t1), int(neuron), float(mV)))
        return self


@dataclass
class Wiring:
    """How many cells, and the synapses between them: (src, dst, dst_comp, contacts, delay_ms), each an AMPA + NMDA
    contact."""
    n: int = 1
    edges: list = field(default_factory=list)


@dataclass
class Result:
    t: np.ndarray            # recorded times (ms)
    V: list                  # [run][cell] -> (recorded compartments, times), float32 by default
    rec: list                # [cell] -> the compartments recorded
    vmax: list               # [run][cell] -> (N,): each compartment's peak over the run
    spikes: list             # [run][cell] -> spike times
    dcaap: list              # [run][cell] -> [(onset time, K, V − θ at onset in mV)]
    syn: list                # [run] -> [(t, cell, comp, contacts)]: every activation (with keep_syn)
    final: list              # [run][cell] -> {"V", "X", "Q", "T"} (with final=True)
    seconds: float           # wall-clock
    steps: int
    ca: list = None          # [run][cell] -> (recorded compartments, times): Ca²⁺ current, pA (with record_ca)

    def v(self, b, comp, cell=0):
        return self.V[b][cell][list(self.rec[cell]).index(comp)]

    def i_ca(self, b, comp, cell=0):
        return self.ca[b][cell][list(self.rec[cell]).index(comp)]


def simulate(stims, T, cells, wiring: Wiring = None, dt=DT, rec_every=REC_EVERY, record=None, keep_syn=False,
             final=False, rec_dtype=np.float32, record_ca=False):
    """cells: one Cell for every neuron of the wiring, or a list (one per neuron). record: compartments to record,
    for every cell (None: the soma; "all": every one). record_ca: also their Ca²⁺ currents."""
    t_start = time.perf_counter()
    wiring = wiring or Wiring()
    nn, B = wiring.n, len(stims)
    kinds = list(cells) if isinstance(cells, (list, tuple)) else [cells] * nn
    assert len(kinds) == nn
    S = B * nn
    K = int(round(T / dt))
    insts = [kinds[s % nn] for s in range(S)]
    off = np.concatenate([[0], np.cumsum([c.N for c in insts])]).astype(np.int64)
    M = int(off[-1])
    sid = lambda b, j: b * nn + j
    seg = lambda s: slice(off[s], off[s + 1])

    # ---- the forest ----
    par = np.concatenate([np.where(c.parent >= 0, c.parent + off[s], -1) for s, c in enumerate(insts)]).astype(np.int64)
    gax = np.concatenate([c.gax for c in insts]).astype(float)
    C = np.concatenate([c.C for c in insts]).astype(float)
    gL = np.concatenate([c.gL for c in insts]).astype(float)
    EL = np.concatenate([c.EL for c in insts]).astype(float)
    V = np.concatenate([c.rest["V"] for c in insts]).astype(float)
    hh = np.zeros(M, np.int8)
    G = np.zeros((8, M))
    vsh2 = np.zeros(M)
    gcad = np.zeros(M)
    X = np.zeros((13, M))
    P = ch.HH(dens={}, vshift2_nat=None).params()
    tg = np.zeros((2, M))                  # Traub's Na⁺, K⁺ totals (nS), and their gates
    ts = np.zeros((3, M))
    qg = np.zeros((2, M))                  # the channel dCaAP: total g_Ca, g_BK (nS); its constants; its state
    qp = np.ones((9, M))
    qs = np.zeros((6, M))
    for s, c in enumerate(insts):
        a, z = off[s], off[s + 1]
        if c.hh is not None:
            hh[a:z] = np.any([c.hh.dens[n] > 0 for n in ch.NAMES], axis=0)
            G[:, a:z] = c.hh.gmax(c.area)
            vsh2[a:z] = c.hh.vshift2_nat
            gcad[a:z] = c.hh.dens["sca"] * c.hh.tadj(23)
            X[:, a:z] = c.rest["X"]
            P = c.hh.params()
        if c.traub is not None:
            tg[:, a:z] = c.traub.gmax(c.area)
            if "T" in c.rest:
                ts[:, a:z] = c.rest["T"]
            else:
                for i in range(a, z):
                    tr.steady(ts, i, V[i])
        if c.cach is not None:
            q, i = c.cach, a + c.cach.zone
            qg[:, i] = (q.g_ca, q.g_bk)
            qp[:, i] = q.q()
            if "Q" in c.rest:
                qs[:, i] = c.rest["Q"]
            else:
                dch.steady(qs, i, V[i], qp)
    ccell = np.repeat(np.arange(S), [c.N for c in insts]).astype(np.int64)
    fa, ia = _cell_arrays(insts, off, [stims[s // nn].spiking for s in range(S)])

    # ---- synapses: per compartment, each cell's kinetics (receptor r: peak per contact, decay factors, reversal) ----
    gn, fd, fr, er = np.zeros((NR, M)), np.ones((NR, M)), np.ones((NR, M)), np.zeros((NR, M))
    mg357, gam = np.zeros(M), np.zeros(M)
    for s, c in enumerate(insts):
        sy = c.syn
        norms, rises, decays, revs = sy.table()
        for r in range(NR):
            gn[r, seg(s)], er[r, seg(s)] = norms[r], revs[r]
            fd[r, seg(s)], fr[r, seg(s)] = math.exp(-dt / decays[r]), math.exp(-dt / rises[r])
        mg357[seg(s)], gam[seg(s)] = sy.mg / 3.57, sy.mg_gamma

    # ---- inputs, as sorted event lists ----
    syn_log = [[] for _ in range(B)]
    ev, cur, cl = [], [], []
    for b, st in enumerate(stims):
        for t, j, c, n, kd in st.syn:
            k = int(round(t / dt))
            if 0 <= k < K:
                ev.append((k, off[sid(b, j)] + c, n, kd))
                if keep_syn:
                    syn_log[b].append((t, j, c, n))
        for t0, t1, j, c, amp in st.steps:
            k0, k1 = max(0, int(round(t0 / dt))), int(round(t1 / dt))
            if k0 < min(k1, K):
                cur.append((k0, off[sid(b, j)] + c, amp))
                cur.append((k1, off[sid(b, j)] + c, -amp))
        for t0, t1, j, c, n, rate, seed, kd in st.poisson:
            rng = np.random.default_rng(seed)
            times = np.sort(rng.uniform(t0, t1, rng.poisson(n * rate * (t1 - t0) / 1000.0)))
            for t in times:
                k = int(t / dt)
                if k < K:
                    ev.append((k, off[sid(b, j)] + c, 1.0, kd))
                    if keep_syn:
                        syn_log[b].append((k * dt, j, c, 1.0))
        for t0, t1, j, v in st.clamps:
            cl.append((max(0, int(round(t0 / dt))), sid(b, j), v))
            cl.append((int(round(t1 / dt)), sid(b, j), np.nan))
    ev.sort(key=lambda e: e[0])
    cur.sort(key=lambda e: e[0])
    cl.sort(key=lambda e: e[0])
    arr = lambda L, i, dtype: np.array([e[i] for e in L], dtype) if L else np.zeros(0, dtype)

    # ---- the wiring: edges by source cell (CSR), delays in steps ----
    out = [[] for _ in range(S)]
    for src, dst, c, n, dl in wiring.edges:
        for b in range(B):
            out[sid(b, src)].append((off[sid(b, dst)] + c, n, max(1, int(round(dl / dt)))))
    e_ptr = np.concatenate([[0], np.cumsum([len(o) for o in out])]).astype(np.int64)
    flat = [e for o in out for e in o]
    e_dst, e_n, e_d = arr(flat, 0, np.int64), arr(flat, 1, float), arr(flat, 2, np.int64)
    D = int(e_d.max()) + 1 if len(e_d) else 1

    # ---- recording ----
    rec_local = [list(range(c.N)) if record == "all" else list(record or [0]) for c in kinds]
    rec_c = np.array([off[sid(b, j)] + i for b in range(B) for j in range(nn) for i in rec_local[j]], np.int64)
    R = K // rec_every
    rec = np.zeros((R, len(rec_c)), rec_dtype)
    rca = np.zeros((R, len(rec_c)) if record_ca else (0, 0), rec_dtype)
    vmax = V.copy()
    cap_s = int(S * (T / 2.0 + 2) + 16)
    cap_d = int(S * (T / max(1.0, min((c.dcaap.ref for c in kinds if c.dcaap), default=200.0)) + 2) + 16)
    spk_t, spk_s = np.zeros(cap_s), np.zeros(cap_s, np.int64)
    dca = np.zeros((cap_d, 4))
    cap_l = 4 * cap_s * max(1, max((len(o) for o in out), default=0)) if keep_syn else 0
    dlog = np.zeros((cap_l, 3))
    counts = _run(K, dt, rec_every, par, gax, C, gL, EL, V, hh, G, vsh2, gcad, X, P, ccell, fa, ia,
                  arr(ev, 0, np.int64), arr(ev, 1, np.int64), arr(ev, 2, float), arr(ev, 3, np.int64),
                  arr(cur, 0, np.int64), arr(cur, 1, np.int64), arr(cur, 2, float),
                  arr(cl, 0, np.int64), arr(cl, 1, np.int64), arr(cl, 2, float),
                  e_ptr, e_dst, e_n, e_d, D, gn, fd, fr, er, mg357, gam,
                  tg, ts, qg, qp, qs, rec_c, rec, rca, vmax, spk_t, spk_s, dca, dlog)
    n_spk, n_dca, n_log = counts
    if n_spk > cap_s or n_dca > cap_d or (keep_syn and n_log > cap_l):
        raise RuntimeError("an output buffer overflowed")

    # ---- results ----
    spikes = [[[] for _ in range(nn)] for _ in range(B)]
    for t, s in zip(spk_t[:n_spk], spk_s[:n_spk]):
        spikes[s // nn][s % nn].append(float(t))
    dcaap = [[[] for _ in range(nn)] for _ in range(B)]
    for t, s, Kv, lv in dca[:n_dca]:
        dcaap[int(s) // nn][int(s) % nn].append((float(t), float(Kv), float(lv)))
    for k, c, n in dlog[:n_log]:
        s = int(np.searchsorted(off, int(c), side="right") - 1)
        syn_log[s // nn].append((k * dt, s % nn, int(c) - int(off[s]), float(n)))
    Vr, Car, i = [[None] * nn for _ in range(B)], [[None] * nn for _ in range(B)] if record_ca else None, 0
    for b in range(B):
        for j in range(nn):
            Vr[b][j] = rec[:, i:i + len(rec_local[j])].T
            if record_ca:
                Car[b][j] = rca[:, i:i + len(rec_local[j])].T
            i += len(rec_local[j])
    vm = [[vmax[seg(sid(b, j))] for j in range(nn)] for b in range(B)]
    fin = [[{"V": V[seg(sid(b, j))].copy(), "X": X[:, seg(sid(b, j))].copy(), "T": ts[:, seg(sid(b, j))].copy(),
             "Q": qs[:, off[sid(b, j)] + kinds[j].cach.zone].copy() if kinds[j].cach is not None else None}
            for j in range(nn)] for b in range(B)] if final else None
    t_rec = (np.arange(R) * rec_every + rec_every - 1) * dt
    return Result(t_rec, Vr, rec_local, vm, spikes, dcaap, syn_log, fin, time.perf_counter() - t_start, K, Car)


# per-cell parameter columns
(F_VT, F_DT, F_A, F_TAUW, F_B, F_VR, F_VDET, F_VSPK, F_TSPK, F_TREF, F_EL,
 F_W, F_TH, F_D, F_VREST, F_TAUA, F_TAUB, F_DELTA, F_REF, F_HHDET) = range(NA)
I_ROOT, I_END, I_KIND, I_ZONE = range(NI)
KIND_PASSIVE, KIND_ADEX, KIND_HH = 0, 1, 2


def _cell_arrays(insts, off, spiking):
    fa = np.zeros((len(insts), NA))
    ia = np.full((len(insts), NI), -1, np.int64)
    for s, c in enumerate(insts):
        ia[s, I_ROOT], ia[s, I_END] = off[s], off[s + 1]
        ia[s, I_KIND] = KIND_ADEX if c.adex else (KIND_HH if (c.hh is not None or c.traub is not None)
                                                  else KIND_PASSIVE)
        if c.adex:
            a = c.adex
            fa[s, :F_EL + 1] = (a.VT, a.DT, a.a, a.tau_w, a.b, a.Vr, a.V_detect, a.V_spike, a.t_spike, a.t_ref,
                                c.EL[0])
        if c.dcaap is not None and c.dcaap.w > 0:
            d = c.dcaap
            ia[s, I_ZONE] = off[s] + d.zone
            fa[s, F_W:F_REF + 1] = (d.w, d.theta, d.D, d.V_rest, d.tauA, d.tauB, d.delta, d.ref)
        fa[s, F_HHDET] = 0.0
    # spiking: the AdEx exponential and spikes, or the HH Na+ conductances; kept in the kind column's sign
    for s, on in enumerate(spiking):
        if not on:
            ia[s, I_KIND] = -ia[s, I_KIND] - 1          # -1 passive, -2 AdEx off, -3 HH off
    return fa, ia


@njit(cache=True)
def _run(K, dt, rec_every, par, gax, C, gL, EL, V, hh, G, vsh2, gcad, X, P, ccell, fa, ia,
         ev_k, ev_c, ev_n, ev_m, cur_k, cur_c, cur_a, cl_k, cl_s, cl_v, e_ptr, e_dst, e_n, e_d, D,
         gn, fd, fr, er, mg357, gam, tg, ts, qg, qp, qs, rec_c, rec, rca, vmax, spk_t, spk_s, dca, dlog):
    M, S = V.shape[0], fa.shape[0]
    sa, sb = np.zeros((NR, M)), np.zeros((NR, M))     # each receptor's two exponentials
    inj, d, r, Vn = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)
    gcs, iph = np.zeros(M), np.zeros(M)          # this step's Ca²⁺ conductance, and the phenomenological dCaAP's pA
    ecs = np.full(M, P[P_ECA])                   # the Ca²⁺ reversal that goes with gcs
    do_ca = rca.shape[0] > 0
    ring = np.zeros((D, M))
    w, hold, ext = np.zeros(S), np.zeros(S), np.full(S, np.nan)
    don, dK, dref = np.full(S, -1.0), np.zeros(S), np.zeros(S)
    ie = ic = il = 0
    n_spk = n_dca = n_log = 0
    has_ring = e_dst.shape[0] > 0
    ena, ek, eca, ehd = P[P_ENA], P[P_EK], P[P_ECA], P[P_EHD]
    for k in range(K):
        t = k * dt
        # 1. inputs
        while ic < cur_k.shape[0] and cur_k[ic] == k:
            inj[cur_c[ic]] += cur_a[ic]
            ic += 1
        while ie < ev_k.shape[0] and ev_k[ie] == k:
            c, n, m = ev_c[ie], ev_n[ie], ev_m[ie]
            for q in range(NR):
                if m & (1 << q):
                    sa[q, c] += n * gn[q, c]
                    sb[q, c] += n * gn[q, c]
            ie += 1
        if has_ring:
            slot = k % D
            for c in range(M):
                n = ring[slot, c]
                if n != 0.0:
                    for q in range(2):                       # a wiring contact: AMPA + NMDA
                        sa[q, c] += n * gn[q, c]
                        sb[q, c] += n * gn[q, c]
                    ring[slot, c] = 0.0
        while il < cl_k.shape[0] and cl_k[il] == k:
            ext[cl_s[il]] = cl_v[il]
            il += 1
        # 2. the dCaAP's onset, from the step's starting voltage (dCaAP.mod's BEFORE BREAKPOINT)
        for s in range(S):
            z = ia[s, I_ZONE]
            if z < 0:
                continue
            th = fa[s, F_TH]
            if dref[s] <= 1e-9 and V[z] > th:
                Kv = min(1.0, math.exp(-(V[z] - th) / ((th - fa[s, F_VREST]) * fa[s, F_D])))
                don[s], dK[s], dref[s] = 0.0, Kv, fa[s, F_REF]
                if n_dca < dca.shape[0]:
                    dca[n_dca, 0], dca[n_dca, 1], dca[n_dca, 2], dca[n_dca, 3] = t, s, Kv, V[z] - th
                n_dca += 1
        # 3. the matrix: diagonal d, right-hand side r; the off-diagonal is -gax
        for i in range(M):
            v = V[i]
            d[i] = C[i] / dt + gL[i]
            r[i] = C[i] / dt * v + gL[i] * EL[i] + inj[i]
            for q in range(NR):
                g = sa[q, i] - sb[q, i]
                if g != 0.0:
                    if q == 1:
                        g = g / (1.0 + mg357[i] * math.exp(-gam[i] * v))
                    d[i] += g
                    r[i] += g * er[q, i]
            gcs[i] = 0.0
            iph[i] = 0.0
            on = 1.0 if ia[ccell[i], I_KIND] >= 0 else 0.0
            if hh[i]:
                gna, gk, gh, gca = ch.conductances(X, i, G, on)
                d[i] += gna + gk + gh + gca
                r[i] += gna * ena + gk * ek + gh * ehd + gca * eca
                gcs[i] = gca
            if tg[0, i] > 0.0 or tg[1, i] > 0.0:
                gna, gk = tr.conductances(ts, i, tg[0, i], tg[1, i], on)
                d[i] += gna + gk
                r[i] += gna * tr.E_NA + gk * tr.E_K
            if qg[0, i] > 0.0:
                gc, ec, gb = dch.conductances(qs, i, qg[0, i], qg[1, i])
                d[i] += gc + gb
                r[i] += gc * ec + gb * dch.EK
                gcs[i] = gc
                ecs[i] = ec
        for i in range(M):
            p = par[i]
            if p >= 0:
                d[i] += gax[i]
                d[p] += gax[i]
        for s in range(S):
            rt = ia[s, I_ROOT]
            if ia[s, I_KIND] == KIND_ADEX:
                r[rt] += gL[rt] * fa[s, F_DT] * math.exp(min((V[rt] - fa[s, F_VT]) / fa[s, F_DT], 20.0)) - w[s]
            z = ia[s, I_ZONE]
            if z >= 0 and don[s] >= 0.0:
                A = 1.0 / (1.0 + 999.0 * math.exp(-don[s] / fa[s, F_TAUA]))
                Bv = 0.0
                if don[s] >= fa[s, F_DELTA]:
                    Bv = 1.0 / (1.0 + 999.0 * math.exp(-(don[s] - fa[s, F_DELTA]) / fa[s, F_TAUB]))
                r[z] += fa[s, F_W] * dK[s] * (A - Bv)
                iph[z] = fa[s, F_W] * dK[s] * (A - Bv)
        # 4. Hines: eliminate children into parents, then solve outward
        for i in range(M - 1, -1, -1):
            p = par[i]
            if p >= 0:
                f = gax[i] / d[i]
                d[p] -= f * gax[i]
                r[p] += f * r[i]
        for s in range(S):
            rt = ia[s, I_ROOT]
            kind = ia[s, I_KIND]
            if (kind == KIND_ADEX or kind == -KIND_ADEX - 1) and hold[s] > 0.0:
                Vn[rt] = fa[s, F_VSPK] if hold[s] > fa[s, F_TREF] + 1e-9 else fa[s, F_VR]
            elif ext[s] == ext[s]:
                Vn[rt] = ext[s]
            else:
                Vn[rt] = r[rt] / d[rt]
        for i in range(M):
            p = par[i]
            if p >= 0:
                Vn[i] = (r[i] + gax[i] * Vn[p]) / d[i]
        # 5. spikes
        for s in range(S):
            rt, kind = ia[s, I_ROOT], ia[s, I_KIND]
            new = False
            if kind == KIND_ADEX or kind == -KIND_ADEX - 1:
                held = hold[s] > 0.0
                if held:
                    hold[s] -= dt
                if kind == KIND_ADEX and not held and Vn[rt] >= fa[s, F_VDET]:
                    new = True
                    Vn[rt] = fa[s, F_VSPK]                    # solve the step again with the soma held (amendment 6)
                    for i in range(rt + 1, ia[s, I_END]):
                        Vn[i] = (r[i] + gax[i] * Vn[par[i]]) / d[i]
                    hold[s] = fa[s, F_TSPK] + fa[s, F_TREF]
                    w[s] += fa[s, F_B]
                if kind == KIND_ADEX:
                    w[s] += dt / fa[s, F_TAUW] * (fa[s, F_A] * (V[rt] - fa[s, F_EL]) - w[s])
            elif kind == KIND_HH:
                new = V[rt] < fa[s, F_HHDET] and Vn[rt] >= fa[s, F_HHDET]
            if new:
                if n_spk < spk_t.shape[0]:
                    spk_t[n_spk] = t
                    spk_s[n_spk] = s
                n_spk += 1
                for e in range(e_ptr[s], e_ptr[s + 1]):
                    if k + e_d[e] < K:
                        ring[(k + e_d[e]) % D, e_dst[e]] += e_n[e]
                        if n_log < dlog.shape[0]:
                            dlog[n_log, 0], dlog[n_log, 1], dlog[n_log, 2] = k + e_d[e], e_dst[e], e_n[e]
                        n_log += 1
        # 6. the dCaAP's clocks; gates, synapses, the new state
        for s in range(S):
            if ia[s, I_ZONE] < 0:
                continue
            if don[s] >= 0.0:
                don[s] += dt
                if don[s] > 100.0:
                    don[s] = -1.0
            dref[s] -= dt
        for i in range(M):
            if hh[i]:
                ch.advance(X, i, Vn[i], dt, vsh2[i], gcad[i], P)
            if tg[0, i] > 0.0 or tg[1, i] > 0.0:
                tr.advance(ts, i, Vn[i], dt)
            if qg[0, i] > 0.0:
                dch.advance(qs, i, Vn[i], dt, gcs[i] * (Vn[i] - ecs[i]), qp)
            for q in range(NR):
                sa[q, i] *= fd[q, i]
                sb[q, i] *= fr[q, i]
            V[i] = Vn[i]
            if Vn[i] > vmax[i]:
                vmax[i] = Vn[i]
        if k % rec_every == rec_every - 1 and k // rec_every < rec.shape[0]:
            row = k // rec_every
            for j in range(rec_c.shape[0]):
                rec[row, j] = V[rec_c[j]]
                if do_ca:
                    c = rec_c[j]
                    rca[row, j] = gcs[c] * (V[c] - ecs[c]) - iph[c]
    return n_spk, n_dca, n_log
