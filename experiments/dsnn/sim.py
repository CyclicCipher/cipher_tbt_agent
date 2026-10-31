"""Stepping cells through time (LAB.md E0b, "The simulator"): one simulator for every experiment and for the viz.

**What runs:** a batch of runs (`Stim`s), each a copy of a `Wiring` of cells. Every compartment of every cell of
every run is one node of a forest (each cell a tree, its soma the root), stepped together by one compiled loop.

**The step** (dt, from t to t + dt):
1. **Inputs** for this step: synaptic contacts (each opens one AMPA + one NMDA conductance), current steps, voltage
   clamps of the soma, and contacts delivered by the wiring after its delays.
2. **Backward Euler** for the leak, the coupling and every conductance (synaptic and channel), each conductance taken
   at the start of the step (NEURON's linearization). From the start of the step: the NMDA Mg²⁺ block, the AdEx
   exponential and adaptation, the dCaAP current, the channel gates.
3. **The solve:** Hines elimination (children into parents, then back out), O(compartments). A held soma (a spike's
   hold, or a clamp) is set, not solved.
4. **Spikes:**
   - an AdEx soma crosses V_detect: it is held at V_spike for t_spike (the spike's waveform, and so the bAP), then at
     Vr for t_ref (LAB.md E0, amendment 3). The crossing step is solved again with the soma held, so the dendrites
     never see the upstroke's overshoot (amendment 6).
   - a Hodgkin–Huxley soma crosses 0 mV upward.
5. **The dCaAP** (Gidon et al. 2020's `dCaAP.mod`, level-triggered): the zone is above θ outside the refractory
   period. K = exp(−(level + τ_K·V̇)/(D·(θ − V_rest))), capped at 1, with level = how far above θ the zone already
   was at the start of the step (0 for a crossing) and V̇ its dV/dt over the step (LAB.md E0b). Its current from the
   next step: w·K·(A − B), A and B logistic from 0.001, B from Δ after the onset.
   **Or the channel dCaAP** (`cell.CaChannel`, LAB.md E0c-A): g·m·h·(V − E_Ca) at the zone, a conductance like any
   other.
6. **Gates** advance by exponential Euler at the new voltage (`channels.advance`, and the channel dCaAP's m and h);
   synapses decay.

**Ca²⁺ current** (`record_ca`, for E1's calcium): each recorded compartment's Ca²⁺ current (pA, inward negative) over
the step: the Hodgkin–Huxley sca current and the channel dCaAP's, at the step's new voltage, and the phenomenological
dCaAP's current, which Gidon et al. identify as Ca²⁺.

**"Spiking off"** (as with TTX): an AdEx soma loses its exponential and its spikes; a Hodgkin–Huxley cell loses its
Na⁺ conductances.
"""
from dataclasses import dataclass, field
import math
import time

import numpy as np
from numba import njit

from . import channels as ch
from .channels import P_ECA, P_EHD, P_EK, P_ENA

DT = 0.05            # ms
REC_EVERY = 10       # record every 0.5 ms
NA, NI = 21, 4       # per-cell parameters: floats, ints (see `_cell_arrays`)


@dataclass
class Stim:
    """One run's interventions. Times in ms; `neuron` indexes the wiring's cells; `comp` is a compartment index."""
    label: str = ""
    spiking: bool = True                                # False: the spike mechanism off (as with TTX)
    syn: list = field(default_factory=list)            # (t, neuron, comp, contacts)
    steps: list = field(default_factory=list)          # (t0, t1, neuron, comp, pA)
    poisson: list = field(default_factory=list)        # (t0, t1, neuron, comp, contacts, rate_hz, seed)
    clamps: list = field(default_factory=list)         # (t0, t1, neuron, mV): the soma held at mV

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
            self.poisson.append((float(t0), float(t1), int(neuron), int(comp), float(contacts), float(rate), seed))
        return self

    def clamp(self, t0, t1, mV, neuron=0):
        self.clamps.append((float(t0), float(t1), int(neuron), float(mV)))
        return self


@dataclass
class Wiring:
    """How many cells, and the synapses between them: (src, dst, dst_comp, contacts, delay_ms)."""
    n: int = 1
    edges: list = field(default_factory=list)


@dataclass
class Result:
    t: np.ndarray            # recorded times (ms)
    V: list                  # [run][cell] -> (recorded compartments, times), float32 by default
    rec: list                # [cell] -> the compartments recorded
    vmax: list               # [run][cell] -> (N,): each compartment's peak over the run
    spikes: list             # [run][cell] -> spike times
    dcaap: list              # [run][cell] -> [(onset time, K, V̇ in mV/ms, level in mV)]
    syn: list                # [run] -> [(t, cell, comp, contacts)]: every activation (with keep_syn)
    final: list              # [run][cell] -> {"V", "X"} (with final=True)
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
    assert len(kinds) == nn and len({c.syn for c in kinds}) == 1, "one synapse type for every cell"
    S = B * nn
    K = int(round(T / dt))
    insts = [kinds[s % nn] for s in range(S)]
    off = np.concatenate([[0], np.cumsum([c.N for c in insts])]).astype(np.int64)
    M = int(off[-1])
    sid = lambda b, j: b * nn + j

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
    for s, c in enumerate(insts):
        if c.hh is not None:
            a, z = off[s], off[s + 1]
            hh[a:z] = np.any([c.hh.dens[n] > 0 for n in ch.NAMES], axis=0)
            G[:, a:z] = c.hh.gmax(c.area)
            vsh2[a:z] = c.hh.vshift2_nat
            gcad[a:z] = c.hh.dens["sca"] * c.hh.tadj(23)
            X[:, a:z] = c.rest["X"]
            P = c.hh.params()
    cg = np.zeros(M)                       # the channel dCaAP: nS, and (V_m, k_m, τ_m, V_h, k_h, τ_h)
    cpar = np.ones((6, M))
    cam, cah = np.zeros(M), np.zeros(M)
    for s, c in enumerate(insts):
        if c.cach is not None:
            q, i = c.cach, off[s] + c.cach.zone
            cg[i] = q.g
            cpar[:, i] = (q.Vm, q.km, q.taum, q.Vh, q.kh, q.tauh)
            cam[i], cah[i] = q.minf(V[i]), q.hinf(V[i])
    ccell = np.repeat(np.arange(S), [c.N for c in insts]).astype(np.int64)
    fa, ia = _cell_arrays(insts, off, [stims[s // nn].spiking for s in range(S)])

    # ---- inputs, as sorted event lists ----
    syn_log = [[] for _ in range(B)]
    ev, cur, cl = [], [], []
    for b, st in enumerate(stims):
        for t, j, c, n in st.syn:
            k = int(round(t / dt))
            if 0 <= k < K:
                ev.append((k, off[sid(b, j)] + c, n))
                if keep_syn:
                    syn_log[b].append((t, j, c, n))
        for t0, t1, j, c, amp in st.steps:
            k0, k1 = max(0, int(round(t0 / dt))), int(round(t1 / dt))
            if k0 < min(k1, K):
                cur.append((k0, off[sid(b, j)] + c, amp))
                cur.append((k1, off[sid(b, j)] + c, -amp))
        for t0, t1, j, c, n, rate, seed in st.poisson:
            rng = np.random.default_rng(seed)
            times = np.sort(rng.uniform(t0, t1, rng.poisson(n * rate * (t1 - t0) / 1000.0)))
            for t in times:
                k = int(t / dt)
                if k < K:
                    ev.append((k, off[sid(b, j)] + c, 1.0))
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
    dca = np.zeros((cap_d, 5))
    cap_l = 4 * cap_s * max(1, max((len(o) for o in out), default=0)) if keep_syn else 0
    dlog = np.zeros((cap_l, 3))
    syn0 = kinds[0].syn
    na, nm = syn0.norms()
    counts = _run(K, dt, rec_every, par, gax, C, gL, EL, V, hh, G, vsh2, gcad, X, P, ccell, fa, ia,
                  arr(ev, 0, np.int64), arr(ev, 1, np.int64), arr(ev, 2, float),
                  arr(cur, 0, np.int64), arr(cur, 1, np.int64), arr(cur, 2, float),
                  arr(cl, 0, np.int64), arr(cl, 1, np.int64), arr(cl, 2, float),
                  e_ptr, e_dst, e_n, e_d, D,
                  na, nm, math.exp(-dt / syn0.ampa_decay), math.exp(-dt / syn0.ampa_rise),
                  math.exp(-dt / syn0.nmda_decay), math.exp(-dt / syn0.nmda_rise), syn0.mg / 3.57, syn0.mg_gamma,
                  cg, cpar, cam, cah, rec_c, rec, rca, vmax, spk_t, spk_s, dca, dlog)
    n_spk, n_dca, n_log = counts
    if n_spk > cap_s or n_dca > cap_d or (keep_syn and n_log > cap_l):
        raise RuntimeError("an output buffer overflowed")

    # ---- results ----
    spikes = [[[] for _ in range(nn)] for _ in range(B)]
    for t, s in zip(spk_t[:n_spk], spk_s[:n_spk]):
        spikes[s // nn][s % nn].append(float(t))
    dcaap = [[[] for _ in range(nn)] for _ in range(B)]
    for t, s, Kv, vd, lv in dca[:n_dca]:
        dcaap[int(s) // nn][int(s) % nn].append((float(t), float(Kv), float(vd), float(lv)))
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
    vm = [[vmax[off[sid(b, j)]:off[sid(b, j) + 1]] for j in range(nn)] for b in range(B)]
    fin = [[{"V": V[off[sid(b, j)]:off[sid(b, j) + 1]].copy(), "X": X[:, off[sid(b, j)]:off[sid(b, j) + 1]].copy()}
            for j in range(nn)] for b in range(B)] if final else None
    t_rec = (np.arange(R) * rec_every + rec_every - 1) * dt
    return Result(t_rec, Vr, rec_local, vm, spikes, dcaap, syn_log, fin, time.perf_counter() - t_start, K, Car)


# per-cell parameter columns
(F_VT, F_DT, F_A, F_TAUW, F_B, F_VR, F_VDET, F_VSPK, F_TSPK, F_TREF, F_EL,
 F_W, F_TAUK, F_TH, F_D, F_VREST, F_TAUA, F_TAUB, F_DELTA, F_REF, F_HHDET) = range(NA)
I_ROOT, I_END, I_KIND, I_ZONE = range(NI)
KIND_PASSIVE, KIND_ADEX, KIND_HH = 0, 1, 2


def _cell_arrays(insts, off, spiking):
    fa = np.zeros((len(insts), NA))
    ia = np.full((len(insts), NI), -1, np.int64)
    for s, c in enumerate(insts):
        ia[s, I_ROOT], ia[s, I_END] = off[s], off[s + 1]
        ia[s, I_KIND] = KIND_ADEX if c.adex else (KIND_HH if c.hh is not None else KIND_PASSIVE)
        if c.adex:
            a = c.adex
            fa[s, :F_EL + 1] = (a.VT, a.DT, a.a, a.tau_w, a.b, a.Vr, a.V_detect, a.V_spike, a.t_spike, a.t_ref,
                                c.EL[0])
        if c.dcaap is not None and c.dcaap.w > 0:
            d = c.dcaap
            ia[s, I_ZONE] = off[s] + d.zone
            fa[s, F_W:F_REF + 1] = (d.w, d.tauK, d.theta, d.D, d.V_rest, d.tauA, d.tauB, d.delta, d.ref)
        fa[s, F_HHDET] = 0.0
    # spiking: the AdEx exponential and spikes, or the HH Na+ conductances; kept in the kind column's sign
    for s, on in enumerate(spiking):
        if not on:
            ia[s, I_KIND] = -ia[s, I_KIND] - 1          # -1 passive, -2 AdEx off, -3 HH off
    return fa, ia


@njit(cache=True)
def _run(K, dt, rec_every, par, gax, C, gL, EL, V, hh, G, vsh2, gcad, X, P, ccell, fa, ia,
         ev_k, ev_c, ev_n, cur_k, cur_c, cur_a, cl_k, cl_s, cl_v, e_ptr, e_dst, e_n, e_d, D,
         na, nm, dAd, dAr, dNd, dNr, mg357, gam, cg, cpar, cam, cah, rec_c, rec, rca, vmax, spk_t, spk_s, dca,
         dlog):
    M, S = V.shape[0], fa.shape[0]
    aA, bA, aN, bN = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)
    inj, d, r, Vn = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)
    gcs, iph = np.zeros(M), np.zeros(M)          # this step's Ca²⁺ conductance, and the phenomenological dCaAP's pA
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
            c, n = ev_c[ie], ev_n[ie]
            aA[c] += n * na
            bA[c] += n * na
            aN[c] += n * nm
            bN[c] += n * nm
            ie += 1
        if has_ring:
            slot = k % D
            for c in range(M):
                n = ring[slot, c]
                if n != 0.0:
                    aA[c] += n * na
                    bA[c] += n * na
                    aN[c] += n * nm
                    bN[c] += n * nm
                    ring[slot, c] = 0.0
        while il < cl_k.shape[0] and cl_k[il] == k:
            ext[cl_s[il]] = cl_v[il]
            il += 1
        # 2. the matrix: diagonal d, right-hand side r; the off-diagonal is -gax
        for i in range(M):
            v = V[i]
            g = (aA[i] - bA[i]) + (aN[i] - bN[i]) / (1.0 + mg357 * math.exp(-gam * v))       # E_syn = 0
            d[i] = C[i] / dt + gL[i] + g
            r[i] = C[i] / dt * v + gL[i] * EL[i] + inj[i]
            gcs[i] = 0.0
            iph[i] = 0.0
            if hh[i]:
                gna, gk, gh, gca = ch.conductances(X, i, G, 1.0 if ia[ccell[i], I_KIND] >= 0 else 0.0)
                d[i] += gna + gk + gh + gca
                r[i] += gna * ena + gk * ek + gh * ehd + gca * eca
                gcs[i] = gca
            if cg[i] > 0.0:
                gc = cg[i] * cam[i] * cah[i]
                d[i] += gc
                r[i] += gc * eca
                gcs[i] += gc
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
        # 3. Hines: eliminate children into parents, then solve outward
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
        # 4. spikes
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
        # 5. the dCaAP
        for s in range(S):
            z = ia[s, I_ZONE]
            if z < 0:
                continue
            if don[s] >= 0.0:
                don[s] += dt
                if don[s] > 100.0:
                    don[s] = -1.0
            dref[s] -= dt
            th = fa[s, F_TH]
            if dref[s] <= 1e-9 and Vn[z] > th:
                level = max(0.0, V[z] - th)
                vdot = max(0.0, (Vn[z] - V[z]) / dt)
                Kv = min(1.0, math.exp(-(level + fa[s, F_TAUK] * vdot) / (fa[s, F_D] * (th - fa[s, F_VREST]))))
                don[s], dK[s], dref[s] = 0.0, Kv, fa[s, F_REF]
                if n_dca < dca.shape[0]:
                    dca[n_dca, 0], dca[n_dca, 1], dca[n_dca, 2], dca[n_dca, 3], dca[n_dca, 4] = t, s, Kv, vdot, level
                n_dca += 1
        # 6. gates, synapses, the new state
        for i in range(M):
            if hh[i]:
                ch.advance(X, i, Vn[i], dt, vsh2[i], gcad[i], P)
            if cg[i] > 0.0:
                vv = Vn[i]
                minf = 1.0 / (1.0 + math.exp(-(vv - cpar[0, i]) / cpar[1, i]))
                hinf = 1.0 / (1.0 + math.exp((vv - cpar[3, i]) / cpar[4, i]))
                cam[i] = minf + (cam[i] - minf) * math.exp(-dt / cpar[2, i])
                cah[i] = hinf + (cah[i] - hinf) * math.exp(-dt / cpar[5, i])
            aA[i] *= dAd
            bA[i] *= dAr
            aN[i] *= dNd
            bN[i] *= dNr
            V[i] = Vn[i]
            if Vn[i] > vmax[i]:
                vmax[i] = Vn[i]
        if k % rec_every == rec_every - 1 and k // rec_every < rec.shape[0]:
            row = k // rec_every
            for j in range(rec_c.shape[0]):
                rec[row, j] = V[rec_c[j]]
                if do_ca:
                    c = rec_c[j]
                    rca[row, j] = gcs[c] * (V[c] - eca) - iph[c]
    return n_spk, n_dca, n_log
