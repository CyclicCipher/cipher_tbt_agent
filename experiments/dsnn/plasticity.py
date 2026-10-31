"""The calcium rule (LAB.md E1): one plasticity rule for every synapse of both cell types.

One concept: what a synapse does with its own calcium.

**A synapse's calcium** c(t), in its spine head, in µM: c′ = −c/τ_Ca + J(t)/(2F·V_sp·(1 + κ)), with:
- **J_N**, its own NMDA receptors: (1/30)·g_N(t)·B(V)·(E_Ca − V). Ca²⁺ is 10% of the NMDA current at −70 mV. g_N is
  one contact's NMDA conductance from its own presynaptic spikes, with the simulator's kinetics and Mg²⁺ block. V is
  its compartment's voltage.
- **J_V**, voltage-gated: its compartment's Ca²⁺ current per µm² × the spine head's membrane area (sim.py's
  `record_ca`).

**Its weight** (Graupner & Brunel 2012, without the bistable term and noise):
τ_ρ·ρ′ = γ_p·(1 − ρ)·[c > θ_p] − γ_d·ρ·[c > θ_d], from ρ = 0.5. The strength is w ∝ 1 + (b − 1)·ρ, with b = 4. The
reported change is w_after/w_before.

**Computed offline:** the weight is held fixed while a protocol runs, as slices express a change after induction. So
c comes from a recorded run, and ρ from c.

**Units:** ms, mV, nS, pA, µM; the rates k_d = γ_d/τ_ρ and k_p = γ_p/τ_ρ are per second.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit

TAU_CA = 20.0              # ms (Graupner & Brunel 2012)
V_SP = 0.1                 # µm³, a spherical spine head
A_SP = 4.0 * math.pi * (3.0 * V_SP / (4.0 * math.pi)) ** (2.0 / 3.0)       # its membrane, 1.04 µm²
KAPPA = 20.0               # the spine's buffer capacity (Sabatini, Oertner & Svoboda 2002)
E_CA = 140.0               # mV
Q_N = 1.0 / 30.0           # Ca²⁺ = 10% of the NMDA current at −70 mV
FARADAY = 96485.309
B_W = 4.0                  # w ∝ 1 + (b − 1)·ρ
RHO0 = 0.5
UM_PER_PA_MS = 1e-9 / (2.0 * FARADAY * V_SP * 1e-15 * (1.0 + KAPPA))            # µM per (pA·ms) into one spine


def nmda_g(pre, t, syn):
    """One contact's NMDA conductance (nS) at times t, from its presynaptic spike times."""
    _, nm = syn.norms()
    g = np.zeros(len(t))
    span = 10.0 * syn.nmda_decay
    for ts in pre:
        i0, i1 = np.searchsorted(t, ts), np.searchsorted(t, ts + span)
        x = t[i0:i1] - ts
        g[i0:i1] += nm * (np.exp(-x / syn.nmda_decay) - np.exp(-x / syn.nmda_rise))
    return g


def calcium(t, v, ica, pre, area, syn):
    """c(t) in µM for one synapse: its compartment's voltage v and Ca²⁺ current ica (pA, inward negative), the
    compartment's area (µm²), and the synapse's presynaptic times."""
    v = np.asarray(v, float)
    B = 1.0 / (1.0 + syn.mg / 3.57 * np.exp(-syn.mg_gamma * v))
    J = Q_N * nmda_g(pre, t, syn) * B * (E_CA - v) - np.asarray(ica, float) * A_SP / area
    return _leaky(J * UM_PER_PA_MS, float(t[1] - t[0]), TAU_CA)


@njit(cache=True)
def _leaky(drive, h, tau):
    """c′ = −c/τ + drive, exactly for a drive held over each sample."""
    c = np.zeros(drive.shape[0])
    a = math.exp(-h / tau)
    x = 0.0
    for i in range(drive.shape[0]):
        x = x * a + drive[i] * tau * (1.0 - a)
        c[i] = x
    return c


@njit(cache=True)
def runs(c, th_d, th_p):
    """Run-length code of the rule's state: 0 below θ_d, 1 between, 2 above θ_p. Returns (states, sample counts)."""
    n = c.shape[0]
    st = np.zeros(n, np.int8)
    ln = np.zeros(n, np.int64)
    k = -1
    prev = -1
    for i in range(n):
        s = 2 if c[i] > th_p else (1 if c[i] > th_d else 0)
        if s != prev:
            k += 1
            st[k] = s
            prev = s
        ln[k] += 1
    return st[:k + 1], ln[:k + 1]


@njit(cache=True)
def rho_end(st, ln, h, kp, kd, rho):
    """ρ after the coded states, integrated exactly (rates per second, h in ms)."""
    for i in range(st.shape[0]):
        d = ln[i] * h / 1000.0
        if st[i] == 1:
            rho *= math.exp(-kd * d)
        elif st[i] == 2:
            inf = kp / (kp + kd)
            rho = inf + (rho - inf) * math.exp(-(kp + kd) * d)
    return rho


def change(rho):
    """w_after / w_before."""
    return (1.0 + (B_W - 1.0) * rho) / (1.0 + (B_W - 1.0) * RHO0)


@dataclass(frozen=True)
class Rule:
    th_d: float                # µM
    th_p: float                # µM
    ratio: float               # γ_p / γ_d
    kd: float                  # γ_d / τ_ρ, per s

    def apply(self, c, h):
        st, ln = runs(np.asarray(c, float), self.th_d, self.th_p)
        return change(rho_end(st, ln, h, self.ratio * self.kd, self.kd, RHO0))


def fit(traces, targets, tols, n_th=40, ratios=np.logspace(-1, 2, 16), kds=np.logspace(-2, 2, 21)):
    """The grid search of LAB.md E1 on the fit set: traces [(c, h)], targets and tolerances of w_after/w_before.
    θ_d < θ_p on a log grid from (the smallest trace peak)/20 to the largest peak. Least squared error, each error
    divided by its tolerance; the first of equals in grid order. Returns (Rule, error, changes)."""
    peaks = [float(np.max(c)) for c, _ in traces]
    th = np.logspace(math.log10(min(peaks) / 20.0), math.log10(max(peaks)), n_th)
    targets, tols = np.asarray(targets, float), np.asarray(tols, float)
    best = (np.inf, None, None)
    for i in range(n_th):
        for j in range(i + 1, n_th):
            coded = [runs(np.asarray(c, float), th[i], th[j]) + (h,) for c, h in traces]
            E, ch, a, b = _grid(coded, ratios, kds, targets, tols)
            if E < best[0]:
                best = (E, Rule(float(th[i]), float(th[j]), float(ratios[a]), float(kds[b])), ch)
    return best[1], best[0], best[2]


def _grid(coded, ratios, kds, targets, tols):
    best, arg, chs = np.inf, (0, 0), None
    for a, r in enumerate(ratios):
        for b, kd in enumerate(kds):
            ch = np.array([change(rho_end(st, ln, h, r * kd, kd, RHO0)) for st, ln, h in coded])
            E = float((((ch - targets) / tols) ** 2).sum())
            if E < best:
                best, arg, chs = E, (a, b), ch
    return best, chs, arg[0], arg[1]
