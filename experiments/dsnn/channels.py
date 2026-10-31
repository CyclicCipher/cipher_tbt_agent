"""Hodgkin–Huxley channels: Bahl et al. 2012's reduced layer 5 cell (ModelDB 146026, `channels/*.mod`), ported.

One concept: the voltage- and calcium-gated channels, their kinetics and their state (LAB.md E0b, type B).

**The channels** (each a density in pS/µm² per compartment):

| name | source file | gates | reversal | temperature factor on g and rates |
|---|---|---|---|---|
| nat | nat.mod (Mainen 1994; Kole 2006) | m³h | ena | q10 2.3 from 23 °C |
| kfast | kfast.mod (Mainen 1995) | n | ek | q10 2.3 from 23 °C |
| kslow | kslow.mod (Korngreen 2002) | a²(b + b1)/2 | ek | on g only: q10 2.3 from 21 °C |
| nap | nap.mod (Traub 2003) | m | ena | none |
| km | IKM.mod | m | ek | rates only: 2.3 from 36 °C |
| ih | h.mod (Kole 2006) | q | ehd | none |
| sca | SlowCa.mod (Reuveni 1993; Mainen) | m²h | eca (fixed) | q10 2.3 from 23 °C |
| kca | kca.mod (Reuveni 1993; Mainen) | n, from [Ca²⁺]i | ek | q10 2.3 from 23 °C |
| cad | cad.mod | [Ca²⁺]i from sca's current, decay τ_r | — | — |

**Port details:**
- NEURON's lookup tables (199 points from −120 to 100 mV) are replaced by the exact functions.
- trap0's singular point: the source tests |v/th| > 1e-6, which guards the wrong point. Here the limit a·q is taken
  at v = th.
- Gates advance by exponential Euler on the new voltage (cnexp). km and cad use Euler in the source; the
  exponential form agrees to first order.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit

NAMES = ("nat", "kfast", "kslow", "nap", "km", "ih", "sca", "kca")
# rows of the state array X
NAT_M, NAT_H, KF_N, KS_A, KS_B, KS_B1, NAP_M, KM_M, IH_Q, SCA_M, SCA_H, KCA_N, CA = range(13)
FARADAY = 96485.309


@dataclass
class HH:
    dens: dict                         # name -> (N,) density, pS/µm²
    vshift2_nat: np.ndarray            # (N,) mV; the initial segment's second shift
    celsius: float = 37.0
    ena: float = 55.0
    ek: float = -80.0
    eca: float = 140.0
    ehd: float = -47.0
    vshift_nat: float = 10.0
    vshift_sca: float = 0.0
    cad_depth: float = 0.1             # µm
    cad_taur: float = 200.0            # ms
    cad_cainf: float = 100e-6          # mM

    def tadj(self, temp):
        return 2.3 ** ((self.celsius - temp) / 10.0)

    def gmax(self, area):
        """(8, N) maximal conductances in nS, with each source's temperature factor on g."""
        f = dict(nat=self.tadj(23), kfast=self.tadj(23), kslow=self.tadj(21), nap=1.0, km=1.0, ih=1.0,
                 sca=self.tadj(23), kca=self.tadj(23))
        return np.array([self.dens[n] * area * 1e-3 * f[n] for n in NAMES])

    def params(self):
        """The constants the kernel needs, as one array (see `P_*`)."""
        return np.array([self.tadj(23), self.tadj(21), 1000.0 / 2.3 ** ((self.celsius - 36.0) / 10.0),
                         self.vshift_nat, self.vshift_sca, self.ena, self.ek, self.eca, self.ehd, self.cad_depth,
                         self.cad_taur, self.cad_cainf])


P_TADJ23, P_TADJ21, P_KM_TAUPEAK, P_VSH_NAT, P_VSH_SCA, P_ENA, P_EK, P_ECA, P_EHD, P_DEPTH, P_TAUR, P_CAINF = range(12)


@njit(cache=True)
def _trap0(v, th, a, q):
    x = v - th
    if abs(x) > 1e-6:
        return a * x / (1.0 - math.exp(-x / q))
    return a * q


@njit(cache=True)
def nat_rates(vm, tadj):
    a = _trap0(vm, -28.0, 0.182, 9.0)
    b = _trap0(-vm, 28.0, 0.124, 9.0)
    minf, mtau = a / (a + b), 1.0 / tadj / (a + b)
    a = _trap0(vm, -50.0, 0.024, 5.0)
    b = _trap0(-vm, 75.0, 0.0091, 5.0)
    return minf, mtau, 1.0 / (1.0 + math.exp((vm + 55.0) / 6.2)), 1.0 / tadj / (a + b)


@njit(cache=True)
def kfast_rates(v, tadj):
    x = v - 25.0
    if abs(x) > 1e-6:
        a = 0.02 * x / (1.0 - math.exp(-x / 9.0))
        b = -0.002 * x / (1.0 - math.exp(x / 9.0))
    else:
        a, b = 0.02 * 9.0, 0.002 * 9.0
    return a / (a + b), 1.0 / tadj / (a + b)


@njit(cache=True)
def kslow_rates(vm):
    x = vm - 11.1
    alpha = 0.0052 * x / (1.0 - math.exp(-x / 13.1)) if abs(x) > 1e-6 else 0.0052 * 13.1
    beta = 0.01938 * math.exp(-(vm + 1.27) / 71.0) - 0.0053
    taua = 1.0 / (alpha + beta)
    ainf = alpha / (alpha + beta)
    taub = 360.0 + (1010.0 + 23.7 * (vm + 54.0)) * math.exp(-((vm + 75.0) / 48.0) ** 2)
    taub1 = 2350.0 + 1380.0 * math.exp(-0.01118 * vm) - 210.0 * math.exp(-0.0306 * vm)
    binf = 1.0 / (1.0 + math.exp((vm + 58.0) / 11.0))
    return ainf, taua, binf, taub, taub1


@njit(cache=True)
def nap_rates(v):
    minf = 1.0 / (1.0 + math.exp((-v - 48.0) / 10.0))
    if v < -40.0:
        return minf, 0.025 + 0.14 * math.exp((v + 40.0) / 10.0)
    return minf, 0.02 + 0.145 * math.exp((-v - 40.0) / 10.0)


@njit(cache=True)
def km_rates(v, taupeak):
    return 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0)), taupeak / (3.3 * math.exp((v + 35.0) / 20.0)
                                                                     + math.exp(-(v + 35.0) / 20.0))


@njit(cache=True)
def ih_rates(v):
    x = v + 154.9
    alpha = 0.001 * 6.43 * x / (math.exp(x / 11.9) - 1.0) if abs(x) > 1e-6 else 0.001 * 6.43 * 11.9
    beta = 0.001 * 193.0 * math.exp(v / 33.1)
    return alpha / (alpha + beta), 1.0 / (alpha + beta)


@njit(cache=True)
def sca_rates(vm, tadj):
    x = -27.0 - vm
    a = 0.055 * x / (math.exp(x / 3.8) - 1.0) if abs(x) > 1e-6 else 0.055 * 3.8
    b = 0.94 * math.exp((-75.0 - vm) / 17.0)
    minf, mtau = a / (a + b), 1.0 / tadj / (a + b)
    a = 0.000457 * math.exp((-13.0 - vm) / 50.0) / 3.0
    b = 0.0065 / (math.exp((-vm - 15.0) / 28.0) + 1.0) / 3.0
    return minf, mtau, a / (a + b), 1.0 / tadj / (a + b)


@njit(cache=True)
def kca_rates(ca, tadj):
    a, b = 0.01 * ca, 0.02
    return a / (a + b), 1.0 / tadj / (a + b)


@njit(cache=True)
def _relax(x, xinf, tau, dt):
    return xinf + (x - xinf) * math.exp(-dt / tau)


@njit(cache=True)
def advance(X, i, v, dt, vshift2, gca_dens, P):
    """Advance compartment i's gates by dt at voltage v; then [Ca²⁺]i from sca's current. gca_dens: sca's maximal
    density times its temperature factor (pS/µm²), 0 where there is no sca."""
    vm = v - P[P_VSH_NAT] - vshift2
    minf, mtau, hinf, htau = nat_rates(vm, P[P_TADJ23])
    X[NAT_M, i] = _relax(X[NAT_M, i], minf, mtau, dt)
    X[NAT_H, i] = _relax(X[NAT_H, i], hinf, htau, dt)
    ninf, ntau = kfast_rates(v, P[P_TADJ23])
    X[KF_N, i] = _relax(X[KF_N, i], ninf, ntau, dt)
    ainf, taua, binf, taub, taub1 = kslow_rates(v)
    X[KS_A, i] = _relax(X[KS_A, i], ainf, taua, dt)
    X[KS_B, i] = _relax(X[KS_B, i], binf, taub, dt)
    X[KS_B1, i] = _relax(X[KS_B1, i], binf, taub1, dt)
    minf, mtau = nap_rates(v)
    X[NAP_M, i] = _relax(X[NAP_M, i], minf, mtau, dt)
    minf, mtau = km_rates(v, P[P_KM_TAUPEAK])
    X[KM_M, i] = _relax(X[KM_M, i], minf, mtau, dt)
    qinf, qtau = ih_rates(v)
    X[IH_Q, i] = _relax(X[IH_Q, i], qinf, qtau, dt)
    minf, mtau, hinf, htau = sca_rates(v - P[P_VSH_SCA], P[P_TADJ23])
    X[SCA_M, i] = _relax(X[SCA_M, i], minf, mtau, dt)
    X[SCA_H, i] = _relax(X[SCA_H, i], hinf, htau, dt)
    if gca_dens > 0.0:
        ica = 1e-4 * gca_dens * X[SCA_M, i] ** 2 * X[SCA_H, i] * (v - P[P_ECA])      # mA/cm²
        drive = -1e4 * ica / (2.0 * FARADAY * P[P_DEPTH])
        if drive < 0.0:
            drive = 0.0
        X[CA, i] += dt * (drive + (P[P_CAINF] - X[CA, i]) / P[P_TAUR])
    ninf, ntau = kca_rates(X[CA, i], P[P_TADJ23])
    X[KCA_N, i] = _relax(X[KCA_N, i], ninf, ntau, dt)


@njit(cache=True)
def init(X, i, v, vshift2, P):
    """NEURON's INITIAL: every gate at its steady state for v; [Ca²⁺]i at rest."""
    minf, _, hinf, _ = nat_rates(v - P[P_VSH_NAT] - vshift2, P[P_TADJ23])
    X[NAT_M, i], X[NAT_H, i] = minf, hinf
    X[KF_N, i] = kfast_rates(v, P[P_TADJ23])[0]
    ainf, _, binf, _, _ = kslow_rates(v)
    X[KS_A, i], X[KS_B, i], X[KS_B1, i] = ainf, binf, binf
    X[NAP_M, i] = nap_rates(v)[0]
    X[KM_M, i] = km_rates(v, P[P_KM_TAUPEAK])[0]
    X[IH_Q, i] = ih_rates(v)[0]
    minf, _, hinf, _ = sca_rates(v - P[P_VSH_SCA], P[P_TADJ23])
    X[SCA_M, i], X[SCA_H, i] = minf, hinf
    X[CA, i] = P[P_CAINF]
    X[KCA_N, i] = kca_rates(P[P_CAINF], P[P_TADJ23])[0]


@njit(cache=True)
def conductances(X, i, G, na_on):
    """(g_na, g_k, g_h, g_ca) in nS for compartment i, from the gates and the maximal conductances G (8, M)."""
    gna = na_on * (G[0, i] * X[NAT_M, i] ** 3 * X[NAT_H, i] + G[3, i] * X[NAP_M, i])
    gk = (G[1, i] * X[KF_N, i] + G[2, i] * X[KS_A, i] ** 2 * (0.5 * X[KS_B, i] + 0.5 * X[KS_B1, i])
          + G[4, i] * X[KM_M, i] + G[7, i] * X[KCA_N, i])
    return gna, gk, G[5, i] * X[IH_Q, i], G[6, i] * X[SCA_M, i] ** 2 * X[SCA_H, i]
