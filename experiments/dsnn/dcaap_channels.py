"""The dCaAP's channels: Gidon et al. 2020's Fig. S12 model (ModelDB 260178), ported (LAB.md E0d).

One concept: the kinetics of Ca_HVA, BK and the Ca²⁺ shell that make a dCaAP. The cell holds them as a
`cell.ChannelDCaAP`; `sim.py` steps them.

| mechanism | source file | gates | reversal |
|---|---|---|---|
| Ca_HVA | Ca_HVA.mod (Reuveni et al. 1993's rates; multipliers ma, mb, ha, hb) | m²h | E_Ca, Nernst from [Ca]i |
| BK | BK.mod (Akemann & Knöpfel 2006, from Khaliq et al. 2003) | m³z²h; z from [Ca]i | E_K = −77 mV |
| cad | capump.mod (Destexhe 1994): a shell, inward current only, first-order removal | [Ca]i | — |

**As the source runs them:** NEURON's defaults, so 6.3 °C (it sets only E_Ca, since no rate here has a
temperature factor), [Ca]o 2 mM, E_K −77 mV. The rates' singular point (v = −27 mV in m's α) takes its limit.

**State rows** (`S_*`): Ca_HVA m, h; BK m, z, h; [Ca]i (mM). **Constant rows** (`Q_*`): ma, mb, ha, hb, BK's
mtau_factor and htau_factor, zhalf (mM), τ_r (ms), the pool's area (µm²).
"""
import math

import numpy as np
from numba import njit

CELSIUS, CAO, EK, CAINF, DEPTH, CTAUZ = 6.3, 2.0, -77.0, 2.4e-4, 1.0, 1.0
FARADAY_CAD = 96489.0                          # capump.mod's constant
RT_2F = 8.31446261815324 * (273.15 + CELSIUS) / (2.0 * 96485.33212) * 1000.0     # mV
S_CM, S_CH, S_BM, S_BZ, S_BH, S_CA = range(6)
Q_MA, Q_MB, Q_HA, Q_HB, Q_MTF, Q_HTF, Q_ZHALF, Q_TAUR, Q_AREA = range(9)


@njit(cache=True)
def eca(cai):
    return RT_2F * math.log(CAO / cai)


@njit(cache=True)
def ca_rates(v, ma, mb, ha, hb):
    """Ca_HVA: (m∞, τ_m, h∞, τ_h), ms."""
    x = -27.0 - v
    a = ma * (0.055 * x / (math.exp(x / 3.8) - 1.0) if abs(x) > 1e-9 else 0.055 * 3.8)
    b = mb * 0.94 * math.exp((-75.0 - v) / 17.0)
    c = ha * 0.000457 * math.exp((-13.0 - v) / 50.0)
    d = hb * 0.0065 / (math.exp((-v - 15.0) / 28.0) + 1.0)
    return a / (a + b), 1.0 / (a + b), c / (c + d), 1.0 / (c + d)


@njit(cache=True)
def bk_rates(v, cai, mtf, htf, zhalf):
    """BK: (m∞, τ_m, z∞, τ_z, h∞, τ_h), ms; the source shifts v by +5 mV."""
    v = v + 5.0
    minf = 1.0 / (1.0 + math.exp(-(v + 28.9) / 6.2))
    taum = 1e3 * (0.000505 + 1.0 / (math.exp(-(v + 86.4) / -10.1) + math.exp(-(v - 33.3) / 10.0))) / mtf
    zinf = 1.0 / (1.0 + zhalf / cai)
    hinf = 0.085 + 0.915 / (1.0 + math.exp(-(v + 32.0) / -5.8))
    tauh = 1e3 * (0.0019 + 1.0 / (math.exp(-(v + 48.5) / -5.2) + math.exp(-(v - 54.2) / 12.9))) / htf
    return minf, taum, zinf, CTAUZ, hinf, tauh


@njit(cache=True)
def steady(S, i, v, Q):
    """Every gate at its steady state at v, and [Ca]i at its resting value (the source's INITIAL blocks)."""
    S[S_CA, i] = CAINF
    m, _, h, _ = ca_rates(v, Q[Q_MA, i], Q[Q_MB, i], Q[Q_HA, i], Q[Q_HB, i])
    bm, _, bz, _, bh, _ = bk_rates(v, CAINF, Q[Q_MTF, i], Q[Q_HTF, i], Q[Q_ZHALF, i])
    S[S_CM, i], S[S_CH, i], S[S_BM, i], S[S_BZ, i], S[S_BH, i] = m, h, bm, bz, bh


@njit(cache=True)
def conductances(S, i, gca_tot, gbk_tot):
    """(g_Ca, E_Ca, g_BK) now, in nS and mV."""
    return (gca_tot * S[S_CM, i] * S[S_CM, i] * S[S_CH, i], eca(S[S_CA, i]),
            gbk_tot * S[S_BM, i] ** 3 * S[S_BZ, i] ** 2 * S[S_BH, i])


@njit(cache=True)
def advance(S, i, v, dt, i_ca, Q):
    """The gates by exponential Euler at the new voltage v; then [Ca]i from the Ca²⁺ current i_ca (pA over the pool's
    area; inward only, as the source's pump)."""
    m, tm, h, th = ca_rates(v, Q[Q_MA, i], Q[Q_MB, i], Q[Q_HA, i], Q[Q_HB, i])
    S[S_CM, i] = m + (S[S_CM, i] - m) * math.exp(-dt / tm)
    S[S_CH, i] = h + (S[S_CH, i] - h) * math.exp(-dt / th)
    bm, tbm, bz, tbz, bh, tbh = bk_rates(v, S[S_CA, i], Q[Q_MTF, i], Q[Q_HTF, i], Q[Q_ZHALF, i])
    S[S_BM, i] = bm + (S[S_BM, i] - bm) * math.exp(-dt / tbm)
    S[S_BZ, i] = bz + (S[S_BZ, i] - bz) * math.exp(-dt / tbz)
    S[S_BH, i] = bh + (S[S_BH, i] - bh) * math.exp(-dt / tbh)
    drive = -1e4 * (0.1 * i_ca / Q[Q_AREA, i]) / (2.0 * FARADAY_CAD * DEPTH)       # mM/ms; 1 pA/µm² = 0.1 mA/cm²
    if drive < 0.0:
        drive = 0.0
    taur = Q[Q_TAUR, i]
    cinf = CAINF + drive * taur
    S[S_CA, i] = cinf + (S[S_CA, i] - cinf) * math.exp(-dt / taur)


def published():
    """Their parameters0.hoc: (g_Ca density S/cm², g_BK density pS/µm², the compartment's area µm², constants Q)."""
    area = math.pi * 99.0 * 100.0
    return 0.0005, 300.0, area, dict(ma=1.0, mb=1.0, ha=1.0, hb=1.0, mtf=0.2, htf=0.02, zhalf=0.01, taur=60.0)
