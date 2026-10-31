"""Traub's spiking channels, as Gidon et al.'s type A soma and axon use them (ModelDB 254217, `Traub.mod`; LAB.md E0e).

One concept: the kinetics of the soma's Na⁺ and K⁺ channels in the human L2/3 model. `sim.py` steps them.

| channel | gates | maximal conductance (their `biophys.hoc`) | reversal |
|---|---|---|---|
| Na⁺ | m²h | 0.1 S/cm² | 90 mV |
| K⁺ | n | 0.045 S/cm² | −80 mV |

**Their rates** (Traub et al. 1991, with the voltage shifted: vt = v + v_shft, v_shft = 50 mV), every time constant
divided by Q = 3^((35 − 32)/10), for 35 °C. Their leak (1/37,000 S/cm² at −74 mV) is the cell's own leak. NEURON's
lookup table (−100 to 70 mV) is replaced by the exact functions, their singular points by their limits.

**State rows** (`T_*`): m, h, n.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit

V_SHFT, E_NA, E_K = 50.0, 90.0, -80.0
Q = 3.0 ** ((35.0 - 32.0) / 10.0)
T_M, T_H, T_N = range(3)


@dataclass
class Traub:
    na: np.ndarray             # (N,) S/cm², Na⁺ maximal conductance density per compartment (0 where absent)
    k: np.ndarray              # (N,) S/cm²

    def gmax(self, area):
        """(2, N) maximal conductances in nS (S/cm² × µm² × 10)."""
        return np.array([self.na * area * 10.0, self.k * area * 10.0])


@njit(cache=True)
def _lin(x, k, a):
    """a·x/(exp(x/k) − 1), and its limit a·k at x = 0."""
    if abs(x) < 1e-9:
        return a * k
    return a * x / (math.exp(x / k) - 1.0)


@njit(cache=True)
def rates(v):
    """(m∞, τ_m, h∞, τ_h, n∞, τ_n), ms."""
    vt = v + V_SHFT
    a, b = _lin(13.1 - vt, 4.0, 0.32), _lin(vt - 40.1, 5.0, 0.28)
    minf, mtau = a / (a + b), 1.0 / (a + b) / Q
    a, b = 0.128 * math.exp((17.0 - vt) / 18.0), 4.0 / (1.0 + math.exp((40.0 - vt) / 5.0))
    hinf, htau = a / (a + b), 1.0 / (a + b) / Q
    a, b = _lin(35.1 - vt, 5.0, 0.016), 0.25 * math.exp((20.0 - vt) / 40.0)
    ninf, ntau = a / (a + b), 1.0 / (a + b) / Q
    return minf, mtau, hinf, htau, ninf, ntau


@njit(cache=True)
def steady(S, i, v):
    minf, _, hinf, _, ninf, _ = rates(v)
    S[T_M, i], S[T_H, i], S[T_N, i] = minf, hinf, ninf


@njit(cache=True)
def conductances(S, i, gna_tot, gk_tot, na_on):
    """(g_Na, g_K) now, in nS."""
    m = S[T_M, i]
    return na_on * gna_tot * m * m * S[T_H, i], gk_tot * S[T_N, i]


@njit(cache=True)
def advance(S, i, v, dt):
    minf, mtau, hinf, htau, ninf, ntau = rates(v)
    S[T_M, i] = minf + (S[T_M, i] - minf) * math.exp(-dt / mtau)
    S[T_H, i] = hinf + (S[T_H, i] - hinf) * math.exp(-dt / htau)
    S[T_N, i] = ninf + (S[T_N, i] - ninf) * math.exp(-dt / ntau)
