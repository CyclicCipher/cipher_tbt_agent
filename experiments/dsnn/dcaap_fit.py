"""The channel dCaAP, fitted to Gidon et al. 2020's step curve (LAB.md E0c-A).

One concept: the fit. The channel is `cell.CaChannel`; the simulator steps it like any conductance.

**The targets**, for 100 ms steps of current I into the zone, spiking off:
- a(I): the largest value, over the step, of V_zone(with the channel) − V_zone(passive);
- I_th: the step with the largest a (Gidon: the amplitude is largest at threshold);
- K(f) = a(f·I_th)/a(I_th) follows exp(−(f − 1)/0.3) at f = 1.5 and 2;
- a(I_th) and the width (time with the difference ≥ a/2) match E0b's phenomenological dCaAP at K = 1 and I = 350 pA;
- I_th near 350 pA; a(0.8·I_th) ≤ 0.2·a(I_th).

**The passive response is linear in I** (the AdEx soma without its exponential, no synapses), so one passive run at
100 pA gives it at every amplitude.

**The search:** 600 random candidates on a 50 pA grid (stage 1); the best 10 refined by Nelder–Mead (stage 2), each
evaluation locating I_th on the 25 pA grid and running 0.8, 1.5 and 2 × I_th exactly. Both stages spread their work
over processes; that changes the wall-clock, not the arithmetic.
"""
from dataclasses import replace
from functools import lru_cache
import math
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

from .cell import CaChannel
from .cells import l23
from .protocols import GIDON_SCALE, T0
from .sim import Stim, Wiring, simulate

NAMES = ("g", "Vm", "km", "taum", "Vh", "kh", "tauh")
LO = np.array([2.0, -50.0, 1.0, 0.5, -50.0, 1.0, 1.0])
HI = np.array([200.0, -15.0, 10.0, 20.0, -5.0, 10.0, 60.0])
LOG = np.array([True, False, False, True, False, False, True])
STEP = 100.0                                   # ms
T = T0 + STEP + 2.0
REC = 2                                        # record every 0.1 ms (dt 0.05)
COARSE = np.arange(100.0, 1500.0 + 1e-9, 50.0)
FINE = np.arange(100.0, 1500.0 + 1e-9, 25.0)
I_TH_E0B = 350.0
N_CAND, N_BEST, MAXFEV, SEED = 600, 10, 60, 0
CHUNK = 25


def from_unit(u):
    """Unit-cube coordinates -> constants (log-uniform where LOG)."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    x = LO + u * (HI - LO)
    x[..., LOG] = np.exp(np.log(LO[LOG]) + u[..., LOG] * (np.log(HI[LOG]) - np.log(LO[LOG])))
    return x


def to_unit(x):
    x = np.asarray(x, float)
    u = (x - LO) / (HI - LO)
    u[..., LOG] = (np.log(x[..., LOG]) - np.log(LO[LOG])) / (np.log(HI[LOG]) - np.log(LO[LOG]))
    return u


@lru_cache(maxsize=1)
def base():
    """Type A with neither dCaAP: the cell every candidate is added to."""
    return replace(l23(), dcaap=None)


def with_channel(x, i_ref=0.0):
    cell = base()
    return replace(cell, cach=CaChannel(cell.site("zone"), *[float(v) for v in x], i_ref=float(i_ref)))


def _stims(amps, n):
    z = base().site("zone")
    out = []
    for a in amps:
        st = Stim(spiking=False)
        for j in range(n):
            st.current(T0, T0 + STEP, z, a, neuron=j)
        out.append(st)
    return out


def zone_runs(cells, amps, dt):
    """Zone voltage and Ca²⁺ current, (cells, amps, times), for 100 ms steps of each amplitude into each cell."""
    z = base().site("zone")
    r = simulate(_stims(amps, len(cells)), T, list(cells), wiring=Wiring(len(cells)), dt=dt, rec_every=REC,
                 record=[z], record_ca=True)
    V = np.array([[r.V[b][j][0] for b in range(len(amps))] for j in range(len(cells))], float)
    I = np.array([[r.ca[b][j][0] for b in range(len(amps))] for j in range(len(cells))], float)
    return r.t, V, I


@lru_cache(maxsize=4)
def passive_unit(dt):
    """(t, the zone's passive response to 1 pA, from rest): linear, so any amplitude is a multiple."""
    cell = base()
    z = cell.site("zone")
    r = simulate([Stim(spiking=False).current(T0, T0 + STEP, z, 100.0)], T, cell, dt=dt, rec_every=REC, record=[z])
    return r.t, (np.asarray(r.V[0][0][0], float) - cell.EL[z]) / 100.0


def event(t, v, amp, dt):
    """(a, width) of one response: the channel's extra depolarization over the passive one, during the step."""
    tp, unit = passive_unit(dt)
    diff = v - (base().EL[base().site("zone")] + amp * unit)
    win = (t >= T0) & (t <= T0 + STEP)
    a = float(diff[win].max())
    width = float(np.count_nonzero(diff[win] >= a / 2.0) * (t[1] - t[0])) if a > 0 else 0.0
    return a, width


@lru_cache(maxsize=4)
def reference(dt):
    """E0b's phenomenological dCaAP at K = 1 (τ_K = 0 makes a crossing's K exactly 1), at I = 350 pA: (a, width)."""
    cell = l23().with_dcaap(tauK=0.0)
    z = cell.site("zone")
    r = simulate([Stim(spiking=False).current(T0, T0 + STEP, z, I_TH_E0B)], T, cell, dt=dt, rec_every=REC, record=[z])
    return event(r.t, np.asarray(r.V[0][0][0], float), I_TH_E0B, dt)


def error(a_th, width, I_th, K15, K2, a08, dt):
    a_ref, w_ref = reference(dt)
    if a_th <= 1e-6 or width <= 0:
        return 1e3
    k = lambda f: math.exp(-(f - 1.0) / GIDON_SCALE)
    lg = lambda x: math.log(max(x, 1e-4))
    E = (lg(K15) - lg(k(1.5))) ** 2 + (lg(K2) - lg(k(2.0))) ** 2
    E += math.log(a_th / a_ref) ** 2 + math.log(width / w_ref) ** 2 + math.log(I_th / I_TH_E0B) ** 2
    return E + (10.0 if a08 > 0.2 * a_th else 0.0)


def _coarse_chunk(args):
    X, dt = args
    cells = [with_channel(x) for x in X]
    t, V, _ = zone_runs(cells, COARSE, dt)
    out = []
    for j in range(len(cells)):
        ev = [event(t, V[j, i], a, dt) for i, a in enumerate(COARSE)]
        amp = np.array([e[0] for e in ev])
        i = int(np.argmax(amp))
        I_th, a_th, width = float(COARSE[i]), float(amp[i]), ev[i][1]
        at = lambda f: float(np.interp(f * I_th, COARSE, amp))
        K15, K2, a08 = at(1.5) / a_th if a_th > 0 else 0, at(2.0) / a_th if a_th > 0 else 0, at(0.8)
        out.append(dict(E=error(a_th, width, I_th, K15, K2, a08, dt), I_th=I_th, a=a_th, width=width, K15=K15,
                        K2=K2, a08=a08))
    return out


def evaluate(x, dt):
    """Stage 2's measure of one candidate: I_th on the 25 pA grid, then 0.8, 1.5 and 2 × I_th exactly."""
    cell = with_channel(x)
    t, V, I = zone_runs([cell], FINE, dt)
    ev = [event(t, V[0, i], a, dt) for i, a in enumerate(FINE)]
    i = int(np.argmax([e[0] for e in ev]))
    I_th, (a_th, width) = float(FINE[i]), ev[i]
    i_ref = float(-I[0, i].min())
    amps = [0.8 * I_th, 1.5 * I_th, 2.0 * I_th]
    t2, V2, _ = zone_runs([cell], amps, dt)
    a08, a15, a2 = (event(t2, V2[0, k], a, dt)[0] for k, a in enumerate(amps))
    K15, K2 = (a15 / a_th, a2 / a_th) if a_th > 0 else (0.0, 0.0)
    return dict(E=error(a_th, width, I_th, K15, K2, a08, dt), I_th=I_th, a=a_th, width=width, K15=K15, K2=K2,
                a08=a08, i_ref=i_ref, x=[float(v) for v in x])


def _refine(args):
    x0, dt = args
    seen = []

    def f(u):
        r = evaluate(from_unit(u), dt)
        seen.append(r)
        return r["E"]

    minimize(f, to_unit(x0), method="Nelder-Mead", options=dict(maxfev=MAXFEV, xatol=1e-3, fatol=1e-4))
    best = min(seen, key=lambda r: r["E"])
    return dict(best, evaluations=len(seen), start=[float(v) for v in x0])


def passes(r, dt):
    a_ref, w_ref = reference(dt)
    k15, k2 = math.exp(-0.5 / GIDON_SCALE), math.exp(-1.0 / GIDON_SCALE)
    checks = dict(K15=abs(r["K15"] - k15) <= 0.10 * k15, K2=abs(r["K2"] - k2) <= 0.10 * k2,
                  a=abs(r["a"] / a_ref - 1) <= 0.2, width=abs(r["width"] / w_ref - 1) <= 0.2,
                  I_th=abs(r["I_th"] / I_TH_E0B - 1) <= 0.2, sharp=r["a08"] <= 0.2 * r["a"])
    return checks, all(checks.values())


def fit(dt=0.05, workers=7):
    """The whole fit: (stage 1 rows, stage 2 results, the best, its checks, passed)."""
    rng = np.random.default_rng(SEED)
    X = from_unit(rng.random((N_CAND, len(NAMES))))
    reference(dt)
    with Pool(workers) as pool:
        chunks = pool.map(_coarse_chunk, [(X[i:i + CHUNK], dt) for i in range(0, N_CAND, CHUNK)])
        rows = [dict(r, x=[float(v) for v in X[i]]) for i, r in enumerate(r for c in chunks for r in c)]
        order = sorted(range(N_CAND), key=lambda i: rows[i]["E"])[:N_BEST]
        refined = pool.map(_refine, [(X[i], dt) for i in order])
    best = min(refined, key=lambda r: r["E"])
    checks, ok = passes(best, dt)
    return dict(stage1=rows, top=[rows[i] for i in order], stage2=refined, best=best, checks=checks, passed=ok,
                reference=dict(zip(("a", "width"), reference(dt))))
