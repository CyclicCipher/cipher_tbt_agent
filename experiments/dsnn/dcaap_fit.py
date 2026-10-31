"""Gidon et al.'s channel dCaAP fitted to the dCaAP recorded in our own cell (LAB.md E0d, step 4).

One concept: the fit. The model is `dcaap_channels.py` at type A's zone (`tests/test_e0d.transplant`); this module
searches the ten constants their files expose, plus the two totals.

**The targets** (cell 171122005, Gidon et al.'s Fig. S8 / 2E data): the first event at I_th is 54.7 mV high and 4.3 ms
wide; at 1.107, 1.214, 1.321, 1.429 × I_th its normalized amplitude is 0.874, 0.824, 0.652, 0.515. Each error is
divided by its D1 tolerance (13.9 mV, 1.3 ms, 0.15).

**Each evaluation:** the candidate settled for 2 s; 300 ms steps into the zone, spiking off; I_th on a 20 pA grid from
20 to 1000 pA, then 5 pA within the bracket; then the four multiples. The event detector is D1's.

**The search:** 600 random candidates (seed 0) in the bounds; the best 10 refined by Nelder–Mead (in unit
coordinates), at most 60 evaluations each. Work spreads over processes; the arithmetic doesn't change.
"""
from dataclasses import replace
import math
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

from .cell import ChannelDCaAP
from .protocols import T0
from .sim import Stim, Wiring, simulate

NAMES = ("g_ca", "g_bk", "ma", "mb", "ha", "hb", "mtf", "htf", "zhalf", "taur")
G_CA, G_BK = 155.50883635269477, 9330.530190344238          # nS: their compartment's totals (the transplant)
LO = np.array([0.1 * G_CA, 0.1 * G_BK, 0.2, 0.2, 0.2, 0.2, 0.2 * 0.2, 0.2 * 0.02, 0.001, 20.0])
HI = np.array([10 * G_CA, 10 * G_BK, 5.0, 5.0, 5.0, 5.0, 5 * 0.2, 5 * 0.02, 0.1, 140.0])
LOG = np.array([True] * 9 + [False])
STEP = 300.0
FS = (1.107, 1.214, 1.321, 1.429)
TARGET = dict(amp=54.7, width=4.3, norm=(0.874, 0.824, 0.652, 0.515))
TOL = dict(amp=13.9, width=1.3, norm=0.15)
N_CAND, N_BEST, MAXFEV, SEED, CHUNK = 600, 10, 60, 0, 20


def from_unit(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    x = LO + u * (HI - LO)
    x[..., LOG] = np.exp(np.log(LO[LOG]) + u[..., LOG] * (np.log(HI[LOG]) - np.log(LO[LOG])))
    return x


def to_unit(x):
    x = np.asarray(x, float)
    u = (x - LO) / (HI - LO)
    u[..., LOG] = (np.log(x[..., LOG]) - np.log(LO[LOG])) / (np.log(HI[LOG]) - np.log(LO[LOG]))
    return u


def cells(X):
    """Type A with each candidate's channels at the zone, each settled for 2 s (as `test_e0d.transplant`)."""
    from .tests.test_e0d import transplant_base
    base = transplant_base()
    cs = [replace(base, cach=replace(base.cach, **dict(zip(NAMES, map(float, x))))) for x in X]
    r = simulate([Stim(spiking=False)], 2000.0, cs, wiring=Wiring(len(cs)), final=True)
    return [replace(c, rest={"V": r.final[0][j]["V"], "Q": r.final[0][j]["Q"]}) for j, c in enumerate(cs)]


def first_events(cs, amps, dt):
    """For each cell and amplitude (a list per cell), the first event (amp, width) of a 300 ms step, or None."""
    from .tests.test_e0d import zone_events
    z = cs[0].site("zone")
    n = len(amps[0])
    stims = []
    for k in range(n):
        st = Stim(spiking=False)
        for j in range(len(cs)):
            st.current(T0, T0 + STEP, z, amps[j][k], neuron=j)
        stims.append(st)
    r = simulate(stims, T0 + STEP + 10.0, cs, wiring=Wiring(len(cs)), dt=dt, rec_every=2, record=[z])
    out = []
    for j in range(len(cs)):
        row = []
        for k in range(n):
            ev = zone_events(r.t, np.asarray(r.V[k][j][0], float), T0, T0 + STEP)
            row.append((ev[0][1], ev[0][2]) if ev else None)
        out.append(row)
    return out


def measure(X, dt):
    """The fit's measure of each candidate: I_th (20 pA grid, then 5 pA), the first event there and at FS × I_th."""
    cs = cells(X)
    coarse = np.arange(20.0, 1000.0 + 1e-9, 20.0)
    ev = first_events(cs, [list(coarse)] * len(cs), dt)
    out = [None] * len(cs)
    fine_amps, live = [], []
    for j in range(len(cs)):
        hit = [k for k, e in enumerate(ev[j]) if e is not None]
        if not hit:
            out[j] = dict(E=1e3, I_th=None)
            continue
        hi = coarse[hit[0]]
        fine_amps.append([hi - 15.0, hi - 10.0, hi - 5.0, hi])
        live.append(j)
    if not live:
        return out
    sub = [cs[j] for j in live]
    evf = first_events(sub, fine_amps, dt)
    I_ths = []
    for i, j in enumerate(live):
        k = next(k for k, e in enumerate(evf[i]) if e is not None)
        I_ths.append(fine_amps[i][k])
    ev2 = first_events(sub, [[f * I for f in (1.0,) + FS] for I in I_ths], dt)
    for i, j in enumerate(live):
        e0 = ev2[i][0]
        if e0 is None:
            out[j] = dict(E=1e3, I_th=I_ths[i])
            continue
        a0, w0 = e0
        norm = [(e[0] / a0 if e is not None else 0.0) for e in ev2[i][1:]]
        E = ((a0 - TARGET["amp"]) / TOL["amp"]) ** 2 + ((w0 - TARGET["width"]) / TOL["width"]) ** 2
        E += sum(((n - t) / TOL["norm"]) ** 2 for n, t in zip(norm, TARGET["norm"]))
        out[j] = dict(E=float(E), I_th=float(I_ths[i]), amp=float(a0), width=float(w0), norm=[float(n) for n in norm])
    return out


def _chunk(args):
    X, dt = args
    return measure(X, dt)


def _refine(args):
    x0, dt = args
    seen = []

    def f(u):
        x = from_unit(u)
        r = dict(measure(np.array([x]), dt)[0], x=[float(v) for v in x])
        seen.append(r)
        return r["E"]

    minimize(f, to_unit(x0), method="Nelder-Mead", options=dict(maxfev=MAXFEV, xatol=1e-3, fatol=1e-4))
    return dict(min(seen, key=lambda r: r["E"]), evaluations=len(seen), start=[float(v) for v in x0])


def fit(dt=0.05, workers=7):
    rng = np.random.default_rng(SEED)
    X = from_unit(rng.random((N_CAND, len(NAMES))))
    with Pool(workers) as pool:
        chunks = pool.map(_chunk, [(X[i:i + CHUNK], dt) for i in range(0, N_CAND, CHUNK)], chunksize=1)
        rows = [dict(r, x=[float(v) for v in X[i]]) for i, r in enumerate(r for c in chunks for r in c)]
        order = sorted(range(N_CAND), key=lambda i: rows[i]["E"])[:N_BEST]
        refined = pool.map(_refine, [(X[i], dt) for i in order], chunksize=1)
    best = min(refined, key=lambda r: r["E"])
    return dict(stage1=rows, top=[rows[i] for i in order], stage2=refined, best=best)
