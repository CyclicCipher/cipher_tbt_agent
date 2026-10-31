"""The classic experiments, as interventions on a cell (LAB.md E0, "The protocols"; E0b, "Protocols and pass criteria"),
and the rules that set input strengths.

One concept: what is done TO a cell. Every builder takes the cell, whose named sites (`cell.groups`) place the input.
The pass criteria are in `tests/test_e0b.py`; the viz shows the same runs.

**Spread inputs** ("n contacts over the tuft") are spread over a group's compartments in proportion to their length,
so the input doesn't change when the cut into compartments does (check 6).
"""
from functools import lru_cache
import math

import numpy as np

from .cells import TYPES
from .sim import Stim, Wiring, simulate

T0 = 10.0                  # ms: when the first input arrives


def spread(cell, group, n):
    """[(comp, contacts)]: n contacts over a group, in proportion to each compartment's length."""
    comps = cell.groups[group]
    L = np.array([cell.length[c] for c in comps])
    return [(c, n * l / L.sum()) for c, l in zip(comps, L)]


def place(cell, seclist, n, rng, min_dist=None):
    """n synapse positions drawn uniformly by length over a section list (Gidon et al.'s `TPointList.setrand`), or,
    with min_dist, only from points farther than min_dist µm from the soma (their `setrand_distance`, by rejection).
    Returns [(compartment, section, x)]."""
    t = cell.tree
    secs = cell.seclists[seclist]
    L = np.array([t.length(s) for s in secs])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    out = []
    while len(out) < n:
        u = rng.uniform(0.0, cum[-1])
        i = min(int(np.searchsorted(cum, u, side="right")) - 1, len(secs) - 1)
        s, x = secs[i], (u - cum[i]) / L[i]
        if min_dist is not None and t.dist(s, x) <= min_dist:
            continue
        out.append((comp_at(cell, s, x), s, x))
    return out


def comp_at(cell, s, x):
    """The compartment that holds point x of section s."""
    if s == "soma":
        return 0
    L = cell.where[s][-1][2]
    for c, a, b in cell.where[s]:
        if x * L <= b + 1e-9:
            return c
    return cell.where[s][-1][0]


# ---- the builders ----------------------------------------------------------------------------------------------
def n1(cell, k=12, where="same", A=1, B=1):
    """Two groups of k contacts: A on b1.d; B on b1.m (the same branch) or b2.d (another). Spiking off."""
    st = Stim(spiking=False)
    if A:
        st.contacts(T0, cell.site("b1.d"), k)
    if B:
        st.contacts(T0, cell.site("b1.m" if where == "same" else "b2.d"), k)
    return st


def n2(cell, pulse=1000.0, n=12, delta=2.0, use_pulse=1, use_tuft=1):
    """A 2 ms somatic pulse at 20 ms; n contacts spread over the tuft at 20 + delta."""
    st = Stim()
    if use_pulse:
        st.current(20.0, 22.0, 0, pulse)
    if use_tuft:
        for c, m in spread(cell, "tuft", n):
            st.contacts(20.0 + delta, c, m)
    return st


def n3(cell, n=10, A=1, B=1):
    """Inputs A and B, n contacts each, on the zone at 10 ms."""
    return Stim().contacts(T0, cell.site("zone"), n * (int(A) + int(B)))


def n4(cell, s=2, interval=2.0, order="inward"):
    """s contacts at each of the 9 N4 sites along basal 1, in order: inward (distal first) or outward. Spiking off."""
    st = Stim(spiking=False)
    sites = cell.groups["n4"] if order == "inward" else cell.groups["n4"][::-1]
    for i, c in enumerate(sites):
        st.contacts(T0 + i * interval, c, s)
    return st


def n5(cell, r_b=4.0, r_a=4.0, seed=0, T=2000.0):
    """Poisson input for T ms: 60 contacts spread over the basal dendrites at r_b each; 60 over the tuft at r_a."""
    st = Stim()
    for c, m in spread(cell, "basal", 60):
        st.train(0.0, T, c, m, r_b, (seed, 1, c))
    for c, m in spread(cell, "tuft", 60):
        st.train(0.0, T, c, m, r_a, (seed, 2, c))
    return st


def n6(cell, c=10, pulse=500.0, delta=50.0, cond=1, test=1):
    """Conditioning: c contacts on b1.d at 10 ms. Test: a 2 ms somatic current pulse at 10 + delta (amendment 5)."""
    st = Stim()
    if cond:
        st.contacts(T0, cell.site("b1.d"), c)
    if test:
        st.current(T0 + delta, T0 + delta + 2.0, 0, pulse)
    return st


def network(cells, rate1=20.0, rate2=0.0, top=0.0, contacts=44, ff=24, fb=12, seed=0, T=300.0):
    """2 cells per layer, `cells` = the cell of each, layer by layer (index = 2 * layer + column).
    - **Two input fibres:** fibre f hits layer 1's cell f with `contacts` spread over its basal dendrites, and the
      other layer-1 cell with half as many.
    - **Feedforward:** every cell of layer l reaches each cell of layer l + 1 with `ff` contacts over its basal side.
    - **Feedback:** every cell of layer l + 1 reaches each cell of layer l with `fb` contacts over its tuft.
    - **Top-down:** a fibre into the top layer's tufts, `contacts` per spike.
    Delays 1.5 ms. Not E2's wiring: a stand-in for the cost and the viz."""
    depth = len(cells) // 2
    wiring = Wiring(len(cells))
    for l in range(depth - 1):
        for i in range(2):
            for j in range(2):
                src, dst = 2 * l + i, 2 * (l + 1) + j
                for c, m in spread(cells[dst], "basal", ff):
                    wiring.edges.append((src, dst, c, m, 1.5))
                for c, m in spread(cells[src], "tuft", fb):
                    wiring.edges.append((dst, src, c, m, 1.5))
    st = Stim()
    rng = np.random.default_rng((seed, 77))
    for f, rate in enumerate((rate1, rate2)):
        for t in _poisson_times(rng, rate, T):
            for col in range(2):
                for c, m in spread(cells[col], "basal", contacts * (1.0 if col == f else 0.5)):
                    st.contacts(t, c, m, neuron=col)
    for t in _poisson_times(rng, top, T):
        for col in range(2):
            j = 2 * (depth - 1) + col
            for c, m in spread(cells[j], "tuft", contacts):
                st.contacts(t, c, m, neuron=j)
    return st, wiring


def _poisson_times(rng, rate, T):
    if rate <= 0:
        return []
    n = rng.poisson(rate * T / 1000.0)
    return np.sort(rng.uniform(5.0, T, n)).tolist()


# ---- the rules that set strengths (LAB.md E0, E0b) -------------------------------------------------------------
def first(values, ok):
    """The first value for which ok holds, or nan."""
    hit = [v for v, o in zip(values, ok) if o]
    return float(hit[0]) if hit else float("nan")


def zone_event(cell, r, b):
    """Run b gave a dCaAP: the phenomenological one fired, or the channel's current reached half its peak in the
    I_th step (LAB.md E0c-A). Channel cells need the zone recorded with its Ca²⁺ current (`zone_record`)."""
    if cell.cach is not None:
        return float(-r.i_ca(b, cell.cach.zone).min()) >= 0.5 * cell.cach.i_ref
    return bool(r.dcaap[b][0])


def zone_record(cell):
    """simulate() keywords that let `zone_event` read a run."""
    return dict(record=[0, cell.site("zone")], record_ca=True, rec_every=2) if cell.cach is not None else {}


def strengths(cell, dt, kind):
    """Every rule-set input strength (N5's rates: `n5_rates`)."""
    out = {}
    pulses = np.arange(50.0, 5050.0, 50.0)
    r = simulate([n2(cell, pulse=a, use_tuft=0) for a in pulses], 120.0, cell, dt=dt)
    out["n2_pulse"] = first(pulses, [len(r.spikes[b][0]) == 1 for b in range(len(pulses))])
    if kind == "A":                    # n: the smallest even n whose tuft input alone gives a dCaAP, minus 2
        zr = zone_record(cell)
        ns = np.arange(2, 402, 2)
        r = simulate([n2(cell, n=n, use_pulse=0) for n in ns], 120.0, cell, dt=dt, **zr)
        out["n2_tuft"] = first(ns, [zone_event(cell, r, b) for b in range(len(ns))]) - 2
        ns = np.arange(1, 121)
        r = simulate([n3(cell, n=n, B=0) for n in ns], 120.0, cell, dt=dt, **zr)
        out["n3_n"] = first(ns, [zone_event(cell, r, b) for b in range(len(ns))])
        cs = np.arange(1, 61)
        d = cell.site("b1.d")
        r = simulate([n6(cell, c=c, test=0) for c in cs], 150.0, cell, dt=dt, record=[0, d])
        out["n6_c"] = first(cs, [longest_above(r.v(b, d), -40.0, r.t) >= 30.0 for b in range(len(cs))])
        r = simulate([n6(cell, pulse=a, delta=0.0, cond=0) for a in pulses], 60.0, cell, dt=dt)
        quiet = [a for b, a in enumerate(pulses) if not r.spikes[b][0]]
        out["n6_pulse"] = float(quiet[-1]) if quiet else 0.0
    else:                              # type B (amendment): the smallest even n whose tuft input alone fires the soma
        ns = np.arange(2, 2002, 2)
        r = simulate([n2(cell, n=n, use_pulse=0) for n in ns], 120.0, cell, dt=dt)
        out["n2_tuft"] = first(ns, [bool(r.spikes[b][0]) for b in range(len(ns))]) - 2
    return out


def n5_rates(cell, dt, seed=0):
    """r: the smallest basal rate (0.5-20 Hz) giving >= 3 events/s with no apical input; r': the smallest apical rate
    giving, with basal at r, a burst fraction >= 0.1. Searches run 2 s, with seed 0."""
    grid = np.arange(0.5, 20.5, 0.5)
    r = simulate([n5(cell, r_b=x, r_a=0.0, seed=seed) for x in grid], 2000.0, cell, dt=dt)
    rb = first(grid, [events(r.spikes[b][0])[0] / 2.0 >= 3.0 for b in range(len(grid))])
    if rb != rb:
        return rb, float("nan")
    r = simulate([n5(cell, r_b=rb, r_a=x, seed=seed) for x in grid], 2000.0, cell, dt=dt)
    ra = first(grid, [events(r.spikes[b][0])[2] >= 0.1 for b in range(len(grid))])
    return rb, ra


def events(spikes, gap=16.0):
    """(events, bursts, burst fraction): an event is a spike > gap ms after the previous one, plus the spikes that
    follow within gap; a burst is an event of >= 2 spikes."""
    sizes = []
    last = -1e9
    for t in spikes:
        if t - last > gap:
            sizes.append(1)
        else:
            sizes[-1] += 1
        last = t
    n = len(sizes)
    bursts = sum(1 for s in sizes if s >= 2)
    return n, bursts, (bursts / n if n else 0.0)


def longest_above(v, level, t):
    """The longest stretch (ms) for which v > level."""
    best = run = 0
    for x in v > level:
        run = run + 1 if x else 0
        best = max(best, run)
    return best * (t[1] - t[0]) if len(t) > 1 else 0.0


@lru_cache(maxsize=8)
def calibrated(kind, dt=0.05, max_len=None):
    """A cell type with its rule-set strengths: (Cell, {name: value}). Type A at its coupled site (E0's protocols
    were written for a site that drives the soma)."""
    if kind == "A":
        return calibrate(TYPES[kind](site="coupled"), kind, dt)
    return calibrate(TYPES[kind](max_len) if max_len else TYPES[kind](), kind, dt)


def calibrate(cell, kind, dt=0.05):
    """(Cell, {name: value}): the rule-set strengths."""
    return cell, strengths(cell, dt, kind)


# ---- the scenarios, for the viz (updated after E0b's run) -------------------------------------------------------
BUILDERS = dict(n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6)
