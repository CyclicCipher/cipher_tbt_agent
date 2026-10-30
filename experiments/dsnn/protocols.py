"""The classic experiments, as interventions on the cell (LAB.md E0, "The protocols"), and the rules that set strengths.

One concept: what is done TO the cell. E0's pass criteria are in `tests/test_e0.py`; the viz shows the same runs.

**Each scenario has:**
- a builder: control values -> a `Stim`, or a (`Stim`, `Wiring`) for the networks;
- its controls, and its default runs, as side-by-side interventions;
- a plain question: what the comparison shows.
"""
from functools import lru_cache

import numpy as np

from .cell import BASAL, INDEX, Params, SOMA, TUFT, ZONE
from .sim import Stim, Wiring, simulate

T0 = 10.0                  # ms: when the first input arrives
B1 = [INDEX[n] for n in ("b1.p", "b1.m", "b1.d")]


# ---- the builders ----------------------------------------------------------------------------------------------
def n1(k=12, where="same", A=1, B=1):
    """Two groups of k contacts: A on b1.d; B on b1.m (the same branch) or b2.d (another). Spiking off."""
    st = Stim(spiking=False)
    if A:
        st.contacts(T0, INDEX["b1.d"], k)
    if B:
        st.contacts(T0, INDEX["b1.m" if where == "same" else "b2.d"], k)
    return st


def n2(pulse=1000.0, n=12, delta=2.0, use_pulse=1, use_tuft=1):
    """A 2 ms somatic pulse at 20 ms; n tuft contacts spread evenly over the 6 tuft segments at 20 + delta."""
    st = Stim()
    if use_pulse:
        st.current(20.0, 22.0, SOMA, pulse)
    if use_tuft:
        for c in TUFT:
            st.contacts(20.0 + delta, c, n / len(TUFT))
    return st


def n3(n=10, A=1, B=1):
    """Inputs A and B, n contacts each, on the zone at 10 ms."""
    st = Stim()
    st.contacts(T0, ZONE, n * (int(A) + int(B)))
    return st


def n4(s=4, interval=2.0, order="inward"):
    """s contacts at each of b1's segments, in order: inward (d, m, p) or outward (p, m, d). Spiking off."""
    st = Stim(spiking=False)
    sites = B1[::-1] if order == "inward" else B1
    for i, c in enumerate(sites):
        st.contacts(T0 + i * interval, c, s)
    return st


def n5(r_b=4.0, r_a=4.0, seed=0, T=2000.0):
    """Poisson input for T ms: 10 contacts per basal segment at r_b; 10 per tuft segment at r_a."""
    st = Stim()
    for c in BASAL:
        st.train(0.0, T, c, 10, r_b, (seed, 1, c))
    for c in TUFT:
        st.train(0.0, T, c, 10, r_a, (seed, 2, c))
    return st


def n6(c=10, pulse=500.0, delta=50.0, cond=1, test=1):
    """Conditioning: c contacts on b1.d at 10 ms. Test: a 2 ms somatic current pulse at 10 + delta (amendment 5)."""
    st = Stim()
    if cond:
        st.contacts(T0, INDEX["b1.d"], c)
    if test:
        st.current(T0 + delta, T0 + delta + 2.0, SOMA, pulse)
    return st


def branch(prefix):
    """A branch's three segments, e.g. branch("b1") -> [b1.p, b1.m, b1.d]."""
    return [INDEX["%s.%s" % (prefix, s)] for s in "pmd"]


def network(depth=2, rate1=20.0, rate2=0.0, top=0.0, contacts=44, ff=24, fb=12, seed=0, T=300.0):
    """2 cells per layer; every connection spreads its contacts evenly over one branch's three segments (a volley on
    one segment saturates it and never fires the soma: 36 contacts over a branch do).
    - **Two input fibres:** fibre f hits layer 1's cell f with `contacts` on branch b(f+1), and the other layer-1 cell
      with half as many on the same branch.
    - **Feedforward:** cell i of layer l reaches each cell of layer l + 1 on branch b(i+1), `ff` contacts.
    - **Feedback:** cell j of layer l + 1 reaches each cell of layer l on tuft branch t(j+1), `fb` contacts.
    - **Top-down:** a fibre into the top layer's tufts, `contacts` per spike over both tuft branches.
    Delays 1.5 ms. Cell index = 2 * layer + column."""
    wiring = Wiring(2 * depth)
    for l in range(depth - 1):
        for i in range(2):
            for j in range(2):
                for c in branch("b%d" % (i + 1)):
                    wiring.edges.append((2 * l + i, 2 * (l + 1) + j, c, ff / 3, 1.5))
                for c in branch("t%d" % (j + 1)):
                    wiring.edges.append((2 * (l + 1) + j, 2 * l + i, c, fb / 3, 1.5))
    st = Stim()
    rng = np.random.default_rng((seed, 77))
    for f, rate in enumerate((rate1, rate2)):
        for t in _poisson_times(rng, rate, T):
            for col in range(2):
                for c in branch("b%d" % (f + 1)):
                    st.contacts(t, c, contacts / 3 * (1.0 if col == f else 0.5), neuron=col)
    for t in _poisson_times(rng, top, T):
        for col in range(2):
            for c in TUFT:
                st.contacts(t, c, contacts / len(TUFT), neuron=2 * (depth - 1) + col)
    return st, wiring


def _poisson_times(rng, rate, T):
    if rate <= 0:
        return []
    n = rng.poisson(rate * T / 1000.0)
    return np.sort(rng.uniform(5.0, T, n)).tolist()


# ---- the rules that set strengths (LAB.md E0) ----------------------------------------------------------------
def _crosses(r, b, comp, level):
    return bool((r.V[b, 0, comp] >= level).any())


def zone_rheobase(p, dt, step=5.0, top=2000.0):
    """The smallest 100 ms step into the zone, in `step` pA, that takes the zone to the dCaAP's threshold (no dCaAP)."""
    q = p.with_(dca_w=0.0)
    amps = np.arange(step, top + step, step)
    r = simulate([Stim(spiking=False).current(T0, T0 + 100, ZONE, a) for a in amps], 120.0, q, dt=dt)
    hit = [b for b in range(len(amps)) if _crosses(r, b, ZONE, p.dca_theta)]
    return float(amps[hit[0]])


def dca_w_rule(p, dt, I_rheo):
    """The smallest w, in 10 pA steps, for which the dCaAP evoked by the threshold-level step drives the zone to 0 mV."""
    ws = np.arange(10.0, 5010.0, 10.0)
    reaches = lambda w: _crosses(simulate([Stim(spiking=False).current(T0, T0 + 100, ZONE, I_rheo)], 150.0,
                                          p.with_(dca_w=w), dt=dt), 0, ZONE, 0.0)
    lo, hi = 0, len(ws) - 1                           # bisection on the grid: the zone's peak rises with w
    if not reaches(ws[hi]):
        raise RuntimeError("no w up to 5000 pA reaches 0 mV")
    while lo < hi:
        mid = (lo + hi) // 2
        if reaches(ws[mid]):
            hi = mid
        else:
            lo = mid + 1
    assert lo == 0 or not reaches(ws[lo - 1])
    return float(ws[lo])


def strengths(p, dt):
    """Every rule-set input strength for N2, N3 and N6 (N5's rates: `n5_rates`)."""
    out = {}
    pulses = np.arange(50.0, 5050.0, 50.0)
    r = simulate([n2(pulse=a, use_tuft=0) for a in pulses], 120.0, p, dt=dt)
    one = [b for b in range(len(pulses)) if len(r.spikes[b][0]) == 1]
    out["n2_pulse"] = float(pulses[one[0]])
    ns = np.arange(2, 202, 2)
    r = simulate([n2(n=n, use_pulse=0) for n in ns], 120.0, p, dt=dt)
    first = [b for b in range(len(ns)) if r.dcaap[b][0]]
    out["n2_tuft"] = float(ns[first[0]] - 2)
    ns = np.arange(1, 121)
    r = simulate([n3(n=n, B=0) for n in ns], 120.0, p, dt=dt)
    first = [b for b in range(len(ns)) if r.dcaap[b][0]]
    out["n3_n"] = float(ns[first[0]])
    cs = np.arange(1, 61)
    r = simulate([n6(c=c, test=0) for c in cs], 150.0, p, dt=dt)
    d = INDEX["b1.d"]
    long = [b for b in range(len(cs)) if _longest_above(r.V[b, 0, d], -40.0, r.t) >= 30.0]
    out["n6_c"] = float(cs[long[0]])
    pulses = np.arange(50.0, 5050.0, 50.0)            # the test: the largest pulse that alone gives no spike
    r = simulate([n6(pulse=a, delta=0.0, cond=0) for a in pulses], 60.0, p, dt=dt)
    quiet = [b for b in range(len(pulses)) if not r.spikes[b][0]]
    out["n6_pulse"] = float(pulses[quiet[-1]]) if quiet else 0.0
    return out


def n5_rates(p, dt, seed=0):
    """r: the smallest basal rate (0.5-20 Hz) giving >= 3 events/s with no apical input; r': the smallest apical rate
    giving, with basal at r, a burst fraction >= 0.1."""
    grid = np.arange(0.5, 20.5, 0.5)
    r = simulate([n5(r_b=x, r_a=0.0, seed=seed) for x in grid], 2000.0, p, dt=dt)
    ev = [events(r.spikes[b][0])[0] / 2.0 for b in range(len(grid))]
    hit = [b for b in range(len(grid)) if ev[b] >= 3.0]
    rb = float(grid[hit[0]]) if hit else float("nan")
    r = simulate([n5(r_b=rb, r_a=x, seed=seed) for x in grid], 2000.0, p, dt=dt)
    bf = [events(r.spikes[b][0])[2] for b in range(len(grid))]
    hit = [b for b in range(len(grid)) if bf[b] >= 0.1]
    ra = float(grid[hit[0]]) if hit else float("nan")
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


def _longest_above(v, level, t):
    """The longest stretch (ms) for which v > level."""
    best = run = 0
    for x in v > level:
        run = run + 1 if x else 0
        best = max(best, run)
    return best * (t[1] - t[0]) if len(t) > 1 else 0.0


@lru_cache(maxsize=4)
def calibrated(dt=0.05, p: Params = Params()):
    """The default cell with dca_w set by its rule, and the rule-set strengths: (Params, {name: value})."""
    I_rheo = zone_rheobase(p, dt)
    p = p.with_(dca_Ith=I_rheo)
    w = dca_w_rule(p, dt, I_rheo)
    q = p.with_(dca_w=w)
    cal = dict(zone_rheobase=I_rheo, dca_w=w, **strengths(q, dt))
    return q, cal


# ---- the scenarios, for the viz --------------------------------------------------------------------------------
def scenarios(cal):
    """[{key, title, question, controls, runs, T, traces, network}]; control kinds: range, select, toggle."""
    rng = lambda name, label, lo, hi, step, default: dict(name=name, label=label, kind="range", lo=lo, hi=hi,
                                                          step=step, default=default)
    tog = lambda name, label, default=1: dict(name=name, label=label, kind="toggle", default=default)
    sel = lambda name, label, options, default: dict(name=name, label=label, kind="select", options=options,
                                                     default=default)
    tr = lambda *names: list(names)
    k, pulse, nt, n3n = 12, cal["n2_pulse"], cal["n2_tuft"], cal["n3_n"]
    c6, s6 = cal["n6_c"], cal["n6_pulse"]
    return [
        dict(key="n1", title="N1 · Do inputs add?", T=100.0, traces=tr("soma", "b1.d", "b1.m", "b2.d"),
             question="Two groups of synapses, first on the same thin branch, then on two different branches. "
                      "Same inputs, different places: compare what reaches the soma. (The soma's spiking is off, "
                      "so you see the raw sum.)",
             controls=[rng("k", "contacts per group", 1, 30, 1, k), sel("where", "B goes on", ["same", "other"], "same"),
                       tog("A", "group A (b1.d)"), tog("B", "group B")],
             runs=[("A alone", dict(k=k, where="same", A=1, B=0)), ("B alone, same branch", dict(k=k, where="same", A=0, B=1)),
                   ("A + B, same branch", dict(k=k, where="same", A=1, B=1)),
                   ("A + B, other branch", dict(k=k, where="other", A=1, B=1))]),
        dict(key="n2", title="N2 · Spike meets tuft input", T=120.0, traces=tr("soma", "trunk", "zone", "t1.d"),
             question="A somatic spike travels back up the dendrite (the bAP). Tuft input arrives Δt ms later. "
                      "Alone, each does little. Does the combination fire a burst, and how tightly must they coincide?",
             controls=[rng("delta", "tuft input Δt (ms)", -20, 40, 1, 2), rng("n", "tuft contacts", 0, 120, 2, nt),
                       rng("pulse", "somatic pulse (pA)", 0, 5000, 50, pulse), tog("use_pulse", "somatic pulse"),
                       tog("use_tuft", "tuft input")],
             runs=[("pulse alone", dict(pulse=pulse, n=nt, delta=2, use_pulse=1, use_tuft=0)),
                   ("tuft alone", dict(pulse=pulse, n=nt, delta=2, use_pulse=0, use_tuft=1)),
                   ("both, Δt = 2 ms", dict(pulse=pulse, n=nt, delta=2, use_pulse=1, use_tuft=1)),
                   ("both, Δt = 20 ms", dict(pulse=pulse, n=nt, delta=20, use_pulse=1, use_tuft=1))]),
        dict(key="n3", title="N3 · XOR in one neuron", T=120.0, traces=tr("soma", "zone", "trunk"),
             question="The human dCaAP is biggest for input just at threshold and shrinks for stronger input. "
                      "So: A alone, B alone, both. Watch the zone's spike and the soma.",
             controls=[rng("n", "contacts per input", 1, 80, 1, n3n), tog("A", "input A"), tog("B", "input B")],
             runs=[("neither", dict(n=n3n, A=0, B=0)), ("A", dict(n=n3n, A=1, B=0)), ("B", dict(n=n3n, A=0, B=1)),
                   ("A + B", dict(n=n3n, A=1, B=1))]),
        dict(key="n4", title="N4 · Does order matter?", T=100.0, traces=tr("soma", "b1.d", "b1.m", "b1.p"),
             question="The same synapses along one branch, activated toward the soma (inward) or away from it "
                      "(outward). Same inputs, different order. (The soma's spiking is off.)",
             controls=[rng("s", "contacts per site", 1, 20, 1, 6), rng("interval", "interval (ms)", 0.5, 10, 0.5, 2),
                       sel("order", "order", ["inward", "outward"], "inward")],
             runs=[("inward", dict(s=6, interval=2, order="inward")), ("outward", dict(s=6, interval=2, order="outward"))]),
        dict(key="n5", title="N5 · Two messages on one axon", T=2000.0, traces=tr("soma", "zone", "b1.d"),
             question="Random input to the basal branches and to the tuft. Does basal input set how often the cell "
                      "fires, and tuft input how often it bursts?",
             controls=[rng("r_b", "basal rate (Hz)", 0, 20, 0.5, 4), rng("r_a", "tuft rate (Hz)", 0, 20, 0.5, 4),
                       rng("seed", "seed", 0, 9, 1, 0)],
             runs=[("basal low, tuft low", dict(r_b=4, r_a=2, seed=0)), ("basal high, tuft low", dict(r_b=8, r_a=2, seed=0)),
                   ("basal low, tuft high", dict(r_b=4, r_a=8, seed=0))]),
        dict(key="n6", title="N6 · A plateau as memory", T=400.0, traces=tr("soma", "b1.d", "b2.d"),
             question="A strong input to one branch starts an NMDA plateau. Δ ms later, a small current pulse into the "
                      "soma, too weak to fire it alone. Does the cell still 'remember' the first input when the "
                      "pulse arrives?",
             controls=[rng("delta", "test delay Δ (ms)", 10, 380, 10, 50), rng("c", "conditioning contacts", 0, 60, 1, c6),
                       rng("pulse", "test pulse (pA)", 0, 5000, 50, s6), tog("cond", "conditioning"), tog("test", "test")],
             runs=[("test alone", dict(c=c6, pulse=s6, delta=50, cond=0, test=1)),
                   ("conditioning alone", dict(c=c6, pulse=s6, delta=50, cond=1, test=0)),
                   ("both, Δ = 50 ms", dict(c=c6, pulse=s6, delta=50, cond=1, test=1)),
                   ("both, Δ = 300 ms", dict(c=c6, pulse=s6, delta=300, cond=1, test=1))]),
    ] + [
        dict(key="net%d" % depth, title="Network 2 × %d" % depth, T=300.0, traces=tr("soma", "zone"), network=depth,
             question=("Fibre 1 drives cell L1a strongly and L1b weakly; fibre 2 the reverse. Each layer feeds the next "
                       "layer's basal branches, and feeds back to the layer below's tufts. One fibre alone, or both: "
                       "which cells fire, and does the activity climb the layers?"
                       if depth == 2 else
                       "The same, three layers deep: the middle layer gets feedforward input on its basal branches and "
                       "feedback on its tufts. Follow a volley up, and the feedback down."),
             controls=[rng("rate1", "fibre 1 rate (Hz)", 0, 100, 5, 20), rng("rate2", "fibre 2 rate (Hz)", 0, 100, 5, 0),
                       rng("top", "top-down rate (Hz)", 0, 100, 5, 0), rng("contacts", "fibre contacts", 0, 80, 1, 44),
                       rng("ff", "feedforward contacts", 0, 60, 1, 24), rng("fb", "feedback contacts", 0, 60, 1, 12),
                       rng("seed", "seed", 0, 9, 1, 0)],
             runs=[("fibre 1", dict(rate1=20, rate2=0)), ("fibre 2", dict(rate1=0, rate2=20)),
                   ("both", dict(rate1=20, rate2=20)), ("both + top-down", dict(rate1=20, rate2=20, top=20))])
        for depth in (2, 3)
    ]


BUILDERS = dict(n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6)


def build(key, controls):
    """(Stim, Wiring or None, T) for a scenario and its control values."""
    if key in ("net2", "net3"):
        st, wiring = network(depth=2 if key == "net2" else 3, **controls)
        return st, wiring
    return BUILDERS[key](**controls), None
