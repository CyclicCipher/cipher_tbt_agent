"""E0c-B: type B's N2 diagnosed with Bahl's own protocol (LAB.md E0c). Run from experiments/:
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e0c                 (writes dsnn/runs/e0c_dt0.05_B.json)
    ../venv312/Scripts/python.exe -m pytest dsnn/tests/test_e0c.py -q    (a fast check)

**What it scores:** the port against Bahl et al. 2012's model 2 numbers (Fig. 8), and the diagnosis of N2.

E0c-A (one inactivating Ca²⁺ channel as the dCaAP) failed and was replaced by E0d's channel model; its code is at
commit ee235ab (`dcaap_fit.py`, this file's part A).
"""
import json
import math
from multiprocessing import Pool
import os
import sys
import time

import numpy as np

from dsnn import protocols as P
from dsnn.cells import l5
from dsnn.sim import Stim, simulate
from dsnn.tests import test_e0b as E

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # dsnn/


# ---- E0c-B ------------------------------------------------------------------------------------------------------
BAHL = dict(soma_threshold=900.0, I_CA=3600.0, I_BAC=1800.0, C=0.5)
SITES = ("apical.0", "apical.1", "apical.2", "apical.3", "apical.4", "tuft.0", "tuft.1")
T_PULSE, T_DEND, T_END = 100.0, 106.0, 300.0
AMPS = np.arange(100.0, 8000.0 + 1e-9, 100.0)


def epsc(st, t0, comp, amp, neuron=0, dur=80.0, piece=0.1):
    """Bahl's dendritic current, I(t) = amp·(1 − e^(−t/2 ms))·e^(−t/8 ms) from t0, as 0.1 ms pieces (each at its
    midpoint's value)."""
    for k in range(int(round(dur / piece))):
        t = (k + 0.5) * piece
        st.current(t0 + k * piece, t0 + (k + 1) * piece, comp, amp * (1.0 - math.exp(-t / 2.0)) * math.exp(-t / 8.0),
                   neuron)
    return st


def tuft_integrals(r, b, tuft, rest_tuft):
    """From the first somatic spike t_s, the tuft's ∫(V − V_rest) in mV·ms: I1 over [t_s, t_s + 50], I2 over
    [t_s + 100, t_s + 150], pre over [t_s − 10, t_s], and the largest over any 10 ms in [t_s, t_s + 50] (0.5 ms
    steps). None without a somatic spike."""
    s = r.spikes[b][0]
    if not s:
        return None
    ts, t = s[0], r.t
    v = np.asarray(r.v(b, tuft), float) - rest_tuft
    h = t[1] - t[0]
    integ = lambda a, z: float(v[(t >= a) & (t < z)].sum() * h)
    return dict(I1=integ(ts, ts + 50), I2=integ(ts + 100, ts + 150), pre=integ(ts - 10, ts),
                win10=max(integ(a, a + 10) for a in np.arange(ts, ts + 40 + 1e-9, 0.5)))


def ca_spike(r, b, tuft, rest_tuft, local):
    """A dendritic Ca²⁺ spike (amendment 8): Bahl's own number, ∫(V − V_rest) > 500 mV·ms within 10 ms of the tuft
    (a mean 50 mV above rest). With `local` (the current alone), over the 10 ms before the first somatic spike, as
    Bahl's local test; otherwise over some 10 ms of the 50 ms after it."""
    x = tuft_integrals(r, b, tuft, rest_tuft)
    if x is None:
        return False
    return x["pre"] > 500 if local else x["win10"] > 500


def _site_thresholds(args):
    site, pulse, dt = args
    cell = l5()
    c, tuft = cell.index(site), cell.site("tuft(0.5)")
    rest = float(cell.rest["V"][tuft])
    alone = [epsc(Stim(), T_DEND, c, a) for a in AMPS]
    bac = [epsc(Stim().current(T_PULSE, T_PULSE + 5.0, 0, pulse), T_DEND, c, a) for a in AMPS]
    r = simulate(alone + bac, T_END, cell, dt=dt, record=[0, tuft], rec_every=2)
    n = len(AMPS)
    i_ca = P.first(AMPS, [ca_spike(r, b, tuft, rest, True) for b in range(n)])
    i_bac = P.first(AMPS, [ca_spike(r, n + b, tuft, rest, False) for b in range(n)])
    b_bac = n + list(AMPS).index(i_bac) if i_bac == i_bac else None
    spikes_bac = len(r.spikes[b_bac][0]) if b_bac is not None else None
    req = tuft_integrals(r, b_bac, tuft, rest) if b_bac is not None else None
    C = (i_ca - i_bac) / i_ca if i_ca == i_ca and i_bac == i_bac else float("nan")
    match = (abs(i_ca / BAHL["I_CA"] - 1) <= 0.2 and abs(i_bac / BAHL["I_BAC"] - 1) <= 0.2
             and abs(C - BAHL["C"]) <= 0.1 and spikes_bac is not None and 2 <= spikes_bac <= 4)
    err = abs(i_ca / BAHL["I_CA"] - 1) + abs(i_bac / BAHL["I_BAC"] - 1) + abs(C - BAHL["C"]) / BAHL["C"]
    return dict(site=site, dist=float(cell.dist[c]), I_CA=i_ca, I_BAC=i_bac, C=C, spikes_at_I_BAC=spikes_bac,
                bahl_requirements_at_I_BAC=req, matches=bool(match), rel_err=err if err == err else None)


def part_b(dt, workers=7):
    t0 = time.time()
    cell = l5()
    tuft = cell.site("tuft(0.5)")
    rest = float(cell.rest["V"][tuft])
    pulses = np.arange(50.0, 3000.0 + 1e-9, 50.0)
    r = simulate([Stim().current(T_PULSE, T_PULSE + 5.0, 0, a) for a in pulses], T_END, cell, dt=dt)
    th = P.first(pulses, [len(r.spikes[b][0]) >= 1 for b in range(len(pulses))])
    out = dict(soma_threshold=th, soma_threshold_ok=abs(th / BAHL["soma_threshold"] - 1) <= 0.1)
    with Pool(workers) as pool:
        sites = pool.map(_site_thresholds, [(s, th, dt) for s in SITES])
    out["sites"] = sites
    out["port_passed"] = bool(out["soma_threshold_ok"] and any(s["matches"] for s in sites))
    scored = [s for s in sites if s["rel_err"] is not None]
    if not scored:
        out["verdict"] = "the port"
        return out
    best = min(scored, key=lambda s: s["rel_err"])
    out["best_site"] = best["site"]
    c = cell.index(best["site"])

    # 1. current against contacts, with the threshold pulse
    ns = np.arange(2, 1002, 2)
    rows = {}
    for label, place in (("at_site", lambda n: [(c, float(n))]), ("tuft_spread", lambda n: P.spread(cell, "tuft", n))):
        stims = []
        for n in ns:
            st = Stim().current(T_PULSE, T_PULSE + 5.0, 0, th)
            for comp, m in place(n):
                st.contacts(T_DEND, comp, m)
            stims.append(st)
        rr = simulate(stims, T_END, cell, dt=dt, record=[0, tuft], rec_every=2)
        ok = [ca_spike(rr, b, tuft, rest, False) for b in range(len(ns))]
        n0 = P.first(ns, ok)
        rows[label] = dict(n=n0, spikes=len(rr.spikes[list(ns).index(n0)][0]) if n0 == n0 else None)
    rr = simulate([epsc(Stim().current(T_PULSE, T_PULSE + 5.0, 0, th), T_DEND, c, best["I_BAC"])], T_END, cell, dt=dt)
    rows["current"] = dict(I=best["I_BAC"], spikes=len(rr.spikes[0][0]), times=rr.spikes[0][0])
    out["current_vs_contacts"] = rows

    # 2. the window: N2's Δt grid (N2's 2 ms pulse at 20 ms) with Bahl's current at I_BAC; and with Bahl's 5 ms pulse
    _, calB = P.calibrated("B", dt)
    deltas = (-10, -5, 0, 2, 5, 10, 20)
    win = {}
    for label, dur, amp in (("n2_pulse", 2.0, calB["n2_pulse"]), ("bahl_pulse", 5.0, th)):
        stims = [epsc(Stim().current(20.0, 20.0 + dur, 0, amp), 20.0 + d, c, best["I_BAC"]) for d in deltas]
        rr = simulate(stims, 150.0, cell, dt=dt)
        win[label] = dict(within30={d: E.spikes_in(rr, i, 20.0, 50.0) for i, d in enumerate(deltas)},
                          at20_in50=E.spikes_in(rr, deltas.index(20), 20.0, 70.0))
    out["window"] = win

    # 3. the verdict, as pre-registered
    cur_spikes = [rows["current"]["spikes"]] + [win[k]["within30"][d] for k in win for d in (0, 2, 5)]
    contact_spikes = [rows[k]["spikes"] for k in ("at_site", "tuft_spread") if rows[k]["spikes"] is not None]
    if not out["port_passed"]:
        out["verdict"] = "the port"
    elif rows["current"]["spikes"] in (2, 3, 4) and max(cur_spikes) < 3:
        out["verdict"] = "the criterion"
    elif max(cur_spikes) >= 3 and all(s < 3 for s in contact_spikes):
        out["verdict"] = "the synapses"
    else:
        out["verdict"] = "none of the three"
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---- the run ----------------------------------------------------------------------------------------------------
def report(res, say=print):
    say("E0c at dt = %g ms (%.1f s)" % (res["dt"], res["seconds"]))
    if "B" in res:
        b = res["B"]
        say("  B: soma threshold (5 ms) %.0f pA (Bahl 900) -> %s" % (b["soma_threshold"],
                                                                     "ok" if b["soma_threshold_ok"] else "off"))
        for s in b["sites"]:
            q = s.get("bahl_requirements_at_I_BAC") or {}
            say("    %-9s %4.0f um: I_CA %s, I_BAC %s, C %s, spikes at I_BAC %s; I1 %s I2 %s%s" % (
                s["site"], s["dist"], s["I_CA"], s["I_BAC"], None if s["C"] != s["C"] else round(s["C"], 3),
                s["spikes_at_I_BAC"], q and round(q["I1"]), q and round(q["I2"]), "  MATCHES" if s["matches"] else ""))
        say("    port %s; best site %s" % ("PASS" if b["port_passed"] else "FAIL", b.get("best_site")))
        if "current_vs_contacts" in b:
            say("    current vs contacts: %s" % b["current_vs_contacts"])
            say("    window: %s" % b["window"])
        say("    verdict: %s" % b["verdict"])


def main(argv):
    dt = 0.05
    t0 = time.time()
    res = dict(dt=dt, amendment=8, B=part_b(dt))
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    with open(os.path.join(HERE, "runs", "e0c_dt0.05_B.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    report(res)


def test_e0c_small():
    """Fast: Bahl's current is the published shape."""
    st = epsc(Stim(), 0.0, 1, 1000.0)
    peak = max(a for _, _, _, _, a in st.steps)
    assert abs(peak - 1000.0 * 0.8 * math.exp(-2 * math.log(5) / 8)) < 5.0


if __name__ == "__main__":
    main(sys.argv[1:])
