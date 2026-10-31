"""E0d: Gidon's Ca²⁺ + BK model, verified, in type A (LAB.md E0d). Run from experiments/:
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e0d                 (steps 1-3; writes dsnn/runs/e0d_dt0.05.json)
    ../venv312/Scripts/python.exe -m pytest dsnn/tests/test_e0d.py -q    (a fast check)

**What it scores:**
- check P: the port against their Fig. S12 (the screenshot, digitized; `REF`);
- step 3, with their model at type A's zone: D1 (our cell's recorded dCaAP), N3 (XOR), the bAP, N1, N4, N6 (required),
  N2, N5 (reported).
"""
from dataclasses import replace
import json
import math
import os
import sys
import time

import numpy as np

from dsnn import dcaap_channels as dch
from dsnn import protocols as P
from dsnn.cell import Cell, ChannelDCaAP, passive
from dsnn.cells import l23
from dsnn.morphology import Geometry
from dsnn.sim import Stim, simulate
from dsnn.tests import test_e0b as E

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # dsnn/

# ---- check P: their figure, digitized (LAB.md E0d; screenshot.png of ModelDB 260178) ------------------------------
AMPS_P = [0.375 + 0.025 * k for k in range(11)]                         # nA
REF = {0.400: dict(heights=[], trough=-46.7),
       0.425: dict(heights=[64.4, 49.4, 43.5], trough=-44.1),
       0.450: dict(heights=[55.2, 36.8, 35.1, 32.6, 31.8, 31.0], trough=-44.0),
       0.475: dict(heights=[23.4, 21.8, 19.2, 18.4, 17.6, 17.6, 16.7], trough=-43.1),
       0.500: dict(heights=[10.9, 10.0, 10.0, 8.4, 9.2, 8.4, 8.4, 8.4, 8.4, 8.4], trough=-40.5),
       0.525: dict(heights=[6.7, 4.2, 4.2, 3.3, 3.3, 3.3, 3.3], trough=-37.9),
       0.550: dict(heights=[5.0], trough=-37.8),
       0.575: dict(heights=[4.2], trough=-37.7)}
REST_REF = -84.3
PX0, PX2000 = 50.0, 541.0                                                  # the screenshot's x axis: px of 0 and 2000 ms


def port_cell():
    """Their compartment (L 100, diam 99; cm 0.45; g_pas 3.5e-5 S/cm²; e_pas -86) with their channels at their values."""
    gca_d, gbk_d, area, k = dch.published()
    geo = Geometry(["dend"], np.array([-1]), np.array([0.0]), np.array([area]), np.array([0.0]), np.array([0.0]), {})
    cell = passive(geo, 0.45, 1.0 / 3.5e-5, -86.0, "Gidon Fig. S12")
    cell.rest = {"V": np.array([-86.0])}
    cell.groups = {"soma": [0], "zone": [0]}
    return replace(cell, cach=ChannelDCaAP(0, g_ca=gca_d * area * 10.0, g_bk=gbk_d * area * 1e-3, area=area, **k))


def columns(t, v):
    """The trace cut into the screenshot's pixel columns: (column centre ms, max, min)."""
    ms_per_px = 2000.0 / (PX2000 - PX0)
    px = np.floor((t / ms_per_px) + PX0 + 0.5).astype(int)
    out = []
    for p in np.unique(px):
        sel = px == p
        out.append(((p - PX0) * ms_per_px, v[sel].max(), v[sel].min()))
    return np.array(out)


def column_events(cols):
    """The digitizer's detector: local maxima of the column maxima in 560-1490 ms, >= 3 mV above the lowest column
    maximum in the 60 ms before. Returns (events [(ms, height)], trough, rest)."""
    st = cols[(cols[:, 0] > 560) & (cols[:, 0] < 1490)]
    top = st[:, 1]
    ev = []
    for i in range(1, len(st) - 1):
        if top[i] >= top[i - 1] and top[i] > top[i + 1]:
            pre = top[(st[:, 0] > st[i, 0] - 60) & (st[:, 0] < st[i, 0])]
            if len(pre) and top[i] - pre.min() >= 3.0:
                ev.append((float(st[i, 0]), float(top[i] - pre.min())))
    rest = float(cols[(cols[:, 0] > 100) & (cols[:, 0] < 450), 1].mean())
    return ev, float(np.median(st[:, 2])), rest


def check_p(dt):
    cell = port_cell()
    stims = [Stim(spiking=False).current(500.0, 1500.0, 0, 1000.0 * a) for a in AMPS_P]
    r = simulate(stims, 2000.0, cell, dt=dt, rec_every=max(1, int(round(0.1 / dt))), rec_dtype=np.float64)
    rows = {}
    for b, a in enumerate(AMPS_P):
        ev, trough, rest = column_events(columns(r.t, np.asarray(r.V[b][0][0])))
        rows[round(a, 3)] = dict(n=len(ev), times=[round(x, 0) for x, _ in ev], heights=[round(h, 1) for _, h in ev],
                                 trough=round(trough, 1), rest=round(rest, 1), peak=round(float(r.vmax[b][0][0]), 1))
    return rows, r.seconds


def score_p(rows):
    """The pre-registered pass criteria of check P."""
    med = lambda h: float(np.median(h[1:])) if len(h) > 1 else float("nan")
    ok = {}
    ok["rest"] = all(abs(x["rest"] - REST_REF) <= 3.0 for x in rows.values())
    ok["none_below"] = rows[0.375]["n"] == 0 and rows[0.4]["n"] == 0
    for a in (0.425, 0.45, 0.475, 0.5):
        ok["count_%g" % a] = abs(rows[a]["n"] - len(REF[a]["heights"])) <= 1
    for a in (0.425, 0.45, 0.475):
        ok["height_%g" % a] = abs(med(rows[a]["heights"]) / med(REF[a]["heights"]) - 1.0) <= 0.2
    ok["height_0.5"] = abs(med(rows[0.5]["heights"]) - med(REF[0.5]["heights"])) <= 3.0
    ok["at_most_one_above"] = rows[0.55]["n"] <= 1 and rows[0.575]["n"] <= 1
    ok["troughs"] = all(abs(rows[a]["trough"] - REF[a]["trough"]) <= 3.0 for a in REF)
    return ok, all(ok.values())


# ---- step 2: their model at type A's zone --------------------------------------------------------------------------
def transplant(**kw):
    """Type A with their channels at the zone (totals as in their compartment; the pool over their area), settled
    for 2 s from E_L. kw: overrides of `ChannelDCaAP`'s constants (step 4)."""
    gca_d, gbk_d, area, k = dch.published()
    base = replace(l23(), dcaap=None)
    k = dict(k, g_ca=gca_d * area * 10.0, g_bk=gbk_d * area * 1e-3, area=area)
    k.update(kw)
    cell = replace(base, cach=ChannelDCaAP(base.site("zone"), **k))
    r = simulate([Stim(spiking=False)], 2000.0, cell, final=True)
    return replace(cell, rest={"V": r.final[0][0]["V"], "Q": r.final[0][0]["Q"]})


def zone_events(t, v, t0, t1):
    """D1's detector at full resolution: local maxima of V in [t0, t1] (the largest within ±2 ms) that rise >= 10 mV
    above the minimum of the 20 ms before, with a peak dV/dt >= 5 mV/ms on the way up. [(peak ms, amplitude, width)]"""
    h = t[1] - t[0]
    dv = np.gradient(v, h)
    w2, w20 = int(round(2.0 / h)), int(round(20.0 / h))
    out = []
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
    for i in range(max(i0, w20, 1), min(i1, len(v) - 1)):
        if not (v[i] >= v[i - 1] and v[i] > v[i + 1] and v[i] >= v[max(0, i - w2):i + w2 + 1].max()):
            continue
        j = i - w20 + int(np.argmin(v[i - w20:i]))
        rise = float(v[i] - v[j])
        if rise < 10.0 or dv[j:i + 1].max() < 5.0:
            continue
        half = v[j] + rise / 2.0
        a = i
        while a > j and v[a] > half:
            a -= 1
        z = i
        while z < len(v) - 1 and v[z] > half:
            z += 1
        out.append((float(t[i]), rise, float((z - a) * h)))
    return out


def d1(cell, dt):
    """D1: 1 s steps into the zone, spiking off. I_th on a 5 pA grid; then 1.107-1.429 x I_th (our cell's recorded
    ratios). The first event's amplitude and width; the repeated events reported."""
    z = cell.site("zone")
    kw = dict(dt=dt, rec_every=2, record=[0, z], record_ca=True)
    grid = np.arange(5.0, 2000.0 + 1e-9, 5.0)
    T = P.T0 + 1000.0 + 20.0
    r = simulate([Stim(spiking=False).current(P.T0, P.T0 + 1000.0, z, a) for a in grid], T, cell, **kw)
    has = [bool(zone_events(r.t, np.asarray(r.v(b, z), float), P.T0, P.T0 + 1000.0)) for b in range(len(grid))]
    I_th = P.first(grid, has)
    if I_th != I_th:
        return dict(I_th=None, passed=False, note="no event up to 2 nA")
    fs = (1.0, 1.107, 1.214, 1.321, 1.429)
    r = simulate([Stim(spiking=False).current(P.T0, P.T0 + 1000.0, z, f * I_th) for f in fs], T, cell, **kw)
    rows = []
    for b, f in enumerate(fs):
        ev = zone_events(r.t, np.asarray(r.v(b, z), float), P.T0, P.T0 + 1000.0)
        ica = np.asarray(r.i_ca(b, z), float)
        first_ica = None
        if ev:
            sel = (r.t >= ev[0][0] - 10.0) & (r.t <= ev[0][0] + 10.0)
            first_ica = float(-ica[sel].min())
        rows.append(dict(f=f, I=f * I_th, events=len(ev), first_ms=ev[0][0] if ev else None,
                         amp=ev[0][1] if ev else 0.0, width=ev[0][2] if ev else None,
                         repeated_amps=[round(e[1], 1) for e in ev[1:6]], first_ica_pA=first_ica))
    a0 = rows[0]["amp"]
    norm = [x["amp"] / a0 if a0 > 0 else 0.0 for x in rows]
    want = (1.0, 0.874, 0.824, 0.652, 0.515)
    ok = dict(amp=abs(a0 - 54.7) <= 13.9, width=rows[0]["width"] is not None and abs(rows[0]["width"] - 4.4) <= 1.3,
              curve=all(abs(n - w) <= 0.15 for n, w in zip(norm[1:], want[1:])))
    return dict(I_th=I_th, rows=rows, norm=[round(n, 3) for n in norm], recorded=want, checks=ok,
                i_ref=rows[0]["first_ica_pA"], passed=all(ok.values()))


def step3(cell, dt):
    t0 = time.time()
    out = dict(rest_zone=float(cell.rest["V"][cell.site("zone")]), rest_soma=float(cell.rest["V"][0]))
    out["D1"] = d1(cell, dt)
    if not out["D1"].get("i_ref"):
        out["note"] = "no event at I_th: the event-based rules can't be set"
        return out
    cell = replace(cell, cach=replace(cell.cach, i_ref=out["D1"]["i_ref"]))
    cell, cal = P.calibrate(cell, "A", dt)
    out["calibration"] = cal
    peaks = lambda stims, T: _peaks(cell, stims, T, dt)
    out["bAP"] = dict(peak_fraction=peaks([P.n2(cell, pulse=cal["n2_pulse"], use_tuft=0)], 120.0)[0])
    out["bAP"]["passed"] = out["bAP"]["peak_fraction"] < 0.5
    for name, fn in (("N1", lambda: E.n1(cell, dt)), ("N2", lambda: E.n2(cell, cal, dt)),
                     ("N3", lambda: E.n3(cell, cal, dt)), ("N4", lambda: E.n4(cell, dt)),
                     ("N5", lambda: E.n5(cell, dt)), ("N6", lambda: E.n6(cell, cal, dt))):
        t = time.time()
        if name in ("N2", "N3", "N6") and any(cal[k] != cal[k] for k in cal):
            out[name] = dict(passed=False, note="a strength rule found no value: %s" % cal)
            continue
        out[name] = fn()
        out[name]["seconds"] = round(time.time() - t, 1)
    if out["N3"].get("n") is not None:
        n = cal["n3_n"]
        out["N3"]["channel_peaks"] = dict(zip(("none", "A", "B", "A+B"), peaks(
            [P.n3(cell, n, a, b) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))], 120.0)))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _peaks(cell, stims, T, dt):
    """The zone's Ca_HVA peak in each run, as a fraction of i_ref."""
    r = simulate(stims, T, cell, dt=dt, **P.zone_record(cell))
    return [round(float(-r.i_ca(b, cell.cach.zone).min()) / cell.cach.i_ref, 3) for b in range(len(stims))]


# ---- the run -----------------------------------------------------------------------------------------------------
def report(o, say=print):
    say("E0d (%s s)" % o["seconds"])
    for dt, (rows, secs) in o["P"].items():
        say("  check P at dt %s (%.1f s):" % (dt, secs))
        for a, x in rows.items():
            say("    %.3f nA: %d events %s heights %s; trough %.1f, rest %.1f, peak %.1f" % (
                float(a), x["n"], x["times"], x["heights"], x["trough"], x["rest"], x["peak"]))
    say("  check P: %s -> %s" % (o["P_checks"], "PASS" if o["P_passed"] else "FAIL"))
    if "step3" in o:
        s = o["step3"]
        say("  type A with their model: rest zone %.1f, soma %.1f mV" % (s["rest_zone"], s["rest_soma"]))
        d = s["D1"]
        say("  D1: I_th %s pA; norm %s (recorded %s); checks %s -> %s" % (
            d.get("I_th"), d.get("norm"), d.get("recorded"), d.get("checks"), "PASS" if d.get("passed") else "FAIL"))
        for x in d.get("rows", []):
            say("    %s" % {k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()})
        if "calibration" in s:
            pf = lambda r: "PASS" if r.get("passed") else "FAIL"
            say("  calibration %s" % s["calibration"])
            say("  bAP: Ca peak %.3f of i_ref -> %s" % (s["bAP"]["peak_fraction"], pf(s["bAP"])))
            say("  N1 max same %s other %s -> %s" % (s["N1"].get("max_same_ratio"), s["N1"].get("other_range"),
                                                     pf(s["N1"])))
            say("  N2 (reported) %s -> %s" % ({k: s["N2"].get(k) for k in ("pulse_alone", "tuft_alone",
                                                                            "spikes_within_30ms")}, pf(s["N2"])))
            say("  N3 n %s spikes %s Ca peaks %s -> %s" % (s["N3"].get("n"), s["N3"].get("spikes"),
                                                           s["N3"].get("channel_peaks"), pf(s["N3"])))
            say("  N4 -> %s;  N5 (reported) -> %s;  N6 %s -> %s" % (pf(s["N4"]), pf(s["N5"]), s["N6"].get("rows"),
                                                                   pf(s["N6"])))
        elif "note" in s:
            say("  %s" % s["note"])


def main(argv):
    t0 = time.time()
    o = dict(P={})
    for dt in (0.0025, 0.00125):
        rows, secs = check_p(dt)
        o["P"][str(dt)] = (rows, round(secs, 1))
    rows = o["P"]["0.0025"][0]
    o["P_checks"], o["P_passed"] = score_p(rows)
    fine = o["P"]["0.00125"][0]
    o["P_step_size"] = all(rows[a]["n"] == fine[a]["n"] for a in rows) and all(
        abs(h1 - h2) <= 0.02 * max(abs(h2), 1.0) for a in rows for h1, h2 in zip(rows[a]["heights"], fine[a]["heights"]))
    if o["P_passed"]:
        o["step3"] = step3(transplant(), 0.05)
    o["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    with open(os.path.join(HERE, "runs", "e0d_dt0.05.json"), "w", encoding="utf-8") as fh:
        json.dump(o, fh, indent=1, default=float)
    report(o)


def test_e0d_small():
    """Fast: the port rests where theirs does; their model at the zone settles; the detector sees a clean spike."""
    cell = port_cell()
    r = simulate([Stim(spiking=False)], 200.0, cell, dt=0.01)
    assert abs(r.V[0][0][0][-1] + 86.0) < 1.0
    t = np.arange(0, 50, 0.1)
    v = -60 + 50 * np.exp(-((t - 25) / 2.0) ** 2)
    ev = zone_events(t, v, 0, 50)
    assert len(ev) == 1 and abs(ev[0][1] - 50) < 1


if __name__ == "__main__":
    main(sys.argv[1:])
