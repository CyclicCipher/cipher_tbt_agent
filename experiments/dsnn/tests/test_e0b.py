"""E0b: two cell types, and the apical side rebuilt (LAB.md E0b). Run from experiments/:
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e0b                 (the run: dt 0.05; writes dsnn/runs/e0b_*.json)
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e0b --dt 0.025      (check 5, against the saved dt 0.05 run)
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e0b --fine          (check 6: A at <= 15 um, B's sections at <= 30 um)
    ../venv312/Scripts/python.exe -m pytest dsnn/tests/test_e0b.py -q    (a fast check)

**What it scores:**
- type A (human L2/3): N1, N3, N4, N6 required; N2, N5 reported;
- type B (L5-like): N2, N5 required;
- the checks on the apparatus: 1 the solver, 2 the reduction, 3 the port, 4 the dCaAP's shape (5 and 6 by flag);
- the cost.
"""
from dataclasses import replace
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from dsnn import protocols as P
from dsnn.cell import AdEx, Cell
from dsnn.cells import l23, l5
from dsnn.cells.l5 import from_v_init
from dsnn.sim import Stim, simulate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # dsnn/
REPO = os.path.dirname(os.path.dirname(HERE))


def dep(r, b, rest, comp=0):
    """Peak depolarization (mV) of a compartment in run b, from rest."""
    return float(r.vmax[b][0][comp] - rest)


def spikes_in(r, b, t0, t1):
    return sum(1 for t in r.spikes[b][0] if t0 <= t < t1)


# ---- checks on the apparatus -----------------------------------------------------------------------------------
def check1():
    """The solver: E0's cell (v0, commit a325c72) with the dCaAP off, E0's N6 protocol (E0's values: c = 9, pulse 1900 pA, Δ = 50,
    both inputs), the new Hines solve against E0's dense solve. Voltages within 1e-6 mV, identical spike times. Both
    record in float64 for this (the reference's copy has its recording's dtype changed; its solve is untouched)."""
    tmp = tempfile.mkdtemp()
    pkg = os.path.join(tmp, "e0ref")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for f in ("cell.py", "sim.py"):
        src = subprocess.run(["git", "show", "a325c72:experiments/dsnn/" + f], cwd=REPO, capture_output=True,
                             text=True, encoding="utf-8", check=True).stdout
        if f == "sim.py":
            assert src.count("np.float32") == 1
            src = src.replace("np.float32", "np.float64")
        open(os.path.join(pkg, f), "w", encoding="utf-8").write(src)
    sys.path.insert(0, tmp)
    try:
        oc, osim = importlib.import_module("e0ref.cell"), importlib.import_module("e0ref.sim")
    finally:
        sys.path.remove(tmp)
    p = oc.Params()
    old = oc.Cell(p)
    N = old.N
    gax = np.array([old.G[i, old.parent[i]] if old.parent[i] >= 0 else 0.0 for i in range(N)])
    v0 = Cell(name="v0", names=list(oc.NAMES), parent=old.parent.copy(), gax=gax, area=old.area, C=old.C, gL=old.gL,
              EL=np.full(N, p.EL), dist=np.zeros(N), length=np.zeros(N), adex=AdEx(), rest={"V": np.full(N, p.EL)})
    assert AdEx() == AdEx(p.VT, p.DT, p.a, p.tau_w, p.b, p.Vr, p.V_detect, p.V_spike, p.t_spike, p.t_ref)
    c, pulse, delta, T = 9, 1900.0, 50.0, 400.0
    d = oc.INDEX["b1.d"]
    st_old = osim.Stim().contacts(10.0, d, c).current(10.0 + delta, 12.0 + delta, oc.SOMA, pulse)
    st_new = Stim().contacts(10.0, d, c).current(10.0 + delta, 12.0 + delta, 0, pulse)
    ro = osim.simulate([st_old], T, p)
    rn = simulate([st_new], T, v0, record="all", rec_dtype=np.float64)
    err = float(np.abs(ro.V[0, 0] - rn.V[0][0]).max())
    same = ro.spikes[0][0] == rn.spikes[0][0]
    return dict(max_abs_mV=err, spikes_old=ro.spikes[0][0], spikes_new=rn.spikes[0][0], passed=err <= 1e-6 and same)


def check2(dt):
    """The reduction (type A) against the full reconstruction, passive: each number within 20%."""
    red = replace(l23(), adex=None, dcaap=None)
    full = l23(full=True)
    out = {}
    for name, cell in (("reduced", red), ("full", full)):
        Z = cell.impedance()
        z, b1 = cell.site("zone"), cell.site("b1.d")
        bap = simulate([Stim().clamp(10.0, 11.0, 30.0).clamp(11.0, 13.0, -58.0)], 40.0, cell, dt=dt)
        out[name] = dict(R_soma=Z[0, 0], R_site=Z[z, z], R_b1tip=Z[b1, b1], transfer_site_soma=Z[0, z],
                         bap_site_mV=float(bap.vmax[0][0][z] - cell.EL[z]), N=cell.N)
    ratio = {k: out["reduced"][k] / out["full"][k] for k in out["reduced"] if k != "N"}
    return dict(out, ratio=ratio, passed=all(0.8 <= x <= 1.2 for x in ratio.values()))


def check3(dt):
    """The port (type B): Bahl's example 1 (0.4 nA for 500 ms at 100 ms, v_init -65): 9 +/- 1 spikes, the first two a
    doublet (first interval < 10 ms); apical(0.5) peak +25 +/- 10 mV; tuft(0.5) peak -42 +/- 8 mV."""
    cell = l5()
    a, tu = cell.site("apical(0.5)"), cell.site("tuft(0.5)")
    r = simulate([Stim().current(100.0, 600.0, 0, 400.0)], 800.0, from_v_init(cell), dt=dt)
    s = r.spikes[0][0]
    pa, pt = float(r.vmax[0][0][a]), float(r.vmax[0][0][tu])
    doublet = len(s) >= 2 and s[1] - s[0] < 10.0
    return dict(spikes=len(s), times=s, first_interval=(s[1] - s[0]) if len(s) > 1 else None, apical_peak=pa,
                tuft_peak=pt, passed=abs(len(s) - 9) <= 1 and doublet and abs(pa - 25) <= 10 and abs(pt + 42) <= 8)


def check4(cell, cal, dt):
    """The dCaAP's shape (type A): K for zone steps of 1, 1.5 and 2 x I_th, within 10% of exp(-(f - 1)/0.3)."""
    fs = (1.0, 1.5, 2.0)
    z = cell.site("zone")
    r = simulate([Stim(spiking=False).current(P.T0, P.T0 + 100, z, f * cal["zone_rheobase"]) for f in fs], 120.0,
                 cell, dt=dt)
    K = [r.dcaap[b][0][0][1] if r.dcaap[b][0] else float("nan") for b in range(len(fs))]
    want = [math.exp(-(f - 1.0) / P.GIDON_SCALE) for f in fs]
    err = [abs(k - w) / w for k, w in zip(K, want)]
    return dict(K=K, expected=want, rel_err=err, passed=all(e <= 0.10 for e in err))


# ---- the experiments -------------------------------------------------------------------------------------------
def n1(cell, dt):
    rows = []
    rest = cell.rest["V"][0]
    for k in range(2, 22, 2):
        runs = [P.n1(cell, k, "same", 1, 0), P.n1(cell, k, "same", 0, 1), P.n1(cell, k, "other", 0, 1),
                P.n1(cell, k, "same", 1, 1), P.n1(cell, k, "other", 1, 1)]
        r = simulate(runs, 100.0, cell, dt=dt)
        a, bs, bo, ab_s, ab_o = (dep(r, i, rest) for i in range(5))
        rows.append(dict(k=k, A=a, B_same=bs, B_other=bo, same_ratio=ab_s / (a + bs), other_ratio=ab_o / (a + bo)))
    same = max(x["same_ratio"] for x in rows)
    other = [x["other_ratio"] for x in rows]
    return dict(rows=rows, max_same_ratio=same, other_range=[min(other), max(other)],
                passed=same >= 1.5 and all(0.85 <= o <= 1.15 for o in other))


def n2(cell, cal, dt):
    deltas = (-10, -5, 0, 2, 5, 10, 20)
    pulse, n = cal["n2_pulse"], cal["n2_tuft"]
    if pulse != pulse or n != n:
        return dict(passed=False, note="a strength rule found no value on its grid")
    runs = [P.n2(cell, pulse, n, 0, 1, 0), P.n2(cell, pulse, n, 0, 0, 1)] + [P.n2(cell, pulse, n, d, 1, 1)
                                                                            for d in deltas]
    tuft = cell.site("tuft(0.5)") if "tuft(0.5)" in cell.groups else cell.site("zone")
    r = simulate(runs, 120.0, cell, dt=dt, record=[0, tuft])
    alone_pulse, alone_tuft = len(r.spikes[0][0]), len(r.spikes[1][0])
    within30 = {d: spikes_in(r, 2 + i, 20.0, 50.0) for i, d in enumerate(deltas)}
    late50 = spikes_in(r, 2 + deltas.index(20), 20.0, 70.0)
    out = dict(pulse_alone=alone_pulse, tuft_alone=alone_tuft, spikes_within_30ms=within30, spikes_50ms_at_20=late50,
               dendrite_peak_mV={d: float(r.vmax[2 + i][0][tuft]) for i, d in enumerate(deltas)},
               dendrite_peak_alone=[float(r.vmax[0][0][tuft]), float(r.vmax[1][0][tuft])])
    if cell.dcaap:
        out["dcaap_K"] = {d: [round(x[1], 3) for x in r.dcaap[2 + i][0]] for i, d in enumerate(deltas)}
        out["dcaap_K_pulse_alone"] = [round(x[1], 3) for x in r.dcaap[0][0]]
        out["dcaap_K_tuft_alone"] = [round(x[1], 3) for x in r.dcaap[1][0]]
    out["passed"] = (alone_pulse == 1 and alone_tuft == 0 and any(within30[d] >= 3 for d in (0, 2, 5))
                     and late50 <= 1)
    return out


def n3(cell, cal, dt):
    n = cal["n3_n"]
    conds = (("none", 0, 0), ("A", 1, 0), ("B", 0, 1), ("A+B", 1, 1))
    r = simulate([P.n3(cell, n, a, b) for _, a, b in conds], 120.0, cell, dt=dt)
    s = {name: len(r.spikes[i][0]) for i, (name, _, _) in enumerate(conds)}
    K = {name: [round(x[1], 3) for x in r.dcaap[i][0]] for i, (name, _, _) in enumerate(conds)}
    return dict(n=n, spikes=s, dcaap_K=K, passed=s["none"] == 0 and s["A"] >= 1 and s["B"] >= 1 and s["A+B"] == 0)


def n4(cell, dt):
    rows = []
    rest = cell.rest["V"][0]
    for s in (1, 2, 3):
        for iv in (1.0, 2.0, 4.0):
            r = simulate([P.n4(cell, s, iv, "inward"), P.n4(cell, s, iv, "outward")], 100.0, cell, dt=dt)
            i, o = dep(r, 0, rest), dep(r, 1, rest)
            rows.append(dict(s=s, interval=iv, inward=i, outward=o, ratio=i / o))
    ok = [s for s in (1, 2, 3) if sum(x["ratio"] >= 1.15 for x in rows if x["s"] == s) >= 2]
    return dict(rows=rows, s_passing=ok, passed=bool(ok))


def n5(cell, dt):
    rb, ra = P.n5_rates(cell, dt, seed=0)
    out = dict(r=rb, r_prime=ra)
    if not (rb == rb and ra == ra):
        return dict(out, passed=False, note="a rate rule found no rate on its grid")
    conds = [(x, y, s) for x in (rb, 2 * rb) for y in (ra, 2 * ra) for s in (0, 1, 2)]
    T = 10000.0
    r = simulate([P.n5(cell, x, y, s, T=T) for x, y, s in conds], T, cell, dt=dt)
    ev, bf = {}, {}
    for i, c in enumerate(conds):
        n, _, f = P.events(r.spikes[i][0])
        ev[c], bf[c] = n / (T / 1000.0), f
    mean = lambda d, pairs: float(np.mean([d[b] - d[a] for a, b in pairs]))
    basal_pairs = [((rb, y, s), (2 * rb, y, s)) for y in (ra, 2 * ra) for s in (0, 1, 2)]
    apical_pairs = [((x, ra, s), (x, 2 * ra, s)) for x in (rb, 2 * rb) for s in (0, 1, 2)]
    d_ev_b, d_ev_a = mean(ev, basal_pairs), mean(ev, apical_pairs)
    d_bf_b, d_bf_a = mean(bf, basal_pairs), mean(bf, apical_pairs)
    table = [dict(r_b=x, r_a=y, seed=s, events_per_s=ev[(x, y, s)], burst_fraction=bf[(x, y, s)]) for x, y, s in conds]
    return dict(out, table=table, d_events_basal=d_ev_b, d_events_apical=d_ev_a, d_bf_basal=d_bf_b,
                d_bf_apical=d_bf_a,
                passed=(d_ev_b > 0 and d_ev_b >= 2 * abs(d_ev_a) and d_bf_a > 0 and d_bf_a >= 2 * abs(d_bf_b)))


def n6(cell, cal, dt):
    c, s = cal["n6_c"], cal["n6_pulse"]
    rows = []
    for d in (50.0, 100.0, 300.0):
        r = simulate([P.n6(cell, c, s, d, 1, 0), P.n6(cell, c, s, d, 0, 1), P.n6(cell, c, s, d, 1, 1)], 400.0, cell,
                     dt=dt)
        w0, w1 = P.T0 + d, P.T0 + d + 30.0
        rows.append(dict(delta=d, cond_alone=spikes_in(r, 0, w0, w1), test_alone=spikes_in(r, 1, w0, w1),
                         both=spikes_in(r, 2, w0, w1)))
    by = {x["delta"]: x for x in rows}
    passed = (all(x["cond_alone"] == 0 and x["test_alone"] == 0 for x in rows)
              and by[50.0]["both"] >= 1 and by[100.0]["both"] >= 1 and by[300.0]["both"] == 0)
    return dict(c=c, pulse=s, rows=rows, passed=passed)


def cost(A, B, dt):
    """Wall-clock seconds per simulated second: one cell of each type, 12 of each as a batch, a 2 x 3 network of 3
    cells of each type. Input: N5's at 4 Hz (one cell, batch); the network's fibres at 20 Hz."""
    out = {}
    for name, cell in (("A", A), ("B", B)):
        simulate([P.n5(cell, 4.0, 4.0, 0, T=100.0)], 100.0, cell, dt=dt)            # compiled already; warm caches
        r = simulate([P.n5(cell, 4.0, 4.0, 0, T=1000.0)], 1000.0, cell, dt=dt)
        out["one_" + name] = r.seconds
        r = simulate([P.n5(cell, 4.0, 4.0, s, T=1000.0) for s in range(12)], 1000.0, cell, dt=dt)
        out["batch12_" + name] = r.seconds
    cells = [A, A, A, B, B, B]
    st, wiring = P.network(cells, T=1000.0)
    r = simulate([st], 1000.0, cells, wiring=wiring, dt=dt)
    out["network_2x3"] = r.seconds
    K = r.steps
    out["compartment_updates_per_s"] = dict(one_A=K * A.N / out["one_A"], one_B=K * B.N / out["one_B"],
                                            batch12_A=K * 12 * A.N / out["batch12_A"],
                                            batch12_B=K * 12 * B.N / out["batch12_B"],
                                            network_2x3=K * sum(c.N for c in cells) / out["network_2x3"])
    return out


def run(dt, fine=False):
    t0 = time.time()
    res = dict(dt=dt, fine=fine)
    A, calA = P.calibrated("A", dt, 15.0 if fine else None)
    B, calB = P.calibrated("B", dt, 30.0 if fine else None)
    res["calibration"] = dict(A=calA, B=calB)
    res["seconds_calibration"] = round(time.time() - t0, 1)
    res["cells"] = dict(A=dict(N=A.N, R_in=A.input_resistance(), R_zone=A.input_resistance(A.site("zone"))),
                        B=dict(N=B.N, R_in=B.input_resistance(), rest_soma=float(B.rest["V"][0])))
    if not fine:
        res["check1"] = check1()
        res["check2"] = check2(dt)
    res["check3"] = check3(dt) if not fine else None
    res["check4"] = check4(A, calA, dt)
    for name, f in (("A_N1", lambda: n1(A, dt)), ("A_N2", lambda: n2(A, calA, dt)), ("A_N3", lambda: n3(A, calA, dt)),
                    ("A_N4", lambda: n4(A, dt)), ("A_N5", lambda: n5(A, dt)), ("A_N6", lambda: n6(A, calA, dt)),
                    ("B_N2", lambda: n2(B, calB, dt)), ("B_N5", lambda: n5(B, dt))):
        t = time.time()
        res[name] = f()
        res[name]["seconds"] = round(time.time() - t, 1)
    if not fine:
        res["cost"] = cost(A, B, dt)
    res["seconds"] = round(time.time() - t0, 1)
    return res


VERDICTS = ("check4", "A_N1", "A_N2", "A_N3", "A_N4", "A_N5", "A_N6", "B_N2", "B_N5")


def numbers(res):
    """The scored numbers, flattened, for checks 5 and 6."""
    out = {}
    for kind in ("A", "B"):
        out.update({"%s_%s" % (kind, k): v for k, v in res["calibration"][kind].items() if not isinstance(v, list)})
    out.update({"dcaK%d" % i: k for i, k in enumerate(res["check4"]["K"])})
    out.update({"N1_same_k%d" % x["k"]: x["same_ratio"] for x in res["A_N1"]["rows"]})
    out.update({"N1_other_k%d" % x["k"]: x["other_ratio"] for x in res["A_N1"]["rows"]})
    for kind in ("A", "B"):
        out.update({"%s_N2_dt%s" % (kind, d): v for d, v in res[kind + "_N2"].get("spikes_within_30ms", {}).items()})
        for k in ("r", "r_prime", "d_events_basal", "d_events_apical", "d_bf_basal", "d_bf_apical"):
            if k in res[kind + "_N5"]:
                out["%s_N5_%s" % (kind, k)] = res[kind + "_N5"][k]
    out.update({"N3_" + k: v for k, v in res["A_N3"]["spikes"].items()})
    out.update({"N4_s%d_i%g" % (x["s"], x["interval"]): x["ratio"] for x in res["A_N4"]["rows"]})
    out.update({"N6_%g_%s" % (x["delta"], k): x[k] for x in res["A_N6"]["rows"] for k in ("cond_alone", "test_alone",
                                                                                          "both")})
    return out


def compare(a, b, label):
    """Checks 5 and 6: verdicts identical; numbers within 5% (N5's within 15%)."""
    na, nb = numbers(a), numbers(b)
    tol = lambda k: 0.15 if "_N5_" in k else 0.05
    diffs = {k: (na[k], nb[k]) for k in na if k in nb and not (na[k] != na[k] and nb[k] != nb[k])
             and abs(na[k] - nb[k]) > tol(k) * max(abs(na[k]), 1e-9)}
    verdicts = {k: (a[k]["passed"], b[k]["passed"]) for k in VERDICTS}
    same = all(x == y for x, y in verdicts.values())
    print("%s: verdicts %s; %d of %d numbers outside tolerance: %s" % (
        label, "IDENTICAL" if same else "DIFFERENT %s" % {k: v for k, v in verdicts.items() if v[0] != v[1]},
        len(diffs), len(na), diffs))
    return dict(verdicts_identical=same, verdicts=verdicts, outside=diffs, n=len(na))


def report(res, say=print):
    pf = lambda r: "PASS" if r and r.get("passed") else "FAIL"
    say("E0b at dt = %g ms%s (%.1f s; calibration %.1f s)" % (res["dt"], ", fine cut" if res["fine"] else "",
                                                             res["seconds"], res["seconds_calibration"]))
    say("  cells: %s" % res["cells"])
    say("  calibration A: %s" % {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["calibration"]["A"].items()})
    say("  calibration B: %s" % res["calibration"]["B"])
    if "check1" in res:
        c = res["check1"]
        say("  check 1 (solver): max |dV| %.2e mV; spikes old %s new %s -> %s" % (c["max_abs_mV"], c["spikes_old"],
                                                                                c["spikes_new"], pf(c)))
        c = res["check2"]
        say("  check 2 (reduction): ratios reduced/full %s -> %s" % ({k: round(v, 3) for k, v in c["ratio"].items()},
                                                                    pf(c)))
        say("      reduced %s" % {k: round(v, 2) for k, v in c["reduced"].items()})
        say("      full    %s" % {k: round(v, 2) for k, v in c["full"].items()})
    if res.get("check3"):
        c = res["check3"]
        say("  check 3 (port): %d spikes, first interval %s ms, apical(0.5) peak %.1f, tuft(0.5) peak %.1f -> %s" % (
            c["spikes"], None if c["first_interval"] is None else round(c["first_interval"], 2), c["apical_peak"],
            c["tuft_peak"], pf(c)))
    c = res["check4"]
    say("  check 4 (dCaAP shape): K %s vs %s -> %s" % (np.round(c["K"], 3).tolist(),
                                                       np.round(c["expected"], 3).tolist(), pf(c)))
    r = res["A_N1"]
    say("  A N1: max same-branch ratio %.2f; other-branch range [%.2f, %.2f] -> %s" % (
        r["max_same_ratio"], r["other_range"][0], r["other_range"][1], pf(r)))
    for kind in ("A", "B"):
        r = res[kind + "_N2"]
        if "note" in r:
            say("  %s N2: %s -> FAIL" % (kind, r["note"]))
            continue
        say("  %s N2%s: pulse alone %d, tuft alone %d; spikes in 30 ms by dt %s; at dt = 20, %d in 50 ms -> %s" % (
            kind, " (reported)" if kind == "A" else "", r["pulse_alone"], r["tuft_alone"], r["spikes_within_30ms"],
            r["spikes_50ms_at_20"], pf(r)))
        say("      dendrite peak by dt %s; alone (pulse, tuft) %s" % (
            {d: round(v, 1) for d, v in r["dendrite_peak_mV"].items()}, np.round(r["dendrite_peak_alone"], 1).tolist()))
        if "dcaap_K" in r:
            say("      dCaAP K by dt %s; pulse alone %s; tuft alone %s" % (r["dcaap_K"], r["dcaap_K_pulse_alone"],
                                                                         r["dcaap_K_tuft_alone"]))
    r = res["A_N3"]
    say("  A N3 (n = %s): spikes %s; dCaAP K %s -> %s" % (r["n"], r["spikes"], r["dcaap_K"], pf(r)))
    r = res["A_N4"]
    say("  A N4: inward/outward %s -> %s" % (
        " ".join("s%d/%gms %.2f" % (x["s"], x["interval"], x["ratio"]) for x in r["rows"]), pf(r)))
    for kind in ("A", "B"):
        r = res[kind + "_N5"]
        tag = " (reported)" if kind == "A" else ""
        if "table" in r:
            say("  %s N5%s: r %.1f Hz, r' %.1f Hz; events/s change: basal %+.2f, apical %+.2f; burst fraction change: "
                "basal %+.3f, apical %+.3f -> %s" % (kind, tag, r["r"], r["r_prime"], r["d_events_basal"],
                                                     r["d_events_apical"], r["d_bf_basal"], r["d_bf_apical"], pf(r)))
        else:
            say("  %s N5%s: r %s, r' %s: %s -> FAIL" % (kind, tag, r["r"], r["r_prime"], r["note"]))
    r = res["A_N6"]
    say("  A N6 (c %s, pulse %s): %s -> %s" % (r["c"], r["pulse"], "; ".join(
        "D %g: cond %d, test %d, both %d" % (x["delta"], x["cond_alone"], x["test_alone"], x["both"])
        for x in r["rows"]), pf(r)))
    if "cost" in res:
        c = res["cost"]
        say("  cost (s per simulated s): %s; compartment updates/s %s" % (
            {k: round(v, 3) for k, v in c.items() if k != "compartment_updates_per_s"},
            {k: "%.2g" % v for k, v in c["compartment_updates_per_s"].items()}))


def main(argv):
    dt = float(argv[argv.index("--dt") + 1]) if "--dt" in argv else 0.05
    fine = "--fine" in argv
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    res = run(dt, fine)
    name = "e0b_dt%g%s.json" % (dt, "_fine" if fine else "")
    report(res)
    if dt != 0.05 or fine:
        base = json.load(open(os.path.join(HERE, "runs", "e0b_dt0.05.json")))
        res["compare"] = compare(base, json.loads(json.dumps(res, default=float)),
                                 "check 6 (compartment size)" if fine else "check 5 (step size)")
    json.dump(res, open(os.path.join(HERE, "runs", name), "w"), indent=1, default=float)


def test_e0b_small():
    """A fast check: the type A cell's passive structure, the Hines solve against a dense solve, one NMDA plateau."""
    A = l23()
    assert 40 < A.input_resistance() < 80
    q = replace(A, adex=None, dcaap=None)
    r = simulate([Stim().current(5.0, 400.0, A.site("zone"), 100.0)], 400.0, q, record="all")
    v_ss = A.EL[0] + A.impedance()[:, A.site("zone")] * 100.0 / 1000.0
    assert np.abs(r.V[0][0][:, -1] - v_ss).max() < 0.05
    r = simulate([Stim(spiking=False).contacts(10, A.site("b1.d"), 20)], 150.0, A, record=[A.site("b1.d")])
    assert P.longest_above(r.v(0, A.site("b1.d")), -40.0, r.t) > 30


if __name__ == "__main__":
    main(sys.argv[1:])
