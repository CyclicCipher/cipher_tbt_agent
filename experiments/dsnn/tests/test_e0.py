"""E0: the neuron (LAB.md E0). Run from experiments/:
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e0 [--dt 0.05] [--check-dt]   (the run; writes dsnn/runs/e0*.json)
    ../venv312/Scripts/python.exe -m pytest dsnn/tests/test_e0.py -q                 (a fast check)

**What it scores:** N1–N6, each against its pre-registered criterion, with every input strength set by its rule
(`protocols.calibrated`). Also the two checks on the apparatus (the dCaAP's shape; the step size) and the cost.
"""
import json
import math
import os
import sys
import time

import numpy as np

from dsnn import protocols as P
from dsnn.cell import Cell, INDEX, Params, SOMA, ZONE
from dsnn.sim import Stim, simulate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # dsnn/


def dep(r, b, comp=SOMA, EL=-70.0):
    """Peak depolarization (mV) of a compartment in run b."""
    return float(r.V[b, 0, comp].max() - EL)


def spikes_in(r, b, t0, t1):
    return sum(1 for t in r.spikes[b][0] if t0 <= t < t1)


def dca_shape(p, cal, dt):
    """Apparatus: K for zone steps at 1, 1.5 and 2 x the threshold current, against exp(-(f - 1) / 0.3)."""
    fs = (1.0, 1.5, 2.0)
    r = simulate([Stim(spiking=False).current(P.T0, P.T0 + 100, ZONE, f * cal["zone_rheobase"]) for f in fs],
                 120.0, p, dt=dt)
    K = [r.dcaap[b][0][0][1] if r.dcaap[b][0] else float("nan") for b in range(len(fs))]
    want = [math.exp(-(f - 1.0) / p.dca_tauK) for f in fs]
    err = [abs(k - w_) / w_ for k, w_ in zip(K, want)]
    return dict(K=K, expected=want, rel_err=err, passed=all(e <= 0.10 for e in err))


def n1(p, dt):
    ks = list(range(2, 22, 2))
    rows = []
    for k in ks:
        runs = [P.n1(k, "same", 1, 0), P.n1(k, "same", 0, 1), P.n1(k, "other", 0, 1),
                P.n1(k, "same", 1, 1), P.n1(k, "other", 1, 1)]
        r = simulate(runs, 100.0, p, dt=dt)
        a, bs, bo, ab_s, ab_o = (dep(r, i) for i in range(5))
        rows.append(dict(k=k, A=a, B_same=bs, B_other=bo, same_ratio=ab_s / (a + bs), other_ratio=ab_o / (a + bo)))
    same = max(x["same_ratio"] for x in rows)
    other = [x["other_ratio"] for x in rows]
    return dict(rows=rows, max_same_ratio=same, other_range=[min(other), max(other)],
                passed=same >= 1.5 and all(0.85 <= o <= 1.15 for o in other))


def n2(p, cal, dt):
    deltas = (-10, -5, 0, 2, 5, 10, 20)
    pulse, n = cal["n2_pulse"], cal["n2_tuft"]
    runs = [P.n2(pulse, n, 0, 1, 0), P.n2(pulse, n, 0, 0, 1)] + [P.n2(pulse, n, d, 1, 1) for d in deltas]
    r = simulate(runs, 120.0, p, dt=dt)
    alone_pulse, alone_tuft = len(r.spikes[0][0]), len(r.spikes[1][0])
    within30 = {d: spikes_in(r, 2 + i, 20.0, 50.0) for i, d in enumerate(deltas)}
    late50 = spikes_in(r, 2 + deltas.index(20), 20.0, 70.0)
    dca = {d: [round(K, 3) for _, K in r.dcaap[2 + i][0]] for i, d in enumerate(deltas)}
    passed = (alone_pulse == 1 and alone_tuft == 0 and any(within30[d] >= 3 for d in (0, 2, 5)) and late50 <= 1)
    return dict(pulse_alone=alone_pulse, tuft_alone=alone_tuft, spikes_within_30ms=within30, spikes_50ms_at_20=late50,
                dcaap_K=dca, dcaap_K_pulse_alone=[round(K, 3) for _, K in r.dcaap[0][0]], passed=passed)


def n3(p, cal, dt):
    n = cal["n3_n"]
    conds = (("none", 0, 0), ("A", 1, 0), ("B", 0, 1), ("A+B", 1, 1))
    r = simulate([P.n3(n, a, b) for _, a, b in conds], 120.0, p, dt=dt)
    s = {name: len(r.spikes[i][0]) for i, (name, _, _) in enumerate(conds)}
    K = {name: [round(k, 3) for _, k in r.dcaap[i][0]] for i, (name, _, _) in enumerate(conds)}
    return dict(spikes=s, dcaap_K=K, passed=s["none"] == 0 and s["A"] >= 1 and s["B"] >= 1 and s["A+B"] == 0)


def n4(p, dt):
    rows = []
    for s in (2, 4, 6, 8):
        for iv in (1.0, 2.0, 4.0):
            r = simulate([P.n4(s, iv, "inward"), P.n4(s, iv, "outward")], 100.0, p, dt=dt)
            rows.append(dict(s=s, interval=iv, inward=dep(r, 0), outward=dep(r, 1), ratio=dep(r, 0) / dep(r, 1)))
    ok = [s for s in (2, 4, 6, 8) if sum(x["ratio"] >= 1.15 for x in rows if x["s"] == s) >= 2]
    return dict(rows=rows, s_passing=ok, passed=bool(ok))


def n5(p, dt):
    rb, ra = P.n5_rates(p, dt, seed=0)
    out = dict(r=rb, r_prime=ra)
    if not (rb == rb and ra == ra):                        # a rule found no rate (nan)
        return dict(out, passed=False, note="a rate rule found no rate on its grid")
    conds = [(x, y, s) for x in (rb, 2 * rb) for y in (ra, 2 * ra) for s in (0, 1, 2)]
    T = 10000.0                                            # amendment 4: 10 s per scored condition
    r = simulate([P.n5(x, y, s, T=T) for x, y, s in conds], T, p, dt=dt)
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
                passed=(d_ev_b > 0 and d_ev_b >= 2 * abs(d_ev_a)            # amendment 4: signed
                        and d_bf_a > 0 and d_bf_a >= 2 * abs(d_bf_b)))


def n6(p, cal, dt):
    c, s = cal["n6_c"], cal["n6_pulse"]
    rows = []
    for d in (50.0, 100.0, 300.0):
        r = simulate([P.n6(c, s, d, 1, 0), P.n6(c, s, d, 0, 1), P.n6(c, s, d, 1, 1)], 400.0, p, dt=dt)
        w0, w1 = P.T0 + d, P.T0 + d + 30.0
        rows.append(dict(delta=d, cond_alone=spikes_in(r, 0, w0, w1), test_alone=spikes_in(r, 1, w0, w1),
                         both=spikes_in(r, 2, w0, w1)))
    by = {x["delta"]: x for x in rows}
    passed = (all(x["cond_alone"] == 0 and x["test_alone"] == 0 for x in rows)
              and by[50.0]["both"] >= 1 and by[100.0]["both"] >= 1 and by[300.0]["both"] == 0)
    return dict(rows=rows, passed=passed)


def cost(p, dt):
    """Wall-clock seconds per simulated second: one cell, 12 cells as a batch, the 2 x 3 network."""
    out = {}
    one = P.n5(4.0, 4.0, 0, T=1000.0)
    r = simulate([one], 1000.0, p, dt=dt)
    out["one_cell"] = r.seconds
    r = simulate([P.n5(4.0, 4.0, s, T=1000.0) for s in range(12)], 1000.0, p, dt=dt)
    out["batch_12"] = r.seconds
    st, wiring = P.network(depth=3, T=1000.0)
    r = simulate([st], 1000.0, p, wiring=wiring, dt=dt)
    out["network_2x3"] = r.seconds
    N = Cell(p).N
    out["compartment_updates_per_s"] = dict(one_cell=r.steps * N / out["one_cell"],
                                            batch_12=r.steps * 12 * N / out["batch_12"],
                                            network_2x3=r.steps * 6 * N / out["network_2x3"])
    return out


def run(dt):
    t0 = time.time()
    p, cal = P.calibrated(dt)
    res = dict(dt=dt, calibration=cal, seconds_calibration=round(time.time() - t0, 1))
    res["check_dcaap_shape"] = dca_shape(p, cal, dt)
    for name, f in (("N1", lambda: n1(p, dt)), ("N2", lambda: n2(p, cal, dt)), ("N3", lambda: n3(p, cal, dt)),
                    ("N4", lambda: n4(p, dt)), ("N5", lambda: n5(p, dt)), ("N6", lambda: n6(p, cal, dt))):
        t = time.time()
        res[name] = f()
        res[name]["seconds"] = round(time.time() - t, 1)
    res["cost"] = cost(p, dt)
    res["input_resistance_MOhm"] = Cell(p).input_resistance()
    res["seconds"] = round(time.time() - t0, 1)
    return res


def numbers(res):
    """The scored numbers, flattened, for the step-size check."""
    out = dict(res["calibration"])
    out.update({"dcaK%d" % i: k for i, k in enumerate(res["check_dcaap_shape"]["K"])})
    out.update({"N1_same_k%d" % x["k"]: x["same_ratio"] for x in res["N1"]["rows"]})
    out.update({"N1_other_k%d" % x["k"]: x["other_ratio"] for x in res["N1"]["rows"]})
    out.update({"N2_dt%d" % d: v for d, v in res["N2"]["spikes_within_30ms"].items()})
    out.update({"N3_" + k: v for k, v in res["N3"]["spikes"].items()})
    out.update({"N4_s%d_i%g" % (x["s"], x["interval"]): x["ratio"] for x in res["N4"]["rows"]})
    for k in ("r", "r_prime", "d_events_basal", "d_events_apical", "d_bf_basal", "d_bf_apical"):
        if k in res["N5"]:
            out["N5_" + k] = res["N5"][k]
    out.update({"N6_%g_%s" % (x["delta"], k): x[k] for x in res["N6"]["rows"] for k in ("cond_alone", "test_alone", "both")})
    return out


def report(res, say=print):
    say("E0 at dt = %g ms (%.1f s; calibration %.1f s)" % (res["dt"], res["seconds"], res["seconds_calibration"]))
    say("  calibration: %s" % res["calibration"])
    say("  input resistance %.1f MOhm" % res["input_resistance_MOhm"])
    c = res["check_dcaap_shape"]
    say("  check, dCaAP shape: K %s vs %s -> %s" % (np.round(c["K"], 3).tolist(), np.round(c["expected"], 3).tolist(),
                                                     "PASS" if c["passed"] else "FAIL"))
    r = res["N1"]
    say("  N1: max same-branch ratio %.2f; other-branch range [%.2f, %.2f] -> %s" % (
        r["max_same_ratio"], r["other_range"][0], r["other_range"][1], "PASS" if r["passed"] else "FAIL"))
    r = res["N2"]
    say("  N2: pulse alone %d, tuft alone %d; spikes in 30 ms by dt %s; at dt = 20, %d in 50 ms -> %s" % (
        r["pulse_alone"], r["tuft_alone"], r["spikes_within_30ms"], r["spikes_50ms_at_20"],
        "PASS" if r["passed"] else "FAIL"))
    say("      dCaAP K by dt %s; pulse alone %s" % (r["dcaap_K"], r["dcaap_K_pulse_alone"]))
    r = res["N3"]
    say("  N3: spikes %s; dCaAP K %s -> %s" % (r["spikes"], r["dcaap_K"], "PASS" if r["passed"] else "FAIL"))
    r = res["N4"]
    say("  N4: ratios inward/outward %s -> %s" % (
        " ".join("s%d/%gms %.2f" % (x["s"], x["interval"], x["ratio"]) for x in r["rows"]),
        "PASS" if r["passed"] else "FAIL"))
    r = res["N5"]
    if "table" in r:
        say("  N5: r %.1f Hz, r' %.1f Hz; events/s change: basal %+.2f, apical %+.2f; burst fraction change: basal "
            "%+.3f, apical %+.3f -> %s" % (r["r"], r["r_prime"], r["d_events_basal"], r["d_events_apical"],
                                           r["d_bf_basal"], r["d_bf_apical"], "PASS" if r["passed"] else "FAIL"))
    else:
        say("  N5: r %s, r' %s: %s -> FAIL" % (r["r"], r["r_prime"], r["note"]))
    r = res["N6"]
    say("  N6: %s -> %s" % ("; ".join("D %g: cond %d, test %d, both %d" % (x["delta"], x["cond_alone"], x["test_alone"],
                                                                          x["both"]) for x in r["rows"]),
                            "PASS" if r["passed"] else "FAIL"))
    c = res["cost"]
    say("  cost (s per simulated s): one cell %.2f, 12 as a batch %.2f, 2x3 network %.2f; compartment updates/s %s" % (
        c["one_cell"], c["batch_12"], c["network_2x3"],
        {k: "%.2g" % v for k, v in c["compartment_updates_per_s"].items()}))


def main(argv):
    dt = float(argv[argv.index("--dt") + 1]) if "--dt" in argv else 0.05
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    res = run(dt)
    json.dump(res, open(os.path.join(HERE, "runs", "e0_dt%g.json" % dt), "w"), indent=1, default=float)
    report(res)
    if "--check-dt" in argv:                                 # the step-size check, against the saved coarse run
        coarse = json.load(open(os.path.join(HERE, "runs", "e0_dt0.05.json")))
        a, b = numbers(coarse), numbers(res)
        diffs = {k: (a[k], b[k]) for k in a if k in b and abs(a[k] - b[k]) > 0.05 * max(abs(a[k]), 1e-9)}
        verdicts = {k: (coarse[k]["passed"], res[k]["passed"]) for k in ("check_dcaap_shape", "N1", "N2", "N3", "N4",
                                                                          "N5", "N6")}
        same = all(x == y for x, y in verdicts.values())
        print("check, step size: verdicts %s (%s); %d of %d numbers differ by > 5%%: %s" % (
            "identical" if same else "DIFFERENT", verdicts, len(diffs), len(a), diffs))


def test_e0_small():
    """A fast check: the passive cell, one NMDA plateau, and the dCaAP's rule-set values."""
    p = Params()
    assert 60 < Cell(p).input_resistance() < 80
    r = simulate([Stim(spiking=False).contacts(10, INDEX["b1.d"], 20)], 150.0, p)
    assert P._longest_above(r.V[0, 0, INDEX["b1.d"]], -40.0, r.t) > 50
    I = P.zone_rheobase(p, 0.05)
    assert 100 < I < 1000


if __name__ == "__main__":
    main(sys.argv[1:])
