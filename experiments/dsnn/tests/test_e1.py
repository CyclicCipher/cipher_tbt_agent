"""E1: one calcium rule, against slice experiments (LAB.md E1). Run from experiments/:
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e1                  (the run, dt 0.05; writes dsnn/runs/e1_dt0.05.json)
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e1 --trains         (run 2, amendment 11: the train protocols)
    ../venv312/Scripts/python.exe -m pytest dsnn/tests/test_e1.py -q     (a fast check)

**The order:** the rule-set settings (EPSP sizes, the DC step, L4's current); every protocol simulated and each
synapse's calcium computed; the rule fitted on F1–F3 alone; then the rule applied to every protocol.

**The cells:** E0b's, with E0b's saved rule-set values (`runs/e0b_dt0.05.json`): type A's τ_K, pulses, n3_n and
n6_c; type B's pulse and n2_tuft. E0c's type A isn't run: its fit failed, so the comparison rule keeps E0b's.
"""
from functools import lru_cache
import json
from multiprocessing import Pool
import os
import sys
import time

import numpy as np

from dsnn import plasticity as R
from dsnn import protocols as P
from dsnn.cells import l23, l5
from dsnn.sim import Stim, simulate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # dsnn/
DT, REC = 0.05, 2                     # ms; record every 0.1 ms
T_START = 500.0                       # the first pairing; a DC step has settled by then
LTP, LTD = 1.1, 0.9
FIT = (("F1", 1.0, 0.1), ("F2", 1.36, 0.095), ("F3", 0.645, 0.065))


@lru_cache(maxsize=1)
def e0b_calibration():
    with open(os.path.join(HERE, "runs", "e0b_dt0.05.json"), encoding="utf-8") as fh:
        return json.load(fh)["calibration"]


@lru_cache(maxsize=2)
def cell_of(kind):
    return l23().with_dcaap(tauK=e0b_calibration()["A"]["tauK"]) if kind == "A" else l5()


# ---- the rule-set settings ---------------------------------------------------------------------------------------
def epsp_n(kind, comp, mV, top=400):
    """The smallest n (1..top) whose single activation at comp raises the soma by >= mV."""
    cell = cell_of(kind)
    ns = np.arange(1, top + 1)
    r = simulate([Stim().contacts(20.0, comp, int(n)) for n in ns], 150.0, cell, dt=DT)
    rest = float(cell.rest["V"][0])
    return P.first(ns, [r.vmax[b][0][0] - rest >= mV for b in range(len(ns))])


def dc_below(kind):
    """The largest DC into the soma, on a 10 pA grid, that gives no spike in 1 s."""
    amps = np.arange(0.0, 2000.0 + 1e-9, 10.0)
    r = simulate([Stim().current(0.0, 1000.0, 0, a) for a in amps], 1000.0, cell_of(kind), dt=DT)
    fire = P.first(amps, [bool(r.spikes[b][0]) for b in range(len(amps))])
    return float(fire - 10.0)


def l4_amp(kind, comp, pulse):
    """The smallest 100 ms step into comp (50 pA grid), from 50 ms before a single post pulse to 50 ms after, at
    which step and pulse together give >= 2 spikes; and the step alone's spikes at that amplitude."""
    cell = cell_of(kind)
    amps = np.arange(50.0, 5000.0 + 1e-9, 50.0)
    both = [Stim().current(100.0, 200.0, comp, a).current(150.0, 152.0, 0, pulse) for a in amps]
    r = simulate(both, 300.0, cell, dt=DT)
    a0 = P.first(amps, [len(r.spikes[b][0]) >= 2 for b in range(len(amps))])
    if a0 != a0:
        return dict(amp=a0, step_alone_spikes=None)
    r1 = simulate([Stim().current(100.0, 200.0, comp, a0)], 300.0, cell, dt=DT)
    return dict(amp=a0, step_alone_spikes=len(r1.spikes[0][0]))


def _setting(job):
    name, f, args = job
    return name, globals()[f](*args)


# ---- the protocols ----------------------------------------------------------------------------------------------
def sjostrom(f):
    """Post-pulse onsets: 15 bursts of 5 pairings at f, each burst followed by 1 s; at 0.1 Hz, 50 pairings 1 s
    apart."""
    if f == 0.1:
        return [T_START + 1000.0 * k for k in range(50)]
    return [T_START + b * (5000.0 / f + 1000.0) + 1000.0 * k / f for b in range(15) for k in range(5)]


def letzkus():
    return [T_START + 1000.0 * k for k in range(100)]


def spec(name, kind, posts, timing, paths, burst=1, dc=0.0, extra=(), pre=None):
    """One protocol. paths: [(label, [(comp, n)])], each contact activated at post − timing (or at `pre`'s times).
    extra: ("tuft", n, offset) = n contacts spread over the tuft at post + offset; ("step", comp, t0, t1, pA)."""
    return dict(name=name, kind=kind, posts=list(posts), timing=timing, paths=paths, burst=burst, dc=dc,
                extra=list(extra), pre=pre)


def build(sp, pulse):
    cell = cell_of(sp["kind"])
    posts = sp["posts"]
    pre = sp["pre"] if sp["pre"] is not None else [t - sp["timing"] for t in posts]
    T = max(posts + pre) + 1000.0
    st = Stim()
    if sp["dc"]:
        st.current(0.0, T, 0, sp["dc"])
    for t in posts:
        for k in range(sp["burst"]):
            st.current(t + 5.0 * k, t + 5.0 * k + 2.0, 0, pulse)
        for ex in sp["extra"]:
            if ex[0] == "tuft":
                for c, m in P.spread(cell, "tuft", ex[1]):
                    st.contacts(t + ex[2], c, m)
            elif ex[0] == "step":
                st.current(t + ex[2], t + ex[3], ex[1], ex[4])
    for _, contacts in sp["paths"]:
        for c, n in contacts:
            for tp in pre:
                st.contacts(tp, c, n)
    return st, pre, T


def per_pairing(spikes, posts):
    """Spikes from each post pulse's onset to the next pairing's, at most 30 ms (amendment 11)."""
    out = []
    for k, t in enumerate(posts):
        end = min(t + 30.0, posts[k + 1]) if k + 1 < len(posts) else t + 30.0
        out.append(sum(1 for x in spikes if t <= x < end))
    return out


def train_pulse(job):
    """Amendment 11: the smallest post pulse (50 pA grid, from the rule-set amplitude to 8000 pA) at which every
    pairing of the protocol's first burst (5 pairings, from rest, with the pathway) fires."""
    sp, base = job
    one = dict(sp, posts=sp["posts"][:5], pre=None)
    amps = np.arange(base, 8000.0 + 1e-9, 50.0)
    built = [build(one, a) for a in amps]
    r = simulate([b[0] for b in built], built[0][2], cell_of(sp["kind"]), dt=DT)
    ok = [min(per_pairing(r.spikes[b][0], one["posts"])) >= 1 for b in range(len(amps))]
    return sp["name"], P.first(amps, ok)


def run_protocol(job):
    """Simulate one protocol; each pathway's calcium per compartment, for one contact there."""
    sp, pulse = job
    cell = cell_of(sp["kind"])
    st, pre, T = build(sp, pulse)
    comps = sorted({c for _, contacts in sp["paths"] for c, _ in contacts})
    r = simulate([st], T, cell, dt=DT, rec_every=REC, record=[0] + comps, record_ca=True)
    s = r.spikes[0][0]
    per = per_pairing(s, sp["posts"])
    out = dict(name=sp["name"], spikes=len(s), T=T, h=float(r.t[1] - r.t[0]),
               post_spikes_per_pairing=dict(mean=float(np.mean(per)) if per else None,
                                            min=int(min(per)) if per else None, max=int(max(per)) if per else None),
               paths={})
    for label, contacts in sp["paths"]:
        out["paths"][label] = [dict(comp=c, n=float(n), c=R.calcium(r.t, r.v(0, c), r.i_ca(0, c), pre, cell.area[c],
                                                                      cell.syn).astype(np.float32),
                                    v_peak=float(np.max(r.v(0, c))))
                               for c, n in contacts]
    return out


def protocols(s, calA, calB):
    """Every E1 protocol, with the settings s (LAB.md E1's tables)."""
    B, A = cell_of("B"), cell_of("A")
    basal, dist, tuft = B.index("basal.0"), B.index("apical.4"), B.site("tuft(0.5)")
    out = []
    small = [("basal", [(basal, s["B_basal_1mV"])])]
    out += [spec("F1", "B", sjostrom(0.1), 10.0, small),
            spec("F2", "B", sjostrom(0.1), 10.0, small, dc=s["B_dc"]),
            spec("F3", "B", sjostrom(0.1), -10.0, small, dc=s["B_dc"])]
    out += [spec("S1_%g" % f, "B", sjostrom(f), 10.0, small) for f in (10, 20, 40, 50)]
    out += [spec("S2_%g" % f, "B", sjostrom(f), -10.0, small) for f in (0.1, 10, 20)]
    out += [spec("S3_50", "B", sjostrom(50), -10.0, small),
            spec("S4", "B", sjostrom(0.1), 10.0, [("basal", [(basal, s["B_basal_3mV"])])])]
    sites = (("proximal", basal, s["B_basal_1mV"]), ("distal", dist, s["B_apical4_1mV"]))
    for lab, c, n in sites:
        out += [spec("L1_" + lab, "B", letzkus(), 10.0, [(lab, [(c, n)])]),
                spec("L2_" + lab, "B", letzkus(), 10.0, [(lab, [(c, n)])], burst=3),
                spec("L3_" + lab, "B", letzkus(), -10.0, [(lab, [(c, n)])], burst=3),
                spec("L4_" + lab, "B", letzkus(), 10.0, [(lab, [(c, n)])],
                     extra=[("step", dist, -50.0, 50.0, s["B_L4"]["amp"])])]
    t5 = [("tuft", [(tuft, s["B_tuft_1mV"])]), ("basal", [(basal, s["B_basal_1mV"])])]
    out += [spec("T5_without", "B", letzkus(), 10.0, t5),
            spec("T5_with", "B", letzkus(), 10.0, t5, extra=[("tuft", calB["n2_tuft"], 2.0)])]
    # type A
    d1, b1m, zone = A.site("b1.d"), A.site("b1.m"), A.site("zone")
    rws = [T_START + 125.0 * k for k in range(480)]                       # 8 Hz for 60 s
    c = calA["n6_c"]
    out += [spec("A1_clustered", "A", [], 0.0, [("cluster", [(d1, c)])], pre=rws),
            spec("A1_dispersed", "A", [], 0.0, [("dispersed", [(x, m) for x, m in P.spread(A, "basal", c)])],
                 pre=rws)]
    smallA = [("b1.m", [(b1m, s["A_b1m_1mV"])])]
    out += [spec("A2_%+d_%g" % (tm, f), "A", sjostrom(f), float(tm), smallA) for tm in (10, -10) for f in (0.1, 20)]
    zp = [("zone_path", [(zone, s["A_zone_1mV"])])]
    n3 = calA["n3_n"]
    out += [spec("A3_event", "A", [], 0.0, zp + [("zone_drive", [(zone, n3)])], pre=letzkus()),
            spec("A3_no_event", "A", [], 0.0, zp + [("zone_drive", [(zone, n3 - 4)])], pre=letzkus())]
    return out


# ---- the run ----------------------------------------------------------------------------------------------------
def changes(rule, res):
    """w_after/w_before per protocol and pathway: one contact per compartment, averaged by contact count."""
    out = {}
    for name, r in res.items():
        out[name] = {}
        for label, rows in r["paths"].items():
            w = np.array([x["n"] for x in rows])
            ch = np.array([rule.apply(x["c"], r["h"]) for x in rows])
            out[name][label] = float((w * ch).sum() / w.sum())
    return out


def verdicts(ch):
    g = lambda n, p="basal": ch[n][p]
    s1 = [g("S1_%g" % f) for f in (10, 20, 40, 50)]
    v = dict(
        S1=all(x > LTP for x in s1) and all(b >= a - 1e-9 for a, b in zip(s1, s1[1:])),
        S2=all(g("S2_%g" % f) < LTD for f in (0.1, 10, 20)),
        S3=g("S3_50") > LTP,
        S4=g("S4") > LTP,
        L1=g("L1_distal", "distal") < LTD and g("L1_distal", "distal") < g("L1_proximal", "proximal"),
        L2=g("L2_proximal", "proximal") > LTP and g("L2_distal", "distal") < LTD,
        L3=g("L3_proximal", "proximal") < LTD and g("L3_distal", "distal") > LTP,
        L4=g("L4_proximal", "proximal") > LTP and g("L4_distal", "distal") < LTD,
        A1=ch["A1_clustered"]["cluster"] > LTP and ch["A1_dispersed"]["dispersed"] <= LTP)
    return v


def run(workers=7):
    t0 = time.time()
    cal = e0b_calibration()
    calA, calB = cal["A"], cal["B"]
    B, A = cell_of("B"), cell_of("A")
    jobs = [("B_basal_1mV", "epsp_n", ("B", B.index("basal.0"), 1.0)),
            ("B_basal_3mV", "epsp_n", ("B", B.index("basal.0"), 3.0)),
            ("B_apical4_1mV", "epsp_n", ("B", B.index("apical.4"), 1.0)),
            ("B_tuft_1mV", "epsp_n", ("B", B.site("tuft(0.5)"), 1.0)),
            ("A_b1m_1mV", "epsp_n", ("A", A.site("b1.m"), 1.0)),
            ("A_zone_1mV", "epsp_n", ("A", A.site("zone"), 1.0)),
            ("B_dc", "dc_below", ("B",)),
            ("B_L4", "l4_amp", ("B", B.index("apical.4"), calB["n2_pulse"]))]
    with Pool(workers) as pool:
        s = dict(pool.map(_setting, jobs))
        t1 = time.time()
        specs = protocols(s, calA, calB)
        pulses = {"A": calA["n2_pulse"], "B": calB["n2_pulse"]}
        sims = pool.map(run_protocol, [(sp, pulses[sp["kind"]]) for sp in specs], chunksize=1)
    t2 = time.time()
    res = {r["name"]: r for r in sims}
    rule, err, fit_ch = R.fit([(res[n]["paths"]["basal"][0]["c"], res[n]["h"]) for n, _, _ in FIT],
                              [x for _, x, _ in FIT], [x for _, _, x in FIT])
    t3 = time.time()
    fit_ok = all(abs(c - x) <= tol for c, (_, x, tol) in zip(fit_ch, FIT))
    out = dict(dt=DT, settings=s, rule=dict(th_d=rule.th_d, th_p=rule.th_p, ratio=rule.ratio, kd=rule.kd),
               fit=dict(error=err, changes=dict(zip([n for n, _, _ in FIT], map(float, fit_ch))), passed=fit_ok),
               seconds=dict(settings=round(t1 - t0, 1), simulations=round(t2 - t1, 1), fit=round(t3 - t2, 1)))
    out["protocols"] = {n: dict(spikes=r["spikes"], T_ms=r["T"], post_spikes_per_pairing=r["post_spikes_per_pairing"],
                                c_peak={lab: [round(float(np.max(x["c"])), 2) for x in rows]
                                        for lab, rows in r["paths"].items()},
                                v_peak={lab: [round(x["v_peak"], 1) for x in rows] for lab, rows in r["paths"].items()},
                                above={lab: [dict(d_ms=round(float((x["c"] > rule.th_d).sum()) * r["h"], 1),
                                                  p_ms=round(float((x["c"] > rule.th_p).sum()) * r["h"], 1))
                                             for x in rows] for lab, rows in r["paths"].items()})
                        for n, r in res.items()}
    if fit_ok:
        ch = changes(rule, res)
        out["changes"] = ch
        out["verdicts"] = verdicts(ch)
        out["T5"] = {p: ch["T5_with"][p] - ch["T5_without"][p] for p in ("tuft", "basal")}
        out["A3"] = ch["A3_event"]["zone_path"] - ch["A3_no_event"]["zone_path"]
    out["seconds"]["total"] = round(time.time() - t0, 1)
    out["simulated_s"] = round(sum(r["T"] for r in res.values()) / 1000.0, 1)
    return out


TRAINS = ("S1_10", "S1_20", "S1_40", "S1_50", "S2_10", "S2_20", "S3_50", "A2_+10_20", "A2_-10_20")


def rerun_trains(workers=7):
    """Run 2 (amendment 11): the train protocols with a post spike at every pairing; run 1's rule and every other
    protocol's change as they were."""
    t0 = time.time()
    with open(os.path.join(HERE, "runs", "e1_run1_dt0.05.json"), encoding="utf-8") as fh:
        run1 = json.load(fh)
    rule = R.Rule(**run1["rule"])
    cal = e0b_calibration()
    calA, calB = cal["A"], cal["B"]
    specs = {sp["name"]: sp for sp in protocols(run1["settings"], calA, calB) if sp["name"] in TRAINS}
    base = {"A": calA["n2_pulse"], "B": calB["n2_pulse"]}
    with Pool(workers) as pool:
        pulses = dict(pool.map(train_pulse, [(sp, base[sp["kind"]]) for sp in specs.values()], chunksize=1))
        t1 = time.time()
        sims = pool.map(run_protocol, [(sp, pulses[n]) for n, sp in specs.items()], chunksize=1)
    res = {r["name"]: r for r in sims}
    ch = dict(run1["changes"])
    ch.update(changes(rule, res))
    out = dict(dt=DT, amendment=11, rule=run1["rule"], train_pulses=pulses,
               protocols={n: dict(spikes=r["spikes"], post_spikes_per_pairing=r["post_spikes_per_pairing"],
                                  c_peak={lab: [round(float(np.max(x["c"])), 2) for x in rows]
                                          for lab, rows in r["paths"].items()},
                                  above={lab: [dict(d_ms=round(float((x["c"] > rule.th_d).sum()) * r["h"], 1),
                                                    p_ms=round(float((x["c"] > rule.th_p).sum()) * r["h"], 1))
                                               for x in rows] for lab, rows in r["paths"].items()})
                          for n, r in res.items()},
               changes=ch, verdicts=verdicts(ch), T5=run1["T5"], A3=run1["A3"],
               seconds=dict(pulses=round(t1 - t0, 1), total=round(time.time() - t0, 1)),
               simulated_s=round(sum(r["T"] for r in res.values()) / 1000.0, 1))
    return out


def report(o, say=print):
    if o.get("amendment") == 11:
        say("E1 run 2 (amendment 11): %s s, %s s simulated" % (o["seconds"], o["simulated_s"]))
        say("  train pulses: %s" % o["train_pulses"])
        for n, p in o["protocols"].items():
            say("    %-12s spikes %4d per pairing %s; c peak %s; above d/p %s; change %s" % (
                n, p["spikes"], p["post_spikes_per_pairing"], p["c_peak"], p["above"],
                {k: round(v, 3) for k, v in o["changes"][n].items()}))
        say("  verdicts: %s" % {k: ("PASS" if v else "FAIL") for k, v in o["verdicts"].items()})
        return
    say("E1 at dt = %g ms (%s s; %s s simulated)" % (o["dt"], o["seconds"], o["simulated_s"]))
    say("  settings: %s" % o["settings"])
    say("  rule: %s" % {k: round(v, 4) for k, v in o["rule"].items()})
    say("  fit: %s -> %s" % ({k: round(v, 3) for k, v in o["fit"]["changes"].items()},
                             "PASS" if o["fit"]["passed"] else "FAIL"))
    for n, p in o["protocols"].items():
        say("    %-14s spikes %5d, per pairing %s; c peak %s; time above d/p %s" % (
            n, p["spikes"], p["post_spikes_per_pairing"], p["c_peak"], p["above"]))
    if "changes" in o:
        for n, c in o["changes"].items():
            say("    %-14s %s" % (n, {k: round(v, 3) for k, v in c.items()}))
        say("  verdicts: %s" % {k: ("PASS" if v else "FAIL") for k, v in o["verdicts"].items()})
        say("  T5 (with - without): %s;  A3 (event - no event): %.3f" % (
            {k: round(v, 3) for k, v in o["T5"].items()}, o["A3"]))


def main(argv):
    o = rerun_trains() if "--trains" in argv else run()
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    name = "e1_run2_dt0.05.json" if "--trains" in argv else "e1_dt0.05.json"
    with open(os.path.join(HERE, "runs", name), "w", encoding="utf-8") as fh:
        json.dump(o, fh, indent=1, default=float)
    report(o)


def test_e1_small():
    """Fast: calcium from a held drive reaches drive·τ; the rule's three states; a pathway's calcium is read."""
    c = R._leaky(np.full(20000, 2.0), 0.1, R.TAU_CA)
    assert abs(c[-1] - 2.0 * R.TAU_CA) < 1e-6
    rule = R.Rule(1.0, 2.0, 2.0, 1.0)
    assert rule.apply(np.zeros(100), 0.1) == 1.0
    assert rule.apply(np.full(10000, 1.5), 0.1) < 1.0 < rule.apply(np.full(10000, 3.0), 0.1)
    cell = l5()
    b = cell.index("basal.0")
    r = simulate([Stim().contacts(10.0, b, 5)], 100.0, cell, rec_every=REC, record=[0, b], record_ca=True)
    ca = R.calcium(r.t, r.v(0, b), r.i_ca(0, b), [10.0], cell.area[b], cell.syn)
    assert ca.max() > 0 and ca[-1] < ca.max()


if __name__ == "__main__":
    main(sys.argv[1:])
