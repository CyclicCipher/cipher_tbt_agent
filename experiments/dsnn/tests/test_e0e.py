"""E0e: type A rebuilt as Gidon et al. built it, and their Fig. 3 replicated (LAB.md E0e). Run from experiments/:
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e0e                 (run 2; writes dsnn/runs/e0e_run2.json)
    ../venv312/Scripts/python.exe -m pytest dsnn/tests/test_e0e.py -q    (a fast check)

**Run 2 (amendment 8):** w is set by their stated rule (the first dCaAP at the threshold step 50 mV tall at the
uncoupled site, 60 mV at the coupled one), and dt is 0.05 ms, as their `plots.hoc` sets it.

**What it scores:**
- check T: their test pulse's dCaAP threshold, 0.22 ± 0.01 nA; check T2: the coupled site's, 0.50 ± 0.02 nA (check M
  is reported);
- F3: XOR at the dendrite (Fig. 3B–E), on each of 3 seeds: X ≥ 3, Y ≥ 3, X+Y ≤ 1, X+Y+inhibition ≥ 3 large dCaAPs
  (site voltage peaking above 0 mV) in 4 s;
- S9: the soma's output (Fig. S9), on each seed: X's and Y's rates above X+Y's, and distal inhibition raising X+Y's.
"""
import json
from multiprocessing import Pool
import os
import sys
import time

import numpy as np

from dsnn import protocols as P
from dsnn.cells.l23 import l23
from dsnn.sim import AMPA, CONTACT, GABA_F, GABA_S, Stim, simulate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # dsnn/
DT = 0.05                           # their plots.hoc: steps_per_ms = 20 (amendment 9)
SEEDS = (0, 1, 2)
DELAY, RATE = 100.0, 20.0           # their syns: from 100 ms, 20 Hz
GRID = dict(uncoupled=[round(0.18 + 0.01 * k, 2) for k in range(13)],     # their test pulses (nA)
            coupled=[round(0.48 + 0.02 * k, 2) for k in range(7)])
TARGET = dict(uncoupled=50.0, coupled=60.0)                                 # mV, the supplement's dCaAP at threshold
THRESHOLD = dict(uncoupled=(0.22, 0.0101), coupled=(0.50, 0.0201))          # nA, ± tolerance
S9_THEIRS = dict(X=1.7, XY=0.54, XY_distal_inh=1.5, X_proximal_inh=0.25)     # Hz, the supplement's Fig. S9
_CELLS = {}


def cell(site, w=None):
    """Type A with its dCaAP at `site`; w (pA) replaces density × segment area when given (amendment 8)."""
    if (site, w) not in _CELLS:
        c = l23(site=site)
        _CELLS[(site, w)] = c.with_dcaap(w=w) if w else c
    return _CELLS[(site, w)]


# ---- check M and check T -------------------------------------------------------------------------------------------
def check_m():
    A = cell("uncoupled")
    t = A.tree
    return dict(N=A.N, dist_apic76=t.dist("apic[76]", 0.214), dist_apic60=t.dist("apic[60]", 0.99),
                R_in_soma=A.input_resistance(), R_in_zone=A.input_resistance(A.site("zone")), w_pA=A.dcaap.w,
                membrane_um2=float(A.area.sum()))


def dcaap_peaks(r, b, z, events):
    """Each dCaAP's (onset ms, K, peak site voltage in the 40 ms from its onset)."""
    v, t = np.asarray(r.v(b, z), float), r.t
    return [(on, K, float(v[(t >= on) & (t <= on + 40.0)].max())) for on, K, _ in events]


def first_amplitude(A, nA):
    """The first dCaAP's height at a 1 s step of nA from 200 ms: its peak minus the mean of the 1 ms before onset."""
    z = A.site("zone")
    r = simulate([Stim().current(200.0, 1200.0, z, 1000.0 * nA)], 700.0, A, dt=DT, rec_every=1, record=[0, z])
    if not r.dcaap[0][0]:
        return None
    on = r.dcaap[0][0][0][0]
    v, t = np.asarray(r.v(0, z), float), r.t
    return float(v[(t >= on) & (t <= on + 40.0)].max() - v[(t >= on - 1.0) & (t < on)].mean())


def calibrate_w(site):
    """Amendment 8: the threshold step on their grid, then w by bisection so that the first dCaAP there is TARGET mV
    tall (± 1 mV)."""
    A = cell(site)
    z = A.site("zone")
    r = simulate([Stim().current(200.0, 1200.0, z, 1000.0 * a) for a in GRID[site]], 1400.0, A, dt=DT)
    th = next((a for b, a in enumerate(GRID[site]) if r.dcaap[b][0]), None)
    lo, hi = 0.05 * A.dcaap.w, 1.5 * A.dcaap.w
    steps = []
    for _ in range(30):
        w = 0.5 * (lo + hi)
        amp = first_amplitude(A.with_dcaap(w=w), th)
        steps.append((round(w, 1), None if amp is None else round(amp, 2)))
        if amp is not None and abs(amp - TARGET[site]) <= 1.0:
            break
        if amp is None or amp > TARGET[site]:
            hi = w
        else:
            lo = w
    return dict(threshold_nA=th, w_pA=w, w_density_pA=A.dcaap.w, amplitude=amp, steps=steps)


def check_t(site="uncoupled", w=None):
    A = cell(site, w)
    z = A.site("zone")
    amps = GRID[site]
    r = simulate([Stim().current(200.0, 1200.0, z, 1000.0 * a) for a in amps], 1400.0, A, dt=DT, rec_every=4,
                 record=[0, z])
    rows = []
    for b, a in enumerate(amps):
        ev = dcaap_peaks(r, b, z, r.dcaap[b][0])
        rows.append(dict(nA=a, dcaaps=len(ev), soma_spikes=len(r.spikes[b][0]),
                         K=[round(k, 3) for _, k, _ in ev], peaks=[round(p, 1) for _, _, p in ev],
                         onsets=[round(o, 1) for o, _, _ in ev]))
    th = next((x["nA"] for x in rows if x["dcaaps"]), None)
    want, tol = THRESHOLD[site]
    return dict(site=site, rows=rows, threshold_nA=th, passed=th is not None and abs(th - want) <= tol,
                seconds=r.seconds)


# ---- F3 and S9 -------------------------------------------------------------------------------------------------------
def placements(site, seed):
    """Every synapse's compartment for one seed (their point lists, drawn by our generator)."""
    A = cell(site)
    rng = lambda k: np.random.default_rng((seed, k))
    n_path = 25 if site == "uncoupled" else 35
    n_inh = 20 if site == "uncoupled" else 35
    out = dict(bg=P.place(A, "all_no_axon", 100, rng(3)), X=P.place(A, "subdomain", n_path, rng(2)),
               Y=P.place(A, "subdomain", n_path, rng(4)))
    if site == "uncoupled":
        out["inh"] = P.place(A, "subdomain", n_inh, rng(5))
    else:
        out["inh_distal"] = P.place(A, "subdomain", n_inh, rng(5), min_dist=700.0)
        out["inh_proximal"] = P.place(A, "basal", n_inh, rng(10))
    return out


KIND = dict(bg=AMPA, X=CONTACT, Y=CONTACT, inh=GABA_S, inh_distal=GABA_S, inh_proximal=GABA_F)
GID = dict(bg=0, X=1, Y=2, inh=3, inh_distal=4, inh_proximal=5)
CONDS = dict(uncoupled=dict(X=("bg", "X"), Y=("bg", "Y"), XY=("bg", "X", "Y"), XY_inh=("bg", "X", "Y", "inh")),
             coupled=dict(X=("bg", "X"), Y=("bg", "Y"), XY=("bg", "X", "Y"),
                          XY_distal_inh=("bg", "X", "Y", "inh_distal"), X_proximal_inh=("bg", "X", "inh_proximal")))
T_RUN = dict(uncoupled=4000.0, coupled=20500.0)


def stim(site, seed, groups):
    """Each synapse its own Poisson train at 20 Hz from 100 ms; the same train for a synapse in every condition."""
    pl = placements(site, seed)
    st = Stim(label="+".join(groups))
    for g in groups:
        for i, (c, _, _) in enumerate(pl[g]):
            st.train(DELAY, T_RUN[site], c, 1, RATE, (seed, GID[g], i), kind=KIND[g])
    return st


def run_condition(job):
    site, seed, cond, w = job
    A = cell(site, w)
    z = A.site("zone")
    r = simulate([stim(site, seed, CONDS[site][cond])], T_RUN[site], A, dt=DT, rec_every=4, record=[0, z])
    ev = dcaap_peaks(r, 0, z, r.dcaap[0][0])
    v = np.asarray(r.v(0, z), float)
    so = np.asarray(r.v(0, 0), float)
    spk = r.spikes[0][0]
    return dict(site=site, seed=seed, cond=cond, dcaaps=len(ev), large=sum(1 for _, _, p in ev if p > 0.0),
                K_mean=float(np.mean([k for _, k, _ in ev])) if ev else None,
                peaks=[round(p, 1) for _, _, p in ev][:40], K=[round(k, 3) for _, k, _ in ev][:40],
                site_median=float(np.median(v[r.t > 200])), site_p90=float(np.percentile(v[r.t > 200], 90)),
                soma_median=float(np.median(so[r.t > 200])), soma_spikes=len(spk),
                soma_rate=len(spk) / ((T_RUN[site] - DELAY) / 1000.0), seconds=round(r.seconds, 1))


def score(rows):
    by = {(x["site"], x["seed"], x["cond"]): x for x in rows}
    f3 = {s: (by[("uncoupled", s, "X")]["large"] >= 3 and by[("uncoupled", s, "Y")]["large"] >= 3
              and by[("uncoupled", s, "XY")]["large"] <= 1 and by[("uncoupled", s, "XY_inh")]["large"] >= 3)
          for s in SEEDS}
    rate = lambda s, c: by[("coupled", s, c)]["soma_rate"]
    s9 = {s: (rate(s, "X") > rate(s, "XY") and rate(s, "Y") > rate(s, "XY")
              and rate(s, "XY_distal_inh") > rate(s, "XY")) for s in SEEDS}
    return dict(F3=f3, F3_passed=all(f3.values()), S9=s9, S9_passed=all(s9.values()))


def run(workers=7):
    t0 = time.time()
    with Pool(workers) as pool:
        cal = dict(zip(("uncoupled", "coupled"), pool.map(calibrate_w, ["uncoupled", "coupled"])))
        W = {site: cal[site]["w_pA"] for site in cal}
        jobs = [(site, s, c, W[site]) for site in ("coupled", "uncoupled") for s in SEEDS for c in CONDS[site]]
        tjobs = [pool.apply_async(check_t, (site, W[site])) for site in ("uncoupled", "coupled")]
        rows = pool.map(run_condition, jobs, chunksize=1)
        ct, ct2 = (j.get() for j in tjobs)
    return dict(dt=DT, amendment=8, calibration=cal, M=check_m(), T=ct, T2=ct2, rows=rows, score=score(rows),
                S9_theirs=S9_THEIRS, seconds=round(time.time() - t0, 1))


def report(o, say=print):
    say("E0e (%s s)" % o["seconds"])
    m = o["M"]
    say("  check M: %d compartments; soma -> apic[76](0.214) %.1f um, apic[60](0.99) %.1f um; R_in soma %.1f MOhm, "
        "zone %.1f; w %.0f pA; membrane %.0f um2" % (m["N"], m["dist_apic76"], m["dist_apic60"], m["R_in_soma"],
                                                   m["R_in_zone"], m["w_pA"], m["membrane_um2"]))
    for site, c in o.get("calibration", {}).items():
        say("  calibration %s: threshold %s nA; w %.0f pA (density x area: %.0f); first dCaAP %s mV; steps %s" % (
            site, c["threshold_nA"], c["w_pA"], c["w_density_pA"], c["amplitude"], c["steps"]))
    for key in ("T", "T2"):
        if key not in o:
            continue
        t = o[key]
        say("  check %s (%s): threshold %s nA -> %s" % (key, t.get("site", "uncoupled"), t["threshold_nA"],
                                                          "PASS" if t["passed"] else "FAIL"))
        for x in t["rows"]:
            say("    %.2f nA: %d dCaAPs, K %s, peaks %s, soma spikes %d" % (x["nA"], x["dcaaps"], x["K"], x["peaks"],
                                                                         x["soma_spikes"]))
    for x in sorted(o["rows"], key=lambda x: (x["site"], x["cond"], x["seed"])):
        say("  %-9s %-14s seed %d: %2d dCaAPs (%2d large), K mean %s; site median %.1f p90 %.1f; soma median %.1f, "
            "%d spikes (%.2f Hz)" % (x["site"], x["cond"], x["seed"], x["dcaaps"], x["large"],
                                     None if x["K_mean"] is None else round(x["K_mean"], 3), x["site_median"],
                                     x["site_p90"], x["soma_median"], x["soma_spikes"], x["soma_rate"]))
    sc = o["score"]
    say("  F3 (XOR at the dendrite) by seed %s -> %s" % (sc["F3"], "PASS" if sc["F3_passed"] else "FAIL"))
    say("  S9 (the soma's output) by seed %s -> %s" % (sc["S9"], "PASS" if sc["S9_passed"] else "FAIL"))
    if "S9_theirs" in o:
        mean = lambda c: np.mean([x["soma_rate"] for x in o["rows"] if x["site"] == "coupled" and x["cond"] == c])
        say("  S9 rates, mean over seeds vs theirs: %s" % {c: (round(float(mean(c)), 2), v)
                                                           for c, v in o["S9_theirs"].items()})


def main(argv):
    o = run()
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    with open(os.path.join(HERE, "runs", "e0e_run2.json"), "w", encoding="utf-8") as fh:
        json.dump(o, fh, indent=1, default=float)
    report(o)


def test_e0e_small():
    """Fast: the rebuilt cell's scale and site; placement stays in its list; GABA hyperpolarizes; the dCaAP fires."""
    A = cell("uncoupled")
    assert abs(A.tree.dist("apic[76]", 0.214) - 536.4) < 1.0
    pl = P.place(A, "subdomain", 10, np.random.default_rng(0))
    assert all(s in A.seclists["subdomain"] for _, s, _ in pl)
    far = P.place(A, "subdomain", 5, np.random.default_rng(1), min_dist=700.0)
    assert all(A.tree.dist(s, x) > 700.0 for _, s, x in far)
    z = A.site("zone")
    sub = Stim().current(10.0, 60.0, z, 50.0)
    r = simulate([sub, Stim(syn=[(20.0, 0, z, 20.0, GABA_S)]).current(10.0, 60.0, z, 50.0),
                  Stim().current(10.0, 400.0, z, 300.0)], 400.0, A, dt=DT, record=[0, z])
    assert r.vmax[1][0][z] < r.vmax[0][0][z] - 0.5 and len(r.dcaap[2][0]) >= 1


if __name__ == "__main__":
    main(sys.argv[1:])
