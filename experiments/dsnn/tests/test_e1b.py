"""E1b: the calcium rule on the corrected type A (LAB.md E1b). Run from experiments/:
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e1b                 (writes dsnn/runs/e1b.json)

**What it runs:** E1's type A protocols (A1 scored; A2, A3 reported) with E1's machinery (`test_e1`), on type A as
rebuilt in E0e (`test_e1.cell_of("A")`), with the rule fitted in E1 run 1; and A4 (reported): the rule at the 25
synapses of pathway X in E0e's XOR setup, for X, X+Y and X+Y+inhibition, 20 s each, seeds 0-2.
"""
import json
from multiprocessing import Pool
import os
import sys
import time

import numpy as np

from dsnn import plasticity as R
from dsnn import protocols as P
from dsnn.sim import Stim, simulate
from dsnn.tests import test_e0e as F
from dsnn.tests import test_e1 as E

HERE = E.HERE
T_A4 = 20000.0
A4_CONDS = dict(X=("bg", "X"), XY=("bg", "X", "Y"), XY_inh=("bg", "X", "Y", "inh"))


def settings_A(job):
    """The rule-set values on the rebuilt type A (E1's rules)."""
    name = job
    A = E.cell_of("A")
    if name == "strengths":
        return name, P.calibrate(A, "A", E.DT)[1]
    comp = A.site("b1.m") if name == "A_b1m_1mV" else A.site("zone")
    return name, E.epsp_n("A", comp, 1.0)


def protocols_A(s, cal):
    A = E.cell_of("A")
    d1, b1m, zone = A.site("b1.d"), A.site("b1.m"), A.site("zone")
    rws = [E.T_START + 125.0 * k for k in range(480)]                     # 8 Hz for 60 s
    c = cal["n6_c"]
    out = [E.spec("A1_clustered", "A", [], 0.0, [("cluster", [(d1, c)])], pre=rws),
           E.spec("A1_dispersed", "A", [], 0.0, [("dispersed", [(x, m) for x, m in P.spread(A, "basal", c)])],
                  pre=rws)]
    smallA = [("b1.m", [(b1m, s["A_b1m_1mV"])])]
    out += [E.spec("A2_%+d_%g" % (tm, f), "A", E.sjostrom(f), float(tm), smallA) for tm in (10, -10) for f in (0.1, 20)]
    if s["A_zone_1mV"] == s["A_zone_1mV"]:
        zp = [("zone_path", [(zone, s["A_zone_1mV"])])]
        n3 = cal["n3_n"]
        out += [E.spec("A3_event", "A", [], 0.0, zp + [("zone_drive", [(zone, n3)])], pre=E.letzkus()),
                E.spec("A3_no_event", "A", [], 0.0, zp + [("zone_drive", [(zone, n3 - 4)])], pre=E.letzkus())]
    return out


def train_times(seed, gid, i, t0, t1, rate):
    """A synapse's spike times, exactly as `sim.simulate` draws them for a train of 1 contact."""
    rng = np.random.default_rng((seed, gid, i))
    return np.sort(rng.uniform(t0, t1, rng.poisson(rate * (t1 - t0) / 1000.0)))


def run_a4(job):
    """A4: X's synapses' calcium and the rule's change, in one condition and seed."""
    seed, cond, rule = job
    A = E.cell_of("A")
    pl = F.placements("uncoupled", seed)
    st = Stim(label=cond)
    pre_X = []
    for g in A4_CONDS[cond]:
        for i, (c, _, _) in enumerate(pl[g]):
            st.train(F.DELAY, T_A4, c, 1, F.RATE, (seed, F.GID[g], i), kind=F.KIND[g])
            if g == "X":
                pre_X.append((c, train_times(seed, F.GID[g], i, F.DELAY, T_A4, F.RATE)))
    comps = sorted({c for c, _ in pre_X})
    z = A.site("zone")
    r = simulate([st], T_A4, A, dt=E.DT, rec_every=E.REC, record=[0, z] + comps, record_ca=True)
    rows = []
    for c, pre in pre_X:
        ca = R.calcium(r.t, r.v(0, c), r.i_ca(0, c), pre, A.area[c], A.syn)
        rows.append(dict(comp=c, dist=float(A.dist[c]), change=float(rule.apply(ca, float(r.t[1] - r.t[0]))),
                         c_peak=float(ca.max()), above_p_ms=float((ca > rule.th_p).sum() * (r.t[1] - r.t[0]))))
    ch = [x["change"] for x in rows]
    v = np.asarray(r.v(0, z), float)
    return dict(seed=seed, cond=cond, mean_change=float(np.mean(ch)), min_change=float(np.min(ch)),
                max_change=float(np.max(ch)), n_ltp=sum(1 for x in ch if x > E.LTP), n_ltd=sum(1 for x in ch if x < E.LTD),
                dcaaps=len(r.dcaap[0][0]), K_mean=float(np.mean([k for _, k, _ in r.dcaap[0][0]])) if r.dcaap[0][0] else None,
                site_median=float(np.median(v[r.t > 200])), soma_spikes=len(r.spikes[0][0]), synapses=rows,
                seconds=round(r.seconds, 1))


def run(workers=7):
    t0 = time.time()
    with open(os.path.join(HERE, "runs", "e1_run1_dt0.05.json"), encoding="utf-8") as fh:
        rule = R.Rule(**json.load(fh)["rule"])
    with Pool(workers) as pool:
        a4 = pool.map_async(run_a4, [(s, c, rule) for s in F.SEEDS for c in A4_CONDS], chunksize=1)
        got = dict(pool.map(settings_A, ["strengths", "A_b1m_1mV", "A_zone_1mV"]))
        cal = got.pop("strengths")
        s = got
        specs = protocols_A(s, cal)
        base = cal["n2_pulse"]
        trains = [sp for sp in specs if sp["name"].endswith("_20")]
        pulses = dict(pool.map(E.train_pulse, [(sp, base) for sp in trains], chunksize=1))
        sims = pool.map(E.run_protocol, [(sp, pulses.get(sp["name"], base)) for sp in specs], chunksize=1)
        a4 = a4.get()
    res = {r["name"]: r for r in sims}
    ch = E.changes(rule, res)
    out = dict(rule=dict(th_d=rule.th_d, th_p=rule.th_p, ratio=rule.ratio, kd=rule.kd), calibration=cal, settings=s,
               train_pulses=pulses, changes=ch,
               A1=ch["A1_clustered"]["cluster"] > E.LTP and ch["A1_dispersed"]["dispersed"] <= E.LTP,
               protocols={n: dict(spikes=r["spikes"], post_spikes_per_pairing=r["post_spikes_per_pairing"],
                                  c_peak={lab: [round(float(np.max(x["c"])), 2) for x in rows]
                                          for lab, rows in r["paths"].items()},
                                  v_peak={lab: [round(x["v_peak"], 1) for x in rows] for lab, rows in r["paths"].items()})
                          for n, r in res.items()},
               A4=a4, seconds=round(time.time() - t0, 1))
    return out


def report(o, say=print):
    say("E1b (%s s)" % o["seconds"])
    say("  calibration %s; settings %s; train pulses %s" % (o["calibration"], o["settings"], o["train_pulses"]))
    for n, p in o["protocols"].items():
        say("    %-14s spikes %5d per pairing %s; c peak %s; v peak %s; change %s" % (
            n, p["spikes"], p["post_spikes_per_pairing"], p["c_peak"] if len(str(p["c_peak"])) < 120 else "(many)",
            p["v_peak"] if len(str(p["v_peak"])) < 120 else "(many)",
            {k: round(v, 3) for k, v in o["changes"][n].items()}))
    say("  A1 -> %s" % ("PASS" if o["A1"] else "FAIL"))
    for x in sorted(o["A4"], key=lambda x: (x["cond"], x["seed"])):
        say("  A4 %-6s seed %d: X's 25 synapses mean change %.3f (min %.3f, max %.3f; %d LTP, %d LTD); %d dCaAPs (K %s); "
            "site median %.1f; soma spikes %d" % (x["cond"], x["seed"], x["mean_change"], x["min_change"],
                                                  x["max_change"], x["n_ltp"], x["n_ltd"], x["dcaaps"],
                                                  None if x["K_mean"] is None else round(x["K_mean"], 3),
                                                  x["site_median"], x["soma_spikes"]))


def main(argv):
    o = run()
    with open(os.path.join(HERE, "runs", "e1b.json"), "w", encoding="utf-8") as fh:
        json.dump(o, fh, indent=1, default=float)
    report(o)


if __name__ == "__main__":
    main(sys.argv[1:])
