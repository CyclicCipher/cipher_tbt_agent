"""E0f: the reduced type A against the full rebuilt cell (LAB.md E0f). Run from experiments/:
    ../venv312/Scripts/python.exe -m dsnn.tests.test_e0f                 (writes dsnn/runs/e0f.json)

**What it scores** (the reduced cell: the five paths and the axon modelled, every other subtree an equivalent
cylinder; the full cell: E0e's):
- R1 passive: R_in at the soma and both dCaAP sites, the steady transfer from each site to the soma (each within
  10%), and the bAP at each site (within 20%);
- R2 thresholds: check T and T2 on the reduced cell, equal to the full's (0.21, 0.50 nA) within one grid step;
- R3 w by the supplement's rule, within 10% of the full's;
- R4 paired behaviour: seeds 0-4, the same synapse points and trains, E0e's F3 and S9 conditions;
- R5 cost.
"""
from dataclasses import replace
import json
from multiprocessing import Pool
import os
import sys
import time

import numpy as np

from dsnn.cells.l23 import l23
from dsnn.sim import Stim, simulate
from dsnn.tests import test_e0e as F

HERE = F.HERE
SEEDS = (0, 1, 2, 3, 4)


def passive(reduced):
    """R1's numbers for one cell (Traub and the dCaAP removed, as E0b's check 2)."""
    out = {}
    for site in ("uncoupled", "coupled"):
        c = l23(site=site, reduced=reduced)
        z = c.site("zone")
        q = replace(c, traub=None, dcaap=None)
        Z = q.impedance()
        bap = simulate([Stim().clamp(10.0, 11.0, 30.0).clamp(11.0, 13.0, -58.0)], 40.0, q, dt=F.DT, record=[0, z])
        out.update({"R_soma": float(Z[0, 0]), "R_" + site: float(Z[z, z]), "transfer_" + site: float(Z[0, z]),
                    "bAP_" + site: float(bap.vmax[0][0][z] - q.rest["V"][z])})
    return out


def cost(job):
    reduced = job
    A = F.cell("uncoupled", None, reduced)
    st = F.stim("uncoupled", 0, ("bg", "X"), reduced)
    simulate([st], 200.0, A, dt=F.DT)                    # warm
    t = time.perf_counter()
    simulate([st], 2000.0, A, dt=F.DT, rec_every=4, record=[0, A.site("zone")])
    return reduced, (time.perf_counter() - t) / 2.0, A.N


def run(workers=7):
    t0 = time.time()
    with open(os.path.join(HERE, "runs", "e0e_run2.json"), encoding="utf-8") as fh:
        full2 = json.load(fh)
    W_full = {s: full2["calibration"][s]["w_pA"] for s in ("uncoupled", "coupled")}
    with Pool(workers) as pool:
        R1 = dict(zip(("full", "reduced"), pool.map(passive, [False, True])))
        cal = dict(zip(("uncoupled", "coupled"), pool.starmap(F.calibrate_w, [("uncoupled", True), ("coupled", True)])))
        W_red = {s: cal[s]["w_pA"] for s in cal}
        T = dict(zip(("uncoupled", "coupled"), pool.starmap(F.check_t, [("uncoupled", W_red["uncoupled"], True),
                                                                         ("coupled", W_red["coupled"], True)])))
        jobs = [(site, s, c, W_red[site], True) for site in ("coupled", "uncoupled") for s in SEEDS
                for c in F.CONDS[site]]
        jobs += [(site, s, c, W_full[site], False) for site in ("coupled", "uncoupled") for s in SEEDS
                 if s not in F.SEEDS for c in F.CONDS[site]]
        rows = pool.map(F.run_condition, jobs, chunksize=1)
        costs = pool.map(cost, [False, True])
    rows += [dict(x, reduced=False) for x in full2["rows"]]                   # seeds 0-2 of the full cell: E0e run 2
    return dict(R1=R1, calibration=cal, w_full=W_full, T=T, rows=rows,
                cost={("reduced" if r else "full"): dict(s_per_s=c, N=n) for r, c, n in costs},
                seconds=round(time.time() - t0, 1))


def score(o):
    f, r = o["R1"]["full"], o["R1"]["reduced"]
    ratio = {k: r[k] / f[k] for k in f}
    R1 = all(abs(v - 1) <= (0.2 if k.startswith("bAP") else 0.1) for k, v in ratio.items())
    R2 = (abs(o["T"]["uncoupled"]["threshold_nA"] - 0.21) <= 0.0101 and abs(o["T"]["coupled"]["threshold_nA"] - 0.50)
          <= 0.0201)
    R3 = all(abs(o["calibration"][s]["w_pA"] / o["w_full"][s] - 1) <= 0.1 for s in o["w_full"])
    by = {(x["site"], x["seed"], x["cond"], x["reduced"]): x for x in o["rows"]}
    pairs = []
    for (site, seed, cond, red), x in by.items():
        if red:
            y = by[(site, seed, cond, False)]
            ok = (abs(x["site_median"] - y["site_median"]) <= 2.0
                  and abs(x["dcaaps"] - y["dcaaps"]) <= 0.2 * y["dcaaps"] + 1)
            if site == "uncoupled":
                ok = ok and abs(x["large"] - y["large"]) <= 2
            else:
                ok = ok and abs(x["soma_spikes"] - y["soma_spikes"]) <= 0.2 * y["soma_spikes"] + 1
            pairs.append(dict(site=site, seed=seed, cond=cond, ok=ok,
                              d_median=round(x["site_median"] - y["site_median"], 2),
                              dcaaps=(x["dcaaps"], y["dcaaps"]), large=(x["large"], y["large"]),
                              soma_spikes=(x["soma_spikes"], y["soma_spikes"])))
    R4 = all(p["ok"] for p in pairs)
    return dict(ratio=ratio, R1=R1, R2=R2, R3=R3, R4=R4, R4_passing=sum(p["ok"] for p in pairs),
                R4_total=len(pairs), pairs=pairs,
                speedup=o["cost"]["full"]["s_per_s"] / o["cost"]["reduced"]["s_per_s"])


def report(o, sc, say=print):
    say("E0f (%s s)" % o["seconds"])
    say("  R1 reduced/full: %s -> %s" % ({k: round(v, 3) for k, v in sc["ratio"].items()}, "PASS" if sc["R1"] else "FAIL"))
    say("  R2 thresholds (reduced): uncoupled %s, coupled %s nA -> %s" % (
        o["T"]["uncoupled"]["threshold_nA"], o["T"]["coupled"]["threshold_nA"], "PASS" if sc["R2"] else "FAIL"))
    say("  R3 w (reduced vs full): %s -> %s" % ({s: (round(o["calibration"][s]["w_pA"]), round(o["w_full"][s]))
                                                for s in o["w_full"]}, "PASS" if sc["R3"] else "FAIL"))
    for p in sorted(sc["pairs"], key=lambda p: (p["site"], p["cond"], p["seed"])):
        say("    %-9s %-14s seed %d: d median %+.2f mV; dCaAPs red/full %s; large %s; soma spikes %s %s" % (
            p["site"], p["cond"], p["seed"], p["d_median"], p["dcaaps"], p["large"], p["soma_spikes"],
            "" if p["ok"] else "<- outside"))
    say("  R4 paired: %d of %d within tolerance -> %s" % (sc["R4_passing"], sc["R4_total"], "PASS" if sc["R4"] else "FAIL"))
    say("  R5 cost: %s; speed-up %.1fx" % (o["cost"], sc["speedup"]))
    for red in (True, False):
        for site in ("uncoupled", "coupled"):
            for cond in F.CONDS[site]:
                xs = [x for x in o["rows"] if x["reduced"] == red and x["site"] == site and x["cond"] == cond]
                say("    %s %-9s %-14s large %s, soma rate %s" % ("reduced" if red else "full   ", site, cond,
                                                                [x["large"] for x in sorted(xs, key=lambda x: x["seed"])],
                                                                [round(x["soma_rate"], 2) for x in sorted(xs, key=lambda x: x["seed"])]))


def main(argv):
    o = run()
    sc = score(o)
    with open(os.path.join(HERE, "runs", "e0f.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(o, score=sc), fh, indent=1, default=float)
    report(o, sc)


if __name__ == "__main__":
    main(sys.argv[1:])
