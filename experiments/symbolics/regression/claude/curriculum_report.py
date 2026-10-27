"""Experiment B's report (docs/CURRICULUM.md): the held-out results, averaged over seeds, and the inputs to P1-P12."""
import json
import math
from collections import defaultdict

import numpy as np

OUT = "runs/curriculum"
TASKS = ["H1_stepped6_tip", "H2_beamhole_xyr", "H3_plate_r_yc", "H4_stepped4_uniform", "H5_clamped3_mid",
         "H6_beamhole_xr_uniform"]
ARMS = ["A0", "A1", "A2", "A3", "A4", "GP", "law_bending", "law_bending_shear"]


def gmean(v):
    """Geometric mean, for errors spanning orders of magnitude; inf if any is non-finite."""
    v = np.asarray(v, float)
    return float(np.exp(np.mean(np.log(np.maximum(v, 1e-12))))) if np.all(np.isfinite(v)) else float("inf")


def report(out_dir=OUT):
    if out_dir != OUT:
        return report_b2(out_dir)
    rs = [json.loads(l) for l in open(f"{OUT}/heldout_results.jsonl")]
    by = defaultdict(list)
    for r in rs:
        by[(r["task"], r["arm"], r["n"])].append(r)
    summary = {}
    for task in TASKS:
        print(f"\n{task}")
        print(f"   {'arm':18s} {'n':>4s} {'in':>8s} {'ex':>8s} {'far':>8s} {'terms':>6s} {'spread ex':>9s} {'limit':>8s} {'s/fit':>6s}")
        for n in (30, 100):
            for arm in ARMS:
                R = by.get((task, arm, n))
                if not R:
                    continue
                row = dict(nin=gmean([r["nrmse_test"] for r in R]), nex=gmean([r["nrmse_extrap"] for r in R]),
                           nfar=gmean([r["nrmse_far"] for r in R]), seconds=float(np.mean([r["seconds"] for r in R])),
                           terms=[r.get("terms") for r in R], spread=[r.get("spread_ex") for r in R],
                           limit=[r.get("limit") for r in R], per_seed=[(r["nrmse_test"], r["nrmse_extrap"], r["nrmse_far"]) for r in R],
                           formula=sorted(R, key=lambda r: r["seed"])[0].get("formula"))
                summary[(task, arm, n)] = row
                lim = [x for x in row["limit"] if x is not None]
                sp = [x for x in row["spread"] if x is not None]
                print(f"   {arm:18s} {n:4d} {row['nin']:8.4f} {row['nex']:8.4f} {row['nfar']:8.4f} "
                      f"{(np.mean([t for t in row['terms'] if t]) if any(row['terms']) else float('nan')):6.1f} "
                      f"{(np.mean(sp) if sp else float('nan')):9.3f} {(np.mean(lim) if lim else float('nan')):8.3f} {row['seconds']:6.1f}")
        for arm in ("A1", "A2", "A4"):
            row = summary.get((task, arm, 100))
            if row and row["formula"]:
                print(f"   {arm} formula (n = 100, seed 0): {row['formula'][:200]}")
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in summary.items()}, open(f"{OUT}/heldout_summary.json", "w"), indent=1)

    # ---- inputs to the predictions -----------------------------------------------------------------------------------
    S = lambda t, a, n=100: summary.get((t, a, n))
    print("\nP1 / P4: A1 in-range error over A4's, n = 100 (>= 2 = A4 at least 2x better)")
    for t in TASKS:
        print(f"   {t:24s} {S(t, 'A1')['nin'] / S(t, 'A4')['nin']:6.2f}")
    print("P3: H1 in range, A2 over A4:", round(S(TASKS[0], "A2")["nin"] / S(TASKS[0], "A4")["nin"], 2))
    print("P5: H2 in range at n = 100:", {a: round(S(TASKS[1], a)["nin"], 4) for a in ("A0", "A1", "A2", "A3", "A4")})
    v2 = [r for r in rs if r["arm"] in ("A1", "A2", "A3", "A4")]
    worst = max(max(r["nrmse_extrap"], r["nrmse_far"]) for r in v2)
    print(f"P6: largest extrapolation or far NRMSE of any finder-v2 fit: {worst:.3g}")
    fails = [r for r in v2 if r["nrmse_extrap"] > 1]
    ok = [r for r in v2 if r["nrmse_extrap"] <= 1]
    if len(fails) >= 3:
        print(f"P7: of {len(fails)} fits with extrapolation NRMSE > 1, {np.mean([r['spread_ex'] > 0.1 for r in fails]):.0%} have spread > 0.1; "
              f"of the other {len(ok)}, {np.mean([r['spread_ex'] > 0.1 for r in ok]):.0%}")
    else:
        print(f"P7: untestable, {len(fails)} fits with extrapolation NRMSE > 1")
    secs = [r["seconds"] for r in v2]
    t30 = np.mean([r["seconds"] for r in v2 if r["n"] == 30])
    t100 = np.mean([r["seconds"] for r in v2 if r["n"] == 100])
    print(f"P8: held-out fit time max {max(secs):.0f}s, mean n=30 {t30:.1f}s, n=100 {t100:.1f}s (ratio {t100 / t30:.1f})")
    print("P9: A4 against the GP at n = 100 (in, ex)")
    for t in TASKS:
        a, g = S(t, "A4"), S(t, "GP")
        print(f"   {t:24s} in {a['nin']:.4f} vs {g['nin']:.4f} {'A4' if a['nin'] < g['nin'] else 'GP'}   "
              f"ex {a['nex']:.4f} vs {g['nex']:.4f} {'A4' if a['nex'] < g['nex'] else 'GP'}")
    print("P11: candidate laws (<= 6 terms, in <= 0.02, far <= 0.1), per seed, A4 and A1, n = 100")
    for t in TASKS:
        for arm in ("A1", "A4"):
            R = by.get((t, arm, 100), [])
            hits = [r for r in R if r.get("terms", 99) <= 6 and r["nrmse_test"] <= 0.02 and r["nrmse_far"] <= 0.1]
            print(f"   {t:24s} {arm}: {len(hits)}/{len(R)}" + (f"   e.g. {hits[0]['formula'][:150]}" if hits else ""))
    print("P12: limit checks (stepped: formula / textbook at h = 0.15, 1 is ideal; plate: (y - 1) / 3 pi r^2 at r = 0.02, "
          "1 is ideal; holes: |y - 1| at r = 0.01 in units of y's std, 0 is ideal), n = 100")
    for t in TASKS:
        print(f"   {t:24s} " + "  ".join(f"{a} {np.mean([x for x in S(t, a)['limit'] if x is not None]):.3g}"
                                          for a in ("A0", "A1", "A2", "A4", "GP") if S(t, a) and any(x is not None for x in S(t, a)['limit'])))


if __name__ == "__main__":
    report()


def summarize(rs):
    by = defaultdict(list)
    for r in rs:
        by[(r["task"], r["arm"], r["n"])].append(r)
    return {k: dict(nin=gmean([r["nrmse_test"] for r in v]), nex=gmean([r["nrmse_extrap"] for r in v]),
                    nfar=gmean([r["nrmse_far"] for r in v]), terms=[r.get("terms") for r in v],
                    seconds=float(np.mean([r["seconds"] for r in v])), rows=v,
                    formula=sorted(v, key=lambda r: r["seed"])[0].get("formula")) for k, v in by.items()}


def report_b2(out_dir):
    """Experiment B2 beside B: A1-A4 refitted with the noise level known; A0, the GP and the textbook laws from B."""
    rb = [json.loads(l) for l in open(f"{OUT}/heldout_results.jsonl")]
    r2 = [json.loads(l) for l in open(f"{out_dir}/heldout_results.jsonl")]
    B, B2 = summarize(rb), summarize(r2)
    for n in (100, 30):
        print(f"\nn = {n}: NRMSE in / ex / far, geometric mean over 3 seeds (B -> B2 for A1-A4)")
        for task in TASKS:
            print(f"   {task}")
            for arm in ("A0", "GP", "law_bending"):
                if (task, arm, n) in B:
                    v = B[(task, arm, n)]
                    print(f"      {arm:12s} {v['nin']:.4f} / {v['nex']:.4f} / {v['nfar']:.4f}")
            for arm in ("A1", "A2", "A3", "A4"):
                b, v = B[(task, arm, n)], B2[(task, arm, n)]
                print(f"      {arm:12s} B {b['nin']:.4f} / {b['nex']:.4f} / {b['nfar']:.4g}  ->  B2 {v['nin']:.4f} / {v['nex']:.4f} / {v['nfar']:.4g}"
                      f"   terms B {np.median([t for t in b['terms'] if t]):.0f} -> B2 {np.median([t for t in v['terms'] if t]):.0f}")
    for task in TASKS:
        print(f"   {task} A4 formula (B2, n = 100, seed 0): {B2[(task, 'A4', 100)]['formula'][:220]}")
    v2 = [r for r in r2 if r["arm"] in ("A1", "A2", "A3", "A4")]
    a4 = [r for r in r2 if r["arm"] == "A4" and r["n"] == 100]
    print("\nQ1 median A4 formula size at n = 100:", np.median([r["terms"] for r in a4]))
    w100 = max(max(r["nrmse_extrap"], r["nrmse_far"]) for r in v2 if r["n"] == 100)
    w30 = max(max(r["nrmse_extrap"], r["nrmse_far"]) for r in v2 if r["n"] == 30)
    print(f"Q2 largest extrapolation or far NRMSE, finder v2: n = 100 {w100:.3g}; n = 30 {w30:.3g}")
    gb = gmean([B[(t, "A4", 100)]["nfar"] for t in TASKS])
    g2 = gmean([B2[(t, "A4", 100)]["nfar"] for t in TASKS])
    print(f"Q3 A4 far NRMSE, geometric mean over tasks at n = 100: B {gb:.3f} -> B2 {g2:.3f} (ratio {gb / g2:.2f})")
    print("Q4 A4 in range, B2 over B, n = 100:", {t[:2]: round(B2[(t, 'A4', 100)]['nin'] / B[(t, 'A4', 100)]['nin'], 2) for t in TASKS})
    print("Q5 candidate laws (<= 6 terms, in <= 0.02, far <= 0.1), B2, n = 100:")
    for r in sorted(v2, key=lambda r: (r["task"], r["arm"], r["seed"])):
        if r["n"] == 100 and r["terms"] <= 6 and r["nrmse_test"] <= 0.02 and r["nrmse_far"] <= 0.1:
            print(f"   {r['task']} {r['arm']} seed {r['seed']}: in {r['nrmse_test']:.4f} far {r['nrmse_far']:.4f} limit {r['limit']}  {r['formula'][:170]}")
    print(f"Q6 H1 A4 far NRMSE at n = 100: {B2[(TASKS[0], 'A4', 100)]['nfar']:.4f}")
    print("Q8 A4 (B2) against the GP, far, n = 100:", {t[:2]: ('A4' if B2[(t, 'A4', 100)]['nfar'] < B[(t, 'GP', 100)]['nfar'] else 'GP') for t in TASKS})
    print("   and in range:", {t[:2]: ('A4' if B2[(t, 'A4', 100)]['nin'] < B[(t, 'GP', 100)]['nin'] else 'GP') for t in TASKS})
    print("Limit checks, B2 A4 at n = 100 (stepped: 1 ideal; plate: 1 ideal; holes: 0 ideal):",
          {t[:2]: round(float(np.mean([r['limit'] for r in B2[(t, 'A4', 100)]['rows'] if r['limit'] is not None])), 3)
           for t in TASKS if any(r['limit'] is not None for r in B2[(t, 'A4', 100)]['rows'])})
