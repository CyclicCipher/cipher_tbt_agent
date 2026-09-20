"""Aggregate `runs/*/eval.json` into tables, plots and the pre-registered verdicts.

    python experiments/mdl_vs_ntp/analyze.py

Statistics: a two-level bootstrap (resample seeds, then tasks within each seed), 10k reps; arm-vs-arm deltas paired by
seed index; means with 95% CIs. A prediction is PASS if the CI excludes 0 (or clears the stated margin) in the predicted
direction, FAIL if it excludes it the other way, INCONCLUSIVE otherwise; a composite passes only if all parts pass.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS, RES = HERE / "runs", HERE / "results"
N_BOOT = 10000
import argparse                                                       # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument("--runs", default=None)
_ap.add_argument("--out", default=None)
_args, _ = _ap.parse_known_args()
if _args.runs:
    RUNS = Path(_args.runs)
if _args.out:
    RES = Path(_args.out)


def load_evals():
    out = {}
    for p in sorted(RUNS.glob("*/eval.json")):
        r = json.load(open(p))
        out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


# ── metric extractors: eval.json -> per-task 0/1 (or value) arrays ───────────────────────────────────────────────────
def per_task(r, metric):
    kind, name, k = metric
    if kind == "e1":
        return np.array(r["e1"][name]["per_task"], dtype=float)
    if kind == "v":
        return np.array(r["e3"][name]["per_task_verified"][str(k)] if str(k) in r["e3"][name]["per_task_verified"]
                        else r["e3"][name]["per_task_verified"][k], dtype=float)
    if kind == "p":
        d = r["e3"][name]["per_task_pass"]
        return np.array(d[str(k)] if str(k) in d else d[k], dtype=float)
    if kind == "nll":
        return np.array(r["e2"][name]["per_task"], dtype=float)
    raise KeyError(metric)


def boot_mean(arm_runs, metric, rng):
    """Two-level bootstrap of the mean over seeds x tasks."""
    seeds = sorted(arm_runs)
    data = [per_task(arm_runs[s], metric) for s in seeds]
    reps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(len(seeds), size=len(seeds))
        vals = [data[i][rng.integers(len(data[i]), size=len(data[i]))].mean() for i in pick]
        reps[b] = np.mean(vals)
    point = float(np.mean([d.mean() for d in data]))
    return point, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def boot_delta(a_runs, b_runs, metric, rng):
    """Paired by seed index: seeds present in both arms."""
    seeds = sorted(set(a_runs) & set(b_runs))
    da = [per_task(a_runs[s], metric) for s in seeds]
    db = [per_task(b_runs[s], metric) for s in seeds]
    reps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(len(seeds), size=len(seeds))
        vals = []
        for i in pick:
            ia = rng.integers(len(da[i]), size=len(da[i]))
            ib = rng.integers(len(db[i]), size=len(db[i]))
            vals.append(da[i][ia].mean() - db[i][ib].mean())
        reps[b] = np.mean(vals)
    point = float(np.mean([x.mean() - y.mean() for x, y in zip(da, db)]))
    return point, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)), len(seeds)


def verdict(lo, hi, direction, margin=0.0):
    """direction +1: predicted delta > margin; -1: predicted delta < margin."""
    if direction > 0:
        return "PASS" if lo > margin else ("FAIL" if hi < margin else "INCONCLUSIVE")
    return "PASS" if hi < margin else ("FAIL" if lo > margin else "INCONCLUSIVE")


def main():
    rng = np.random.default_rng(0)
    evals = load_evals()
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    (RES / "plots").mkdir(parents=True, exist_ok=True)
    arms = sorted(evals)
    metrics = {"id_em": ("e1", "e1_id", None)}
    for sp in ("hfresh", "hcomp", "hdepth", "hnovel"):
        metrics[f"v16_{sp}"] = ("v", f"e3_{sp}", 16)
        metrics[f"p128_{sp}"] = ("p", f"e3_{sp}", 128)
    metrics["noisy_digit_nll"] = ("nll", "e2_noisy", None)
    metrics["partial_nll"] = ("nll", "e2_partial", None)

    # main.csv: arm x split x metric with CIs
    rows = []
    for arm in arms:
        for mname, m in metrics.items():
            pt, lo, hi = boot_mean(evals[arm], m, rng)
            rows.append(dict(arm=arm, metric=mname, mean=pt, ci_lo=lo, ci_hi=hi, seeds=len(evals[arm])))
        for sp in ("hfresh", "hcomp", "hdepth", "hnovel"):
            med = [evals[arm][s]["e3"][f"e3_{sp}"]["samples_to_first_correct_median"] for s in evals[arm]]
            rows.append(dict(arm=arm, metric=f"samples_to_solve_{sp}", mean=float(np.nanmean(med)), ci_lo="", ci_hi="",
                             seeds=len(evals[arm])))
    _csv(RES / "tables" / "main.csv", rows)

    # paired deltas
    for a, b in (("mdl", "ntp"), ("mdl", "exit"), ("mdl", "mdl_nolib"), ("mdl", "mdl_insample"), ("curio", "mdl")):
        if a in evals and b in evals:
            rows = []
            for mname, m in metrics.items():
                pt, lo, hi, n = boot_delta(evals[a], evals[b], m, rng)
                rows.append(dict(metric=mname, delta=pt, ci_lo=lo, ci_hi=hi, paired_seeds=n))
            _csv(RES / "tables" / f"paired_{a}_vs_{b}.csv", rows)

    # library and curriculum tables
    lib_rows, cur_rows = [], []
    for arm in arms:
        for s, r in evals[arm].items():
            e4 = r["e4"]
            lib_rows.append(dict(arm=arm, seed=s, n_macros=e4["n_macros"], recovered_exact=e4["recovered_exact"],
                                 recovered_partial=e4["recovered_partial"], precision_aligned=e4["precision_aligned"],
                                 family_specific_rate=e4["family_specific_rate"]))
            share = r["e5"]["realized_share"]
            noisy_share = share.get("noisy", 0.0)
            cur_rows.append(dict(arm=arm, seed=s, noisy_share=noisy_share, partial_share=share.get("partial", 0.0),
                                 **{f"mass_{k}": v for k, v in share.items()}))
    _csv(RES / "tables" / "library.csv", lib_rows)
    _csv(RES / "tables" / "curriculum.csv", cur_rows)

    # compute.csv
    comp = []
    for arm in arms:
        for s in evals[arm]:
            cfgp = RUNS / f"{arm}_s{s}" / "train.jsonl"
            last = None
            if cfgp.exists():
                for line in open(cfgp):
                    last = json.loads(line)
            if last:
                comp.append(dict(arm=arm, seed=s, te=last["te"], steps=last["step"], wall=last["wall"],
                                 cpu_seconds=last["cpu_seconds"]))
    _csv(RES / "tables" / "compute.csv", comp)

    # pre-registered predictions
    preds = []

    def delta(a, b, m):
        return boot_delta(evals[a], evals[b], m, rng) if (a in evals and b in evals) else None

    d = delta("mdl", "ntp", metrics["id_em"])
    preds.append(("H1 competence", "ID EM lower CI(mdl-ntp) >= -2pts", d,
                  ("PASS" if d[1] >= -0.02 else "FAIL") if d else "MISSING"))
    for sp in ("hcomp", "hdepth"):
        for other in ("ntp", "exit"):
            d = delta("mdl", other, metrics[f"v16_{sp}"])
            preds.append((f"H2 search efficiency [{sp}, vs {other}]", "v@16 mdl > other", d,
                          verdict(d[1], d[2], +1) if d else "MISSING"))
    d = delta("mdl", "ntp", metrics["p128_hdepth"])
    preds.append(("H3 discovery", "pass@128 H-depth mdl > ntp", d, verdict(d[1], d[2], +1) if d else "MISSING"))
    if "mdl" in evals:
        rec = [evals["mdl"][s]["e4"]["recovered_exact"] + evals["mdl"][s]["e4"]["recovered_partial"] for s in evals["mdl"]]
        ok = sum(1 for r in rec if r >= 5)
        preds.append(("H4a invention", ">=5/10 planted in >=2 of 3 seeds", tuple(rec), "PASS" if ok >= 2 else "FAIL"))
    d = delta("mdl", "mdl_nolib", metrics["v16_hcomp"])
    preds.append(("H4b library matters", "H-comp v@16 mdl > mdl_nolib", d, verdict(d[1], d[2], +1) if d else "MISSING"))
    if "mdl" in evals and "mdl_insample" in evals:
        fs_in = np.mean([r["e4"]["family_specific_rate"] for r in evals["mdl_insample"].values()])
        fs_mdl = np.mean([r["e4"]["family_specific_rate"] for r in evals["mdl"].values()])
        d = delta("mdl", "mdl_insample", metrics["v16_hcomp"])
        v = verdict(d[1], d[2], +1)
        preds.append(("H4c anti-leakage", "fam_specific(insample) > fam_specific(mdl) AND H-comp v@16 mdl > insample",
                      (fs_in, fs_mdl, d), "PASS" if (fs_in > fs_mdl and v == "PASS") else ("FAIL" if v == "FAIL" else "INCONCLUSIVE")))
    if "curio" in evals and "mdl" in evals:
        cs = np.mean([r["e5"]["realized_share"].get("noisy", 0) for r in evals["curio"].values()])
        ms = np.mean([r["e5"]["realized_share"].get("noisy", 0) for r in evals["mdl"].values()])
        preds.append(("H5 noisy TV", "noisy share curio >= 30% and mdl <= 8%", (cs, ms),
                      "PASS" if (cs >= 0.30 and ms <= 0.08) else "FAIL"))
    d = delta("mdl", "ntp", metrics["v16_hnovel"])
    preds.append(("H6 no harm", "H-novel v@16 lower CI(mdl-ntp) >= -3pts", d,
                  ("PASS" if d[1] >= -0.03 else "FAIL") if d else "MISSING"))
    _csv(RES / "tables" / "predictions_check.csv",
         [dict(id=i, rule=r, numbers=str(n), verdict=v) for i, r, n, v in preds])
    for i, r, n, v in preds:
        print(f"{v:<13} {i:<40} {r}   {n}")
    plots(evals, arms, metrics)


def plots(evals, arms, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    # verified@k / pass@k curves per split
    for sp in ("hfresh", "hcomp", "hdepth", "hnovel"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for arm in arms:
            for ax, kind in zip(axes, ("v", "p")):
                pts, los, his = [], [], []
                for k in (1, 4, 16, 64, 128):
                    pt, lo, hi = boot_mean(evals[arm], (kind, f"e3_{sp}", k), rng)
                    pts.append(pt)
                    los.append(lo)
                    his.append(hi)
                ax.plot([1, 4, 16, 64, 128], pts, marker="o", label=arm)
                ax.fill_between([1, 4, 16, 64, 128], los, his, alpha=0.15)
                ax.set_xscale("log")
                ax.set_title(f"{'verified' if kind == 'v' else 'oracle pass'}@k — {sp}")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(RES / "plots" / f"search_{sp}.png", dpi=120)
        plt.close(fig)
    # ID EM vs TE (from e1_curve), noisy-digit NLL by arm, library size / J_S / reward / regret, curriculum mass
    fig, ax = plt.subplots(figsize=(5, 4))
    for arm in arms:
        xs, ys = [], []
        for s, r in evals[arm].items():
            c = r.get("e1_curve", {})
            for tag, v in sorted(c.items(), key=lambda kv: int(kv[0])):
                xs.append(int(tag))
                ys.append(v["e1_id"])
            xs.append(100)
            ys.append(r["e1"]["e1_id"]["em"])
        if xs:
            ax.plot(xs, ys, marker="o", label=arm)
    ax.set_xlabel("% of B")
    ax.set_ylabel("ID EM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "plots" / "id_em_vs_te.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(arms, [np.mean([r["e2"]["e2_noisy"]["bits"] for r in evals[a].values()]) for a in arms])
    ax.axhline(np.log2(10), ls="--", c="k")
    ax.set_ylabel("noisy-digit NLL (bits)")
    fig.tight_layout()
    fig.savefig(RES / "plots" / "noisy_digit_nll.png", dpi=120)
    plt.close(fig)
    for arm in arms:
        for s, r in evals[arm].items():
            tr = r["e4"]["trajectory"]
            if not tr:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
            axes[0].plot([t["round"] for t in tr], [t["size"] for t in tr])
            axes[0].set_title(f"{arm} s{s}: library size")
            axes[1].plot([t["round"] for t in tr], [t["J_S"] for t in tr])
            axes[1].set_title("J_S")
            axes[2].plot([t["round"] for t in tr], [t["reward"] for t in tr], label="reward")
            axes[2].plot([t["round"] for t in tr], [t["regret"] for t in tr], label="regret")
            axes[2].legend()
            fig.tight_layout()
            fig.savefig(RES / "plots" / f"library_{arm}_s{s}.png", dpi=120)
            plt.close(fig)
            rounds = r["e5"]["rounds"]
            if rounds:
                cats = sorted({k for rr in rounds for k in rr["mass"]})
                fig, ax = plt.subplots(figsize=(7, 4))
                for c in cats:
                    ax.plot([rr["round"] for rr in rounds], [rr["mass"].get(c, 0) for rr in rounds], label=c)
                ax.set_title(f"{arm} s{s}: curriculum mass by category")
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(RES / "plots" / f"curriculum_{arm}_s{s}.png", dpi=120)
                plt.close(fig)


def _csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
