"""Experiment A, part 2 (docs/SURROGATES.md): what limits the function finder? Each diagnostic isolates one candidate.
    D1 representation   the best the term library can do: every term at once (ridge, penalty by 5-fold CV), and the
                        best k-term formula for k up to 30, both from all 300 training simulations
    D2 search           the formula symreg chose at n = 100, refound with a beam 8x wider (32): how much lower is
                        its training error?
    D3 term cap         how often cross-validation picks the largest size allowed (8 terms) at n = 100
    D4 locality         bracket: the share of squared test error in the region where a hole matters most (near the
                        root, xc < 1.2, or within 0.1 of an edge), against that region's share of designs
    python -m regression.claude.diagnostics      (from experiments/symbolics, after surrogates simulate and fit)
"""
import json

import numpy as np

from regression.claude.surrogates import OUT, PROBLEMS, make, nrmse
from regression.claude.symreg import SymbolicRegressor, ols_beam_path


def pooled(z):
    return (np.concatenate([z[f"X_train{s}"] for s in range(3)]), np.concatenate([z[f"y_train{s}"] for s in range(3)]))


def d1(name, z):
    X, y = pooled(z)
    names = PROBLEMS[name][1]
    sr = SymbolicRegressor(names)
    out = {}
    for tr in ("identity", "log"):
        t = np.log(y) if tr == "log" else y
        F = sr.lib.numerator(X)
        mu, sd = F.mean(0), F.std(0) + 1e-300
        keep = sd > 1e-12
        G = (F[:, keep] - mu[keep]) / sd[keep]
        Gt = (sr.lib.numerator(z["X_test"])[:, keep] - mu[keep]) / sd[keep]
        Ge = (sr.lib.numerator(z["X_extrap"])[:, keep] - mu[keep]) / sd[keep]
        fold = np.random.default_rng(0).permutation(len(t)) % 5
        best = (np.inf, None)
        for lam in 10.0 ** np.arange(-10, 3):
            err = 0.0
            for f in range(5):
                a, b = fold != f, fold == f
                w = np.linalg.solve(G[a].T @ G[a] + lam * np.eye(G.shape[1]), G[a].T @ (t[a] - t[a].mean()))
                p = G[b] @ w + t[a].mean()
                err += np.sum(((np.exp(np.clip(p, -50, 50)) if tr == "log" else p) - y[b]) ** 2)
            if err < best[0]:
                best = (err, lam)
        w = np.linalg.solve(G.T @ G + best[1] * np.eye(G.shape[1]), G.T @ (t - t.mean()))
        inv = (lambda p: np.exp(np.clip(p, -50, 50))) if tr == "log" else (lambda p: p)
        out[f"ridge_{tr}"] = dict(cv=best[0], nrmse_in=nrmse(inv(Gt @ w + t.mean()), z["y_test"]),
                                  nrmse_ex=nrmse(inv(Ge @ w + t.mean()), z["y_extrap"]), terms=int(keep.sum()))
        A = sr.lib.numerator(X)
        path = ols_beam_path(A, t, 30)
        curve = []
        for s in path:
            design = np.column_stack([np.ones(len(t))] + [A[:, j] for j in s])
            coef = np.linalg.lstsq(design, t, rcond=None)[0]
            At = sr.lib.numerator(z["X_test"])
            curve.append(nrmse(inv(np.column_stack([np.ones(len(At))] + [At[:, j] for j in s]) @ coef), z["y_test"]))
        out[f"kcurve_{tr}"] = curve
    return out


def d2(name, z):
    rows = []
    for seed in range(3):
        X, y = z[f"X_train{seed}"][:100], z[f"y_train{seed}"][:100]
        sr = SymbolicRegressor(PROBLEMS[name][1]).fit(X, y)
        t = np.log(y) if sr.transform == "log" else y
        A, _ = sr._columns(X, t)
        k = len(sr.support)
        valid = lambda s: sr._polynomial(s) or not sr._has_pole(s, sr._coef(X, t, s, A))
        mse = {}
        for beam in (4, 32):
            path = ols_beam_path(A, t, k, beam=beam, valid=valid) if k else [[]]
            s = path[-1] if path else []
            c = sr._coef(X, t, s, A)
            design = np.column_stack([np.ones(len(t))] + [A[:, j] for j in s])
            mse[beam] = float(np.mean((design @ c - t) ** 2))
        rows.append(dict(seed=seed, k=k, train_mse_beam4=mse[4], train_mse_beam32=mse[32],
                         reduction=1 - mse[32] / mse[4] if mse[4] > 0 else 0.0))
    return rows


def d3(name):
    rs = [json.loads(l) for l in open(f"{OUT}/results.jsonl")]
    terms = [r["terms"] for r in rs if r["problem"] == name and r["n"] == 100 and r["method"] == "symreg"]
    return dict(terms=terms, at_cap=sum(t == 9 for t in terms))


def d4(z):
    X, Xt, yt = z["X_train0"][:100], z["X_test"], z["y_test"]
    region = (Xt[:, 0] < 1.2) | (Xt[:, 1] - Xt[:, 2] < 0.1) | (1 - Xt[:, 1] - Xt[:, 2] < 0.1)
    out = dict(design_share=float(region.mean()))
    for method in ("symreg", "gp"):
        e2 = (make(method, "bracket").fit(X, z["y_train0"][:100]).predict(Xt) - yt) ** 2
        out[f"{method}_error_share"] = float(e2[region].sum() / e2.sum())
    return out


def main():
    report = {}
    for name in PROBLEMS:
        z = np.load(f"{OUT}/sims_{name}.npz")
        r = dict(d1=d1(name, z), d2=d2(name, z), d3=d3(name))
        if name == "bracket":
            r["d4"] = d4(z)
        report[name] = r
        print(f"\n{name}")
        for tr in ("identity", "log"):
            rr = r["d1"][f"ridge_{tr}"]
            curve = r["d1"][f"kcurve_{tr}"]
            print(f"   D1 every term ({rr['terms']}, ridge), {tr:8s}: NRMSE in {rr['nrmse_in']:.4f} ex {rr['nrmse_ex']:.4f};  "
                  f"best k-term NRMSE in at k = 4/8/16/30: {[round(curve[min(k, len(curve)) - 1], 4) for k in (4, 8, 16, 30)]}")
        for row in r["d2"]:
            print(f"   D2 seed {row['seed']}: k = {row['k']}, training error with a 32-wide beam {row['reduction']:+.1%} lower")
        print(f"   D3 terms chosen at n = 100: {r['d3']['terms']} (at the cap: {r['d3']['at_cap']} of 3)")
        if "d4" in r:
            d = r["d4"]
            print(f"   D4 region holds {d['design_share']:.0%} of designs, {d['symreg_error_share']:.0%} of symreg's squared "
                  f"error, {d['gp_error_share']:.0%} of the GP's")
    json.dump(report, open(f"{OUT}/diagnostics.json", "w"))


if __name__ == "__main__":
    main()
