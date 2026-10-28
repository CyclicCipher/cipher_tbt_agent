"""Experiment A (docs/SURROGATES.md): can the function finder turn a few simulations into a compact, accurate formula?

Four problems from fem2d.py, three training budgets (10, 30, 100 simulations; 3 random seeds each), the surrogates:
    symreg      symreg.py: greedy residual term selection with refitting and a beam, size by cross-validation
    gp          Gaussian process (Matern 5/2, one length scale per input, noise term; scikit-learn), inputs scaled to
                the box; y or log y chosen by 5-fold cross-validation, as for symreg
    quadratic   a full quadratic polynomial (the classical response surface), least squares
  part 2 (docs/SURROGATES.md, "Part 2"):
    symreg_margins, gp_margins   the same, given the design's constraint margins as extra inputs (plate, bracket)
    hybrid      symreg's formula as the trend, plus a Gaussian process on its residuals (in symreg's y or log y)
Measures: NRMSE = RMSE / standard deviation of y, on 500 random in-range designs and on 200 extrapolation designs;
for the bracket, the regret of optimizing the surrogate instead of the simulator over a grid of hole positions.
    python -m regression.claude.surrogates simulate      (from experiments/symbolics; ~1 min, 8 processes)
    python -m regression.claude.surrogates fit
    python -m regression.claude.surrogates report
"""
import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np

from regression.claude import fem2d
from regression.claude.symreg import SymbolicRegressor

OUT = "runs/surrogates"
BUDGETS, SEEDS, POOL, N_TEST, N_EXTRAP = (10, 30, 100), (0, 1, 2), 100, 500, 200
OPT_R, OPT_GRID = 0.15, (np.linspace(0.6, 3.4, 15), np.linspace(0.25, 0.65, 9))

PROBLEMS = {   # name: (simulator, input names, box low, box high, extrapolation low, high, constraint[, extrap constraint])
    "beam": (fem2d.beam, ["L"], [2.0], [10.0], [10.0], [14.0], None),
    "plate": (fem2d.plate_hole, ["r", "yc"], [0.05, 0.5], [0.3, 0.7], [0.3, 0.5], [0.38, 0.52],
              lambda x: x[1] + x[0] <= 0.9),
    "bracket": (fem2d.bracket, ["xc", "yc", "r"], [0.6, 0.25, 0.05], [3.4, 0.65, 0.2], [0.6, 0.25, 0.2],
                [3.4, 0.65, 0.26], lambda x: x[1] - x[2] >= 0.05 and x[1] + x[2] <= 0.95),
    "stepped": (fem2d.stepped_beam, [f"h{i}" for i in range(1, 7)], [0.4] * 6, [1.0] * 6, [0.3] * 6, [1.0] * 6, None,
                lambda x: x.min() < 0.4),
}
MARGINS = {    # the design's constraint margins: names, function of X, range low, high
    "plate": (["s"], lambda X: (0.9 - X[:, 1] - X[:, 0])[:, None], [0.0], [0.35]),
    "bracket": (["s1", "s2"], lambda X: np.stack([X[:, 1] - X[:, 2] - 0.05, 0.95 - X[:, 1] - X[:, 2]], 1),
                [0.0, 0.1], [0.55, 0.65]),
}
METHODS = ("symreg", "gp", "quadratic", "symreg_margins", "gp_margins", "hybrid")


def sample(rng, lo, hi, n, ok):
    out = []
    while len(out) < n:
        x = rng.uniform(lo, hi)
        if ok is None or ok(x):
            out.append(x)
    return np.array(out)


def run_sim(job):
    name, x = job
    return PROBLEMS[name][0](*x)


def simulate():
    os.makedirs(OUT, exist_ok=True)
    for name, (_, names, lo, hi, elo, ehi, ok, *ok_ex) in PROBLEMS.items():
        if os.path.exists(f"{OUT}/sims_{name}.npz"):
            continue
        t0 = time.time()
        rng = np.random.default_rng([99, len(name)])
        sets = {f"train{s}": sample(np.random.default_rng([s, len(name)]), lo, hi, POOL, ok) for s in SEEDS}
        sets["test"] = sample(rng, lo, hi, N_TEST, ok)
        sets["extrap"] = sample(rng, elo, ehi, N_EXTRAP, ok_ex[0] if ok_ex else ok)
        if name == "bracket":
            sets["opt"] = np.array([[xc, yc, OPT_R] for xc in OPT_GRID[0] for yc in OPT_GRID[1]])
        mid = (np.array(lo) + np.array(hi)) / 2                         # roughness: 1-D sweeps through the centre
        for i in range(len(names)):
            sweep = np.repeat(mid[None], 101, 0)
            sweep[:, i] = np.linspace(lo[i], hi[i], 101)
            sets[f"sweep{i}"] = sweep
        jobs = [(name, x) for k in sets for x in sets[k]]
        with Pool(8) as pool:
            ys = pool.map(run_sim, jobs, chunksize=8)
        out, at = {}, 0
        for k, X in sets.items():
            out[f"X_{k}"], out[f"y_{k}"] = X, np.array(ys[at:at + len(X)])
            at += len(X)
        np.savez(f"{OUT}/sims_{name}.npz", **out)
        print(f"{name}: {len(jobs)} simulations in {time.time() - t0:.0f}s", flush=True)


# ---- surrogates -------------------------------------------------------------------------------------------------------
class GP:
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def _fit_one(self, X, t):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        d = X.shape[1]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(d), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(X, t)

    def _u(self, X):
        return (np.asarray(X, float) - self.lo) / (self.hi - self.lo)

    def fit(self, X, y):
        U, n = self._u(X), len(y)
        fold = np.random.default_rng(0).permutation(n) % min(5, n)
        best = (np.inf, None)
        for tr in ["identity"] + (["log"] if (y > 0).all() else []):
            t = np.log(y) if tr == "log" else y
            err = 0.0
            for f in range(fold.max() + 1):
                a, b = fold != f, fold == f
                p = self._fit_one(U[a], t[a]).predict(U[b])
                err += np.sum(((np.exp(p) if tr == "log" else p) - y[b]) ** 2)
            if err < best[0]:
                best = (err, tr)
        self.transform = best[1]
        self.model = self._fit_one(U, np.log(y) if self.transform == "log" else y)
        return self

    def predict(self, X):
        p = self.model.predict(self._u(X))
        return np.exp(p) if self.transform == "log" else p


class Quadratic:
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def _design(self, X):
        U = 2 * (np.asarray(X, float) - self.lo) / (self.hi - self.lo) - 1
        d = U.shape[1]
        cols = [np.ones(len(U))] + [U[:, i] for i in range(d)] + [U[:, i] * U[:, j] for i in range(d) for j in range(i, d)]
        return np.stack(cols, 1)

    def fit(self, X, y):
        A = self._design(X)
        self.coef = np.linalg.solve(A.T @ A + 1e-8 * np.eye(A.shape[1]), A.T @ y)
        return self

    def predict(self, X):
        return self._design(X) @ self.coef


class Hybrid:
    """symreg's formula as the trend, plus a Gaussian process on its residuals."""
    def __init__(self, names, lo, hi):
        self.sr, self.gp = SymbolicRegressor(names), GP(lo, hi)

    def fit(self, X, y):
        self.sr.fit(X, y)
        t = np.log(y) if self.sr.transform == "log" else y
        self.res = self.gp._fit_one(self.gp._u(X), t - self.sr.predict_t(X))
        return self

    def predict(self, X):
        t = self.sr.predict_t(X) + self.res.predict(self.gp._u(X))
        return np.exp(t) if self.sr.transform == "log" else t


class Augmented:
    """A surrogate given extra input columns computed from the inputs (the constraint margins)."""
    def __init__(self, base, fn):
        self.base, self.fn = base, fn

    def _x(self, X):
        return np.column_stack([X, self.fn(np.asarray(X, float))])

    def fit(self, X, y):
        self.base.fit(self._x(X), y)
        return self

    def predict(self, X):
        return self.base.predict(self._x(X))

    def formula(self):
        return self.base.formula()

    def n_terms(self):
        return self.base.n_terms()


def make(method, name):
    _, names, lo, hi, *_rest = PROBLEMS[name]
    if method == "symreg_margins":
        return Augmented(SymbolicRegressor(names + MARGINS[name][0]), MARGINS[name][1])
    if method == "gp_margins":
        mnames, mfn, mlo, mhi = MARGINS[name]
        return Augmented(GP(list(lo) + mlo, list(hi) + mhi), mfn)
    return {"symreg": lambda: SymbolicRegressor(names), "gp": lambda: GP(lo, hi), "quadratic": lambda: Quadratic(lo, hi),
            "hybrid": lambda: Hybrid(names, lo, hi)}[method]()


def nrmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)) / y.std()) if np.all(np.isfinite(pred)) else float("inf")


def fit():
    path = f"{OUT}/results.jsonl"
    have = {(r["problem"], r["n"], r["seed"], r["method"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    for name, (_, names, lo, hi, *_rest) in PROBLEMS.items():
        z = np.load(f"{OUT}/sims_{name}.npz")
        for n in BUDGETS:
            for seed in SEEDS:
                X, y = z[f"X_train{seed}"][:n], z[f"y_train{seed}"][:n]
                for method in METHODS:
                    if (name, n, seed, method) in have or (method.endswith("_margins") and name not in MARGINS):
                        continue
                    t0 = time.time()
                    m = make(method, name).fit(X, y)
                    rec = dict(problem=name, n=n, seed=seed, method=method, seconds=time.time() - t0,
                               nrmse_in=nrmse(m.predict(z["X_test"]), z["y_test"]),
                               nrmse_ex=nrmse(m.predict(z["X_extrap"]), z["y_extrap"]),
                               max_in=float(np.max(np.abs(m.predict(z["X_test"]) - z["y_test"])) / z["y_test"].std()))
                    if name == "bracket":
                        yo = z["y_opt"]
                        pick = int(np.argmin(m.predict(z["X_opt"])))
                        rec["regret"] = float((yo[pick] - yo.min()) / (yo.max() - yo.min()))
                    if method in ("symreg", "symreg_margins"):
                        rec["formula"], rec["terms"] = m.formula(), m.n_terms()
                    with open(path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{name:7s} n={n:3d} seed {seed} {method:9s}: NRMSE in {rec['nrmse_in']:.4f} ex {rec['nrmse_ex']:.4f}"
                          + (f" regret {rec['regret']:.3f}" if "regret" in rec else "")
                          + (f" terms {rec['terms']}" if "terms" in rec else "") + f"  {rec['seconds']:.1f}s", flush=True)


def report():
    rs = [json.loads(l) for l in open(f"{OUT}/results.jsonl")]
    for name, (_, names, lo, hi, *_rest) in PROBLEMS.items():
        z = np.load(f"{OUT}/sims_{name}.npz")
        floor = []
        for i in range(len(names)):
            x, y = z[f"X_sweep{i}"][:, i], z[f"y_sweep{i}"]
            u = (x - x.mean()) / x.std()
            floor.append(np.std(y - np.polyval(np.polyfit(u, y, 8), u)) / z["y_test"].std())
        print(f"\n{name}: simulator roughness (residual of a degree-8 fit along each 1-D sweep, / std y) "
              f"{', '.join(f'{f:.1e}' for f in floor)}")
        print(f"   {'n':>4s} {'method':10s} {'NRMSE in':>10s} {'NRMSE ex':>10s} {'max in':>8s}" + ("  regret" if name == "bracket" else "") + "  terms")
        for n in BUDGETS:
            for method in METHODS:
                R = [r for r in rs if r["problem"] == name and r["n"] == n and r["method"] == method]
                if not R:
                    continue
                mean = lambda k: np.mean([r[k] for r in R])
                line = f"   {n:4d} {method:10s} {mean('nrmse_in'):10.4f} {mean('nrmse_ex'):10.4f} {mean('max_in'):8.3f}"
                if name == "bracket":
                    line += f"  {mean('regret'):6.3f}"
                if "terms" in R[0]:
                    line += f"  {[r['terms'] for r in R]}"
                print(line)
        for method in ("symreg", "symreg_margins"):
            best = [r for r in rs if r["problem"] == name and r["n"] == 100 and r["method"] == method]
            if best:
                print(f"   {method} formula at n = 100 (seed {best[0]['seed']}): {best[0]['formula']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("simulate", "fit", "report"))
    a = p.parse_args()
    {"simulate": simulate, "fit": fit, "report": report}[a.stage]()


if __name__ == "__main__":
    main()
