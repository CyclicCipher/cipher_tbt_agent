"""Symbolic regression by greedy residual term selection (docs/SURROGATES.md): Gemini's "symbolic ResNet" done
properly.

The model is t(y) = P(x) / (1 + D(x)), with t = y or log y, P a sum of terms and D a sum of denominator terms. Fitting
it as t = P - t * D is linear, so every term, numerator or denominator, is one column of a library:
    powers          prod_i x_i^a_i, a_i in {-2, -1, -1/2, 0, 1/2, 1, 2, 3, 4}, sum |a_i| <= 4 (inputs must be positive);
                    with 4 or more inputs, at most 2 inputs per term (the full set grows as 9^d: 19,188 terms at d = 6)
    logs            log x_i
    denominators    -t * prod_i x_i^b_i, b_i >= 0 integers, 1 <= sum b_i <= 2
Selection grows the formula one term at a time on the residual (the residual stream), refitting every coefficient
after each addition. Each step adds the column that most reduces the squared error after the refit (orthogonal
least squares), and a beam keeps the 4 best formulas of each size, so an early wrong choice can be outgrown (the
lookahead lesson of the subset-sum experiment). K-fold cross-validation picks t and the size, by the
one-standard-error rule (the smallest formula within one standard error of the best). A formula whose denominator
comes within 0.05 of zero anywhere in the sampled box (1,000 uniform points over the training range) has a pole
there and is rejected.
Relatives: FFX (McConaghy 2011), SINDy (Brunton et al. 2016), orthogonal least squares (Chen et al. 1989).
"""
import itertools

import numpy as np

EXPONENTS = (-2, -1, -0.5, 0, 0.5, 1, 2, 3, 4)
BEAM = 4


class Library:
    def __init__(self, names):
        self.names = list(names)
        d = len(self.names)
        max_vars = d if d <= 3 else 2
        self.pow = [a for a in itertools.product(EXPONENTS, repeat=d)
                    if 0 < sum(map(abs, a)) <= 4 and sum(x != 0 for x in a) <= max_vars]
        self.den = [b for b in itertools.product(range(3), repeat=d) if 1 <= sum(b) <= 2]

    def numerator(self, X):
        cols = [np.prod(X ** np.array(a, float), 1) for a in self.pow] + [np.log(X[:, i]) for i in range(X.shape[1])]
        return np.stack(cols, 1)

    def denominator(self, X):
        return np.stack([np.prod(X ** np.array(b, float), 1) for b in self.den], 1)

    def labels(self):
        def mono(exps):
            return "*".join(s if e == 1 else f"{s}^{e:g}" for e, s in zip(exps, self.names) if e != 0)
        return [mono(a) for a in self.pow] + [f"log({n})" for n in self.names], [mono(b) for b in self.den]


def ols_beam_path(A, t, k_max, beam=BEAM, valid=None):
    """For each size k = 1..k_max, the support (column indices besides the constant) with the lowest training error
    found by a beam of forward orthogonal-least-squares steps. valid(support) -> bool drops supports (poles)."""
    n = len(t)
    scale = np.linalg.norm(A - A.mean(0), axis=0)
    usable = scale > 1e-12 * max(scale.max(), 1e-300)
    frontier, best = [()], []
    for _ in range(k_max):
        cand = {}
        for s in frontier:
            Q, _ = np.linalg.qr(np.column_stack([np.ones(n)] + [A[:, j] for j in s]))
            r = t - Q @ (Q.T @ t)
            Ap = A - Q @ (Q.T @ A)
            norm2 = np.einsum("ij,ij->j", Ap, Ap)
            gain = np.where(usable & (norm2 > 1e-12 * np.maximum(scale ** 2, 1e-300)), (Ap.T @ r) ** 2 / np.maximum(norm2, 1e-300), -1)
            gain[list(s)] = -1
            base = r @ r
            for j in np.argsort(-gain)[:beam]:
                if gain[j] > 0:
                    key = tuple(sorted(s + (int(j),)))
                    cand[key] = min(cand.get(key, np.inf), base - gain[j])
        ranked = [c for c in sorted(cand, key=cand.get) if valid is None or valid(list(c))]
        if not ranked:
            break
        frontier = ranked[:beam]
        best.append(list(frontier[0]))
    return best


class SymbolicRegressor:
    def __init__(self, names, k_max=8):
        self.lib, self.k_max = Library(names), k_max

    def _columns(self, X, t):
        num = self.lib.numerator(X)
        return np.column_stack([num, -t[:, None] * self.lib.denominator(X)]), num.shape[1]

    def _coef(self, X, t, support, A=None):
        A = self._columns(X, t)[0] if A is None else A
        design = np.column_stack([np.ones(len(t))] + [A[:, j] for j in support])
        return np.linalg.lstsq(design, t, rcond=None)[0]

    def _polynomial(self, support):
        return all(j < self.n_num_lib for j in support)

    def _has_pole(self, support, coef):
        D = 1 + sum(c * self.box_den[:, j - self.n_num_lib] for c, j in zip(coef[1:], support) if j >= self.n_num_lib)
        return np.min(D) <= 0.05

    def _predict(self, X, transform, support, coef):
        num, den = self.lib.numerator(X), self.lib.denominator(X)
        m = num.shape[1]
        P = coef[0] + sum((c * num[:, j] for c, j in zip(coef[1:], support) if j < m), np.zeros(len(X)))
        D = sum((c * den[:, j - m] for c, j in zip(coef[1:], support) if j >= m), np.zeros(len(X)))
        with np.errstate(all="ignore"):
            t = P / (1 + D)
            return np.exp(np.clip(t, -50, 50)) if transform == "log" else t

    def fit(self, X, y, folds=10, seed=0):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n = len(y)
        rng = np.random.default_rng(seed + 1)
        self.box = rng.uniform(X.min(0), X.max(0), (1000, X.shape[1]))
        self.n_num_lib = self.lib.numerator(X[:1]).shape[1]
        self.box_den = self.lib.denominator(self.box)
        k_max = max(1, min(self.k_max, n // 3))
        fold = np.random.default_rng(seed).permutation(n) % min(folds, n)
        n_folds = fold.max() + 1
        results = []                                          # (mean CV error, its standard error, transform, k)
        for tr in ["identity"] + (["log"] if (y > 0).all() else []):
            t = np.log(y) if tr == "log" else y
            err = np.full((n_folds, k_max + 1), np.inf)
            for f in range(n_folds):
                a, b = fold != f, fold == f
                A, _ = self._columns(X[a], t[a])
                valid = lambda s: self._polynomial(s) or not self._has_pole(s, self._coef(X[a], t[a], s, A))
                path = [[]] + ols_beam_path(A, t[a], k_max, valid=valid)
                for k in range(k_max + 1):
                    s = path[min(k, len(path) - 1)]
                    pred = self._predict(X[b], tr, s, self._coef(X[a], t[a], s, A))
                    if np.all(np.isfinite(pred)):
                        err[f, k] = np.mean((pred - y[b]) ** 2)
            for k in range(k_max + 1):
                results.append((err[:, k].mean(), err[:, k].std() / np.sqrt(n_folds), tr, k))
        finite = [r for r in results if np.isfinite(r[0])]
        best = min(finite, key=lambda r: r[0])
        chosen = min((r for r in finite if r[0] <= best[0] + best[1]), key=lambda r: (r[3], r[0]))
        _, _, self.transform, k = chosen
        t = np.log(y) if self.transform == "log" else y
        A, self.n_num = self._columns(X, t)
        path = [[]] + ols_beam_path(A, t, k, valid=lambda s: self._polynomial(s) or not self._has_pole(s, self._coef(X, t, s, A)))
        self.support = path[min(k, len(path) - 1)]
        self.coef = self._coef(X, t, self.support, A)
        return self

    def predict(self, X):
        return self._predict(np.asarray(X, float), self.transform, self.support, self.coef)

    def predict_t(self, X):
        """The prediction in the fitted space (y or log y)."""
        y = self.predict(X)
        return np.log(np.maximum(y, 1e-300)) if self.transform == "log" else y

    def formula(self):
        num_labels, den_labels = self.lib.labels()
        num = [f"{self.coef[0]:.4g}"] + [f"{c:+.4g}*{num_labels[j]}" for c, j in zip(self.coef[1:], self.support)
                                          if j < self.n_num]
        den = [f"{c:+.4g}*{den_labels[j - self.n_num]}" for c, j in zip(self.coef[1:], self.support) if j >= self.n_num]
        body = " ".join(num) if not den else f"({' '.join(num)}) / (1 {' '.join(den)})"
        return f"log y = {body}" if self.transform == "log" else f"y = {body}"

    def n_terms(self):
        return 1 + len(self.support)
