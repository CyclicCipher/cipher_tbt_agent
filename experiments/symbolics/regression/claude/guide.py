"""The guide (docs/CURRICULUM.md): which terms to try first, from the data alone. No neural network.

G1 deduction       the residual for sums is the finder's core; log y for products, when G2 finds separability
G2 statistics      additivity: a first-order functional-ANOVA fit (a degree-4 polynomial in each input, summed, least
                   squares) on y and on log y; R^2 >= ADDITIVE splits the task into 1-D fits, one per input, assembled
                   and then refined jointly. Exponent estimates p = 1 + median slope of log|f_i'| against log x_i.
                   Exchangeable inputs: inputs whose exponent estimates agree within 0.3 form a group (amended
                   2026-10-26 before any held-out run: comparing the 1-D fits' term syntax missed equivalent forms,
                   e.g. x^-3 and (x + 0.0004)^-3).
                   Separability: the size of the cross terms of a full quadratic fit of log y, relative to the rest.
G2 pairs (B3)      a second-order sensitivity fit in the chosen transform: t = b0 + sum_i P4(z_i) + sum_{i<j}
                   (c1 z_i z_j + c2 z_i^2 z_j + c3 z_i z_j^2), least squares with a 1e-6 ridge; a pair's share is the
                   variance of its fitted interaction part over the variance of t; detected: share >= 1/4 of the largest
                   and >= 1e-4. Input names are never used.
G3 case memory     signature = (inputs, the two additivity R^2, sorted exponent estimates, separability); the 3 nearest
                   solved tasks (training tasks and fantasies) multiply the prior of the templates they used by 3.
"""
import itertools

import numpy as np

ADDITIVE, MIN_SHARE, GROUP_TOL, BOOST, K_NEAR, MAX_D = 0.98, 0.01, 0.3, 3.0, 3, 6


def scaled(X, lo, hi):
    return 2 * (X - lo) / (hi - lo) - 1


def additive_fit(X, t, lo, hi, degree=4):
    """First-order additive fit: t = b0 + sum_i sum_k b_ik z_i^k. Returns (R^2, coefficients [d, degree], b0)."""
    Z = scaled(X, lo, hi)
    n, d = Z.shape
    A = np.column_stack([np.ones(n)] + [Z[:, i] ** k for i in range(d) for k in range(1, degree + 1)])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    r = t - A @ coef
    r2 = 1 - (r @ r) / max(np.sum((t - t.mean()) ** 2), 1e-300)
    return float(r2), coef[1:].reshape(d, degree), float(coef[0])


def component(B, i, z):
    return sum(B[i, k - 1] * z ** k for k in range(1, B.shape[1] + 1))


def exponent_estimates(B, lo, hi, total_var):
    """p_i = 1 + median d log|f_i'| / d log x_i over the range, for inputs whose component carries >= MIN_SHARE of the
    variance (else None)."""
    out = {}
    for i in range(B.shape[0]):
        x = np.linspace(lo[i], hi[i], 60)
        z = scaled(x, lo[i], hi[i])
        f = component(B, i, z)
        if np.var(f) < MIN_SHARE * total_var:
            continue
        dz = 2 / (hi[i] - lo[i])
        fp = sum(k * B[i, k - 1] * z ** (k - 1) for k in range(1, B.shape[1] + 1)) * dz
        ok = np.abs(fp) > 1e-12 * np.max(np.abs(fp))
        if ok.sum() < 10:
            continue
        lx, lf = np.log(x[ok]), np.log(np.abs(fp[ok]))
        out[i] = float(1 + np.median(np.diff(lf) / np.diff(lx)))
    return out


def pair_shares(X, t, lo, hi):
    """{(i, j): share of the variance of t in the pair's fitted interaction part}."""
    Z = scaled(np.asarray(X, float), lo, hi)
    n, d = Z.shape
    pairs = list(itertools.combinations(range(d), 2))
    if not pairs:
        return {}
    first = [np.ones(n)] + [Z[:, i] ** k for i in range(d) for k in range(1, 5)]
    blocks = [np.column_stack([Z[:, i] * Z[:, j], Z[:, i] ** 2 * Z[:, j], Z[:, i] * Z[:, j] ** 2]) for i, j in pairs]
    A = np.column_stack(first + blocks)
    G = A.T @ A
    coef = np.linalg.solve(G + 1e-6 * np.mean(np.diag(G)) * np.eye(len(G)), A.T @ t)
    at, var, out = len(first), max(np.var(t), 1e-300), {}
    for (i, j), B in zip(pairs, blocks):
        out[(i, j)] = float(np.var(B @ coef[at:at + 3]) / var)
        at += 3
    return out


def detect_pairs(shares):
    if not shares:
        return []
    top = max(shares.values())
    return sorted(p for p, v in shares.items() if v >= 0.25 * top and v >= 1e-4)


def separability(X, y, lo, hi):
    """Cross-term share of a full quadratic fit of log y: near 0 for y = g(x_1) h(x_2) ..., larger otherwise."""
    if X.shape[1] < 2 or np.any(y <= 0):
        return 0.0
    Z = scaled(X, lo, hi)
    n, d = Z.shape
    pairs = list(itertools.combinations(range(d), 2))
    A = np.column_stack([np.ones(n)] + [Z[:, i] for i in range(d)] + [Z[:, i] ** 2 for i in range(d)] +
                        [Z[:, i] * Z[:, j] for i, j in pairs])
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    cross = np.abs(coef[1 + 2 * d:]).mean()
    rest = np.abs(coef[1:1 + 2 * d]).mean()
    return float(cross / max(cross + rest, 1e-300))


class Guide:
    """Statistics for one task, and the decomposition into 1-D fits when the task is additive."""
    def __init__(self, X, y, lo, hi):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.r2 = {"identity": additive_fit(X, y, self.lo, self.hi)[0]}
        if np.all(y > 0):
            self.r2["log"] = additive_fit(X, np.log(y), self.lo, self.hi)[0]
        self.transform = max(self.r2, key=self.r2.get)
        t = np.log(y) if self.transform == "log" else y
        _, self.B, self.b0 = additive_fit(X, t, self.lo, self.hi)
        self.exponents = exponent_estimates(self.B, self.lo, self.hi, np.var(t))
        self.sep = separability(X, y, self.lo, self.hi)
        self.additive = self.r2[self.transform] >= ADDITIVE
        self.groups = exchangeable(self.exponents)
        self.shares = pair_shares(X, t, self.lo, self.hi)
        self.pairs = detect_pairs(self.shares)

    def signature(self, d):
        e = sorted(self.exponents.values()) + [0.0] * MAX_D
        return np.array([d, self.r2.get("identity", 0.0), self.r2.get("log", 0.0)] + e[:MAX_D] + [self.sep])

    def hints(self):
        return {"exponents": {i: [p] for i, p in self.exponents.items()}, "groups": [list(g) for g in self.groups],
                "pairs": [tuple(p) for p in self.pairs]}

    def decompose(self, X, y, make_finder, margins=None, domain=None):
        """Fit each input's component with a 1-D finder on its partial residual (backfitting), and return the
        assembled formula as a start state (blocks re-bound to the full input indices), and the 1-D fits."""
        from regression.claude.symreg2 import Block, State
        X, y = np.asarray(X, float), np.asarray(y, float)
        t = np.log(y) if self.transform == "log" else y
        Z = scaled(X, self.lo, self.hi)
        parts = {i: component(self.B, i, Z[:, i]) for i in range(X.shape[1])}
        blocks, fits = [], {}
        total = np.var(t)
        for i in range(X.shape[1]):
            if np.var(parts[i]) < MIN_SHARE * total:
                continue
            r = t - self.b0 - sum(parts[j] for j in parts if j != i)
            yi = np.exp(r) if self.transform == "log" else r
            f = make_finder(i).fit(X[:, [i]], yi, None, {"X": domain["X"][:, [i]], "M": np.zeros((len(domain["X"]), 0))},
                                   hints={"exponents": {0: [self.exponents[i]]} if i in self.exponents else {}},
                                   transforms=[self.transform])
            fits[i] = f
            for b in f.best["state"].blocks:
                blocks.append(Block(b.factors, [tuple(i for _ in bind) for bind in b.bindings], b.theta, b.free, b.template))
        return State(blocks, []), fits


def exchangeable(exponents):
    """Groups of inputs whose exponent estimates agree within GROUP_TOL (greedy, in order of the estimate)."""
    groups, cur = [], []
    for i, p in sorted(exponents.items(), key=lambda kv: kv[1]):
        if cur and p - exponents[cur[0]] > GROUP_TOL:
            if len(cur) >= 2:
                groups.append(sorted(cur))
            cur = []
        cur.append(i)
    if len(cur) >= 2:
        groups.append(sorted(cur))
    return groups


class CaseMemory:
    """G3: solved tasks (and fantasies) by signature; a lookup boosts the templates the nearest ones used."""
    def __init__(self):
        self.sigs, self.used = [], []

    def add(self, signature, templates):
        self.sigs.append(np.asarray(signature, float))
        self.used.append(set(templates))

    def boosts(self, signature):
        if not self.sigs:
            return {}
        S = np.stack(self.sigs)
        mu, sd = S.mean(0), S.std(0) + 1e-9
        dist = np.linalg.norm((S - mu) / sd - (np.asarray(signature) - mu) / sd, axis=1)
        out = {}
        for k in np.argsort(dist)[:K_NEAR]:
            for name in self.used[k]:
                out[name] = BOOST
        return out
