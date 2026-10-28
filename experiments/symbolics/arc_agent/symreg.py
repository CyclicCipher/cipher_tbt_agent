"""The function finder, version 2 (legacy/docs/CURRICULUM.md, the experiments that built it; a copy for the agent): terms with continuous parameters, safety over the design
domain, selection by BIC with a library prior, and a frontier of near-equal formulas as an ensemble.

A formula is t = (c0 + sum_k B_k(x; theta_k) . c_k) / (1 + D(x)), with t = y or log y. A block B_k is a product of
factors and gives one column, or one column per input of a group (shared parameters, one coefficient each).
Factor kinds and their parameters:
    pow     x_i^p                        p
    shift   (x_i + b)^p                  beta, p      (b = -min x_i + range * (0.1 + softplus(beta)): the base stays
                                                       positive on the box, and a singularity sits at least 10% of the
                                                       range below it, so modest extrapolation can't reach it)
    margin  (s_j + b)^p, b > 0           beta, p      (s_j a constraint margin, >= 0 on the domain)
    log     log x_i                      -
    exp     exp(k z_i), z_i in [-1, 1]   k
    den     -t * prod x_i^m (integers, degree 1-2): a denominator column (no parameters)
Search (polynomial): grow one block at a time on the residual. Each step screens every candidate on a parameter grid
by the exact drop in squared error after a linear refit, and refines the best REFINE candidates' own parameters by
bounded least squares (variable projection: the linear coefficients are solved exactly inside; at most LM_EVALS
residual evaluations, finite-difference Jacobians included). The BEAM best formulas of each size then get a joint
refit of every free parameter, under the same budget.
Safety: every block is evaluated on the task's domain sample (feasible designs, with boundary points) and rejected
if non-finite or if its largest magnitude exceeds 1e4 x its median; a denominator must stay above 0.05 there.
Selection: BIC on the y scale, n log(RSS/n) + parameters * log n (+ 2 sum log y for a log-y formula)
- 2 sum log P(block template); or, given the noise level sigma (experiment B2), BIC with sigma known:
sum (y_hat - y)^2 / sigma^2 + parameters * log n - 2 sum log P, residuals on the y scale for both transforms.
Frontier: every formula within 6 BIC of the best, at most 8, weighted exp(-BIC/2).

Version 3 (experiment B3, docs/VOCABULARY.md), each behind a flag:
    outer        composition by inversion, depth 2: y = f(T(x)); the inner sum T is searched on t = f^-1(y). The family:
                 y; log(y - b); (y - b)^q, q in {-1, -1/2, 1/2, 2}; b in {0, min y - 0.05 range y}. Every member is
                 scored by how well a short sum fits its target: a cheap 3-term greedy screen (grid starts, no
                 refinement), measured on the y scale; the top 3, plus y and log y, are searched. (Amended before any
                 run: first-order additivity, as first specified, can't discriminate in 1-D and penalizes products.)
    affine       atoms over a fitted combination of inputs: (a.u + b)^p and log(a.u + b), u the inputs scaled to the
                 box, a a direction, b keeping the base positive with any singularity 10% of its range outside.
    relational   one term x_i^p x_j^q summed over a set of input pairs (shared p, q; a coefficient per pair), over all
                 pairs and over the pairs the guide detects as interacting.
"""
import itertools
import math

import numpy as np
from scipy.optimize import least_squares

GRID = (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
EXP_GRID = (-2.0, -1.0, 1.0, 2.0)
BETA_GRID = (-2.0, 0.0)
BEAM, K_MAX, REFINE, LM_EVALS, FRONTIER, FRONTIER_BIC = 4, 8, 8, 50, 8, 6.0
K_CONTINUE = 4                     # blocks the search may add after a start formula (the guide's assembled 1-D fits)
PRIMITIVES = ("pow", "shift", "log", "exp", "pow*pow", "margin", "den", "affine", "alog", "pairgroup")
AFFINE_P = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


# ---- outer functions (B3) --------------------------------------------------------------------------------------------
def as_transform(tr):
    return ("identity",) if tr == "identity" else ("log", 0.0) if tr == "log" else tuple(tr)


def t_forward(tr, y):
    tr = as_transform(tr)
    with np.errstate(all="ignore"):
        if tr[0] == "identity":
            return y
        if tr[0] == "log":
            return np.log(y - tr[1])
        return (y - tr[2]) ** tr[1]


def t_inverse(tr, t):
    tr = as_transform(tr)
    with np.errstate(all="ignore"):
        if tr[0] == "identity":
            return t
        if tr[0] == "log":
            return np.exp(np.clip(t, -50, 50)) + tr[1]
        return np.power(t, 1.0 / tr[1]) + tr[2]


def t_valid(tr, y):
    tr = as_transform(tr)
    if tr[0] == "identity":
        return True
    b = tr[1] if tr[0] == "log" else tr[2]
    return bool(np.all(y - b > 0))


def t_label(tr):
    tr = as_transform(tr)
    if tr[0] == "identity":
        return "y"
    b = tr[1] if tr[0] == "log" else tr[2]
    yb = "y" if b == 0 else f"(y{-b:+.4g})"
    return f"log {yb}" if tr[0] == "log" else f"{yb}^{tr[1]:g}"


def outer_family(y):
    bs = [0.0, float(np.min(y) - 0.05 * (np.max(y) - np.min(y)))]
    fam = [("identity",)] + [("log", b) for b in bs] + [("pow", q, b) for q in (-1.0, -0.5, 0.5, 2.0) for b in bs]
    return [tr for tr in fam if t_valid(tr, y)]
N_PARAMS = {"pow": 1, "shift": 2, "margin": 2, "log": 0, "exp": 1}
P_BOUND, P_MIN = 6.0, 0.25


def softplus(v):
    return np.log1p(np.exp(-abs(v))) + max(v, 0.0)


class Context:
    """The data a formula is evaluated on: inputs, margins, and the per-input offsets that keep shifted bases positive."""
    def __init__(self, X, margins, domain):
        self.lo, self.hi = domain["X"].min(0), domain["X"].max(0)
        self.X, self.M = X, margins
        self.dom_X, self.dom_M = domain["X"], domain["M"]
        self.mlo = domain["M"].min(0) if domain["M"].size else np.zeros(0)
        self.mrange = (domain["M"].max(0) - self.mlo) if domain["M"].size else np.zeros(0)


def shift_offset(ctx, var, beta):
    return -ctx.lo[var] + (ctx.hi[var] - ctx.lo[var]) * (0.1 + softplus(beta))


def margin_offset(ctx, var, beta):
    # b > 0 from the constraint boundary s = 0, so every feasible design (s >= 0) has a positive base (fixed
    # 2026-10-26: it was measured from the training domain's smallest margin, which extrapolation can undercut)
    return ctx.mrange[var] * (1e-3 + softplus(beta))


def factor_value(kind, var, params, X, M, ctx):
    if kind == "pow":
        return X[:, var] ** params[0]
    if kind == "shift":
        rng = ctx.hi[var] - ctx.lo[var]
        return (X[:, var] + shift_offset(ctx, var, params[0])) ** params[1]
    if kind == "margin":
        return (M[:, var] + margin_offset(ctx, var, params[0])) ** params[1]
    if kind == "log":
        return np.log(X[:, var])
    if kind == "exp":
        z = 2 * (X[:, var] - ctx.lo[var]) / (ctx.hi[var] - ctx.lo[var]) - 1
        return np.exp(params[0] * z)
    raise ValueError(kind)


class Block:
    """A product of factors. factors: [(kind, slot)], slots index into each binding; bindings: [tuple of input or
    margin indices], one column per binding (one binding = an ordinary term, several = a group). theta: all factor
    parameters; free: which are fitted. template: the library entry it came from (for the prior)."""
    def __init__(self, factors, bindings, theta, free, template):
        self.factors, self.bindings = list(factors), [tuple(b) for b in bindings]
        self.theta, self.free, self.template = np.asarray(theta, float), np.asarray(free, bool), template

    def copy(self):
        return Block(self.factors, self.bindings, self.theta.copy(), self.free.copy(), self.template)

    def columns(self, X, M, ctx, theta=None):
        theta = self.theta if theta is None else theta
        cols = []
        for bind in self.bindings:
            v, at = np.ones(len(X)), 0
            for kind, slot in self.factors:
                k = N_PARAMS[kind]
                v = v * factor_value(kind, bind[slot], theta[at:at + k], X, M, ctx)
                at += k
            cols.append(v)
        return np.stack(cols, 1)

    def n_free(self):
        return int(self.free.sum())

    def beta_mask(self):
        roles = []
        for kind, _ in self.factors:
            roles += (["beta", "p"] if kind in ("shift", "margin") else ["p"] * N_PARAMS[kind])
        return np.array([r == "beta" for r in roles], bool)

    def exponent_mask(self):
        roles = []
        for kind, _ in self.factors:
            roles += {"pow": [True], "shift": [False, True], "margin": [False, True], "log": [], "exp": [True]}[kind]
        return np.array(roles, bool)

    def key(self):
        return (tuple(self.factors), tuple(self.bindings), tuple(np.round(self.theta[~self.free], 3)))

    def safe(self, ctx):
        with np.errstate(all="ignore"):
            C = self.columns(ctx.dom_X, ctx.dom_M, ctx)
        if not np.all(np.isfinite(C)):
            return False
        A = np.abs(C)
        return bool(np.all(A.max(0) <= 1e4 * np.maximum(np.median(A, 0), 1e-300)))

    def label(self, names, mnames, ctx=None):
        parts, at = [], 0
        for kind, slot in self.factors:
            k = N_PARAMS[kind]
            p = self.theta[at:at + k]
            at += k
            v = "{" + ",".join(names[b[slot]] if kind != "margin" else mnames[b[slot]] for b in self.bindings) + "}" \
                if len(self.bindings) > 1 else (mnames[self.bindings[0][slot]] if kind == "margin" else names[self.bindings[0][slot]])
            off = lambda f: f(ctx, self.bindings[0][slot], p[0]) if ctx is not None else float("nan")
            parts.append({"pow": lambda: f"{v}^{p[0]:.3g}", "shift": lambda: f"({v}{off(shift_offset):+.3g})^{p[1]:.3g}",
                          "margin": lambda: f"({v}{off(margin_offset):+.3g})^{p[1]:.3g}", "log": lambda: f"log({v})",
                          "exp": lambda: f"exp({p[0]:.3g} z_{v})"}[kind]())
        return ("sum over " if len(self.bindings) > 1 else "") + "*".join(parts)


class AffineBlock(Block):
    """(a.u + b)^p or log(a.u + b) over all inputs (B3): u = inputs scaled to [0, 1] over the box, a a direction
    (normalized inside), b = -min(a.u) + (0.1 + softplus(beta)) range(a.u) over the domain. Params: a (d), beta, p."""
    def __init__(self, kind, d, theta, free=None, template=None):
        k = d + (2 if kind == "affine" else 1)
        super().__init__([(kind, 0)], [(0,)], theta, np.ones(k, bool) if free is None else free, template or kind)
        self.kind, self.d = kind, d

    def copy(self):
        return AffineBlock(self.kind, self.d, self.theta.copy(), self.free.copy(), self.template)

    def base(self, X, ctx, theta):
        a = np.asarray(theta[:self.d], float)
        nrm = np.linalg.norm(a)
        if nrm < 1e-9:
            return None
        a = a / nrm
        span = np.where(ctx.hi > ctx.lo, ctx.hi - ctx.lo, 1.0)
        s, sd = ((X - ctx.lo) / span) @ a, ((ctx.dom_X - ctx.lo) / span) @ a
        rng = max(float(sd.max() - sd.min()), 1e-9)
        return s - sd.min() + (0.1 + softplus(theta[self.d])) * rng

    def columns(self, X, M, ctx, theta=None):
        theta = self.theta if theta is None else theta
        bs = self.base(X, ctx, theta)
        if bs is None:
            return np.full((len(X), 1), np.nan)
        with np.errstate(all="ignore"):
            v = bs ** theta[self.d + 1] if self.kind == "affine" else np.log(bs)
        return v[:, None]

    def beta_mask(self):
        m = np.zeros(len(self.theta), bool)
        m[self.d] = True
        return m

    def exponent_mask(self):
        m = np.zeros(len(self.theta), bool)
        if self.kind == "affine":
            m[self.d + 1] = True
        return m

    def label(self, names, mnames, ctx=None):
        a = self.theta[:self.d] / max(np.linalg.norm(self.theta[:self.d]), 1e-9)
        comb = "".join(f"{c:+.2g}*u_{n}" for c, n in zip(a, names) if abs(c) > 0.05)
        return f"({comb}+b)^{self.theta[self.d + 1]:.3g}" if self.kind == "affine" else f"log({comb}+b)"


class State:
    """A formula: blocks, denominator monomials, fitted coefficients."""
    def __init__(self, blocks, dens):
        self.blocks, self.dens = blocks, list(dens)

    def copy(self):
        return State([b.copy() for b in self.blocks], self.dens)

    def key(self):
        return (tuple(sorted(b.key() for b in self.blocks)), tuple(sorted(self.dens)))


def den_cols(dens, X, t):
    return np.stack([-t * np.prod(X ** np.array(m, float), 1) for m in dens], 1) if dens else np.zeros((len(X), 0))


class Finder:
    def __init__(self, names, mnames=(), library=None, prior=None, k_max=K_MAX, outer=False, affine=False,
                 relational=False):
        """library: list of template specs (see library.py); None = primitives only. prior: {primitive or template name:
        weight}, default 1 each; P(name) = weight / total."""
        self.names, self.mnames, self.k_max = list(names), list(mnames), k_max
        self.library = library or []
        self.prior = dict(prior or {})
        self.outer, self.affine, self.relational = outer, affine, relational

    # ---- candidates ---------------------------------------------------------------------------------------------------
    def primitive_candidates(self, d, n_margins, hints):
        """(Block with initial theta, list of initial thetas for its free parameters)."""
        out = []
        extra = hints.get("exponents", {})
        for i in range(d):
            grid = sorted(set(GRID) | {round(float(e), 2) for e in extra.get(i, [])})
            out.append((Block([("pow", 0)], [(i,)], [1.0], [True], "pow"), [[p] for p in grid]))
            out.append((Block([("shift", 0)], [(i,)], [0.0, 1.0], [True, True], "shift"),
                        [[bb, p] for bb in BETA_GRID for p in GRID if p < 0]))
            out.append((Block([("log", 0)], [(i,)], [], [], "log"), [[]]))
            out.append((Block([("exp", 0)], [(i,)], [1.0], [True], "exp"), [[k] for k in EXP_GRID]))
        for i, j in itertools.combinations(range(d), 2):
            out.append((Block([("pow", 0), ("pow", 1)], [(i, j)], [1.0, 1.0], [True, True], "pow*pow"),
                        [[p, q] for p in GRID for q in GRID if abs(p) + abs(q) <= 6]))
        for j in range(n_margins):
            out.append((Block([("margin", 0)], [(j,)], [0.0, -1.0], [True, True], "margin"),
                        [[bb, p] for bb in BETA_GRID for p in GRID if p < 0]))
        if self.affine and d >= 2:
            dirs = [np.eye(d)[i] + s * np.eye(d)[j] for i, j in itertools.combinations(range(d), 2) for s in (1, -1)]
            dirs.append(np.ones(d))
            for kind in ("affine", "alog"):
                inits = [list(a) + [bb] + ([p] if kind == "affine" else []) for a in dirs for bb in BETA_GRID
                         for p in (AFFINE_P if kind == "affine" else [None])]
                out.append((AffineBlock(kind, d, inits[0]), inits))
        if self.relational and d >= 2:
            pair_sets = [list(itertools.combinations(range(d), 2))]
            det = [tuple(p) for p in hints.get("pairs", [])]
            if len(det) >= 2 and det != pair_sets[0]:
                pair_sets.append(det)
            for ps in pair_sets:
                if len(ps) >= 2:
                    out.append((Block([("pow", 0), ("pow", 1)], ps, [1.0, 1.0], [True, True], "pairgroup"),
                                [[p, q] for p in GRID for q in GRID if abs(p) + abs(q) <= 6]))
        return out

    def library_candidates(self, d, hints):
        """Every library template, bound to every ordered choice of distinct inputs for its slots, or, for a group
        template, to each group the guide found and to all inputs. Free parameters start on their grids."""
        out = []
        for tpl in self.library:
            s, free = tpl["slots"], np.asarray(tpl["free"], bool)
            probe = Block(tpl["factors"], [tuple(range(s))], tpl["theta"], free, tpl["name"])
            beta = probe.beta_mask()
            rate = np.array([kind == "exp" for kind, _ in probe.factors for _ in range(N_PARAMS[kind])], bool)
            grids = [BETA_GRID if beta[k] else EXP_GRID if rate[k] else GRID for k in np.nonzero(free)[0]]
            inits = [list(p) for p in itertools.product(*grids)][:64] if free.any() else [[]]
            if tpl.get("group") and s == 2:                      # a relational template: sets of input pairs
                sets = [[tuple(p) for p in hints.get("pairs", [])], list(itertools.combinations(range(d), 2))]
                bindings_list = [ps for ps in sets if len(ps) >= 2]
            elif tpl.get("group"):
                groups = [list(g) for g in hints.get("groups", []) if len(g) >= 2] + ([list(range(d))] if d >= 2 else [])
                bindings_list = [[(v,) for v in g] for g in groups]
            else:
                bindings_list = [[b] for b in itertools.permutations(range(d), s)] if s <= d else []
            for bindings in bindings_list:
                out.append((Block(tpl["factors"], bindings, tpl["theta"], free, tpl["name"]), inits))
        return out

    def den_candidates(self, d):
        return [m for m in itertools.product(range(3), repeat=d) if 1 <= sum(m) <= 2]

    # ---- fitting ------------------------------------------------------------------------------------------------------
    def design(self, state, X, M, t, ctx, thetas=None):
        cols = [np.ones((len(X), 1))]
        for k, b in enumerate(state.blocks):
            cols.append(b.columns(X, M, ctx, None if thetas is None else thetas[k]))
        cols.append(den_cols(state.dens, X, t))
        return np.column_stack(cols)

    def fit_linear(self, state, X, M, t, ctx, thetas=None):
        with np.errstate(all="ignore"):
            A = self.design(state, X, M, t, ctx, thetas)
        if not np.all(np.isfinite(A)):
            return None, np.inf
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        r = t - A @ coef
        return coef, float(r @ r)

    def refit_joint(self, state, X, M, t, ctx, only_last=False):
        """Refine the free parameters (all blocks, or only the last) by variable projection, bounded, at most
        LM_EVALS residual evaluations including the finite-difference Jacobian's."""
        free = [b for b in (state.blocks[-1:] if only_last else state.blocks) if b.n_free()]
        if not free:
            return self.fit_linear(state, X, M, t, ctx)
        x0 = np.concatenate([b.theta[b.free] for b in free])
        beta = np.concatenate([b.beta_mask()[b.free] for b in free])
        expo = np.concatenate([b.exponent_mask()[b.free] for b in free])
        lo = np.where(beta, -10.0, -P_BOUND)
        hi = np.where(beta, 5.0, P_BOUND)
        # an exponent or rate keeps its sign and stays >= P_MIN from zero (c x^eps fakes log x, c exp(eps z) fakes z)
        lo = np.where(expo & (x0 > 0), P_MIN, lo)
        hi = np.where(expo & (x0 <= 0), -P_MIN, hi)
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)

        def unpack(z):
            out, at = [], 0
            for b in state.blocks:
                th = b.theta.copy()
                if b.n_free() and any(b is f for f in free):
                    th[b.free] = z[at:at + b.n_free()]
                    at += b.n_free()
                out.append(th)
            return out

        def resid(z):
            th = unpack(z)
            with np.errstate(all="ignore"):
                A = self.design(state, X, M, t, ctx, th)
            if not np.all(np.isfinite(A)):
                return np.full(len(t), 1e6)
            coef, *_ = np.linalg.lstsq(A, t, rcond=None)
            return t - A @ coef

        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=max(2, LM_EVALS // (1 + len(x0))))
            z = sol.x
        except (ValueError, np.linalg.LinAlgError):
            z = x0
        for b, th in zip(state.blocks, unpack(z)):
            b.theta = th
        return self.fit_linear(state, X, M, t, ctx)

    @staticmethod
    def _is_exponent(block):
        return block.exponent_mask()

    @staticmethod
    def _is_beta(block):
        return block.beta_mask()

    def screen(self, state, cands, X, M, t, ctx):
        """Score every (candidate, initial theta) by the drop in squared error after a linear refit."""
        with np.errstate(all="ignore"):
            A0 = self.design(state, X, M, t, ctx)
        Q, _ = np.linalg.qr(A0)
        r = t - Q @ (Q.T @ t)
        scored = []
        for ci, (block, inits) in enumerate(cands):
            for init in inits:
                th = block.theta.copy()
                if block.n_free():
                    th[block.free] = init
                with np.errstate(all="ignore"):
                    C = block.columns(X, M, ctx, th)
                if not np.all(np.isfinite(C)):
                    continue
                Cp = C - Q @ (Q.T @ C)
                norm = np.linalg.norm(Cp, axis=0)
                if np.any(norm < 1e-10 * max(np.linalg.norm(C), 1e-300)):
                    continue
                c, *_ = np.linalg.lstsq(Cp, r, rcond=None)
                gain = float(np.sum((Cp @ c) ** 2))
                scored.append((gain, ci, init))
        scored.sort(key=lambda s: -s[0])
        return scored

    def bic(self, state, rss, n, transform, y):
        # a numerical floor, the same precision on the y scale for both transforms (exact data can't overfit)
        floor = 1e-12 * float(np.sum((y - y.mean()) ** 2))
        is_log = as_transform(transform) == ("log", 0.0)
        rss = max(rss, floor / float(np.exp(2 * np.mean(np.log(y)))) if is_log else floor)
        p = 1 + sum(len(b.bindings) + b.n_free() for b in state.blocks) + len(state.dens)
        jac = 2 * float(np.sum(np.log(y))) if is_log else 0.0
        prior = sum(math.log(self.prior_prob(b.template)) for b in state.blocks) + len(state.dens) * math.log(self.prior_prob("den"))
        return n * math.log(max(rss, 1e-300) / n) + p * math.log(n) + jac - 2 * prior

    def prior_prob(self, name):
        names = list(PRIMITIVES) + [t["name"] for t in self.library]
        total = sum(self.prior.get(k, 1.0) for k in names)
        return self.prior.get(name, 1.0) / total

    def pole_free(self, state, coef, ctx):
        if not state.dens:
            return True
        m = len(state.dens)
        D = np.stack([np.prod(ctx.dom_X ** np.array(mm, float), 1) for mm in state.dens], 1) @ coef[-m:]
        return bool(np.min(1 + D) > 0.05)

    # ---- the search ---------------------------------------------------------------------------------------------------
    def fit(self, X, y, margins=None, domain=None, hints=None, start=None, start_transform=None, transforms=None,
            noise=None):
        """The plain search from an empty formula, for each transform (default y and, if positive, log y); and, if a
        start formula is given, a further search from it in start_transform, adding at most K_CONTINUE blocks. Every
        formula met goes into one pool, from which BIC picks the frontier."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        M = np.zeros((n, 0)) if margins is None else np.asarray(margins, float)
        self.noise, self._train = noise, (X, M)
        domain = domain or {"X": X, "M": M}
        ctx = Context(X, M, domain)
        hints = hints or {}
        safe_cache = {}
        cands = self.primitive_candidates(d, M.shape[1], hints) + self.library_candidates(d, hints)
        dens = self.den_candidates(d)
        if transforms is None and self.outer:
            scores = {}
            for tr in outer_family(y):
                t = t_forward(tr, y)
                if np.all(np.isfinite(t)) and np.std(t) > 0:
                    scores[tr] = self.quick_score(tr, X, M, y, t, ctx, cands)
            self.outer_scores = scores
            top = sorted(scores, key=scores.get)[:3]
            transforms = [("identity",)] + ([("log", 0.0)] if ("log", 0.0) in scores else []) + top
        transforms = list(dict.fromkeys(as_transform(tr) for tr in (transforms or
                                        (["identity"] + (["log"] if np.all(y > 0) else [])))))
        runs = [(tr, None) for tr in transforms]
        if start is not None:
            runs.append((as_transform(start_transform), start))
        self.records = []
        for transform, init in runs:
            t = t_forward(transform, y)
            beam = [init.copy() if init is not None else State([], [])]
            if init is not None:
                coef, rss = self.refit_joint(beam[0], X, M, t, ctx)
                steps = max(0, min(K_CONTINUE, (n - 2) // 2 - len(init.blocks)))
            else:
                coef, rss = self.fit_linear(beam[0], X, M, t, ctx)
                steps = max(1, min(self.k_max, (n - 2) // 2))
            self._record(beam[0], coef, rss, transform, y, n, ctx)
            for _ in range(steps):
                new = {}
                for st in beam:
                    scored = self.screen(st, cands, X, M, t, ctx)
                    tried = set()                                      # refine the REFINE best distinct candidates
                    for gain, ci, init in scored:
                        if len(tried) >= REFINE:
                            break
                        if ci in tried:
                            continue
                        block = cands[ci][0].copy()
                        if block.n_free():
                            block.theta[block.free] = init
                        nxt = State([b.copy() for b in st.blocks] + [block], st.dens)
                        coef, rss = self.refit_joint(nxt, X, M, t, ctx, only_last=True)
                        if coef is None:
                            continue
                        tried.add(ci)
                        if not all(self._safe(b, ctx, safe_cache) for b in nxt.blocks) or not self.pole_free(nxt, coef, ctx):
                            continue
                        k = nxt.key()
                        if k not in new or rss < new[k][2]:
                            new[k] = (nxt, coef, rss)
                    for m in dens[:0] if len(st.dens) >= 2 else dens:          # denominators, at most 2
                        nxt = State([b.copy() for b in st.blocks], st.dens + [m])
                        coef, rss = self.fit_linear(nxt, X, M, t, ctx)
                        if coef is None or not self.pole_free(nxt, coef, ctx):
                            continue
                        k = nxt.key()
                        if k not in new or rss < new[k][2]:
                            new[k] = (nxt, coef, rss)
                if not new:
                    break
                ranked = sorted(new.values(), key=lambda v: v[2])[:BEAM]
                beam = []
                for st_, coef_, rss_ in ranked:
                    trial = st_.copy()
                    c2, r2 = self.refit_joint(trial, X, M, t, ctx)
                    if c2 is not None and r2 < rss_ and all(self._safe(b, ctx, safe_cache) for b in trial.blocks)                             and self.pole_free(trial, c2, ctx):
                        st_, coef_, rss_ = trial, c2, r2
                    beam.append(st_)
                    self._record(st_, coef_, rss_, transform, y, n, ctx)
        self._finish(n)
        return self

    def quick_score(self, tr, X, M, y, t, ctx, cands, terms=3):
        """How well a short sum fits the target t = f^-1(y): greedy screening for `terms` blocks at their grid starts
        (no refinement), the y-scale residual sum of squares of the result (lower is better)."""
        st = State([], [])
        for _ in range(terms):
            scored = self.screen(st, cands, X, M, t, ctx)
            if not scored:
                break
            _, ci, init = scored[0]
            b = cands[ci][0].copy()
            if b.n_free():
                b.theta[b.free] = init
            st = State(st.blocks + [b], [])
        coef, _ = self.fit_linear(st, X, M, t, ctx)
        if coef is None:
            return np.inf
        pred = t_inverse(tr, self.design(st, X, M, t, ctx) @ coef)
        return float(np.sum((pred - y) ** 2)) if np.all(np.isfinite(pred)) else np.inf

    def _safe(self, block, ctx, cache):
        k = (block.key(), tuple(np.round(block.theta, 6)))
        if k not in cache:
            cache[k] = block.safe(ctx)
        return cache[k]

    def _record(self, state, coef, rss, transform, y, n, ctx):
        if coef is None or not self.pole_free(state, coef, ctx):
            return
        rec = dict(state=state.copy(), coef=coef, rss=rss, transform=transform, ctx=ctx)
        if self.noise is None:
            rec["bic"] = self.bic(state, rss, n, transform, y)
        else:
            pred = self.predict_one(rec, *self._train)
            rss_y = float(np.sum((pred - y) ** 2)) if np.all(np.isfinite(pred)) else np.inf
            p = 1 + sum(len(b.bindings) + b.n_free() for b in state.blocks) + len(state.dens)
            prior = sum(math.log(self.prior_prob(b.template)) for b in state.blocks) + \
                len(state.dens) * math.log(self.prior_prob("den"))
            rec["bic"] = rss_y / self.noise ** 2 + p * math.log(n) - 2 * prior
        if np.isfinite(rec["bic"]):
            self.records.append(rec)

    def _finish(self, n):
        recs = sorted(self.records, key=lambda r: r["bic"])
        seen, front = set(), []
        for r in recs:
            k = (r["state"].key(), r["transform"])
            if k in seen:
                continue
            seen.add(k)
            if r["bic"] <= recs[0]["bic"] + FRONTIER_BIC and len(front) < FRONTIER:
                front.append(r)
        b0 = front[0]["bic"]
        w = np.array([math.exp(-(r["bic"] - b0) / 2) for r in front])
        self.frontier, self.weights = front, w / w.sum()
        self.best = front[0]

    # ---- prediction ---------------------------------------------------------------------------------------------------
    def predict_one(self, rec, X, margins=None):
        X = np.asarray(X, float)
        M = np.zeros((len(X), 0)) if margins is None else np.asarray(margins, float)
        st, coef = rec["state"], rec["coef"]
        m = len(st.dens)
        with np.errstate(all="ignore"):
            A = self.design(State(st.blocks, []), X, M, np.zeros(len(X)), rec["ctx"])
            P = A @ coef[:A.shape[1]]
            D = (np.stack([np.prod(X ** np.array(mm, float), 1) for mm in st.dens], 1) @ coef[-m:]) if m else 0.0
            t = P / (1 + D)
            return t_inverse(rec["transform"], t)

    def predict(self, X, margins=None):
        preds = np.stack([self.predict_one(r, X, margins) for r in self.frontier])
        return self.weights @ preds

    def spread(self, X, margins=None, scale=1.0):
        preds = np.stack([self.predict_one(r, X, margins) for r in self.frontier])
        mean = self.weights @ preds
        return float(np.mean(np.sqrt(self.weights @ (preds - mean) ** 2)) / scale)

    def formula(self, rec=None):
        rec = rec or self.best
        st, coef = rec["state"], rec["coef"]
        terms, at = [f"{coef[0]:.4g}"], 1
        for b in st.blocks:
            k = len(b.bindings)
            c = coef[at:at + k]
            at += k
            terms.append((f"{c[0]:+.4g}*" if k == 1 else "+[" + ",".join(f"{x:.3g}" for x in c) + "]*") + b.label(self.names, self.mnames, rec["ctx"]))
        body = " ".join(terms)
        if st.dens:
            dc = coef[at:]
            den = " ".join(f"{c:+.3g}*" + "*".join(f"{self.names[i]}^{e}" for i, e in enumerate(m) if e) for c, m in zip(dc, st.dens))
            body = f"({body}) / (1 {den})"
        return f"{t_label(rec['transform'])} = " + body

    def n_terms(self, rec=None):
        rec = rec or self.best
        return 1 + sum(len(b.bindings) for b in rec["state"].blocks) + len(rec["state"].dens)
