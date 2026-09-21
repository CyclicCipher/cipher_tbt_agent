"""Evaluate INNER objectives -- rules that decide how the weights change -- with no model, no training, no inference.

A training step sees one gradient per environment (task family). The outer objective is the task loss; the INNER
objective is how those per-environment gradients are combined into an update. Plain SGD/Adam uses their mean, which
rewards whatever explains the current batch. The candidates here judge an update coordinate by AGREEMENT across
environments: a direction that helps everywhere is structure, one that helps one environment is memorisation, and
noise agrees with nothing.

The world is linear so that everything is exact. Each environment e has loss L_e(w) = 1/2 ||w - w*_e||^2 and its
optimum decomposes over three orthogonal blocks of the weight vector:
    invariant   w*_e = w_inv          the same in every environment   (the primitives every composition shares)
    spurious    w*_e = s_e            random per environment           (a regularity real within e, useless elsewhere)
    irrelevant  w*_e = 0              nothing to learn there
Per-environment gradient: g_e = (w - w*_e) + sigma * xi_e (finite-sample noise). NOISE environments contribute
g_e = sigma_noise * xi_e only -- the noisy TV.

TRANSFER ERROR is what a fresh environment (same w_inv, new s) costs beyond its unavoidable floor:
    ||P_inv (w - w_inv)||^2 + ||P_sp w||^2 + ||P_irr w||^2
Any weight left in the spurious block is training-family memorisation and hurts a novel family in expectation.

Update rules, all consuming {g_e}:
    mean     u = mean_e g_e                                                    (the baseline)
    andmask  m_i = 1[ |sum_e sign g_ei| >= tau K ],  u = m * mean_e g_e       (Parascandolo et al. 2020)
    soft     m_i = |mean_e sign g_ei|,               u = m * mean_e g_e
    snr      m_i = clip(|mean_e g_ei| / (std_e g_ei + eps), 0, 1), u = m * mean_e g_e
    geomean  sign-consistent coordinates get the geometric mean of |g_ei|, others 0
    invvar   m_i = 1 / (1 + Var_e(g_ei) / median_i Var_e(g_ei))     -- trust what environments AGREE ON IN VALUE
    varmask  m_i = 1[ Var_e(g_ei) <= 2 * median_i Var_e(g_ei) ]      -- the hard version
    invvar+  as invvar, with each ENVIRONMENT weighted by its agreement with the leave-one-out consensus
    varmask+ as varmask, likewise                                     -- the criterion recursed one level up
    ftest    two half-batches per environment; an environment is trusted by agreement with ITSELF (reproducible =
             learnable, which noise never is); a coordinate is kept, HARD, only if its variance across trusted
             environments is <= c times its variance within an environment.
    ftest+wd as ftest, and a masked-out coordinate DECAYS toward zero: if the environments cannot agree what a weight
             should be, it should be nothing.
    consolidate  the F-test gates RETENTION, not learning: full gradient everywhere, decay on what is not shared. The
             model experiment showed `ftest+wd` cannot bootstrap in a real network (nothing is shared at init, so it
             refuses the first steps of learning); this is the repair, run here first.

The first four judge agreement of SIGN, and the first run of this script showed why that is the wrong criterion: from a
random start every environment agrees on the direction toward its own spurious target, so a sign gate lets the weights
drift to the centroid of the environment-specific targets -- memorisation -- and only freezes them once they are there.
What distinguishes shared structure from a family-specific regularity is the VARIANCE of the gradient across
environments -- zero in the invariant block, Var(s) in the spurious one, wherever the weights currently are -- and the
threshold is set from the data (the median over coordinates) so it tracks the noise level.

Two measurements per rule: the DIRECTION at a random point (how much of the update lies in each block, cosine to the
ideal update that moves only along invariant + irrelevant error) and the IDEALISED LEARNER (iterate the rule with fresh
noise each step; transfer error after T steps; steps to bring invariant error under 10%). Swept over the number of
environments, the noise level and the presence of noise environments -- the phase diagram of where an agreement
criterion wins, and where it merely cuts the learning rate.

Runs in seconds:  python experiments/inner_objective/verify_objective.py
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

D, K_INV, K_SP = 64, 16, 16                              # weight dims: 16 invariant, 16 spurious, 32 irrelevant
INV, SP, IRR = slice(0, K_INV), slice(K_INV, K_INV + K_SP), slice(K_INV + K_SP, D)
RULES = ["mean", "andmask", "soft", "snr", "geomean", "invvar", "varmask", "invvar+", "varmask+", "ftest", "ftest+wd",
         "consolidate"]


def make_world(K, n_noise, rng):
    w_inv = rng.normal(size=K_INV)
    s = rng.normal(size=(K, K_SP))
    return w_inv, s, n_noise


def grads(w, world, sigma, sigma_noise, rng, halves=False):
    """Per-environment gradients at w: (K + n_noise, D); with `halves`, two independent samples per environment,
    shape (2, K + n_noise, D) -- the two half-batches the F-test rule needs."""
    if halves:
        return np.stack([grads(w, world, sigma, sigma_noise, rng) for _ in range(2)])
    w_inv, s, n_noise = world
    K = s.shape[0]
    target = np.zeros((K, D))
    target[:, INV] = w_inv
    target[:, SP] = s
    g = (w[None, :] - target) + sigma * rng.normal(size=(K, D))
    if n_noise:
        g = np.vstack([g, sigma_noise * rng.normal(size=(n_noise, D))])
    return g


def combine(g, rule, tau=0.75, eps=1e-8, c=3.0, w=None, wd=0.1):
    """The inner objective: per-environment gradients (E, D) -> one update (D,). `ftest` takes (2, E, D)."""
    if rule.startswith("ftest") or rule == "consolidate":
        # THE CANDIDATE. Two half-batch gradients per environment. (1) An environment is trusted by agreement with
        # ITSELF -- cos(g1, g2) -- which is whether its gradient is reproducible, i.e. learnable; a noise family fails
        # this at every stage of training, and it does not degenerate once shared structure is learned. (2) Per
        # coordinate, a HARD mask: keep it only if the variance ACROSS trusted environments is no more than c times the
        # variance WITHIN an environment (the two halves). Across-variance far above within-variance means the
        # environments genuinely want different values there: family-specific, so leave it alone.
        g1, g2 = g[0], g[1]
        t = np.array([max(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps)), 0.0) for a, b in zip(g1, g2)])
        keep = t > 0.2
        if keep.sum() < 2:
            return np.zeros(g.shape[-1])
        g1, g2 = g1[keep], g2[keep]
        gm = 0.5 * (g1 + g2)
        within = 0.25 * ((g1 - g2) ** 2).mean(0)              # variance of a half-batch mean, per coordinate
        across = gm.var(0)
        mask = across <= c * within + eps
        if rule == "consolidate":
            # Gate RETENTION, not learning: every coordinate takes its full gradient (a real network must first build
            # structure before any of it can be shared), and only the coordinates the environments disagree on are
            # decayed toward zero. Learn freely; keep what agrees.
            u = gm.mean(0)
            if w is not None:
                u = u + wd * w * (~mask)
            return u
        u = gm.mean(0) * mask
        if rule == "ftest+wd" and w is not None:
            u = u + wd * w * (~mask)                          # what the environments cannot agree on decays to nothing
        return u
    if g.ndim == 3:
        g = 0.5 * (g[0] + g[1])                                # other rules see the full-batch gradient
    mean = g.mean(0)
    if rule == "mean":
        return mean
    sg = np.sign(g)
    agree = np.abs(sg.sum(0)) / g.shape[0]               # fraction of environments agreeing on the sign, in [0, 1]
    if rule == "andmask":
        return mean * (agree >= tau)
    if rule == "soft":
        return mean * agree
    if rule == "snr":
        return mean * np.clip(np.abs(mean) / (g.std(0) + eps), 0.0, 1.0)
    if rule == "geomean":
        consistent = agree >= 1.0 - 1e-9
        gm = np.exp(np.log(np.abs(g) + eps).mean(0)) * np.sign(mean)
        return np.where(consistent, gm, 0.0)
    if rule.endswith("+"):
        # PER-ENVIRONMENT TRUST, the fix the first sweep demanded: a noise environment's gradient is ~0 where the real
        # ones are large, so it inflates the between-environment variance in exactly the coordinates carrying the most
        # signal. Weight each environment by how much its gradient agrees with the leave-one-out consensus (noise
        # agrees with nothing), then take the trusted mean and the trusted variance. The same criterion, one level up.
        E = g.shape[0]
        tot = g.sum(0)
        t = np.empty(E)
        for e in range(E):
            other = (tot - g[e]) / max(E - 1, 1)
            c = g[e] @ other / (np.linalg.norm(g[e]) * np.linalg.norm(other) + eps)
            t[e] = max(c, 0.0) ** 2
        t = t / (t.sum() + eps)
        mean = (t[:, None] * g).sum(0)
        var = (t[:, None] * (g - mean) ** 2).sum(0)
        rule = rule[:-1]
    else:
        var = g.var(0)
    scale = np.median(var) + eps
    if rule == "invvar":
        return mean / (1.0 + var / scale)
    if rule == "varmask":
        return mean * (var <= 2.0 * scale)
    raise KeyError(rule)


def ideal_update(w, world):
    """Move only along invariant and irrelevant error; nothing along the spurious block."""
    u = w.copy()
    u[INV] = w[INV] - world[0]
    u[SP] = 0.0
    return u


def transfer_error(w, world):
    return float(((w[INV] - world[0]) ** 2).sum() + (w[SP] ** 2).sum() + (w[IRR] ** 2).sum())


def block_fractions(u):
    n = (u ** 2).sum() + 1e-12
    return (u[INV] ** 2).sum() / n, (u[SP] ** 2).sum() / n, (u[IRR] ** 2).sum() / n


def direction_probe(world, sigma, sigma_noise, rng, init, n=50):
    """At random points: cosine to the ideal update and the block split of each rule's update."""
    out = {r: dict(cos=[], inv=[], sp=[], irr=[], kept=[]) for r in RULES}
    for _ in range(n):
        w = init * rng.normal(size=D)
        g = grads(w, world, sigma, sigma_noise, rng)
        ideal = ideal_update(w, world)
        g2 = grads(w, world, sigma, sigma_noise, rng, halves=True)
        for r in RULES:
            u = combine(g2 if (r.startswith("ftest") or r == "consolidate") else g, r, w=w)
            fi, fs, fr = block_fractions(u)
            c = float(u @ ideal / (np.linalg.norm(u) * np.linalg.norm(ideal) + 1e-12))
            out[r]["cos"].append(c)
            out[r]["inv"].append(fi)
            out[r]["sp"].append(fs)
            out[r]["irr"].append(fr)
            out[r]["kept"].append(float((u != 0).mean()))
    return {r: {k: float(np.mean(v)) for k, v in d.items()} for r, d in out.items()}


def learner(world, rule, sigma, sigma_noise, rng, init, lr=0.1, T=300):
    """The idealised learner: iterate the rule. Returns transfer error over time and steps-to-10%-invariant-error.
    `init` is the weight scale at the start: 1.0 is a pretrained network far from every target, 0.1 is a fresh one
    near zero -- and the two regimes give different verdicts, which is part of the finding."""
    w = init * rng.normal(size=D)
    inv0 = float(((w[INV] - world[0]) ** 2).sum())
    errs, hit = [], None
    for t in range(T):
        w = w - lr * combine(grads(w, world, sigma, sigma_noise, rng, halves=rule.startswith("ftest") or rule == "consolidate"), rule, w=w)
        errs.append(transfer_error(w, world))
        if hit is None and ((w[INV] - world[0]) ** 2).sum() < 0.1 * inv0:
            hit = t + 1
    return errs, hit, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "verify_objective.json"))
    ap.add_argument("--show", default="mean,ftest+wd,consolidate", help="rules printed (all are in the json)")
    args = ap.parse_args()
    args.show = set(args.show.split(","))
    Ks, sigmas, noises, inits = [2, 4, 8, 16], [0.0, 0.3, 1.0], [0, 4], [0.1, 1.0]
    table = []
    print(f"{'init':>4} {'K':>3} {'sigma':>5} {'noise':>5} | {'rule':<8} {'cos':>6} {'inv%':>5} {'sp%':>5} {'irr%':>5} "
          f"{'kept%':>6} | {'transfer@T':>10} {'inv err':>8} {'sp err':>7} {'steps->10%':>10}")
    for init, K, sigma, n_noise in itertools.product(inits, Ks, sigmas, noises):
        rows = {r: dict(cos=[], inv=[], sp=[], irr=[], kept=[], te=[], inv_err=[], sp_err=[], steps=[]) for r in RULES}
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            world = make_world(K, n_noise, rng)
            dp = direction_probe(world, sigma, 1.0, rng, init)
            for r in RULES:
                errs, hit, w = learner(world, r, sigma, 1.0, np.random.default_rng(1000 + seed), init)
                rows[r]["cos"].append(dp[r]["cos"])
                rows[r]["inv"].append(dp[r]["inv"])
                rows[r]["sp"].append(dp[r]["sp"])
                rows[r]["irr"].append(dp[r]["irr"])
                rows[r]["kept"].append(dp[r]["kept"])
                rows[r]["te"].append(errs[-1])
                rows[r]["inv_err"].append(float(((w[INV] - world[0]) ** 2).sum()))
                rows[r]["sp_err"].append(float((w[SP] ** 2).sum()))
                rows[r]["steps"].append(hit if hit is not None else np.nan)
        for r in RULES:
            m = {k: float(np.nanmean(v)) for k, v in rows[r].items()}
            m.update(K=K, sigma=sigma, n_noise=n_noise, rule=r, init=init)
            table.append(m)
            if r not in args.show:
                continue
            print(f"{init:>4} {K:>3} {sigma:>5.1f} {n_noise:>5} | {r:<8} {m['cos']:6.2f} {100*m['inv']:5.0f} {100*m['sp']:5.0f} "
                  f"{100*m['irr']:5.0f} {100*m['kept']:6.0f} | {m['te']:10.3f} {m['inv_err']:8.3f} {m['sp_err']:7.3f} "
                  f"{m['steps']:10.0f}")
        print()
    json.dump(table, open(args.out, "w"), indent=1)

    # The verdicts the model experiment inherits as predictions.
    def cell(rule, K, sigma, n_noise, init, key):
        return next(t[key] for t in table if t["rule"] == rule and t["K"] == K and t["sigma"] == sigma
                    and t["n_noise"] == n_noise and t["init"] == init)
    print("== verdicts: transfer error after T steps, baseline `mean` vs the best gate; the noise multiplier is the ==")
    print("== factor by which adding 4 noise environments raises each rule's error (1.0 = immune)                 ==")
    for init in inits:
        print(f"-- init scale {init} ({'pretrained, far from targets' if init >= 1 else 'fresh, near zero'}) --")
        for K in Ks:
            for sigma in sigmas:
                base = cell("mean", K, sigma, 0, init, "te")
                winner = min(RULES[1:], key=lambda r: cell(r, K, sigma, 0, init, "te"))
                best = cell(winner, K, sigma, 0, init, "te")
                hurt = cell("mean", K, sigma, 4, init, "te") / max(base, 1e-9)
                hurt_g = cell(winner, K, sigma, 4, init, "te") / max(best, 1e-9)
                tag = "GATE WINS" if best < 0.8 * base else ("gate loses" if best > 1.25 * base else "no difference")
                print(f"  K={K:2d} sigma={sigma:.1f}: mean {base:6.3f} | best gate {best:6.3f} [{winner:<7}] {tag:<13} "
                      f"| noise x{hurt:.2f} (mean) x{hurt_g:.2f} ({winner})")


if __name__ == "__main__":
    main()
