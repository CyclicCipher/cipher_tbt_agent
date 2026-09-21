"""Experiment A -- geometric mixing of TWO experts with a Bayesian mixture over a GRID of exponents (no learning; a
Cover-style universal mixture over hypotheses, each hypothesis one fixed pair of exponents).

Follows E34 (experiments/ziplearn/e34.py): the chain expert (E33's blended order-8 table) and ONE other expert
(`word`, `match`, `order4`, one run each) are trained by counting on a training slice (E34's `run_stream`, which also
trains E34's linear `Mixer` with 2 experts per mixing context), then the held-out book is coded ONLINE (counts and
posteriors keep growing) under five predictors at once:
    chain alone          P_chain(c)
    other alone          P_other(c)
    linear               E34's Mixer over the 2 experts (Bayesian weights per mixing context, share 0.02)
    grid-Bayes           sum_k w_k * G_k(c),  G_k(c) = P_chain(c)^a_k * P_other(c)^b_k / Z_k   over a grid of (a, b);
                         w_k = the posterior over grid points (prior uniform; w_k <- w_k * G_k(c) / sum, then a fixed
                         share ALPHA of the mass spread uniformly so the posterior can move); ALPHA = 0.01
    grid-Bayes, alpha 0  the same with no share = the pure Bayesian model average over the grid; guaranteed within
                         log2(K)/n bits/char of the best FIXED grid point in hindsight (K grid points, n characters)
plus the cumulative bits of EVERY grid point (so the best fixed (a, b) in hindsight is read off), and two closed-form
variants of the grid posterior that condition on what the mixer can see: one posterior per (other's context seen /
unseen), and one per E34 mixing context (previous character's class x the chain's deepest seen order, 36 contexts).

Theory under test: the geometric mixture sharpens only when the two experts carry INDEPENDENT evidence; then the best
exponents sum to more than 1 (a + b > 1), and the grid posterior finds them without a gradient. For an expert whose
evidence the chain already holds (order4 is one of the chain's own backoff levels) the best b should be ~0 and a + b ~ 1.

    PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe experiments/ziplearn/research/expA_exponent_grid.py \
        --train_chars 300000 --test_chars 60000            (CPU, ~1-2 min) -> research/expA_exponent_grid.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e34 import Chain, Mixer, Stream, build_experts, run_stream    # noqa: E402
from textlm import load_corpus                                     # noqa: E402

N_CTX = 4 * 9                                                      # E34's mixing contexts


class GridPosterior:
    """A Bayesian posterior over K fixed hypotheses (grid points), one weight vector per condition (n_cond of them),
    with fixed share alpha. No parameters are learned: the weights ARE the posterior, updated in closed form."""

    def __init__(self, K, n_cond=1, alpha=0.01):
        self.w = np.full((n_cond, K), 1.0 / K)
        self.alpha = alpha
        self.K = K
        self.wsum = np.zeros((n_cond, K))                           # time-integrated weights (for the average favourite)
        self.n = np.zeros(n_cond)

    def code(self, cond, pg):
        """Bits of the posterior-weighted mixture for the probabilities pg (K,) the grid points gave the character; then
        the Bayes update of the posterior for this condition."""
        w = self.w[cond]
        p = float(w @ pg)
        self.wsum[cond] += w
        self.n[cond] += 1
        post = w * pg / p
        if self.alpha:
            post = (1 - self.alpha) * post + self.alpha / self.K
        self.w[cond] = post
        return -math.log2(p)


def evaluate(chain, other, mixer, stream, test, A, B):
    """One online pass over `test`. Returns the bits of every predictor and the per-grid-point cumulative bits."""
    K = len(A)
    bits = dict(chain=0.0, other=0.0, linear=0.0, grid_bayes=0.0, grid_bayes_alpha0=0.0,
                grid_bayes_by_seen=0.0, grid_bayes_by_ctx=0.0)
    cum = np.zeros(K)                                              # cumulative bits per fixed grid point
    cum_seen = np.zeros(K)                                         # ... over the characters where other's context was seen
    n_seen = 0
    post_all = GridPosterior(K, 1, alpha=0.01)
    post_a0 = GridPosterior(K, 1, alpha=0.0)
    post_seen = GridPosterior(K, 2, alpha=0.01)
    post_ctx = GridPosterior(K, N_CTX, alpha=0.01)
    probs = np.zeros(2)
    s = stream
    for c in test.tolist():
        keys = chain.predict(s)
        dC, _ = chain.dist(keys)
        k, N, cnt = other.predict(s)
        dO, seen = other.dist(k, N, cnt)
        logC = np.log(np.clip(dC, 1e-300, None))
        logO = np.log(np.clip(dO, 1e-300, None))
        L = A[:, None] * logC[None, :] + B[:, None] * logO[None, :]       # (K, V)
        L -= L.max(1, keepdims=True)
        G = np.exp(L)
        G /= G.sum(1, keepdims=True)
        pg = G[:, c]
        # the solo experts and E34's linear mixer
        probs[0], probs[1] = dC[c], dO[c]
        bits["chain"] -= math.log2(probs[0])
        bits["other"] -= math.log2(probs[1])
        deep = chain.deepest(keys)
        ctx = s.cls() * 9 + deep
        p_lin, _w = mixer.mix(ctx, probs)
        bits["linear"] -= math.log2(p_lin)
        mixer.update(ctx, probs, p_lin)
        # the grid mixtures
        bits["grid_bayes"] += post_all.code(0, pg)
        bits["grid_bayes_alpha0"] += post_a0.code(0, pg)
        bits["grid_bayes_by_seen"] += post_seen.code(int(seen), pg)
        bits["grid_bayes_by_ctx"] += post_ctx.code(ctx, pg)
        lg = -np.log2(np.clip(pg, 1e-300, None))
        cum += lg
        if seen:
            cum_seen += lg
            n_seen += 1
        # learn
        chain.update(keys, c)
        other.update(k, c)
        s.push(c)
    n = len(test)
    return {k: v / n for k, v in bits.items()}, cum / n, cum_seen / max(n_seen, 1), n_seen, post_all, post_a0, post_seen, post_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_chars", type=int, default=300000)
    ap.add_argument("--test_chars", type=int, default=60000)
    ap.add_argument("--others", nargs="+", default=["word", "match", "order4"])
    ap.add_argument("--a", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--b", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--out", default=str(HERE / "expA_exponent_grid.json"))
    args = ap.parse_args()
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)[:args.train_chars]   # E34's --train_chars convention: a prefix of the concatenated books
    test = test[:args.test_chars]
    grid = [(a, b) for a in args.a for b in args.b]
    A = np.array([g[0] for g in grid]); B = np.array([g[1] for g in grid])
    K = len(grid)
    print(f"Experiment A: {len(text)} training characters, {len(test)} held out (online); V = {V}; grid K = {K} "
          f"(a in {args.a}, b in {args.b}); share alpha 0.01", flush=True)
    report = dict(V=V, n_train=int(len(text)), n_test=int(len(test)), grid=grid, alpha=0.01, runs={})
    for name in args.others:
        t0 = time.time()
        ex = build_experts(V, alphabet)
        chain = ex[0]
        other = next(e for e in ex if e.name == name)
        experts = [chain, other]
        mixer = Mixer(len(experts), N_CTX)                                # E34's linear mixer, 2 experts, share 0.02
        _tb, _solo, s, mixer = run_stream(experts, text, V, alphabet, mixer=mixer, track_solo=False)
        t_train = time.time() - t0
        bits, cum, cum_seen, n_seen, post_all, post_a0, post_seen, post_ctx = evaluate(chain, other, mixer, s, test, A, B)
        t_test = time.time() - t0 - t_train
        best = int(np.argmin(cum))
        best_seen = int(np.argmin(cum_seen))
        fav_end = int(np.argmax(post_all.w[0]))
        fav_avg = int(np.argmax(post_all.wsum[0]))
        fav_a0 = int(np.argmax(post_a0.w[0]))
        fav_seen = [int(np.argmax(post_seen.w[i])) for i in range(2)]
        # posterior-mean exponents (time-averaged over the pass)
        wbar = post_all.wsum[0] / post_all.n[0]
        mean_ab = (float(wbar @ A), float(wbar @ B))
        row = dict(bits=bits,
                   grid_bits={f"a={a:.1f},b={b:.1f}": float(v) for (a, b), v in zip(grid, cum)},
                   grid_bits_seen_only={f"a={a:.1f},b={b:.1f}": float(v) for (a, b), v in zip(grid, cum_seen)},
                   frac_other_seen=n_seen / len(test),
                   best_fixed=dict(a=grid[best][0], b=grid[best][1], bits=float(cum[best])),
                   best_fixed_on_seen=dict(a=grid[best_seen][0], b=grid[best_seen][1], bits=float(cum_seen[best_seen])),
                   chain_tempered_best=dict(a=float(A[np.argmin(np.where(B == 0, cum, np.inf))]),
                                            bits=float(cum[np.argmin(np.where(B == 0, cum, np.inf))])),
                   posterior_favourite_end=dict(a=grid[fav_end][0], b=grid[fav_end][1], w=float(post_all.w[0][fav_end])),
                   posterior_favourite_avg=dict(a=grid[fav_avg][0], b=grid[fav_avg][1], w=float(wbar[fav_avg])),
                   posterior_mean_ab=dict(a=mean_ab[0], b=mean_ab[1]),
                   posterior_alpha0_favourite=dict(a=grid[fav_a0][0], b=grid[fav_a0][1], w=float(post_a0.w[0][fav_a0])),
                   posterior_by_seen_favourite=dict(unseen=dict(a=grid[fav_seen[0]][0], b=grid[fav_seen[0]][1]),
                                                    seen=dict(a=grid[fav_seen[1]][0], b=grid[fav_seen[1]][1])),
                   mixer_weight_chain_mean=float(mixer.w[:, 0].mean()),
                   seconds=dict(train=t_train, test=t_test))
        report["runs"][name] = row
        print(f"  [{name}] bits/char online: chain {bits['chain']:.3f} | {name} {bits['other']:.3f} | linear (E34 Mixer) {bits['linear']:.3f} | "
              f"grid-Bayes {bits['grid_bayes']:.3f} (alpha0 {bits['grid_bayes_alpha0']:.3f}; by seen {bits['grid_bayes_by_seen']:.3f}; "
              f"by ctx {bits['grid_bayes_by_ctx']:.3f}) | best fixed a={grid[best][0]:.1f} b={grid[best][1]:.1f} -> {cum[best]:.3f} "
              f"| chain tempered best a={row['chain_tempered_best']['a']:.1f} -> {row['chain_tempered_best']['bits']:.3f} "
              f"| posterior favourite at end a={grid[fav_end][0]:.1f} b={grid[fav_end][1]:.1f} (w {post_all.w[0][fav_end]:.2f}), "
              f"time-avg favourite a={grid[fav_avg][0]:.1f} b={grid[fav_avg][1]:.1f}, mean (a,b)=({mean_ab[0]:.2f},{mean_ab[1]:.2f}) "
              f"| other's context seen {100 * n_seen / len(test):.0f}% of chars; on those, best fixed a={grid[best_seen][0]:.1f} b={grid[best_seen][1]:.1f} "
              f"| mixer mean weight on chain {mixer.w[:, 0].mean():.2f}  [train {t_train:.0f}s, test {t_test:.0f}s]", flush=True)
        # the (a, b) table of bits, for the eye
        print("      fixed-grid bits/char (rows a, columns b):  b = " + "  ".join(f"{b:5.1f}" for b in args.b))
        for a in args.a:
            print(f"        a = {a:.1f}: " + "  ".join(f"{cum[grid.index((a, b))]:5.3f}" for b in args.b))
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
