"""Round-1 check (the reviser, 2026-09-22): the COMPARATOR of a per-context grid posterior.

The dominance inequality bounds a per-context Bayesian mixture over a grid G of exponent vectors against the best
PER-CONTEXT fixed assignment in hindsight (36 independent argmins), at a prior cost of n_ctx * log2|G| / n bits/char --
not against the best GLOBAL grid point, which is what expA_exponent_grid.py's `best_fixed` and
mix_temperature_bayes.py's `grid2_best` report.  Neither script accumulates the per-context per-point totals, so this
script re-runs their two online passes UNCHANGED except for one (n_ctx, K) accumulator and per-10k-block bookkeeping:

  part A -- expA chain + word, the fine grid of expA_exponent_grid_fine.json (a in 0.5..1.1, b in 0.0..0.8, K = 63),
            prefix 300k / first 60k held out: global best fixed, per-context oracle, grid_bayes_by_ctx (share 0.01).
  part B -- mix_temperature_bayes.py's 2-D grid (BETA_C x BETA_R, K = 42) over the 18 experts with the DECAYING switch
            (A2-min of mix_temperature_bayes_decay.json): global best point, per-context oracle, the posterior.

Every difference is a paired difference on the same 60k characters with its standard error over six 10k blocks.

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/check_per_context_oracle.py
        -> research/check_per_context_oracle.json   (CPU, ~3 min)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e34 import Chain, Mixer, build_experts, run_stream                  # noqa: E402
from textlm import load_corpus                                           # noqa: E402
from expA_exponent_grid import GridPosterior, N_CTX                      # noqa: E402
from mix_temperature_bayes import BETA_C, BETA_R                         # noqa: E402

BLOCK = 10000


def blocks(per_char, n):
    b = np.array([per_char[i:i + BLOCK].sum() / BLOCK for i in range(0, n - BLOCK + 1, BLOCK)])
    return b


def paired(x_blocks, y_blocks):
    d = x_blocks - y_blocks
    return dict(mean=float(d.mean()), se_blocks=float(d.std(ddof=1) / math.sqrt(len(d))), per_block=[float(v) for v in d])


def part_a(alphabet, text, test, V):
    A_ = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    B_ = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    grid = [(a, b) for a in A_ for b in B_]
    A = np.array([g[0] for g in grid]); B = np.array([g[1] for g in grid]); K = len(grid)
    ex = build_experts(V, alphabet)
    chain = ex[0]
    other = next(e for e in ex if e.name == "word")
    mixer = Mixer(2, N_CTX)
    _tb, _solo, s, mixer = run_stream([chain, other], text, V, alphabet, mixer=mixer, track_solo=False)
    post_ctx = GridPosterior(K, N_CTX, alpha=0.01)
    n = len(test)
    cum_ctx = np.zeros((N_CTX, K))
    lg_chain = np.zeros(n); lg_ctx = np.zeros(n); lg_pt = np.zeros((n, K), dtype=np.float32); ctx_of = np.zeros(n, dtype=np.int32)
    for i, c in enumerate(test.tolist()):
        keys = chain.predict(s)
        dC, _ = chain.dist(keys)
        k, N, cnt = other.predict(s)
        dO, seen = other.dist(k, N, cnt)
        logC = np.log(np.clip(dC, 1e-300, None)); logO = np.log(np.clip(dO, 1e-300, None))
        L = A[:, None] * logC[None, :] + B[:, None] * logO[None, :]
        L -= L.max(1, keepdims=True)
        G = np.exp(L); G /= G.sum(1, keepdims=True)
        pg = G[:, c]
        deep = chain.deepest(keys)
        ctx = s.cls() * 9 + deep
        lg_chain[i] = -math.log2(dC[c])
        lg_ctx[i] = post_ctx.code(ctx, pg)
        lg = -np.log2(np.clip(pg, 1e-300, None))
        cum_ctx[ctx] += lg
        lg_pt[i] = lg; ctx_of[i] = ctx
        chain.update(keys, c); other.update(k, c); s.push(c)
    return finish("expA chain+word fine grid, prefix 300k", grid, K, cum_ctx, lg_chain, lg_ctx, lg_pt, ctx_of, n, "grid_bayes_by_ctx (share 0.01)")


def part_b(alphabet, text, test, V):
    grid2 = np.array([(bc, br) for bc in BETA_C for br in BETA_R]); K = len(grid2)
    grid = [tuple(map(float, g)) for g in grid2.tolist()]
    experts = build_experts(V, alphabet)
    _tb, _ts, s, mixer = run_stream(experts, text, V, alphabet, track_solo=False)
    chain = experts[0]
    n_ctx = mixer.w.shape[0]
    post2 = np.full((n_ctx, K), 1.0 / K)
    used = np.zeros(n_ctx)
    probs = np.zeros(len(experts))
    n = len(test)
    cum_ctx = np.zeros((n_ctx, K))
    lg_chain = np.zeros(n); lg_ctx = np.zeros(n); lg_pt = np.zeros((n, K), dtype=np.float32); ctx_of = np.zeros(n, dtype=np.int32)
    for i, c in enumerate(test.tolist()):
        preds = [e.predict(s) for e in experts]
        dists, seen = [], []
        for e, pr in zip(experts, preds):
            d, ok = e.dist(pr) if isinstance(e, Chain) else e.dist(pr[0], pr[1], pr[2])
            dists.append(d); seen.append(ok)
        D = np.stack(dists)
        probs[:] = D[:, c]
        ctx = s.cls() * 9 + chain.deepest(preds[0])
        p_lin, w = mixer.mix(ctx, probs)
        sm = np.array(seen)
        log_chain = np.log(np.clip(D[0], 1e-12, None))
        rest = sm.copy(); rest[0] = False
        if rest.any():
            wr = w[rest] / w[rest].sum()
            log_rest = wr @ np.log(np.clip(D[rest], 1e-12, None))
        else:
            log_rest = np.zeros(V)
        Z2 = grid2[:, :1] * log_chain[None, :] + grid2[:, 1:] * log_rest[None, :]
        Z2 -= Z2.max(1, keepdims=True)
        G2 = np.exp(Z2); G2 /= G2.sum(1, keepdims=True)
        p2 = np.clip(G2[:, c], 1e-12, None)
        pi2 = post2[ctx]
        pm2 = float(pi2 @ p2)
        a = 1.0 / (used[ctx] + 2)                                        # the decaying switch of the A2-min run
        post2[ctx] = (1 - a) * (pi2 * p2 / pm2) + a / K
        used[ctx] += 1
        lg_chain[i] = -math.log2(max(D[0, c], 1e-12))
        lg_ctx[i] = -math.log2(pm2)
        lg = -np.log2(p2)
        cum_ctx[ctx] += lg
        lg_pt[i] = lg; ctx_of[i] = ctx
        mixer.update(ctx, probs, p_lin)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        s.push(c)
    return finish("A2-min: 42-point (beta_chain, beta_rest) grid, 18 experts, decaying switch, prefix 300k", grid, K, cum_ctx,
                  lg_chain, lg_ctx, lg_pt, ctx_of, n, "bayes_over_grid2 (decay)")


def finish(label, grid, K, cum_ctx, lg_chain, lg_ctx, lg_pt, ctx_of, n, post_name):
    cum = cum_ctx.sum(0)
    best_global = int(np.argmin(cum))
    best_per_ctx = cum_ctx.argmin(1)                                     # 36 independent argmins
    n_ctx_used = int((cum_ctx.sum(1) > 0).sum())
    oracle_total = float(cum_ctx[np.arange(len(best_per_ctx)), best_per_ctx].sum())
    # per-character code length of the oracle assignment and of the global best point (for the block SEs)
    lg_oracle = lg_pt[np.arange(n), best_per_ctx[ctx_of]].astype(float)
    lg_global = lg_pt[:, best_global].astype(float)
    prior_cost = n_ctx_used * math.log2(K) / n
    b_chain, b_ctx, b_or, b_gl = blocks(lg_chain, n), blocks(lg_ctx, n), blocks(lg_oracle, n), blocks(lg_global, n)
    out = dict(label=label, K=K, n=n, n_ctx_used=n_ctx_used,
               bits=dict(chain=float(lg_chain.sum() / n), posterior=float(lg_ctx.sum() / n),
                         best_global_point=float(cum[best_global] / n), per_context_oracle=oracle_total / n,
                         per_context_oracle_plus_prior=oracle_total / n + prior_cost),
               posterior_name=post_name,
               prior_cost_bits_per_char=prior_cost,
               best_global_point=grid[best_global],
               per_context_best_points={int(i): grid[int(j)] for i, j in enumerate(best_per_ctx) if cum_ctx[i].sum() > 0},
               diff_posterior_minus_oracle=paired(b_ctx, b_or),
               diff_posterior_minus_global_best=paired(b_ctx, b_gl),
               diff_oracle_minus_global_best=paired(b_or, b_gl),
               diff_posterior_minus_chain=paired(b_ctx, b_chain))
    print(f"[{label}] chain {out['bits']['chain']:.4f} | {post_name} {out['bits']['posterior']:.4f} | best global point {grid[best_global]} "
          f"{out['bits']['best_global_point']:.4f} | per-context oracle ({n_ctx_used} contexts) {out['bits']['per_context_oracle']:.4f} "
          f"(+ prior {prior_cost:.4f} = {out['bits']['per_context_oracle_plus_prior']:.4f}) | posterior - oracle "
          f"{out['diff_posterior_minus_oracle']['mean']:+.4f} +- {out['diff_posterior_minus_oracle']['se_blocks']:.4f} | "
          f"posterior - global best {out['diff_posterior_minus_global_best']['mean']:+.4f} +- {out['diff_posterior_minus_global_best']['se_blocks']:.4f}",
          flush=True)
    return out


def main():
    t0 = time.time()
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)[:300000]
    test = test[:60000]
    report = dict(V=V, n_train=int(len(text)), n_test=int(len(test)), block=BLOCK)
    report["A_expA_chain_word_fine"] = part_a(alphabet, text, test, V)
    print(f"  part A done at {time.time() - t0:.0f}s", flush=True)
    report["B_A2min_grid2_decay"] = part_b(alphabet, text, test, V)
    print(f"  part B done at {time.time() - t0:.0f}s", flush=True)
    report["seconds"] = time.time() - t0
    json.dump(report, open(HERE / "check_per_context_oracle.json", "w"), indent=1)


if __name__ == "__main__":
    main()
