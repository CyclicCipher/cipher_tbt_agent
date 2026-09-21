"""Feel-experiment for refs/bayesian_alternatives_to_learned_mixing.md, candidate A1: is the missing degree of freedom in
E34's geometric mixture its total exponent mass?  E34 measured, on the held-out book, the geometric mixture with the
Bayesian weights (renormalised over the seen experts, so the exponents sum to 1) at 1.801 bits/char and the plain product
(exponents sum to the number of seen experts) at 7.665.  Here the exponents are beta * w_Bayes for beta on a grid, and a
BAYESIAN MIXTURE over the grid (per mixing context, fixed share) replaces choosing beta: closed form, no learned parameter,
regret <= log2(len(grid)) bits per mixing context against the best beta in hindsight (the dominance inequality).

Reuses e34.py unchanged (build_experts, run_stream, Mixer, Stream).  Subsampled by default so it runs in ~1-2 minutes:

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/mix_temperature_bayes.py \
        --train_chars 300000 --test_chars 60000     -> experiments/ziplearn/research/mix_temperature_bayes.json

Reports, all online bits/char on the held-out slice: the Bayesian linear mixture (E34's `linear`), the geometric mixture at
each beta alone (beta = 1 is E34's `geometric`), the Bayesian mixture over the beta grid, and the product (E34's `product`).
Second block (candidate A2 in its smallest form): a 2-D grid -- exponent beta_c on the chain, exponent beta_r on the
Bayesian-weighted geometric mean of the OTHER seen experts -- again a Bayesian mixture over the grid per mixing context.
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
from e34 import Chain, Mixer, Stream, build_experts, run_stream      # noqa: E402
from textlm import load_corpus                                       # noqa: E402

BETAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)
BETA_C = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)                             # exponent on the chain
BETA_R = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)                         # exponent on the geometric mean of the other seen experts


def run_temperatures(experts, seq, V, mixer, stream, betas=BETAS, alpha=0.02, switch="fixed"):
    """One online pass: linear, geometric at every beta, the Bayesian mixture over betas, the product.
    switch = "fixed": fixed share alpha (E34's Mixer); "decay": alpha_t = 1/(t+1) per mixing context (Veness et al. 2012,
    CTS Lemma 1: total switching price (switches+1)(log2 K + log2 t) bits instead of a per-step slack)."""
    s = stream
    chain = experts[0]
    B = len(betas)
    beta = np.asarray(betas)
    n_ctx = mixer.w.shape[0]
    post = np.full((n_ctx, B), 1.0 / B)                              # the posterior over betas per mixing context
    bits = dict(linear=0.0, product=0.0, bayes_over_beta=0.0, bayes_over_grid2=0.0)
    bits_beta = np.zeros(B)
    grid2 = np.array([(bc, br) for bc in BETA_C for br in BETA_R])   # (K, 2)
    K = len(grid2)
    post2 = np.full((n_ctx, K), 1.0 / K)
    bits_grid2 = np.zeros(K)
    used = np.zeros(n_ctx)
    probs = np.zeros(len(experts))
    for c in seq.tolist():
        preds = [e.predict(s) for e in experts]
        dists, seen = [], []
        for e, pr in zip(experts, preds):
            d, ok = e.dist(pr) if isinstance(e, Chain) else e.dist(pr[0], pr[1], pr[2])
            dists.append(d); seen.append(ok)
        D = np.stack(dists)
        probs[:] = D[:, c]
        ctx = s.cls() * 9 + chain.deepest(preds[0])
        p_lin, w = mixer.mix(ctx, probs)
        bits["linear"] -= math.log2(p_lin)
        sm = np.array(seen)
        logD = np.log(np.clip(D[sm], 1e-12, None))
        wg = w[sm] / w[sm].sum()
        base = wg @ logD                                             # (V,) the log of the geometric mixture, unnormalised
        Z = beta[:, None] * base[None, :]                            # (B, V)
        Z -= Z.max(1, keepdims=True)
        G = np.exp(Z); G /= G.sum(1, keepdims=True)
        pb = np.clip(G[:, c], 1e-12, None)                           # (B,) each beta's probability of the character
        bits_beta -= np.log2(pb)
        pi = post[ctx]
        p_mix = float(pi @ pb)
        bits["bayes_over_beta"] -= math.log2(p_mix)
        a = alpha if switch == "fixed" else 1.0 / (used[ctx] + 2)
        post[ctx] = (1 - a) * (pi * pb / p_mix) + a / B
        used[ctx] += 1
        # the 2-D grid: chain exponent x rest exponent (the rest = Bayesian-weighted geometric mean of the other seen experts)
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
        bits_grid2 -= np.log2(p2)
        pi2 = post2[ctx]
        pm2 = float(pi2 @ p2)
        bits["bayes_over_grid2"] -= math.log2(pm2)
        post2[ctx] = (1 - a) * (pi2 * p2 / pm2) + a / K
        pr_ = np.exp(logD.sum(0) - logD.sum(0).max()); pr_ /= pr_.sum()
        bits["product"] -= math.log2(max(pr_[c], 1e-12))
        mixer.update(ctx, probs, p_lin)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        s.push(c)
    n = len(seq)
    out = {k: v / n for k, v in bits.items()}
    out["geometric_by_beta"] = {str(b): float(x / n) for b, x in zip(betas, bits_beta)}
    out["best_beta"] = float(betas[int(np.argmin(bits_beta))])
    out["posterior_mean_beta_by_ctx"] = {int(i): float(post[i] @ beta) for i in range(n_ctx) if used[i] > 1000}
    k = int(np.argmin(bits_grid2))
    out["grid2_best"] = dict(beta_chain=float(grid2[k, 0]), beta_rest=float(grid2[k, 1]), bits=float(bits_grid2[k] / n))
    out["grid2_by_point"] = {f"{bc},{br}": float(x / n) for (bc, br), x in zip(grid2.tolist(), bits_grid2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_chars", type=int, default=300000)
    ap.add_argument("--test_chars", type=int, default=60000)
    ap.add_argument("--switch", default="fixed", choices=("fixed", "decay"), help="switching prior of the grid posteriors")
    ap.add_argument("--out", default=str(HERE / "mix_temperature_bayes.json"))
    args = ap.parse_args()
    if args.switch == "decay" and args.out == str(HERE / "mix_temperature_bayes.json"):
        args.out = str(HERE / "mix_temperature_bayes_decay.json")
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)
    if args.train_chars:
        text = text[:args.train_chars]
    if args.test_chars:
        test = test[:args.test_chars]
    print(f"A1: {len(text)} training characters, {len(test)} held out; V = {V}; betas {BETAS}", flush=True)
    t0 = time.time()
    experts = build_experts(V, alphabet)
    _tb, _ts, s, mixer = run_stream(experts, text, V, alphabet, track_solo=False)
    print(f"   experts + mixer trained  [{time.time() - t0:.0f}s]", flush=True)
    res = run_temperatures(experts, test, V, mixer, s, switch=args.switch)
    print(f"   held-out online bits/char: linear {res['linear']:.3f}; geometric by beta: " +
          ", ".join(f"{b} -> {x:.3f}" for b, x in res["geometric_by_beta"].items()) +
          f"; Bayes over beta {res['bayes_over_beta']:.3f}; product {res['product']:.3f}; best beta {res['best_beta']}  [{time.time() - t0:.0f}s]",
          flush=True)
    g = res["grid2_best"]
    print(f"   2-D grid (chain exponent x rest exponent): Bayes over the {len(BETA_C) * len(BETA_R)} points {res['bayes_over_grid2']:.3f}; "
          f"best point in hindsight beta_chain {g['beta_chain']}, beta_rest {g['beta_rest']} -> {g['bits']:.3f}", flush=True)
    json.dump(dict(V=V, n_train=int(len(text)), n_test=int(len(test)), betas=BETAS, switch=args.switch, result=res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
