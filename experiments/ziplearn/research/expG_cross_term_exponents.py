"""Experiment G -- the two closed-form CROSS-TERM exponent rules on the Latin stream, per mixing context (item 1 of
notes/gradient_free_mixing_and_features.md section 3; theses T6 and I.4 hinge on it).

Question. E34's geometric mixture needs exponents. The closed forms measured so far (the per-context grid posterior of
expA, the temperature of mix_temperature_bayes, the SSE table of expC) compute NO cross-term between experts from data.
Two closed forms do, and are linear solves on counted second moments, not gradients:
  C2  Lindley's supra-Bayesian model, Genest & Zidek 1986 eq. (4.1) = linear discriminant analysis on the experts'
      stretched forecasts: w = Xi^-1 (mu_1 - mu_0), from the class-conditional means and the pooled within-class
      covariance of the experts' stretched predictions, all accumulated by counting. Needs the outcome. The pool it
      defines for a binary event is  logit P(y=1 | s) = ln(pi_1/pi_0) + w' (s - (mu_1 + mu_0)/2), the log-odds pool
      with exponents w -- the exponents are NOT on the simplex; their sum is whatever the covariance says.
  C3  Satopaa, Pemantle & Ungar 2016 (JASA 111(516), "Modeling probability forecasts via information diversity"), the
      partial-information "revealed aggregator": on the probit scale, delta_m = Var(s_m)/(1 + Var(s_m)),
      X_m = (s_m - E s_m) sqrt(1 - delta_m), Sigma = Cov(X), w = Sigma^-1 diag(Sigma), E[Z|X] = w'X,
      Var(Z|X) = 1 - diag(Sigma)' w, P(y=1|X) = Phi((E[Z|X] - theta) / sqrt(Var(Z|X))) with theta = Phi^-1(1 - 1/V) the
      structural base rate of a one-vs-rest event. From the forecasts ALONE (no outcome), under the reliability
      assumption Cov(X_m, Z) = Var(X_m) (each forecast is a calibrated conditional expectation) -- which the KT count
      tables violate per context (research/round1_checks.txt); that violation is what the number measures.

THE BINARY EVENT (stated, as the task asks). PAQ mixes binary decisions; our experts emit 70-way distributions. The
binarisation used here is ONE-VS-REST: at position t every character c is a binary event y_{t,c} = 1[x_t = c] with
expert m's forecast p_m(c) and stretched forecast s_m(c) = ln(p_m(c)/(1 - p_m(c))) (C2, PAQ's stretch) or Phi^-1(p_m(c))
(C3, the probit the SPU model is written in). One position contributes ONE positive event (the character that came)
and V - 1 = 69 negative events (the characters that did not). Per mixing context the class-conditional sums, outer-
product sums and counts are accumulated over these (position, c) pairs. Two poolings of the within-class covariance
are reported: EQUAL, Xi = (C_1 + C_0)/2 (research/pool_math_check.py's form; the positive class weighted up to parity),
and COUNT, Xi = (n_1 C_1 + n_0 C_0)/(n_1 + n_0) (standard LDA, 69:1 toward the negatives). Feature scale: the stretch
(the faithful C2), and additionally ln p_m(c) (so that the solved exponents live on the SAME functional form as expA's
grid, P(c) ~ prod_m p_m(c)^{w_m}, and can be compared with the grid's hindsight-best exponents per context).

Mixing over the 70 characters, from the same exponents (two POOL SCALES -- the exponents of a pool are only defined
with its scale: the probability pool P(c) ~ prod_m p_m(c)^{w_m} (expA's grid) and the ODDS pool
P(c) ~ prod_m (p_m(c)/(1 - p_m(c)))^{w_m} = exp(sum_m w_m stretch(p_m(c))) (PAQ's, renormalised) are different
distributions for the same w; a second grid posterior + oracle is run on the odds scale, `gridS_*`, and the
stretch-LDA exponents are scored on BOTH pools, `C2_stretch_exp` (odds) and `C2_stretch_onlog` (probability);
`C2_stretch_cnt_onlog` is the count-pooled LDA on the probability pool):
  sig   P(c) ~ sigma(b + w' s(c)), the eq.-4.1 binary posterior of every one-vs-rest event, renormalised over c
        (the faithful reading; the intercept b = ln(n_1/n_0) - w'(mu_1 + mu_0)/2 is counted too);
  exp   P(c) ~ exp(w' s(c)), the geometric pool on the feature scale (intercept-free; equals prod p_m^{w_m}/Z on the
        ln p scale).
Accumulation is CAUSAL: it starts in the training slice (subsampled: every --acc_every-th training position, stated in
the JSON; the experts still update at every position) and continues at every test position, the weights of the
current mixing context re-solved after each of its first 16 test visits and then every 8th visit ("counts up to the
current position", with a lag of at most 8 of that context's samples -- for speed; contexts have 10^2-10^4 visits). A TEST-ONLY variant starts
the accumulators at zero at the first test character (the grid posterior's footing: it too starts uniform at the test
start); a context with no positive event yet falls back to the chain alone. A HINDSIGHT variant solves once from the
whole test slice's statistics (an oracle, like the grid's hindsight-best point; charged below).
Regularisation, fixed before the run and not tuned: Xi (and Sigma) + 0.01 * (trace/M) * I; forecasts clipped to
[1e-7, 1 - 1e-7] before stretching; output probabilities floored at 1e-6 per character before renormalising (so a
failed rule costs at most ~20 bits/char, not infinity); Var(Z|X) floored at 0.01 (the number of contexts at the floor
is recorded).

Comparators in the SAME pass on the SAME slice (paired, per-10k-block SE): the chain alone; E34's linear Mixer over the
run's experts; expA's per-context grid posterior (GridPosterior(K, 36, share 0.01) -- the ledger's 2.152 for
chain + word on the 63-point grid); the hindsight-best fixed grid point per mixing context (36 independent argmins
over the per-context cumulative bits -- the comparator round1_checks asks for) and globally.

PARAMETER COST, charged: the grid posterior pays at most 36 log2 K bits (the dominance inequality) = 36 * log2(63)/60000
= 0.0036 bits/char for K = 63 (0.0050 for K = 336); the per-context hindsight grid oracle is charged the same
36 log2 K. The causal C2/C3 rules carry NO prior charge in the prequential sense (plug-in estimators pay through their
early predictions); their state is 36 x (2 (M + M(M+1)/2) + 2) sufficient statistics for C2 and 36 x (M + M(M+1)/2 + 1)
for C3, i.e. M exponents + 1 intercept per context as free quantities. The hindsight C2 (fitted on the test slice) is
charged 36 (M + 1) * (1/2) log2(60000) = 8.0 bits per real parameter (Rissanen) = 0.0058 bits/char for M = 2, 0.0077
for M = 3, and stated beside its number.

Runs: chain ALONE (`--runs chain`, a separate invocation: M = 1, the diagnostic -- the exponent the LDA gives ONE
expert, which is 1 iff its Gaussian model is self-consistent on this stream); chain + word (the ledger's pair); chain + word + match (the task's three); chain + order4 (the redundant control:
order4 is one of the chain's own backoff levels, so its exact exponent given the chain is ~0 and a cross-term rule must
find it -- expA's grid found b = 0.3 for it).
Slice: E34's --train_chars convention (the first 300 000 characters of the concatenated training books), the first
60 000 characters of the held-out book, online. Baselines on this slice from the note: chain 2.229, A2-min 2.168,
chain + word per-context grid posterior 2.152.

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/expG_cross_term_exponents.py
        (CPU, ~3.5 min measured: 202 s) -> experiments/ziplearn/research/expG_cross_term_exponents.json (+ .log)
    ... --runs chain --out experiments/ziplearn/research/expG_cross_term_exponents_chain.json   (55 s; the M = 1 diagnostic)
The JSON merges runs across invocations (same n_train / n_test), so runs may be split.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from e34 import Chain, Mixer, Stream, build_experts                 # noqa: E402
from textlm import load_corpus                                     # noqa: E402
from expA_exponent_grid import GridPosterior, N_CTX                # noqa: E402

BLOCK = 10000
P_CLIP = 1e-7          # forecasts clipped to [P_CLIP, 1 - P_CLIP] before stretching
P_FLOOR = 1e-6         # output floor per character before renormalising
RIDGE = 0.01           # relative ridge on Xi / Sigma
CVAR_FLOOR = 0.01      # floor on Var(Z|X) in C3
RESOLVE_EVERY = 8      # after a context's first 16 test visits, its weights are re-solved every 8th visit (a lag of <= 8 samples)
LN2 = math.log(2)


# ----------------------------------------------------------------------------------------------------------------------
# the two closed-form rules as per-context sufficient statistics + a solve
# ----------------------------------------------------------------------------------------------------------------------
class LDAStats:
    """C2: per-context class-conditional sums of an M-vector feature over one-vs-rest events. add() takes the (M, V)
    feature matrix of one position and the character that came."""

    def __init__(self, n_ctx, M):
        self.M = M
        self.S1 = np.zeros((n_ctx, M)); self.Q1 = np.zeros((n_ctx, M, M)); self.n1 = np.zeros(n_ctx)
        self.S0 = np.zeros((n_ctx, M)); self.Q0 = np.zeros((n_ctx, M, M)); self.n0 = np.zeros(n_ctx)

    def add(self, ctx, F, x):
        f1 = F[:, x]
        o1 = np.outer(f1, f1)
        self.S1[ctx] += f1; self.Q1[ctx] += o1; self.n1[ctx] += 1
        self.S0[ctx] += F.sum(1) - f1; self.Q0[ctx] += F @ F.T - o1; self.n0[ctx] += F.shape[1] - 1

    def solve(self, ctx, pooling="equal"):
        """(w, b) of logit P(y=1|s) = b + w's, or None when the context has no positive event yet."""
        n1, n0 = self.n1[ctx], self.n0[ctx]
        if n1 < 1:
            return None
        mu1 = self.S1[ctx] / n1; mu0 = self.S0[ctx] / n0
        C1 = self.Q1[ctx] / n1 - np.outer(mu1, mu1)
        C0 = self.Q0[ctx] / n0 - np.outer(mu0, mu0)
        Xi = 0.5 * (C1 + C0) if pooling == "equal" else (n1 * C1 + n0 * C0) / (n1 + n0)
        Xi = Xi + (RIDGE * np.trace(Xi) / self.M + 1e-9) * np.eye(self.M)
        w = np.linalg.solve(Xi, mu1 - mu0)
        b = math.log(n1 / n0) - float(w @ (mu1 + mu0)) / 2
        return w, b


class SPUStats:
    """C3: per-context sums of the probit forecasts over ALL one-vs-rest events (no outcome)."""

    def __init__(self, n_ctx, M):
        self.M = M
        self.S = np.zeros((n_ctx, M)); self.Q = np.zeros((n_ctx, M, M)); self.n = np.zeros(n_ctx)
        self.floor_hits = 0; self.solves = 0

    def add(self, ctx, F):
        self.S[ctx] += F.sum(1); self.Q[ctx] += F @ F.T; self.n[ctx] += F.shape[1]

    def solve(self, ctx, theta=None):
        """(mu, scale, w, cvar) or None when the context has no events yet. X_m = (s_m - mu_m) * scale_m; with
        theta given, mu_m = -theta / scale_m is the MODEL-implied mean of a calibrated forecaster (then X_m = s_m scale_m +
        theta and the rule with w = e_m reproduces expert m exactly); otherwise mu_m is the empirical mean."""
        n = self.n[ctx]
        if n < 1:
            return None
        mu = self.S[ctx] / n
        C = self.Q[ctx] / n - np.outer(mu, mu)
        v = np.clip(np.diag(C), 0, None)
        delta = v / (1 + v)
        sc = np.sqrt(1 - delta)
        Sig = C * np.outer(sc, sc)
        Sig = Sig + (RIDGE * np.trace(Sig) / self.M + 1e-9) * np.eye(self.M)
        d = np.diag(Sig).copy()
        w = np.linalg.solve(Sig, d)
        cvar = 1.0 - float(d @ w)
        self.solves += 1
        if cvar < CVAR_FLOOR:
            self.floor_hits += 1; cvar = CVAR_FLOOR
        if theta is not None:
            mu = -theta / sc
        return mu, sc, w, cvar


def stretch_of(logD):
    """logit of the clipped forecasts from their logs, (M, V)."""
    p = np.clip(np.exp(logD), P_CLIP, 1 - P_CLIP)
    return np.log(p) - np.log1p(-p)


def probit_of(logD):
    p = np.clip(np.exp(logD), P_CLIP, 1 - P_CLIP)
    return ndtri(p)


def mix_sig(F, w, b):
    z = b + w @ F
    e = np.exp(-np.abs(z))
    p = np.where(z > 0, 1 / (1 + e), e / (1 + e))
    p = np.maximum(p, P_FLOOR)
    return p / p.sum()


def mix_exp(F, w):
    z = w @ F
    z = z - z.max()
    p = np.maximum(np.exp(z), P_FLOOR)
    return p / p.sum()


def mix_spu(F, mu, sc, w, cvar, theta):
    X = (F - mu[:, None]) * sc[:, None]
    z = (w @ X - theta) / math.sqrt(cvar)
    p = np.maximum(ndtr(z), P_FLOOR)
    return p / p.sum()


# ----------------------------------------------------------------------------------------------------------------------
# the stream loop (mirrors e34.run_stream / run_mixtures, with the feature hook)
# ----------------------------------------------------------------------------------------------------------------------
def dists_of(experts, preds):
    out = []
    for e, pr in zip(experts, preds):
        d, _ok = e.dist(pr) if isinstance(e, Chain) else e.dist(pr[0], pr[1], pr[2])
        out.append(d)
    return np.stack(out)


def update_all(experts, preds, c):
    for e, pr in zip(experts, preds):
        if isinstance(e, Chain):
            e.update(pr, c)
        else:
            e.update(pr[0], c)


def train_pass(experts, mixer, seq, V, alphabet, acc_every, lda_s, lda_l, spu):
    """E34's training pass (experts + linear Mixer learn at every position) with the C2/C3 accumulators fed at every
    acc_every-th position."""
    s = Stream(V, alphabet)
    chain = experts[0]
    probs = np.zeros(len(experts))
    n_acc = 0
    for t, c in enumerate(seq.tolist()):
        preds = [e.predict(s) for e in experts]
        for i, (e, pr) in enumerate(zip(experts, preds)):
            probs[i] = e.prob(pr, c) if isinstance(e, Chain) else e.prob(pr[0], pr[1], pr[2], c)
        ctx = s.cls() * 9 + chain.deepest(preds[0])
        p_lin, _w = mixer.mix(ctx, probs)
        mixer.update(ctx, probs, p_lin)
        if t % acc_every == 0:
            logD = np.log(np.clip(dists_of(experts, preds), 1e-300, None))
            Fs = stretch_of(logD)
            lda_s.add(ctx, Fs, c)
            lda_l.add(ctx, logD, c)
            spu.add(ctx, probit_of(logD))
            n_acc += 1
        update_all(experts, preds, c)
        s.push(c)
    return s, n_acc


def test_pass(experts, mixer, stream, test, V, grid, theta, lda_s, lda_l, spu):
    """One online pass; every predictor scored from the same expert distributions at every position."""
    M = len(experts)
    K = len(grid)
    chain = experts[0]
    s = stream
    n = len(test)
    nb = math.ceil(n / BLOCK)
    names = ["chain", "linear", "grid_by_ctx", "gridS_by_ctx",
             "C2_stretch_sig", "C2_stretch_exp", "C2_stretch_sig_cnt", "C2_stretch_onlog", "C2_stretch_cnt_onlog", "C2_log_exp",
             "C2_stretch_exp_testonly", "C2_log_exp_testonly", "C3_probit", "C3_probit_testonly", "C3_probit_emp"]
    blk = {k: np.zeros(nb) for k in names}
    cum_ctx_blk = np.zeros((nb, N_CTX, K))                             # per-block, per-context bits of every grid point (probability pool)
    cumS_ctx_blk = np.zeros((nb, N_CTX, K))                            # the same grid on the ODDS scale: P(c) ~ exp(sum_m w_m stretch(p_m(c)))
    post_ctx = GridPosterior(K, N_CTX, alpha=0.01)
    postS_ctx = GridPosterior(K, N_CTX, alpha=0.01)
    # test-only accumulators
    lda_s_t = LDAStats(N_CTX, M); lda_l_t = LDAStats(N_CTX, M); spu_t = SPUStats(N_CTX, M)
    # the solved weights per context (from the training statistics, then re-solved after each visit)
    W = {}
    def resolve(ctx):
        W[("s_eq", ctx)] = lda_s.solve(ctx, "equal")
        W[("s_cnt", ctx)] = lda_s.solve(ctx, "count")
        W[("l_eq", ctx)] = lda_l.solve(ctx, "equal")
        W[("s_eq_t", ctx)] = lda_s_t.solve(ctx, "equal")
        W[("l_eq_t", ctx)] = lda_l_t.solve(ctx, "equal")
        W[("spu", ctx)] = spu.solve(ctx, theta)
        W[("spu_t", ctx)] = spu_t.solve(ctx, theta)
        W[("spu_emp", ctx)] = spu.solve(ctx)
    for ctx in range(N_CTX):
        resolve(ctx)
    e_chain = np.zeros(M); e_chain[0] = 1.0
    stored_logD = np.zeros((n, M, V), dtype=np.float32)
    stored_ctx = np.zeros(n, dtype=np.int16)
    probs = np.zeros(M)
    visits = np.zeros(N_CTX)
    for t, c in enumerate(test.tolist()):
        b = t // BLOCK
        preds = [e.predict(s) for e in experts]
        D = dists_of(experts, preds)
        logD = np.log(np.clip(D, 1e-300, None))
        probs[:] = D[:, c]
        ctx = s.cls() * 9 + chain.deepest(preds[0])
        visits[ctx] += 1
        stored_logD[t] = logD; stored_ctx[t] = ctx
        # baselines
        blk["chain"][b] -= math.log2(probs[0])
        p_lin, _w = mixer.mix(ctx, probs)
        blk["linear"][b] -= math.log2(p_lin)
        mixer.update(ctx, probs, p_lin)
        # the grid
        L = grid @ logD                                                # (K, V)
        L -= L.max(1, keepdims=True)
        G = np.exp(L); G /= G.sum(1, keepdims=True)
        pg = np.clip(G[:, c], 1e-300, None)
        blk["grid_by_ctx"][b] += post_ctx.code(ctx, pg)
        cum_ctx_blk[b, ctx] -= np.log2(pg)
        Fs = stretch_of(logD); Fp = probit_of(logD)
        LS = grid @ Fs
        LS -= LS.max(1, keepdims=True)
        GS = np.exp(LS); GS /= GS.sum(1, keepdims=True)
        pgS = np.clip(GS[:, c], 1e-300, None)
        blk["gridS_by_ctx"][b] += postS_ctx.code(ctx, pgS)
        cumS_ctx_blk[b, ctx] -= np.log2(pgS)
        # the closed-form rules, from the weights solved BEFORE seeing this character
        def score(key, name, mixfn):
            wb = W[key]
            if wb is None:
                p = mix_exp(logD, e_chain)                             # chain alone
            else:
                p = mixfn(wb)
            blk[name][b] -= math.log2(p[c])
        score(("s_eq", ctx), "C2_stretch_sig", lambda wb: mix_sig(Fs, wb[0], wb[1]))
        score(("s_eq", ctx), "C2_stretch_exp", lambda wb: mix_exp(Fs, wb[0]))
        score(("s_cnt", ctx), "C2_stretch_sig_cnt", lambda wb: mix_sig(Fs, wb[0], wb[1]))
        score(("s_eq", ctx), "C2_stretch_onlog", lambda wb: mix_exp(logD, wb[0]))
        score(("s_cnt", ctx), "C2_stretch_cnt_onlog", lambda wb: mix_exp(logD, wb[0]))
        score(("l_eq", ctx), "C2_log_exp", lambda wb: mix_exp(logD, wb[0]))
        score(("s_eq_t", ctx), "C2_stretch_exp_testonly", lambda wb: mix_exp(Fs, wb[0]))
        score(("l_eq_t", ctx), "C2_log_exp_testonly", lambda wb: mix_exp(logD, wb[0]))
        score(("spu", ctx), "C3_probit", lambda wb: mix_spu(Fp, *wb, theta))
        score(("spu_t", ctx), "C3_probit_testonly", lambda wb: mix_spu(Fp, *wb, theta))
        score(("spu_emp", ctx), "C3_probit_emp", lambda wb: mix_spu(Fp, *wb, theta))
        # learn: accumulate, re-solve this context, update the experts
        lda_s.add(ctx, Fs, c); lda_l.add(ctx, logD, c); spu.add(ctx, Fp)
        lda_s_t.add(ctx, Fs, c); lda_l_t.add(ctx, logD, c); spu_t.add(ctx, Fp)
        if visits[ctx] <= 16 or visits[ctx] % RESOLVE_EVERY == 0:
            resolve(ctx)
        update_all(experts, preds, c)
        s.push(c)
    sizes = np.array([min(BLOCK, n - i * BLOCK) for i in range(nb)])
    # the grid oracles from the per-context cumulative bits
    cum_ctx = cum_ctx_blk.sum(0)                                       # (N_CTX, K)
    best_ctx = cum_ctx.argmin(1)                                       # hindsight-best point per context
    blk["grid_oracle_ctx"] = np.array([cum_ctx_blk[i, np.arange(N_CTX), best_ctx].sum() for i in range(nb)])
    cum_all = cum_ctx.sum(0)
    best_all = int(cum_all.argmin())
    blk["grid_oracle_global"] = cum_ctx_blk[:, :, best_all].sum(1)
    cumS_ctx = cumS_ctx_blk.sum(0)
    bestS_ctx = cumS_ctx.argmin(1)
    blk["gridS_oracle_ctx"] = np.array([cumS_ctx_blk[i, np.arange(N_CTX), bestS_ctx].sum() for i in range(nb)])
    bestS_all = int(cumS_ctx.sum(0).argmin())
    blk["gridS_oracle_global"] = cumS_ctx_blk[:, :, bestS_all].sum(1)
    # hindsight C2 (weights from the whole test slice, per context), scored on the same positions
    blk["C2_stretch_exp_hindsight"] = np.zeros(nb); blk["C2_stretch_onlog_hindsight"] = np.zeros(nb); blk["C2_log_exp_hindsight"] = np.zeros(nb)
    hind = {}
    for ctx in range(N_CTX):
        idx = np.nonzero(stored_ctx == ctx)[0]
        if len(idx) == 0:
            continue
        logD_c = stored_logD[idx].astype(np.float64)                   # (n_c, M, V)
        xs = test[idx].astype(np.int64)
        pc_ = np.clip(np.exp(logD_c), P_CLIP, 1 - P_CLIP)
        Fs_c = np.log(pc_) - np.log1p(-pc_)
        for name, F, mixfn in (("C2_stretch_exp_hindsight", Fs_c, "exp"),
                               ("C2_log_exp_hindsight", logD_c, "exp")):
            st = LDAStats(1, M)
            f1 = F[np.arange(len(idx)), :, xs]                          # (n_c, M)
            st.S1[0] = f1.sum(0); st.Q1[0] = f1.T @ f1; st.n1[0] = len(idx)
            st.S0[0] = F.sum(2).sum(0) - f1.sum(0)
            st.Q0[0] = np.einsum("nmv,nkv->mk", F, F) - f1.T @ f1; st.n0[0] = len(idx) * (V - 1)
            w, bb = st.solve(0, "equal")
            hind[(name, ctx)] = (w, bb)
            z = np.einsum("m,nmv->nv", w, F)
            if mixfn == "sig":
                z = z + bb
                p = np.where(z > 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))
            else:
                z = z - z.max(1, keepdims=True); p = np.exp(z)
            p = np.maximum(p, P_FLOOR); p /= p.sum(1, keepdims=True)
            bits_pos = -np.log2(p[np.arange(len(idx)), xs])
            np.add.at(blk[name], idx // BLOCK, bits_pos)
            if name == "C2_stretch_exp_hindsight":                     # the same exponents on the probability pool
                z2 = np.einsum("m,nmv->nv", w, logD_c); z2 -= z2.max(1, keepdims=True)
                p2 = np.maximum(np.exp(z2), P_FLOOR); p2 /= p2.sum(1, keepdims=True)
                np.add.at(blk["C2_stretch_onlog_hindsight"], idx // BLOCK, -np.log2(p2[np.arange(len(idx)), xs]))
    return blk, sizes, cum_ctx, best_ctx, best_all, W, hind, visits, (lda_s, lda_l, spu, lda_s_t, lda_l_t, spu_t), (bestS_ctx, bestS_all)


def summarize(tag, blk, sizes, ref="chain"):
    n = sizes.sum()
    tot = {k: float(v.sum() / n) for k, v in blk.items()}
    diffs = {}
    for k, v in blk.items():
        if k == ref:
            continue
        d = v / sizes - blk[ref] / sizes
        diffs[k] = dict(mean=float((d * sizes).sum() / n), se_blocks=float(d.std(ddof=1) / math.sqrt(len(d))),
                        n_blocks_negative=int((d < 0).sum()), per_block=[float(x) for x in d])
    return tot, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_chars", type=int, default=300000)
    ap.add_argument("--test_chars", type=int, default=60000)
    ap.add_argument("--acc_every", type=int, default=4, help="feed the C2/C3 accumulators at every k-th TRAINING position (the experts update at every position)")
    ap.add_argument("--runs", nargs="+", default=["word", "word+match", "order4"])
    ap.add_argument("--out", default=str(HERE / "expG_cross_term_exponents.json"))
    args = ap.parse_args()
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)[:args.train_chars]
    test = test[:args.test_chars]
    theta = float(ndtri(1 - 1 / V))
    print(f"Experiment G: {len(text)} training characters (prefix), {len(test)} held out (online); V = {V}; "
          f"accumulators fed at every {args.acc_every}th training position and every test position; theta = {theta:.3f}", flush=True)
    a_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    report = dict(V=V, n_train=int(len(text)), n_test=int(len(test)), acc_every=args.acc_every, block=BLOCK, theta=theta,
                  ridge=RIDGE, p_clip=P_CLIP, p_floor=P_FLOOR, cvar_floor=CVAR_FLOOR,
                  binary_event="one-vs-rest: y_{t,c} = 1[x_t = c] for every character c at every position; 1 positive + 69 negative events per position",
                  runs={})
    for run in args.runs:
        t0 = time.time()
        others = [] if run == "chain" else run.split("+")
        ex = build_experts(V, alphabet)
        experts = [ex[0]] + [next(e for e in ex if e.name == o) for o in others]
        names = [e.name for e in experts]
        M = len(experts)
        if M == 1:
            grid = np.array([(a,) for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]])   # the chain's own temperature
        elif M == 2:
            b_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            grid = np.array([(a, b) for a in a_grid for b in b_grid])
        else:
            b1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]; b2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            grid = np.array([(a, x, y) for a in a_grid for x in b1 for y in b2])
        K = len(grid)
        mixer = Mixer(M, N_CTX)
        lda_s = LDAStats(N_CTX, M); lda_l = LDAStats(N_CTX, M); spu = SPUStats(N_CTX, M)
        s, n_acc = train_pass(experts, mixer, text, V, alphabet, args.acc_every, lda_s, lda_l, spu)
        t_train = time.time() - t0
        blk, sizes, cum_ctx, best_ctx, best_all, W, hind, visits, accs, (bestS_ctx, bestS_all) = test_pass(experts, mixer, s, test, V, grid, theta, lda_s, lda_l, spu)
        t_test = time.time() - t0 - t_train
        tot, diffs = summarize(run, blk, sizes)
        _tot, diffs_vs_grid = summarize(run, blk, sizes, ref="grid_by_ctx")
        n = int(sizes.sum())
        cost = dict(grid_posterior_bound_bits_per_char=N_CTX * math.log2(K) / n,
                    grid_oracle_ctx_bits_per_char=N_CTX * math.log2(K) / n,
                    C2_hindsight_bits_per_char=N_CTX * (M + 1) * 0.5 * math.log2(n) / n,
                    causal_rules="no prior charge (plug-in, prequential); state = per context 2(M + M(M+1)/2 + 1) sums (C2), M + M(M+1)/2 + 1 sums (C3)")
        # per-context weights at the end of the pass, for the contexts visited >= 500 times in the test
        per_ctx = {}
        for ctx in range(N_CTX):
            if visits[ctx] < 500:
                continue
            row = dict(visits=int(visits[ctx]), cls=ctx // 9, deep=ctx % 9,
                       grid_oracle=grid[best_ctx[ctx]].tolist(), gridS_oracle=grid[bestS_ctx[ctx]].tolist())
            for key, label in (("s_eq", "C2_stretch"), ("s_cnt", "C2_stretch_cnt"), ("l_eq", "C2_log"),
                               ("s_eq_t", "C2_stretch_testonly"), ("l_eq_t", "C2_log_testonly")):
                wb = W[(key, ctx)]
                row[label] = dict(w=[float(x) for x in wb[0]], b=float(wb[1]), sum_w=float(wb[0].sum())) if wb else None
            for key, label in (("spu", "C3"), ("spu_t", "C3_testonly")):
                wb = W[(key, ctx)]
                row[label] = dict(w=[float(x) for x in wb[2]], delta=[float(1 - x * x) for x in wb[1]], cvar=float(wb[3])) if wb else None
            hw = hind.get(("C2_log_exp_hindsight", ctx))
            row["C2_log_hindsight"] = dict(w=[float(x) for x in hw[0]]) if hw else None
            hw = hind.get(("C2_stretch_exp_hindsight", ctx))
            row["C2_stretch_hindsight"] = dict(w=[float(x) for x in hw[0]]) if hw else None
            per_ctx[str(ctx)] = row
        # does the linear solve recover the grid's structure?  compare C2_log weights with the grid oracle per context
        ctxs = [int(k) for k in per_ctx]
        A_or = np.array([grid[best_ctx[c]] for c in ctxs])
        A_l = np.array([W[("l_eq", c)][0] for c in ctxs])
        A_h = np.array([hind[("C2_log_exp_hindsight", c)][0] for c in ctxs])
        wts = np.array([visits[c] for c in ctxs], dtype=float)
        agree = {}
        for lab, A in (("C2_log_causal", A_l), ("C2_log_hindsight", A_h)):
            d = A - A_or
            agree[lab] = dict(mean_abs_diff_per_expert=[float(x) for x in (np.abs(d) * wts[:, None]).sum(0) / wts.sum()],
                              visit_weighted_mean_w=[float(x) for x in (A * wts[:, None]).sum(0) / wts.sum()],
                              visit_weighted_mean_sum_w=float(((A.sum(1)) * wts).sum() / wts.sum()),
                              corr_per_expert=[float(np.corrcoef(A[:, j], A_or[:, j])[0, 1]) if A_or[:, j].std() > 0 and A[:, j].std() > 0 else None for j in range(M)])
        agree["grid_oracle_visit_weighted_mean"] = [float(x) for x in (A_or * wts[:, None]).sum(0) / wts.sum()]
        agree["grid_oracle_visit_weighted_mean_sum"] = float(((A_or.sum(1)) * wts).sum() / wts.sum())
        A_orS = np.array([grid[bestS_ctx[c]] for c in ctxs])
        agree["gridS_oracle_visit_weighted_mean"] = [float(x) for x in (A_orS * wts[:, None]).sum(0) / wts.sum()]
        agree["gridS_oracle_visit_weighted_mean_sum"] = float(((A_orS.sum(1)) * wts).sum() / wts.sum())
        Aw = np.array([W[("s_eq", c)][0] for c in ctxs])
        agree["C2_stretch_visit_weighted_mean_w"] = [float(x) for x in (Aw * wts[:, None]).sum(0) / wts.sum()]
        agree["C2_stretch_visit_weighted_mean_sum_w"] = float(((Aw.sum(1)) * wts).sum() / wts.sum())
        for lab, ref in (("C2_stretch_vs_grid_oracle", A_or), ("C2_stretch_vs_gridS_oracle", A_orS)):
            dd = Aw - ref
            agree[lab] = dict(mean_abs_diff_per_expert=[float(x) for x in (np.abs(dd) * wts[:, None]).sum(0) / wts.sum()],
                              corr_per_expert=[float(np.corrcoef(Aw[:, j], ref[:, j])[0, 1]) if ref[:, j].std() > 0 and Aw[:, j].std() > 0 else None for j in range(M)])
        Ac = np.array([W[("spu", c)][2] for c in ctxs])
        agree["C3_visit_weighted_mean_w"] = [float(x) for x in (Ac * wts[:, None]).sum(0) / wts.sum()]
        agree["C3_cvar_floor_hits"] = dict(hits=int(accs[2].floor_hits), solves=int(accs[2].solves))
        agree["C3_testonly_cvar_floor_hits"] = dict(hits=int(accs[5].floor_hits), solves=int(accs[5].solves))
        agree["C3_cvar_visit_weighted_mean"] = float(sum(W[("spu", c)][3] * visits[c] for c in ctxs) / wts.sum())
        row = dict(experts=names, M=M, K=K, grid_shape=("a" if M == 1 else "a x b" if M == 2 else "a x b_word x b_match"), n_train_acc_samples=int(n_acc),
                   bits=tot, diff_vs_chain={k: {kk: vv for kk, vv in v.items() if kk != "per_block"} for k, v in diffs.items()},
                   diff_vs_grid_by_ctx={k: dict(mean=v["mean"], se_blocks=v["se_blocks"]) for k, v in diffs_vs_grid.items()},
                   per_block_bits={k: [float(x) for x in v / sizes] for k, v in blk.items()},
                   grid_best_global=grid[best_all].tolist(), grid_oracle_ctx_points={str(c): grid[best_ctx[c]].tolist() for c in range(N_CTX) if visits[c] > 0},
                   parameter_cost=cost, structure_agreement=agree, per_context=per_ctx,
                   seconds=dict(train=t_train, test=t_test))
        report["runs"][run] = row
        order = ["chain", "linear", "grid_by_ctx", "grid_oracle_ctx", "grid_oracle_global", "gridS_by_ctx", "gridS_oracle_ctx", "gridS_oracle_global",
                 "C2_stretch_exp", "C2_stretch_onlog", "C2_stretch_cnt_onlog", "C2_stretch_sig", "C2_stretch_sig_cnt", "C2_log_exp",
                 "C2_stretch_exp_testonly", "C2_log_exp_testonly", "C2_stretch_exp_hindsight", "C2_stretch_onlog_hindsight", "C2_log_exp_hindsight",
                 "C3_probit", "C3_probit_testonly", "C3_probit_emp"]
        print(f"  [{run}] experts {names}; K = {K}; train {t_train:.0f}s, test {t_test:.0f}s; {n_acc} training samples fed to the accumulators", flush=True)
        for k in order:
            d = diffs.get(k)
            extra = f"  vs chain {d['mean']:+.4f} (SE {d['se_blocks']:.4f}, {d['n_blocks_negative']}/{len(sizes)} blocks < 0)" if d else ""
            dg = diffs_vs_grid.get(k)
            extra += f"  vs grid_by_ctx {dg['mean']:+.4f} (SE {dg['se_blocks']:.4f})" if dg else ""
            print(f"      {k:28s} {tot[k]:.4f}{extra}", flush=True)
        print(f"      costs: grid posterior bound {cost['grid_posterior_bound_bits_per_char']:.4f}, C2 hindsight {cost['C2_hindsight_bits_per_char']:.4f} bits/char; "
              f"grid best global {grid[best_all].tolist()}", flush=True)
        print(f"      visit-weighted mean exponents over contexts >= 500 visits: grid oracle {np.round(agree['grid_oracle_visit_weighted_mean'], 2).tolist()} "
              f"(sum {agree['grid_oracle_visit_weighted_mean_sum']:.2f}); odds-grid oracle {np.round(agree['gridS_oracle_visit_weighted_mean'], 2).tolist()} "
              f"(sum {agree['gridS_oracle_visit_weighted_mean_sum']:.2f}); C2_log causal {np.round(agree['C2_log_causal']['visit_weighted_mean_w'], 2).tolist()} "
              f"(sum {agree['C2_log_causal']['visit_weighted_mean_sum_w']:.2f}); C2_log hindsight {np.round(agree['C2_log_hindsight']['visit_weighted_mean_w'], 2).tolist()}; "
              f"C2_stretch {np.round(agree['C2_stretch_visit_weighted_mean_w'], 2).tolist()} (sum {agree['C2_stretch_visit_weighted_mean_sum_w']:.2f}); "
              f"C3 {np.round(agree['C3_visit_weighted_mean_w'], 2).tolist()}; C3 cvar floor hits {agree['C3_cvar_floor_hits']}", flush=True)
        print(f"      |C2_stretch - grid oracle| per expert {np.round(agree['C2_stretch_vs_grid_oracle']['mean_abs_diff_per_expert'], 3).tolist()} (corr {np.round(agree['C2_stretch_vs_grid_oracle']['corr_per_expert'], 2).tolist()}); "
              f"|C2_stretch - odds-grid oracle| {np.round(agree['C2_stretch_vs_gridS_oracle']['mean_abs_diff_per_expert'], 3).tolist()} (corr {np.round(agree['C2_stretch_vs_gridS_oracle']['corr_per_expert'], 2).tolist()})", flush=True)
        print(f"      |C2_log - grid oracle| per expert: causal {np.round(agree['C2_log_causal']['mean_abs_diff_per_expert'], 3).tolist()}, "
              f"hindsight {np.round(agree['C2_log_hindsight']['mean_abs_diff_per_expert'], 3).tolist()}; corr across contexts: causal {agree['C2_log_causal']['corr_per_expert']}", flush=True)
    if Path(args.out).exists():                                        # merge: runs may be split across invocations
        try:
            old = json.load(open(args.out))
            if old.get("n_train") == report["n_train"] and old.get("n_test") == report["n_test"]:
                old["runs"].update(report["runs"]); old.update({k: v for k, v in report.items() if k != "runs"}); report = old
        except Exception:
            pass
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
