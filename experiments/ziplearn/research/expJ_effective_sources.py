"""Experiment J (research, 2026-09-22) -- the effective number of independent sources among E34's 18 context experts,
measured from DATA before any mixer is built, and ONE closed-form use of it. No learned parameter, no gradient.

Apparatus: experiments/ziplearn/e34.py unchanged (Stream, Expert, Chain, Mixer, build_experts, run_stream). The
standard text slice: the first 300 000 characters of the concatenated training books (E34's --train_chars convention),
the first 60 000 characters of the held-out book (De Bello Civili), online (prequential) bits/char. Same-slice references
(research/): chain 2.229, linear 2.211 and E34 geometric 2.191 (expC_e34_mixtures_300k.json), A2-min 2.168
(mix_temperature_bayes_decay.json, replicated in this pass so that it has per-block paired differences).

PART 1 -- the measurement (the note's I.3 item 2, "the information structure of the 18 experts"). During the LAST
`--stat_chars` characters of the training pass (default 100 000: characters 200 000-300 000, the experts and the E34
Mixer learning exactly as run_stream does, so the trained state at the end is the same as every other script's), record
per position and per mixing context ctx = cls * 9 + deepest order (E34's 36 contexts):
  (i)   each expert's code length l_m(t) = -log2 D_m(c_t) and the KT unigram's l_u(t)   -> mean code length per (ctx, m),
        and the GAIN over the unigram, gain_m(ctx) = mean(l_u - l_m | m's context seen);
  (ii)  the pairwise symmetrised KL between the experts' predictive distributions, S_ab(t) = sum_x (D_a - D_b)(log2 D_a
        - log2 D_b) -> mean per ctx: Heskes 1997's ambiguity matrix (his eq. 4-5, A = 1/2 sum_ab w_a w_b S_ab), and the
        check his decomposition invites: predicted code length of the UNIFORM simplex geometric pool over all 18 = mean_m
        l_m - A, against the pool's measured in-window code length;
  (iii) the pairwise conditional mutual information between the experts' context KEYS given the next character, by
        plug-in counting over the positions where both contexts were seen (expB's r_cmi recipe, applied to the KEYS
        instead of the argmax predictions), normalised r = I(K_m; K_j | C) / min(H(K_m|C), H(K_j|C)), WITH a shuffle null
        (K_j permuted within each value of C; one draw, seed 0) subtracted from both numerator and ceiling, because a
        plug-in CMI between near-unique keys saturates at the ceiling by chance alone (pairs whose ceiling minus null is
        below 0.05 bits or below 20 % of the ceiling are reported as UNRESOLVABLE and excluded);
        also expB's own argmax-CMI r (cardinality <= 71, resolvable), and the correlation rho_mj of the two experts'
        code lengths over both-seen positions (Clemen & Winkler's literal setting: correlated errors).
From these, per context and per pair, the Clemen-Winkler-style effective number of sources: for M sources with mean
pairwise redundancy rbar, k_eff = M / (1 + (M - 1) rbar) (per pair: 2 / (1 + r)); and for the code-length correlation
matrix R also the general form 1' R^-1 1 (ridge 0.05; the precision of the best linear combination of unit-variance
correlated errors), clipped to [1, M]. CAVEAT (round1_checks, check:mathematics on T5): Clemen & Winkler's k/(1+(k-1)rho)
is a precision-of-linear-combination result and S_w of a log pool is scale-dependent, so "k_eff" here is a COUNT-BASED
CANDIDATE for the exponent mass, tested below, not a theorem about the log pool.

PART 2 -- ONE closed-form use, per mixing context, mixed geometrically over the SEEN experts (expD's form, no prior
division: p(c) proportional to prod_m D_m(c)^e_m):
    e_m = MASS(ctx, seen set) * a_m / sum_{j seen} a_j,
    a_m = max(gain_m(ctx), 0) * (1 - max_{j seen, j != m} r(m, j))                   ["sym", the task's formula]
with MASS = k_eff of the seen set from the SAME redundancy matrix r, for r in {argmax-CMI, key-CMI, code-length
correlation}. The control that answers "does the ALLOCATION carry information, or only the mass?" is `flat` at the SAME
mass (expD's nb_flat logic). Secondary rows: `gain` (no redundancy factor) isolates the redundancy factor from the
code-length gain; `asym` applies the discount only against STRONGER experts (max over j with gain_j > gain_m -- a greedy
one-pass stand-in for Allard eq. 17's sequential conditioning, because a SYMMETRIC r cannot say which of a nested pair
is the function of the other: r(chain, order8) = 1 zeroes both under `sym`); fixed masses 1 and 2 and the R^-1 mass
separate the mass question from the allocation question. If every a_m is 0 on the seen set the row falls back to flat
(counted). Baselines in the SAME pass: chain, linear (E34 Mixer), E34 geometric (Bayesian exponents over seen), raw
product, and A2-min (the 42-point (beta_chain, beta_rest) per-context grid posterior with decaying switch, from
mix_temperature_bayes.py). Every number is reported as a paired difference vs the chain with its SE over the six
10k-character blocks (check_text_slice_noise's convention).

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/expJ_effective_sources.py
        -> experiments/ziplearn/research/expJ_effective_sources.json      (CPU, ~4 min)
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
from textlm import load_corpus                                        # noqa: E402
from e34 import Chain, Expert, Mixer, Stream, build_experts, run_stream   # noqa: E402

N_CTX = 4 * 9
LN2 = math.log(2)
BLOCK = 10000
BETA_C = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)                             # A2-min's grid (mix_temperature_bayes.py)
BETA_R = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
RIDGE = 0.05
R_NAMES = ("argmax_cmi", "key_cmi", "codelen_corr")
ALLOCS = ("flat", "gain") + tuple(f"{k}_{r}" for r in R_NAMES for k in ("sym", "asym"))
MASSES = ("m1", "m2") + tuple(f"keff_{r}" for r in R_NAMES) + ("keff_inv_codelen",)


# ----------------------------------------------------------------------------------------------------- the measurement

def chain_key_and_argmax(chain, keys):
    """The chain's context KEY = (deepest seen order, that order's context tuple); its argmax = that level's mode."""
    for r, k in zip(range(chain.R, -1, -1), keys):
        totals, counts = chain.tables[r]
        if totals.get(k, 0):
            cnt = counts[k]
            return (r, k), max(cnt, key=cnt.get)
    return (0, ()), chain.V


def record_pass(experts, seq, V, alphabet, s, mixer):
    """E34's learning pass over `seq` (experts + Mixer learn as run_stream does), recording what Part 1 needs."""
    M = len(experts)
    chain = experts[0]
    uni_expert = next(e for e in experts if isinstance(e, Expert))
    n = len(seq)
    key_ids = [dict() for _ in range(M)]
    K = np.zeros((n, M), dtype=np.int64)
    A = np.zeros((n, M), dtype=np.int16)
    Lb = np.zeros((n, M))                                                # code lengths in bits
    Lu = np.zeros(n)
    Sn = np.zeros((n, M), dtype=bool)
    ctxs = np.zeros(n, dtype=np.int16)
    S_sum = np.zeros((N_CTX, M, M))                                      # symmetrised KL (bits), summed per ctx
    unif_bits = np.zeros(N_CTX)                                          # measured uniform mass-1 pool over ALL 18
    probs = np.zeros(M)
    for t, c in enumerate(seq.tolist()):
        preds = [e.predict(s) for e in experts]
        dists = []
        for i, (e, pr) in enumerate(zip(experts, preds)):
            if isinstance(e, Chain):
                d, ok = e.dist(pr)
                key, am = chain_key_and_argmax(e, pr)
            else:
                d, ok = e.dist(pr[0], pr[1], pr[2])
                key = pr[0]
                am = max(pr[2], key=pr[2].get) if pr[1] else V
            dists.append(d); Sn[t, i] = ok; A[t, i] = am
            kid = key_ids[i].get(key)
            if kid is None:
                kid = len(key_ids[i]); key_ids[i][key] = kid
            K[t, i] = kid
        D = np.stack(dists)
        probs[:] = D[:, c]
        L2 = np.log2(np.clip(D, 1e-300, None))
        Lb[t] = -L2[:, c]
        u = uni_expert.uni[c] + 0.5
        Lu[t] = -math.log2(u / (uni_expert.uni_total + V / 2))
        deep = chain.deepest(preds[0])
        ctx = s.cls() * 9 + deep
        ctxs[t] = ctx
        PL = (D * L2).sum(1)
        Cross = D @ L2.T
        S_sum[ctx] += PL[:, None] + PL[None, :] - Cross - Cross.T
        lg = L2.mean(0); lg -= lg.max(); g = np.exp2(lg); g /= g.sum()
        unif_bits[ctx] -= math.log2(max(g[c], 1e-300))
        # learn exactly as run_stream does
        p_lin, _w = mixer.mix(ctx, probs)
        mixer.update(ctx, probs, p_lin)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        s.push(c)
    return dict(K=K, A=A, L=Lb, Lu=Lu, seen=Sn, ctx=ctxs, C=seq.astype(np.int64), S_sum=S_sum, unif_bits=unif_bits,
                n_keys=[len(d) for d in key_ids])


def _H(codes):
    _u, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def _compact(x):
    return np.unique(x, return_inverse=True)[1].astype(np.int64)


def cmi_matrix(Kmat, C, seen, V, rng, unseen_code=None):
    """Plug-in normalised conditional MI between the columns of Kmat given C over BOTH-SEEN positions, with a
    shuffle null. Returns (r_raw, r_corr [NaN = unresolvable], I, I_null, ceiling, n_both). `seen` masks the
    positions where a column's context was seen (an unseen expert's argmax is V; its key exists but carries no count)."""
    n, M = Kmat.shape
    r_raw = np.full((M, M), np.nan); r_cor = np.full((M, M), np.nan)
    I = np.zeros((M, M)); I0 = np.zeros((M, M)); ceil = np.zeros((M, M)); nb = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            m = seen[:, i] & seen[:, j]
            k = int(m.sum())
            nb[i, j] = nb[j, i] = k
            if k < 200:
                continue
            ki, kj, c = _compact(Kmat[m, i]), _compact(Kmat[m, j]), _compact(C[m])
            nc = c.max() + 1; nj = kj.max() + 1
            Hc = _H(c)
            Hic = _H(ki * nc + c) - Hc
            Hjc = _H(kj * nc + c) - Hc
            Hijc = _H((ki * nj + kj) * nc + c) - Hc
            cmi = max(Hic + Hjc - Hijc, 0.0)
            # the null: kj permuted within each value of c
            q = rng.permutation(k)
            s0 = np.argsort(c, kind="stable"); sq = np.argsort(c[q], kind="stable")
            kj0 = np.empty_like(kj); kj0[s0] = kj[q[sq]]
            Hij0 = _H((ki * nj + kj0) * nc + c) - Hc
            cmi0 = max(Hic + Hjc - Hij0, 0.0)
            den = min(Hic, Hjc)
            I[i, j] = I[j, i] = cmi; I0[i, j] = I0[j, i] = cmi0; ceil[i, j] = ceil[j, i] = den
            if den > 1e-9:
                r_raw[i, j] = r_raw[j, i] = min(cmi / den, 1.0)
            head = den - cmi0
            if head >= 0.05 and head >= 0.2 * den:
                r_cor[i, j] = r_cor[j, i] = float(np.clip((cmi - cmi0) / head, 0.0, 1.0))
    return r_raw, r_cor, I, I0, ceil, nb


def codelen_corr(L, seen, sel, C=None):
    """Correlation matrix of the experts' code lengths over both-seen positions among `sel`; NaN where < 200.
    With C given, each expert's code length is first centred by its mean over the positions with the same next
    character (its seen positions), so the correlation is CONDITIONAL on the target -- a rare character costs every
    expert more bits, and that common cause is not redundancy (the count-based analogue of the CMI given C)."""
    Ls, Ss = L[sel].copy(), seen[sel].astype(np.float64)
    if C is not None:
        Cs = C[sel]
        for v in np.unique(Cs):
            m = Cs == v
            cnt = Ss[m].sum(0)
            mean = (Ss[m] * Ls[m]).sum(0) / np.maximum(cnt, 1)
            Ls[m] -= mean[None, :]
    SL = Ss * Ls
    n_ij = Ss.T @ Ss
    s1 = SL.T @ Ss                                                        # sum over both-seen of l_i   (i, j)
    s2 = (SL * Ls).T @ Ss                                                 # sum over both-seen of l_i^2 (i, j)
    s12 = SL.T @ SL                                                       # sum over both-seen of l_i l_j
    with np.errstate(invalid="ignore", divide="ignore"):
        mi = s1 / n_ij; mj = s1.T / n_ij
        vi = s2 / n_ij - mi ** 2; vj = s2.T / n_ij - mj ** 2
        rho = (s12 / n_ij - mi * mj) / np.sqrt(np.clip(vi, 1e-12, None) * np.clip(vj, 1e-12, None))
    rho = np.where(n_ij >= 200, rho, np.nan)
    np.fill_diagonal(rho, 1.0)
    return rho, n_ij


def keff_cw(R):
    """M / (1 + (M-1) rbar) with rbar the nan-mean of the off-diagonal (NaN pairs excluded); M = R.shape[0]."""
    M = R.shape[0]
    if M <= 1:
        return 1.0
    off = R[~np.eye(M, dtype=bool)]
    good = off[~np.isnan(off)]
    rbar = float(np.clip(good.mean(), 0.0, 1.0)) if good.size else 0.0
    return M / (1.0 + (M - 1) * rbar)


def keff_inv(R, ridge=RIDGE):
    """1' (R + ridge I)^-1 1 on a correlation-like matrix (NaN -> 0), clipped to [1, M]."""
    M = R.shape[0]
    if M <= 1:
        return 1.0
    Rf = np.nan_to_num(np.clip(R, 0.0, 1.0)); np.fill_diagonal(Rf, 1.0)
    try:
        x = np.linalg.solve(Rf + ridge * np.eye(M), np.ones(M))
        return float(np.clip(x.sum(), 1.0, M))
    except np.linalg.LinAlgError:
        return float(M)


def build_stats(rec, V, names, min_ctx, rng):
    """Per-context statistics (with the global ones as the fallback for contexts with < min_ctx positions)."""
    M = len(names)
    L, Lu, seen, ctx, C = rec["L"], rec["Lu"], rec["seen"], rec["ctx"], rec["C"]
    n = len(C)
    n_ctx = np.bincount(ctx, minlength=N_CTX).astype(np.float64)
    all_ = np.ones(n, dtype=bool)
    out = dict(n_ctx=n_ctx.tolist(), min_ctx=min_ctx, fallback_ctx=[int(i) for i in range(N_CTX) if n_ctx[i] < min_ctx])

    def gains(sel):
        Ss = seen[sel].astype(np.float64)
        ns = np.maximum(Ss.sum(0), 1)
        g_seen = (Ss * (Lu[sel][:, None] - L[sel])).sum(0) / ns
        mean_all = L[sel].mean(0)
        return g_seen, mean_all, Ss.sum(0)

    t0 = time.time()
    g_glob, mean_glob, ns_glob = gains(all_)
    ra_glob = cmi_matrix(rec["A"].astype(np.int64), C, seen, V, rng)
    rk_glob = cmi_matrix(rec["K"], C, seen, V, rng)
    rc_glob, nij_glob = codelen_corr(L, seen, all_, C)                    # partialled on the next character (used)
    rc_raw_glob, _ = codelen_corr(L, seen, all_)                          # raw (reported only)
    print(f"   global CMI matrices [{time.time() - t0:.0f}s]", flush=True)
    G = np.zeros((N_CTX, M)); MEAN = np.zeros((N_CTX, M)); NS = np.zeros((N_CTX, M))
    RA = np.zeros((N_CTX, M, M)); RK = np.zeros((N_CTX, M, M)); RC = np.zeros((N_CTX, M, M))
    RA_raw = np.zeros((N_CTX, M, M)); RK_raw = np.zeros((N_CTX, M, M))
    resolv = np.zeros(N_CTX)
    for i in range(N_CTX):
        sel = ctx == i
        if n_ctx[i] < min_ctx:
            G[i], MEAN[i], NS[i] = g_glob, mean_glob, ns_glob
            RA[i], RK[i], RC[i] = ra_glob[1], rk_glob[1], rc_glob
            RA_raw[i], RK_raw[i] = ra_glob[0], rk_glob[0]
            continue
        G[i], MEAN[i], NS[i] = gains(sel)
        ra = cmi_matrix(rec["A"][sel].astype(np.int64), C[sel], seen[sel], V, rng)
        rk = cmi_matrix(rec["K"][sel], C[sel], seen[sel], V, rng)
        rc, _nij = codelen_corr(L, seen, sel, C)
        RA[i], RK[i], RC[i] = ra[1], rk[1], rc
        RA_raw[i], RK_raw[i] = ra[0], rk[0]
        off = ~np.eye(M, dtype=bool)
        resolv[i] = float(np.mean(~np.isnan(rk[1][off])))
    print(f"   per-context matrices [{time.time() - t0:.0f}s]", flush=True)
    # the symmetrised-KL matrix and Heskes' check
    S = rec["S_sum"] / np.maximum(n_ctx, 1)[:, None, None]
    S_glob = rec["S_sum"].sum(0) / n
    amb = S.sum((1, 2)) / (2 * M * M)
    amb_glob = S_glob.sum() / (2 * M * M)
    unif_meas = rec["unif_bits"] / np.maximum(n_ctx, 1)
    unif_glob = rec["unif_bits"].sum() / n
    heskes = dict(global_=dict(mean_solo=float(mean_glob.mean()), ambiguity=float(amb_glob), predicted=float(mean_glob.mean() - amb_glob),
                               measured_uniform_pool=float(unif_glob), exact_ambiguity=float(mean_glob.mean() - unif_glob)),
                  by_ctx={int(i): dict(n=int(n_ctx[i]), mean_solo=float(MEAN[i].mean()), ambiguity=float(amb[i]),
                                       predicted=float(MEAN[i].mean() - amb[i]), measured_uniform_pool=float(unif_meas[i]))
                          for i in range(N_CTX) if n_ctx[i] >= min_ctx})
    out.update(gain=G, mean_codelen=MEAN, n_seen=NS, RA=RA, RK=RK, RC=RC, RA_raw=RA_raw, RK_raw=RK_raw, S=S, S_glob=S_glob,
               ra_glob=ra_glob, rk_glob=rk_glob, rc_glob=rc_glob, rc_raw_glob=rc_raw_glob, nij_glob=nij_glob, g_glob=g_glob, mean_glob=mean_glob,
               ns_glob=ns_glob, unigram_mean=float(Lu.mean()), heskes=heskes, resolvable_frac_by_ctx=resolv.tolist())
    return out


# ------------------------------------------------------------------------------------------------ the closed-form use

def exponent_rows(seen, gain_ctx, R_ctx, nS):
    """All (allocation x mass) exponent rows for one position. Returns (E (n_rows, M), fallback flags, masses)."""
    M = len(seen)
    idx = np.where(seen)[0]
    g = np.maximum(gain_ctx[idx], 0.0)
    allocs = {"flat": np.ones(nS), "gain": g.copy()}
    masses = {"m1": 1.0, "m2": 2.0}
    eye = np.eye(nS, dtype=bool)
    stronger = g[None, :] > g[:, None]                                    # [m, j]: j stronger than m
    for name, R in zip(R_NAMES, R_ctx):
        Rs = R[np.ix_(idx, idx)].copy()
        Rs[eye] = np.nan
        masses[f"keff_{name}"] = keff_cw(np.where(eye, 1.0, Rs))
        if name == "codelen_corr":
            masses["keff_inv_codelen"] = keff_inv(np.where(eye, 1.0, Rs))
        with np.errstate(all="ignore"):
            mx_sym = np.nanmax(np.where(np.isnan(Rs), -np.inf, Rs), axis=1)
            mx_sym = np.where(np.isfinite(mx_sym), mx_sym, 0.0)
            Ra = np.where(stronger, Rs, np.nan)
            mx_asym = np.nanmax(np.where(np.isnan(Ra), -np.inf, Ra), axis=1)
            mx_asym = np.where(np.isfinite(mx_asym), mx_asym, 0.0)
        allocs[f"sym_{name}"] = g * (1.0 - np.clip(mx_sym, 0.0, 1.0))
        allocs[f"asym_{name}"] = g * (1.0 - np.clip(mx_asym, 0.0, 1.0))
    E = np.zeros((len(ALLOCS) * len(MASSES), M))
    fb = np.zeros(len(ALLOCS), dtype=bool)
    row = 0
    for ai, an in enumerate(ALLOCS):
        a = allocs[an]
        tot = a.sum()
        if tot <= 1e-12:
            a = np.ones(nS); tot = float(nS); fb[ai] = True
        a = a / tot
        for mn in MASSES:
            E[row, idx] = masses[mn] * a
            row += 1
    return E, fb, masses


def test_pass(experts, seq, V, s, mixer, stats):
    M = len(experts)
    chain = experts[0]
    n = len(seq)
    nblk = math.ceil(n / BLOCK)
    nrows = len(ALLOCS) * len(MASSES)
    bits = np.zeros((nrows, nblk)); ent = np.zeros(nrows); mass_sum = np.zeros(nrows); esum = np.zeros((nrows, M))
    base_names = ("chain", "linear", "geo_bayes", "product", "a2min")
    base = {k: np.zeros(nblk) for k in base_names}
    fb_count = np.zeros(len(ALLOCS))
    keff_sum = {k: 0.0 for k in MASSES}
    nS_sum = 0.0
    grid2 = np.array([(bc, br) for bc in BETA_C for br in BETA_R])
    Kg = len(grid2)
    post2 = np.full((N_CTX, Kg), 1.0 / Kg); used = np.zeros(N_CTX)
    probs = np.zeros(M)
    G, RA, RK, RC = stats["gain"], stats["RA"], stats["RK"], stats["RC"]
    t0 = time.time()
    for t, c in enumerate(seq.tolist()):
        b = t // BLOCK
        preds = [e.predict(s) for e in experts]
        dists, seen = [], []
        for e, pr in zip(experts, preds):
            d, ok = e.dist(pr) if isinstance(e, Chain) else e.dist(pr[0], pr[1], pr[2])
            dists.append(d); seen.append(ok)
        D = np.stack(dists); seen = np.array(seen)
        probs[:] = D[:, c]
        logD = np.log(np.clip(D, 1e-300, None))
        deep = chain.deepest(preds[0]); ctx = s.cls() * 9 + deep
        p_lin, w = mixer.mix(ctx, probs)
        base["linear"][b] -= math.log2(p_lin)
        base["chain"][b] -= math.log2(probs[0])
        wg = w[seen] / w[seen].sum()
        lg = wg @ logD[seen]; lg -= lg.max(); gb = np.exp(lg); gb /= gb.sum()
        base["geo_bayes"][b] -= math.log2(max(gb[c], 1e-300))
        lp = logD[seen].sum(0); lp -= lp.max(); pr_ = np.exp(lp); pr_ /= pr_.sum()
        base["product"][b] -= math.log2(max(pr_[c], 1e-300))
        # A2-min (mix_temperature_bayes.run_temperatures, switch = decay)
        rest = seen.copy(); rest[0] = False
        if rest.any():
            wr = w[rest] / w[rest].sum(); log_rest = wr @ logD[rest]
        else:
            log_rest = np.zeros(V)
        Z2 = grid2[:, :1] * logD[0][None, :] + grid2[:, 1:] * log_rest[None, :]
        Z2 -= Z2.max(1, keepdims=True); G2 = np.exp(Z2); G2 /= G2.sum(1, keepdims=True)
        p2 = np.clip(G2[:, c], 1e-12, None); pi2 = post2[ctx]; pm2 = float(pi2 @ p2)
        base["a2min"][b] -= math.log2(pm2)
        a = 1.0 / (used[ctx] + 2); post2[ctx] = (1 - a) * (pi2 * p2 / pm2) + a / Kg; used[ctx] += 1
        # the closed-form rows
        nS = int(seen.sum()); nS_sum += nS
        E, fb, masses = exponent_rows(seen, G[ctx], (RA[ctx], RK[ctx], RC[ctx]), nS)
        fb_count += fb
        for k in MASSES:
            keff_sum[k] += masses[k]
        logits = E @ logD
        mx = logits.max(1, keepdims=True); z = np.exp(logits - mx); Zs = z.sum(1, keepdims=True)
        logp = logits - mx - np.log(Zs)
        bits[:, b] -= logp[:, c] / LN2
        p = z / Zs
        ent -= (p * logp).sum(1) / LN2
        mass_sum += E.sum(1); esum += E
        # learn online as E34 does
        mixer.update(ctx, probs, p_lin)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        s.push(c)
        if (t + 1) % 20000 == 0:
            print(f"   {t + 1} test characters [{time.time() - t0:.0f}s]", flush=True)
    sizes = np.array([min(BLOCK, n - i * BLOCK) for i in range(nblk)], dtype=np.float64)
    return dict(bits=bits, ent=ent / n, mass=mass_sum / n, mean_exp=esum / n, base=base, sizes=sizes, fallback=fb_count / n,
                keff_mean={k: v / n for k, v in keff_sum.items()}, n_seen_mean=nS_sum / n)


def paired(blk, ref, sizes):
    """Paired difference (blk - ref) per block, its weighted mean and the SE over blocks."""
    d = (blk - ref) / sizes
    return float((blk - ref).sum() / sizes.sum()), float(d.std(ddof=1) / math.sqrt(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_chars", type=int, default=300_000)
    ap.add_argument("--stat_chars", type=int, default=100_000, help="the LAST stat_chars of the training slice are recorded")
    ap.add_argument("--test_chars", type=int, default=60_000)
    ap.add_argument("--min_ctx", type=int, default=1000, help="contexts with fewer recorded positions use the global statistics")
    ap.add_argument("--out", default=str(HERE / "expJ_effective_sources.json"))
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)[:args.train_chars]
    test = test[:args.test_chars]
    n_pre = len(text) - args.stat_chars
    print(f"expJ: {len(text)} training characters (prefix), statistics from the last {args.stat_chars}, {len(test)} held out; V = {V}", flush=True)
    experts = build_experts(V, alphabet)
    names = [e.name for e in experts]
    M = len(experts)
    _tb, _ts, s, mixer = run_stream(experts, text[:n_pre], V, alphabet, track_solo=False)
    print(f"   first {n_pre} characters trained [{time.time() - t0:.0f}s]", flush=True)
    rec = record_pass(experts, text[n_pre:], V, alphabet, s, mixer)
    print(f"   recording pass done [{time.time() - t0:.0f}s]; distinct keys per expert: " + ", ".join(f"{nm} {k}" for nm, k in zip(names, rec["n_keys"])), flush=True)
    stats = build_stats(rec, V, names, args.min_ctx, rng)
    t_stats = time.time() - t0
    print(f"   statistics built [{t_stats:.0f}s]", flush=True)

    # ---- report Part 1
    off = ~np.eye(M, dtype=bool)
    ra_raw, ra, *_r = stats["ra_glob"]; rk_raw, rk, Ik, I0k, ceilk, nbk = stats["rk_glob"]; rc = stats["rc_glob"]; rc_raw = stats["rc_raw_glob"]
    print(f"\nPART 1 -- {args.stat_chars} recorded positions; unigram code length {stats['unigram_mean']:.3f} bits/char")
    print("   mean code length (all positions) | gain over the unigram where seen | seen fraction:")
    for i, nm in enumerate(names):
        print(f"      {nm:>9}: {stats['mean_glob'][i]:.3f} | {stats['g_glob'][i]:+.3f} | {stats['ns_glob'][i] / args.stat_chars:.2f}")
    S = stats["S_glob"]
    print("   symmetrised KL (bits), global, selected pairs: " + ", ".join(
        f"{a}-{b} {S[names.index(a), names.index(b)]:.2f}" for a, b in (("chain", "order8"), ("chain", "order4"), ("chain", "word"), ("chain", "match"), ("word", "match"), ("order1", "line"), ("skip3", "skip4"))))
    h = stats["heskes"]["global_"]
    print(f"   Heskes: mean solo {h['mean_solo']:.3f}, quadratic ambiguity A_approx {h['ambiguity']:.3f} -> predicted uniform pool {h['predicted']:.3f}; measured {h['measured_uniform_pool']:.3f} "
          f"(exact ambiguity = mean solo - measured = {h['mean_solo'] - h['measured_uniform_pool']:.3f})")
    print(f"   key-CMI: resolvable pairs (global) {int(np.sum(~np.isnan(rk[off])) // 2)} of {M * (M - 1) // 2}; argmax-CMI resolvable {int(np.sum(~np.isnan(ra[off])) // 2)}")

    def show(Rm, tag):
        print(f"   {tag}:")
        for i, nm in enumerate(names):
            print(f"      {nm:>9}: " + " ".join("  . " if np.isnan(Rm[i, j]) else f"{Rm[i, j]:.2f}" for j in range(M)))
    show(ra, "r argmax-CMI, shuffle-corrected (global)")
    show(rk_raw, "r key-CMI RAW plug-in (global)")
    show(rk, "r key-CMI shuffle-corrected (global; '.' = unresolvable)")
    show(rc, "rho code-length correlation over both-seen positions, PARTIALLED on the next character (global; used below)")
    print("   rho RAW (not partialled), selected pairs: " + ", ".join(
        f"{a}-{b} {rc_raw[names.index(a), names.index(b)]:.2f}" for a, b in (("chain", "order8"), ("chain", "word"), ("chain", "match"), ("word", "match"), ("order1", "line"), ("skip3", "skip4"))))
    keffs = {tag: dict(cw=keff_cw(Rm), inv=keff_inv(Rm)) for tag, Rm in (("argmax_cmi", ra), ("key_cmi", rk), ("codelen_corr", rc))}
    print("   effective number of sources over all 18 (global): " + "; ".join(f"{k}: CW {v['cw']:.2f}, 1'R^-1 1 {v['inv']:.2f}" for k, v in keffs.items()))
    pairs = (("chain", "order8"), ("chain", "order4"), ("chain", "word"), ("chain", "match"), ("word", "match"), ("order1", "line"), ("skip3", "skip4"), ("word", "prevword"))
    print("   per pair 2/(1+r) [argmax-CMI / key-CMI / codelen]: " + ", ".join(
        f"{a}-{b} " + "/".join("." if np.isnan(Rm[names.index(a), names.index(b)]) else f"{2 / (1 + Rm[names.index(a), names.index(b)]):.2f}" for Rm in (ra, rk, rc))
        for a, b in pairs))
    n_ctx = np.array(stats["n_ctx"])
    print("   per context (ctx = cls*9 + deep; n >= min_ctx): n | k_eff CW argmax / key / codelen | chain gain | word gain | Heskes predicted vs measured")
    keff_ctx = {}
    for i in range(N_CTX):
        if n_ctx[i] < args.min_ctx:
            continue
        kc = dict(argmax_cmi=keff_cw(stats["RA"][i]), key_cmi=keff_cw(stats["RK"][i]), codelen_corr=keff_cw(stats["RC"][i]), inv_codelen=keff_inv(stats["RC"][i]))
        keff_ctx[int(i)] = kc
        hb = stats["heskes"]["by_ctx"][int(i)]
        print(f"      ctx {i:2d} (cls {i // 9}, deep {i % 9}): {int(n_ctx[i]):6d} | {kc['argmax_cmi']:.2f} / {kc['key_cmi']:.2f} / {kc['codelen_corr']:.2f} (inv {kc['inv_codelen']:.2f}) | "
              f"{stats['gain'][i, 0]:+.2f} | {stats['gain'][i, names.index('word')]:+.2f} | {hb['predicted']:.3f} vs {hb['measured_uniform_pool']:.3f}")

    # ---- Part 2
    res = test_pass(experts, test, V, s, mixer, stats)
    sizes = res["sizes"]; n = sizes.sum()
    chain_blk = res["base"]["chain"]
    base_tot = {k: float(v.sum() / n) for k, v in res["base"].items()}
    print(f"\nPART 2 -- held-out online bits/char ({int(n)} characters, {len(sizes)} blocks); [{time.time() - t0:.0f}s]")
    print("   baselines: " + ", ".join(f"{k} {v:.4f}" for k, v in base_tot.items()))
    base_diff = {}
    for k in ("linear", "geo_bayes", "product", "a2min"):
        m_, se = paired(res["base"][k], chain_blk, sizes); base_diff[k] = dict(mean=m_, se=se)
        print(f"      {k} - chain: {m_:+.4f} (SE {se:.4f})")
    print(f"   mean seen experts per position {res['n_seen_mean']:.2f}; mean mass per rule: " + ", ".join(f"{k} {v:.2f}" for k, v in res["keff_mean"].items()))
    print("   fallback-to-flat fraction per allocation: " + ", ".join(f"{a} {f:.3f}" for a, f in zip(ALLOCS, res["fallback"])))
    rows = []
    r = 0
    for an in ALLOCS:
        for mn in MASSES:
            tot = float(res["bits"][r].sum() / n)
            m_, se = paired(res["bits"][r], chain_blk, sizes)
            rows.append(dict(alloc=an, mass=mn, bpc=tot, vs_chain=m_, se=se, entropy=float(res["ent"][r]), gap=float(tot - res["ent"][r]), mean_mass=float(res["mass"][r]),
                             mean_exponent={nm: float(round(res["mean_exp"][r, i], 4)) for i, nm in enumerate(names)}))
            r += 1
    tab = {(x["alloc"], x["mass"]): x for x in rows}
    print("\n   bits/char (vs chain, SE over blocks)  rows = allocation, columns = mass rule")
    print("   " + " " * 22 + "".join(f"{mn:>26}" for mn in MASSES))
    for an in ALLOCS:
        print(f"   {an:>22}" + "".join(f"  {tab[(an, mn)]['bpc']:.4f} ({tab[(an, mn)]['vs_chain']:+.3f}±{tab[(an, mn)]['se']:.3f})" for mn in MASSES))
    print("\n   PRIMARY: the task's formula sym_r at mass keff_r, against flat at the SAME mass (paired vs chain, SE):")
    primary = {}
    for rn in R_NAMES:
        a, f = tab[(f"sym_{rn}", f"keff_{rn}")], tab[("flat", f"keff_{rn}")]
        ia = ALLOCS.index(f"sym_{rn}") * len(MASSES) + MASSES.index(f"keff_{rn}"); i_f = ALLOCS.index("flat") * len(MASSES) + MASSES.index(f"keff_{rn}")
        d_, se_ = paired(res["bits"][ia], res["bits"][i_f], sizes)
        primary[rn] = dict(sym=a, flat=f, sym_minus_flat=dict(mean=d_, se=se_))
        print(f"      {rn:>13}: sym {a['bpc']:.4f} ({a['vs_chain']:+.4f}±{a['se']:.4f}, mass {a['mean_mass']:.2f}) | flat {f['bpc']:.4f} ({f['vs_chain']:+.4f}±{f['se']:.4f}) | sym - flat {d_:+.4f}±{se_:.4f}")
    best = sorted(rows, key=lambda x: x["bpc"])[:6]
    print("   best 6 rows (secondary; selected on this slice): " + "; ".join(f"{x['alloc']}@{x['mass']} {x['bpc']:.4f} (mass {x['mean_mass']:.2f}, gap {x['gap']:+.2f})" for x in best))

    def top_exp(x, k=5):
        me = sorted(x["mean_exponent"].items(), key=lambda kv: -kv[1])[:k]
        return ", ".join(f"{nm} {v:.2f}" for nm, v in me)
    print("   mean exponent per expert (top 5) -- where each rule actually puts its mass:")
    for x in [tab[("flat", "m2")], tab[("gain", "m2")]] + [tab[(f"sym_{rn}", f"keff_{rn}")] for rn in R_NAMES] + [tab[(f"sym_{rn}", "m1")] for rn in R_NAMES] + [tab[(f"asym_{rn}", "m1")] for rn in R_NAMES]:
        print(f"      {x['alloc']:>18}@{x['mass']:<18}: {top_exp(x)}")
    total = time.time() - t0
    print(f"\n[total {total:.0f}s]")

    def r3(x):
        return np.round(np.asarray(x, dtype=np.float64), 4).tolist()

    report = dict(V=V, n_train=int(len(text)), stat_chars=args.stat_chars, n_test=int(n), min_ctx=args.min_ctx, seconds=dict(stats=t_stats, total=total),
                  experts=names, unigram_mean_codelen=stats["unigram_mean"],
                  part1=dict(mean_codelen_global=dict(zip(names, r3(stats["mean_glob"]))), gain_global=dict(zip(names, r3(stats["g_glob"]))),
                             seen_frac_global=dict(zip(names, r3(stats["ns_glob"] / args.stat_chars))), n_keys=dict(zip(names, rec["n_keys"])),
                             symKL_global=r3(S), r_argmax_cmi_raw=r3(ra_raw), r_argmax_cmi=r3(ra), r_key_cmi_raw=r3(rk_raw), r_key_cmi=r3(rk),
                             key_cmi_I=r3(Ik), key_cmi_I_null=r3(I0k), key_cmi_ceiling=r3(ceilk), n_both_seen=r3(nbk), rho_codelen=r3(rc), rho_codelen_raw=r3(rc_raw),
                             keff_global=keffs, keff_by_ctx=keff_ctx, heskes=stats["heskes"], n_ctx=stats["n_ctx"], fallback_ctx=stats["fallback_ctx"],
                             resolvable_frac_key_cmi_by_ctx=stats["resolvable_frac_by_ctx"],
                             by_ctx=dict(gain=r3(stats["gain"]), mean_codelen=r3(stats["mean_codelen"]), symKL=r3(stats["S"]),
                                         r_argmax_cmi=r3(stats["RA"]), r_key_cmi=r3(stats["RK"]), rho_codelen=r3(stats["RC"]))),
                  part2=dict(baselines=base_tot, baseline_vs_chain=base_diff, blocks_chain=r3(chain_blk / sizes),
                             n_seen_mean=res["n_seen_mean"], mass_mean=res["keff_mean"], fallback_frac=dict(zip(ALLOCS, r3(res["fallback"]))),
                             rows=rows, primary=primary, best=best,
                             references_same_slice=dict(chain=2.229, linear=2.211, geometric_e34=2.191, a2min=2.168,
                                                        note="expD's uniform-at-mass-2 (2.161) is on the SPREAD 300k slice (chain 2.030), recomputed here as flat@m2")))
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
