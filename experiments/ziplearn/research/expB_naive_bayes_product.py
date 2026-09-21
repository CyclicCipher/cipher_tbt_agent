"""Experiment B (research, 2026-09-22) -- the naive-Bayes product with the prior divided out, and a redundancy
correction from COUNTS. No learned parameter anywhere.

The question (E34, DESIGN §20). E34's raw product of the seen experts over-sharpens (7.665 bits/char on the held-out
book against the chain's 1.823), the linear (Bayesian) mixture cannot combine evidence (1.820), and PAQ's gain lives in
LEARNED exponents. Theory to test: the over-sharpening is REDUNDANCY, and two further defects of the raw product are
fixable in closed form:
  (i)  the raw product multiplies the PRIOR M times over -- every expert's distribution carries the unigram inside it;
  (ii) E34's KT tables smooth toward UNIFORM, so a sparse context, divided by the prior, votes FOR rare characters.
The exact rule when the contexts k_1..k_M are conditionally independent given the next character c (naive Bayes) is
    P(c | k_1..k_M)  ∝  P_uni(c) · ∏_m P_m(c) / P_uni(c)  =  ∏_m P_m(c) / P_uni(c)^(M-1),
normalised over the V = 70 characters. An expert whose context was never seen has P_m = P_uni and drops out (ratio 1).
The contexts are NOT conditionally independent (order2 ⊂ order4 ⊂ chain), so the same evidence is counted several
times. The correction, with nothing learned: shrink expert m's exponent by its redundancy with the OTHER SEEN experts,
    lambda_m = 1 / (1 + sum_{j seen, j != m} r(m, j)),      P(c) ∝ P_uni(c) · ∏_m (P_m(c) / P_uni(c))^lambda_m,
with r(m, j) in [0, 1] estimated on the training slice by counting, two ways:
    r_agree = the fraction of positions where both contexts were seen at which the two experts' argmax agree;
    r_cmi   = I(A_m; A_j | C) / min(H(A_m | C), H(A_j | C)): the conditional mutual information between the two experts'
              argmax predictions A given the next character C (plug-in counts over the both-seen positions) -- exactly the
              quantity naive Bayes assumes to be zero, normalised to [0, 1].
lambda = 1 for identical experts would double-count; r = 1 gives each of M identical experts 1/M -> their geometric mean =
one expert's worth; r = 0 gives lambda = 1 = the naive-Bayes product. Brackets reported: lambda = 1 (`nb`, independent)
and lambda = 1 / M_seen (`nb_geo`, fully redundant). Two controls: `nb_flat` = the SAME total exponent as `nb_agree`
spread uniformly over the seen experts (does the pairwise matrix's ALLOCATION matter, or only the total?); `nb_ref` = the
reference expert (the shortest prequential code on the training slice; the chain here) keeps exponent 1 and each other
seen expert gets 1 - r_agree(m, ref) (the share of its argmax the reference does not already give).
Smoothing of an expert's table (defect ii): `kt` = E34's Expert.dist (KT toward uniform); `prior` = a Dirichlet prior
centred on the running unigram, P_m(c) = (n_c + (V/2) P_uni(c)) / (N + V/2). The chain's blended backoff is used as is.

Two expert sets, as instructed: DIVERSE = chain + word + match + line (different context functions), and REDUNDANT =
chain + order2 + order4 (both nested inside the chain); plus, if time allows, ALL 18 of E34's library.
Measured, online (prequential) on the first 60k characters of the held-out book after a 300k-character training prefix
(E34's `--train_chars` convention; the full corpus is too slow for a 5-minute script): bits/char of the chain alone
(baseline), each expert solo, the linear mixture (E34's Mixer), E34's raw product and geometric mixture, and the
naive-Bayes variants under both smoothings. Sibling reference at this slice (expA): chain 2.229, linear 2.230.

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/expB_naive_bayes_product.py
        -> experiments/ziplearn/research/expB_naive_bayes_product.json   (CPU, ~3 min)
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
from textlm import load_corpus                                                  # noqa: E402
from e34 import Stream, Expert, Chain, Mixer, build_experts                     # noqa: E402

SETS = {
    "diverse": ["chain", "word", "match", "line"],
    "redundant": ["chain", "order2", "order4"],
    "all18": None,
}
N_CTX = 4 * 9                                                                   # E34's mixing contexts: class x deepest order


def pick(V, alphabet, names):
    ex = build_experts(V, alphabet)
    if names is None:
        return ex
    by = {e.name: e for e in ex}
    return [by[n] for n in names]                                               # the chain stays first (run_stream's convention)


def argmax_of(e, pr, V):
    """The expert's argmax prediction from its counts (V = 'unseen'); the chain: its deepest seen level's counts."""
    if isinstance(e, Chain):
        for r, k in zip(range(e.R, -1, -1), pr):
            totals, counts = e.tables[r]
            if totals.get(k, 0):
                cnt = counts[k]
                return max(cnt, key=cnt.get)
        return V
    _k, N, cnt = pr
    return max(cnt, key=cnt.get) if N else V


def train_pass(experts, seq, V, alphabet):
    """E34's prequential training pass (experts + the linear mixer learn), recording each expert's argmax per position."""
    s, M = Stream(V, alphabet), len(experts)
    mixer = Mixer(M, N_CTX)
    chain = experts[0]
    probs = np.zeros(M)
    solo = np.zeros(M)                                                          # each expert's prequential bits on the slice
    A = np.full((len(seq), M), V, dtype=np.int16)
    for t, c in enumerate(seq.tolist()):
        preds = [e.predict(s) for e in experts]
        for i, (e, pr) in enumerate(zip(experts, preds)):
            probs[i] = e.prob(pr, c) if isinstance(e, Chain) else e.prob(pr[0], pr[1], pr[2], c)
            A[t, i] = argmax_of(e, pr, V)
        solo -= np.log2(probs)
        ctx = s.cls() * 9 + chain.deepest(preds[0])
        p, _w = mixer.mix(ctx, probs)
        mixer.update(ctx, probs, p)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        s.push(c)
    return s, mixer, A, solo / len(seq)


def _H(keys):
    _u, n = np.unique(keys, return_counts=True)
    p = n / n.sum()
    return float(-(p * np.log2(p)).sum())


def redundancy(A, C, V):
    """Pairwise r_agree and r_cmi (module docstring) from the recorded argmax matrix A (n, M) and the truth C (n)."""
    n, M = A.shape
    A, C = A.astype(np.int64), C.astype(np.int64)
    R_agree, R_cmi, both = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            m = (A[:, i] < V) & (A[:, j] < V)
            nb = int(m.sum())
            both[i, j] = both[j, i] = nb / n
            if nb < 100:
                continue
            ai, aj, c = A[m, i], A[m, j], C[m]
            R_agree[i, j] = R_agree[j, i] = float((ai == aj).mean())
            Hc = _H(c)
            Hi, Hj = _H(ai * V + c) - Hc, _H(aj * V + c) - Hc
            Hij = _H((ai * V + aj) * V + c) - Hc
            cmi = max(Hi + Hj - Hij, 0.0)
            den = min(Hi, Hj)
            R_cmi[i, j] = R_cmi[j, i] = min(cmi / den, 1.0) if den > 1e-9 else 1.0
    return R_agree, R_cmi, both


def test_pass(experts, seq, V, alphabet, s, mixer, R_agree, R_cmi, ref):
    """One online pass computing every variant at once; experts and the mixer keep learning (prequential).
    `ref` = the index of the reference expert (shortest training code) for the `nb_ref_*` control."""
    M = len(experts)
    chain = experts[0]
    uni = np.zeros(V)                                                           # the running unigram (same counts every E34 Expert keeps)
    for e in experts:
        if isinstance(e, Expert):
            uni[:] = e.uni
            break
    else:                                                                       # chain alone: its order-0 table
        for ch, n in chain.tables[0][1].get((), {}).items():
            uni[ch] = n
    beta = V / 2
    names = ["linear", "product_kt", "geometric_kt",
             "nb_kt", "nb_prior", "nb_geo_kt", "nb_geo_prior",
             "nb_agree_kt", "nb_agree_prior", "nb_cmi_kt", "nb_cmi_prior",
             "nb_flat_kt", "nb_flat_prior", "nb_ref_kt", "nb_ref_prior"]
    bits = {k: 0.0 for k in names}
    solo = np.zeros(M)
    lam_sum = dict(agree=np.zeros(M), cmi=np.zeros(M), ref=np.zeros(M))
    tot_sum = dict(nb=0.0, agree=0.0, cmi=0.0, ref=0.0)                        # the SUM of exponents, averaged over positions
    seen_count = np.zeros(M)
    probs = np.zeros(M)
    D_kt, D_pr = np.zeros((M, V)), np.zeros((M, V))

    def nb(logu, ratio, lam, c):
        lp = logu + lam @ ratio
        lp -= lp.max()
        p = np.exp(lp)
        return -math.log2(max(p[c] / p.sum(), 1e-300))

    for c in seq.tolist():
        u = (uni + 0.5) / (uni.sum() + beta)                                    # the KT unigram = P_uni
        preds = [e.predict(s) for e in experts]
        seen = np.zeros(M, dtype=bool)
        for i, (e, pr) in enumerate(zip(experts, preds)):
            if isinstance(e, Chain):
                d, ok = e.dist(pr)
                D_kt[i] = d; D_pr[i] = d
            else:
                d, ok = e.dist(pr[0], pr[1], pr[2])
                D_kt[i] = d
                if ok:
                    _k, N, cnt = pr
                    v = beta * u
                    for ch, n_ in cnt.items():
                        v[ch] += n_
                    D_pr[i] = v / (N + beta)
                else:
                    D_pr[i] = u
            seen[i] = ok
        probs[:] = D_kt[:, c]
        solo -= np.log2(np.clip(probs, 1e-300, None))
        ctx = s.cls() * 9 + chain.deepest(preds[0])
        p_lin, w = mixer.mix(ctx, probs)
        bits["linear"] -= math.log2(p_lin)
        # E34's two multiplicative stand-ins over the seen experts
        logD = np.log(np.clip(D_kt[seen], 1e-12, None))
        wg = w[seen] / w[seen].sum()
        g = np.exp(wg @ logD); g /= g.sum()
        bits["geometric_kt"] -= math.log2(max(g[c], 1e-12))
        pr_ = np.exp(logD.sum(0) - logD.sum(0).max()); pr_ /= pr_.sum()
        bits["product_kt"] -= math.log2(max(pr_[c], 1e-12))
        # naive Bayes: the prior divided out; exponents 1 / (1 + redundancy with the other SEEN experts)
        logu = np.log(u)
        sf = seen.astype(np.float64)
        lam_geo = sf / sf.sum()
        lam_agree = sf / (1.0 + R_agree @ sf)
        lam_cmi = sf / (1.0 + R_cmi @ sf)
        lam_flat = sf * (lam_agree.sum() / sf.sum())                            # control: agree's TOTAL, spread uniformly
        lam_ref = sf * (1.0 - R_agree[:, ref]) if seen[ref] else lam_agree      # control: the reference expert keeps 1; others 1 - r(m, ref)
        lam_sum["agree"] += lam_agree; lam_sum["cmi"] += lam_cmi; lam_sum["ref"] += lam_ref; seen_count += sf
        tot_sum["nb"] += sf.sum(); tot_sum["agree"] += lam_agree.sum(); tot_sum["cmi"] += lam_cmi.sum(); tot_sum["ref"] += lam_ref.sum()
        for tag, D in (("kt", D_kt), ("prior", D_pr)):
            ratio = (np.log(np.clip(D, 1e-12, None)) - logu) * sf[:, None]
            bits[f"nb_{tag}"] += nb(logu, ratio, sf, c)
            bits[f"nb_geo_{tag}"] += nb(logu, ratio, lam_geo, c)
            bits[f"nb_agree_{tag}"] += nb(logu, ratio, lam_agree, c)
            bits[f"nb_cmi_{tag}"] += nb(logu, ratio, lam_cmi, c)
            bits[f"nb_flat_{tag}"] += nb(logu, ratio, lam_flat, c)
            bits[f"nb_ref_{tag}"] += nb(logu, ratio, lam_ref, c)
        # learn
        mixer.update(ctx, probs, p_lin)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        uni[c] += 1
        s.push(c)
    n = len(seq)
    return ({k: v / n for k, v in bits.items()}, solo / n,
            {k: (v / np.maximum(seen_count, 1)).tolist() for k, v in lam_sum.items()}, (seen_count / n).tolist(),
            {k: v / n for k, v in tot_sum.items()})


def run_set(name, names, text, test, V, alphabet):
    experts = pick(V, alphabet, names)
    ex_names = [e.name for e in experts]
    t0 = time.time()
    s, mixer, A, solo_train = train_pass(experts, text, V, alphabet)
    t_train = time.time() - t0
    R_agree, R_cmi, both = redundancy(A, text, V)
    t_red = time.time() - t0 - t_train
    ref = int(np.argmin(solo_train))
    res, solo, lam_mean, seen_frac, tot = test_pass(experts, test, V, alphabet, s, mixer, R_agree, R_cmi, ref)
    t_test = time.time() - t0 - t_train - t_red
    row = dict(experts=ex_names, bpc=res, solo={n: float(v) for n, v in zip(ex_names, solo)},
               solo_train={n: float(v) for n, v in zip(ex_names, solo_train)}, reference=ex_names[ref],
               r_agree=R_agree.round(3).tolist(), r_cmi=R_cmi.round(3).tolist(), both_seen_frac=both.round(3).tolist(),
               lambda_mean_agree=dict(zip(ex_names, lam_mean["agree"])), lambda_mean_cmi=dict(zip(ex_names, lam_mean["cmi"])),
               lambda_mean_ref=dict(zip(ex_names, lam_mean["ref"])), exponent_sum_mean=tot,
               seen_frac_test=dict(zip(ex_names, seen_frac)),
               seconds=dict(train=t_train, redundancy=t_red, test=t_test))
    print(f"\n[{name}] experts {ex_names}  [train {t_train:.0f}s, redundancy {t_red:.0f}s, test {t_test:.0f}s]")
    print("   solo bits/char: " + ", ".join(f"{n} {v:.3f}" for n, v in row["solo"].items()) + f" | reference (shortest training code): {ex_names[ref]}")
    print("   r_agree (pairwise, training slice):")
    for i, n in enumerate(ex_names):
        print(f"      {n:>9}: " + " ".join(f"{R_agree[i, j]:.2f}" for j in range(len(ex_names))))
    print("   r_cmi (normalised conditional MI of the argmax given the next character):")
    for i, n in enumerate(ex_names):
        print(f"      {n:>9}: " + " ".join(f"{R_cmi[i, j]:.2f}" for j in range(len(ex_names))))
    print("   mean exponent at test (over positions where seen): agree " + ", ".join(f"{n} {v:.2f}" for n, v in row["lambda_mean_agree"].items())
          + " | cmi " + ", ".join(f"{n} {v:.2f}" for n, v in row["lambda_mean_cmi"].items())
          + " | ref " + ", ".join(f"{n} {v:.2f}" for n, v in row["lambda_mean_ref"].items()))
    print("   mean SUM of exponents per position: " + ", ".join(f"{k} {v:.2f}" for k, v in tot.items()) + " (geo 1.00)")
    ch = row["solo"]["chain"]
    print(f"   bits/char online, held out: CHAIN alone {ch:.3f} | " + " | ".join(f"{k} {v:.3f}" for k, v in res.items()))
    print("   vs chain: " + ", ".join(f"{k} {v - ch:+.3f}" for k, v in res.items()))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_chars", type=int, default=300000)
    ap.add_argument("--test_chars", type=int, default=60000)
    ap.add_argument("--sets", nargs="+", default=["diverse", "redundant", "all18"])
    ap.add_argument("--out", default=str(HERE / "expB_naive_bayes_product.json"))
    args = ap.parse_args()
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)[:args.train_chars]           # E34's --train_chars convention
    test = test[:args.test_chars]
    print(f"expB: {len(text)} training characters (prefix of the concatenated books), {len(test)} held out (prefix of the book); V = {V}", flush=True)
    report = dict(V=V, n_train=int(len(text)), n_test=int(len(test)), sets={})
    t0 = time.time()
    for name in args.sets:
        report["sets"][name] = run_set(name, SETS[name], text, test, V, alphabet)
        json.dump(report, open(args.out, "w"), indent=1)
        sys.stdout.flush()
    print(f"\ntotal {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
