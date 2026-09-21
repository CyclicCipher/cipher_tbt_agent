"""Experiment D -- geometric mixing of E34's 18 context experts with exponents that are a DETERMINISTIC function of each
expert's own evidence (no learned parameter anywhere).

Setting (E34's apparatus, `experiments/ziplearn/e34.py`): a training pass of `--train_chars` characters (default 300k,
spread over the 15 training books by `textlm.training_slice`) fits the 18 experts and the Bayesian per-context Mixer;
then ONE online pass over the first `--test_chars` characters (default 60k) of the held-out book (De Bello Civili),
where every variant below is scored from the SAME expert distributions D (18 x 70) at every position, then the experts
and the Mixer are updated as in `e34.run_mixtures`.

For expert m with current-context count N_m (0 if the context was never seen; for the chain, the count of the DEEPEST
order whose context was seen, `chain_mode = deepest`, or forced to full weight, `chain_mode = full`), the mixture is
    p(c) proportional to prod_m D_m(c) ** e_m,   renormalised over the 70 characters,
with
    e_m = g(N_m) / n_active ** q,            g in { N/(N+2), N/(N+8), N/(N+32), 1[N>0], N/(N+V/2) = KT confidence },
                                              q in {0, 0.5, 1},  n_active = number of experts with N_m > 0
    (the "main grid", 5 g x 3 q x 2 chain modes x 2 subsets = 60 variants), and additionally
    e_m = T * g(N_m) / sum_m g(N_m),          T in {0.5, 1, 1.5, 2, 3}   (the "temperature" family: total exponent mass
                                              fixed at T -- asks how much sharpness is right once redundancy is removed),
    e_m = T * w_m g(N_m) / sum_m w_m g(N_m),  the Bayesian weight w_m of the current mixing context times the confidence
                                              (closed-form parameters only; the "bayes x conf" family).
Subsets: `all` = the 18 experts; `nonnested` = the 18 minus order1..order8 (which the chain already contains) -- the
direct test of the theory that the product's over-sharpening is the nested orders each counted at full weight.

Baselines in the same pass: the chain alone (E33's table); the linear Bayesian mixture (E34); E34's geometric mixture
with the Bayesian weights as exponents over the seen experts; the raw product of the seen experts (= g=1[N>0], q=0).
Also recorded per variant: the mean total exponent mass, and the mean ENTROPY of the mixture's distribution in bits --
bits/char minus entropy is the calibration gap (positive = over-confident, the product's disease).

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/expD_confidence_exponents.py
        (CPU, ~2 min) -> experiments/ziplearn/research/expD_confidence_exponents.json
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
ZIPLEARN = HERE.parent
sys.path.insert(0, str(ZIPLEARN))
from textlm import load_corpus, training_slice                       # noqa: E402
from e34 import Chain, build_experts, run_stream                     # noqa: E402

K_LIST = (2, 8, 32)                    # g(N) = N/(N+k)
Q_LIST = (0.0, 0.5, 1.0)               # divide the exponents by n_active ** q
T_LIST = (0.5, 1.0, 1.5, 2.0, 3.0)     # total exponent mass in the temperature family
LN2 = math.log(2)


def chain_evidence(chain, keys):
    """(deepest seen order, its count) for the chain's current context."""
    for r, k in zip(range(chain.R, -1, -1), keys):
        N = chain.tables[r][0].get(k, 0)
        if N:
            return r, N
    return 0, 0


def g_names(V):
    return [f"N/(N+{k})" for k in K_LIST] + ["1[N>0]", f"KT N/(N+{V / 2:g})"]


def g_matrix(N, seen, V):
    """(5, M): the five confidence functions of the evidence vector N."""
    rows = [N / (N + k) for k in K_LIST] + [seen.astype(np.float64), N / (N + V / 2)]
    return np.stack(rows)


def build_variant_index(V):
    """The ordered list of variant specs (dicts); `exponent_rows` builds the exponent rows in the same order."""
    names_g = g_names(V)
    specs = []
    for subset in ("all", "nonnested"):
        for cm in ("deepest", "full"):
            for gn in names_g:
                for q in Q_LIST:
                    specs.append(dict(family="main", subset=subset, chain_mode=cm, g=gn, q=q))
    for subset in ("all", "nonnested"):
        for cm in ("deepest", "full"):
            for gn in names_g:
                for T in T_LIST:
                    specs.append(dict(family="temperature", subset=subset, chain_mode=cm, g=gn, T=T))
    for subset in ("all", "nonnested"):
        for gn in names_g:
            for T in T_LIST:
                specs.append(dict(family="bayes_x_conf", subset=subset, chain_mode="deepest", g=gn, T=T))
    return specs


def exponent_rows(N, seen, w, S, V):
    """All variants' exponent vectors, (n_variants, M), in the order of `build_variant_index`.
    N: (M,) evidence counts with the chain's deepest-order count; seen: (M,) bool; w: (M,) the Bayesian weights of the
    current mixing context; S: (2, M) subset masks (all, nonnested)."""
    G_deep = g_matrix(N, seen, V)                                      # (5, M)
    G_full = G_deep.copy(); G_full[:, 0] = 1.0                         # the chain at full weight
    G2 = np.stack([G_deep, G_full])                                    # (2 cm, 5 g, M)
    B = G2[None, :, :, :] * S[:, None, None, :]                        # (2 subset, 2 cm, 5 g, M)
    n_active = np.maximum((seen[None, :] & (S > 0)).sum(1), 1).astype(np.float64)   # (2 subset,)
    div = n_active[:, None] ** np.array(Q_LIST)[None, :]               # (2 subset, 3 q)
    E_main = B[:, :, :, None, :] / div[:, None, None, :, None]         # (2, 2, 5, 3, M)
    Bsum = np.maximum(B.sum(-1, keepdims=True), 1e-12)                 # (2, 2, 5, 1)
    E_T = np.array(T_LIST)[None, None, None, :, None] * (B / Bsum)[:, :, :, None, :]   # (2, 2, 5, nT, M)
    BW = G_deep[None, :, :] * S[:, None, :] * w[None, None, :]         # (2 subset, 5 g, M)
    BWsum = np.maximum(BW.sum(-1, keepdims=True), 1e-12)
    E_W = np.array(T_LIST)[None, None, :, None] * (BW / BWsum)[:, :, None, :]        # (2, 5, nT, M)
    M = len(N)
    return np.concatenate([E_main.reshape(-1, M), E_T.reshape(-1, M), E_W.reshape(-1, M)]), n_active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_chars", type=int, default=300_000, help="training characters, spread over the 15 training books")
    ap.add_argument("--test_chars", type=int, default=60_000, help="held-out characters (prefix of De Bello Civili)")
    ap.add_argument("--out", default=str(HERE / "expD_confidence_exponents.json"))
    args = ap.parse_args()
    t0 = time.time()
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(training_slice(train, args.train_chars)).astype(np.int16)
    test = test[:args.test_chars]
    print(f"expD: {len(text)} training characters (spread over {len(train)} books), {len(test)} held out; V = {V}", flush=True)

    experts = build_experts(V, alphabet)
    names = [e.name for e in experts]
    M = len(experts)
    _tb, _ts, s, mixer = run_stream(experts, text, V, alphabet, track_solo=False)
    t_train = time.time() - t0
    print(f"   training pass done [{t_train:.0f}s]", flush=True)

    specs = build_variant_index(V)
    nv = len(specs)
    nested = np.array([nm.startswith("order") for nm in names])
    S = np.stack([np.ones(M), (~nested).astype(np.float64)])           # subset masks: all, nonnested
    chain = experts[0]
    bits = np.zeros(nv); ent = np.zeros(nv); mass = np.zeros(nv)
    base = dict(chain=0.0, linear=0.0, geo_bayes=0.0, product=0.0)
    solo = np.zeros(M)
    n_active_sum = np.zeros(2)
    seen_frac = np.zeros(M)
    t1 = time.time()
    n = len(test)
    for i, c in enumerate(test.tolist()):
        preds = [e.predict(s) for e in experts]
        dists, seen, Nv = [], [], []
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                d, ok = e.dist(pr)
                _r, N = chain_evidence(e, pr)
            else:
                d, ok = e.dist(pr[0], pr[1], pr[2])
                N = pr[1]
            dists.append(d); seen.append(ok); Nv.append(N)
        D = np.stack(dists)                                            # (M, V)
        seen = np.array(seen); Nv = np.array(Nv, dtype=np.float64)
        probs = D[:, c]
        solo -= np.log2(probs)
        seen_frac += seen
        logD = np.log(np.clip(D, 1e-12, None))
        # -- baselines (E34's) --
        deep = chain.deepest(preds[0])
        ctx = s.cls() * 9 + deep
        p_lin, w = mixer.mix(ctx, probs)
        base["linear"] -= math.log2(p_lin)
        base["chain"] -= math.log2(probs[0])
        wg = w[seen] / w[seen].sum()
        lg = wg @ logD[seen]; lg -= lg.max(); gb = np.exp(lg); gb /= gb.sum()
        base["geo_bayes"] -= math.log2(max(gb[c], 1e-12))
        lp = logD[seen].sum(0); lp -= lp.max(); pr_ = np.exp(lp); pr_ /= pr_.sum()
        base["product"] -= math.log2(max(pr_[c], 1e-12))
        # -- the variants, all at once --
        E, n_act = exponent_rows(Nv, seen, w, S, V)                   # (nv, M)
        n_active_sum += n_act
        logits = E @ logD                                              # (nv, V), natural log
        mx = logits.max(1, keepdims=True)
        z = np.exp(logits - mx)
        Z = z.sum(1, keepdims=True)
        logp = logits - mx - np.log(Z)                                 # log-softmax
        bits -= logp[:, c] / LN2
        p = z / Z
        ent -= (p * logp).sum(1) / LN2
        mass += E.sum(1)
        # -- learn online, as E34 does --
        mixer.update(ctx, probs, p_lin)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        s.push(c)
        if (i + 1) % 20000 == 0:
            print(f"   {i + 1} test characters [{time.time() - t1:.0f}s]", flush=True)
    t_test = time.time() - t1

    bits /= n; ent /= n; mass /= n; solo /= n; seen_frac /= n; n_active_mean = n_active_sum / n
    base = {k: v / n for k, v in base.items()}
    rows = []
    for j, sp in enumerate(specs):
        r = dict(sp); r.update(bpc=float(bits[j]), entropy=float(ent[j]), gap=float(bits[j] - ent[j]), mean_mass=float(mass[j]))
        rows.append(r)

    def find(**kw):
        return next(r for r in rows if all(r.get(k) == v for k, v in kw.items()))

    print(f"\nbaselines (bits/char, online, {n} held-out chars after {len(text)} training chars): chain {base['chain']:.3f}, "
          f"linear Bayesian {base['linear']:.3f}, geometric w/ Bayesian exponents {base['geo_bayes']:.3f}, raw product {base['product']:.3f}")
    print(f"mean number of ACTIVE experts (context seen): all {n_active_mean[0]:.2f} of 18, nonnested {n_active_mean[1]:.2f} of 10")
    print("solo bits/char: " + ", ".join(f"{nm} {solo[i]:.2f} (seen {seen_frac[i]:.2f})" for i, nm in enumerate(names)))
    print("\nMAIN GRID  e_m = g(N_m) / n_active^q   (bits/char | entropy | mean total exponent mass)")
    for subset in ("all", "nonnested"):
        for cm in ("deepest", "full"):
            print(f"  subset={subset:9s} chain={cm:7s}")
            for gn in g_names(V):
                cells = []
                for q in Q_LIST:
                    r = find(family="main", subset=subset, chain_mode=cm, g=gn, q=q)
                    cells.append(f"q={q:<3g} {r['bpc']:.3f}|H {r['entropy']:.2f}|m {r['mean_mass']:.2f}")
                print(f"    g={gn:14s} " + "   ".join(cells))
    print("\nTEMPERATURE  e_m = T g(N_m)/sum g   (bits/char per T = " + ", ".join(f"{T:g}" for T in T_LIST) + ")")
    for subset in ("all", "nonnested"):
        for cm in ("deepest", "full"):
            print(f"  subset={subset:9s} chain={cm:7s}")
            for gn in g_names(V):
                cells = [find(family="temperature", subset=subset, chain_mode=cm, g=gn, T=T)["bpc"] for T in T_LIST]
                print(f"    g={gn:14s} " + "  ".join(f"{b:.3f}" for b in cells))
    print("\nBAYES x CONF  e_m = T w_m g(N_m)/sum w g   (bits/char per T)")
    for subset in ("all", "nonnested"):
        print(f"  subset={subset:9s}")
        for gn in g_names(V):
            cells = [find(family="bayes_x_conf", subset=subset, g=gn, T=T)["bpc"] for T in T_LIST]
            print(f"    g={gn:14s} " + "  ".join(f"{b:.3f}" for b in cells))
    best = sorted(rows, key=lambda r: r["bpc"])[:8]
    print("\nbest 8 variants:")
    for r in best:
        print(f"   {r['bpc']:.3f}  {r['family']:12s} subset={r['subset']:9s} chain={r['chain_mode']:7s} g={r['g']:14s} "
              + (f"q={r['q']:g}" if 'q' in r else f"T={r['T']:g}") + f"   entropy {r['entropy']:.2f}  mass {r['mean_mass']:.2f}")
    total = time.time() - t0
    print(f"\n[train {t_train:.0f}s, test {t_test:.0f}s, total {total:.0f}s]")
    json.dump(dict(V=V, n_train=int(len(text)), n_test=int(n), train_secs=t_train, test_secs=t_test,
                   experts=names, solo_bpc={nm: float(solo[i]) for i, nm in enumerate(names)},
                   seen_frac={nm: float(seen_frac[i]) for i, nm in enumerate(names)},
                   n_active_mean=dict(all=float(n_active_mean[0]), nonnested=float(n_active_mean[1])),
                   baselines=base, variants=rows, best=best),
              open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
