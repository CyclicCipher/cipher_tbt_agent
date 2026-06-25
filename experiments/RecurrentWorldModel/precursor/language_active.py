"""Active sensorimotor next-token learning — predicting a token IS a motor action (architecture doc §5/§6).

`language.py` learns PASSIVELY: count every co-occurrence, factorize once (batch SVD/eigh), done. No agency.
This is the ACTIVE counterpart — the same sensorimotor loop the column uses everywhere (efference copy ->
reafference -> error), now driving language:

  state    : the column sits at a LOCATION in the frame — the code E[x_t] of the current token (L6).
  MOTOR ACT: it must emit the next token, so it NAVIGATES — applies the learned displacement operator Op (L5)
             to predict where the next token sits:  p = Op @ E[x_t].  Read-out = the token nearest p (L4).
  reafference: the world reveals the actual next token x_{t+1}, at its true location E[x_{t+1}].
  error    : the navigation was wrong by  δ = E[x_{t+1}] − p  — the reafferent prediction error in the frame.
  learn    : a local delta-rule corrects the operator toward where it ACTUALLY landed, and a contrastive
             (negative-sampling) push shapes the codes so prediction works — online, streaming, surprise-
             weighted (active inference spends learning where it was most surprised).

So the model learns the corpus by READING it and anticipating each word, correcting itself from the miss —
not by tallying a co-occurrence matrix. Same representational class as the passive model (codes E + a linear
operator Op), so the comparison is clean: PASSIVE batch factorization vs ACTIVE sensorimotor learning, head to
head on held-out perplexity. ONE shared operator across all contexts (the Broca/L5 principle) — and, per the
SGNS<->PMI<->SR-frame equivalence (Levy & Goldberg 2014), the active route should converge toward the same
geometry the passive route gets in closed form, but by acting. We pool Latin + Middle/Old High German into one
frame and watch whether reading anticipation improves and whether the geometry stays coherent per language.

Run:  python -m precursor.language_active        (from experiments/RecurrentWorldModel/)
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from precursor.language import (                                     # the shared passive harness  # noqa: E402
    build_vocab, chunk_docs, directed_embed, encode, forward_cooc, interp_bigram_nll, load_docs,
    nearest, pairs_of, ppmi, split_docs, sr_lm_nll, strict_bigram, unigram_nll,
)

LAT = ["rex", "bellum", "caesar", "miles", "deus", "urbs"]
GER = ["lant", "sprach", "niht", "helde", "recken", "künec"]


def load_pooled(subdirs=("latin books", "mittelhochdeutsch", "althochdeutsch")):
    docs = []
    for sub in subdirs:
        docs += chunk_docs(load_docs(sub))
    return docs


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class ActiveColumn:
    """Codes E (V×d) + ONE displacement operator Op (d×d), learned by the sensorimotor loop above."""

    def __init__(self, V, d, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((V, d)) * 0.1
        self.Op = np.eye(d) + rng.standard_normal((d, d)) * 0.01     # start near "stay put", learn to navigate
        self.rng = rng

    def train(self, pairs, uni, uni_logp, epochs=8, lr=0.025, neg=5, batch=256, surprise=True, monitor=3000, thr=1e-3):
        V, d = self.E.shape
        I = np.array([i for i, _ in pairs]); J = np.array([j for _, j in pairs])
        f = uni / uni.sum()
        keep = np.minimum(np.sqrt(thr / np.maximum(f, 1e-12)), 1.0)   # word2vec subsampling: tame the function words
        negp = (uni ** 0.75); negp /= negp.sum()
        mon = self.rng.choice(len(pairs), size=min(monitor, len(pairs)), replace=False)
        eye, curve = np.eye(d), []
        for ep in range(epochs):
            idx = np.nonzero(self.rng.random(len(pairs)) < keep[I])[0]   # drop frequent contexts (fresh each epoch)
            self.rng.shuffle(idx)
            eta = lr * (1 - ep / max(1, epochs)) + 1e-3
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                i, j = I[b], J[b]
                H = self.E[i]                                         # current location  (B,d)
                P = H @ self.Op.T                                     # MOTOR ACT: navigate to predicted loc
                k = self.rng.choice(V, size=(len(b), neg), p=negp)    # negative tokens (B,neg)
                spos = _sigmoid(np.sum(P * self.E[j], axis=1))        # P(land on the true next | predicted)
                sneg = _sigmoid(np.einsum("bd,bkd->bk", P, self.E[k]))
                wgt = (1.0 - spos) if surprise else np.ones_like(spos)   # spend learning where SURPRISED
                ep_pos = ((spos - 1.0) * wgt)[:, None]                # reafferent error on the true target
                ep_neg = (sneg * wgt[:, None])[:, :, None]
                gP = ep_pos * self.E[j] + np.sum(ep_neg * self.E[k], axis=1)   # error back to the predicted loc
                gP *= np.minimum(1.0, 5.0 / (np.linalg.norm(gP, axis=1, keepdims=True) + 1e-9))   # clip
                self.Op -= eta * (gP.T @ H) / len(b)                  # correct the navigation operator (L5)
                self.Op -= 1e-5 * (self.Op - eye)                     # whisper of decay — bounded, but free to navigate
                gH = gP @ self.Op                                     # error back to the current code
                np.add.at(self.E, i, -eta * gH)
                np.add.at(self.E, j, -eta * ep_pos * P)
                np.add.at(self.E, k.ravel(), -eta * (ep_neg.reshape(-1, 1) * np.repeat(P, neg, axis=0)))
            n = np.linalg.norm(self.E, axis=1, keepdims=True)         # cap code norms each epoch (no blow-up/collapse)
            self.E *= np.minimum(1.0, 4.0 / (n + 1e-9))
            curve.append(self._monitor_ppl(I[mon], J[mon], uni_logp))
        return curve

    def _monitor_ppl(self, i, j, uni_logp):
        scores = (self.E[i] @ self.Op.T) @ self.E.T + uni_logp[None, :]   # anticipate next token as we read
        mx = scores.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(scores - mx).sum(1))
        return float(np.exp((lse - scores[np.arange(len(i)), j]).mean()))

    def current_next(self):
        """(predicted-location vectors, target codes) for the shared perplexity evaluator."""
        return self.E @ self.Op.T, self.E


# ── run: passive vs active, head to head on the pooled corpus ───────────────────────────────────────────
def run(pooled=True, subdir="latin books", V=5000, d=200, window=5, epochs=6, seed=0):
    docs = load_pooled() if pooled else chunk_docs(load_docs(subdir))
    train, held = split_docs(docs, 0.15, seed)
    stoi, itos = build_vocab(train, V); V = len(itos)
    enc_tr, enc_te = encode(train, stoi), encode(held, stoi)
    tr_pairs, test = pairs_of(enc_tr), pairs_of(enc_te)
    dev = list(np.array(tr_pairs, dtype=object)[np.random.default_rng(1).choice(len(tr_pairs), 5000)])
    dev = [(int(i), int(j)) for i, j in dev]

    Cbg, uni = strict_bigram(enc_tr, V)
    rs = np.asarray(Cbg.sum(1)).ravel()
    uni_logp = np.log((uni + 1.0) / (uni.sum() + V)); uni_p = np.exp(uni_logp)
    ppl = lambda nll: math.exp(nll)

    # PASSIVE: batch factorization of the co-occurrence graph (language.py)
    w_p, c_p = directed_embed(ppmi(forward_cooc(enc_tr, V, window)), d)
    bP = min([0.5, 1.0, 2.0], key=lambda b: sr_lm_nll(w_p, c_p, uni_logp, dev, b))

    # ACTIVE: learn the same corpus by reading + anticipating (sensorimotor)
    col = ActiveColumn(V, d, seed)
    curve = col.train(tr_pairs, uni, uni_logp, epochs=epochs)
    op_drift = float(np.linalg.norm(col.Op - np.eye(d)) / math.sqrt(d))   # did the navigation operator move off I?
    w_a, c_a = col.current_next()
    bA = min([0.5, 1.0, 2.0], key=lambda b: sr_lm_nll(w_a, c_a, uni_logp, dev, b))

    lam = min([0.0, 0.1, 0.3, 0.5, 0.7, 0.9], key=lambda l: interp_bigram_nll(Cbg, rs, uni_p, dev, l))
    freq_rank = {i: r for r, i in enumerate(np.argsort(-uni))}
    rare = [(i, j) for i, j in test if freq_rank[i] > 500]

    res = {
        "unigram": ppl(unigram_nll(uni_logp, test)),
        "bigram(interp)": ppl(interp_bigram_nll(Cbg, rs, uni_p, test, lam)),
        "passive SR-frame": ppl(sr_lm_nll(w_p, c_p, uni_logp, test, bP)),
        "ACTIVE sensorimotor": ppl(sr_lm_nll(w_a, c_a, uni_logp, test, bA)),
    }
    rare_res = {
        "bigram(interp)": ppl(interp_bigram_nll(Cbg, rs, uni_p, rare, lam)),
        "passive": ppl(sr_lm_nll(w_p, c_p, uni_logp, rare, bP)),
        "ACTIVE": ppl(sr_lm_nll(w_a, c_a, uni_logp, rare, bA)),
    }
    probes = [p for p in LAT + GER if p in stoi]
    return dict(V=V, d=d, ndocs=len(docs), n_tr=sum(map(len, enc_tr)), n_te=sum(map(len, enc_te)),
                res=res, rare=rare_res, n_test=len(test), n_rare=len(rare), curve=curve, op_drift=op_drift,
                active_nn={p: nearest(col.E, itos, stoi, p) for p in probes},
                passive_nn={p: nearest(w_p, itos, stoi, p) for p in probes})


if __name__ == "__main__":
    print("active sensorimotor language — predicting a token as a MOTOR ACT, vs passive factorization\n")
    r = run(pooled=True)
    print(f"=== POOLED Latin + Middle/Old High German  ({r['ndocs']} chunks, {r['n_tr']:,} train / "
          f"{r['n_te']:,} test tokens, V={r['V']}, d={r['d']}) ===\n")
    print(f"  next-token perplexity (held-out, {r['n_test']:,} pairs):")
    for name, p in r["res"].items():
        print(f"      {name:>20}:  {p:8.1f}")
    print(f"\n  rare-context (ctx rank>500, {r['n_rare']:,} pairs):")
    for name, p in r["rare"].items():
        print(f"      {name:>20}:  {p:8.1f}")
    print(f"\n  active reading-anticipation by epoch (held-out monitor perplexity, lower=better; "
          f"operator drift ‖Op−I‖={r['op_drift']:.2f}):")
    print("      " + "  ".join(f"e{i+1}:{p:.0f}" for i, p in enumerate(r["curve"])))
    print("\n  nearest neighbours — ACTIVE codes (learned by reading):")
    for p, nn in r["active_nn"].items():
        if nn:
            print(f"      {p:>8} -> {', '.join(nn)}")
    print("\n  nearest neighbours — PASSIVE codes (batch factorization), same probes:")
    for p, nn in r["passive_nn"].items():
        if nn:
            print(f"      {p:>8} -> {', '.join(nn)}")
