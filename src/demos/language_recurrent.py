"""Recurrent (selective state-space) active column — give the reader a MEMORY of the sequence (architecture doc).

The active model so far is Markov-1: it predicts the next token from the CURRENT token's code only — it can't
tell "B after A" from "B after X". TBT's answer is path integration: the column's LOCATION is the integrated
history. So we make L6 a DYNAMIC, path-integrated state, updated each token by a SELECTIVE linear recurrence —
Mamba's mechanism, adapted to this model and learned online with a 1-step-truncated local rule (no BPTT, no NN):

    α(x) = σ(Wₐ·E[x] + bₐ) ∈ (0,1)^d                  # FULL (per-channel) selectivity — a learned projection
    h_t  = α(x_t) ⊙ h_{t-1} + (1−α(x_t)) ⊙ E[x_t]     # selective path integration in the SR frame (L6)
    p_t  = Op · h_t                                   # MOTOR ACT: navigate from the integrated location (L5)
    P(next=j) ∝ exp( E_j · p_t )                      # soft retrieval (L4), + unigram prior

Per-channel α (Mamba's diagonal selective state) lets a token RESET some state channels while CARRYING others
(reset "which phrase", keep "which sentence"). The convex form keeps h inside the convex hull of the codes, so
it stays bounded even when we CARRY THE STATE ACROSS THE WHOLE CHUNK (~1500 tokens) instead of resetting every
segment — effectively unbounded context, reset only at chunk boundaries (= the train/test split, so no leakage).
The gate learns from the 1-step gradient dL/dα = (Opᵀ gP) ⊙ (h_{t-1} − E[x]); we don't need lossless memory
(no attention) — just a cheap structured state, which is exactly what an SSM is.

Head to head on the pooled corpus: Markov active (no memory) vs this recurrent active, with passive + n-gram refs.

Run:  python -m demos.language_recurrent      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np


import torch                                                          # eval matmuls only  # noqa: E402

from demos.language import (                                      # shared harness  # noqa: E402
    build_vocab, directed_embed, encode, forward_cooc, interp_bigram_nll, nearest, pairs_of, ppmi,
    split_docs, sr_lm_nll, strict_bigram, unigram_nll,
)
from demos.language_active import ActiveColumn, GER, LAT, _sigmoid, load_pooled   # noqa: E402
from tbt.recurrence import SelectiveRecurrence                # the ONE canonical selective-gated recurrence  # noqa: E402


def _pad(batch):
    lens = np.array([len(c) for c in batch])
    S = np.zeros((len(batch), lens.max()), dtype=int)
    for bi, c in enumerate(batch):
        S[bi, :len(c)] = c
    return S, lens


class SSMColumn:
    """Codes E + displacement operator Op + the canonical selective recurrence (per-token, per-channel gate)."""

    def __init__(self, V, d, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((V, d)) * 0.1
        self.Op = np.eye(d) + rng.standard_normal((d, d)) * 0.01
        self.rec = SelectiveRecurrence(d, n_keys=V)                   # the shared recurrence: A=identity, drive=E[x]
        self.rng = rng

    def train(self, chunks, uni, epochs=5, lr=0.05, lr_g=0.5, neg=5, batch=96, surprise=True):
        V, d = self.E.shape
        eye = np.eye(d)
        negp = uni ** 0.75; negp /= negp.sum()
        chunks = [c for c in chunks if len(c) > 2]
        for ep in range(epochs):
            self.rng.shuffle(chunks)
            eta = lr * (1 - ep / max(1, epochs)) + 5e-3
            for s in range(0, len(chunks), batch):
                S, lens = _pad(chunks[s:s + batch])
                B = len(lens)
                h = np.zeros((B, d))
                for t in range(S.shape[1] - 1):
                    valid = ((t + 1) < lens).astype(float)            # mask: predict t→t+1 only where both exist
                    x, y = S[:, t], S[:, t + 1]
                    ex = self.E[x]
                    a = self.rec.gate(x)                              # canonical per-channel gate, computed ONCE
                    h_prev = h
                    h = self.rec.step(h_prev, ex, a)                  # h = a⊙h_prev + (1−a)⊙ex (A = identity)
                    p = h @ self.Op.T
                    k = self.rng.choice(V, size=(B, neg), p=negp)
                    spos = _sigmoid(np.sum(p * self.E[y], 1))
                    sneg = _sigmoid(np.einsum("bd,bkd->bk", p, self.E[k]))
                    wgt = ((1 - spos) if surprise else np.ones(B)) * valid   # surprise-weight × validity mask
                    ep_pos = ((spos - 1) * wgt)[:, None]
                    ep_neg = (sneg * wgt[:, None])[:, :, None]
                    gP = ep_pos * self.E[y] + np.sum(ep_neg * self.E[k], 1)
                    gP *= np.minimum(1.0, 5.0 / (np.linalg.norm(gP, axis=1, keepdims=True) + 1e-9))
                    gH = gP @ self.Op
                    self.Op -= eta * (gP.T @ h) / B; self.Op -= 1e-5 * (self.Op - eye)
                    self.rec.learn_gate(x, a, gH, h_prev, ex, lr_g)   # canonical gate learning (reuses gate a)
                    idx = np.concatenate([x, y, k.ravel()])           # ONE scatter for drive + pos + neg (np.add.at
                    vals = np.concatenate([-eta * (1 - a) * gH, -eta * ep_pos * p,   # is the hot path: 3 calls → 1)
                                           -eta * (ep_neg.reshape(-1, 1) * np.repeat(p, neg, 0))])
                    np.add.at(self.E, idx, vals)
            n = np.linalg.norm(self.E, axis=1, keepdims=True)
            self.E *= np.minimum(1.0, 4.0 / (n + 1e-9))
            self.rec.clip()

    @torch.no_grad()
    def eval_nll(self, chunks, uni_logp, beta, rare_rank=None, rare_thresh=500, batch=96):
        E = torch.tensor(self.E, dtype=torch.float32); Op = torch.tensor(self.Op, dtype=torch.float32)
        G = torch.tensor(self.rec.G, dtype=torch.float32); up = torch.tensor(uni_logp, dtype=torch.float32)
        chunks = [c for c in chunks if len(c) > 2]
        tot, n, rtot, rn = 0.0, 0, 0.0, 0
        for s in range(0, len(chunks), batch):
            S, lens = _pad(chunks[s:s + batch])
            St = torch.tensor(S, dtype=torch.long); lt = torch.tensor(lens)
            h = torch.zeros(len(lens), E.shape[1])
            for t in range(S.shape[1] - 1):
                valid = (t + 1) < lt
                x, y = St[:, t], St[:, t + 1]
                a = torch.sigmoid(G[x])
                h = a * h + (1 - a) * E[x]
                logits = beta * (h @ Op.t()) @ E.t() + up[None, :]
                ll = (torch.logsumexp(logits, 1) - logits[torch.arange(len(lens)), y]) * valid
                tot += ll.sum().item(); n += int(valid.sum())
                if rare_rank is not None:
                    m = valid & torch.tensor([rare_rank[int(i)] > rare_thresh for i in x])
                    rtot += ll[m].sum().item(); rn += int(m.sum())
        return math.exp(tot / n), (math.exp(rtot / rn) if rn else float("nan"))


def run(V=5000, d=200, epochs=5, seed=0):
    docs = load_pooled()
    train, held = split_docs(docs, 0.15, seed)
    stoi, itos = build_vocab(train, V); V = len(itos)
    enc_tr, enc_te = encode(train, stoi), encode(held, stoi)
    test = pairs_of(enc_te)
    rng = np.random.default_rng(1)
    tr_pairs = pairs_of(enc_tr)
    dev = [(int(tr_pairs[i][0]), int(tr_pairs[i][1])) for i in rng.choice(len(tr_pairs), 5000)]
    Cbg, uni = strict_bigram(enc_tr, V); rs = np.asarray(Cbg.sum(1)).ravel()
    uni_logp = np.log((uni + 1.0) / (uni.sum() + V)); uni_p = np.exp(uni_logp)
    freq_rank = {i: r for r, i in enumerate(np.argsort(-uni))}
    rare = [(i, j) for i, j in test if freq_rank[i] > 500]
    ppl = lambda nll: math.exp(nll)

    # references (predict from token t only)
    w_p, c_p = directed_embed(ppmi(forward_cooc(enc_tr, V, 5)), d)
    bP = min([0.5, 1.0, 2.0], key=lambda b: sr_lm_nll(w_p, c_p, uni_logp, dev, b))
    mk = ActiveColumn(V, d, seed); mk.train(tr_pairs, uni, uni_logp, epochs=max(epochs, 6), track=False)
    w_m, c_m = mk.current_next()
    bM = min([0.5, 1.0, 2.0], key=lambda b: sr_lm_nll(w_m, c_m, uni_logp, dev, b))
    lam = min([0.0, 0.3, 0.6, 0.9], key=lambda l: interp_bigram_nll(Cbg, rs, uni_p, dev, l))

    # the recurrent model (predict from the integrated prefix, carried across the whole chunk)
    ssm = SSMColumn(V, d, seed); ssm.train(enc_tr, uni, epochs=epochs)
    dev_chunks = enc_tr[:8]
    bR = min([0.5, 1.0, 2.0], key=lambda b: ssm.eval_nll(dev_chunks, uni_logp, b)[0])
    ssm_all, ssm_rare = ssm.eval_nll(enc_te, uni_logp, bR, freq_rank)

    res = {"unigram": ppl(unigram_nll(uni_logp, test)),
           "bigram(interp)": ppl(interp_bigram_nll(Cbg, rs, uni_p, test, lam)),
           "passive SR-frame": ppl(sr_lm_nll(w_p, c_p, uni_logp, test, bP)),
           "Markov active (no memory)": ppl(sr_lm_nll(w_m, c_m, uni_logp, test, bM)),
           "RECURRENT active (memory)": ssm_all}
    rare_res = {"bigram(interp)": ppl(interp_bigram_nll(Cbg, rs, uni_p, rare, lam)),
                "passive": ppl(sr_lm_nll(w_p, c_p, uni_logp, rare, bP)),
                "Markov active": ppl(sr_lm_nll(w_m, c_m, uni_logp, rare, bM)),
                "RECURRENT active": ssm_rare}
    alpha = _sigmoid(ssm.rec.G).mean(1)                             # mean per-channel decay per token
    common = [i for i in np.argsort(-uni)[:800] if itos[i] != "<unk>"]
    hi = sorted(common, key=lambda i: -alpha[i])[:8]; lo = sorted(common, key=lambda i: alpha[i])[:8]
    probes = [p for p in LAT + GER if p in stoi]
    return dict(V=V, d=d, ndocs=len(docs), n_tr=sum(map(len, enc_tr)), n_test=len(test), n_rare=len(rare),
                res=res, rare=rare_res, bR=bR,
                gate_hi=[(itos[i], alpha[i]) for i in hi], gate_lo=[(itos[i], alpha[i]) for i in lo],
                nn={p: nearest(ssm.E, itos, stoi, p) for p in probes})


if __name__ == "__main__":
    print("recurrent active column — FULL per-channel selectivity + state carried across the chunk\n")
    r = run()
    print(f"=== POOLED Latin + Middle/Old High German  ({r['ndocs']} chunks, {r['n_tr']:,} train tokens, "
          f"V={r['V']}, d={r['d']}) ===\n")
    print(f"  next-token perplexity (held-out, {r['n_test']:,} pairs):")
    for k, p in r["res"].items():
        print(f"      {k:>26}:  {p:8.1f}")
    print(f"\n  rare-context (ctx rank>500, {r['n_rare']:,} pairs):")
    for k, p in r["rare"].items():
        print(f"      {k:>26}:  {p:8.1f}")
    print(f"\n  learned mean decay α (high = carry context through this token; low = overwrite/reset):")
    print("      high-α (transparent): " + ", ".join(f"{w}:{a:.2f}" for w, a in r["gate_hi"]))
    print("      low-α  (resets ctx) : " + ", ".join(f"{w}:{a:.2f}" for w, a in r["gate_lo"]))
    print("\n  nearest neighbours — recurrent codes:")
    for p, nn in r["nn"].items():
        if nn:
            print(f"      {p:>8} -> {', '.join(nn)}")
