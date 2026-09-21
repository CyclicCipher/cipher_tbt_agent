"""E34 — gradient-free context mixing on the Latin stream (DESIGN §19–§20; pre-registered 2026-09-21; second in the
approved order). How much of the way from a table (E33: 1.822 bits/char online) toward a context-mixing compressor can
be had with no gradient anywhere: a LIBRARY of context functions, each a count table with the KT rate, mixed by
Bayesian weights in closed form, selected by the prequential code.

The experts (each maps a position to a context key; a KT count table from key to the next character; an unseen key
backs off to the unigram):
  chain      E33's blended backoff over the last 8 characters (the table of E33, as one expert)
  order k    the last k characters alone, k = 1..8 (no backoff -- what the mixer must learn to trust only when seen)
  skip d     the character d back alone, d = 2, 3, 4;  pair (1, d): the previous character with the one d back, d = 3, 4
  word       the characters of the current word so far (since the last space, up to 12)
  prevword   the previous complete word (up to 12 characters) with the current word so far
  line       the position in the line (capped at 80) with the previous character
  match      E18's induction head as a context: the character that FOLLOWED the last earlier occurrence of the current
             6-gram, with the length of that match in buckets of 4 (up to 32) -- the table learns how far to trust it
Mixing: Bayesian (exponential) weights over the experts, one weight vector per MIXING CONTEXT -- the previous character's
class (letter / space / punctuation / newline) x the longest order whose context was seen (0..8) -- updated by the
experts' probabilities of the character that came (the posterior), with fixed-share switching (a share alpha of the mass
is spread uniformly each step, so a newly good expert can recover). No trained mixer.
Selection: greedy drop of experts while the mixture's prequential code on a training prefix gets shorter (E33's price).
Measured: online bits per character on the held-out book against the table's 1.822 and xz's 2.263; every expert's own
online bits/char; the survivors. Pass: at or below 1.6. Refute: no gain over the table.

    python experiments/ziplearn/e34.py           (CPU, ~10 min) -> runs/e34/e34.json
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
sys.path.insert(0, str(HERE))
from textlm import load_corpus                                       # noqa: E402

WORD_CAP = 12
LINE_CAP = 80
MATCH_N = 6


class Stream:
    """The running state of a character stream: what every context function reads."""

    def __init__(self, V, alphabet):
        self.V = V
        self.hist = []                                                 # the characters so far
        self.space = {i for i, ch in enumerate(alphabet) if ch == " "}
        self.newline = {i for i, ch in enumerate(alphabet) if ch == "\n"}
        self.letter = {i for i, ch in enumerate(alphabet) if ch.isalpha()}
        self.word, self.prev_word, self.line_pos = [], [], 0
        self.last_seen = {}                                            # 6-gram key -> last position of its end
        self.match_pred, self.match_len = None, 0

    def back(self, k):
        return self.hist[-k] if len(self.hist) >= k else self.V       # BORDER = V

    def cls(self):
        c = self.back(1)
        return 0 if c in self.letter else 1 if c in self.space else 2 if c in self.newline else 3

    def push(self, c):
        """Advance the stream by one character; maintain the word, line and match states."""
        self.hist.append(c)
        n = len(self.hist)
        if c in self.space or c in self.newline:
            if self.word:
                self.prev_word = self.word
            self.word = []
        else:
            self.word = (self.word + [c])[-WORD_CAP:]
        self.line_pos = 0 if c in self.newline else min(self.line_pos + 1, LINE_CAP)
        # the match model: where did the current 6-gram last end?
        if n >= MATCH_N:
            key = tuple(self.hist[-MATCH_N:])
            j = self.last_seen.get(key)
            if j is not None:
                self.match_pred = self.hist[j] if j < n else None      # the character that followed the earlier occurrence
                length = 0
                while length < 32 and j - 1 - length >= 0 and n - 1 - length >= 0 and self.hist[j - 1 - length] == self.hist[n - 1 - length]:
                    length += 1
                self.match_len = length
            else:
                self.match_pred, self.match_len = None, 0
            self.last_seen[key] = n                                    # the position AFTER this 6-gram (= index of what follows)
        else:
            self.match_pred, self.match_len = None, 0


class Expert:
    """A count table with the KT rate over one context function; unseen key -> the unigram."""

    def __init__(self, name, fn, V):
        self.name, self.fn, self.V = name, fn, V
        self.totals, self.counts, self.uni_total, self.uni = {}, {}, 0, [0] * V
        self.bits = 0.0

    def predict(self, s):
        k = self.fn(s)
        N = self.totals.get(k, 0)
        if N:
            cnt = self.counts.get(k)
            return k, N, cnt
        return k, 0, None

    def prob(self, k, N, cnt, c):
        V = self.V
        if N:
            return (cnt.get(c, 0) + 0.5) / (N + V / 2)
        return (self.uni[c] + 0.5) / (self.uni_total + V / 2)

    def update(self, k, c):
        self.totals[k] = self.totals.get(k, 0) + 1
        d = self.counts.setdefault(k, {})
        d[c] = d.get(c, 0) + 1
        self.uni[c] += 1
        self.uni_total += 1

    def dist(self, k, N, cnt):
        """The KT distribution over all V characters, and whether the context was seen."""
        V = self.V
        if N:
            v = np.full(V, 0.5 / (N + V / 2))
            for ch, n in cnt.items():
                v[ch] += n / (N + V / 2)
            return v, True
        u = np.array(self.uni, dtype=np.float64)
        return (u + 0.5) / (self.uni_total + V / 2), False


class Chain:
    """E33's blended backoff over the last R characters, as one expert."""

    def __init__(self, V, R=8):
        self.name, self.V, self.R = "chain", V, R
        self.tables = [({}, {}) for _ in range(R + 1)]                 # order r: (totals, counts)
        self.bits = 0.0

    def predict(self, s):
        keys = [tuple(s.back(k) for k in range(1, r + 1)) for r in range(self.R, -1, -1)]
        return keys

    def prob(self, keys, c):
        V = self.V
        p, mass = 0.0, 1.0
        for r, k in zip(range(self.R, -1, -1), keys):
            totals, counts = self.tables[r]
            N = totals.get(k, 0)
            if N:
                p += mass * counts[k].get(c, 0) / (N + V / 2)
                mass *= (V / 2) / (N + V / 2)
        return p + mass / V

    def update(self, keys, c):
        for r, k in zip(range(self.R, -1, -1), keys):
            totals, counts = self.tables[r]
            totals[k] = totals.get(k, 0) + 1
            d = counts.setdefault(k, {})
            d[c] = d.get(c, 0) + 1

    def deepest(self, keys):
        for r, k in zip(range(self.R, -1, -1), keys):
            if self.tables[r][0].get(k, 0):
                return r
        return 0

    def dist(self, keys):
        V = self.V
        p = np.zeros(V)
        mass = 1.0
        for r, k in zip(range(self.R, -1, -1), keys):
            totals, counts = self.tables[r]
            N = totals.get(k, 0)
            if N:
                for ch, n in counts[k].items():
                    p[ch] += mass * n / (N + V / 2)
                mass *= (V / 2) / (N + V / 2)
        return p + mass / V, True


def build_experts(V, alphabet):
    ex = [Chain(V)]
    for k in range(1, 9):
        ex.append(Expert(f"order{k}", (lambda k: lambda s: tuple(s.back(i) for i in range(1, k + 1)))(k), V))
    for d in (2, 3, 4):
        ex.append(Expert(f"skip{d}", (lambda d: lambda s: (s.back(d),))(d), V))
    for d in (3, 4):
        ex.append(Expert(f"pair1_{d}", (lambda d: lambda s: (s.back(1), s.back(d)))(d), V))
    ex.append(Expert("word", lambda s: tuple(s.word), V))
    ex.append(Expert("prevword", lambda s: (tuple(s.prev_word), tuple(s.word)), V))
    ex.append(Expert("line", lambda s: (s.line_pos, s.back(1)), V))
    ex.append(Expert("match", lambda s: (s.match_pred, s.match_len // 4) if s.match_pred is not None else ("none",), V))
    return ex


class Mixer:
    """Bayesian mixing per mixing context with fixed-share switching; no trained parameters."""

    def __init__(self, n_experts, n_contexts, alpha=0.02):
        self.w = np.full((n_contexts, n_experts), 1.0 / n_experts)
        self.alpha = alpha

    def mix(self, ctx, probs):
        w = self.w[ctx]
        p = float(w @ probs)
        return p, w

    def update(self, ctx, probs, p):
        w = self.w[ctx]
        post = w * probs / p                                           # the Bayesian posterior over experts
        self.w[ctx] = (1 - self.alpha) * post + self.alpha / len(w)    # fixed share


def run_stream(experts, seq, V, alphabet, mixer=None, stream=None, learn=True, track_solo=True):
    """One prequential pass: returns (mixture bits, solo bits per expert, the stream, the mixer)."""
    s = stream or Stream(V, alphabet)
    n_ctx = 4 * 9
    mixer = mixer or Mixer(len(experts), n_ctx)
    total = 0.0
    solo = np.zeros(len(experts))
    probs = np.zeros(len(experts))
    chain = experts[0] if isinstance(experts[0], Chain) else None
    for c in seq.tolist():
        preds = [e.predict(s) for e in experts]
        for i, (e, pr) in enumerate(zip(experts, preds)):
            probs[i] = e.prob(pr, c) if isinstance(e, Chain) else e.prob(pr[0], pr[1], pr[2], c)
        deep = chain.deepest(preds[0]) if chain else 0
        ctx = s.cls() * 9 + deep
        p, _w = mixer.mix(ctx, probs)
        total -= math.log2(p)
        if track_solo:
            solo -= np.log2(probs)
        if learn:
            mixer.update(ctx, probs, p)
            for e, pr in zip(experts, preds):
                if isinstance(e, Chain):
                    e.update(pr, c)
                else:
                    e.update(pr[0], c)
        s.push(c)
    return total, solo, s, mixer


def run_mixtures(experts, seq, V, alphabet, mixer, stream):
    """One online pass over `seq` computing three mixtures at once: the Bayesian (linear) mixture per mixing context, the
    weighted GEOMETRIC mixture of the experts whose context was seen (exponents = the Bayesian weights renormalised
    over them), and the plain PRODUCT of the seen experts (exponent 1 each) -- the gradient-free stand-ins for PAQ's
    logistic mixer, which combines evidence multiplicatively where a linear mixture can only choose."""
    s = stream
    chain = experts[0]
    bits = dict(linear=0.0, geometric=0.0, product=0.0)
    probs = np.zeros(len(experts))
    for c in seq.tolist():
        preds = [e.predict(s) for e in experts]
        dists, seen = [], []
        for e, pr in zip(experts, preds):
            d, ok = e.dist(pr) if isinstance(e, Chain) else e.dist(pr[0], pr[1], pr[2])
            dists.append(d); seen.append(ok)
        D = np.stack(dists)                                            # (M, V)
        probs[:] = D[:, c]
        deep = chain.deepest(preds[0])
        ctx = s.cls() * 9 + deep
        p_lin, w = mixer.mix(ctx, probs)
        bits["linear"] -= math.log2(p_lin)
        sm = np.array(seen)
        logD = np.log(np.clip(D[sm], 1e-12, None))
        wg = w[sm] / w[sm].sum()
        g = np.exp(wg @ logD); g /= g.sum()
        bits["geometric"] -= math.log2(max(g[c], 1e-12))
        pr_ = np.exp(logD.sum(0) - logD.sum(0).max()); pr_ /= pr_.sum()
        bits["product"] -= math.log2(max(pr_[c], 1e-12))
        mixer.update(ctx, probs, p_lin)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        s.push(c)
    return {k: v / len(seq) for k, v in bits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select_chars", type=int, default=30000, help="training prefix on which experts are selected")
    ap.add_argument("--train_chars", type=int, default=0, help="0 = the whole training text")
    ap.add_argument("--test_chars", type=int, default=0, help="0 = the whole held-out book")
    ap.add_argument("--mixtures", type=int, default=0, help="1: no selection; train on --train_chars, then compare linear / geometric / product mixing online on the held-out book")
    ap.add_argument("--out", default=str(HERE / "runs" / "e34"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)
    if args.train_chars:
        text = text[:args.train_chars]
    if args.test_chars:
        test = test[:args.test_chars]
    print(f"E34: {len(text)} training characters, {len(test)} held out; V = {V}", flush=True)
    if args.mixtures:
        experts = build_experts(V, alphabet)
        t1 = time.time()
        _tb, _ts, s, mixer = run_stream(experts, text, V, alphabet, track_solo=False)
        res = run_mixtures(experts, test, V, alphabet, mixer, s)
        print(f"   trained on {len(text)} characters; held-out book online, bits/char: linear (Bayesian) {res['linear']:.3f}, "
              f"geometric (Bayesian exponents over seen experts) {res['geometric']:.3f}, product of seen experts {res['product']:.3f}  [{time.time() - t1:.0f}s]", flush=True)
        json.dump(dict(V=V, n_train=int(len(text)), n_test=int(len(test)), mixtures=res), open(out / "e34_mixtures.json", "w"), indent=1)
        return
    # 1. selection on a prefix: greedy drop while the mixture's prequential code falls
    prefix = text[:args.select_chars] if args.select_chars else text[:1]
    names_all = [e.name for e in build_experts(V, alphabet)]
    keep = list(range(len(names_all)))
    t0 = time.time()
    def bits_of(idx):
        ex = [e for i, e in enumerate(build_experts(V, alphabet)) if i in idx]
        total, solo, _s, _m = run_stream(ex, prefix, V, alphabet, track_solo=True)
        return total, solo
    cur = 0.0
    if args.select_chars:
        cur, solo = bits_of(keep)
        print(f"   all {len(keep)} experts on the first {len(prefix)} characters: mixture {cur / len(prefix):.3f} bits/char; solo: " +
              ", ".join(f"{names_all[i]} {solo[j] / len(prefix):.2f}" for j, i in enumerate(keep)) + f"  [{time.time() - t0:.0f}s]", flush=True)
    for round_ in range(2 if args.select_chars else 0):               # two rounds of batch drops (the greedy one at a time is too slow); 0 = keep all, the mixer selects online
        gains = {}
        for i in keep:
            if i == 0:
                continue                                               # the chain stays: it is the reference the mixture must beat
            b, _ = bits_of([j for j in keep if j != i])
            gains[i] = cur - b
        drop = [i for i, g in gains.items() if g > 0]
        if not drop:
            break
        keep = [j for j in keep if j not in drop]
        cur, _ = bits_of(keep)
        print(f"   round {round_ + 1}: dropped {[names_all[i] for i in drop]} -> mixture {cur / len(prefix):.3f} bits/char  [{time.time() - t0:.0f}s]", flush=True)
    survivors = [names_all[i] for i in keep]
    print(f"   survivors: {survivors}", flush=True)
    # 2. the full training pass, then the held-out book online
    experts = [e for i, e in enumerate(build_experts(V, alphabet)) if i in keep]
    t1 = time.time()
    _tb, _ts, s, mixer = run_stream(experts, text, V, alphabet, track_solo=False)
    train_secs = time.time() - t1
    total, solo, _s, _m = run_stream(experts, test, V, alphabet, mixer=mixer, stream=s, learn=True, track_solo=True)
    bpc = total / len(test)
    solo_bpc = {e.name: float(solo[i] / len(test)) for i, e in enumerate(experts)}
    weights = {e.name: float(_m.w[:, i].mean()) for i, e in enumerate(experts)}
    print("   mixer weights (mean over mixing contexts): " + ", ".join(f"{k} {v:.2f}" for k, v in weights.items()), flush=True)
    print(f"   held-out book, online: MIXTURE {bpc:.3f} bits/char (table alone, E33: 1.822; xz: 2.263) | solo: " +
          ", ".join(f"{k} {v:.3f}" for k, v in solo_bpc.items()) + f"  [train {train_secs:.0f}s, test {time.time() - t1 - train_secs:.0f}s]", flush=True)
    verdict = "PASS" if bpc <= 1.6 else "REFUTED" if bpc >= 1.822 else "INCONCLUSIVE"
    print(f"\nE34 verdict: {verdict} — mixture {bpc:.3f} bits/char online (pass <= 1.6; table 1.822)", flush=True)
    json.dump(dict(V=V, n_train=int(len(text)), n_test=int(len(test)), survivors=survivors, select_chars=args.select_chars,
                   bpc=bpc, solo_bpc=solo_bpc, weights=weights, verdict=verdict),
              open(out / ("e34.json" if args.select_chars else "e34_all.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
