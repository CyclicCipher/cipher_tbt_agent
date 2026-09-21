"""Experiment C -- PAQ's gradient-free refinement: secondary symbol estimation (SSE / APM) as a COUNT table, on top of
E34's chain (E33's blended order-8 KT backoff table, `e34.Chain`).

The question: are the chain's probabilities miscalibrated in context-dependent ways, so that a count-based map from
(small context, the chain's own probability) to the rate at which that character actually came fixes part of what
PAQ's learned logistic mixer fixes -- with no learned parameter, everything by counting, online?

The mechanism (one SSE table per small context; tables sharing a bucket geometry are batched in a `Bank`):
  raw       p_raw[c] for every character c, from `Chain.dist` (the blended KT backoff distribution over the 70 symbols)
  bucket    b[c] = the bucket of stretch(p_raw[c]) = ln(p / (1 - p)), B equal-width buckets on [LO, HI] = [-10, 6]
            (default B = 24; a stretch outside the range lands in the end bucket)
  table     hits[ctx, b] = how often a character whose raw probability fell in bucket b under context ctx was the one
            that came;  n[ctx, b] = how often a character fell in that bucket (every position contributes all 70
            characters to n and one hit)
  refined   r[c] = (hits[ctx, b[c]] + p_raw[c]) / (n[ctx, b[c]] + 1)  -- the bucket's empirical rate, with a
            pseudo-count of 1 at the raw probability (the identity map as the prior, PAQ's APM initialisation)
  mix       p[c] ∝ (1 - lam) p_raw[c] + lam r[c], renormalised over the 70 characters; lam = 3/4 is PAQ's rule,
            lam = 1/2 and 1 are also scored in the same pass
  update    hits[ctx, b[actual]] += 1;  n[ctx, b[c]] += 1 for every c;  optional count cap (PAQ's limited counts:
            when n exceeds `cap`, hits and n are scaled down to cap, so the rate tracks the chain as it grows; checked
            every 4th step, so n overshoots the cap by at most 4 x 70)
  interp    optional: PAQ's APM interpolates the rate between the two nearest bucket nodes and updates the nearer one
The small contexts tried (each one keys its own table):
  none      the bucket alone (pure calibration of the chain)
  cls       the previous character's class: letter / space / punctuation / newline (`Stream.cls`)
  prev      the previous character (71 values incl. BORDER)
  deep      the deepest backoff order whose context was seen (0..8, `Chain.deepest`)
  conf      (deep, log2 bucket of the count N at that order, whether only one symbol was seen there) -- PPM's
            confidence in the deepest context, 9 x 6 x 2 = 108 cells
  char      the candidate character itself (a per-character calibration, 70 rows of B)
  cls_deep  class x deep (36);  prev_deep  previous character x deep (639);  conf_cls  conf x class (432)
Also: an average of two tables' refined rates (PAQ8's two APMs averaged), and a two-stage chain (the second table keyed
by the FIRST table's lam = 3/4 mixed output), for chosen pairs.
Everything online and in ONE pass over ONE chain: the SSE tables learn from the first training character on (the
`fresh` variants forget at the start of the test); the chain is E34's, learning throughout.
Measured: online bits/char on the first --test_chars of the held-out book after a --train_chars training slice, for
the chain alone and for every chain+SSE variant; plus the calibration table of the `none` context (per bucket: the
mean raw probability vs the empirical rate) -- the direct test of the miscalibration theory.

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/expC_sse_refinement.py
        --train_chars 300000 --test_chars 60000            (CPU, ~2.5 min) -> research/expC_sse_refinement.json
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
from textlm import load_corpus                                       # noqa: E402
from e34 import Stream, Chain                                        # noqa: E402

LAMS = np.array([0.5, 0.75, 1.0])
LO, HI = -10.0, 6.0
CTX_NAMES = ("none", "cls", "prev", "deep", "conf", "cls_deep", "prev_deep", "conf_cls")   # the scalar contexts, in ctx_vals order
CAP_EVERY = 4


def place(st, B, interp):
    """Bucket positions of every character's stretch: (b0, b1, w, b_update)."""
    if interp:
        x = np.clip((st - LO) * ((B - 1) / (HI - LO)), 0.0, B - 1 - 1e-9)
        b0 = x.astype(np.int64)
        w = x - b0
        b1 = np.minimum(b0 + 1, B - 1)
        return b0, b1, w, np.where(w > 0.5, b1, b0)                   # nodes, weight, the node PAQ updates
    b = np.clip((st - LO) * (B / (HI - LO)), 0.0, B - 1e-9).astype(np.int64)
    return b, None, None, b


class Bank:
    """K SSE count tables sharing one bucket geometry (B, interp), batched in one (rows, B) array with row offsets.
    Scalar-context tables come first (K_s of them), per-character (`char`) tables last."""

    def __init__(self, specs, n_ctx, V, B, interp, track=False):
        specs = [sp for sp in specs if sp["ctx"] != "char"] + [sp for sp in specs if sp["ctx"] == "char"]
        self.specs, self.B, self.interp, self.V, self.track = specs, B, interp, V, track
        self.names = [sp["name"] for sp in specs]
        self.K = len(specs)
        self.K_s = sum(sp["ctx"] != "char" for sp in specs)
        offsets = np.cumsum([0] + [n_ctx[sp["ctx"]] for sp in specs])
        self.offsets, rows = offsets[:-1], int(offsets[-1])
        self.hits, self.n = np.zeros((rows, B)), np.zeros((rows, B))
        self.sump = np.zeros((rows, B)) if track else None
        self.s_ctx = np.array([CTX_NAMES.index(sp["ctx"]) for sp in specs[:self.K_s]], dtype=np.int64)
        self.s_off = self.offsets[:self.K_s]
        self.v_off = [int(self.offsets[k]) for k in range(self.K_s, self.K)]
        self.ar = np.arange(V)
        caps = np.array([float(sp["cap"]) if sp["cap"] else np.inf for sp in specs[:self.K_s]])
        self.capped = np.flatnonzero(np.isfinite(caps))
        self.caps = caps[self.capped][:, None]
        self.fresh = [k for k, sp in enumerate(specs) if sp["fresh"]]
        self.bits = np.zeros((len(LAMS), self.K))

    def rows(self, ctx_vals):
        return self.s_off + ctx_vals[self.s_ctx]

    def reset_fresh(self):
        for k in self.fresh:
            lo, hi = int(self.offsets[k]), int(self.offsets[k + 1]) if k + 1 < self.K else len(self.n)
            self.hits[lo:hi] = 0; self.n[lo:hi] = 0

    def refine(self, rows, p, placed):
        """(K, V) refined rates."""
        b0, b1, w, _bu = placed
        R = np.empty((self.K, self.V))
        if self.interp:
            r0 = (self.hits[rows[:, None], b0[None, :]] + p) / (self.n[rows[:, None], b0[None, :]] + 1.0)
            r1 = (self.hits[rows[:, None], b1[None, :]] + p) / (self.n[rows[:, None], b1[None, :]] + 1.0)
            R[:self.K_s] = (1.0 - w) * r0 + w * r1
            for k, off in enumerate(self.v_off):
                r0 = (self.hits[off + self.ar, b0] + p) / (self.n[off + self.ar, b0] + 1.0)
                r1 = (self.hits[off + self.ar, b1] + p) / (self.n[off + self.ar, b1] + 1.0)
                R[self.K_s + k] = (1.0 - w) * r0 + w * r1
        else:
            R[:self.K_s] = (self.hits[rows[:, None], b0[None, :]] + p) / (self.n[rows[:, None], b0[None, :]] + 1.0)
            for k, off in enumerate(self.v_off):
                R[self.K_s + k] = (self.hits[off + self.ar, b0] + p) / (self.n[off + self.ar, b0] + 1.0)
        return R

    def update(self, rows, actual, p, placed, step):
        b = placed[3]
        if self.K_s:
            self.n[rows] += np.bincount(b, minlength=self.B)[None, :]
            self.hits[rows, b[actual]] += 1.0
            if self.track:
                self.sump[rows] += np.bincount(b, weights=p, minlength=self.B)[None, :]
            if len(self.capped) and step % CAP_EVERY == 0:
                rc = rows[self.capped]
                sub = self.n[rc]
                over = sub > self.caps
                if over.any():
                    f = np.where(over, self.caps / np.maximum(sub, 1.0), 1.0)
                    self.hits[rc] *= f
                    self.n[rc] = np.minimum(sub, self.caps)
        for off in self.v_off:                                         # per-character rows: (row, b) pairs are unique
            self.n[off + self.ar, b] += 1.0
            self.hits[off + actual, b[actual]] += 1.0

    def score(self, p, R, actual):
        """Accumulate -log2 of the actual under the renormalised mix, for every lam and table."""
        Q = (1.0 - LAMS)[:, None, None] * p[None, None, :] + LAMS[:, None, None] * R[None]     # (L, K, V)
        self.bits -= np.log2(Q[:, :, actual] / Q.sum(2))

    def calibration(self, name, ctx=0):
        """Per bucket of one context row of one table: n, hits, mean raw p, empirical rate (needs track=True)."""
        row = int(self.offsets[self.names.index(name)]) + ctx
        out = []
        for b in range(self.B):
            n, h = self.n[row, b], self.hits[row, b]
            if n > 0:
                out.append(dict(bucket=b, stretch_lo=LO + b * (HI - LO) / self.B, n=int(n), hits=int(h),
                                mean_raw_p=float(self.sump[row, b] / n), rate=float(h / n)))
        return out


def log2_bucket(N):
    return 0 if N <= 1 else 1 if N <= 3 else 2 if N <= 7 else 3 if N <= 15 else 4 if N <= 63 else 5


def deepest_info(chain, keys, deep):
    """(log2 bucket of N at the deepest seen order, 1 if only one symbol was seen there)."""
    if deep == 0 and not chain.tables[0][0]:
        return 0, 0
    k = keys[chain.R - deep]
    totals, counts = chain.tables[deep]
    N = totals.get(k, 0)
    det = 1 if N and len(counts[k]) == 1 else 0
    return log2_bucket(N), det


def n_cells(V):
    return {"none": 1, "cls": 4, "prev": V + 1, "deep": 9, "conf": 108, "cls_deep": 36, "prev_deep": (V + 1) * 9, "conf_cls": 432, "char": V}


def ctx_values(s, deep, info):
    """The scalar contexts of this position, in CTX_NAMES order."""
    cls, prev = s.cls(), s.back(1)
    conf = (deep * 6 + info[0]) * 2 + info[1]
    return np.array([0, cls, prev, deep, conf, cls * 9 + deep, prev * 9 + deep, conf * 4 + cls], dtype=np.int64)


def run(text, test, V, alphabet, specs, pairs=(), stacks=(), log=print):
    """One online pass (training slice, then test) over ONE chain; every SSE spec is scored on the test part.
    specs: dicts(name, ctx, B, cap, interp, fresh); pairs: (name_a, name_b) whose refined rates are averaged; stacks:
    (name_a, name_b): a second table with b's context and geometry, keyed by a's lam = 3/4 mixed output."""
    chain = Chain(V)
    s = Stream(V, alphabet)
    nc = n_cells(V)
    groups = {}
    for sp in specs:
        groups.setdefault((sp["B"], sp["interp"]), []).append(sp)
    banks = {g: Bank(sps, nc, V, g[0], g[1], track=(g == (specs[0]["B"], specs[0]["interp"]))) for g, sps in groups.items()}
    where = {nm: (g, k) for g, bank in banks.items() for k, nm in enumerate(bank.names)}
    pair_bits = np.zeros((len(LAMS), len(pairs)))
    if pairs:
        assert all(where[a][0] == where[b][0] for a, b in pairs), "pair members must share a geometry"
        pair_g = where[pairs[0][0]][0]
        pair_ia = np.array([where[a][1] for a, _b in pairs]); pair_ib = np.array([where[b][1] for _a, b in pairs])
    by_first = {}
    for a, b in stacks:
        by_first.setdefault(a, []).append(b)
    stack_banks = {}
    for a, bs in by_first.items():
        sp_b = [dict(specs[[sp["name"] for sp in specs].index(b)]) for b in bs]
        g = (sp_b[0]["B"], sp_b[0]["interp"])
        assert all((sp["B"], sp["interp"]) == g for sp in sp_b), "one geometry per first-stage table"
        for sp in sp_b:
            sp["name"] = f"{a}>{sp['name']}"; sp["fresh"] = False
        stack_banks[a] = Bank(sp_b, nc, V, g[0], g[1])
    chain_bits = 0.0
    n_train = len(text)
    seq = np.concatenate([text, test]).tolist()
    t0 = time.time()
    for i, c in enumerate(seq):
        testing = i >= n_train
        if i == n_train:
            for bank in banks.values():
                bank.reset_fresh()
        keys = chain.predict(s)
        p_raw, _ = chain.dist(keys)
        deep = chain.deepest(keys)
        info = deepest_info(chain, keys, deep)
        st = np.log(p_raw / (1.0 - p_raw))
        cv = ctx_values(s, deep, info)
        R, rows, placed = {}, {}, {}
        for g, bank in banks.items():
            placed[g] = place(st, g[0], g[1])
            rows[g] = bank.rows(cv)
            R[g] = bank.refine(rows[g], p_raw, placed[g])
            if testing:
                bank.score(p_raw, R[g], c)
        if testing:
            chain_bits -= math.log2(p_raw[c])
            if pairs:
                Rp = 0.5 * (R[pair_g][pair_ia] + R[pair_g][pair_ib])                          # (P, V)
                Q = (1.0 - LAMS)[:, None, None] * p_raw[None, None, :] + LAMS[:, None, None] * Rp[None]
                pair_bits -= np.log2(Q[:, :, c] / Q.sum(2))
        for a, bank in stack_banks.items():
            ga, ka = where[a]
            q = 0.25 * p_raw + 0.75 * R[ga][ka]
            q = q / q.sum()
            pl = place(np.log(q / (1.0 - q)), bank.B, bank.interp)
            rw = bank.rows(cv)
            R2 = bank.refine(rw, q, pl)
            if testing:
                bank.score(q, R2, c)
            bank.update(rw, c, q, pl, i)
        for g, bank in banks.items():
            bank.update(rows[g], c, p_raw, placed[g], i)
        chain.update(keys, c)
        s.push(c)
        if i == n_train - 1:
            log(f"   training slice done ({n_train} chars) [{time.time() - t0:.0f}s]")
    n_test = len(test)
    sse = {nm: {str(l): float(bank.bits[j, k] / n_test) for j, l in enumerate(LAMS)} for bank in banks.values() for k, nm in enumerate(bank.names)}
    stack_res = {nm: {str(l): float(bank.bits[j, k] / n_test) for j, l in enumerate(LAMS)} for bank in stack_banks.values() for k, nm in enumerate(bank.names)}
    res = dict(chain=chain_bits / n_test, sse=sse,
               pairs={f"{a}+{b}": {str(l): float(pair_bits[j, i] / n_test) for j, l in enumerate(LAMS)} for i, (a, b) in enumerate(pairs)},
               stacks=stack_res, seconds=time.time() - t0)
    main_bank = banks[(specs[0]["B"], specs[0]["interp"])]
    if "none" in main_bank.names:
        res["calibration_none"] = main_bank.calibration("none", 0)
    if "deep" in main_bank.names:
        res["calibration_deep"] = {str(d): main_bank.calibration("deep", d) for d in range(9)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_chars", type=int, default=300000)
    ap.add_argument("--test_chars", type=int, default=60000)
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--quick", type=int, default=0, help="1: only the single-context tables at the default geometry")
    ap.add_argument("--out", default=str(HERE / "expC_sse_refinement.json"))
    args = ap.parse_args()
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)[:args.train_chars]   # E34's convention: the first train_chars of the concatenated books
    test = test[:args.test_chars]
    print(f"Experiment C (SSE as a count table): {len(text)} training characters (the first of the concatenated books), "
          f"{len(test)} held-out characters (the start of De Bello Civili); V = {V}; B = {args.B} buckets on stretch [{LO:.0f}, {HI:.0f}]", flush=True)
    base = dict(B=args.B, cap=0, interp=False, fresh=False)
    specs = [dict(base, name=n, ctx=n) for n in ("none", "cls", "prev", "deep", "conf", "cls_deep", "prev_deep", "conf_cls", "char")]
    pairs, stacks = (), ()
    if not args.quick:
        for ctx in ("conf", "prev", "cls_deep"):                       # the count / geometry variants on three contexts
            specs += [dict(base, name=f"{ctx}_cap1024", ctx=ctx, cap=1024), dict(base, name=f"{ctx}_fresh", ctx=ctx, fresh=True),
                      dict(base, name=f"{ctx}_interp", ctx=ctx, interp=True)]
        pairs = (("conf", "prev"), ("conf", "char"), ("prev", "char"), ("conf_cls", "prev"), ("conf_cls", "char"), ("cls_deep", "prev"),
                 ("cls_deep", "char"), ("prev_deep", "char"), ("cls_deep", "conf"))
        stacks = (("cls_deep", "prev"), ("cls_deep", "char"), ("cls_deep", "conf"))
    print(f"   {len(specs)} SSE tables, {len(pairs)} averages, {len(stacks)} stacks, one pass", flush=True)
    res = run(text, test, V, alphabet, specs, pairs=pairs, stacks=stacks)
    print(f"   chain alone: {res['chain']:.4f} bits/char online  [{res['seconds']:.0f}s]", flush=True)
    print("   chain + SSE, bits/char at lam = 1/2 / 3/4 / 1 (gain at 3/4 vs the chain):", flush=True)
    for k, v in sorted(res["sse"].items(), key=lambda kv: kv[1]["0.75"]):
        print(f"      {k:<18} " + " / ".join(f"{v[str(l)]:.4f}" for l in LAMS) + f"   ({res['chain'] - v['0.75']:+.4f})", flush=True)
    for label, d in (("average of two", res["pairs"]), ("two-stage", res["stacks"])):
        if d:
            print(f"   {label}:", flush=True)
            for k, v in sorted(d.items(), key=lambda kv: kv[1]["0.75"]):
                print(f"      {k:<18} " + " / ".join(f"{v[str(l)]:.4f}" for l in LAMS) + f"   ({res['chain'] - v['0.75']:+.4f})", flush=True)
    print("   calibration of the chain at the end (context `none`; bucket: stretch_lo, n, mean raw p -> empirical rate):", flush=True)
    for row in res["calibration_none"]:
        print(f"      b{row['bucket']:>2} st>={row['stretch_lo']:6.2f} n={row['n']:>9} raw {row['mean_raw_p']:.5f} -> rate {row['rate']:.5f}"
              f"  ({'over' if row['mean_raw_p'] > row['rate'] else 'under'}-confident x{row['mean_raw_p'] / max(row['rate'], 1e-9):.2f})", flush=True)
    print("   the same by deepest order (mean raw p -> rate for the buckets with p >= 0.15, i.e. b12+, n >= 50):", flush=True)
    for d, rows in res["calibration_deep"].items():
        hi = [r for r in rows if r["bucket"] >= 12 and r["n"] >= 50]
        if hi:
            print(f"      deep {d}: " + "  ".join(f"{r['mean_raw_p']:.2f}->{r['rate']:.2f}" for r in hi), flush=True)
    report = dict(V=V, n_train=int(len(text)), n_test=int(len(test)), B=args.B, lams=LAMS.tolist(), stretch_range=[LO, HI],
                  specs=specs, pairs=[list(p) for p in pairs], stacks=[list(s) for s in stacks], result=res)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"   saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
