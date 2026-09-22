"""Experiment H -- do the two best closed-form gains ADD, and is SSE's gain the chain's escape calibration?

Two questions from notes/gradient_free_mixing_and_features.md section I.3 (items 3 and 4), on the standard slice
(300 000 training characters = the first of the concatenated books; the first 60 000 characters of De Bello Civili,
online / prequential bits per character; V = 70):

  (1) ADDITIVITY. expC's SSE count tables (`conf_cls`, and the two-stage `cls_deep>char`, lam = 3/4) applied on top of
      the OUTPUT of expA's per-context grid-posterior geometric mixture chain + word (`grid_bayes_by_ctx`, the 63-point
      fine grid a in 0.5..1.1, b in 0.0..0.8, share 0.01), versus each alone. Then the same on top of A2-min
      (mix_temperature_bayes_decay: 18 experts, per-context posterior over the 42 (beta_chain, beta_rest) points,
      decaying switch alpha_t = 1/(t+2)).
  (2) THE COUNTED ESCAPE. `e34.Chain` gives each backoff level the KT escape (V/2)/(N + V/2). `EscChain` below replaces
      it by an escape probability COUNTED per (order r, floor(log2 N), number of distinct symbols d) bucket (PPM-style
      secondary escape estimation, Bloom's SEE keys): e = (escapes + 1/2)/(events + 1), where an event is one position
      at one seen level and an escape is "the character that came had count 0 at that level". Two structures: `blend`
      keeps the chain's blending (each seen level, deepest first, gives its symbols (1 - e) cnt/N of the remaining mass
      and passes e on); `excl` is PPM exclusion (the escaped mass is spread by the next-shorter level over the symbols
      NOT seen at this one); `ppm` (top-down PPM with exclusions, conditional counts); `cal` (the blend's
      interpolation kept, the counted escape setting the seen/unseen split) and `cal_cls` (`cal` with the previous
      character's class in the key) -- see EscChain. The chain alone, SSE on top, the grid on top, and grid + SSE are
      re-measured per chain.

RESULTS (2026-09-21, the log and JSON beside this file; paired vs the KT chain 2.2288, SE over 6 blocks of 10k):
  additivity: SSE conf_cls / cls_deep>char on the chain -0.048 / -0.050; grid alone -0.078; grid + SSE -0.083 / -0.083
  (SSE on top of the grid -0.005, SE 0.001; interaction +0.043, SE 0.005). A2-min -0.062; A2 + SSE -0.070 / -0.076
  (SSE on top of A2 -0.007 / -0.013; interaction +0.040 / +0.038). The gains do NOT add: ~90% of what SSE fixes on the
  chain, the per-context exponents (a = 0.5-0.9 on the chain) fix as well.
  the escape: `blend` 2.901 (the minimal edit is catastrophic: the escaped mass compounds through 8 levels);
  `excl` 2.388, `ppm` 2.413 (backoff without interpolation loses to the KT blend); `cal` 2.221 (-0.007, SE 0.005);
  `cal_cls` 2.189 (-0.040, SE 0.006 -- 82% of SSE's gain from the escape alone, once the previous character's class
  is in the key); SSE's residual on `cal_cls` -0.030 / -0.035. The full stack cal(_cls) + grid + cls_deep>char reaches
  2.117 (-0.112, 5.0%), the same floor from either chain.

Everything is ONE online pass per chain variant (training slice, then test), the SSE tables and the grid posterior
learning from the first training character on (expC's convention for the tables; expA's posterior was fresh at the
test start -- that variant is kept as the reproduction check `grid_fresh`). Per 10 000-character test block the bits of
every arm are stored, so every difference is a PAIRED difference on the same slice with its standard error over the 6
blocks (research/check_text_slice_noise.py's convention). Per the round-1 checks, every per-context posterior number is
accompanied by its prior cost 36 * log2(K) / n and by the best per-context FIXED grid assignment in hindsight (36
independent argmins), which is the comparator the dominance bound refers to.

Reused unchanged by import: e34 (Chain, Mixer, Stream, build_experts, run_stream), expA_exponent_grid.GridPosterior,
expC_sse_refinement (Bank, place, ctx_values, deepest_info, n_cells, LAMS), mix_temperature_bayes (BETA_C, BETA_R).

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/expH_additivity_and_escape.py --part grid --chain kt
    ... --part grid --chain esc_blend | esc_excl | esc_ppm | esc_cal | esc_cal_cls      ... --part a2 [--warm 240000]      ... --part summary
Each invocation is one pass (< 5 min, CPU) and merges its results into research/expH_additivity_and_escape.json;
`--part summary` prints the cross-pass paired tables from the stored per-block bits.
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
sys.path.insert(0, str(HERE))
from e34 import Chain, Mixer, Stream, build_experts, run_stream      # noqa: E402
from textlm import load_corpus                                       # noqa: E402
import expA_exponent_grid as expA                                    # noqa: E402
import expC_sse_refinement as expC                                   # noqa: E402
from mix_temperature_bayes import BETA_C, BETA_R                     # noqa: E402

N_CTX = expA.N_CTX                                                   # 36 mixing contexts (class x deepest order)
BLOCK = 10000
A_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]                         # expA_exponent_grid_fine's grid
B_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LAM_I = 1                                                            # expC.LAMS[1] = 0.75, the headline lambda
OUT = HERE / "expH_additivity_and_escape.json"
LOG = HERE / "expH_additivity_and_escape.log"
EPS = 1e-12


def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ----------------------------------------------------------------------------------------------------------------------
class EscChain(Chain):
    """e34.Chain with the KT escape replaced by a COUNTED escape per (order, floor(log2 N), distinct symbols) bucket.
    Three structures:
      blend  the chain's own top-down blending kept, only the estimator swapped: each seen level gives its symbols
             (1 - e) cnt/N of the remaining mass and passes e on; e counted UNCONDITIONALLY at every seen level
             (an event per position per seen level; an escape when the character had count 0 there).
      excl   PPM exclusion, built from order 0 up: the symbols seen at a level get (1 - e) cnt/N, the others share e in
             the proportions of the next-shorter level restricted to them (nested levels: S_r is a subset of S_{r-1});
             e counted unconditionally as in `blend`.
      ppm    textbook PPM with exclusions, top-down: at the deepest seen level the symbols get (1 - e) cnt/N; on an
             escape the next level distributes (1 - e) over its symbols NOT already excluded (counts renormalised over
             them), a level whose symbols are all excluded is skipped, the mass left after order 0 is spread over the
             never-seen symbols. The escape is counted CONDITIONALLY on the same walk (an event only at a level that
             was reached; an escape when the character was not there) and the bucket key carries a flag (deepest seen
             level / reached by escape), so e = P(c not in S_r | the level was reached with these exclusions).
      cal    the blend's INTERPOLATION kept, the counted escape fixing the seen/unseen split: bottom-up,
             p_r = w p_{r-1} + (1 - w) cnt/N with w = min(1, e / p_{r-1}(V \ S_r)), so the symbols seen at the level
             get exactly 1 - e in total (the counted rate) and the lower level's opinion about WHICH of them is kept;
             e counted unconditionally as in `blend`.
      cal_cls  `cal` with the previous character's class (letter / space / punctuation / newline, `Stream.cls`) added
             to the bucket key -- the key SSE's best tables carry and the escape lacks.
    `excl` with unconditional counts approximates the conditional escape through its renormalisation over the unseen
    set; feeding the conditional count into that same renormalisation conditions twice (tried, worse), hence `ppm`
    is the top-down form."""

    def __init__(self, V, R=8, mode="blend", ncap=17, dcap=16, cls_of=None):
        super().__init__(V, R)
        self.name, self.mode = f"chain_esc_{mode}", mode
        self.cond = mode == "ppm"
        self.by_cls = mode == "cal_cls"                                # `cal` with the previous character's class in the key
        self.cls_of = cls_of                                           # symbol -> class (letter / space / punct / newline), BORDER -> 3
        self.ncap, self.dcap = ncap, dcap
        self.esc = np.zeros((R + 1, ncap + 1, dcap + 1, 4))
        self.tot = np.zeros((R + 1, ncap + 1, dcap + 1, 4))

    def _bucket(self, r, N, d, flag=0):
        return r, min(N.bit_length() - 1, self.ncap), min(d, self.dcap), flag

    def _cls(self, keys):
        return int(self.cls_of[keys[self.R - 1][0]]) if self.by_cls else 0

    def _e(self, r, N, d, flag=0):
        b = self._bucket(r, N, d, flag)
        return (self.esc[b] + 0.5) / (self.tot[b] + 1.0)

    def prob(self, keys, c):
        if self.mode != "blend":
            return float(self.dist(keys)[0][c])
        p, mass = 0.0, 1.0
        for r, k in zip(range(self.R, -1, -1), keys):
            totals, counts = self.tables[r]
            N = totals.get(k, 0)
            if N:
                cnt = counts[k]
                e = self._e(r, N, len(cnt))
                p += mass * (1.0 - e) * cnt.get(c, 0) / N
                mass *= e
        return p + mass / self.V

    def dist(self, keys):
        V = self.V
        if self.mode == "ppm":
            return self._dist_ppm(keys)
        if self.mode == "excl":
            return self._dist_excl(keys)
        if self.mode in ("cal", "cal_cls"):
            return self._dist_cal(keys)
        p = np.zeros(V)
        mass = 1.0
        for r, k in zip(range(self.R, -1, -1), keys):
            totals, counts = self.tables[r]
            N = totals.get(k, 0)
            if N:
                cnt = counts[k]
                e = self._e(r, N, len(cnt))
                f = mass * (1.0 - e) / N
                for ch, n in cnt.items():
                    p[ch] += f * n
                mass *= e
        return p + mass / V, True

    def _seen_levels(self, keys):
        """[(r, key)] for the seen orders 0..deepest (nested: a prefix of the orders)."""
        out = []
        for r, k in zip(range(0, self.R + 1), reversed(keys)):
            if not self.tables[r][0].get(k, 0):
                break
            out.append((r, k))
        return out

    def _dist_ppm(self, keys):
        V = self.V
        p = np.zeros(V)
        mass = 1.0
        excluded = np.zeros(V, dtype=bool)
        flag = 0
        for r, k in reversed(self._seen_levels(keys)):
            totals, counts = self.tables[r]
            cnt = counts[k]
            idx = np.fromiter(cnt.keys(), dtype=np.int64, count=len(cnt))
            keep = ~excluded[idx]
            if not keep.any():
                continue
            vals = np.fromiter(cnt.values(), dtype=np.float64, count=len(cnt))[keep]
            idx = idx[keep]
            e = self._e(r, totals[k], len(cnt), flag)
            p[idx] = mass * (1.0 - e) * vals / vals.sum()
            mass *= e
            excluded[idx] = True
            flag = 1
        free = ~excluded
        nf = int(free.sum())
        if nf:
            p[free] += mass / nf
        else:
            p /= p.sum()
        return p, True

    def _dist_cal(self, keys):
        V = self.V
        p = np.full(V, 1.0 / V)
        cls = self._cls(keys)
        for r, k in self._seen_levels(keys):
            totals, counts = self.tables[r]
            N = totals[k]
            cnt = counts[k]
            e = self._e(r, N, len(cnt), cls)
            idx = np.fromiter(cnt.keys(), dtype=np.int64, count=len(cnt))
            vals = np.fromiter(cnt.values(), dtype=np.float64, count=len(cnt))
            rest = 1.0 - p[idx].sum()
            if rest > 1e-12:
                w = min(1.0, e / rest)
                q = w * p
                q[idx] += (1.0 - w) * vals / N
                p = q
        return p, True

    def _dist_excl(self, keys):
        V = self.V
        p = np.full(V, 1.0 / V)
        for r, k in self._seen_levels(keys):
            totals, counts = self.tables[r]
            N = totals[k]
            cnt = counts[k]
            e = self._e(r, N, len(cnt))
            idx = np.fromiter(cnt.keys(), dtype=np.int64, count=len(cnt))
            vals = np.fromiter(cnt.values(), dtype=np.float64, count=len(cnt))
            q = p.copy()
            q[idx] = 0.0
            rest = q.sum()
            if rest > 0:
                q *= e / rest
                q[idx] = (1.0 - e) * vals / N
            else:                                                      # every symbol seen here: nothing to escape to
                q[idx] = vals / N
            p = q
        return p, True

    def update(self, keys, c):
        if not self.cond:
            cls = self._cls(keys)
            for r, k in zip(range(self.R, -1, -1), keys):
                totals, counts = self.tables[r]
                N = totals.get(k, 0)
                if N:
                    cnt = counts[k]
                    b = self._bucket(r, N, len(cnt), cls)
                    self.tot[b] += 1.0
                    if c not in cnt:
                        self.esc[b] += 1.0
        else:
            flag = 0
            excluded = set()
            for r, k in reversed(self._seen_levels(keys)):             # deepest first; stop at the level that held c
                totals, counts = self.tables[r]
                cnt = counts[k]
                if excluded and all(ch in excluded for ch in cnt):
                    continue                                           # the level is skipped in `_dist_ppm` too
                b = self._bucket(r, totals[k], len(cnt), flag)
                self.tot[b] += 1.0
                if c in cnt:
                    break
                self.esc[b] += 1.0
                excluded.update(cnt.keys())
                flag = 1
        super().update(keys, c)

    def escape_table(self):
        """The counted escape rate per (order, log2 N) with d and the flag marginalised -- for the eye."""
        out = {}
        for r in range(self.R + 1):
            row = {}
            for nb in range(self.ncap + 1):
                t, e = self.tot[r, nb].sum(), self.esc[r, nb].sum()
                if t >= 50:
                    row[str(nb)] = dict(events=int(t), rate=float(e / t))
            out[str(r)] = row
        return out


def make_chain(kind, V, alphabet=None):
    if kind == "kt":
        return Chain(V)
    mode = kind.split("_", 1)[1]
    cls_of = None
    if mode == "cal_cls":
        s = Stream(V, alphabet)
        cls_of = np.array([0 if i in s.letter else 1 if i in s.space else 2 if i in s.newline else 3 for i in range(V)] + [3], dtype=np.int64)
    return EscChain(V, mode=mode, cls_of=cls_of)


# ----------------------------------------------------------------------------------------------------------------------
class SSE:
    """expC's Bank machinery on an arbitrary input distribution: the single tables `conf_cls` and `cls_deep`, and the
    two-stage `cls_deep>char` (a per-character table on the first table's lam = 3/4 output), B = 24, no cap, no interp."""

    def __init__(self, nc, V, B=24):
        base = dict(B=B, cap=0, interp=False, fresh=False)
        self.bank = expC.Bank([dict(base, name="conf_cls", ctx="conf_cls"), dict(base, name="cls_deep", ctx="cls_deep")], nc, V, B, False)
        self.stack = expC.Bank([dict(base, name="cls_deep>char", ctx="char")], nc, V, B, False)
        self.i_first = self.bank.names.index("cls_deep")
        self.names = list(self.bank.names) + list(self.stack.names)
        self.snaps = []

    @staticmethod
    def _stretch(p):
        pc = np.clip(p, EPS, 1.0 - EPS)
        return np.log(pc / (1.0 - pc))

    def step(self, p, cv, c, testing, i):
        placed = expC.place(self._stretch(p), self.bank.B, False)
        rows = self.bank.rows(cv)
        R = self.bank.refine(rows, p, placed)
        if testing:
            self.bank.score(p, R, c)
        q = 0.25 * p + 0.75 * R[self.i_first]
        q = q / q.sum()
        pl2 = expC.place(self._stretch(q), self.stack.B, False)
        rows2 = self.stack.rows(cv)
        R2 = self.stack.refine(rows2, q, pl2)
        if testing:
            self.stack.score(q, R2, c)
        self.stack.update(rows2, c, q, pl2, i)
        self.bank.update(rows, c, p, placed, i)

    def snapshot(self):
        self.snaps.append(np.concatenate([self.bank.bits, self.stack.bits], axis=1).copy())     # (L, 3)

    def blocks(self, sizes):
        """name -> {lam: per-block bits/char}."""
        cum = np.stack([np.zeros_like(self.snaps[0])] + self.snaps)                            # (nb + 1, L, 3)
        per = np.diff(cum, axis=0)                                                             # (nb, L, 3)
        return {nm: {str(l): (per[:, j, k] / sizes).tolist() for j, l in enumerate(expC.LAMS)} for k, nm in enumerate(self.names)}


# ----------------------------------------------------------------------------------------------------------------------
def paired(blk_a, blk_b, sizes):
    """Paired difference a - b (bits/char) over the test blocks: mean (size-weighted total), SE over blocks."""
    a, b = np.asarray(blk_a), np.asarray(blk_b)
    d = a - b
    return dict(mean=float(((a - b) * sizes).sum() / sizes.sum()), se=float(d.std(ddof=1) / math.sqrt(len(d))),
                per_block=[float(x) for x in d], n_neg=int((d < 0).sum()))


def total(blk, sizes):
    return float((np.asarray(blk) * sizes).sum() / sizes.sum())


def fmt(d):
    return f"{d['mean']:+.4f} (SE {d['se']:.4f}, {d['n_neg']}/{len(d['per_block'])} blocks < 0)"


# ----------------------------------------------------------------------------------------------------------------------
def run_grid_pass(kind, text, test, V, alphabet):
    """One online pass: the chain of `kind` + word, the fine-grid per-context posterior, SSE on the chain and on the
    grid output. Returns per-block bits of every arm."""
    chain = make_chain(kind, V, alphabet)
    word = next(e for e in build_experts(V, alphabet) if e.name == "word")
    mixer = Mixer(2, N_CTX)                                            # E34's linear mixer, for the check only
    s = Stream(V, alphabet)
    grid = [(a, b) for a in A_GRID for b in B_GRID]
    A = np.array([g[0] for g in grid]); B = np.array([g[1] for g in grid])
    K = len(grid)
    post_warm = expA.GridPosterior(K, N_CTX, alpha=0.01)              # learns from the first training character
    post_fresh = expA.GridPosterior(K, N_CTX, alpha=0.01)             # expA's convention: from the test start
    per_ctx = np.zeros((N_CTX, K))                                     # test bits per (context, grid point)
    ctx_n = np.zeros(N_CTX)
    nc = expC.n_cells(V)
    sse = {"chain": SSE(nc, V), "grid": SSE(nc, V)}
    n_train, n_test = len(text), len(test)
    nb = math.ceil(n_test / BLOCK)
    sizes = np.array([min(BLOCK, n_test - b * BLOCK) for b in range(nb)], dtype=np.float64)
    names = ("chain", "word", "linear", "grid_warm", "grid_fresh")
    blk = {k: np.zeros(nb) for k in names}
    seq = np.concatenate([text, test]).tolist()
    probs = np.zeros(2)
    t0 = time.time()
    for i, c in enumerate(seq):
        testing = i >= n_train
        b = (i - n_train) // BLOCK if testing else -1
        keys = chain.predict(s)
        dC, _ = chain.dist(keys)
        deep = chain.deepest(keys)
        info = expC.deepest_info(chain, keys, deep)
        cv = expC.ctx_values(s, deep, info)
        ctx = s.cls() * 9 + deep
        k, N, cnt = word.predict(s)
        dO, _seen = word.dist(k, N, cnt)
        logC = np.log(np.clip(dC, 1e-300, None)); logO = np.log(np.clip(dO, 1e-300, None))
        L = A[:, None] * logC[None, :] + B[:, None] * logO[None, :]
        L -= L.max(1, keepdims=True)
        G = np.exp(L); G /= G.sum(1, keepdims=True)
        pg = G[:, c]
        q_grid = post_warm.w[ctx] @ G                                  # the posterior-weighted mixture: the grid's OUTPUT
        bits_warm = post_warm.code(ctx, pg)                            # = -log2(q_grid[c]); updates the posterior
        probs[0], probs[1] = dC[c], dO[c]
        p_lin, _w = mixer.mix(ctx, probs)
        mixer.update(ctx, probs, p_lin)
        if testing:
            blk["chain"][b] -= math.log2(dC[c])
            blk["word"][b] -= math.log2(dO[c])
            blk["linear"][b] -= math.log2(p_lin)
            blk["grid_warm"][b] += bits_warm
            blk["grid_fresh"][b] += post_fresh.code(ctx, pg)
            per_ctx[ctx] -= np.log2(np.clip(pg, 1e-300, None))
            ctx_n[ctx] += 1
        sse["chain"].step(dC, cv, c, testing, i)
        sse["grid"].step(np.clip(q_grid, EPS, None), cv, c, testing, i)
        chain.update(keys, c)
        word.update(k, c)
        s.push(c)
        if testing and ((i - n_train + 1) % BLOCK == 0 or i == len(seq) - 1):
            for v in sse.values():
                v.snapshot()
        if i == n_train - 1:
            log(f"   [{kind}] training slice done ({n_train} chars) [{time.time() - t0:.0f}s]")
    secs = time.time() - t0
    res = dict(kind=kind, n_train=n_train, n_test=n_test, block=BLOCK, sizes=sizes.tolist(), seconds=secs, grid=grid,
               blocks={k: (v / sizes).tolist() for k, v in blk.items()})
    for src, v in sse.items():
        for nm, d in v.blocks(sizes).items():
            res["blocks"][f"{src}+sse_{nm}"] = d[str(expC.LAMS[LAM_I])]
            for l, arr in d.items():
                res.setdefault("all_lams", {})[f"{src}+sse_{nm}@{l}"] = total(arr, sizes)
    # the per-context comparator (round-1 checks): best fixed grid point per context in hindsight, and the prior cost
    best_ctx = per_ctx.min(1)
    res["best_per_ctx_fixed"] = float(best_ctx.sum() / n_test)
    res["best_global_fixed"] = float(per_ctx.sum(0).min() / n_test)
    res["best_global_fixed_ab"] = grid[int(np.argmin(per_ctx.sum(0)))]
    res["prior_cost_per_ctx"] = float(N_CTX * math.log2(K) / n_test)
    res["best_per_ctx_ab"] = {str(cx): grid[int(np.argmin(per_ctx[cx]))] for cx in range(N_CTX) if ctx_n[cx] >= 500}
    res["posterior_mean_ab_by_ctx"] = {str(cx): [float(post_warm.w[cx] @ A), float(post_warm.w[cx] @ B)] for cx in range(N_CTX) if ctx_n[cx] >= 500}
    if isinstance(chain, EscChain):
        res["escape_table"] = chain.escape_table()
    return res


def report_grid(res):
    sizes = np.asarray(res["sizes"])
    bl = res["blocks"]
    T = {k: total(v, sizes) for k, v in bl.items()}
    kind = res["kind"]
    log(f"   [{kind}] totals, bits/char online on the test slice ({res['n_test']} chars, {len(sizes)} blocks) [{res['seconds']:.0f}s]:")
    for k in ("chain", "word", "linear", "grid_fresh", "grid_warm", "chain+sse_conf_cls", "chain+sse_cls_deep>char",
              "grid+sse_conf_cls", "grid+sse_cls_deep>char"):
        log(f"      {k:<26} {T[k]:.4f}   vs chain {fmt(paired(bl[k], bl['chain'], sizes))}")
    log(f"      best per-context fixed (36 argmins, hindsight) {res['best_per_ctx_fixed']:.4f}; best global fixed {res['best_global_fixed']:.4f} at (a,b)={tuple(res['best_global_fixed_ab'])}; "
        f"per-context prior cost 36*log2(63)/n = {res['prior_cost_per_ctx']:.4f}")
    d_sse = paired(bl["chain+sse_conf_cls"], bl["chain"], sizes)
    d_grid = paired(bl["grid_warm"], bl["chain"], sizes)
    d_both = paired(bl["grid+sse_conf_cls"], bl["chain"], sizes)
    inter = np.asarray(d_both["per_block"]) - np.asarray(d_sse["per_block"]) - np.asarray(d_grid["per_block"])
    log(f"      ADDITIVITY (conf_cls): sse alone {d_sse['mean']:+.4f}, grid alone {d_grid['mean']:+.4f}, sum {d_sse['mean'] + d_grid['mean']:+.4f}, "
        f"measured grid+sse {d_both['mean']:+.4f}; interaction {inter.mean():+.4f} (SE {inter.std(ddof=1) / math.sqrt(len(inter)):.4f})")
    d_sse2 = paired(bl["chain+sse_cls_deep>char"], bl["chain"], sizes)
    d_both2 = paired(bl["grid+sse_cls_deep>char"], bl["chain"], sizes)
    inter2 = np.asarray(d_both2["per_block"]) - np.asarray(d_sse2["per_block"]) - np.asarray(d_grid["per_block"])
    log(f"      ADDITIVITY (cls_deep>char): sse alone {d_sse2['mean']:+.4f}, grid alone {d_grid['mean']:+.4f}, sum {d_sse2['mean'] + d_grid['mean']:+.4f}, "
        f"measured {d_both2['mean']:+.4f}; interaction {inter2.mean():+.4f} (SE {inter2.std(ddof=1) / math.sqrt(len(inter2)):.4f})")
    log(f"      sse on top of the grid, vs the grid: conf_cls {fmt(paired(bl['grid+sse_conf_cls'], bl['grid_warm'], sizes))}; "
        f"cls_deep>char {fmt(paired(bl['grid+sse_cls_deep>char'], bl['grid_warm'], sizes))}")
    if "escape_table" in res:
        log("      counted escape rate by (order, floor(log2 N)) [events >= 50], d marginalised; KT's (V/2)/(N+V/2) at the bucket's lower N in brackets:")
        for r, row in res["escape_table"].items():
            cells = "  ".join(f"2^{nb}: {v['rate']:.3f} [{35 / (2 ** int(nb) + 35):.3f}]" for nb, v in row.items())
            log(f"         order {r}: {cells}")


# ----------------------------------------------------------------------------------------------------------------------
def run_a2_pass(text, test, V, alphabet, warm):
    """A2-min (mix_temperature_bayes --switch decay) with SSE on its output. The 18 experts and the linear mixer are
    trained by run_stream on the first n_train - warm characters; the last `warm` training characters and the test run
    through the full mixture machinery (the grid2 posterior and the SSE tables learning), so both the chain's SSE and
    A2's SSE have the same warm-up."""
    experts = build_experts(V, alphabet)
    chain = experts[0]
    n_pre = len(text) - warm
    t0 = time.time()
    _tb, _ts, s, mixer = run_stream(experts, text[:n_pre], V, alphabet, track_solo=False)
    log(f"   [a2] experts + mixer trained on {n_pre} chars [{time.time() - t0:.0f}s]; warm-up {warm} chars, then {len(test)} test")
    grid2 = np.array([(bc, br) for bc in BETA_C for br in BETA_R])
    K = len(grid2)
    post_warm = np.full((N_CTX, K), 1.0 / K); used_warm = np.zeros(N_CTX)
    post_fresh = np.full((N_CTX, K), 1.0 / K); used_fresh = np.zeros(N_CTX)
    per_ctx = np.zeros((N_CTX, K))
    nc = expC.n_cells(V)
    sse = {"chain": SSE(nc, V), "a2": SSE(nc, V)}
    n_test = len(test)
    nb = math.ceil(n_test / BLOCK)
    sizes = np.array([min(BLOCK, n_test - b * BLOCK) for b in range(nb)], dtype=np.float64)
    names = ("chain", "linear", "geometric", "a2_warm", "a2_fresh")
    blk = {k: np.zeros(nb) for k in names}
    seq = np.concatenate([text[n_pre:], test]).tolist()
    probs = np.zeros(len(experts))
    for i, c in enumerate(seq):
        testing = i >= warm
        b = (i - warm) // BLOCK if testing else -1
        preds = [e.predict(s) for e in experts]
        dists, seen = [], []
        for e, pr in zip(experts, preds):
            d, ok = e.dist(pr) if isinstance(e, Chain) else e.dist(pr[0], pr[1], pr[2])
            dists.append(d); seen.append(ok)
        D = np.stack(dists)
        probs[:] = D[:, c]
        keys = preds[0]
        deep = chain.deepest(keys)
        info = expC.deepest_info(chain, keys, deep)
        cv = expC.ctx_values(s, deep, info)
        ctx = s.cls() * 9 + deep
        p_lin, w = mixer.mix(ctx, probs)
        sm = np.array(seen)
        logD = np.log(np.clip(D[sm], 1e-12, None))
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
        pi = post_warm[ctx]
        q_a2 = pi @ G2                                                  # A2-min's OUTPUT distribution
        pm = float(pi @ p2)
        a = 1.0 / (used_warm[ctx] + 2)
        post_warm[ctx] = (1 - a) * (pi * p2 / pm) + a / K
        used_warm[ctx] += 1
        if testing:
            blk["chain"][b] -= math.log2(D[0, c])
            blk["linear"][b] -= math.log2(p_lin)
            wg = w[sm] / w[sm].sum()
            g = np.exp(wg @ logD); g /= g.sum()
            blk["geometric"][b] -= math.log2(max(g[c], 1e-12))
            blk["a2_warm"][b] -= math.log2(pm)
            pf = post_fresh[ctx]
            pmf = float(pf @ p2)
            blk["a2_fresh"][b] -= math.log2(pmf)
            af = 1.0 / (used_fresh[ctx] + 2)
            post_fresh[ctx] = (1 - af) * (pf * p2 / pmf) + af / K
            used_fresh[ctx] += 1
            per_ctx[ctx] -= np.log2(p2)
        sse["chain"].step(D[0], cv, c, testing, i)
        sse["a2"].step(np.clip(q_a2, EPS, None), cv, c, testing, i)
        mixer.update(ctx, probs, p_lin)
        for e, pr in zip(experts, preds):
            if isinstance(e, Chain):
                e.update(pr, c)
            else:
                e.update(pr[0], c)
        s.push(c)
        if testing and ((i - warm + 1) % BLOCK == 0 or i == len(seq) - 1):
            for v in sse.values():
                v.snapshot()
        if i == warm - 1:
            log(f"   [a2] warm-up done [{time.time() - t0:.0f}s]")
    secs = time.time() - t0
    res = dict(kind="a2", n_train=len(text), warm=warm, n_test=n_test, block=BLOCK, sizes=sizes.tolist(), seconds=secs,
               grid2=grid2.tolist(), blocks={k: (v / sizes).tolist() for k, v in blk.items()})
    for src, v in sse.items():
        for nm, d in v.blocks(sizes).items():
            res["blocks"][f"{src}+sse_{nm}"] = d[str(expC.LAMS[LAM_I])]
            for l, arr in d.items():
                res.setdefault("all_lams", {})[f"{src}+sse_{nm}@{l}"] = total(arr, sizes)
    res["best_per_ctx_fixed"] = float(per_ctx.min(1).sum() / n_test)
    res["best_global_fixed"] = float(per_ctx.sum(0).min() / n_test)
    res["best_global_fixed_point"] = grid2[int(np.argmin(per_ctx.sum(0)))].tolist()
    res["prior_cost_per_ctx"] = float(N_CTX * math.log2(K) / n_test)
    return res


def report_a2(res):
    sizes = np.asarray(res["sizes"])
    bl = res["blocks"]
    T = {k: total(v, sizes) for k, v in bl.items()}
    log(f"   [a2] totals, bits/char online on the test slice ({res['n_test']} chars; SSE warm-up {res['warm']} chars) [{res['seconds']:.0f}s]:")
    for k in ("chain", "linear", "geometric", "a2_fresh", "a2_warm", "chain+sse_conf_cls", "chain+sse_cls_deep>char",
              "a2+sse_conf_cls", "a2+sse_cls_deep>char"):
        log(f"      {k:<26} {T[k]:.4f}   vs chain {fmt(paired(bl[k], bl['chain'], sizes))}")
    log(f"      best per-context fixed (36 argmins, hindsight) {res['best_per_ctx_fixed']:.4f}; best global fixed {res['best_global_fixed']:.4f} at "
        f"(beta_chain, beta_rest)={tuple(res['best_global_fixed_point'])}; per-context prior cost 36*log2(42)/n = {res['prior_cost_per_ctx']:.4f}")
    for nm in ("conf_cls", "cls_deep>char"):
        d_sse = paired(bl[f"chain+sse_{nm}"], bl["chain"], sizes)
        d_a2 = paired(bl["a2_warm"], bl["chain"], sizes)
        d_both = paired(bl[f"a2+sse_{nm}"], bl["chain"], sizes)
        inter = np.asarray(d_both["per_block"]) - np.asarray(d_sse["per_block"]) - np.asarray(d_a2["per_block"])
        log(f"      ADDITIVITY ({nm}): sse alone {d_sse['mean']:+.4f}, A2 alone {d_a2['mean']:+.4f}, sum {d_sse['mean'] + d_a2['mean']:+.4f}, "
            f"measured A2+sse {d_both['mean']:+.4f}; interaction {inter.mean():+.4f} (SE {inter.std(ddof=1) / math.sqrt(len(inter)):.4f}); "
            f"sse on top of A2 vs A2 {fmt(paired(bl[f'a2+sse_{nm}'], bl['a2_warm'], sizes))}")


# ----------------------------------------------------------------------------------------------------------------------
def summary(report):
    """Cross-pass paired tables from the stored per-block bits (the same test slice and blocks in every pass)."""
    parts = report.get("parts", {})
    if "grid_kt" not in parts:
        log("   no grid_kt pass stored")
        return
    kt = parts["grid_kt"]
    sizes = np.asarray(kt["sizes"])
    base = np.asarray(kt["blocks"]["chain"])
    log("\n=== SUMMARY: every arm as a paired difference vs the KT chain alone (same 60k slice, SE over the 6 blocks of 10k) ===")
    rows = []
    for key, label in (("grid_kt", "KT chain"), ("grid_esc_blend", "counted escape, blend"), ("grid_esc_excl", "counted escape, exclusion"), ("grid_esc_ppm", "counted escape, PPM (conditional)"), ("grid_esc_cal", "counted escape, calibrated blend"), ("grid_esc_cal_cls", "counted escape, calibrated blend, key + class")):
        if key not in parts:
            continue
        r = parts[key]
        for arm in ("chain", "chain+sse_conf_cls", "chain+sse_cls_deep>char", "grid_warm", "grid+sse_conf_cls", "grid+sse_cls_deep>char"):
            d = paired(r["blocks"][arm], base, sizes)
            rows.append((f"{label}: {arm}", total(r["blocks"][arm], sizes), d))
    for key in sorted(k for k in parts if k.startswith("a2")):
        r = parts[key]
        for arm in ("chain", "a2_fresh", "a2_warm", "chain+sse_conf_cls", "chain+sse_cls_deep>char", "a2+sse_conf_cls", "a2+sse_cls_deep>char"):
            d = paired(r["blocks"][arm], base, sizes)
            rows.append((f"A2 pass (SSE warm {r['warm']}): {arm}", total(r["blocks"][arm], sizes), d))
    for label, t, d in rows:
        log(f"   {label:<58} {t:.4f}   {fmt(d)}")
    # the escape question: does SSE's gain on the chain shrink once the escape is counted?
    for key, label in (("grid_esc_blend", "blend"), ("grid_esc_excl", "excl"), ("grid_esc_ppm", "ppm"), ("grid_esc_cal", "cal"), ("grid_esc_cal_cls", "cal_cls")):
        if key in parts:
            r = parts[key]
            for nm in ("conf_cls", "cls_deep>char"):
                d_kt = paired(kt["blocks"][f"chain+sse_{nm}"], kt["blocks"]["chain"], sizes)
                d_esc = paired(r["blocks"][f"chain+sse_{nm}"], r["blocks"]["chain"], sizes)
                log(f"   SSE {nm}'s gain on its own chain: KT {d_kt['mean']:+.4f} (SE {d_kt['se']:.4f}) -> counted escape ({label}) {d_esc['mean']:+.4f} (SE {d_esc['se']:.4f}); "
                    f"counted-escape chain alone vs KT chain + SSE: {fmt(paired(r['blocks']['chain'], kt['blocks'][f'chain+sse_{nm}'], sizes))}")
            d_g = paired(r["blocks"]["grid_warm"], r["blocks"]["chain"], sizes)
            d_gk = paired(kt["blocks"]["grid_warm"], kt["blocks"]["chain"], sizes)
            log(f"   grid's gain on its own chain: KT {d_gk['mean']:+.4f} -> counted escape ({label}) {d_g['mean']:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="grid", choices=("grid", "a2", "summary"))
    ap.add_argument("--chain", default="kt", choices=("kt", "esc_blend", "esc_excl", "esc_ppm", "esc_cal", "esc_cal_cls"))
    ap.add_argument("--train_chars", type=int, default=300000)
    ap.add_argument("--test_chars", type=int, default=60000)
    ap.add_argument("--warm", type=int, default=60000, help="a2: training characters run through the full machinery before the test")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    report = json.load(open(args.out)) if Path(args.out).exists() else dict(parts={})
    if args.part == "summary":
        summary(report)
        return
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)[:args.train_chars]
    test = test[:args.test_chars]
    log(f"Experiment H ({args.part}{', chain ' + args.chain if args.part == 'grid' else ''}): {len(text)} training characters (prefix of the concatenated books), "
        f"{len(test)} held-out characters (start of De Bello Civili), V = {V}; blocks of {BLOCK}")
    if args.part == "grid":
        res = run_grid_pass(args.chain, text, test, V, alphabet)
        report_grid(res)
        key = f"grid_{args.chain}"
    else:
        res = run_a2_pass(text, test, V, alphabet, args.warm)
        report_a2(res)
        key = f"a2_w{args.warm // 1000}k"
    report = json.load(open(args.out)) if Path(args.out).exists() else dict(parts={})   # re-read: passes may run in parallel
    report.setdefault("parts", {})[key] = res
    report["V"], report["n_train"], report["n_test"] = V, int(len(text)), int(len(test))
    json.dump(report, open(args.out, "w"), indent=1)
    log(f"   saved {args.out}")


if __name__ == "__main__":
    main()
