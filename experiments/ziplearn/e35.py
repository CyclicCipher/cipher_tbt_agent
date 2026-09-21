"""E35 — the written denoising chain (DESIGN §20; pre-registered 2026-09-21). The looped block fitted BLOCKWISE by
counting, no gradient anywhere: the diffusion interpretation gives every block a known target (the data at the next
noise level), so each block is a table learned from its own (noisier, cleaner) pairs, priced by the two-part code and
compressed by the sleep pass (the price alone deciding, `strict=False`: with noisy targets the no-new-exceptions rule
of E24 forbids every drop); running the chain from pure noise is generation, from level k is repair.

Domain 1 — the replica games' frames. Clean data: the frames stood in during exploration of LockPath and CollectAll
(levels 0–1 for training, level 2 held out). Corruption: each cell replaced by a uniform random colour with probability
t, on the schedule t_k = k/K (K = 8), coupled across levels (a cell corrupted at level k-1 keeps its noise value at level
k). Block k is a `LocalRule` (radius 1, the same table at every cell) from the window at t_k to the cell's value at
t_{k-1}. Prediction for a window never seen: the Hamming-NEAREST stored window (E28's default) -- the block form's soft
attention, done exactly. Measured: REPAIR of held-out frames corrupted at t = 0.25 / 0.5 / 0.75, chain against the
ONE-SHOT control (one table from the corrupted window straight to the clean cell, same window, same default);
GENERATION from pure noise (agreement with the nearest training frame; distinct samples); each block's table before and
after sleep.

Domain 2 — text (`corpora/latin books`, E33's split): masked-character diffusion, absorbing (a masked character stays
masked until a block fills it); block k fills masked characters from a window of ±3 characters in which masks are
visible; K = 4; prediction for an unseen window backs off by shrinking the window symmetrically. Measured:
reconstruction accuracy of masked characters on the held-out book at t = 0.15 / 0.5, chain against one shot.

    python experiments/ziplearn/e35.py           (CPU, ~10 min) -> runs/e35/e35.json
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
sys.path.insert(0, str(HERE.parent.parent / "src"))
from arcgames import LocalRule, play, V as VC                      # noqa: E402
from textlm import load_corpus, training_slice                     # noqa: E402
from tasks.games import LockPath, CollectAll                       # noqa: E402
from tasks.harness import Environment                              # noqa: E402


# ── domain 1: frames ─────────────────────────────────────────────────────────────────────────────────────────────
def collect_frames(cls, levels, budget=150):
    """The distinct frames the explorer stood in, per level."""
    out = {}
    trace = []
    play(Environment(cls()), budget_per_level=budget, max_levels=max(levels) + 1, sleep=True, trace=trace)
    for r in trace:
        if r["level"] in levels:
            out.setdefault(r["level"], {})[str(r["frame"])] = np.array(r["frame"], dtype=np.int16)
    return {lvl: list(d.values()) for lvl, d in out.items()}


def corrupt_chain(x0, schedule, rng):
    """Coupled corruptions x_{t_0}=x0, x_{t_1}, ..., x_{t_K}: a cell corrupted at a level keeps its noise value after."""
    xs = [x0.copy()]
    corrupted = np.zeros(x0.shape, dtype=bool)
    cur = x0.copy()
    for k in range(1, len(schedule)):
        t_prev, t = schedule[k - 1], schedule[k]
        p_new = (t - t_prev) / (1.0 - t_prev) if t_prev < 1.0 else 0.0
        new = (~corrupted) & (rng.random(x0.shape) < p_new)
        cur = cur.copy()
        cur[new] = rng.integers(0, VC, size=int(new.sum()))
        corrupted |= new
        xs.append(cur)
    return xs


class NearestRule(LocalRule):
    """A LocalRule whose prediction for a window never seen is the Hamming-nearest stored window's majority."""

    def _keys_array(self):
        if getattr(self, "_karr", None) is None or self._karr_n != len(self.table):
            keys = list(self.table.keys())
            self._karr = np.stack([np.frombuffer(k, dtype=np.int16) for k in keys]) if keys else np.zeros((0, int(self.mask.sum())), np.int16)
            self._kval = np.array([self.majority[k] for k in keys], dtype=np.int16)
            self._karr_n = len(self.table)
        return self._karr, self._kval

    def predict_nearest(self, frame):
        _full, masked = self._rows(frame)
        out = frame.copy().reshape(-1)
        unknown = []
        for t in range(out.size):
            y = self.majority.get(masked[t].tobytes())
            if y is None:
                unknown.append(t)
            else:
                out[t] = y
        if unknown:
            karr, kval = self._keys_array()
            if len(karr):
                q = masked[unknown]                                    # (u, cells)
                for s in range(0, len(q), 256):
                    d = (q[s:s + 256, None, :] != karr[None, :, :]).sum(-1)
                    out[np.array(unknown[s:s + 256])] = kval[d.argmin(1)]
        return out.reshape(frame.shape), len(unknown)


def fit_block(pairs, sleep=True, strict=False):
    rule = NearestRule(1)
    for before, after in pairs:
        rule.observe(before, after)
    n_before = len(rule.table)
    b, a, cells = rule.sleep(strict=strict) if sleep else (rule.cost, rule.cost, 9)   # strict=False: the price alone (noisy targets)
    return rule, dict(entries_before=n_before, entries_after=len(rule.table), bits_before=round(b), bits_after=round(a), cells=cells)


def run_frames(rng, K=8, draws=4, n_gen=100, verbose=True, strict=False):
    schedule = [k / K for k in range(K + 1)]
    lp = collect_frames(LockPath, [0, 1, 2])
    ca = collect_frames(CollectAll, [0, 1, 2])
    train = lp[0] + lp[1] + ca[0] + ca[1]
    held = lp.get(2, []) + ca.get(2, [])
    if verbose:
        print(f"   frames: {len(train)} training (LockPath 0-1: {len(lp[0]) + len(lp[1])}, CollectAll 0-1: {len(ca[0]) + len(ca[1])}), "
              f"{len(held)} held out (level 2 of each)", flush=True)
    # coupled corruption chains of the training frames
    chains = [corrupt_chain(x, schedule, rng) for x in train for _ in range(draws)]
    blocks, block_stats = {}, {}
    t0 = time.time()
    for k in range(1, K + 1):
        pairs = [(c[k], c[k - 1]) for c in chains]
        blocks[k], block_stats[k] = fit_block(pairs, strict=strict)
        if verbose:
            s = block_stats[k]
            print(f"   block {k} (t {schedule[k]:.3f} -> {schedule[k - 1]:.3f}): {s['entries_before']} -> {s['entries_after']} entries, "
                  f"{s['bits_before']} -> {s['bits_after']} bits, {s['cells']} cells kept  [{time.time() - t0:.0f}s]", flush=True)
    # the one-shot controls: from the corrupted level straight to clean
    oneshot = {}
    for k in (2, 4, 6):
        pairs = [(c[k], c[0]) for c in chains]
        oneshot[k], _ = fit_block(pairs, strict=strict)
    # repair on held-out frames
    repair = {}
    for k in (2, 4, 6):
        acc_chain, acc_one, acc_noisy = [], [], []
        for x in held:
            for _ in range(3):
                xs = corrupt_chain(x, schedule, rng)
                noisy = xs[k]
                cur = noisy
                for j in range(k, 0, -1):
                    cur, _u = blocks[j].predict_nearest(cur)
                one, _u = oneshot[k].predict_nearest(noisy)
                acc_chain.append(float((cur == x).mean()))
                acc_one.append(float((one == x).mean()))
                acc_noisy.append(float((noisy == x).mean()))
        repair[k] = dict(t=schedule[k], chain=float(np.mean(acc_chain)), one_shot=float(np.mean(acc_one)), noisy=float(np.mean(acc_noisy)))
        if verbose:
            r = repair[k]
            print(f"   repair at t = {r['t']:.2f}: chain {r['chain']:.3f}, one shot {r['one_shot']:.3f} (corrupted input {r['noisy']:.3f} correct)", flush=True)
    # generation from pure noise, at the training frames' shape
    shapes = {}
    for x in train:
        shapes.setdefault(x.shape, 0)
        shapes[x.shape] += 1
    gen_stats = []
    samples = []
    for i in range(n_gen):
        shape = list(shapes)[i % len(shapes)]
        cur = rng.integers(0, VC, size=shape).astype(np.int16)
        for j in range(K, 0, -1):
            cur, _u = blocks[j].predict_nearest(cur)
        same = [x for x in train if x.shape == cur.shape]
        agree = max(float((cur == x).mean()) for x in same)
        gen_stats.append(agree)
        samples.append(cur.tobytes())
    generation = dict(agreement_nearest_training=float(np.mean(gen_stats)), distinct=len(set(samples)), n=n_gen)
    if verbose:
        print(f"   generation from noise: agreement with the nearest training frame {generation['agreement_nearest_training']:.3f}, "
              f"{generation['distinct']} distinct of {n_gen}", flush=True)
        cur = rng.integers(0, VC, size=list(shapes)[0]).astype(np.int16)
        for j in range(K, 0, -1):
            cur, _u = blocks[j].predict_nearest(cur)
        print("   one sample:\n" + "\n".join("      " + " ".join(f"{v:2d}" for v in row) for row in cur), flush=True)
    return dict(schedule=schedule, n_train=len(train), n_held=len(held), blocks=block_stats, repair=repair, generation=generation)


# ── domain 2: text, masked-character diffusion ─────────────────────────────────────────────────────────────────
class SeqRule:
    """A bidirectional window table over a 1-D sequence: window of ±r around a position (MASK visible) -> the centre's
    value at the cleaner level; unseen windows back off by shrinking the window symmetrically."""

    def __init__(self, V, r=3):
        self.V, self.r = V, r
        self.MASK = V                                                  # symbol index of the mask
        self.base = V + 2                                              # symbols, MASK, BORDER
        self.BORDER = V + 1
        self.tables = {}                                               # radius -> dict key -> counts dict

    def _windows(self, seq, r):
        pad = np.concatenate([np.full(r, self.BORDER, np.int64), seq.astype(np.int64), np.full(r, self.BORDER, np.int64)])
        n = len(seq)
        key = np.zeros(n, dtype=np.int64)
        for off in range(-r, r + 1):
            key = key * self.base + pad[r + off: r + off + n]
        return key

    def observe(self, before, after):
        for r in range(self.r, -1, -1):
            keys = self._windows(before, r)
            tab = self.tables.setdefault(r, {})
            for k, y in zip(keys.tolist(), after.tolist()):
                d = tab.get(k)
                if d is None:
                    tab[k] = {y: 1}
                else:
                    d[y] = d.get(y, 0) + 1

    def predict(self, seq):
        out = seq.copy()
        pending = np.ones(len(seq), dtype=bool)
        for r in range(self.r, -1, -1):
            keys = self._windows(seq, r)
            tab = self.tables[r]
            for i in np.flatnonzero(pending):
                d = tab.get(int(keys[i]))
                if d is not None:
                    out[i] = max(d, key=d.get)
                    pending[i] = False
        return out


def mask_chain(x0, schedule, rng, MASK):
    xs = [x0.copy()]
    masked = np.zeros(len(x0), dtype=bool)
    cur = x0.copy()
    for k in range(1, len(schedule)):
        t_prev, t = schedule[k - 1], schedule[k]
        p_new = (t - t_prev) / (1.0 - t_prev) if t_prev < 1.0 else 0.0
        new = (~masked) & (rng.random(len(x0)) < p_new)
        cur = cur.copy()
        cur[new] = MASK
        masked |= new
        xs.append(cur)
    return xs


def run_text(rng, K=4, n_train=300000, n_test=60000, chunk=64, verbose=True):
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    seqs = training_slice(train, n_train)
    text = np.concatenate(seqs).astype(np.int16)
    test = test[:n_test].astype(np.int16)
    schedule = [k / K for k in range(K + 1)]
    MASK = V
    chunks = [text[i:i + chunk] for i in range(0, len(text) - chunk, chunk)]
    blocks = {}
    t0 = time.time()
    chains = [mask_chain(c, schedule, rng, MASK) for c in chunks]
    for k in range(1, K + 1):
        rule = SeqRule(V)
        for ch in chains:
            rule.observe(ch[k], ch[k - 1])
        blocks[k] = rule
        if verbose:
            print(f"   text block {k} (mask {schedule[k]:.2f} -> {schedule[k - 1]:.2f}): {len(rule.tables[3])} full windows  [{time.time() - t0:.0f}s]", flush=True)
    oneshot = {}
    for k in (1, 2):
        rule = SeqRule(V)
        for ch in chains:
            rule.observe(ch[k], ch[0])
        oneshot[k] = rule
    tchunks = [test[i:i + chunk] for i in range(0, len(test) - chunk, chunk)]
    results = {}
    for t_eval, k in ((0.15, 1), (0.5, 2)):                          # run the chain from the first level at or above t
        acc_chain, acc_one, n_masked = 0, 0, 0
        for c in tchunks:
            m = rng.random(len(c)) < t_eval
            noisy = c.copy(); noisy[m] = MASK
            cur = noisy
            for j in range(k, 0, -1):
                cur = blocks[j].predict(cur)
            one = oneshot[k].predict(noisy)
            acc_chain += int((cur[m] == c[m]).sum()); acc_one += int((one[m] == c[m]).sum()); n_masked += int(m.sum())
        results[str(t_eval)] = dict(t=t_eval, chain=acc_chain / n_masked, one_shot=acc_one / n_masked, n_masked=n_masked)
        if verbose:
            r = results[str(t_eval)]
            print(f"   text repair at mask rate {t_eval}: chain {r['chain']:.3f}, one shot {r['one_shot']:.3f} over {n_masked} masked characters", flush=True)
    return dict(schedule=schedule, n_train=int(len(text)), n_test=int(len(test)), repair=results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_text", type=int, default=0)
    ap.add_argument("--strict", type=int, default=0, help="1: E24's strict sleep (no new exceptions) -- under noise it keeps every window: a nearest-neighbour denoiser")
    ap.add_argument("--out", default=str(HERE / "runs" / "e35"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print("E35 (1) frames: the written denoising chain, blockwise counting, K = 8", flush=True)
    frames = run_frames(rng, strict=bool(args.strict))
    text = None
    if not args.skip_text:
        print("E35 (2) text: masked-character diffusion, K = 4", flush=True)
        text = run_text(rng)
    r4 = frames["repair"][4]
    ok_frames = r4["chain"] >= r4["one_shot"] + 0.05
    ok_text = (text is None) or (text["repair"]["0.5"]["chain"] >= text["repair"]["0.5"]["one_shot"])
    shrink = all(s["entries_after"] < s["entries_before"] for s in frames["blocks"].values())
    verdict = "PASS" if (ok_frames and ok_text and shrink) else ("REFUTED" if not ok_frames and (text is None or not ok_text) else "INCONCLUSIVE")
    print(f"\nE35 verdict: {verdict} — frames at t = 0.5: chain {r4['chain']:.3f} vs one shot {r4['one_shot']:.3f}; tables shrink under sleep: {shrink}"
          + (f"; text at 0.5: chain {text['repair']['0.5']['chain']:.3f} vs one shot {text['repair']['0.5']['one_shot']:.3f}" if text else ""), flush=True)
    json.dump(dict(frames=frames, text=text, verdict=verdict, strict=bool(args.strict)), open(out / ("e35_strict.json" if args.strict else "e35.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
