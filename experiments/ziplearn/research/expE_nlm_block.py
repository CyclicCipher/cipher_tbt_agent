"""Experiment E -- the denoising block as NON-LOCAL MEANS instead of nearest-1: generalisation by a similarity kernel.

Follows E35 (experiments/ziplearn/e35.py). The apparatus is E35's, imported and unchanged: `collect_frames` (the frames
the explorer stood in on LockPath and CollectAll, levels 0-1 training, level 2 held out), `corrupt_chain` (coupled
corruptions on the schedule t_k = k/8), `fit_block` with `strict=True` (E24's strict sleep: every window kept, 9 cells,
the memory block of E35's `--strict 1` run), the same seed and the same draw order, so the k = 1 kernel below MUST
reproduce E35's strict numbers (chain 0.880 / 0.800, one shot 0.854 / 0.732 at t = 0.5 / 0.75) -- that is the check of
the apparatus.

The ONE change is the prediction for a window. E35's `NearestRule.predict_nearest`: a seen window predicts its stored
majority colour; an unseen window predicts the majority colour of the single Hamming-NEAREST stored window (first index
among ties). Here a window's prediction is a VOTE over the k nearest stored windows (Hamming distance d over the 9
cells; ties broken by the stored window's index, as E35's argmin does), each voting its majority colour with weight
    w = exp(-d / h)                           (`weighting = majority`)
    w = exp(-d / h) * n                       (`weighting = count`: n = the stored window's total count)
    w = exp(-d / h) * counts[colour]          (`weighting = counts`: the stored window votes its whole colour
                                               distribution, not only its majority -- an extra, not asked for)
for k in {1, 3, 7, 15} and h in {0.5, 1, 2}, plus k = all (the full kernel over EVERY stored window: non-local means
proper). Ties in the vote go to the nearest window's colour, so k = 1 is E35 exactly. Two modes: `all` applies the
vote to every window (a seen window is the d = 0 voter, weight 1, and its neighbours vote with it); `unseen` keeps
E35's rule for seen windows (the exact stored majority) and votes only where E35 fell back to the nearest window.

Measured as E35: repair accuracy (cell agreement with the clean frame) on the 135 held-out level-2 frames corrupted at
t = 0.5 (4 chain blocks) and t = 0.75 (6 chain blocks), the chain against the one-shot control (one table from the
corrupted window straight to the clean cell, the SAME kernel). E35's 3 corruption draws per frame are drawn in E35's
order (the t = 0.25 draws drawn and discarded) so the random stream is aligned; the k = 1 kernel is measured on all
3 draws (`--check_all_draws 1`, the apparatus check: it reproduces 0.880 / 0.800 and 0.854 / 0.732 exactly), and the
73-kernel grid on the FIRST draw of every frame (`--grid_draws 1`: 135 corrupted frames per t, the 5-minute budget;
the k = 1 row on that subset differs from the all-draws numbers by at most 0.004). Also reported: the distance profile
of the noisy inputs (stored windows per Hamming distance; the share of the full kernel's mass beyond the 15 nearest).

Theory under test: E35's memory block is non-local means with a HARD kernel (k = 1); a soft kernel over the stored
windows is the gradient-free way a block generalises by shared structure, and it should raise repair at high noise.

Cost: the Hamming distances from a query window to the ~30k stored windows of a block are one matmul of one-hot rows
(9 cells x 17 values); the 15 nearest and the full-kernel votes are cached per window bytes and per block, so the
chains of the different kernels share the work (their inputs coincide at the first block and mostly after).

    PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe experiments/ziplearn/research/expE_nlm_block.py
        (CPU, 211 s measured: `collect_frames` + fitting 30 s, the k=1 check 83 s, the 73-kernel grid 100 s)
        -> research/expE_nlm_block.json, .log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))
from e35 import collect_frames, corrupt_chain, fit_block            # noqa: E402
from arcgames import V as VC                                        # noqa: E402
from tasks.games import LockPath, CollectAll                        # noqa: E402

torch.set_grad_enabled(False)                                       # tensor math only; no gradient anywhere
NVAL = VC + 1                                                       # colours 0..15 and BORDER (-1): 17 values per cell
CELLS = 9                                                           # radius 1
KMAX = 15
E35_STRICT = {"0.5": dict(chain=0.880, one_shot=0.854, noisy=0.532), "0.75": dict(chain=0.800, one_shot=0.732, noisy=0.297)}


def onehot_rows(rows):
    """(Q, cells) int16 windows in -1..15 -> (Q, cells*17) float32 one-hot; the dot product of two rows = cells - Hamming
    distance. cells = 9 for the chain blocks; E35's strict sleep drops cells from a one-shot block when that costs no
    exception (the t = 0.75 one-shot table keeps 7), and the distance is then over the cells the block kept."""
    cells = rows.shape[1]
    q = torch.from_numpy(rows.astype(np.int64) + 1)
    oh = torch.zeros(len(rows), cells * NVAL, dtype=torch.float32)
    oh.scatter_(1, q + torch.arange(cells)[None, :] * NVAL, 1.0)
    return oh


class KernelIndex:
    """One block's stored windows as a matrix, and the kernel evidence for a query window: its KMAX nearest stored
    windows (Hamming distance, ties by stored index) and the full-kernel votes exp(-d/h) summed over ALL stored windows,
    for each h and each weighting. Cached per window bytes."""

    def __init__(self, rule, hs, chunk=2048):
        keys = list(rule.table.keys())                               # the same order E35's `_keys_array` uses
        self.rule, self.hs, self.chunk = rule, list(hs), chunk
        self.N = len(keys)
        self.keys = np.stack([np.frombuffer(k, dtype=np.int16) for k in keys])
        self.cells = self.keys.shape[1]
        self.kohT = onehot_rows(self.keys).T.contiguous()            # (153, N)
        maj = np.array([rule.majority[k] for k in keys], dtype=np.int64)
        counts = np.zeros((self.N, VC), np.float32)
        for i, k in enumerate(keys):
            for c, v in rule.table[k].items():
                counts[i, c] = v
        m_maj = np.zeros((self.N, VC), np.float32)
        m_maj[np.arange(self.N), maj] = 1.0
        self.M = {"majority": torch.from_numpy(m_maj), "count": torch.from_numpy(m_maj * counts.sum(1, keepdims=True)),
                  "counts": torch.from_numpy(counts)}
        self.wnames = list(self.M)
        self.Mcat = torch.cat([self.M[n] for n in self.wnames], dim=1)          # (N, 48): one matmul per h
        self.maj = torch.from_numpy(maj)
        self.tie = torch.arange(self.N, dtype=torch.float32) / (self.N + 1.0)     # ties -> the lowest stored index
        self.cache = {}                                              # window bytes -> row in the arrays below
        self.top_idx = np.zeros((0, KMAX), np.int64)
        self.top_d = np.zeros((0, KMAX), np.float32)
        self.full = np.zeros((0, len(self.hs), len(self.M), VC), np.float32)
        self.misses = 0
        self.seconds = 0.0

    def _compute(self, rows):
        """Evidence for new windows: (top_idx, top_d, full votes)."""
        outs_i, outs_d, outs_f = [], [], []
        for s in range(0, len(rows), self.chunk):
            qoh = onehot_rows(rows[s:s + self.chunk])
            D = self.cells - torch.mm(qoh, self.kohT)                # (m, N) exact integer Hamming distances
            top = torch.topk(D + self.tie[None, :], KMAX, dim=1, largest=False)
            outs_i.append(top.indices.numpy())
            outs_d.append(D.gather(1, top.indices).numpy())
            f = np.zeros((len(qoh), len(self.hs), len(self.M), VC), np.float32)
            for hi, h in enumerate(self.hs):
                f[:, hi] = torch.mm(torch.exp(-D / h), self.Mcat).numpy().reshape(len(qoh), len(self.M), VC)
            outs_f.append(f)
        return np.concatenate(outs_i), np.concatenate(outs_d), np.concatenate(outs_f)

    def rows_of(self, rows):
        """Cache rows for every query window (Q, 9)."""
        uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        keys = [u.tobytes() for u in uniq]
        miss = [i for i, k in enumerate(keys) if k not in self.cache]
        if miss:
            t0 = time.time()
            ti, td, tf = self._compute(uniq[miss])
            base = len(self.top_idx)
            self.top_idx = np.concatenate([self.top_idx, ti])
            self.top_d = np.concatenate([self.top_d, td])
            self.full = np.concatenate([self.full, tf])
            for j, i in enumerate(miss):
                self.cache[keys[i]] = base + j
            self.misses += len(miss)
            self.seconds += time.time() - t0
        ids = np.array([self.cache[k] for k in keys], dtype=np.int64)
        return ids[inv]

    def predict_rows(self, rows, k, h, weighting, mode):
        ids = self.rows_of(rows)
        nearest = self.maj[torch.from_numpy(self.top_idx[ids, 0])]                       # (Q,)
        if k == "all":
            votes = torch.from_numpy(self.full[ids, self.hs.index(h), self.wnames.index(weighting)]).clone()
        else:
            idx = torch.from_numpy(self.top_idx[ids, :k])                                # (Q, k)
            d = torch.from_numpy(self.top_d[ids, :k])
            w = torch.exp(-d / (h if h is not None else 1.0))
            votes = (w[:, :, None] * self.M[weighting][idx]).sum(1)                       # (Q, 16)
        votes[torch.arange(len(ids)), nearest] += 1e-6                                    # ties -> the nearest's colour
        pred = votes.argmax(1)
        if mode == "unseen":
            exact = torch.from_numpy(self.top_d[ids, 0] == 0)
            pred = torch.where(exact, nearest, pred)
        return pred.numpy().astype(np.int16)

    def predict_frames(self, frames, cfg):
        """Apply the block to a list of frames (any shapes) at once."""
        rows, sizes = [], []
        for f in frames:
            _full, masked = self.rule._rows(f)
            rows.append(masked)
            sizes.append(masked.shape[0])
        pred = self.predict_rows(np.concatenate(rows), cfg["k"], cfg["h"], cfg["weighting"], cfg["mode"])
        out, s = [], 0
        for f, n in zip(frames, sizes):
            out.append(pred[s:s + n].reshape(f.shape))
            s += n
        return out


def distance_profile(index, frames, hs, max_windows=3000):
    """For the unique query windows of `frames`: the mean number of stored windows at each Hamming distance, the share
    of queries with an exact match, and per h the share of the full kernel's mass that lies beyond the 15 nearest."""
    rows = np.unique(np.concatenate([index.rule._rows(f)[1] for f in frames]), axis=0)[:max_windows]
    qoh = onehot_rows(rows)
    D = index.cells - torch.mm(qoh, index.kohT)
    hist = torch.stack([(D == d).float().sum(1) for d in range(index.cells + 1)], 1).mean(0)
    top = torch.topk(D + index.tie[None, :], KMAX, dim=1, largest=False)
    dtop = D.gather(1, top.indices)
    beyond = {}
    for h in hs:
        w_all = torch.exp(-D / h).sum(1)
        w_top = torch.exp(-dtop / h).sum(1)
        beyond[str(h)] = float((1 - w_top / w_all).mean())
    return dict(n_queries=int(len(rows)), seen_share=float((D.min(1).values == 0).float().mean()),
                mean_stored_at_distance={str(d): round(float(hist[d]), 1) for d in range(index.cells + 1)},
                kernel_mass_beyond_15_nearest=beyond)


def build_configs(ks, hs, modes, weightings):
    cfgs = [dict(name="k=1 (E35 nearest)", k=1, h=None, weighting="majority", mode="all")]
    for mode in modes:
        for w in weightings:
            for k in ks:
                if k == 1:
                    continue
                for h in hs:
                    cfgs.append(dict(name=f"k={k} h={h} {w} {mode}", k=k, h=h, weighting=w, mode=mode))
            for h in hs:
                cfgs.append(dict(name=f"k=all h={h} {w} {mode}", k="all", h=h, weighting=w, mode=mode))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held", type=int, default=0, help="0 = all held-out frames (E35: 135); a smaller number takes every n-th and breaks the E35 rng alignment")
    ap.add_argument("--draws", type=int, default=3, help="corruption draws per held-out frame (E35: 3)")
    ap.add_argument("--grid_draws", type=int, default=1, help="the kernel grid is measured on the first n draws of every held-out frame (time budget)")
    ap.add_argument("--check_all_draws", type=int, default=1, help="1: the k=1 kernel is also measured on ALL draws, which must reproduce E35's strict numbers")
    ap.add_argument("--ks", default="1,3,7,15")
    ap.add_argument("--hs", default="0.5,1,2")
    ap.add_argument("--modes", default="all,unseen")
    ap.add_argument("--weightings", default="majority,count,counts")
    ap.add_argument("--out", default=str(HERE / "expE_nlm_block.json"))
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    hs = [float(x) for x in args.hs.split(",")]
    modes = args.modes.split(",")
    weightings = args.weightings.split(",")
    K = 8
    schedule = [k / K for k in range(K + 1)]
    rng = np.random.default_rng(args.seed)
    T0 = time.time()

    print("expE: the denoising block as non-local means (E35 strict apparatus, K = 8, radius 1)", flush=True)
    lp = collect_frames(LockPath, [0, 1, 2])
    ca = collect_frames(CollectAll, [0, 1, 2])
    train = lp[0] + lp[1] + ca[0] + ca[1]
    held_all = lp.get(2, []) + ca.get(2, [])
    print(f"   frames: {len(train)} training, {len(held_all)} held out  [{time.time() - T0:.0f}s]", flush=True)
    chains = [corrupt_chain(x, schedule, rng) for x in train for _ in range(4)]      # E35: draws = 4
    blocks, block_stats = {}, {}
    for k in range(1, K + 1):
        blocks[k], block_stats[k] = fit_block([(c[k], c[k - 1]) for c in chains], strict=True)
    oneshot = {}
    for k in (4, 6):
        oneshot[k], _ = fit_block([(c[k], c[0]) for c in chains], strict=True)
    print(f"   blocks fitted (strict): entries " + ", ".join(f"{k}:{block_stats[k]['entries_after']}" for k in blocks)
          + f"; one-shot 4:{len(oneshot[4].table)} 6:{len(oneshot[6].table)}  [{time.time() - T0:.0f}s]", flush=True)

    # held-out corruptions in E35's order: k = 2 drawn and discarded, k = 4 and 6 kept
    held = held_all if args.held <= 0 or args.held >= len(held_all) else [held_all[i] for i in np.linspace(0, len(held_all) - 1, args.held).astype(int)]
    aligned = (held is held_all) and args.draws == 3
    noisy, clean = {4: [], 6: []}, []
    for k in (2, 4, 6):
        for x in held:
            for _ in range(args.draws):
                xs = corrupt_chain(x, schedule, rng)
                if k in noisy:
                    noisy[k].append(xs[k])
                if k == 4:
                    clean.append(x)
    print(f"   held out: {len(held)} frames x {args.draws} draws = {len(clean)} corrupted frames per t; rng aligned with E35: {aligned}", flush=True)

    idx_blocks = {k: KernelIndex(blocks[k], hs) for k in range(1, 7)}
    idx_one = {k: KernelIndex(oneshot[k], hs) for k in (4, 6)}
    cfgs = build_configs(ks, hs, modes, weightings)
    print(f"   {len(cfgs)} kernel configurations  [{time.time() - T0:.0f}s]", flush=True)

    def evaluate(cfg, ids):
        """Chain and one-shot repair accuracy of one kernel on the corrupted frames `ids` (E35's cell-agreement mean)."""
        row = dict(cfg)
        for k in (4, 6):
            cur = [noisy[k][i] for i in ids]
            for j in range(k, 0, -1):
                cur = idx_blocks[j].predict_frames(cur, cfg)
            one = idx_one[k].predict_frames([noisy[k][i] for i in ids], cfg)
            row[f"chain_{schedule[k]}"] = float(np.mean([float((p == clean[i]).mean()) for p, i in zip(cur, ids)]))
            row[f"one_shot_{schedule[k]}"] = float(np.mean([float((p == clean[i]).mean()) for p, i in zip(one, ids)]))
        return row

    def n_misses():
        return sum(v.misses for v in list(idx_blocks.values()) + list(idx_one.values()))

    all_ids = list(range(len(clean)))
    grid_ids = all_ids if args.grid_draws >= args.draws else [i for i in all_ids if i % args.draws < args.grid_draws]
    apparatus = None
    if args.check_all_draws and grid_ids is not all_ids:
        t0 = time.time()
        apparatus = evaluate(cfgs[0], all_ids)
        apparatus["seconds"] = time.time() - t0
        print(f"   apparatus check, k=1 on all {len(all_ids)} corrupted frames per t (E35's measurement): chain {apparatus['chain_0.5']:.3f} / "
              f"{apparatus['chain_0.75']:.3f}, one shot {apparatus['one_shot_0.5']:.3f} / {apparatus['one_shot_0.75']:.3f}  "
              f"(E35 strict: 0.880 / 0.800, 0.854 / 0.732)  [{apparatus['seconds']:.0f}s]", flush=True)
    print(f"   the kernel grid is measured on {len(grid_ids)} corrupted frames per t ({len(held)} frames x the first {min(args.grid_draws, args.draws)} draw(s))", flush=True)
    results = []
    for ci, cfg in enumerate(cfgs):
        t0, m0 = time.time(), n_misses()
        row = evaluate(cfg, grid_ids)
        row["seconds"] = time.time() - t0
        row["new_windows"] = n_misses() - m0
        results.append(row)
        print(f"   [{ci + 1:2d}/{len(cfgs)}] {cfg['name']:<28s} chain {row['chain_0.5']:.3f} / {row['chain_0.75']:.3f}   "
              f"one shot {row['one_shot_0.5']:.3f} / {row['one_shot_0.75']:.3f}   ({row['seconds']:.1f}s, {row['new_windows']} new windows, total {time.time() - T0:.0f}s)", flush=True)

    profile = {}
    for k, name in ((4, "block4_t0.5"), (6, "block6_t0.75")):
        profile[name] = distance_profile(idx_blocks[k], [noisy[k][i] for i in grid_ids[:40]], hs)
        pr = profile[name]
        print(f"   distance profile of {name} inputs ({pr['n_queries']} unique windows): seen {pr['seen_share']:.2f}; stored windows at d = 0..3: "
              + ", ".join(f"{pr['mean_stored_at_distance'][str(d)]}" for d in range(4))
              + "; full-kernel mass beyond the 15 nearest: " + ", ".join(f"h={h}: {v:.2f}" for h, v in pr['kernel_mass_beyond_15_nearest'].items()), flush=True)
    noisy_acc = {str(schedule[k]): float(np.mean([float((noisy[k][i] == clean[i]).mean()) for i in grid_ids])) for k in (4, 6)}
    base = results[0]
    best = {}
    for key in ("chain_0.5", "chain_0.75", "one_shot_0.5", "one_shot_0.75"):
        b = max(results, key=lambda r: r[key])
        best[key] = dict(name=b["name"], value=b[key], delta_vs_k1=b[key] - base[key])
    cache = {f"block{k}": dict(misses=v.misses, seconds=round(v.seconds, 1), stored=v.N) for k, v in idx_blocks.items()}
    cache.update({f"one_shot{k}": dict(misses=v.misses, seconds=round(v.seconds, 1), stored=v.N) for k, v in idx_one.items()})
    out = dict(
        e35_strict_reference=E35_STRICT, apparatus_check=dict(k1_all_draws=apparatus, rng_aligned=aligned),
        n_train=len(train), n_held=len(held), draws=args.draws, grid_frames_per_t=len(grid_ids), noisy_input_accuracy=noisy_acc, blocks=block_stats,
        configs=results, best=best, distance_profile=profile, cache=cache, seconds=time.time() - T0, seed=args.seed)
    json.dump(out, open(args.out, "w"), indent=1)
    print("\n   best per column: " + "; ".join(f"{k}: {v['name']} {v['value']:.3f} ({v['delta_vs_k1']:+.3f} vs k=1)" for k, v in best.items()), flush=True)
    print(f"   cache: " + ", ".join(f"{k} {v['misses']} windows/{v['seconds']}s" for k, v in cache.items()), flush=True)
    print(f"wrote {args.out}  [{time.time() - T0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
