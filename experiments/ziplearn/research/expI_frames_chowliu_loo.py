"""Experiment I -- frames: the closed-form removal of double counting (a Chow-Liu tree over the 9 cells given the
centre target), HONEST selection of the soft Hamming kernel (k, h by leave-one-out on the TRAINING windows only), and
MORE HELD-OUT BOARDS (five instead of two), with the trivial baselines on every board.

Answers the round-1 checks on Part II of notes/gradient_free_mixing_and_features.md (research/round1_checks.txt):
  - "naive Bayes over 9 cells ... NOT over-sharpened" was refuted by the calibration gap (bits - entropy +0.33 / +0.19 /
    +0.04 at t = 0.25 / 0.5 / 0.75) and by naive Bayes over centre + 4 neighbours beating all 9 cells (0.934 / 0.881 /
    0.828 vs 0.926 / 0.874 / 0.820): the corners' evidence is double-counted. The fix named there: the Chow-Liu tree.
  - "k and h were selected on the held-out frames ... the bias is <= 0.01" was asserted, not measured; the fix: select
    by leave-one-out on the training windows and report the held-out number at THAT choice.
  - "135 held-out level-2 frames" are TWO boards (85 near-identical LockPath L2 frames + 50 CollectAll L2 frames); the
    fix: more held-out boards, every number per board, a frame-bootstrap CI, and the trivial baselines (colour prior,
    the 1-cell centre table) on every board.

Apparatus (E35's, imported unchanged): `e35.collect_frames` (the frames the explorer stood in), `e35.corrupt_chain`
(coupled corruptions on the schedule t_k = k/8, K = 8), `arcgames.LocalRule._rows` (radius-1 windows, 9 cells, 16
colours + BORDER = 17 values per cell). Training boards as before: LockPath levels 0-1 and CollectAll levels 0-1
(4 boards, 115 frames, 4 corruption chains per frame). Held-out boards: LockPath L2 (85 frames, 8x11), LockPath L3
(63 frames, 7x13 -- the explorer never solves L2 in `play`, so L3 is collected as level 0 of `LockPath(levels=
[_LEVELS[3]])`), CollectAll L2 (50, 7x11), MultiKey L1 (18, 5x13) and Toggle L0 (17, 5x9): 5 boards, 233 frames; the
last two are games the blocks were never trained on. `MultiKey.level_count = 2`, `Toggle.level_count = 1`, so
MultiKey L1 and Toggle L0 exist; every held-out board contains colours (6, 7, 8, 9, 10) that never occur as a training
TARGET, so the fraction of held-out cells whose clean colour is in the training palette is reported per board as the
ceiling every block shares (copy-input excepted).

Every block is the strict (every window kept) count table of its (noisier, cleaner) pairs, built with numpy as
r_window_neighbours.py does: the chain blocks k = 1..8 from (c[k], c[k-1]), the one-shot blocks at t = 0.25 / 0.5 /
0.75 from (c[2], c[0]), (c[4], c[0]), (c[6], c[0]) of the SAME chains (expE's construction). Predictors, all
closed-form or count-based, no gradient anywhere:
  input          copy the corrupted centre;
  colour_prior   the training targets' most frequent colour;
  centre         the 1-cell table P(y | centre colour);
  nb5 / nb9      naive Bayes P(y) prod_i P(w_i | y) over centre + 4 edge neighbours / all 9 cells (alpha = 0.5);
  nb_corners     naive Bayes over centre + the 4 corners (the complement of nb5's neighbours);
  tan9 / tan5    the Chow-Liu tree (Friedman, Geiger & Goldszmidt 1997, tree-augmented naive Bayes): the maximum-weight
                 spanning tree over the cells on the pairwise conditional mutual information I(w_i; w_j | y) estimated
                 by counting (plug-in), rooted at the centre cell, P(y | w) ~ P(y) P(w_c | y) prod_i P(w_i | w_pa(i), y);
                 the pairwise conditionals are m-estimates toward P(w_i | y) with m = 5 (Friedman's prior weight), the
                 1-cell conditionals alpha = 0.5; tan9_m1 / m20 / m100 = the same tree at other prior weights (m -> inf
                 is nb9), a sensitivity of the tree's calibration to the one smoothing constant it has;
  hard           E35's nearest stored window (Hamming, ties by stored index), its counts as the distribution;
  soft_*         the Hamming kernel P(y | w) ~ sum_{s in kNN(w)} exp(-d(w, s) / h) n_s(y) (count-weighted, the full
                 colour distribution of each neighbour = expE's `counts` weighting), k in {1, 3, 7, 15, 50, 100, 200,
                 all}, h in {0.25, 0.5, 0.75, 1, 1.5, 2, 3}, (k, h) selected PER BLOCK by leave-one-out on that block's own stored
                 windows: `loo_window` leaves the whole stored window out (the query is a window never seen -- the
                 held-out regime at t >= 0.5), `loo_pair` leaves one OBSERVATION out (the window stays, with its own
                 count for the held-out colour reduced by one -- the exact leave-one-out of the kernel estimator,
                 which also prices a seen window); both selected by the leave-one-out LOG-LIKELIHOOD (the proper score),
                 the accuracy-selected choice is recorded too;
  soft_expE      the kernel expE selected ON THE HELD-OUT frames (one shot k = all, h = 0.5; chain k = 15, h = 2), the
                 same configuration on the new boards -- the selection bias made measurable.
The chain applies blocks k..1 with the same predictor at every block (per-block (k, h) for soft_*). Measured per t and
per board: accuracy (cell agreement with the clean frame), online code length (bits per cell, -log2 P(true colour))
and the entropy of the predictive distribution; the calibration gap = bits - entropy (positive = over-confident);
mean max-probability against accuracy; the same over the in-palette cells only; the pairwise CMI matrix and the tree;
paired frame-bootstrap CIs of the differences the note reads as real; the board-level mean +- SE (n = 5 boards) and
the number of boards on which each difference keeps its sign. Chains report accuracy only (their intermediate argmax
steps make the last block's distribution not a probability of the clean cell given the noisy input).

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/expI_frames_chowliu_loo.py
        (CPU, 139 s measured: collect_frames 17 s, trees 2 s, the 50-point LOO grid on 9 blocks 19 s, evaluation 100 s)
        -> research/expI_frames_chowliu_loo.json, .log
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
from e35 import collect_frames, corrupt_chain                     # noqa: E402
from arcgames import LocalRule, V                                  # noqa: E402
from tasks.games import LockPath, CollectAll, MultiKey, Toggle     # noqa: E402
from tasks.games.lockpath import _LEVELS as LP_LEVELS              # noqa: E402

torch.set_grad_enabled(False)
NCOL = V + 1                                                       # 16 colours + BORDER (index 0)
CELLS = 9
CENTRE = 4
EDGES = [1, 3, 4, 5, 7]                                            # centre + 4 edge neighbours (raster order)
CORNERS = [0, 2, 4, 6, 8]                                          # centre + 4 corners
KS = [1, 3, 7, 15, 50, 100, 200, "all"]
HS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
TAN_MS = [1.0, 5.0, 20.0, 100.0]                                    # the m-estimate's prior weight (5 = the default)
EPS = 1e-3                                                         # r_window's numerator floor for the kernel
LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def rows(frame):
    return LocalRule(1)._rows(frame)[1] + 1                        # (H*W, 9) in 0..16


def onehot(rows_):
    q = torch.from_numpy(rows_.astype(np.int64))
    oh = torch.zeros(len(rows_), CELLS * NCOL, dtype=torch.float32)
    oh.scatter_(1, q + torch.arange(CELLS)[None, :] * NCOL, 1.0)
    return oh


# ── one block: the strict count table + its closed-form factorisations ──────────────────────────────────────────
class Block:
    def __init__(self, W, Y, alpha=0.5, m=5.0):
        self.W, self.Y = W, Y
        self.n_pairs = len(W)
        keys, inv = np.unique(W, axis=0, return_inverse=True)
        self.keys = keys
        self.N = len(keys)
        counts = np.zeros((self.N, V), dtype=np.float32)
        np.add.at(counts, (inv.reshape(-1), Y), 1)
        self.counts = torch.from_numpy(counts)
        self.n_s = self.counts.sum(1)
        self.kohT = onehot(keys).T.contiguous()                    # (153, N)
        self.tie = torch.arange(self.N, dtype=torch.float32) / (self.N + 1.0)
        # first-order tables
        self.Ny = np.bincount(Y, minlength=V).astype(np.float64)
        self.py = (self.Ny + alpha) / (self.Ny.sum() + V * alpha)
        self.J1 = np.zeros((CELLS, NCOL, V), dtype=np.float64)     # N(x_i, y)
        for i in range(CELLS):
            np.add.at(self.J1[i], (W[:, i], Y), 1)
        self.p_x_given_y = (self.J1 + alpha) / (self.J1 + alpha).sum(1, keepdims=True)       # sums over x_i
        self.log_p1 = np.log(self.p_x_given_y)
        # pairwise tables N(x_i, x_j, y) and the plug-in conditional mutual information I(x_i; x_j | y)
        self.J2 = {}
        self.cmi = np.zeros((CELLS, CELLS))
        for i in range(CELLS):
            for j in range(i + 1, CELLS):
                idx = (W[:, i].astype(np.int64) * NCOL + W[:, j]) * V + Y
                J = np.bincount(idx, minlength=NCOL * NCOL * V).reshape(NCOL, NCOL, V).astype(np.float64)
                self.J2[(i, j)] = J
                self.cmi[i, j] = self.cmi[j, i] = self._cmi(J)
        self.mi_y = np.array([self._mi(self.J1[i]) for i in range(CELLS)])   # I(x_i; y)
        self.alpha, self.m = alpha, m
        self.trees = {}
        self.kcfgs, self.kcache, self.kpred, self.kmisses = [], {}, {}, 0

    @staticmethod
    def _cmi(J):
        """I(X; Z | Y) in bits from counts J[x, z, y] (plug-in)."""
        n = J.sum()
        pxzy = J / n
        py = J.sum((0, 1)) / n
        pxy = J.sum(1) / n                                          # (x, y)
        pzy = J.sum(0) / n                                          # (z, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log2(pxzy * py[None, None, :] / (pxy[:, None, :] * pzy[None, :, :]))
        r[~np.isfinite(r)] = 0.0
        return float((pxzy * r).sum())

    @staticmethod
    def _mi(J):
        n = J.sum()
        p = J / n
        px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log2(p / (px * py))
        r[~np.isfinite(r)] = 0.0
        return float((p * r).sum())

    def tree(self, cells, m=None):
        """Chow-Liu maximum-weight spanning tree over `cells` on the CMI, rooted at the centre; parents dict + edges;
        the conditionals' m-estimate weight `m` (default self.m)."""
        cells = tuple(cells)
        m = self.m if m is None else m
        if (cells, m) in self.trees:
            return self.trees[(cells, m)]
        edges = sorted([(self.cmi[i, j], i, j) for a, i in enumerate(cells) for j in cells[a + 1:]], reverse=True)
        parent = {c: c for c in cells}

        def find(c):
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            return c
        adj = {c: [] for c in cells}
        kept = []
        for w, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                adj[i].append(j); adj[j].append(i)
                kept.append((i, j, w))
        pa = {CENTRE: None}
        stack = [CENTRE]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in pa:
                    pa[v] = u
                    stack.append(v)
        # conditional log-tables P(x_i | x_pa, y): m-estimate toward P(x_i | y)
        logc = {}
        for i, p in pa.items():
            if p is None:
                continue
            J = self.J2[(i, p)] if (i, p) in self.J2 else self.J2[(p, i)].transpose(1, 0, 2)   # (x_i, x_pa, y)
            prior = self.p_x_given_y[i]                                                        # (x_i, y)
            num = J + m * prior[:, None, :]
            den = J.sum(0, keepdims=True) + m
            logc[i] = np.log(num / den)
        self.trees[(cells, m)] = dict(parents=pa, edges=kept, logc=logc)
        return self.trees[(cells, m)]

    # ── predictive distributions (Q, 16) for query windows (Q, 9) ──
    def nb(self, Wq, cells):
        logp = np.log(self.py)[None, :] + sum(self.log_p1[i][Wq[:, i]] for i in cells)
        return _softmax(logp)

    def tan(self, Wq, cells, m=None):
        tr = self.tree(cells, m)
        logp = np.log(self.py)[None, :] + self.log_p1[CENTRE][Wq[:, CENTRE]]
        for i, p in tr["parents"].items():
            if p is not None:
                logp = logp + tr["logc"][i][Wq[:, i], Wq[:, p]]
        return _softmax(logp)

    def kernel(self, Wq, cfg, chunk=1024):
        """Nadaraya-Watson over the k nearest stored windows (Hamming, ties by stored index) or all of them, for
        cfg = (k, h). The Hamming distances of a unique query window are computed ONCE per block and every
        configuration in `self.kcfgs` (fixed after the leave-one-out selection) is served from that computation and
        cached (expE's `KernelIndex`), so the chains of the different kernels share the work."""
        cfgs = self.kcfgs
        assert cfg in cfgs
        uniq, inv = np.unique(Wq, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        keys = [u.tobytes() for u in uniq]
        miss = [i for i, kb in enumerate(keys) if kb not in self.kcache]
        if miss:
            rows_ = uniq[miss]
            kfin = [c[0] for c in cfgs if c[0] != "all"]
            kmax = max(kfin) if kfin else 0
            h_all = sorted(set(c[1] for c in cfgs if c[0] == "all"))
            new = {c: [] for c in cfgs}
            for s in range(0, len(rows_), chunk):
                qoh = onehot(rows_[s:s + chunk])
                D = CELLS - torch.mm(qoh, self.kohT)                # (m, N) exact integer Hamming distances
                if kmax:
                    top = torch.topk(D + self.tie[None, :], kmax, dim=1, largest=False)
                    dtop = D.gather(1, top.indices)
                    Ctop = self.counts[top.indices]                 # (m, kmax, 16)
                full = {h: torch.mm(torch.exp(-D / h), self.counts) for h in h_all}
                for c in cfgs:
                    k, h = c
                    if k == "all":
                        votes = full[h]
                    elif k == 1:
                        votes = Ctop[:, 0]
                    else:
                        votes = (torch.exp(-dtop[:, :k] / h)[:, :, None] * Ctop[:, :k]).sum(1)
                    votes = votes + EPS
                    new[c].append((votes / votes.sum(1, keepdim=True)).numpy())
            base = len(self.kcache)
            for j, i in enumerate(miss):
                self.kcache[keys[i]] = base + j
            for c in cfgs:
                arr = np.concatenate(new[c])
                self.kpred[c] = arr if c not in self.kpred else np.concatenate([self.kpred[c], arr])
            self.kmisses += len(miss)
        ids = np.array([self.kcache[kb] for kb in keys], dtype=np.int64)[inv]
        return self.kpred[cfg][ids]

    # ── leave-one-out selection of (k, h) on the stored windows ──
    def loo_select(self, rng, n_sub=3000, chunk=1024):
        sub = np.sort(rng.choice(self.N, size=min(n_sub, self.N), replace=False))
        kmax = max(k for k in KS if k != "all")
        C_sub = self.counts[sub]                                    # (Q, 16)
        n_sub_ = C_sub.sum(1, keepdim=True)
        grid = [(1, None)] + [(k, h) for k in KS if k != 1 for h in HS]
        ll_w = {g: 0.0 for g in grid}; ll_p = {g: 0.0 for g in grid}
        ok_w = {g: 0.0 for g in grid}; ok_p = {g: 0.0 for g in grid}
        for s in range(0, len(sub), chunk):
            ids = sub[s:s + chunk]
            qoh = onehot(self.keys[ids])
            D = CELLS - torch.mm(qoh, self.kohT)
            D[torch.arange(len(ids)), torch.from_numpy(ids)] = 1e6         # leave the window itself out
            top = torch.topk(D + self.tie[None, :], kmax, dim=1, largest=False)
            dtop = D.gather(1, top.indices)                                  # (m, kmax)
            Ctop = self.counts[top.indices]                                  # (m, kmax, 16)
            Cs = C_sub[s:s + chunk]; ns = n_sub_[s:s + chunk]
            full = {h: torch.mm(torch.exp(-D / h), self.counts) for h in HS}
            for g in grid:
                k, h = g
                if k == 1:
                    others = Ctop[:, :1].sum(1)                              # nearest other window
                    others_p = torch.zeros_like(others)                      # pair-out: the window itself (k-1 = 0 others)
                elif k == "all":
                    others = full[h]; others_p = full[h]
                else:
                    w = torch.exp(-dtop[:, :k] / h)
                    others = (w[:, :, None] * Ctop[:, :k]).sum(1)
                    wp = torch.exp(-dtop[:, :k - 1] / h)
                    others_p = (wp[:, :, None] * Ctop[:, :k - 1]).sum(1)
                # leave-window-out: predict the window's own counts from the others
                p = (others + EPS) / (others + EPS).sum(1, keepdim=True)
                ll_w[g] += float((Cs * torch.log2(p)).sum())
                ok_w[g] += float(Cs.gather(1, p.argmax(1, keepdim=True)).sum())
                # leave-pair-out: the window stays with its held-out observation removed
                num = others_p + Cs - 1.0 + EPS                              # numerator for colour y when y is held out
                den = others_p.sum(1, keepdim=True) + ns - 1.0 + V * EPS
                pp = num / den
                mask = Cs > 0
                ll_p[g] += float((Cs * torch.log2(pp.clamp_min(1e-12)) * mask).sum())
                # accuracy of the pair-out argmax: colour y is predicted iff its reduced numerator is the largest
                base = others_p + Cs                                          # full numerators
                for y in range(V):
                    red = base.clone(); red[:, y] -= 1.0
                    ok_p[g] += float((Cs[:, y] * (red.argmax(1) == y).float()).sum())
        tot = float(n_sub_.sum())
        res = {}
        for g in grid:
            res[_gname(g)] = dict(k=g[0], h=g[1], loo_window_bits=-ll_w[g] / tot, loo_window_acc=ok_w[g] / tot,
                                  loo_pair_bits=-ll_p[g] / tot, loo_pair_acc=ok_p[g] / tot)
        pick = dict(loo_window=_gname(min(grid, key=lambda g: -ll_w[g])), loo_pair=_gname(min(grid, key=lambda g: -ll_p[g])),
                    loo_window_by_acc=_gname(max(grid, key=lambda g: ok_w[g])), loo_pair_by_acc=_gname(max(grid, key=lambda g: ok_p[g])))
        return dict(n_sub=int(len(sub)), n_obs=tot, grid=res, pick=pick)


def _gname(g):
    return f"k={g[0]}" + ("" if g[1] is None else f" h={g[1]}")


def _gparse(name):
    parts = name.split()
    k = parts[0][2:]
    k = "all" if k == "all" else int(k)
    h = float(parts[1][2:]) if len(parts) > 1 else 1.0
    return (k, 1.0) if k == 1 else (k, h)


def _softmax(logp):
    logp = logp - logp.max(1, keepdims=True)
    p = np.exp(logp)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


# ── statistics ──────────────────────────────────────────────────────────────────────────────────────────────────
def boot_diff(a, b, w, rng, B=2000):
    """Paired bootstrap over frames of the cell-weighted mean of (a - b); a, b = per-frame correct counts, w = cells."""
    a = np.asarray(a, float); b = np.asarray(b, float); w = np.asarray(w, float)
    n = len(a)
    idx = rng.integers(0, n, (B, n))
    diffs = (a[idx].sum(1) - b[idx].sum(1)) / w[idx].sum(1)
    return dict(point=float((a.sum() - b.sum()) / w.sum()), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=4, help="corruption chains per training frame (E35: 4)")
    ap.add_argument("--held_draws", type=int, default=3, help="corruption chains per held-out frame (E35: 3)")
    ap.add_argument("--loo_sub", type=int, default=3000, help="stored windows per block in the leave-one-out selection")
    ap.add_argument("--out", default=str(HERE / "expI_frames_chowliu_loo.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    brng = np.random.default_rng(args.seed + 1)
    T0 = time.time()
    K = 8
    schedule = [k / K for k in range(K + 1)]
    say("expI: Chow-Liu tree vs naive Bayes, LOO-selected Hamming kernel, five held-out boards (E35 apparatus, K = 8, radius 1)")

    # ── boards ──
    lp = collect_frames(LockPath, [0, 1, 2])
    lp3 = collect_frames(lambda: LockPath(levels=[LP_LEVELS[3]]), [0])
    ca = collect_frames(CollectAll, [0, 1, 2])
    mk = collect_frames(MultiKey, [0, 1])
    tg = collect_frames(Toggle, [0])
    train_boards = {"lockpath_l0": lp[0], "lockpath_l1": lp[1], "collectall_l0": ca[0], "collectall_l1": ca[1]}
    held_boards = {"lockpath_l2": lp[2], "lockpath_l3": lp3[0], "collectall_l2": ca[2], "multikey_l1": mk[1], "toggle_l0": tg[0]}
    train = sum(train_boards.values(), [])
    train_palette = set(int(c) for x in train for c in np.unique(x))
    board_info = {}
    for name, fr in list(train_boards.items()) + list(held_boards.items()):
        F = np.stack(fr)
        cols = sorted(int(c) for c in np.unique(F))
        board_info[name] = dict(role="train" if name in train_boards else "held", n_frames=len(fr), shape=list(F.shape[1:]),
                                colours=cols, colours_not_in_training_targets=[c for c in cols if c not in train_palette],
                                frac_cells_in_training_palette=float(np.isin(F, list(train_palette)).mean()),
                                frac_cells_varying=float((F != F[0]).any(0).mean()))
    say(f"   training: {len(train)} frames on {len(train_boards)} boards (LockPath L0-1, CollectAll L0-1), target palette {sorted(train_palette)}  [{time.time() - T0:.0f}s]")
    for name, b in board_info.items():
        say(f"   {b['role']:5s} {name:14s} n={b['n_frames']:3d} shape={b['shape']} colours={b['colours']} not-in-training-targets={b['colours_not_in_training_targets']} "
            f"in-palette cells {b['frac_cells_in_training_palette']:.3f} varying {b['frac_cells_varying']:.3f}")

    # ── blocks from the training chains ──
    chains = [corrupt_chain(x, schedule, rng) for x in train for _ in range(args.draws)]

    def make_block(k_from, k_to):
        Wl = [rows(c[k_from]) for c in chains]
        Yl = [c[k_to].reshape(-1) for c in chains]
        return Block(np.concatenate(Wl).astype(np.int16), np.concatenate(Yl).astype(np.int16))
    blocks = {k: make_block(k, k - 1) for k in range(1, 7)}       # the chain uses blocks 1..6 (t <= 0.75)
    oneshot = {k: make_block(k, 0) for k in (2, 4, 6)}
    say(f"   blocks: chain " + ", ".join(f"{k}:{b.N}" for k, b in blocks.items()) + "; one shot " + ", ".join(f"{k}:{b.N}" for k, b in oneshot.items())
        + f" stored windows (from {blocks[1].n_pairs} pairs each)  [{time.time() - T0:.0f}s]")

    # ── the Chow-Liu matrices and trees ──
    trees = {}
    for label, b in [(f"one_shot_{schedule[k]}", oneshot[k]) for k in (2, 4, 6)] + [(f"block{k}", blocks[k]) for k in blocks]:
        tr9 = b.tree(range(CELLS)); tr5 = b.tree(EDGES)
        trees[label] = dict(mi_cell_target_bits=[round(float(v), 4) for v in b.mi_y],
                            cmi_matrix_bits=[[round(float(v), 4) for v in row] for row in b.cmi],
                            tree9_edges=[(int(i), int(j), round(float(w), 4)) for i, j, w in tr9["edges"]],
                            tree9_parents={str(i): (None if p is None else int(p)) for i, p in tr9["parents"].items()},
                            tree5_edges=[(int(i), int(j), round(float(w), 4)) for i, j, w in tr5["edges"]])
        if label.startswith("one_shot"):
            corner_edge = np.mean([b.cmi[c, e] for c in (0, 2, 6, 8) for e in (1, 3, 5, 7) if abs(c - e) in (1, 3)])
            edge_edge = np.mean([b.cmi[1, 3], b.cmi[1, 5], b.cmi[3, 7], b.cmi[5, 7]])
            corner_centre = np.mean([b.cmi[c, CENTRE] for c in (0, 2, 6, 8)])
            edge_centre = np.mean([b.cmi[e, CENTRE] for e in (1, 3, 5, 7)])
            trees[label]["cmi_summary_bits"] = dict(corner_adjacent_edge=float(corner_edge), edge_edge_adjacent=float(edge_edge),
                                                     corner_centre=float(corner_centre), edge_centre=float(edge_centre),
                                                     mean_all_pairs=float(b.cmi[np.triu_indices(CELLS, 1)].mean()))
            say(f"   {label}: CMI matrix (bits, cells in raster order 0..8, centre = 4):" + "".join(chr(10) + "      " + " ".join(f"{v:5.2f}" for v in row) for row in b.cmi))
            say(f"   {label}: I(cell; y) = {[round(float(v), 2) for v in b.mi_y]} bits; CMI given y: corner-adjacent-edge {corner_edge:.3f}, "
                f"edge-edge {edge_edge:.3f}, corner-centre {corner_centre:.3f}, edge-centre {edge_centre:.3f}, mean of 36 pairs {b.cmi[np.triu_indices(CELLS, 1)].mean():.3f}; "
                f"tree9 edges {[(int(i), int(j), round(float(w), 2)) for i, j, w in tr9['edges']]}")
    say(f"   trees built  [{time.time() - T0:.0f}s]")

    # ── leave-one-out selection of (k, h) per block ──
    loo = {}
    for label, b in [(f"one_shot_{schedule[k]}", oneshot[k]) for k in (2, 4, 6)] + [(f"block{k}", blocks[k]) for k in blocks]:
        loo[label] = b.loo_select(rng, n_sub=args.loo_sub)
        pk = loo[label]["pick"]; g = loo[label]["grid"]
        say(f"   LOO {label:14s} ({loo[label]['n_sub']} windows, {loo[label]['n_obs']:.0f} obs): window-out picks {pk['loo_window']:14s} "
            f"({g[pk['loo_window']]['loo_window_bits']:.3f} bits, acc {g[pk['loo_window']]['loo_window_acc']:.3f}; k=1 {g['k=1']['loo_window_bits']:.3f} / {g['k=1']['loo_window_acc']:.3f}); "
            f"pair-out picks {pk['loo_pair']:14s} ({g[pk['loo_pair']]['loo_pair_bits']:.3f} bits, acc {g[pk['loo_pair']]['loo_pair_acc']:.3f}; k=1 {g['k=1']['loo_pair_bits']:.3f} / {g['k=1']['loo_pair_acc']:.3f}); "
            f"by-acc picks {pk['loo_window_by_acc']} / {pk['loo_pair_by_acc']}  [{time.time() - T0:.0f}s]")

    # the kernel configurations each block will be asked for (fixed now; the block computes them together)
    for k_t in (2, 4, 6):
        b, lab = oneshot[k_t], f"one_shot_{schedule[k_t]}"
        b.kcfgs = sorted({(1, 1.0), _gparse(loo[lab]["pick"]["loo_window"]), _gparse(loo[lab]["pick"]["loo_pair"]), ("all", 0.5)}, key=str)
    for j, b in blocks.items():
        b.kcfgs = sorted({(1, 1.0), _gparse(loo[f"block{j}"]["pick"]["loo_window"]), _gparse(loo[f"block{j}"]["pick"]["loo_pair"]), (15, 2.0)}, key=str)

    # ── held-out corruptions ──
    held = []                                                      # (board, frame index, clean, chain)
    for bname, fr in held_boards.items():
        for fi, x in enumerate(fr):
            for _ in range(args.held_draws):
                held.append((bname, fi, x, corrupt_chain(x, schedule, rng)))
    say(f"   held out: {sum(len(v) for v in held_boards.values())} frames x {args.held_draws} draws = {len(held)} corrupted frames per t  [{time.time() - T0:.0f}s]")
    frame_id = np.concatenate([np.full(x.size, i) for i, (_, _, x, _) in enumerate(held)])
    frame_board = np.array([bn for bn, _, _, _ in held])
    frame_key = np.array([f"{bn}:{fi}" for bn, fi, _, _ in held])     # bootstrap unit = clean frame (its draws together)
    ukeys, frame_unit = np.unique(frame_key, return_inverse=True)
    unit_board = np.array([k.split(":")[0] for k in ukeys])
    Yh = np.concatenate([x.reshape(-1) for _, _, x, _ in held]).astype(np.int16)
    in_pal = np.isin(Yh, list(train_palette))
    cells_per_frame = np.bincount(frame_id, minlength=len(held)).astype(float)
    colour_prior = int(blocks[1].Ny.argmax())

    def predict_frames(block, frames_noisy, method, cfg=None):
        """Predictive (Q, 16) for the stacked windows of `frames_noisy` under `method`."""
        Wq = np.concatenate([rows(f) for f in frames_noisy]).astype(np.int16)
        if method == "nb9":
            return block.nb(Wq, range(CELLS))
        if method == "nb5":
            return block.nb(Wq, EDGES)
        if method == "nb_corners":
            return block.nb(Wq, CORNERS)
        if method == "centre":
            return block.nb(Wq, [CENTRE])
        if method == "tan9":
            return block.tan(Wq, range(CELLS))
        if method == "tan5":
            return block.tan(Wq, EDGES)
        if method.startswith("tan9_m"):
            return block.tan(Wq, range(CELLS), float(method[6:]))
        if method == "hard":
            return block.kernel(Wq, (1, 1.0))
        if method == "soft":
            return block.kernel(Wq, cfg)
        raise ValueError(method)

    def split(P, frames_noisy):
        out, s = [], 0
        for f in frames_noisy:
            out.append(P[s:s + f.size].argmax(1).astype(np.int16).reshape(f.shape))
            s += f.size
        return out

    def summarise(ok, P=None):
        """Per-board and pooled accuracy (+ bits, entropy, gap when P is given), per-frame correct counts."""
        res = dict(acc=float(ok.mean()), per_board={})
        pf = np.bincount(frame_id, weights=ok.astype(float), minlength=len(held))
        if P is not None:
            bits = -np.log2(np.maximum(P[np.arange(len(Yh)), Yh], 1e-30))
            Hq = -(P * np.log2(np.maximum(P, 1e-30))).sum(1)
            pmax = P.max(1)
            res.update(bits=float(bits.mean()), entropy=float(Hq.mean()), gap=float(bits.mean() - Hq.mean()),
                       mean_max_prob=float(pmax.mean()), acc_in_palette=float(ok[in_pal].mean()),
                       bits_in_palette=float(bits[in_pal].mean()), entropy_in_palette=float(Hq[in_pal].mean()),
                       bits_out_of_palette=float(bits[~in_pal].mean()), frac_out_of_palette=float((~in_pal).mean()),
                       gap_in_palette=float(bits[in_pal].mean() - Hq[in_pal].mean()), mean_max_prob_in_palette=float(pmax[in_pal].mean()))
        for bn in held_boards:
            m = frame_board[frame_id] == bn
            u = frame_unit[frame_id][m]
            per_unit = np.bincount(u - u.min(), weights=ok[m].astype(float)) / np.bincount(u - u.min())
            d = dict(acc=float(ok[m].mean()), se_frames=float(per_unit.std(ddof=1) / np.sqrt(len(per_unit))) if len(per_unit) > 1 else 0.0, cells=int(m.sum()))
            if P is not None:
                d.update(bits=float(bits[m].mean()), entropy=float(Hq[m].mean()), gap=float(bits[m].mean() - Hq[m].mean()),
                         acc_in_palette=float(ok[m & in_pal].mean()), gap_in_palette=float(bits[m & in_pal].mean() - Hq[m & in_pal].mean()),
                         mean_max_prob=float(pmax[m].mean()))
            res["per_board"][bn] = d
        res["board_mean_acc"] = float(np.mean([v["acc"] for v in res["per_board"].values()]))
        return res, pf

    results = {"per_t": {}}
    for k_t in (2, 4, 6):
        t = schedule[k_t]
        noisy = [c[k_t] for _, _, _, c in held]
        Wq_noisy = np.concatenate([rows(f) for f in noisy]).astype(np.int16)
        ob = oneshot[k_t]
        preds, dists = {}, {}
        # trivial baselines
        preds["input"] = (Wq_noisy[:, CENTRE] - 1).astype(np.int16)
        preds["colour_prior"] = np.full(len(Yh), colour_prior, dtype=np.int16)
        # one-shot closed forms and kernels
        soft_cfgs = {"soft_loo_window": _gparse(loo[f"one_shot_{t}"]["pick"]["loo_window"]),
                     "soft_loo_pair": _gparse(loo[f"one_shot_{t}"]["pick"]["loo_pair"]),
                     "soft_expE_heldout_selected": ("all", 0.5)}
        for mname in ("centre", "nb5", "nb9", "nb_corners", "tan5", "tan9", "hard") + tuple(f"tan9_m{m:g}" for m in TAN_MS if m != 5.0):
            dists[mname] = predict_frames(ob, noisy, mname)
        for mname, cfg in soft_cfgs.items():
            dists[mname] = predict_frames(ob, noisy, "soft", cfg)
        for mname, P in dists.items():
            preds[mname] = P.argmax(1).astype(np.int16)
        say(f"   t={t}: one-shot predictors done  [{time.time() - T0:.0f}s]")
        # chains: blocks k_t..1 with the same predictor at every block
        chain_methods = {"chain_hard": ("hard", None), "chain_soft_loo_window": ("soft", "loo_window"), "chain_soft_loo_pair": ("soft", "loo_pair"),
                         "chain_soft_expE_heldout_selected": ("soft", (15, 2.0)), "chain_nb9": ("nb9", None), "chain_nb5": ("nb5", None), "chain_tan9": ("tan9", None)}
        for cname, (mname, cfg) in chain_methods.items():
            cur = noisy
            for j in range(k_t, 0, -1):
                c = cfg
                if mname == "soft" and isinstance(cfg, str):
                    c = _gparse(loo[f"block{j}"]["pick"][cfg])
                cur = split(predict_frames(blocks[j], cur, mname, c), cur)
            preds[cname] = np.concatenate([f.reshape(-1) for f in cur]).astype(np.int16)
            say(f"   t={t}: {cname} done  [{time.time() - T0:.0f}s]")
        # summaries
        per_t = dict(t=t, held_cells=int(len(Yh)), frac_cells_in_palette=float(in_pal.mean()), methods={}, boot={}, board_level={},
                     soft_cfgs={k: dict(k=str(v[0]), h=v[1]) for k, v in soft_cfgs.items()},
                     chain_soft_cfgs={cfg: {f"block{j}": loo[f"block{j}"]["pick"][cfg] for j in range(1, k_t + 1)} for cfg in ("loo_window", "loo_pair")})
        pfs = {}
        for mname, pr in preds.items():
            ok = pr == Yh
            per_t["methods"][mname], pfs[mname] = summarise(ok, dists.get(mname))
        pairs = [("soft_loo_window", "hard"), ("soft_loo_pair", "hard"), ("soft_expE_heldout_selected", "soft_loo_window"), ("tan9", "nb9"), ("tan9", "nb5"),
                 ("nb5", "nb9"), ("tan5", "nb5"), ("soft_loo_window", "tan9"), ("nb9", "hard"), ("tan9", "hard"), ("nb9", "centre"),
                 ("chain_soft_loo_window", "soft_loo_window"), ("chain_hard", "hard"), ("chain_tan9", "tan9"), ("chain_nb5", "nb5"),
                 ("chain_soft_loo_window", "chain_hard"), ("chain_soft_expE_heldout_selected", "chain_soft_loo_window"), ("chain_tan9", "chain_nb9"),
                 ("tan9_m1", "nb9"), ("tan9_m20", "nb9"), ("tan9_m100", "nb9")]
        # bootstrap over clean frames (draws of a frame resampled together)
        n_units = len(ukeys)
        unit_cells = np.bincount(frame_unit, weights=cells_per_frame, minlength=n_units)
        for a, b in pairs:
            ua = np.bincount(frame_unit, weights=pfs[a], minlength=n_units); ub = np.bincount(frame_unit, weights=pfs[b], minlength=n_units)
            d = dict(pooled=boot_diff(ua, ub, unit_cells, brng), per_board={})
            for bn in held_boards:
                m = unit_board == bn
                d["per_board"][bn] = boot_diff(ua[m], ub[m], unit_cells[m], brng)
            diffs = np.array([per_t["methods"][a]["per_board"][bn]["acc"] - per_t["methods"][b]["per_board"][bn]["acc"] for bn in held_boards])
            d["board_level"] = dict(mean=float(diffs.mean()), se=float(diffs.std(ddof=1) / np.sqrt(len(diffs))), n_boards=int(len(diffs)),
                                    boards_positive=int((diffs > 0).sum()), boards_negative=int((diffs < 0).sum()))
            per_t["boot"][f"{a} - {b}"] = d
        # apparatus check: LockPath L2 + CollectAll L2 pooled by cells, against check_frames_apparatus.json (1 draw, its own rng)
        ref_path = HERE / "check_frames_apparatus.json"
        ref = json.load(open(ref_path))["per_t"][str(t)]["acc"] if ref_path.exists() else {}
        m2 = np.isin(frame_board[frame_id], ["lockpath_l2", "collectall_l2"])
        per_t["apparatus_check_lp2_ca2"] = {}
        for mname, rname in (("hard", "hard"), ("nb9", "nb9"), ("nb5", "nb_centre_plus4"), ("centre", "centre_only"), ("input", "input"), ("soft_expE_heldout_selected", "soft")):
            per_t["apparatus_check_lp2_ca2"][mname] = dict(here=float((preds[mname] == Yh)[m2].mean()), check_frames_apparatus=ref.get(rname))
        results["per_t"][str(t)] = per_t
        # print
        order = ["input", "colour_prior", "centre", "nb_corners", "nb9", "nb5", "tan5", "tan9"] + [f"tan9_m{m:g}" for m in TAN_MS if m != 5.0] + ["hard", "soft_loo_window", "soft_loo_pair", "soft_expE_heldout_selected",
                 "chain_hard", "chain_nb9", "chain_nb5", "chain_tan9", "chain_soft_loo_window", "chain_soft_loo_pair", "chain_soft_expE_heldout_selected"]
        say(f"\n   t = {t}: {len(Yh)} held-out cells ({in_pal.mean():.3f} in the training palette); kernels: " + ", ".join(f"{k} = {_gname(v)}" for k, v in soft_cfgs.items())
            + "; chain kernels " + "; ".join(f"{cfg}: " + ", ".join(v.values()) for cfg, v in per_t["chain_soft_cfgs"].items()))
        hdr = f"   {'predictor':32s} {'pooled':>7s} {'b-mean':>7s} " + " ".join(f"{bn[:12]:>12s}" for bn in held_boards) + "   bits  entr   gap  pmax | in-palette: acc  gap"
        say(hdr)
        for mname in order:
            r = per_t["methods"][mname]
            line = f"   {mname:32s} {r['acc']:7.3f} {r['board_mean_acc']:7.3f} " + " ".join(f"{r['per_board'][bn]['acc']:12.3f}" for bn in held_boards)
            if "bits" in r:
                line += f"   {r['bits']:5.3f} {r['entropy']:5.3f} {r['gap']:+5.3f} {r['mean_max_prob']:5.3f} | {r['acc_in_palette']:.3f} {r['gap_in_palette']:+.3f}"
            say(line)
        say("   apparatus check, LockPath L2 + CollectAll L2 pooled (check_frames_apparatus.json in brackets): " + ", ".join(
            f"{k} {v['here']:.3f} [{v['check_frames_apparatus']:.3f}]" if v['check_frames_apparatus'] is not None else f"{k} {v['here']:.3f}" for k, v in per_t["apparatus_check_lp2_ca2"].items()))
        say("   out-of-palette cells (" + f"{(~in_pal).mean():.3f} of cells, clean colour never a training target): bits per such cell " + ", ".join(
            f"{m} {per_t['methods'][m]['bits_out_of_palette']:.1f}" for m in ("centre", "nb9", "tan9", "hard", "soft_loo_window")))
        for name, d in per_t["boot"].items():
            bl = d["board_level"]
            say(f"      {name:60s} pooled {d['pooled']['point']:+.3f} [{d['pooled']['lo']:+.3f}, {d['pooled']['hi']:+.3f}]  boards mean {bl['mean']:+.3f} +- {bl['se']:.3f} "
                f"(+{bl['boards_positive']}/-{bl['boards_negative']} of {bl['n_boards']})  per board " + " ".join(f"{d['per_board'][bn]['point']:+.3f}" for bn in held_boards))
        say(f"   [{time.time() - T0:.0f}s]")

    results.update(boards=board_info, training_palette=sorted(train_palette), colour_prior=colour_prior, trees=trees, loo=loo,
                   n_train_frames=len(train), n_held_frames=sum(len(v) for v in held_boards.values()), draws=args.draws, held_draws=args.held_draws,
                   loo_grid=dict(ks=[str(k) for k in KS], hs=HS), seed=args.seed, seconds=time.time() - T0)
    json.dump(results, open(args.out, "w"), indent=1)
    Path(args.out).with_suffix(".log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say(f"wrote {args.out}  [{time.time() - T0:.0f}s]")


if __name__ == "__main__":
    main()
