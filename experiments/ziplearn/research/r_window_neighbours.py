"""R1 (reader's feel-experiment for refs/gradient_free_features_and_denoisers.md) -- is E35's memory block a
non-local-means denoiser with a hard kernel, and how much of the held-out data falls outside the lookup?

Reuses E35's apparatus unchanged (`collect_frames`, `corrupt_chain` from e35.py; `LocalRule` from arcgames.py).
One-shot setting only (corrupted at t -> clean centre), radius 1 (9 cells, 16 colours + border), the same frames as
E35 (LockPath + CollectAll levels 0-1 train, level 2 held out).

Measured, per t in {0.25, 0.5, 0.75}:
  (a) distinct training windows; fraction of held-out windows unseen; Hamming distance to the nearest stored window;
  (b) accuracy of the HARD nearest-window rule (E35's `NearestRule`) split by that distance;
  (c) the SOFT kernel: P(y | w) = sum_s K(d(w, s)) n_s(y) / sum_s K(d(w, s)) n_s, K(d) = exp(-d / h), over ALL stored
      windows s (Nadaraya-Watson on Hamming distance = non-local means with an exponential kernel on a discrete
      alphabet); h chosen by leave-one-out log-likelihood on the training windows (closed-form selection over a grid,
      no gradient); h -> 0 recovers the hard rule;
  (d) the FACTORED block: naive Bayes P(y | w) ~ P(y) prod_i P(w_i | y) from the same counts (the multiplicative
      combination with unit exponents -- E34's raw product on frames), and the per-cell conditionals mixed linearly
      (the Bayesian mixture); both generalise by shared structure, not lookup.

    python experiments/ziplearn/research/r_window_neighbours.py    (CPU, ~1 min) -> research/r_window_neighbours.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))
from e35 import collect_frames, corrupt_chain                     # noqa: E402
from arcgames import LocalRule, V                                  # noqa: E402
from tasks.games import LockPath, CollectAll                       # noqa: E402

NCOL = V + 1                                                       # colours + the border value (-1 -> index 0)


def rows(frame):
    full, masked = LocalRule(1)._rows(frame)
    return masked + 1                                              # shift so border = 0, colours = 1..16


def main(seed=0, draws=4):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    lp = collect_frames(LockPath, [0, 1, 2])
    ca = collect_frames(CollectAll, [0, 1, 2])
    train = lp[0] + lp[1] + ca[0] + ca[1]
    held = lp.get(2, []) + ca.get(2, [])
    print(f"frames: {len(train)} training, {len(held)} held out  [{time.time() - t0:.0f}s]", flush=True)
    out = {"frames_train": len(train), "frames_held": len(held), "per_t": {}}
    for t in (0.25, 0.5, 0.75):
        schedule = [0.0, t]
        # training pairs: corrupted window -> clean centre
        W, Y = [], []
        for x in train:
            for _ in range(draws):
                xc = corrupt_chain(x, schedule, rng)[1]
                W.append(rows(xc)); Y.append(x.reshape(-1))
        W = np.concatenate(W).astype(np.int16); Y = np.concatenate(Y).astype(np.int16)
        # distinct stored windows with their count vectors n_s(y)
        keys, inv = np.unique(W, axis=0, return_inverse=True)
        counts = np.zeros((len(keys), V), dtype=np.int32)
        np.add.at(counts, (inv.reshape(-1), Y), 1)
        n_s = counts.sum(1)
        maj = counts.argmax(1)
        # held-out windows
        Wh, Yh = [], []
        for x in held:
            xc = corrupt_chain(x, schedule, rng)[1]
            Wh.append(rows(xc)); Yh.append(x.reshape(-1))
        Wh = np.concatenate(Wh).astype(np.int16); Yh = np.concatenate(Yh).astype(np.int16)
        # (a) nearest stored window, Hamming
        d_all = np.empty((len(Wh), len(keys)), dtype=np.int8)
        for s in range(0, len(Wh), 512):
            d_all[s:s + 512] = (Wh[s:s + 512, None, :] != keys[None, :, :]).sum(-1)
        dmin = d_all.min(1)
        nearest = d_all.argmin(1)
        hard_pred = maj[nearest]
        hard_ok = hard_pred == Yh
        dist_hist = {int(d): int((dmin == d).sum()) for d in np.unique(dmin)}
        acc_by_d = {int(d): float(hard_ok[dmin == d].mean()) for d in np.unique(dmin)}
        # (c) soft Hamming kernel; h by leave-one-out on a training subsample (each stored window against the others)
        sub = rng.choice(len(keys), size=min(2000, len(keys)), replace=False)
        d_tr = (keys[sub, None, :] != keys[None, :, :]).sum(-1).astype(np.float32)
        d_tr[np.arange(len(sub)), sub] = 99                        # leave the window itself out
        best_h, best_ll = None, -np.inf
        for h in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            K = np.exp(-d_tr / h)
            num = K @ counts.astype(np.float32) + 1e-3             # (sub, V)
            p = num / num.sum(1, keepdims=True)
            # log-likelihood of the held-out window's own counts under p
            ll = float((counts[sub] * np.log(p)).sum() / counts[sub].sum())
            if ll > best_ll:
                best_ll, best_h = ll, h
        K = np.exp(-d_all.astype(np.float32) / best_h)
        num = K @ counts.astype(np.float32) + 1e-3
        p_soft = num / num.sum(1, keepdims=True)
        soft_pred = p_soft.argmax(1)
        soft_ok = soft_pred == Yh
        soft_bits = float(-np.log2(p_soft[np.arange(len(Yh)), Yh]).mean())
        # (d) factored: naive Bayes over the 9 cells, and the linear mixture of the 9 per-cell conditionals
        # per-cell tables P(w_i = c | y) and P(y | w_i = c) from the same pairs
        joint = np.zeros((9, NCOL, V), dtype=np.float64)          # cell i, colour c at i, target y
        for i in range(9):
            np.add.at(joint[i], (W[:, i], Y), 1)
        py = np.bincount(Y, minlength=V).astype(np.float64) + 0.5
        p_c_given_y = (joint + 0.5) / (joint + 0.5).sum(1, keepdims=True)        # (9, NCOL, V): sums over c
        p_y_given_c = (joint + 0.5) / (joint + 0.5).sum(2, keepdims=True)        # (9, NCOL, V): sums over y
        logp = np.log(py)[None, :] + sum(np.log(p_c_given_y[i][Wh[:, i]]) for i in range(9))
        logp -= logp.max(1, keepdims=True)
        p_nb = np.exp(logp); p_nb /= p_nb.sum(1, keepdims=True)
        nb_ok = p_nb.argmax(1) == Yh
        nb_bits = float(-np.log2(p_nb[np.arange(len(Yh)), Yh]).mean())
        p_lin = sum(p_y_given_c[i][Wh[:, i]] for i in range(9)) / 9
        lin_ok = p_lin.argmax(1) == Yh
        lin_bits = float(-np.log2(p_lin[np.arange(len(Yh)), Yh]).mean())
        # the corrupted input itself and the majority-of-seen baseline
        input_ok = (Wh[:, 4] - 1) == Yh
        res = dict(
            t=t, n_pairs=int(len(W)), distinct_windows=int(len(keys)), held_windows=int(len(Wh)),
            frac_unseen=float((dmin > 0).mean()), dist_hist=dist_hist, hard_acc_by_dist=acc_by_d,
            input_acc=float(input_ok.mean()), hard_acc=float(hard_ok.mean()),
            soft_h=best_h, soft_acc=float(soft_ok.mean()), soft_bits=soft_bits,
            naive_bayes_acc=float(nb_ok.mean()), naive_bayes_bits=nb_bits,
            linear_mix_acc=float(lin_ok.mean()), linear_mix_bits=lin_bits,
        )
        out["per_t"][str(t)] = res
        print(f"t={t}: {len(keys)} distinct windows from {len(W)} pairs; held-out unseen {res['frac_unseen']:.3f}, "
              f"dist hist {dist_hist}; acc input {res['input_acc']:.3f} | hard-NN {res['hard_acc']:.3f} "
              f"(by dist {{{', '.join(f'{k}: {v:.3f}' for k, v in acc_by_d.items())}}}) | soft h={best_h} "
              f"{res['soft_acc']:.3f} ({soft_bits:.2f} bits) | naive Bayes {res['naive_bayes_acc']:.3f} ({nb_bits:.2f} bits) "
              f"| linear mix {res['linear_mix_acc']:.3f} ({lin_bits:.2f} bits)  [{time.time() - t0:.0f}s]", flush=True)
    (HERE / "r_window_neighbours.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
