"""Adversarial check (numbers lens) of Part II of notes/gradient_free_mixing_and_features.md.

Questions the note does not answer with a number:
  (1) How many DISTINCT boards are the 135 held-out frames? (collect_frames keeps every frame the explorer stood in on
      level 2 of LockPath and CollectAll -- if the frames differ only by the agent's cell, the effective test set is ~2
      boards and cell-level SEs are meaningless.)
  (2) How close are the held-out boards to the training boards (levels 0-1)? Cell agreement with the nearest training
      frame = the memorisation ceiling.
  (3) Frame-level bootstrap CIs (paired) for the differences the note reads as real: soft - hard, nb - hard, soft - nb.
  (4) Trivial baselines: copy the input; the 1-cell table P(y | centre colour); the 9 single-cell tables' product
      restricted to centre + 4-neighbours. If a 1-cell table already reaches most of the score, the window carries
      little.
  (5) Calibration of naive Bayes and the soft kernel (mean entropy of the predictive vs mean code length): the note
      says naive Bayes is "NOT over-sharpened" without a calibration number.

Reuses r_window_neighbours.py's apparatus (e35.collect_frames / corrupt_chain, arcgames.LocalRule) unchanged, seed 0.
    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/check_frames_apparatus.py  (~1-2 min)
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

NCOL = V + 1


def rows(frame):
    full, masked = LocalRule(1)._rows(frame)
    return masked + 1


def frame_stats(name, frames):
    F = np.stack(frames)
    n = len(F)
    varying = (F != F[0]).any(0).mean()
    # pairwise agreement
    flat = F.reshape(n, -1)
    agree = np.array([[(flat[i] == flat[j]).mean() for j in range(n)] for i in range(n)])
    off = agree[~np.eye(n, dtype=bool)]
    return dict(n=n, shape=list(F.shape[1:]), frac_cells_varying=float(varying),
                pairwise_agreement_mean=float(off.mean()), pairwise_agreement_min=float(off.min()))


def nearest_train_agreement(held, train):
    out = []
    for x in held:
        same = [t for t in range(len(train)) if train[t].shape == x.shape]
        if not same:
            out.append(0.0); continue
        Ts = np.stack([train[t] for t in same]).reshape(len(same), -1)
        out.append(float((Ts == x.reshape(-1)).mean(1).max()))
    return out


def boot_ci(per_frame_a, per_frame_b, w, rng, B=2000):
    """Paired bootstrap over frames of the weighted mean of (a - b). per_frame_* = correct counts, w = cells per frame."""
    a = np.asarray(per_frame_a, float); b = np.asarray(per_frame_b, float); w = np.asarray(w, float)
    n = len(a)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        diffs.append((a[idx].sum() - b[idx].sum()) / w[idx].sum())
    diffs = np.array(diffs)
    return dict(point=float((a.sum() - b.sum()) / w.sum()), lo=float(np.percentile(diffs, 2.5)),
                hi=float(np.percentile(diffs, 97.5)))


def main(seed=0, draws=4):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    lp = collect_frames(LockPath, [0, 1, 2])
    ca = collect_frames(CollectAll, [0, 1, 2])
    train = lp[0] + lp[1] + ca[0] + ca[1]
    held = lp.get(2, []) + ca.get(2, [])
    out = dict(frames_train=len(train), frames_held=len(held))
    out["held_boards"] = dict(lockpath_l2=frame_stats("LockPath L2", lp[2]), collectall_l2=frame_stats("CollectAll L2", ca[2]))
    out["train_boards"] = {k: frame_stats(k, v) for k, v in
                           dict(lockpath_l0=lp[0], lockpath_l1=lp[1], collectall_l0=ca[0], collectall_l1=ca[1]).items()}
    nta = nearest_train_agreement(held, train)
    out["held_vs_nearest_training_frame_cell_agreement"] = dict(mean=float(np.mean(nta)), min=float(np.min(nta)), max=float(np.max(nta)))
    print(f"frames: {len(train)} train, {len(held)} held  [{time.time() - t0:.0f}s]")
    for k, v in out["held_boards"].items():
        print(f"  held {k}: n={v['n']} shape={v['shape']} cells varying across frames {v['frac_cells_varying']:.3f}, "
              f"pairwise agreement mean {v['pairwise_agreement_mean']:.3f} min {v['pairwise_agreement_min']:.3f}")
    for k, v in out["train_boards"].items():
        print(f"  train {k}: n={v['n']} shape={v['shape']} cells varying {v['frac_cells_varying']:.3f}, pairwise agreement mean {v['pairwise_agreement_mean']:.3f}")
    print(f"  held frame vs NEAREST training frame, cell agreement: mean {np.mean(nta):.3f} min {np.min(nta):.3f} max {np.max(nta):.3f}")
    # colour prior of the held frames
    Hall = np.concatenate([x.reshape(-1) for x in held])
    bc = np.bincount(Hall, minlength=V)
    out["held_colour_prior_top"] = dict(colour=int(bc.argmax()), frac=float(bc.max() / bc.sum()))
    print(f"  held colour prior: most frequent colour {bc.argmax()} = {bc.max() / bc.sum():.3f} of cells")

    out["per_t"] = {}
    for t in (0.25, 0.5, 0.75):
        schedule = [0.0, t]
        W, Y = [], []
        for x in train:
            for _ in range(draws):
                xc = corrupt_chain(x, schedule, rng)[1]
                W.append(rows(xc)); Y.append(x.reshape(-1))
        W = np.concatenate(W).astype(np.int16); Y = np.concatenate(Y).astype(np.int16)
        keys, inv = np.unique(W, axis=0, return_inverse=True)
        counts = np.zeros((len(keys), V), dtype=np.int32)
        np.add.at(counts, (inv.reshape(-1), Y), 1)
        maj = counts.argmax(1)
        Wh, Yh, fid = [], [], []
        for i, x in enumerate(held):
            xc = corrupt_chain(x, schedule, rng)[1]
            Wh.append(rows(xc)); Yh.append(x.reshape(-1)); fid.append(np.full(x.size, i))
        Wh = np.concatenate(Wh).astype(np.int16); Yh = np.concatenate(Yh).astype(np.int16); fid = np.concatenate(fid)
        d_all = np.empty((len(Wh), len(keys)), dtype=np.int8)
        for s in range(0, len(Wh), 512):
            d_all[s:s + 512] = (Wh[s:s + 512, None, :] != keys[None, :, :]).sum(-1)
        hard_ok = maj[d_all.argmin(1)] == Yh
        # soft kernel at r_window's selected h (0.5 / 0.75 / 0.75)
        h = {0.25: 0.5, 0.5: 0.75, 0.75: 0.75}[t]
        K = np.exp(-d_all.astype(np.float32) / h)
        num = K @ counts.astype(np.float32) + 1e-3
        p_soft = num / num.sum(1, keepdims=True)
        soft_ok = p_soft.argmax(1) == Yh
        soft_bits = -np.log2(p_soft[np.arange(len(Yh)), Yh])
        soft_H = -(p_soft * np.log2(p_soft)).sum(1)
        # naive Bayes over 9 cells; and over the centre only; and centre + 4-neighbours
        joint = np.zeros((9, NCOL, V), dtype=np.float64)
        for i in range(9):
            np.add.at(joint[i], (W[:, i], Y), 1)
        py = np.bincount(Y, minlength=V).astype(np.float64) + 0.5
        p_c_given_y = (joint + 0.5) / (joint + 0.5).sum(1, keepdims=True)

        def nb(cells):
            logp = np.log(py)[None, :] + sum(np.log(p_c_given_y[i][Wh[:, i]]) for i in cells)
            logp -= logp.max(1, keepdims=True)
            p = np.exp(logp); p /= p.sum(1, keepdims=True)
            return p
        p_nb = nb(range(9)); nb_ok = p_nb.argmax(1) == Yh
        nb_bits = -np.log2(p_nb[np.arange(len(Yh)), Yh]); nb_H = -(p_nb * np.log2(p_nb)).sum(1)
        p_c = nb([4]); c_ok = p_c.argmax(1) == Yh                     # the 1-cell table P(y | centre)
        p_5 = nb([1, 3, 4, 5, 7]); n5_ok = p_5.argmax(1) == Yh          # centre + 4-neighbours
        input_ok = (Wh[:, 4] - 1) == Yh
        prior_ok = np.full(len(Yh), int(bc.argmax())) == Yh
        # per-frame bootstrap
        nf = len(held)
        w = np.bincount(fid, minlength=nf)
        pf = lambda ok: np.bincount(fid, weights=ok.astype(float), minlength=nf)
        brng = np.random.default_rng(1)
        res = dict(t=t, held_cells=int(len(Yh)),
                   acc=dict(input=float(input_ok.mean()), colour_prior=float(prior_ok.mean()), centre_only=float(c_ok.mean()),
                            nb_centre_plus4=float(n5_ok.mean()), nb9=float(nb_ok.mean()), hard=float(hard_ok.mean()),
                            soft=float(soft_ok.mean())),
                   bits=dict(nb9=float(nb_bits.mean()), soft=float(soft_bits.mean())),
                   entropy=dict(nb9=float(nb_H.mean()), soft=float(soft_H.mean())),
                   calib_gap_bits_minus_entropy=dict(nb9=float(nb_bits.mean() - nb_H.mean()), soft=float(soft_bits.mean() - soft_H.mean())),
                   nb9_mean_max_prob=float(p_nb.max(1).mean()), soft_mean_max_prob=float(p_soft.max(1).mean()),
                   boot=dict(soft_minus_hard=boot_ci(pf(soft_ok), pf(hard_ok), w, brng),
                             nb9_minus_hard=boot_ci(pf(nb_ok), pf(hard_ok), w, brng),
                             soft_minus_nb9=boot_ci(pf(soft_ok), pf(nb_ok), w, brng),
                             nb9_minus_centre=boot_ci(pf(nb_ok), pf(c_ok), w, brng)),
                   per_frame_acc_spread=dict(hard=[float(x) for x in np.percentile(pf(hard_ok) / w, [0, 25, 50, 75, 100])],
                                             soft=[float(x) for x in np.percentile(pf(soft_ok) / w, [0, 25, 50, 75, 100])]))
        # per game (LockPath L2 = the first len(lp[2]) frames, CollectAll L2 = the rest): does each difference hold on each board?
        g_lp = fid < len(lp[2])
        res["per_game"] = {}
        for gname, gm in (("lockpath_l2", g_lp), ("collectall_l2", ~g_lp)):
            res["per_game"][gname] = dict(cells=int(gm.sum()), hard=float(hard_ok[gm].mean()), soft=float(soft_ok[gm].mean()),
                                          nb9=float(nb_ok[gm].mean()), centre_only=float(c_ok[gm].mean()), input=float(input_ok[gm].mean()))
        out["per_t"][str(t)] = res
        a = res["acc"]
        for gname, gv in res["per_game"].items():
            print(f"      {gname}: cells {gv['cells']} input {gv['input']:.3f} centre {gv['centre_only']:.3f} nb9 {gv['nb9']:.3f} hard {gv['hard']:.3f} soft {gv['soft']:.3f}")
        print(f"t={t}: acc input {a['input']:.3f} | colour prior {a['colour_prior']:.3f} | centre-only table {a['centre_only']:.3f} | "
              f"nb centre+4 {a['nb_centre_plus4']:.3f} | nb9 {a['nb9']:.3f} | hard-NN {a['hard']:.3f} | soft h={h} {a['soft']:.3f}")
        print(f"      calibration: nb9 bits {res['bits']['nb9']:.3f} entropy {res['entropy']['nb9']:.3f} (gap {res['calib_gap_bits_minus_entropy']['nb9']:+.3f}, mean max-prob {res['nb9_mean_max_prob']:.3f}) | "
              f"soft bits {res['bits']['soft']:.3f} entropy {res['entropy']['soft']:.3f} (gap {res['calib_gap_bits_minus_entropy']['soft']:+.3f}, mean max-prob {res['soft_mean_max_prob']:.3f})")
        for k, v in res["boot"].items():
            print(f"      frame-bootstrap {k}: {v['point']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
        print(f"      per-frame accuracy (min/q1/med/q3/max): hard {['%.3f' % x for x in res['per_frame_acc_spread']['hard']]} soft {['%.3f' % x for x in res['per_frame_acc_spread']['soft']]}  [{time.time() - t0:.0f}s]")
    (HERE / "check_frames_apparatus.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {HERE / 'check_frames_apparatus.json'}")


if __name__ == "__main__":
    main()
