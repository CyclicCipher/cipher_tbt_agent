"""Adversarial check (numbers lens) of Part I of notes/gradient_free_mixing_and_features.md.

The note's headline closed-form gain is expA's per-context grid posterior (chain + word): 2.152 vs chain 2.229 at
prefix 300k (-0.077), 1.776 vs 1.817 at 1.2M (-0.041), all on ONE 60k test slice (the first 60 000 characters of
De Bello Civili). The note asserts "differences of 0.005 bits/char or less are inside what another 60k slice could
move" without measuring it. This script:
  (1) re-runs expA's evaluate (imported unchanged) for `word` at prefix 300k and records the bits of chain / linear /
      grid_bayes_alpha0 / grid_bayes_by_ctx per 10k-character block -> block-wise paired differences and their SE;
  (2) evaluates the SAME trained experts on a SECOND, disjoint 60k slice of the book (characters 60 000-120 000, the
      experts having never seen 0-60 000) -> does the ordering and the size of the gain survive a slice change?
Everything else is expA's code: e34.build_experts / run_stream / Mixer, the 63-point grid, share 0.01.
    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/check_text_slice_noise.py   (~2 min)
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from e34 import Mixer, build_experts, run_stream                   # noqa: E402
from textlm import load_corpus                                     # noqa: E402
import expA_exponent_grid as expA                                  # noqa: E402

BLOCK = 10000


def evaluate_blocks(chain, other, mixer, stream, test, A, B):
    """expA.evaluate's loop with per-block bits for four predictors (the grid posteriors are expA.GridPosterior)."""
    K = len(A)
    names = ("chain", "linear", "grid_bayes_alpha0", "grid_bayes_by_ctx", "best_fixed_hindsight")
    n = len(test)
    nb = math.ceil(n / BLOCK)
    blk = {k: np.zeros(nb) for k in names}
    cum = np.zeros((nb, K))
    post_a0 = expA.GridPosterior(K, 1, alpha=0.0)
    post_ctx = expA.GridPosterior(K, expA.N_CTX, alpha=0.01)
    probs = np.zeros(2)
    s = stream
    for i, c in enumerate(test.tolist()):
        b = i // BLOCK
        keys = chain.predict(s)
        dC, _ = chain.dist(keys)
        k, N, cnt = other.predict(s)
        dO, seen = other.dist(k, N, cnt)
        logC = np.log(np.clip(dC, 1e-300, None)); logO = np.log(np.clip(dO, 1e-300, None))
        L = A[:, None] * logC[None, :] + B[:, None] * logO[None, :]
        L -= L.max(1, keepdims=True)
        G = np.exp(L); G /= G.sum(1, keepdims=True)
        pg = G[:, c]
        probs[0], probs[1] = dC[c], dO[c]
        blk["chain"][b] -= math.log2(probs[0])
        deep = chain.deepest(keys)
        ctx = s.cls() * 9 + deep
        p_lin, _w = mixer.mix(ctx, probs)
        blk["linear"][b] -= math.log2(p_lin)
        mixer.update(ctx, probs, p_lin)
        blk["grid_bayes_alpha0"][b] += post_a0.code(0, pg)
        blk["grid_bayes_by_ctx"][b] += post_ctx.code(ctx, pg)
        cum[b] -= np.log2(np.clip(pg, 1e-300, None))
        chain.update(keys, c)
        other.update(k, c)
        s.push(c)
    sizes = np.array([min(BLOCK, n - b * BLOCK) for b in range(nb)])
    best = int(np.argmin(cum.sum(0)))
    blk["best_fixed_hindsight"] = cum[:, best]
    return {k: v / sizes for k, v in blk.items()}, sizes, best


def summarize(tag, blk, sizes, grid, best):
    n = sizes.sum()
    tot = {k: float((v * sizes).sum() / n) for k, v in blk.items()}
    out = dict(total=tot, best_fixed=dict(a=grid[best][0], b=grid[best][1]), blocks={k: [float(x) for x in v] for k, v in blk.items()})
    print(f"  [{tag}] bits/char: " + " | ".join(f"{k} {v:.3f}" for k, v in tot.items()) + f" | best fixed (a,b)={grid[best]}")
    for k in ("linear", "grid_bayes_alpha0", "grid_bayes_by_ctx"):
        d = blk[k] - blk["chain"]
        se = d.std(ddof=1) / math.sqrt(len(d))
        out[f"diff_{k}_minus_chain"] = dict(mean=float(d.mean()), se_blocks=float(se), per_block=[float(x) for x in d],
                                             n_blocks_negative=int((d < 0).sum()))
        print(f"      {k} - chain per 10k block: " + " ".join(f"{x:+.3f}" for x in d) + f"  -> mean {d.mean():+.4f}, SE over blocks {se:.4f}, blocks < 0: {(d < 0).sum()}/{len(d)}")
    d = blk["grid_bayes_by_ctx"] - blk["linear"]
    print(f"      by_ctx - linear per block: " + " ".join(f"{x:+.3f}" for x in d) + f"  -> mean {d.mean():+.4f}, SE {d.std(ddof=1) / math.sqrt(len(d)):.4f}")
    out["diff_by_ctx_minus_linear"] = dict(mean=float(d.mean()), se_blocks=float(d.std(ddof=1) / math.sqrt(len(d))))
    return out


def main(train_chars=300000, test_chars=60000, other_name="word"):
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    text = np.concatenate(train).astype(np.int16)[:train_chars]
    a_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]; b_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    grid = [(a, b) for a in a_grid for b in b_grid]
    A = np.array([g[0] for g in grid]); B = np.array([g[1] for g in grid])
    print(f"check: {len(text)} training chars (prefix), two disjoint {test_chars}-char test slices of the held-out book ({len(test)} chars); V = {V}; K = {len(grid)}")
    t0 = time.time()
    ex = build_experts(V, alphabet)
    chain = ex[0]; other = next(e for e in ex if e.name == other_name)
    mixer = Mixer(2, expA.N_CTX)
    _tb, _solo, s, mixer = run_stream([chain, other], text, V, alphabet, mixer=mixer, track_solo=False)
    print(f"  trained [{time.time() - t0:.0f}s]")
    trained = (copy.deepcopy(chain), copy.deepcopy(other), copy.deepcopy(mixer), copy.deepcopy(s))
    report = dict(V=V, n_train=int(len(text)), test_chars=test_chars, other=other_name, grid=grid, block=BLOCK, slices={})
    for tag, lo in (("slice1_0-60k", 0), ("slice2_60k-120k", test_chars)):
        c_, o_, m_, s_ = (copy.deepcopy(x) for x in trained)
        if lo:
            # the stream context must be the characters just before the slice; push them without learning
            for ch in test[lo - 16:lo].tolist():
                s_.push(ch)
        seg = test[lo:lo + test_chars]
        blk, sizes, best = evaluate_blocks(c_, o_, m_, s_, seg, A, B)
        report["slices"][tag] = summarize(tag, blk, sizes, grid, best)
        print(f"  [{time.time() - t0:.0f}s]")
    (HERE / "check_text_slice_noise.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {HERE / 'check_text_slice_noise.json'}")


if __name__ == "__main__":
    main()
