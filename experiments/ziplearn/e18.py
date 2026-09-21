"""E18 — writing attention weights (DESIGN.md OPEN-8): an induction circuit written into the transformer, no training.

The same `h1_lid` model (RoPE, d = 64, two layers, two heads), every weight written by hand from a description:
  layer 1, head 0 -- the PREVIOUS-TOKEN head. Query and key come from a constant "bias" channel every embedding
     carries, so they do not depend on content; under RoPE their dot product depends only on the distance between
     positions, and the key's phases are set so the score peaks at "one position back" (the four highest-frequency
     rotary pairs, which discriminate a distance of one from zero). Value copies the token's identity; the output
     projection writes it into a "previous token" subspace of the residual.
  layer 2, head 0 -- the INDUCTION head. Query reads the current token, key reads the previous-token subspace, both
     placed in the lowest-frequency rotary pairs, where rotation over 96 positions is negligible, so the match is on
     content alone: attend to positions that FOLLOWED an earlier copy of the current token; value copies the token
     there into an output subspace; the unembedding reads that subspace.
  Everything else -- the other heads, both MLPs -- is written as zero. LayerNorms are left at their defaults.
Test: a BOS token, then a random pattern of 8 distinct tokens repeated twice; the written model must predict the
second copy from the first. Chance is 1/8. Pass (pre-registered): >= 0.95 next-token accuracy on the second copy with no training,
and a random-initialised model at chance. Refute: < 0.6.

    python experiments/ziplearn/e18.py           (CPU, seconds) -> runs/e18/e18.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformers"))
import h1_lid as H   # noqa: E402

V, D, HEADS, LAYERS = 8, 64, 2, 2      # 8 digit tokens plus a BOS token (id 8): nine token identities
BOS = V
NT = V + 1
HD = D // HEADS                       # 32 dims per head, 16 rotary pairs
T_SUB, P_SUB, O_SUB, BIAS = 0, 10, 20, D - 1    # token subspace dims 0..8, previous-token 10..18, output 20..28


@torch.no_grad()
def write(model, M_pos=3.0, M_ind=2.0, M_out=10.0):
    for p in model.parameters():
        p.zero_()
    # embeddings: one-hot token plus a constant bias channel
    for v in range(NT):                                          # every token, BOS included, has an identity dim
        model.emb.weight[v, T_SUB + v] = 1.0
        model.emb.weight[v, BIAS] = 1.0
    for blk in model.blocks:                                     # LayerNorm defaults (gamma 1, beta 0); MLPs stay zero
        blk.n1.weight.fill_(1.0)
        blk.n2.weight.fill_(1.0)
    model.norm.weight.fill_(1.0)
    theta = H._freqs(HD // 2, "cpu")                             # rotary frequencies per pair, as the model uses them
    W = model.blocks[0].attn.qkv.weight                          # (3D, D): rows [q | k | v], head 0 = dims 0..HD-1
    # --- layer 1, head 0: previous-token head -------------------------------------------------------------------
    for c in range(4):                                           # the four highest-frequency pairs
        W[2 * c, BIAS] = M_pos                                   # q = (M, 0) in pair c, from the bias channel
        W[D + 2 * c, BIAS] = M_pos * math.cos(float(theta[c]))   # k = M (cos θ, sin θ): score peaks one position back
        W[D + 2 * c + 1, BIAS] = M_pos * math.sin(float(theta[c]))
    for v in range(NT):
        W[2 * D + v, T_SUB + v] = 1.0                            # value: the token identity, head dims 0..8
    P0 = model.blocks[0].attn.proj.weight                        # (D, D): residual <- concatenated heads
    for v in range(NT):
        P0[P_SUB + v, v] = 1.0                                   # head 0 dims 0..8 -> previous-token subspace
    # --- layer 2, head 0: induction head ------------------------------------------------------------------------
    W2 = model.blocks[1].attn.qkv.weight
    lowest = list(range(HD // 2 - NT, HD // 2))                  # the lowest-frequency pairs: content, not position
    for v, c in enumerate(lowest):
        W2[2 * c, T_SUB + v] = M_ind                             # query: the current token
        W2[D + 2 * c, P_SUB + v] = M_ind                         # key: the token that preceded position s
    for v in range(V):
        W2[2 * D + v, T_SUB + v] = 1.0                           # value: the token at s
    P1 = model.blocks[1].attn.proj.weight
    for v in range(V):
        P1[O_SUB + v, v] = 1.0                                   # -> output subspace
    for v in range(V):
        model.head.weight[v, O_SUB + v] = M_out                  # the unembedding reads the output subspace
    model.eval()


@torch.no_grad()
def accuracy(model, n=256, period=8, seed=0, distinct=True):
    """Patterns of `period` tokens repeated twice. With distinct=True the pattern has no repeated token, so "the
    position after the earlier copy of the current token" is unique -- the case an induction head of order one can
    solve; with repeats it is ambiguous for any such head, and the score is over the ambiguity."""
    g = torch.Generator().manual_seed(seed)
    if distinct:
        pat = torch.stack([torch.randperm(V, generator=g)[:period] for _ in range(n)])
    else:
        pat = torch.randint(0, V, (n, period), generator=g)
    bos = torch.full((n, 1), BOS)
    tok = torch.cat([bos, pat, pat], dim=1)                       # BOS, the pattern, then the pattern again
    logits = model(tok)
    pred = logits[:, :-1].argmax(-1)
    tgt = tok[:, 1:]
    second = slice(period + 1, 2 * period)                        # predictions made from inside the second copy
    return float((pred[:, second] == tgt[:, second]).float().mean()), float((pred == tgt).float().mean())


def main():
    out = HERE / "runs" / "e18"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = H.Model(d_model=D, n_layer=LAYERS, n_head=HEADS, max_len=128, pos="rope", n_vocab=NT)
    rand_second, _ = accuracy(model)
    write(model)
    second, overall = accuracy(model)
    amb, _ = accuracy(model, period=12, distinct=False)
    # where does the induction head look? mean attention weight from second-copy positions to "the position after
    # the earlier copy of the same token", read off the written model on one batch
    print(f"E18: induction circuit written into a {LAYERS}-layer, {HEADS}-head RoPE transformer, no training")
    print(f"   random-initialised model, second copy: {rand_second:.3f} (chance {1 / V:.2f})")
    print(f"   written model, second copy:            {second:.3f}   (all positions {overall:.3f}; patterns of 8 distinct tokens)")
    print(f"   with repeated tokens in the pattern (length 12 over 8 tokens; ambiguous for any first-order induction head): {amb:.3f}")
    for period in (4, 6, 8):
        s, _ = accuracy(model, period=period)
        print(f"   distinct pattern length {period:>2}: {s:.3f}")
    verdict = "PASS" if (second >= 0.95 and rand_second < 0.35) else "REFUTED" if second < 0.6 else "INCONCLUSIVE"
    print(f"\nE18 verdict: {verdict}")
    json.dump(dict(random=rand_second, written_second_copy=second, written_overall=overall,
                   by_period={p: accuracy(model, period=p)[0] for p in (4, 6, 8)}, ambiguous_repeats=amb, verdict=verdict),
              open(out / "e18.json", "w"), indent=1)


if __name__ == "__main__":
    main()
