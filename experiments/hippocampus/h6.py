"""Experiment H6: the hippocampus on real text, character-level language modelling (EXPERIMENT_H6.md).

Every prediction sees [MEM] + a memory passage of R*32 = 128 characters + [SEP] + the previous P = 128 characters + the
window of W = 64 characters being predicted (teacher-forced, causal). The memory policy differs by arm, nothing else:
  none     the memory is padding (no store)
  recent   the 128 characters just before the previous context: a longer context, no retrieval (the control)
  retrieve the store chunk sharing the most rarity-weighted (idf) words with the last 64 characters before the window,
           delivered with its continuation (RETRO-style). Retrieval is cued only by text BEFORE the predicted window.
Training: Cicero + Livy (95%), random windows, the store being the HIST characters before each window; on 25% of
windows ("re-reads") an earlier copy of the current region is also in the store (only `retrieve` can find it).
Evaluation: streams read once, front to back; the store holds every earlier chunk of the stream.
  heldout (Latin, in distribution), caesar (another author), mhg (Middle High German), and the SECOND READING:
  a 20,000-character passage read twice, bits per character on each pass.

Usage:  python h6.py --arm retrieve --seed 0   -> runs/h6/retrieve_s0.json (refuses to re-run)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "intelligent_constraints"))
from model import Net                                   # noqa: E402
from text6 import CHUNK, MAXW, V, chunk_sets, load       # noqa: E402

DEV = "cuda"
OUT = HERE / "runs" / "h6"
P, W, R, HIST = 128, 64, 4, 8192
MAXQ = 32                                               # distinct words of the previous context used as the cue
MEM, SEP, PAD = V, V + 1, V + 2
VOCAB = V + 3
SEQ = 1 + R * CHUNK + 1 + P + W - 1
CUE, REREAD = 64, 0.25                                  # cue: the last 64 characters; share of re-read training windows
MODEL = dict(d=128, n_layer=4, n_head=4, m=512)
STREAM, PASSAGE = 50000, 20000
STEPS = 3000                                            # set from a measured run before pre-registration


def on_dev(t):
    return {k: t[k].to(DEV).long() if k != "ids" else t[k].to(DEV) for k in ("ids", "wid", "sets")}


def unique_rows(x):
    """Distinct non-negative ids per row, packed into the last MAXQ columns (-1 elsewhere)."""
    s, _ = x.sort(dim=1)
    dup = torch.zeros_like(s, dtype=torch.bool)
    dup[:, 1:] = s[:, 1:] == s[:, :-1]
    s = torch.where(dup, torch.full_like(s, -1), s)
    return s.sort(dim=1).values[:, -MAXQ:]


def select(policy, T, idf, t, gen, train=True, reread=None):
    """The first chunk id (B,) of the memory passage (R contiguous chunks = 128 characters), -1 for none.
      recent    the passage that ends where the previous context begins
      retrieve  the chunk that shares the most (idf-weighted) words with the last CUE characters before the window,
                delivered WITH its continuation (RETRO-style), so the reader can copy what followed it then. Only
                passages that end before the previous context are candidates, except on a re-read training example
                (`reread`), where an earlier copy of the current region is also in the store."""
    B = len(t)
    p0 = t - P
    cb = p0 // CHUNK
    if policy == "none":
        return torch.full((B,), -1, device=DEV, dtype=torch.long)
    if policy == "recent":
        j = cb - R
        return torch.where(j >= 0, j, torch.full_like(j, -1))
    if train:
        H = HIST // CHUNK
        cand = cb[:, None] - H + torch.arange(H + 3, device=DEV)[None]      # +3: the current region (re-reads)
    else:
        cand = torch.arange(int(cb.max()), device=DEV)[None].expand(B, -1)
    valid = (cand >= 0) & (cand + R - 1 < cb[:, None])                     # the whole passage before the context
    if reread is not None:                                                  # an earlier copy of this region exists
        cue_chunk = (t - CUE) // CHUNK
        region = (cand >= cue_chunk[:, None] - 1) & (cand <= (t // CHUNK)[:, None])
        valid = valid | (region & reread[:, None])
    sets = T["sets"][cand.clamp(min=0)]                                     # (B, H, MAXW)
    q = unique_rows(T["wid"][(t - CUE)[:, None] + torch.arange(CUE, device=DEV)[None]])
    hit = ((sets[..., None] == q[:, None, None, :]) & (sets[..., None] >= 0)).any(-1)
    score = (hit * idf[sets.clamp(min=0)]).sum(-1)
    score = score + 1e-3 * torch.rand(score.shape, device=DEV, generator=gen)
    score = torch.where(valid, score, torch.full_like(score, -1e9))
    best = score.argmax(1)
    j = torch.gather(cand, 1, best[:, None])[:, 0]
    ok = torch.gather(valid, 1, best[:, None])[:, 0]
    return torch.where(ok, j, torch.full_like(j, -1))


def inputs(T, t, j):
    """[MEM] + the memory passage (R*CHUNK characters, or padding) + [SEP] + the previous context + the window."""
    B = len(t)
    pos = j.clamp(min=0)[:, None] * CHUNK + torch.arange(R * CHUNK, device=DEV)[None]
    m = torch.where(j[:, None] >= 0, T["ids"][pos.clamp(max=len(T["ids"]) - 1)].long(), torch.full_like(pos, PAD))
    sep = torch.where(j >= 0, torch.full_like(j, SEP), torch.full_like(j, PAD))[:, None]
    ctx = T["ids"][(t - P)[:, None] + torch.arange(P + W - 1, device=DEV)[None]].long()
    y = T["ids"][t[:, None] + torch.arange(W, device=DEV)[None]].long()
    return torch.cat([torch.full((B, 1), MEM, device=DEV), m, sep, ctx], 1), y


@torch.no_grad()
def stream_bits(net, T, idf, policy, gen, start, end):
    """Bits per character over [start, end) of a stream, window by window; also the share of retrieved slots filled."""
    ts = torch.arange(max(start, P), end - W + 1, W, device=DEV)
    tot, n, filled = 0.0, 0, 0.0
    net.eval()
    for i in range(0, len(ts), 64):
        t = ts[i:i + 64]
        mem = select(policy, T, idf, t, gen, train=False)
        x, y = inputs(T, t, mem)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            lp = net(x, last=W).float().log_softmax(-1)
        tot += -lp.gather(-1, y[..., None]).sum().item()
        n += y.numel()
        filled += (mem >= 0).float().sum().item()
    net.train()
    return round(tot / n / math.log(2), 4), round(filled / len(ts), 3)


def run(policy, seed, steps, batch=48, lr=1e-3):
    t0 = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    d = load()
    idf = d["idf"].to(DEV)
    T = on_dev(d["train"])
    net = Net(VOCAB, ctx=SEQ, pope=True, **MODEL).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01, fused=True)
    f = torch.compile(net)
    lo, hi = P + CHUNK * R, len(T["ids"]) - W - R * CHUNK
    log = []
    for s in range(1, steps + 1):
        fr = min(1.0, s / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups:
            g["lr"] = lr * fr
        t = torch.randint(lo, hi, (batch,), device=DEV, generator=gen)
        rr = torch.rand(batch, device=DEV, generator=gen) < REREAD
        x, y = inputs(T, t, select(policy, T, idf, t, gen, reread=rr))
        with torch.autocast(DEV, dtype=torch.bfloat16):
            loss = Fn.cross_entropy(f(x, last=W).float().reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 500 == 0 or s == steps:
            log.append(dict(step=s, loss_bits=round(loss.item() / math.log(2), 4)))
    train_secs = round(time.time() - t0, 1)
    res = dict(policy=policy, seed=seed, steps=steps, batch=batch, lr=lr, P=P, W=W, R=R, chunk=CHUNK, hist=HIST,
               cue=CUE, reread=REREAD,
               model=MODEL, pope=True, log=log, train_secs=train_secs, streams={})
    egen = torch.Generator(device=DEV).manual_seed(1234)
    for name in ("heldout", "caesar", "mhg"):
        S = {k: v[:STREAM] if k != "sets" else v[:STREAM // CHUNK] for k, v in d[name].items()}
        bits, filled = stream_bits(net, on_dev(S), idf, policy, egen, P, STREAM)
        res["streams"][name] = dict(bpc=bits, memory_filled=filled)
    h = d["heldout"]
    ids2 = torch.cat([h["ids"][:PASSAGE], h["ids"][:PASSAGE]])
    wid2 = torch.cat([h["wid"][:PASSAGE], h["wid"][:PASSAGE]])
    S2 = on_dev(dict(ids=ids2, wid=wid2, sets=torch.from_numpy(chunk_sets(wid2.numpy()))))
    first = stream_bits(net, S2, idf, policy, egen, P, PASSAGE)[0]
    second = stream_bits(net, S2, idf, policy, egen, PASSAGE, 2 * PASSAGE)[0]
    res["second_reading"] = dict(first_bpc=first, second_bpc=second, ratio=round(second / first, 4))
    res["secs"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["none", "recent", "retrieve"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="0 = the pre-registered budget; >0 = a smoke test, not saved")
    a = ap.parse_args()
    out = OUT / f"{a.arm}_s{a.seed}.json"
    if not a.steps and out.exists():
        print(f"{out} exists: not re-run")
        raise SystemExit
    r = run(a.arm, a.seed, a.steps or STEPS)
    print(json.dumps({k: v for k, v in r.items() if k != "log"}, indent=1))
    if not a.steps:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(r, open(out, "w"), indent=1)
