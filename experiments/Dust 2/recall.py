"""Task 2: in-context associative recall (a structurally different task from task 1, for the diversity rule).

A sequence holds 12 random key→value pairs (keys distinct, from 16 key symbols; values from 16 value symbols), then 20
queries "key value" drawn from those 12 keys. Only the 20 answers are scored (other targets are -100, ignored by the
loss). Chance is log 16 = 2.77 nats; solving it needs an induction-style attention circuit. T = 64, V = 33
(0-15 keys, 16-31 values, 32 unused separator slot kept for symmetry).
"""
from __future__ import annotations

import torch

V = 33
NP, NQ = 12, 20


def batch(B, T, g, dev="cuda"):
    assert T == 2 * NP + 2 * NQ
    keys = torch.rand(B, 16, generator=g, device=dev).argsort(-1)[:, :NP]                 # distinct keys
    vals = torch.randint(0, 16, (B, NP), generator=g, device=dev) + 16
    qi = torch.randint(0, NP, (B, NQ), generator=g, device=dev)
    qk, qv = keys.gather(1, qi), vals.gather(1, qi)
    seq = torch.cat([torch.stack([keys, vals], -1).view(B, -1), torch.stack([qk, qv], -1).view(B, -1)], 1)  # (B, 64)
    idx = seq
    tgt = torch.full_like(seq, -100)
    pos = 2 * NP + 2 * torch.arange(NQ, device=dev)       # query-key positions; the target there is the value
    tgt[:, pos] = qv
    return idx, tgt
