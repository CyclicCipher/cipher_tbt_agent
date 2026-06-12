"""MotifEcho -- a fair, stable autoregressive-rollout task for studying drift.

A short random motif (length m) is repeated forever: the sequence is periodic with
period m. Once the model has seen >= one period it KNOWS the whole continuation, so a
perfect model rolls out perfectly forever -- any drift in free-running generation is the
*architecture's*, not the task's (the lesson from the DriftField post-mortem: only test on
questions whose answer is determined by the inputs). This is the sequence analog of the
TBAF repo's "preserve the image's features over 10,000 frames".

Training: teacher-forced next-token prediction (the model must induct the period and copy
from m steps back). Eval: free-running rollout from a fixed context, measuring per-step
accuracy vs rollout depth -- the decay curve whose flatness is the whole question.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MotifBatch:
    tokens: torch.Tensor   # (B, L) the periodic sequence
    m: torch.Tensor        # (B,)   the period of each row (diagnostics)

    def to(self, device) -> "MotifBatch":
        return MotifBatch(self.tokens.to(device), self.m.to(device))


class MotifEcho:
    def __init__(self, vocab_size: int = 16, motif_min: int = 2, motif_max: int = 6,
                 context_len: int = 18, horizon: int = 60, seed: int = 0) -> None:
        assert context_len >= 2 * motif_max, "context must cover >=2 periods of any motif"
        self.V = vocab_size
        self.motif_min, self.motif_max = motif_min, motif_max
        self.context_len, self.horizon = context_len, horizon
        self.seq_len = context_len + horizon
        self.seed = seed

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> MotifBatch:
        g = generator
        m = torch.randint(self.motif_min, self.motif_max + 1, (batch_size,), generator=g)  # (B,)
        motif = torch.randint(0, self.V, (batch_size, self.motif_max), generator=g)         # (B, motif_max)
        pos = torch.arange(self.seq_len)                                                     # (L,)
        idx = pos[None, :] % m[:, None]                                                      # (B, L) in [0, m-1]
        tokens = torch.gather(motif, 1, idx)                                                 # (B, L) periodic
        return MotifBatch(tokens=tokens, m=m)
