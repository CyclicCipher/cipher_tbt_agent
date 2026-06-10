"""ShiftSeq -- a distribution-shift task to test Δ-encoding vs absolute encoding
(Stage 2 of the temporal fork; Makushkin's "learn the derivative, not the value").

A latent value starts at v0 and takes L small random increments (the "derivatives").
The target is the TOTAL CHANGE (sum of increments) -- which is **shift-invariant**:
it does not depend on v0 at all. The same target is produced by two input
representations:

  * absolute: the running ABSOLUTE values [v0, v1, ..., vL]. The change is a tiny
    signal (0..(D-1)L) buried in large absolutes (v0 ~ 1000), and at test v0 SHIFTS
    to an unseen range -- so the model must extract a small difference from large,
    out-of-distribution numbers.
  * delta: the increments themselves [0, d0, ..., d_{L-1}] -- which DON'T depend on
    v0, so they're identical in train and test. The model just sums them.

Prediction: delta generalizes to the shift for free (it never sees the shift);
absolute must learn to difference large shifted values and should degrade OOD.
This is the concrete form of "distribution shift lives in the absolute; the
derivative is invariant" -- and the compressing-the-source-not-the-sample idea.

Inputs are REAL values (use a continuous-input transformer). Target is the integer
total change (classification, 0..(D-1)L). Score the last position.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class ShiftBatch:
    abs_input: torch.Tensor    # (B, T) float -- absolute values [v0..vL]
    delta_input: torch.Tensor  # (B, T) float -- [0, d0..d_{L-1}]
    target: torch.Tensor       # (B, T) long  -- total change at the last position
    loss_mask: torch.Tensor    # (B, T)

    def to(self, device) -> "ShiftBatch":
        return ShiftBatch(self.abs_input.to(device), self.delta_input.to(device),
                          self.target.to(device), self.loss_mask.to(device))


class ShiftSeq:
    def __init__(self, length: int = 8, n_deltas: int = 4, seed: int = 0) -> None:
        self.L = length
        self.D = n_deltas
        self.target_classes = (n_deltas - 1) * length + 1   # 0 .. (D-1)*L
        self.vocab_size = self.target_classes               # output classes
        self.seq_len = length + 1                            # v0 + L increments

    def sample(self, batch_size: int, v0_lo: float, v0_hi: float,
               rng: random.Random | None = None) -> ShiftBatch:
        rng = rng or random.Random()
        T = self.seq_len
        absin = torch.zeros(batch_size, T, dtype=torch.float32)
        delin = torch.zeros(batch_size, T, dtype=torch.float32)
        tgt = torch.zeros(batch_size, T, dtype=torch.long)
        mask = torch.zeros(batch_size, T, dtype=torch.float32)
        for i in range(batch_size):
            v0 = rng.uniform(v0_lo, v0_hi)
            deltas = [rng.randint(0, self.D - 1) for _ in range(self.L)]
            v = v0
            absin[i, 0] = v0
            for t, d in enumerate(deltas):
                v += d
                absin[i, t + 1] = v               # [v0, v1, ..., vL]
                delin[i, t + 1] = float(d)        # [0, d0, ..., d_{L-1}]
            tgt[i, T - 1] = sum(deltas)            # total change (shift-invariant)
            mask[i, T - 1] = 1.0
        return ShiftBatch(absin, delin, tgt, mask)
