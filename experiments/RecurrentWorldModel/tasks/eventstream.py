"""EventStream -- a timing-dependent task where the answer needs REAL elapsed time,
not token count. The test bed for continuous-time PoPE (Stage 0/1 of the temporal fork).

A latent value is set by the most recent VAL event and **decays 1 unit per unit of
TIME**; the query asks for its current value:

    events at irregular cumulative times tau_1 < tau_2 < ... (random gaps)
    a VAL(k) event sets value = k at its time; NOISE events do nothing
    answer at the query (time tau_q) = max(0, k* - (tau_q - tau*))
        where (k*, tau*) is the most recent VAL event before the query

Crucially the gaps are random, so **token distance != elapsed time**. A model given
only integer positions cannot compute the decay; a model given the timestamps (as a
continuous PoPE coordinate, or as an input feature) can. That is the whole point of
the comparison. Each example returns its per-token timestamps.

Token ids: PAD=0, QUERY=1, VAL(k)=2+k (k in 0..V-1), NOISE=2+V.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

PAD = 0
QUERY = 1


@dataclass
class TemporalBatch:
    input_ids: torch.Tensor   # (B, T)
    targets: torch.Tensor     # (B, T)
    loss_mask: torch.Tensor   # (B, T)  -- 1.0 at the query position
    timestamps: torch.Tensor  # (B, T)  -- cumulative real time of each token

    def to(self, device) -> "TemporalBatch":
        return TemporalBatch(self.input_ids.to(device), self.targets.to(device),
                             self.loss_mask.to(device), self.timestamps.to(device))


class EventStream:
    def __init__(self, n_levels: int = 6, n_events: int = 10, max_gap: int = 3,
                 noise_frac: float = 0.3, seed: int = 0) -> None:
        self.V = n_levels
        self.n_events = n_events
        self.max_gap = max_gap
        self.noise_frac = noise_frac
        self.NOISE = 2 + self.V
        self.vocab_size = 3 + self.V          # PAD, QUERY, V values, NOISE
        self.seq_len = n_events + 1           # events + the query

    def _val(self, k: int) -> int:
        return 2 + k

    def sample(self, batch_size: int, gap_min: int = 1, gap_max: int | None = None,
               rng: random.Random | None = None) -> TemporalBatch:
        gap_max = self.max_gap if gap_max is None else gap_max
        rng = rng or random.Random()
        T = self.seq_len
        ids = torch.zeros(batch_size, T, dtype=torch.long)
        ts = torch.zeros(batch_size, T, dtype=torch.float32)
        tgt = torch.full((batch_size, T), PAD, dtype=torch.long)
        mask = torch.zeros(batch_size, T, dtype=torch.float32)
        for i in range(batch_size):
            tau = 0.0
            last_k: int | None = None
            last_tau: float | None = None
            for j in range(self.n_events):
                tau += rng.randint(gap_min, gap_max)
                ts[i, j] = tau
                if rng.random() < self.noise_frac:
                    ids[i, j] = self.NOISE
                else:
                    k = rng.randint(0, self.V - 1)
                    ids[i, j] = self._val(k)
                    last_k, last_tau = k, tau
            tau += rng.randint(gap_min, gap_max)            # the query
            q = self.n_events
            ids[i, q] = QUERY
            ts[i, q] = tau
            cur = 0 if last_k is None else max(0, last_k - int(round(tau - last_tau)))
            tgt[i, q] = self._val(cur)
            mask[i, q] = 1.0
        return TemporalBatch(input_ids=ids, targets=tgt, loss_mask=mask, timestamps=ts)
