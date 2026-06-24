"""HexTorus -- the hexagonal / conformal-isometry test (BLUEPRINT §A final piece; PURE_MATH §8).

The isotropic 2-D lattice. Three movement directions at 120° on a torus, chosen to **sum to zero**
(d0+d1+d2 = 0), so going around them returns to start: `R0·R1·R2 = I` (the hexagonal closure that makes
this a triangular/hex lattice rather than a square one). `R0,R1` already generate the 2-torus; `R2` is
the third (negative-diagonal) direction. The **conformal-isometry** property -- equal step size in every
direction (Schøyen "hexagons all the way down") -- is enforced by an isotropy term in training and is
the distinctive thing this tests: does optimising for isotropy produce the hexagonal symmetry?

Vocab: PAD,EQ, STEP_0/1/2, then VAL(cell)=VAL0+flat_index. coord = sequence position only.
"""

from __future__ import annotations

import torch

from .number_line import NLBatch

PAD, EQ = 0, 1


class HexTorus:
    DIRS = [(1, 0), (0, 1), (-1, -1)]                    # sum to zero -> R0·R1·R2 = I (hex closure)

    def __init__(self, m: int = 6, seed: int = 0) -> None:
        self.m = m
        self.N = m * m
        self.VAL0 = 2 + 3                                # ops 2,3,4
        self.vocab_size = self.VAL0 + self.N
        self.coords = [(q, r) for q in range(m) for r in range(m)]   # row-major: q outer, r inner
        self.cindex = {c: i for i, c in enumerate(self.coords)}
        self._seed = seed

    def val(self, c: tuple[int, int]) -> int:
        return self.VAL0 + self.cindex[c]

    def step(self, c: tuple[int, int], k: int) -> tuple[int, int]:
        d = self.DIRS[k]
        return ((c[0] + d[0]) % self.m, (c[1] + d[1]) % self.m)

    def step_batch(self, k: int) -> NLBatch:
        T = 4
        op = 2 + k
        ids = torch.zeros(self.N, T, dtype=torch.long)
        coord = torch.zeros(self.N, T, 2, dtype=torch.float)
        coord[:, :, 0] = torch.arange(T, dtype=torch.float)
        for i, c in enumerate(self.coords):
            nc = self.step(c, k)
            ids[i] = torch.tensor([self.val(c), op, EQ, self.val(nc)])
        targets = torch.full_like(ids, PAD)
        targets[:, :-1] = ids[:, 1:]
        loss_mask = torch.zeros(self.N, T)
        loss_mask[:, 2] = 1.0
        return NLBatch(ids, targets, loss_mask, coord, score_pos=2)
