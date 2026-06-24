"""TorusND -- the n-D generality test (BLUEPRINT §A gap, after the 2-torus).

Generalises Torus2D to an n-torus ℤ/m_0 × ... × ℤ/m_{n-1} with **n commuting generators** (one per axis).
Function (BLUEPRINT §H): an n-torus codes a **multi-attribute state** -- position, colour, shape, count,
... as independent axes -- with independent movement along each. It is the bridge from "a number line"
to "a state space", and toward gap C (feature⊗location binding). Same machinery as 2-D, just n generators.

Self-contained vocab: PAD,EQ, STEP_0..STEP_{n-1}, then VAL(cell)=VAL0+flat_index (row-major).
`coord` carries sequence position only (the frame must be learned, not injected).
"""

from __future__ import annotations

from itertools import product

import torch

from .number_line import NLBatch

PAD, EQ = 0, 1


class TorusND:
    def __init__(self, shape: tuple[int, ...] = (4, 4, 4), seed: int = 0) -> None:
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.N = 1
        for s in self.shape:
            self.N *= s
        self.VAL0 = 2 + self.ndim                         # ops occupy 2 .. 2+ndim-1
        self.vocab_size = self.VAL0 + self.N
        self.coords = list(product(*[range(s) for s in self.shape]))   # row-major (last axis fastest)
        self.cindex = {c: i for i, c in enumerate(self.coords)}
        self._seed = seed

    def val(self, c: tuple[int, ...]) -> int:
        return self.VAL0 + self.cindex[c]

    def succ(self, c: tuple[int, ...], axis: int) -> tuple[int, ...]:
        c = list(c)
        c[axis] = (c[axis] + 1) % self.shape[axis]
        return tuple(c)

    def step_batch(self, axis: int) -> NLBatch:
        """[VAL(c), STEP_axis, EQ, VAL(succ_axis(c))] for all cells in row-major order; score EQ (idx 2)."""
        T = 4
        op = 2 + axis
        ids = torch.zeros(self.N, T, dtype=torch.long)
        coord = torch.zeros(self.N, T, 2, dtype=torch.float)
        coord[:, :, 0] = torch.arange(T, dtype=torch.float)
        for i, c in enumerate(self.coords):
            nc = self.succ(c, axis)
            ids[i] = torch.tensor([self.val(c), op, EQ, self.val(nc)])
        targets = torch.full_like(ids, PAD)
        targets[:, :-1] = ids[:, 1:]
        loss_mask = torch.zeros(self.N, T)
        loss_mask[:, 2] = 1.0
        return NLBatch(ids, targets, loss_mask, coord, score_pos=2)
