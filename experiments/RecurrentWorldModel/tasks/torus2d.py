"""Torus2D -- the 2-D / multi-generator generality test (BLUEPRINT §A, §G #4, §H gap A).

The first step past a single cyclic group: a 2-torus (ℤ/mx × ℤ/my) with **two** movement generators --
step in x (Rₓ) and step in y (R_y). Discovery must learn a *factored* 2-D frame `z(x,y)` such that
`z(x+1,y)=Rₓ·z(x,y)` and `z(x,y+1)=R_y·z(x,y)`, both orbits closing, and **Rₓ,R_y commuting** (moving
x-then-y == y-then-x: path independence = reference-frame coherence in 2-D). Then 2-D navigation should
fall out zero-shot: `Rₓᵃ·R_yᵇ·z(x,y) = z(x+a, y+b)` -- move by a vector. This is the literal grid-cell
geometry (a square 2-torus; hexagonal is a later refinement), and the honest pre-agent generality stress.

Self-contained vocab (one token per grid cell): PAD,EQ,STEP_X,STEP_Y, then VAL(cell)=VAL0+index.
`coord` carries sequence position only (value_coord=False: the frame must be learned, not injected).
"""

from __future__ import annotations

import torch

from .number_line import NLBatch

PAD, EQ, STEP_X, STEP_Y, VAL0 = 0, 1, 2, 3, 4


class Torus2D:
    def __init__(self, mx: int = 5, my: int = 5, seed: int = 0) -> None:
        self.mx, self.my = mx, my
        self.N = mx * my
        self.P = self.N                                  # probe-compat (# elements)
        self.vocab_size = VAL0 + self.N
        self._seed = seed

    def idx(self, x: int, y: int) -> int:
        return x * self.my + y                           # row-major: x outer, y inner

    def val(self, x: int, y: int) -> int:
        return VAL0 + self.idx(x, y)

    def succ_x(self, x: int, y: int) -> tuple[int, int]:
        return ((x + 1) % self.mx, y)

    def succ_y(self, x: int, y: int) -> tuple[int, int]:
        return (x, (y + 1) % self.my)

    def step_batch(self, axis: int) -> NLBatch:
        """[VAL(x,y), STEP_axis, EQ, VAL(succ_axis(x,y))] for all cells in index order; score EQ (idx 2)."""
        op = STEP_X if axis == 0 else STEP_Y
        succ = self.succ_x if axis == 0 else self.succ_y
        T = 4
        ids = torch.zeros(self.N, T, dtype=torch.long)
        coord = torch.zeros(self.N, T, 2, dtype=torch.float)
        coord[:, :, 0] = torch.arange(T, dtype=torch.float)
        for x in range(self.mx):
            for y in range(self.my):
                i = self.idx(x, y)
                nx, ny = succ(x, y)
                ids[i] = torch.tensor([self.val(x, y), op, EQ, self.val(nx, ny)])
        targets = torch.full_like(ids, PAD)
        targets[:, :-1] = ids[:, 1:]
        loss_mask = torch.zeros(self.N, T)
        loss_mask[:, 2] = 1.0
        return NLBatch(ids, targets, loss_mask, coord, score_pos=2)
