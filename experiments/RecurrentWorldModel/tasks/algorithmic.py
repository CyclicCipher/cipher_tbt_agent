"""ModularChain -- a compositional reasoning task with a difficulty knob.

Each example composes a chain of affine maps over Z_P and asks for the result:

    v0  o1 o2 ... on  =  ans          where  ans = (o_n o ... o o_1)(v0)  mod P
    each o_i is a fixed affine map  x |-> (a_i * x + b_i) mod P

Why this task (Stage 0):
  * **Difficulty = chain length n** -- an explicit, recordable label. Tests the
    Risk-1 adaptive-compute claim (do iterations-to-converge rise with n?) and
    the gate (does iteration beat fixed depth as n grows past training range?).
  * **Genuinely compositional** -- the answer is a composition of n operations,
    so a fixed-compute model must pattern-match while an iterative one can
    actually compose. (It also pre-stages the Stage 2 A-then-B composition probe.)
  * **Cheap & memorization-resistant** -- the space is P * M^n; for P=7, M=8,
    n>=3 there are >3500 chains, far more than a small model can memorize.

Sequence layout (fixed length for batching; PAD at the end):
    [VAL(v0), OP(o1), ..., OP(on), EQ, VAL(ans), PAD, PAD, ...]
We score the EQ position, whose next token is the answer.

Token ids:  PAD=0, EQ=1, VAL(v)=2+v (v in 0..P-1), OP(o)=2+P+o (o in 0..M-1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

PAD = 0
EQ = 1


@dataclass
class Batch:
    input_ids: torch.Tensor   # (B, T) long
    targets: torch.Tensor     # (B, T) long  -- next-token targets
    loss_mask: torch.Tensor   # (B, T) float -- 1.0 at the scored (answer) position
    difficulty: torch.Tensor  # (B,) long    -- chain length n

    def to(self, device) -> "Batch":
        return Batch(
            self.input_ids.to(device), self.targets.to(device),
            self.loss_mask.to(device), self.difficulty.to(device),
        )


class ModularChain:
    def __init__(self, modulus: int = 7, n_ops: int = 8, max_len: int = 8, seed: int = 0) -> None:
        if modulus < 2:
            raise ValueError("modulus must be >= 2")
        self.P = modulus
        self.M = n_ops
        self.max_len = max_len
        rng = random.Random(seed)
        # fixed bank of invertible affine maps x -> (a*x + b) mod P
        self.ops: list[tuple[int, int]] = []
        while len(self.ops) < n_ops:
            a = rng.randrange(1, modulus)   # a != 0 keeps maps non-degenerate
            b = rng.randrange(0, modulus)
            if (a, b) not in self.ops and (a, b) != (1, 0):
                self.ops.append((a, b))
        self.vocab_size = 2 + self.P + self.M
        # sequence: v0 + max_len ops + EQ + ans
        self.seq_len = 1 + max_len + 1 + 1

    # -- token helpers --
    def _val(self, v: int) -> int:
        return 2 + v

    def _op(self, o: int) -> int:
        return 2 + self.P + o

    def _apply(self, v0: int, op_ids: list[int]) -> int:
        x = v0
        for o in op_ids:
            a, b = self.ops[o]
            x = (a * x + b) % self.P
        return x

    def sample(
        self, batch_size: int, len_min: int, len_max: int, rng: random.Random | None = None
    ) -> Batch:
        if len_max > self.max_len:
            raise ValueError(f"len_max {len_max} exceeds max_len {self.max_len}")
        rng = rng or random.Random()
        T = self.seq_len
        ids = torch.zeros(batch_size, T, dtype=torch.long)
        diffs = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            n = rng.randint(len_min, len_max)
            v0 = rng.randrange(self.P)
            op_ids = [rng.randrange(self.M) for _ in range(n)]
            ans = self._apply(v0, op_ids)
            seq = [self._val(v0)] + [self._op(o) for o in op_ids] + [EQ, self._val(ans)]
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            diffs[i] = n
        # next-token targets; score only the EQ position (predicts the answer)
        targets = torch.full_like(ids, PAD)
        targets[:, :-1] = ids[:, 1:]
        loss_mask = (ids == EQ).float()
        return Batch(input_ids=ids, targets=targets, loss_mask=loss_mask, difficulty=diffs)
