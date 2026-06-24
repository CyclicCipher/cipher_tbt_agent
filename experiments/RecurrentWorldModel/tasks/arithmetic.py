"""ModularArithmetic -- the few-shot arithmetic task for FEW_SHOT_ARITHMETIC.md.

A single binary operation over Z_m: `a op b = c`, op in {add, mul}. The point is to learn the
*operation* from very few labeled pairs (target: addition in 2, multiplication in 3, <=10), so the
task exposes a **train/held-out split by pair** and an `N` knob.

Sequence (matches ModularChain conventions; causal LM, score the EQ position whose next token is the
answer -- the answer sits at the masked-from-attention final slot, no leak under causal attention):
    [VAL(a), OP, VAL(b), EQ, VAL(c)]            score index 3 (EQ) -> predict VAL(c)
Token ids: shared scheme in vocab.py (PAD,EQ,ADD,MUL,SUCC,PRED,CMP, then VAL(v)=VAL0+v) so the value
rows align with the discovery phase (number_line.py) for embedding transfer.

`coord` (B, T, 2) drives 2-axis PoPE:
    axis 0 = sequence position (where-in-sequence),
    axis 1 = the numeral's **value** (the number line / Z_m phase) for VAL tokens, 0 elsewhere.
This supplies the number line as the legitimate prior (PURE_MATH_FOR_ML.md / FEW_SHOT §"two levers"),
leaving the *operation* as the thing the objective must learn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from . import vocab
from .vocab import EQ, PAD, VAL0


@dataclass
class ArithBatch:
    input_ids: torch.Tensor   # (B, T) long
    targets: torch.Tensor     # (B, T) long   -- next-token targets
    loss_mask: torch.Tensor   # (B, T) float  -- 1.0 at the scored (EQ) position
    coord: torch.Tensor       # (B, T, 2) float -- [position, value]

    def to(self, device) -> "ArithBatch":
        return ArithBatch(self.input_ids.to(device), self.targets.to(device),
                          self.loss_mask.to(device), self.coord.to(device))


class ModularArithmetic:
    def __init__(self, modulus: int = 17, op: str = "add", seed: int = 0,
                 value_coord: bool = False) -> None:
        if modulus < 2:
            raise ValueError("modulus must be >= 2")
        if op not in ("add", "mul"):
            raise ValueError("op must be 'add' or 'mul'")
        self.P = modulus
        self.op = op
        self.op_tok = vocab.OP[op]
        # value_coord=False (default, the honest setting per PURE_MATH_FOR_ML.md §9): the numeral's
        # value is NOT injected via the PoPE coord. The number line must be the *learned* representation
        # geometry (from the discovery phase), not a coordinate handed in through the attention channel.
        self.value_coord = value_coord
        self.vocab_size = vocab.vocab_size(modulus)
        self.seq_len = 5                       # VAL(a) OP VAL(b) EQ VAL(c)
        self._fn = ((lambda a, b: (a + b) % self.P) if op == "add"
                    else (lambda a, b: (a * b) % self.P))
        self.all_pairs = [(a, b) for a in range(self.P) for b in range(self.P)]
        self._seed = seed

    def _val(self, v: int) -> int:
        return vocab.val(v)

    def apply(self, a: int, b: int) -> int:
        return self._fn(a, b)

    def split(self, n_train: int, seed: int | None = None) -> tuple[list, list]:
        """Shuffle all m^2 pairs, take the first `n_train` as the labeled set, rest held out."""
        rng = random.Random(self._seed if seed is None else seed)
        pairs = self.all_pairs[:]
        rng.shuffle(pairs)
        return pairs[:n_train], pairs[n_train:]

    def batch(self, pairs: list[tuple[int, int]]) -> ArithBatch:
        B, T = len(pairs), self.seq_len
        ids = torch.zeros(B, T, dtype=torch.long)
        coord = torch.zeros(B, T, 2, dtype=torch.float)
        coord[:, :, 0] = torch.arange(T, dtype=torch.float)            # position axis
        for i, (a, b) in enumerate(pairs):
            c = self._fn(a, b)
            ids[i] = torch.tensor([self._val(a), self.op_tok, self._val(b), EQ, self._val(c)])
            if self.value_coord:
                coord[i, 0, 1] = a                                     # value axis (VAL tokens only)
                coord[i, 2, 1] = b
                coord[i, 4, 1] = c
        targets = torch.full_like(ids, PAD)
        targets[:, :-1] = ids[:, 1:]                                   # next-token; EQ(idx3)->VAL(c)
        loss_mask = torch.zeros(B, T)
        loss_mask[:, 3] = 1.0                                          # score the EQ position
        return ArithBatch(ids, targets, loss_mask, coord)
