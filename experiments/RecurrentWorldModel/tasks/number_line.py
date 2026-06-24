"""NumberLineDiscovery -- phase-0 "teach the number line first" task (FEW_SHOT_ARITHMETIC.md, idea #2).

The diagnostic showed coherence has nothing to anchor to while value embeddings are random: it reaches
a self-consistent NON-addition solution (held-out flat at chance). So before few-shot addition we must
let a *metric space* emerge in the value rows. This task supplies the raw experience from which the
ring Z_m should be discovered -- NOT by hardcoding a Fourier embedding, but from:

  * successor   [VAL(a), SUCC, EQ, VAL((a+1) mod m)]   -- the cyclic generator (+1) as a movement
  * predecessor [VAL(a), PRED, EQ, VAL((a-1) mod m)]   -- its inverse (-1)
  * comparison  [VAL(a), CMP, VAL(b), EQ, VAL(dist)]   -- circular distance min(d, m-d): the ring metric

Successor/predecessor are the sensorimotor half (predict where a movement lands); comparison is the
metric half (recover distance on the circle). With coherence on the successor movement (compose +1 k
times == +k) and SIGReg, the value rows should organise into the ring -- the orbit of the generator.
That emergent geometry is what makes phase-1 addition few-shot. The scientific question this task lets
us ask: *does an ordered/cyclic manifold actually appear in the embeddings, and does it generalise
(held-out comparison accuracy), rather than memorise the pairs?*

Shares vocab.py with arithmetic.py so VAL rows align across phases. Same coord scheme as arithmetic:
axis0 = sequence position, axis1 = the numeral's value on VAL slots (0 elsewhere).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from . import vocab
from .vocab import CMP, EQ, PAD, PRED, SUCC


@dataclass
class NLBatch:
    input_ids: torch.Tensor   # (B, T) long
    targets: torch.Tensor     # (B, T) long   -- next-token targets
    loss_mask: torch.Tensor   # (B, T) float  -- 1.0 at the scored (EQ) position
    coord: torch.Tensor       # (B, T, 2) float -- [position, value]
    score_pos: int            # index of the scored EQ slot (its next token is the answer)

    def to(self, device) -> "NLBatch":
        return NLBatch(self.input_ids.to(device), self.targets.to(device),
                       self.loss_mask.to(device), self.coord.to(device), self.score_pos)


class NumberLineDiscovery:
    def __init__(self, modulus: int = 17, seed: int = 0, value_coord: bool = False) -> None:
        if modulus < 2:
            raise ValueError("modulus must be >= 2")
        self.P = modulus
        self.vocab_size = vocab.vocab_size(modulus)
        self.values = list(range(modulus))
        self.all_compare = [(a, b) for a in range(modulus) for b in range(modulus)]
        self._seed = seed
        # value_coord=False is the *discovery* setting: the numeral's value is NOT handed to the model
        # via the PoPE coord (which would supply the ring metric through the attention phase for free).
        # The model must build the number-line representation in its content embeddings from experience.
        self.value_coord = value_coord

    # --- the ring's structure (ground truth; supplied as experience, not as a prior in the model) ---
    def succ(self, a: int) -> int:
        return (a + 1) % self.P

    def pred(self, a: int) -> int:
        return (a - 1) % self.P

    def dist(self, a: int, b: int) -> int:
        d = (a - b) % self.P
        return min(d, self.P - d)                          # circular distance, 0..P//2

    # --- batch builders -------------------------------------------------------------------------
    def _movement_batch(self, values, op_tok, fn) -> NLBatch:
        """[VAL(a), MOVE, EQ, VAL(fn(a))]; score the EQ slot at index 2 (predicts the landing)."""
        B, T = len(values), 4
        ids = torch.zeros(B, T, dtype=torch.long)
        coord = torch.zeros(B, T, 2, dtype=torch.float)
        coord[:, :, 0] = torch.arange(T, dtype=torch.float)
        for i, a in enumerate(values):
            ans = fn(a)
            ids[i] = torch.tensor([vocab.val(a), op_tok, EQ, vocab.val(ans)])
            if self.value_coord:
                coord[i, 0, 1] = a
                coord[i, 3, 1] = ans
        targets = torch.full_like(ids, PAD)
        targets[:, :-1] = ids[:, 1:]                       # EQ(idx2) -> VAL(ans)
        loss_mask = torch.zeros(B, T)
        loss_mask[:, 2] = 1.0
        return NLBatch(ids, targets, loss_mask, coord, score_pos=2)

    def successor_batch(self, values=None) -> NLBatch:
        return self._movement_batch(self.values if values is None else values, SUCC, self.succ)

    def predecessor_batch(self, values=None) -> NLBatch:
        return self._movement_batch(self.values if values is None else values, PRED, self.pred)

    def compare_batch(self, pairs) -> NLBatch:
        """[VAL(a), CMP, VAL(b), EQ, VAL(dist(a,b))]; score the EQ slot at index 3."""
        B, T = len(pairs), 5
        ids = torch.zeros(B, T, dtype=torch.long)
        coord = torch.zeros(B, T, 2, dtype=torch.float)
        coord[:, :, 0] = torch.arange(T, dtype=torch.float)
        for i, (a, b) in enumerate(pairs):
            d = self.dist(a, b)
            ids[i] = torch.tensor([vocab.val(a), CMP, vocab.val(b), EQ, vocab.val(d)])
            if self.value_coord:
                coord[i, 0, 1] = a
                coord[i, 2, 1] = b
                coord[i, 4, 1] = d
        targets = torch.full_like(ids, PAD)
        targets[:, :-1] = ids[:, 1:]                       # EQ(idx3) -> VAL(dist)
        loss_mask = torch.zeros(B, T)
        loss_mask[:, 3] = 1.0
        return NLBatch(ids, targets, loss_mask, coord, score_pos=3)

    def split_compare(self, n_train: int, seed: int | None = None) -> tuple[list, list]:
        """Held-out split of comparison pairs -- lets us check the metric *generalises* (the line is
        real) rather than memorising pairs."""
        rng = random.Random(self._seed if seed is None else seed)
        pairs = self.all_compare[:]
        rng.shuffle(pairs)
        return pairs[:n_train], pairs[n_train:]
