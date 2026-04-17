"""
task.py - Modular arithmetic task for grokking experiments.

Generates all (a, op, b, c) triples for modular addition and subtraction
on Z/pZ with a clean train/test split (no (a, b, op) triple appears in both).

Token vocabulary:
    0 .. p-1   : numbers
    p          : '+' operator
    p+1        : '-' operator
    p+2        : '=' separator
    VOCAB_SIZE = p + 3

Sequence format: [a, op, b, =]  (length 4)
Target: c = (a op b) mod p  -- predicted from logits at position 3.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class ModArithTask:
    """Modular arithmetic task on Z/pZ.

    Attributes:
        p           : prime modulus
        vocab_size  : p + 3
        op_add      : token index for '+'
        op_sub      : token index for '-'
        eq          : token index for '='
        seq_len     : 4  ([a, op, b, =])
        pred_pos    : 3  (position where the model predicts the answer)
    """

    def __init__(self, p: int = 97, train_frac: float = 0.5, seed: int = 42):
        assert p >= 2, "p must be at least 2"
        self.p          = p
        self.train_frac = train_frac
        self.seed       = seed

        self.vocab_size = p + 3
        self.op_add     = p
        self.op_sub     = p + 1
        self.eq         = p + 2
        self.seq_len    = 4
        self.pred_pos   = 3   # logits[:, pred_pos, :] vs target

        self._all_triples  = self._generate_all()
        self._train, self._test = self._split()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_all(self) -> List[Tuple[int, int, int, int]]:
        """Return all (a, op_tok, b, c) triples for + and -."""
        p   = self.p
        out = []
        for a in range(p):
            for b in range(p):
                out.append((a, self.op_add, b, (a + b) % p))
                out.append((a, self.op_sub, b, (a - b) % p))
        return out

    def _split(self) -> Tuple[List, List]:
        rng   = np.random.default_rng(self.seed)
        idx   = rng.permutation(len(self._all_triples))
        n_tr  = int(len(idx) * self.train_frac)
        tr    = [self._all_triples[i] for i in idx[:n_tr]]
        te    = [self._all_triples[i] for i in idx[n_tr:]]
        return tr, te

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train(self) -> List:
        return self._train

    @property
    def test(self) -> List:
        return self._test

    def stats(self) -> str:
        return (
            f"p={self.p}  total={len(self._all_triples)}  "
            f"train={len(self._train)}  test={len(self._test)}"
        )

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def make_batch(
        self,
        triples: List[Tuple[int, int, int, int]],
        batch_size: int,
        device: torch.device,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Tensor]:
        """Sample a random batch from triples.

        Returns dict with:
            tokens   : (B, 4) long -- [a, op, b, =]
            targets  : (B,)   long -- c (the answer, 0..p-1)
        """
        if rng is None:
            rng = np.random.default_rng()
        idx  = rng.choice(len(triples), size=batch_size, replace=True)
        rows = [triples[i] for i in idx]

        tok_list = [[a, op, b, self.eq] for a, op, b, _ in rows]
        tgt_list = [c for _, _, _, c in rows]

        tokens  = torch.tensor(tok_list, dtype=torch.long, device=device)
        targets = torch.tensor(tgt_list, dtype=torch.long, device=device)
        return {"tokens": tokens, "targets": targets}

    def make_shifted_batch(
        self,
        batch: Dict[str, Tensor],
        shift: int = 1,
    ) -> Dict[str, Tensor]:
        """Return a batch where 'a' is shifted by +shift mod p.

        For both + and -, shifting a by +shift shifts c by +shift.
        Used for the alignment loss: h(a+1, op, b) ~ T @ h(a, op, b).
        """
        tokens  = batch["tokens"].clone()   # (B, 4)
        targets = batch["targets"].clone()  # (B,)

        tokens[:, 0]  = (tokens[:, 0]  + shift) % self.p
        targets[:]    = (targets        + shift) % self.p

        return {"tokens": tokens, "targets": targets}

    # ------------------------------------------------------------------
    # Full-split evaluation (no sampling)
    # ------------------------------------------------------------------

    def eval_batch_iter(
        self,
        triples: List,
        batch_size: int,
        device: torch.device,
    ):
        """Yield batches covering all triples exactly once (no sampling)."""
        for start in range(0, len(triples), batch_size):
            rows = triples[start : start + batch_size]
            tok_list = [[a, op, b, self.eq] for a, op, b, _ in rows]
            tgt_list = [c for _, _, _, c in rows]
            tokens  = torch.tensor(tok_list, dtype=torch.long, device=device)
            targets = torch.tensor(tgt_list, dtype=torch.long, device=device)
            yield {"tokens": tokens, "targets": targets}

    # ------------------------------------------------------------------
    # Accuracy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def accuracy(logits_at_pred: Tensor, targets: Tensor) -> float:
        """logits_at_pred: (B, vocab), targets: (B,)"""
        preds = logits_at_pred.argmax(dim=-1)
        return (preds == targets).float().mean().item()


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    task = ModArithTask(p=97, train_frac=0.5)
    print(task.stats())

    # Check no triple appears in both splits
    train_set = {(a, op, b) for a, op, b, _ in task.train}
    test_set  = {(a, op, b) for a, op, b, _ in task.test}
    overlap   = train_set & test_set
    print(f"Train/test overlap: {len(overlap)} (should be 0)")

    # Sample a batch
    device = torch.device("cpu")
    rng    = np.random.default_rng(0)
    batch  = task.make_batch(task.train, 8, device, rng)
    print("Sample batch tokens:", batch["tokens"])
    print("Sample batch targets:", batch["targets"])

    shifted = task.make_shifted_batch(batch, shift=1)
    print("Shifted targets:", shifted["targets"])
    print("Expected (targets+1)%97:", (batch["targets"] + 1) % 97)
