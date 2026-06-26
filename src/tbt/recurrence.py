"""The selective-gated recurrence — ONE canonical implementation (architecture doc §14 stage 11, R8).

Both the language SSM (`precursor/language_recurrent.py`) and the column's L6 path-integration (`column.py`)
are the SAME Mamba/SSD update:

    h_t  =  gate ⊙ A(h_{t-1})  +  (1 − gate) ⊙ drive_t          (per-channel selective gate)

differing only in the transition A and the drive: A = a decay/identity over CONTENT for language; A = the L5
displacement operator over LOCATION for the column. So the recurrence STEP and the per-channel LEARNED GATE
live HERE, once, and every experiment USES this — never a per-test reimplementation that quietly forgets the
gate (see memory `feedback_reuse_canonical_components`: the original mistake was rebuilding a scalar-gate copy
on L6 instead of using this). The caller applies A(h_{t-1}) in its own framework and passes the result + the
drive; this module owns σ(G) and its 1-step-truncated learning.

Pure numpy: the perf-sensitive language loop stays numpy-native; the torch column converts at its rare
correction step. Both hold their OWN instance (own gate weights) of this one class — like two attention layers
sharing the Attention class with different weights."""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SelectiveRecurrence:
    def __init__(self, d, n_keys=1, init=0.5):
        self.d, self.n_keys = d, n_keys
        self.G = np.full((n_keys, d), float(init))           # per-key, per-channel gate logits; σ(G) ∈ (0,1)^d

    def gate(self, key):
        """The per-channel selective gate for input `key` (an int, or a (B,) array of keys → (B, d)). Compute
        this ONCE per step and pass it to `step`/`learn_gate` — don't let them recompute the sigmoid."""
        return sigmoid(self.G[key])

    def step(self, transitioned_h, drive, gate):
        """h ← gate ⊙ A(h_prev) + (1 − gate) ⊙ drive.  `gate` = the value from .gate(key) (or a scalar
        override); `transitioned_h` = A(h_prev), applied by the caller."""
        return gate * transitioned_h + (1.0 - gate) * drive

    def learn_gate(self, key, gate, err_h, transitioned_h, drive, lr):
        """1-step-truncated gate gradient: move G to reduce  err_h · ∂h/∂gate = err_h ⊙ (A(h_prev) − drive),
        reusing the already-computed `gate`. Scatter-add so repeated keys in a batch accumulate."""
        dz = (err_h * (transitioned_h - drive)) * gate * (1.0 - gate)
        np.add.at(self.G, key, -lr * dz)

    def clip(self, lo=-6.0, hi=6.0):
        np.clip(self.G, lo, hi, out=self.G)
