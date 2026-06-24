"""Learning agent for the ARC-AGI-3 replica.

See LEARNING_AGENT.md for the design. Built incrementally:
  layouts.py   procedural LockPath distributions + train/held-out split  (done)
  encoders.py  swappable binding channels (none / 2D PoPE / 2D+1 PoPE / ...) (next)
  trunk.py     fixed transformer trunk + policy/value heads
  dataset.py   oracle-trajectory + DAgger datasets
  train_bc.py  Phase 1 behavior cloning
"""

from .layouts import (
    GENERATORS,
    Layout,
    make_game,
    sample_layouts,
    train_test_split,
)

__all__ = [
    "GENERATORS",
    "Layout",
    "make_game",
    "sample_layouts",
    "train_test_split",
]
