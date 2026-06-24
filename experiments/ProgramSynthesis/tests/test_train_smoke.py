"""Integration smoke test for the Phase-1 BC pipeline (CPU, a few steps).

Does NOT do real training (Mistake #36) — it just exercises dataset -> model ->
train loop -> metrics end to end so integration bugs surface here, not on the GPU.
"""

from __future__ import annotations

import os
import sys

import torch

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from agent.dataset import build_dataset  # noqa: E402
from agent.train_bc import make_layout_splits, train_one  # noqa: E402


def _tiny_splits():
    return make_layout_splits(["key_door"], "key_door", n_train=5, n_test=4, seed=0)


def test_dataset_shapes_and_labels():
    train_layouts, _ = _tiny_splits()
    ds = build_dataset(train_layouts, window=2)
    assert len(ds) > 0
    fw, aw, tg = ds[0]
    assert fw.shape == (2, 64, 64) and fw.dtype == torch.long
    assert aw.shape == (1,)
    assert int(tg) in (1, 2, 3, 4)                 # a directional action id
    # First decision of an episode has no prior action (padded with -1).
    assert any(int(ds[i][1][0]) == -1 for i in range(len(ds)))
    # All inter-frame actions are -1 or a valid direction.
    assert all(int(ds[i][1][0]) in (-1, 1, 2, 3, 4) for i in range(len(ds)))


def test_train_loop_runs_and_logs():
    train_layouts, test_layouts = _tiny_splits()
    train_ds = build_dataset(train_layouts, window=2)
    test_ds = build_dataset(test_layouts, window=2)
    for binding in ("none", "pope2d1"):
        history = train_one(
            binding, train_ds, test_ds,
            steps=3, batch=8, lr=1e-3, device="cpu",
            eval_every=1, window=2, seed=0,
        )
        assert len(history) >= 2
        for h in history:
            assert 0.0 <= h["test_masked"] <= 1.0
            assert 0.0 <= h["train_masked"] <= 1.0
            assert h["train_loss"] == h["train_loss"]  # not NaN
