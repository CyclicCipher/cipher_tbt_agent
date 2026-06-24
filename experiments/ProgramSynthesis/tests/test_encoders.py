"""Tests for the binding-channel encoders and the trunk (CPU forward only)."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from agent.encoders import BINDINGS, make_scheme  # noqa: E402
from agent.trunk import build_model  # noqa: E402


def _batch(b=2, w=2, grid=64):
    frames = torch.randint(0, 16, (b, w, grid, grid))
    actions = torch.randint(1, 5, (b, w - 1))
    return frames, actions


@pytest.mark.parametrize("binding", BINDINGS)
def test_forward_shapes_and_finite(binding):
    model = build_model(binding)
    frames, actions = _batch()
    logits, value = model(frames, actions)
    assert logits.shape == (2, 8)
    assert value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_arms_differ_only_in_channel():
    # none / pope2d / pope2d1 add ZERO parameters over each other — the binding is
    # the only difference. content adds learned position embeddings, so it has more.
    counts = {b: sum(p.numel() for p in build_model(b).parameters()) for b in BINDINGS}
    assert counts["none"] == counts["pope2d"] == counts["pope2d1"]
    assert counts["content"] > counts["none"]


def test_rotary_binding_distinction():
    sch2d = make_scheme("pope2d", d=96, head_dim=24, max_xy=16, max_t=2)
    sch2d1 = make_scheme("pope2d1", d=96, head_dim=24, max_xy=16, max_t=2)
    same_xy_diff_t = (
        torch.tensor([[[3, 4, 0]]]),
        torch.tensor([[[3, 4, 1]]]),
    )
    diff_x = torch.tensor([[[5, 4, 0]]])
    # 2D rotary ignores time; 2D+1 does not.
    assert torch.allclose(sch2d.angles(same_xy_diff_t[0]), sch2d.angles(same_xy_diff_t[1]))
    assert not torch.allclose(sch2d1.angles(same_xy_diff_t[0]), sch2d1.angles(same_xy_diff_t[1]))
    # Both encode space.
    assert not torch.allclose(sch2d.angles(same_xy_diff_t[0]), sch2d.angles(diff_x))
    assert not torch.allclose(sch2d1.angles(same_xy_diff_t[0]), sch2d1.angles(diff_x))


def test_content_scheme_changes_tokens_none_does_not():
    coords = torch.tensor([[[1, 2, 0], [3, 4, 1], [0, 0, 0]]])
    tok = torch.randn(1, 3, 96)
    content = make_scheme("content", d=96, head_dim=24, max_xy=16, max_t=2)
    none = make_scheme("none", d=96, head_dim=24, max_xy=16, max_t=2)
    assert not torch.allclose(content.add_pos(tok, coords), tok)
    assert torch.allclose(none.add_pos(tok, coords), tok)


def test_gradients_flow():
    model = build_model("pope2d1")
    frames, actions = _batch()
    logits, value = model(frames, actions)
    (logits.sum() + value.sum()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
