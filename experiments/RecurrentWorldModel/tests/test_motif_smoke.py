"""Smoke + correctness tests for the TBAF drift test (MotifEcho + injected activation).

Run: ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests/test_motif_smoke.py -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import (CommonMode, FixedDepthConfig, FixedDepthTransformer,  # noqa: E402
                       TBAFPerToken, TBAFVerbatim)
from tasks import MotifEcho  # noqa: E402
from train_motif import ARMS, MotifConfig, run_motif  # noqa: E402


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_motif_is_periodic_and_fair():
    task = MotifEcho(vocab_size=16, motif_min=2, motif_max=6, context_len=18, horizon=60, seed=0)
    b = task.sample(64, generator=_g(0))
    assert b.tokens.shape == (64, task.seq_len)
    # every row is exactly periodic with its period m -> a perfect model could roll out forever
    for i in range(64):
        m = int(b.m[i])
        seq = b.tokens[i]
        assert torch.equal(seq[:-m], seq[m:])               # token[t] == token[t+m] everywhere


def test_tbaf_pertoken_is_common_mode_invariant():
    """|a-b| is unchanged by a common shift of the triple -> common-mode rejection."""
    act = TBAFPerToken()
    x = torch.randn(4, 5, 12)
    shift = torch.randn(4, 5, 1)                              # same shift added to all 3 of each triple
    assert torch.allclose(act(x), act(x + shift), atol=1e-5)
    assert (act(x) >= 0).all()                               # distances are non-negative
    assert act(x).shape == x.shape                           # dim-preserving


def test_tbaf_verbatim_collapses_positions():
    """The repo op (control) broadcasts one batch-level vector to every position -- the bug we
    flagged: all tokens identical, and it depends on the rest of the batch."""
    act = TBAFVerbatim()
    x = torch.randn(4, 5, 12)
    out = act(x)
    assert out.shape == x.shape
    assert torch.allclose(out[0, 0], out[2, 3], atol=1e-6)   # every (b,t) position identical
    # and it is batch-dependent: perturbing ONE channel of another position (breaking that
    # triple's common mode) shifts this position's output too
    x2 = x.clone(); x2[1, 1, 0] += 5.0
    assert not torch.allclose(act(x)[0, 0], act(x2)[0, 0], atol=1e-4)


def test_commonmode_removes_mean():
    act = CommonMode()
    x = torch.randn(3, 4, 9)
    assert torch.allclose(act(x).mean(-1), torch.zeros(3, 4), atol=1e-5)


def test_injected_transformer_runs_for_each_arm():
    task = MotifEcho(seed=1)
    b = task.sample(4, generator=_g(1))
    for act in ("none", "gelu", "tbaf", "tbaf_verbatim", "commonmode"):
        m = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.V, dim=24, n_layers=2,
                                                   max_seq=task.seq_len, pos_mode="pope",
                                                   inject_act=act))
        out = m(b.tokens[:, :10])
        assert out.shape == (4, 10, task.V) and torch.isfinite(out).all()
        if act == "none":
            assert m.inject is None
        else:
            assert m.inject is not None


def test_injection_default_off_leaves_trunk_unchanged():
    """A field-style model (default inject_act='none') must be byte-for-byte unaffected."""
    cfg = FixedDepthConfig(vocab_size=12, dim=24, n_layers=2, max_seq=16, pos_mode="pope")
    assert cfg.inject_act == "none"
    assert FixedDepthTransformer(cfg).inject is None


def test_run_motif_smoke():
    out = run_motif(MotifConfig(smoke=True))
    for arm in ARMS:
        f = out["arms"][arm]["final"]
        assert "decay" in f and len(f["decay"]) == out["horizon"]
        assert "acc_step1" in f and "acc_mean" in f
