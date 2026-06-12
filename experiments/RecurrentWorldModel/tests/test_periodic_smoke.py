"""Smoke + correctness tests for the latent-rollout TBAF test (PeriodicField + LeWM inject).

Run: ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests/test_periodic_smoke.py -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import LeWorldModel  # noqa: E402
from baselines.bottleneck import ActivationFFN  # noqa: E402
from tasks import PeriodicField  # noqa: E402
from train_lewm_periodic import ARMS, PeriodicLeWMConfig, run  # noqa: E402


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_periodic_is_bounded_and_fair():
    task = PeriodicField(seed=0)
    b = task.sample(64, generator=_g(0))
    # the wave stays inside the value range (bounded -> no information trap)
    assert b.obs_v.min() >= task.v_min - 1e-4 and b.obs_v.max() <= task.v_max + 1e-4
    # the observation window covers >= one full period (so the future is determined)
    assert task.t_obs >= task.period_range[1]
    assert b.field_target.shape == (64, task.t_bins, task.v_bins)
    assert task.time_mask.sum() > 0 and (1 - task.time_mask).sum() > 0   # in + OOD bins exist


def test_periodic_continuation_is_determined_by_one_period():
    """Value at t and t+P are equal -> once a period is seen the whole future is fixed."""
    task = PeriodicField(seed=1)
    b = task.sample(8, generator=_g(1))
    P, phase = b.period, b.phase
    t = torch.tensor([2.0, 5.0, 9.0])
    v_t = task._wave(t[None, :], P[:, None], phase[:, None])
    v_tP = task._wave((t[None, :] + P[:, None]), P[:, None], phase[:, None])
    assert torch.allclose(v_t, v_tP, atol=1e-4)


def test_lewm_predictor_injection_wires_in():
    task = PeriodicField(seed=2)
    base = LeWorldModel(dim=24, depth=2, v_bins=task.v_bins, reg="sigreg", inject_act="none")
    tb = LeWorldModel(dim=24, depth=2, v_bins=task.v_bins, reg="sigreg", inject_act="tbaf")
    assert base.predictor.inject is None
    assert isinstance(tb.predictor.inject, ActivationFFN)
    # both still run a rollout end to end
    b = task.sample(4, generator=_g(2))
    centers = task.t_centers.to(torch.float32)
    for m in (base, tb):
        what = m.rollout_what(b.obs_v, b.obs_t, centers, task.t_obs)
        assert what.shape == (4, task.t_bins, task.v_bins) and torch.isfinite(what).all()


def test_run_periodic_smoke():
    out = run(PeriodicLeWMConfig(smoke=True))
    for arm in ARMS:
        f = out["arms"][arm]["final"]
        assert "per_bin" in f and len(f["per_bin"]) == len(out["t_centers"])
        assert "acc_in" in f and "acc_ood" in f
