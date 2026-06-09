"""Smoke tests for the Stage 0 settling core. No training -- shapes, convergence
plumbing, gradient flow, and the Risk-1 diagnostics only.

Run:  ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests -q
"""

from __future__ import annotations

import os
import sys

import torch

# allow `import core` / `import probes` when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DEQConfig, DEQFixedPoint, SettlingBlock, SettlingBlockConfig  # noqa: E402
from core.halting import ChainHalt, converged  # noqa: E402
from probes import ConvergenceMonitor, basin_consistency  # noqa: E402


def _make(dim=64, heads=4, **deq_kw):
    block = SettlingBlock(SettlingBlockConfig(dim=dim, n_heads=heads))
    deq = DEQFixedPoint(block, DEQConfig(max_iter=40, tol=1e-3, **deq_kw))
    return deq


def test_forward_shapes_and_info():
    torch.manual_seed(0)
    deq = _make()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        h_star, info = deq(x)
    assert h_star.shape == x.shape
    assert 1 <= info.iters <= 40
    assert isinstance(info.converged, bool)
    assert info.final_rel_residual >= 0.0


def test_one_step_gradient_flows():
    torch.manual_seed(0)
    deq = _make(grad_mode="one_step")
    x = torch.randn(2, 8, 64, requires_grad=True)
    h_star, _ = deq(x)
    loss = h_star.pow(2).mean()
    loss.backward()
    # gradient reaches both the block params and the input injection
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in deq.block.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_anderson_matches_picard_fixed_point():
    torch.manual_seed(1)
    block = SettlingBlock(SettlingBlockConfig(dim=64, n_heads=4))
    x = torch.randn(2, 8, 64)
    picard, ip = DEQFixedPoint(block, DEQConfig(anderson=False, max_iter=80, tol=1e-4))(x)
    ander, ia = DEQFixedPoint(block, DEQConfig(anderson=True, max_iter=80, tol=1e-4))(x)
    if ip.converged and ia.converged:
        rel = (picard - ander).flatten(1).norm(dim=1) / picard.flatten(1).norm(dim=1)
        assert rel.mean().item() < 1e-2


def test_convergence_monitor_summary():
    torch.manual_seed(0)
    deq = _make()
    mon = ConvergenceMonitor()
    diffs = []
    with torch.no_grad():
        for t in (4, 8, 16, 32):  # "difficulty" proxy = sequence length
            _, info = deq(torch.randn(1, t, 64))
            mon.record(info)
            diffs.append(float(t))
    s = mon.summary()
    assert s["n_solves"] == 4
    assert 0.0 <= s["convergence_rate"] <= 1.0
    # correlation is just plumbing here, must return a real number or nan
    r = mon.difficulty_correlation(diffs)
    assert isinstance(r, float)


def test_basin_consistency_runs():
    torch.manual_seed(0)
    deq = _make()
    x = torch.randn(1, 8, 64)
    out = basin_consistency(deq, x, n_restarts=3)
    assert out["n_restarts"] == 3
    assert out["mean_pairwise_rel_dist"] >= 0.0


def test_halting_predicates():
    assert converged(1e-4, tol=1e-3) is True
    assert converged(1e-2, tol=1e-3) is False
    ch = ChainHalt(tol=1e-3, max_steps=3)
    c = torch.ones(1, 4)
    assert ch.should_halt(c) is False          # first call, primes prev
    assert ch.should_halt(c.clone()) is True    # unchanged -> halt
