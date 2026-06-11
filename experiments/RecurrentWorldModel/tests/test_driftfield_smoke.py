"""Smoke + correctness tests for Stage 3 (DriftField generator).

The closed-form targets are the confound-critical piece, so these check them hard:
deterministic targets land in the exact analytic bin, stochastic targets are valid
distributions, and the Inverse-Gaussian first-passage target is cross-checked against
a Monte-Carlo simulation of drifted Brownian motion.

Run:  ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests/test_driftfield_smoke.py -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import (FixedDepthConfig, FixedDepthTransformer,  # noqa: E402
                       FunctionalFieldModel, SeparateHeadsModel, UnifiedFieldModel)
from tasks import DriftField  # noqa: E402
from train_field import ALL_ARMS, FieldConfig, run_field  # noqa: E402


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_shapes_and_valid_distributions():
    task = DriftField(sigma=2.0, seed=0)
    b = task.sample(64, generator=_gen())
    assert b.obs_v.shape == (64, task.n_obs) and b.obs_t.shape == (64, task.n_obs)
    assert b.field_target.shape == (64, task.t_bins, task.v_bins)
    assert b.when_target.shape == (64, task.v_bins, task.t_bins)
    # every row of each stack is a valid distribution
    assert torch.allclose(b.field_target.sum(-1), torch.ones(64, task.t_bins), atol=1e-4)
    assert torch.allclose(b.when_target.sum(-1), torch.ones(64, task.v_bins), atol=1e-4)
    assert torch.isfinite(b.field_target).all() and torch.isfinite(b.when_target).all()
    assert (b.field_target >= 0).all() and (b.when_target >= 0).all()
    # validity mask is per (sample, threshold), and both reached + trivial thresholds occur
    assert b.when_valid.shape == (64, task.v_bins)
    assert b.when_valid.max() == 1.0 and b.when_valid.min() == 0.0


def test_observations_increase_in_time():
    task = DriftField(sigma=0.0, seed=1)
    b = task.sample(32, generator=_gen(1))
    assert torch.all(torch.diff(b.obs_t, dim=1) >= 0)            # sorted times
    assert b.obs_t.max() <= task.t_obs + 1e-5                    # within observation window


def test_deterministic_field_is_exact_ridge():
    """sigma=0: field is one-hot at the bin containing v0 + mu * t_center."""
    task = DriftField(sigma=0.0, seed=2)
    b = task.sample(48, generator=_gen(2))
    # one-hot rows
    assert torch.all(b.field_target.max(-1).values == 1.0)
    tc = task.t_centers.to(torch.float32)
    mean = b.v0[:, None] + b.mu[:, None] * tc[None, :]           # (B,Tb)
    expect = torch.bucketize(mean.double(), task.v_edges).clamp(1, task.v_bins) - 1
    got = b.field_target.argmax(-1)
    assert torch.equal(got, expect)


def test_deterministic_when_is_exact_passage():
    """sigma=0: when is one-hot at the bin containing (theta - v0)/mu (first passage)."""
    task = DriftField(sigma=0.0, seed=3)
    b = task.sample(48, generator=_gen(3))
    vc = task.v_centers.to(torch.float32)
    a = vc[None, :] - b.v0[:, None]                              # (B,Vb)
    got = b.when_target.argmax(-1)
    # thresholds at/below the start collapse to time-bin 0
    assert torch.all(got[a <= 0] == 0)
    reachable = (a > 0) & ((a / b.mu[:, None]) <= task.t_max)
    t_pass = (a / b.mu[:, None]).double()
    expect = (torch.bucketize(t_pass, task.t_edges).clamp(1, task.t_bins) - 1)
    assert torch.equal(got[reachable], expect[reachable])


def test_stochastic_field_mean_matches_drift():
    """Stochastic field column mean (over value centres) tracks v0 + mu*t."""
    task = DriftField(sigma=2.0, seed=4)
    b = task.sample(256, generator=_gen(4))
    vc = task.v_centers.to(torch.float32)
    pred_mean = (b.field_target * vc[None, None, :]).sum(-1)     # (B,Tb)
    tc = task.t_centers.to(torch.float32)
    true_mean = b.v0[:, None] + b.mu[:, None] * tc[None, :]
    # interior time bins (avoid edge censoring); within one value-bin width
    bin_w = (task.v_max - task.v_min) / task.v_bins
    assert (pred_mean - true_mean).abs().mean() < bin_w


def test_inverse_gaussian_target_matches_monte_carlo():
    """The closed-form first-passage target == simulated drifted-Brownian first passage."""
    task = DriftField(sigma=2.0, t_max=20.0, t_bins=20, seed=5)
    b = task.sample(1, generator=_gen(5))
    v0, mu = float(b.v0[0]), float(b.mu[0])
    vc = task.v_centers
    # pick a threshold comfortably above the start so a barrier exists
    ti = int((task.v_bins * 3) // 4)
    theta = float(vc[ti])
    a = theta - v0
    assert a > 0
    # Monte-Carlo first passage of v(t) = v0 + mu t + sigma W(t) on a fine grid
    torch.manual_seed(0)
    n_paths, n_steps = 40000, 4000
    dt = task.t_max / n_steps
    steps = mu * dt + task.sigma * torch.randn(n_paths, n_steps) * (dt ** 0.5)
    path = v0 + torch.cumsum(steps, dim=1)                       # (n_paths, n_steps)
    crossed = path >= theta
    ever = crossed.any(dim=1)
    first_idx = torch.where(ever, crossed.float().argmax(dim=1), torch.full((n_paths,), n_steps - 1))
    first_t = (first_idx.float() + 1) * dt
    first_t = torch.where(ever, first_t, torch.full((n_paths,), task.t_max * 1.5))  # censored
    mc_hist = torch.histc(first_t.clamp(max=task.t_max - 1e-6), bins=task.t_bins, min=0.0, max=task.t_max)
    mc = mc_hist / mc_hist.sum()
    analytic = b.when_target[0, ti]                              # (Tb,)
    # total-variation distance between analytic and MC histograms should be small
    tv = 0.5 * (analytic - mc).abs().sum()
    assert tv < 0.06, f"IG target disagrees with MC (TV={tv:.3f})"


def test_decorrelation_both_latents_vary():
    """v0 and mu both vary across the batch -> neither query is solvable from one axis."""
    task = DriftField(sigma=1.0, seed=6)
    b = task.sample(128, generator=_gen(6))
    assert b.v0.std() > 1.0 and b.mu.std() > 0.1


def test_ood_horizon_split():
    task = DriftField(sigma=1.0, t_obs=10.0, t_max=20.0, seed=7)
    b = task.sample(8, generator=_gen(7))
    assert b.time_mask.sum() > 0 and (1 - b.time_mask).sum() > 0     # both in-dist and OOD bins
    # in-dist time bins are exactly those within the observation window
    assert torch.all((task.t_centers <= task.t_obs).float() == b.time_mask)


def test_trunk_consumes_path():
    """The continuous-input PoPE trunk runs on (obs_v, coord=obs_t)."""
    task = DriftField(sigma=1.0, seed=8)
    b = task.sample(4, generator=_gen(8))
    m = FixedDepthTransformer(FixedDepthConfig(
        vocab_size=task.v_bins, dim=32, n_layers=2, max_seq=task.n_obs,
        pos_mode="pope", continuous_input=True))
    out = m(b.obs_v, coord=b.obs_t)
    assert out.shape == (4, task.n_obs, task.v_bins)
    assert torch.isfinite(out).all()


def test_field_readouts_have_matching_interface():
    """Unified, separate, and functional readouts emit same-shaped what/when -> comparable."""
    task = DriftField(sigma=1.0, seed=10)
    b = task.sample(4, generator=_gen(10))
    tcn = (task.t_centers / task.t_max).float()
    vcn = (task.v_centers / task.v_max).float()
    uni = UnifiedFieldModel(32, 4, 2, task.n_obs, task.t_bins, task.v_bins)
    sep = SeparateHeadsModel(32, 4, 2, task.n_obs, task.t_bins, task.v_bins, tcn, vcn)
    fun = FunctionalFieldModel(32, 4, 2, task.n_obs, task.t_bins, task.v_bins, tcn, vcn)
    for m in (uni, sep, fun):
        what, when = m.what_logp(b.obs_v, b.obs_t), m.when_logp(b.obs_v, b.obs_t)
        assert what.shape == (4, task.t_bins, task.v_bins)
        assert when.shape == (4, task.v_bins, task.t_bins)
        # rows are valid log-distributions
        assert torch.allclose(what.exp().sum(-1), torch.ones(4, task.t_bins), atol=1e-4)
        assert torch.allclose(when.exp().sum(-1), torch.ones(4, task.v_bins), atol=1e-4)


def test_functional_readout_is_coordinate_evaluated():
    """The functional field has no fixed grid head -- it evaluates g(z, coord); and its
    `when` read uses no when-specific parameters (so C1 still applies)."""
    task = DriftField(sigma=0.0, seed=12)
    tcn = (task.t_centers / task.t_max).float()
    vcn = (task.v_centers / task.v_max).float()
    fun = FunctionalFieldModel(32, 4, 2, task.n_obs, task.t_bins, task.v_bins, tcn, vcn)
    names = [n for n, _ in fun.named_parameters()]
    assert not any("when" in n for n in names)               # no when head
    assert any(n.startswith("g.") for n in names)            # the shared evaluator g(z, coord)


def test_unified_when_is_parameter_free():
    """The unified `when` read uses no when-specific parameters -- only the field."""
    task = DriftField(sigma=0.0, seed=11)
    uni = UnifiedFieldModel(32, 4, 2, task.n_obs, task.t_bins, task.v_bins)
    names = [n for n, _ in uni.named_parameters()]
    assert not any("when" in n for n in names)               # no when head at all
    assert any("field" in n for n in names)


def test_run_field_smoke_both_regimes():
    for sigma in (0.0, 2.0):
        out = run_field(FieldConfig(sigma=sigma, arms=ALL_ARMS, smoke=True))
        assert out["regime"] == ("stochastic" if sigma > 0 else "deterministic")
        for arm in ALL_ARMS:                                 # cover all 6 training paths
            f = out["arms"][arm]["final"]
            assert "what" in f and "when" in f
            assert "field_entropy" in f


def test_entropy_floor_is_sane():
    task = DriftField(sigma=2.0, seed=9)
    b = task.sample(16, generator=_gen(9))
    h_field = DriftField.entropy(b.field_target)
    assert (h_field >= 0).all()                                 # entropy non-negative
    # a deterministic field has ~zero entropy (the floor); stochastic is strictly positive
    det = DriftField(sigma=0.0, seed=9).sample(16, generator=_gen(9))
    assert DriftField.entropy(det.field_target).max() < 1e-4
    assert h_field.mean() > 0.1
