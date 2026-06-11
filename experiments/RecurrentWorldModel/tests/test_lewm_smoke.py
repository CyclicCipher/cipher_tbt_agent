"""Smoke + correctness tests for the LeWorldModel build (SIGReg first).

Run: ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests/test_lewm_smoke.py -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.lewm import LeWorldModel  # noqa: E402
from baselines.sigreg import MultiSubspaceSIGReg, SIGReg  # noqa: E402
from tasks import DriftField  # noqa: E402


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_sigreg_minimal_on_isotropic_gaussian():
    """SIGReg is small for a true N(0,I) batch and much larger for collapsed / shifted /
    anisotropic batches -- the property that makes it an anti-collapse term."""
    torch.manual_seed(0)
    reg = SIGReg(n_slices=512)
    N, d = 4096, 32
    iso = torch.randn(N, d)                                  # N(0, I): the target
    collapsed = torch.zeros(N, d) + 0.01 * torch.randn(1, d)  # all (near) one point
    shifted = torch.randn(N, d) + 5.0                        # mean far from 0
    aniso = torch.randn(N, d) * 6.0                          # variance far from 1

    s_iso = reg(iso, generator=_g(1)).item()
    s_col = reg(collapsed, generator=_g(1)).item()
    s_shift = reg(shifted, generator=_g(1)).item()
    s_aniso = reg(aniso, generator=_g(1)).item()

    assert s_iso < s_col and s_iso < s_shift and s_iso < s_aniso
    # isotropic should be a small fraction of the collapsed statistic
    assert s_iso < 0.05 * s_col


def test_sigreg_is_differentiable_and_pulls_toward_gaussian():
    """A gradient step on SIGReg should *reduce* it -- i.e. it actually regularises."""
    torch.manual_seed(0)
    reg = SIGReg(n_slices=256)
    z = torch.nn.Parameter(torch.randn(1024, 16) * 4.0 + 3.0)  # anisotropic + shifted
    opt = torch.optim.SGD([z], lr=0.1)
    before = reg(z, generator=_g(2)).item()
    for _ in range(50):
        opt.zero_grad()
        loss = reg(z, generator=_g(2))
        loss.backward()
        opt.step()
    after = reg(z, generator=_g(2)).item()
    assert after < before                                   # the regulariser is doing work


def test_subjepa_regularizer_minimal_on_isotropic_gaussian():
    """MultiSubspaceSIGReg (Sub-JEPA) is small for an isotropic-Gaussian embedding batch and
    much larger for a collapsed one, on the (B, T, D) layout LeWorldModel feeds it."""
    torch.manual_seed(0)
    reg = MultiSubspaceSIGReg(embed_dim=64, num_subspaces=8, num_proj=128)
    B, T, D = 256, 12, 64
    iso = torch.randn(B, T, D)
    collapsed = torch.zeros(B, T, D) + 0.01 * torch.randn(1, 1, D)
    s_iso = reg(iso, generator=_g(1)).item()
    s_col = reg(collapsed, generator=_g(1)).item()
    assert s_iso < 0.1 * s_col and s_iso >= 0.0


def test_lewm_defaults_to_subjepa_and_sigreg_is_selectable():
    task = DriftField(sigma=0.0, seed=0)
    m = LeWorldModel(dim=64, v_bins=task.v_bins, v_min=task.v_min, v_max=task.v_max)
    assert m.reg_kind == "subjepa" and isinstance(m.reg, MultiSubspaceSIGReg)
    m2 = LeWorldModel(dim=64, v_bins=task.v_bins, v_min=task.v_min, v_max=task.v_max, reg="sigreg")
    assert m2.reg_kind == "sigreg" and isinstance(m2.reg, SIGReg)


def test_lewm_losses_and_grad_step():
    """LeWorldModel produces the three loss terms and a training step runs end-to-end."""
    task = DriftField(sigma=0.0, seed=0)
    b = task.sample(8, generator=_g(0))
    m = LeWorldModel(dim=32, heads=4, depth=2, v_bins=task.v_bins,
                     v_min=task.v_min, v_max=task.v_max, num_proj=64, num_subspaces=8)
    out = m.losses(b.obs_v, b.obs_t)
    for k in ("total", "mse", "sigreg", "decode"):
        assert k in out and torch.isfinite(out[k])
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    opt.zero_grad()
    out["total"].backward()
    opt.step()                                              # no NaNs, gradients flow


def test_lewm_rollout_shapes_and_extrapolates_past_observations():
    """Rollout produces a what-field and the matching when-field over the FULL query grid,
    including OOD time bins beyond the observation window (reached by composed steps)."""
    task = DriftField(sigma=0.0, t_obs=10.0, t_max=20.0, seed=1)
    b = task.sample(4, generator=_g(1))
    m = LeWorldModel(dim=32, heads=4, depth=2, v_bins=task.v_bins,
                     v_min=task.v_min, v_max=task.v_max, num_proj=64, num_subspaces=8)
    centers = task.t_centers.to(torch.float32)
    assert centers.max() > task.t_obs                       # the grid really does extend OOD
    what, when = m.what_when(b.obs_v, b.obs_t, centers, task.t_obs)
    assert what.shape == (4, task.t_bins, task.v_bins)
    assert when.shape == (4, task.v_bins, task.t_bins)
    assert torch.allclose(what.exp().sum(-1), torch.ones(4, task.t_bins), atol=1e-4)
    assert torch.isfinite(what).all() and torch.isfinite(when).all()


def test_run_lewm_smoke_both_regimes():
    from train_lewm import LeWMConfig, run_lewm
    for sigma in (0.0, 2.0):
        out = run_lewm(LeWMConfig(sigma=sigma, smoke=True))
        assert out["regime"] == ("stochastic" if sigma > 0 else "deterministic")
        f = out["final"]
        assert "what" in f and "when" in f
        for term in ("mse", "sigreg", "decode"):
            assert term in f
