"""Smoke + correctness tests for Muon / Aurora.

Run:  ./venv/Scripts/python.exe -m pytest experiments/RecurrentWorldModel/tests -q
"""

from __future__ import annotations

import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import FixedDepthConfig, FixedDepthTransformer  # noqa: E402
from optim import MuonAuroraAdamW, aurora_transform, newton_schulz, split_matrix_params  # noqa: E402
from tasks import ModularChain  # noqa: E402
from train_transformer import TConfig, run_transformer  # noqa: E402


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    M = torch.randn(64, 24)
    U = newton_schulz(M, steps=5)
    sv = torch.linalg.svdvals(U)
    # NS5 is a coarse polar approx -> singular values clustered near 1
    assert sv.mean() > 0.7 and sv.std() < 0.25


def test_aurora_revives_dead_rows_and_uniformizes():
    """Aurora's core claim: revive starved rows + uniform row norms, vs Muon."""
    torch.manual_seed(0)
    M = torch.randn(96, 32)
    M[:10] *= 0.02                       # 10 starved (near-dead) rows
    muon = newton_schulz(M, steps=5)
    aur = aurora_transform(M, K=2, beta=0.5, ns_steps=5)
    cv = lambda U: (U.norm(dim=1).std() / U.norm(dim=1).mean()).item()
    # Aurora's row norms are far more uniform than Muon's
    assert cv(aur) < 0.5 * cv(muon)
    # and the starved rows are revived (no longer near-zero)
    muon_dead = muon.norm(dim=1)[:10].mean()
    aur_dead = aur.norm(dim=1)[:10].mean()
    assert aur_dead > 5 * muon_dead


def test_param_routing():
    task = ModularChain(seed=0)
    m = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.vocab_size, dim=32, n_layers=2,
                                               max_seq=task.seq_len, pos_mode="pope"))
    matrix, other = split_matrix_params(m)
    # all matrix-group params are 2D; embeddings are NOT in it (compare by identity)
    assert all(p.ndim == 2 for p in matrix)
    assert not any(p is m.embed.weight for p in matrix)
    assert any(p is m.embed.weight for p in other)
    assert len(matrix) > 0 and len(other) > 0


def test_muon_aurora_optimizers_train():
    task = ModularChain(seed=0)
    rng = random.Random(0)
    for variant in ("muon", "aurora"):
        torch.manual_seed(0)
        m = FixedDepthTransformer(FixedDepthConfig(vocab_size=task.vocab_size, dim=32, n_layers=2,
                                                   max_seq=task.seq_len, pos_mode="pope"))
        matrix, other = split_matrix_params(m)
        opt = MuonAuroraAdamW([
            dict(params=matrix, use_muon=True, variant=variant, lr=0.02, momentum=0.9),
            dict(params=other, use_muon=False, lr=1e-3),
        ])
        L0 = L1 = None
        for step in range(1, 21):
            b = task.sample(64, 1, 4, rng)
            v = m(b.input_ids).shape[-1]
            ce = F.cross_entropy(m(b.input_ids).reshape(-1, v), b.targets.reshape(-1), reduction="none")
            loss = (ce.reshape(b.loss_mask.shape) * b.loss_mask).sum() / b.loss_mask.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
            if step == 1:
                L0 = loss.item()
            L1 = loss.item()
        assert L1 < L0, f"{variant} did not reduce loss ({L0:.3f}->{L1:.3f})"


def test_run_transformer_smoke_muon_aurora():
    out = run_transformer(TConfig(smoke=True, arms=("adamw", "muon", "aurora")))
    for arm in ("adamw", "muon", "aurora"):
        assert arm in out["arms"] and "acc_ood" in out["arms"][arm]["final"]
