"""Phase-2 prototype — relation/edge as a conditionable manifold (docs/phase2/VOLUME_CONCEPTS.md
§10C, case 1): a rule learned as the low-variance constraints of the joint (input, output) data,
applied by conditioning (predict). Deterministic via seeded RNG."""

import numpy as np

from volume.relation import fit_relation


def test_recovers_linear_function_and_predicts():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 300)
    r = fit_relation(x, 2 * x + 1)
    assert r.codim == 1
    assert abs(float(r.predict([0.5])[0]) - 2.0) < 0.02


def test_recovers_planar_relation():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (300, 2))
    y = 0.5 * X[:, 0] - 1.5 * X[:, 1] + 0.3
    r = fit_relation(X, y)
    assert r.codim == 1
    assert abs(float(r.predict([1.0, 1.0])[0]) - (-0.7)) < 0.02


def test_declines_on_independent_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 300)
    y = rng.uniform(-1, 1, 300)              # output carries no signal about input
    r = fit_relation(x, y)
    assert r.codim == 0
    assert r.predict([0.5]) is None         # the rule declines to predict — invents no relation


def test_recovers_two_constraints_and_predicts_both_outputs():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 300)
    yz = np.stack([x, 2 * x], axis=1)        # codimension-2 manifold: y = x, z = 2x
    r = fit_relation(x, yz)
    assert r.codim == 2
    pred = r.predict([0.5])
    assert np.allclose(pred, [0.5, 1.0], atol=0.02)


def test_robust_to_noise():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 300)
    y = 2 * x + 1 + rng.normal(0, 0.05, 300)
    r = fit_relation(x, y)
    assert r.codim == 1
    assert abs(float(r.predict([0.5])[0]) - 2.0) < 0.05
