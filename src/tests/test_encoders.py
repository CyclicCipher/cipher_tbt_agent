"""The SDR encoder/decoder library (tbt.encoders) — the three SDR rules (overlap=similarity, determinism, sparsity),
decode round-trips (each encoder is bidirectional), and the MOTOR-decode design (a motor SDR 'thought' interpreted by
the same encoder that would sense it — the anti-parallel-system property)."""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.encoders import (SDR, ScalarEncoder, CategoryEncoder, GridEncoder,   # noqa: E402
                          MultiEncoder, ConjunctiveEncoder, SpatialPooler)


# ── SDR type ────────────────────────────────────────────────────────────────────────────────────────────────
def test_sdr_overlap_and_concat():
    a, b = SDR(10, {1, 2, 3}), SDR(10, {2, 3, 4})
    assert a.overlap(b) == 2 and a.w == 3
    c = SDR.concat([a, b])                                    # fields laid side by side
    assert c.n == 20 and c.active == {1, 2, 3, 12, 13, 14}


# ── ScalarEncoder ───────────────────────────────────────────────────────────────────────────────────────────
def test_scalar_rules_and_roundtrip():
    e = ScalarEncoder(0.0, 100.0, n=100, w=21)
    assert e.encode(50).w == 21                              # SPARSITY: fixed w
    assert e.encode(50) == e.encode(50)                      # DETERMINISM
    assert abs(e.decode(e.encode(50)) - 50) < 2.0            # round-trip within resolution
    assert abs(e.decode(e.encode(12.5)) - 12.5) < 2.0
    near = e.encode(50).overlap(e.encode(52))                # OVERLAP=similarity: near > far
    far = e.encode(50).overlap(e.encode(90))
    assert near > far and far == 0


def test_scalar_periodic_wraps():
    e = ScalarEncoder(0.0, 360.0, n=90, w=11, periodic=True)
    assert e.encode(0).overlap(e.encode(359)) > 0            # 0 and max are NEAR (wraparound overlap)
    assert e.encode(0).overlap(e.encode(180)) == 0           # opposite phases disjoint
    d = e.decode(e.encode(350))
    assert min(abs(d - 350), 360 - abs(d - 350)) < 8         # circular round-trip


# ── CategoryEncoder ─────────────────────────────────────────────────────────────────────────────────────────
def test_category_disjoint_and_roundtrip():
    e = CategoryEncoder(categories=["red", "green", "blue"], w=8, capacity=16)
    assert e.encode("red").overlap(e.encode("blue")) == 0    # categories are disjoint (no false nearness)
    assert e.decode(e.encode("green")) == "green"            # round-trip
    e.add("cyan")                                            # grows online within capacity, fixed length
    assert e.encode("red").n == e.encode("cyan").n and e.decode(e.encode("cyan")) == "cyan"


# ── GridEncoder (the LOCATION encoder — the point of the SDR pivot) ──────────────────────────────────────────
def test_grid_graded_metric_overlap():
    """The property the localist state_node LACKS: overlap GRADED by distance, a-priori (before any learning)."""
    g = GridEncoder(scales=(7, 11, 13, 17), dims=2, mw=3, bounds=[(0, 63), (0, 63)])
    o = g.encode((30, 30))
    assert o.overlap(g.encode((31, 30))) > o.overlap(g.encode((34, 30))) > o.overlap(g.encode((30, 30))) - o.w
    assert o.overlap(g.encode((31, 30))) > o.overlap(g.encode((2, 30)))   # near shares more than far


def test_grid_roundtrip_and_capacity():
    g = GridEncoder(scales=(7, 11, 13, 17), dims=2, mw=3, bounds=[(0, 63), (0, 63)])
    for p in [(0, 0), (12, 12), (30, 47), (63, 63)]:
        assert g.decode(g.encode(p)) == p                    # exact round-trip over the board
    assert g.encode((10, 10)) != g.encode((40, 40))          # distinct far positions are distinct codes


def test_grid_is_path_integrable():
    """A translation shifts each phase additively (mod scale) -> encode(p+d) == the phase-shifted code; the operator
    can path-integrate the SDR (ARCHITECTURE §2 L6)."""
    g = GridEncoder(scales=(7, 11, 13), dims=1, mw=3, bounds=[(0, 100)])
    assert g.decode(g.encode([40])) == (40,)
    assert g.decode(g.encode([40 + 5])) == (45,)             # a +5 translation lands exactly


# ── MultiEncoder (feature-at-location = colour ⊗ location) ───────────────────────────────────────────────────
def test_multi_encode_decode_roundtrip():
    m = MultiEncoder([("colour", CategoryEncoder(["red", "blue"], w=8, capacity=8)),
                      ("loc", GridEncoder(scales=(7, 11, 13), dims=2, mw=3, bounds=[(0, 31), (0, 31)]))])
    sdr = m.encode({"colour": "blue", "loc": (10, 20)})
    out = m.decode(sdr)
    assert out["colour"] == "blue" and out["loc"] == (10, 20)


# ── SpatialPooler (the learned general SENSORY encoder; MICROCIRCUIT §7 — the canonical HTM spatial pooler) ────
def test_spatial_pooler_rules():
    """The three SDR rules on the LEARNED encoder: FIXED SPARSITY regardless of input density (the point of the SP —
    notes 99), DETERMINISM with learn=False (duty frozen → boost frozen), and OVERLAP=similarity (near inputs → near
    codes)."""
    import numpy as np
    sp = SpatialPooler(n_inputs=200, n_cols=512, w=20, seed=0)
    dense = (np.random.default_rng(1).random(200) < 0.4).astype(float)   # 40% of the input on
    sparse = np.zeros(200); sparse[np.random.default_rng(2).integers(0, 200, 6)] = 1   # a handful on
    assert sp.encode(dense, learn=False).w == 20                     # SPARSITY: fixed w no matter the input density
    assert sp.encode(sparse, learn=False).w == 20
    x = (np.random.default_rng(1).random(200) < 0.1).astype(float)
    assert sp.encode(x, learn=False) == sp.encode(x, learn=False)     # DETERMINISM (no learning)
    y = x.copy(); flip = np.flatnonzero(x)[:2]; y[flip] = 0           # a SIMILAR input (2 bits off)
    z = np.zeros(200); z[:20] = 1                                     # a DISSIMILAR input
    sim = sp.encode(x, learn=False).overlap(sp.encode(y, learn=False))
    dis = sp.encode(x, learn=False).overlap(sp.encode(z, learn=False))
    assert sim > dis                                                  # OVERLAP=similarity


def test_spatial_pooler_learns_a_stable_code():
    """LEARNING: the Hebbian carve converges — after training on a small set of inputs, each input's code STABILISES
    (barely moves under more learning) and DISTINCT inputs get well-SEPARATED codes. The SP has learned a representation
    of its input space (notes 105/107)."""
    import numpy as np
    rng = np.random.default_rng(0)
    sp = SpatialPooler(n_inputs=200, n_cols=512, w=20, seed=0)
    inputs = [(rng.random(200) < 0.1).astype(float) for _ in range(4)]   # 4 distinct inputs, round-robin (no duty blowup)
    for _ in range(40):
        for xi in inputs:
            sp.encode(xi, learn=True)
    before = [sp.encode(xi, learn=False) for xi in inputs]
    for _ in range(5):                                               # a little MORE learning...
        for xi in inputs:
            sp.encode(xi, learn=True)
    after = [sp.encode(xi, learn=False) for xi in inputs]
    for b, a in zip(before, after):
        assert b.overlap(a) >= 0.7 * sp.w                           # CONVERGED: the code barely moves under more learning
    for i in range(len(inputs)):                                    # DISTINCT inputs → well-separated codes
        for j in range(i + 1, len(inputs)):
            assert after[i].overlap(after[j]) <= 0.5 * sp.w


def test_spatial_pooler_homeostasis_no_dead_columns():
    """HOMEOSTASIS: boosting + dead-column revival spread the code across the WHOLE population — over a stream of random
    inputs (almost) every minicolumn gets used, so no small clique hogs the representation (notes 108/112). This is the
    property a plain top-k WITHOUT boosting lacks."""
    import numpy as np
    rng = np.random.default_rng(0)
    sp = SpatialPooler(n_inputs=200, n_cols=256, w=20, seed=0)
    ever = set()
    for _ in range(300):
        ever |= sp.encode((rng.random(200) < 0.1).astype(float), learn=True).active
    assert len(ever) >= 0.8 * sp.n                                  # ≥80% of columns were used at least once
    assert sp.overlap_duty.min() >= sp.min_overlap_duty            # no column is stuck below the revival floor


# ── ConjunctiveEncoder (the tensor-product / SE(2) representation space) ─────────────────────────────────────
def test_conjunctive_encoder_tensor_product_roundtrip():
    """The TENSOR product of position ⊗ heading — a cell active iff both fields are; n = ∏ n, decode projects back."""
    import numpy as np
    c = ConjunctiveEncoder([("pos", GridEncoder(scales=(5, 7, 9), dims=2, mw=1, bounds=[(0, 20), (0, 20)])),
                            ("head", ScalarEncoder(0.0, 2 * np.pi, n=4, w=1, periodic=True))])
    assert c.n == c.fields[0][1].n * c.fields[1][1].n            # tensor: n = n_pos * n_head
    out = c.decode(c.encode({"pos": (10, 12), "head": np.pi / 2}))
    assert out["pos"] == (10, 12)                                # position recovered
    assert min(abs(out["head"] - np.pi / 2), 2 * np.pi - abs(out["head"] - np.pi / 2)) < 1.0   # heading recovered


# ── MOTOR DECODE: a motor 'thought' (SDR) is interpreted by the SAME encoder that would sense it ─────────────
def test_motor_decode_is_the_encoder_inverse():
    """The anti-parallel-system property (ARCHITECTURE §4): L5 emits an SDR; the motor ORGAN decodes it with the
    encoder — a discrete action via a CategoryEncoder, a click (x,y) via the GridEncoder. No separate decoder."""
    actions = CategoryEncoder(["ACTION1", "ACTION2", "ACTION3", "ACTION4"], w=8, capacity=8)
    thought = actions.encode("ACTION3")                      # a motor 'thought' emitted by L5
    assert actions.decode(thought) == "ACTION3"              # the effector reads it back -> the action

    clicks = GridEncoder(scales=(7, 11, 13, 17), dims=2, mw=3, bounds=[(0, 63), (0, 63)])
    target = clicks.encode((41, 12))                         # a target-location 'thought' (ARC ACTION6)
    assert clicks.decode(target) == (41, 12)                 # decoded to the click coordinate
