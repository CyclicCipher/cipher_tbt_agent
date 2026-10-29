"""Tests of HDClibrary: each checks a property the library's docstrings claim. Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/tests/test_hdc.py -q
"""
import numpy as np

from arc_agent_v2.architecture.HDClibrary import (SDR, Associations, Blocks, CategoryEncoder, Conjunction, GridEncoder,
                                     ItemMemory, Modules, Permutation,
                                     ScalarEncoder, SpatialPooler, Synapses, bundle, cdt, chance_overlap, jaccard,
                                     kwta, overlap, shift, thin, union)

D, W = 2048, 40


def rng(seed=0):
    return np.random.default_rng(seed)


def test_sdr_set_algebra_matches_dense():
    r = rng()
    a, b = SDR.random(D, W, r), SDR.random(D, W, r)
    da, db = a.dense(bool), b.dense(bool)
    assert a.w == W and abs(a.sparsity - W / D) < 1e-12
    assert np.array_equal((a & b).dense(bool), da & db)
    assert np.array_equal((a | b).dense(bool), da | db)
    assert np.array_equal((a - b).dense(bool), da & ~db)
    assert np.array_equal((a ^ b).dense(bool), da ^ db)
    assert SDR.from_dense(da) == a and hash(SDR.from_dense(da)) == hash(a)
    assert int(a.idx[3]) in a and (int(a.idx[3]) + 1 in a) == bool(da[a.idx[3] + 1])
    assert overlap(a, a) == W and jaccard(a, a) == 1.0


def test_random_overlap_is_chance():
    r = rng(1)
    sdrs = [SDR.random(D, W, r) for _ in range(200)]
    o = [overlap(sdrs[i], sdrs[i + 1]) for i in range(199)]
    assert abs(np.mean(o) - chance_overlap(D, W)) < 0.3          # 0.78 expected
    assert max(o) <= 6


def test_kwta_and_thin():
    s = np.zeros(D)
    s[[5, 9, 700]] = [3, 2, 5]
    assert kwta(s, 2).idx.tolist() == [5, 700]
    assert kwta(s, 10, floor=0).w == 3
    a = SDR.random(D, W, rng(2))
    t = thin(a, 10, rng(3))
    assert t.w == 10 and (t - a).w == 0


def test_bundle_keeps_its_members():
    r = rng(4)
    members = [SDR.random(D, W, r) for _ in range(5)]
    b = bundle(members, W)
    assert b.w == W
    assert all(overlap(m, b) >= 5 for m in members)               # chance is 0.8
    assert overlap(SDR.random(D, W, r), b) <= 4


def test_cdt_binds_and_keeps_similarity():
    r = rng(5)
    a, b, c = (SDR.random(D, W, r) for _ in range(3))
    b2 = thin(b, 30, r) | thin(SDR.random(D, W, r), 10, r)       # b with a quarter of its bits replaced
    ab, ab2, ac = cdt([a, b], W), cdt([a, b2], W), cdt([a, c], W)
    assert ab.w == W
    assert overlap(ab, ab2) > overlap(ab, ac) + 5                  # similar inputs, similar bindings
    assert overlap(ab, a) < W and overlap(ab, b) < W               # it depends on both: neither input survives whole


def test_permutation_inverse_and_power():
    p = Permutation(D, rng(6))
    a = SDR.random(D, W, rng(7))
    assert p.inverse()(p(a)) == a
    assert p.power(3)(a) == p(p(p(a)))
    assert p.power(-2)(p.power(2)(a)) == a
    assert shift(shift(a, 5), -5) == a and overlap(shift(a, 1), a) < 5


def test_block_binding_is_an_invertible_group():
    bl = Blocks(D, 32)
    r = rng(8)
    a, b, c = bl.random(r), bl.random(r), bl.random(r)
    assert bl.is_code(a) and a.w == 32
    assert bl.unbind(bl.bind(a, b), b) == a
    assert bl.bind(a, b) == bl.bind(b, a)
    assert bl.bind(bl.bind(a, b), c) == bl.bind(a, bl.bind(b, c))
    assert bl.bind(a, bl.inverse(a)) == bl.identity()
    assert overlap(bl.bind(a, b), a) <= 4                          # a binding resembles neither argument


def test_power_turns_the_hand():
    """Modular addition as rotation: argument 1 turns the hand to itself, argument 2 turns it by itself."""
    bl = Blocks(D, 32)
    g = bl.random(rng(9))
    for x, y in [(3, 4), (17, 50), (60, 9), (-5, 12)]:
        assert bl.bind(bl.power(g, x), bl.power(g, y)) == bl.power(g, x + y)
    assert bl.power(g, bl.L) == bl.identity()                     # each block is a ring of L = 64
    near = np.mean([overlap(bl.power(g, x), bl.power(g, x + 1)) for x in range(20)])
    assert near <= 4                                               # an exact group code: neighbours share ~nothing


def test_lattice_code_is_exact_on_integers():
    bl = Blocks(D, 32)
    r = rng(10)
    gx, gy = bl.random(r), bl.random(r)
    place = lambda x, y: bl.bind(bl.power(gx, x), bl.power(gy, y))
    assert bl.bind(place(3, 7), place(-2, 5)) == place(1, 12)
    assert overlap(place(3, 7), place(4, 7)) <= 6


def test_block_bundle_is_a_vote():
    bl = Blocks(D, 32)
    r = rng(11)
    a, b = bl.random(r), bl.random(r)
    assert bl.bundle([a, a, b]) == a
    assert bl.is_code(bl.bundle([a, b, bl.random(r)]))


def test_scalar_encoder_is_metric():
    e = ScalarEncoder(D, W, 0, 100)
    o = [overlap(e.encode(50), e.encode(50 + dx)) for dx in (0.5, 1.0, 1.5, 3.0)]   # the run spans 2 units here
    assert o[0] > o[1] > o[2] > o[3] == 0
    assert abs(e.decode(e.encode(37)) - 37) < 0.1
    ring = ScalarEncoder(D, W, 0, 360, periodic=True)
    assert overlap(ring.encode(0), ring.encode(359.5)) >= W - 4    # the ring wraps (359.5 is 2.8 bits before 0)
    assert abs(ring.decode(ring.encode(200)) - 200) < 0.5


def test_grid_encoder_overlap_falls_with_distance():
    g = GridEncoder(D, 32, seed=0)
    r = rng(12)
    pts = r.uniform(0, 64, (100, 2))

    def mean_overlap(dist):
        ang = r.uniform(0, 2 * np.pi, len(pts))
        return np.mean([overlap(g.encode(p), g.encode(p + dist * np.array([np.cos(t), np.sin(t)])))
                        for p, t in zip(pts, ang)])

    o1, o4, o32 = mean_overlap(1), mean_overlap(4), mean_overlap(32)
    assert g.blocks.is_code(g.encode(pts[0]))
    assert o1 > o4 > o32
    assert o1 >= 8 and o32 < 3                                     # chance is 32 / 64 = 0.5 bits


def test_category_encoder_has_no_structure_and_decodes():
    e = CategoryEncoder(D, W, seed=0)
    s = [e.encode(i) for i in range(20)]
    assert e.encode(3) == s[3]                                     # deterministic
    assert max(overlap(s[i], s[i + 1]) for i in range(19)) <= 6     # neighbours are not alike: no number line given
    noisy = thin(s[7], 25, rng(13)) | SDR.random(D, 15, rng(14))
    assert e.decode(noisy) == 7


def test_item_memory_recalls_nearest():
    r = rng(15)
    m = ItemMemory(D)
    items = [SDR.random(D, W, r) for _ in range(300)]
    for it in items:
        m.add(it)
    q = thin(items[123], 20, r) | SDR.random(D, 20, r)
    i, o = m.nearest(q)
    assert i == 123 and o >= 20
    top, ov = m.nearest(items[5], k=3)
    assert top[0] == 5 and ov[0] == W and len(top) == 3


def test_synapses_learn_a_pattern():
    syn = Synapses(10, D, potential=1.0, seed=0)
    x = SDR.random(D, W, rng(16))
    before = syn.overlaps(x)[3]
    for _ in range(10):
        syn.learn(x, [3])
    assert syn.overlaps(x)[3] == W and syn.overlaps(x)[3] > before
    assert syn.overlaps(SDR.random(D, W, rng(17)))[3] < 10        # the inactive inputs were weakened


def test_spatial_pooler_keeps_overlap_and_spreads_use():
    r = rng(18)
    n_in = 400
    sp = SpatialPooler(n_in, 1024, 20, seed=0)
    protos = [SDR.random(n_in, 30, r) for _ in range(10)]

    def noisy(p):
        return thin(p, 24, r) | SDR.random(n_in, 6, r)

    for _ in range(30):
        for p in protos:
            sp.compute(noisy(p), learn=True)
    same = np.mean([overlap(sp.compute(noisy(p), learn=False), sp.compute(noisy(p), learn=False)) for p in protos])
    diff = np.mean([overlap(sp.compute(protos[i], learn=False), sp.compute(protos[i + 1], learn=False))
                    for i in range(9)])
    assert same > 12 and diff < 4
    used = union(sp.compute(noisy(p), learn=False) for p in protos)
    assert used.w > 100                                            # no single clique codes everything


def test_conjunction_separates_patterns():
    r = rng(19)
    conj = Conjunction(D, W, seed=0)
    p, q, x, y = (SDR.random(D, W, r) for _ in range(4))
    assert conj(p, x) == conj(p, x) and conj(p, x).w == W
    assert overlap(conj(p, x), conj(p, y)) <= 6                    # sharing the place is not enough
    assert overlap(conj(p, x), conj(q, x)) <= 6                    # nor sharing the input
    assert overlap(conj(p, x), p) <= 6 and overlap(conj(p, x), x) <= 6


def test_associations_store_and_recall_by_overlap():
    r = rng(20)
    m = Associations(D, theta=W // 2)
    k1, k2, v1, v2 = (SDR.random(D, W, r) for _ in range(4))
    m.store(k1, v1)
    m.store(k1, v2)
    m.store(k1, v1)
    m.store(k2, v2)
    assert len(m) == 2 and m.recall(k1) == {v1: 2, v2: 1} and m.best(k1) == v1
    assert m.best(thin(k2, 30, r) | SDR.random(D, 10, r)) == v2   # a noisy key still finds its value
    assert m.recall(SDR.random(D, W, r)) == {} and m.best(SDR.random(D, W, r)) is None


def test_modules_bind_adds_phases_per_module():
    m = Modules(D, (3, 5, 7), R=14, seed=0)
    a, b = m.code([1, 2, 3]), m.code([2, 4, 6])
    assert a.w == 42 and m.is_code(a) and not m.is_code(SDR.random(D, 42, rng(21)))
    assert m.bind(a, b) == m.code([0, 1, 2]) and m.unbind(m.bind(a, b), b) == a
    assert m.phases(a).tolist() == [1, 2, 3] and m.bind(a, m.identity()) == a
    assert overlap(a, m.code([1, 0, 3])) == 2 * 14 and overlap(a, m.code([0, 0, 0])) == 0   # R bits per agreeing module
