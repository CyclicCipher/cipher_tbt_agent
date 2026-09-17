import numpy as np
import torch

from env.tokens import M0, OP0
from mdl.curriculum import Curriculum, water_fill_cap
from model_adapter import ModelAdapter, causality_test


def test_causality_and_size():
    a = ModelAdapter("4M", amp=False)
    assert causality_test(a) < 1e-5
    assert a.cache_matches_model() < 1e-4                    # the adapter's KV-cached decode equals the model's forward
    assert 3.5e6 < a.num_params() < 4.5e6, a.num_params()
    assert a.set_loop_depth(2) is False


def test_init_slot_changes_only_intended_rows():
    a = ModelAdapter("1M", amp=False)
    emb0, head0, bias0 = (a.model.emb.weight.clone(), a.model.head.weight.clone(), a.model.head.bias.clone())
    a.init_slot(3, [OP0 + 1, OP0 + 4])
    row = M0 + 3
    d_emb = (a.model.emb.weight != emb0).any(1)
    d_head = (a.model.head.weight != head0).any(1)
    d_bias = a.model.head.bias != bias0
    assert d_emb.nonzero().flatten().tolist() == [row]
    assert d_head.nonzero().flatten().tolist() == [row]
    assert d_bias.nonzero().flatten().tolist() == [row]
    assert torch.allclose(a.model.emb.weight[row], emb0[[OP0 + 1, OP0 + 4]].mean(0))
    a.reset_slot(3)
    assert (a.model.emb.weight != emb0).any(1).nonzero().flatten().tolist() == [row]


def test_water_fill_cap():
    q = np.array([0.5, 0.3, 0.1, 0.1])
    c = water_fill_cap(q, 0.35)
    assert abs(c.sum() - 1) < 1e-9 and c.max() <= 0.35 + 1e-9 and c[1] >= 0.3


def test_noise_gets_floor_under_gain_and_top_under_curio():
    fids = list(range(100))
    noise = set(range(90, 100))
    R = 12
    for mode in ("gain", "curio"):
        cur = Curriculum(fids, mode=mode)
        for r in range(R):
            losses = {f: (3.3 if f in noise else max(0.2, 2.5 - 0.25 * r)) for f in fids}   # flat-high vs falling
            cur.update(losses)
        p = {f: float(cur.p[i]) for i, f in enumerate(fids)}
        floor = cur.mix / len(fids)
        if mode == "gain":
            for f in noise:
                assert p[f] < floor + 0.01, p                    # only the uniform floor
        else:
            top = sorted(fids, key=lambda f: -p[f])[:10]
            assert set(top) == noise, p
