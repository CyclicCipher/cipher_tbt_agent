"""Tests of the general grid-world framework with a second world (Canvas), the context coder, and moving a model's
trunk between worlds. Run: python -m pytest tests/test_gridworld.py """
import math

import numpy as np
import torch

from worlds import canvas as cv
from worlds import rollouts
from worlds import village
from nets.context_coder import ContextCoder
from nets.worldmodel import TRUNK, WorldModel
from worlds.gridworld import crop, relabeling


def test_canvas_moves_paints_and_keeps_the_screen():
    w = cv.Canvas(0)
    r, c = w.pos
    w.step(cv.PAINT + 3)
    assert w.colour[r, c] == 3 and w.painted[r, c]
    before = w.colour[w.screen].copy()
    w.step(cv.WAIT)
    assert not np.array_equal(before, w.colour[w.screen])              # the screen is redrawn (same: p = 4^-9)
    sr, sc = w.patches[cv.SCREEN]
    w.pos = (sr, sc)
    w.step(cv.PAINT + 1)
    assert not w.painted[sr, sc]                                        # the screen cannot be painted


def test_canvas_obs_matches_its_spec_and_walls_stop_nothing_outside():
    w = cv.Canvas(1)
    cells, glob = w.obs()
    assert cells.shape == (cv.N * cv.N, len(cv.CELL_VOCAB)) and glob.shape == (len(cv.GLOBAL_VOCAB),)
    assert (cells.max(0) < np.array(cv.CELL_VOCAB)).all() and cells[:, 1].sum() == 1
    w.pos = (0, 0)
    w.step(0)                                                           # north at the edge: stays
    assert w.pos == (0, 0)
    v = w.view()
    assert v.shape == (5, 5) and (v[:2] == cv.WALL).all() and (v[:, :2] == cv.WALL).all()


def test_clone_is_independent_and_hides_the_future_noise():
    w = cv.Canvas(2)
    c = w.clone(rng=np.random.default_rng(7))
    c.step(cv.PAINT + 2)
    c.step(1)
    assert w.t == 0 and w.stats["painted"] == 0 and c.t == 2
    w.step(cv.WAIT)
    c2 = cv.Canvas(2)
    c2.step(cv.WAIT)
    assert np.array_equal(w.colour, c2.colour)                          # cloning did not consume the world's randomness


def test_family_relabels_actions_and_codes():
    w = cv.Canvas(3, family="train")
    assert sorted(w.ph.action_perm) == list(range(cv.N_ACTIONS)) and w.ph.code_map[cv.WALL] == cv.WALL
    cells, _ = w.obs()
    assert np.array_equal(cells[:, 0], np.array(w.ph.code_map)[w.colour.ravel()])
    rng = np.random.default_rng(0)
    perm, codes = relabeling(rng, 8, 9, fixed=(0, 1))
    assert codes[:2] == (0, 1) and sorted(codes) == list(range(9)) and sorted(perm) == list(range(8))


def test_rollouts_serve_any_world():
    out = rollouts.run_lanes(("canvas", 0, 2, 50, 0, (1.0,), None, None))
    assert out["obs"].shape == (2, 50, cv.N * cv.N, 2) and "events" not in out
    out = rollouts.run_lanes(("village", 0, 1, 20, 0, (0.5,), None, None))
    assert out["obs"].shape == (1, 20, village.N * village.N, len(village.CELL_VOCAB)) and "events" in out


def test_crop_fills_outside():
    g = np.arange(16).reshape(4, 4)
    v = crop(g, (0, 0), 1, -1)
    assert v.tolist() == [[-1, -1, -1], [-1, 0, 1], [-1, 4, 5]]


def test_context_coder_learns_structure_and_starts_uniform():
    coder = ContextCoder(5)
    img = cv.stripes(np.random.default_rng(0), (6, 6))
    assert np.allclose(coder.bits(img), math.log2(5))
    first = coder.bits(img).mean()
    for _ in range(5):
        coder.update(img)
    assert coder.bits(img).mean() < 0.5 * first


def test_trunk_moves_between_worlds():
    torch.manual_seed(0)
    src = WorldModel(village.SPEC, 32, 2, 4, pos="local", persist=5.0)
    dst = WorldModel(cv.SPEC, 32, 2, 4, pos="local", persist=5.0)
    dst.load_trunk(src)
    sp, dp = dict(src.named_parameters()), dict(dst.named_parameters())
    for name, w in dp.items():
        if name.startswith(TRUNK) and name != "rel":
            assert torch.equal(w, sp[name]), name
    k = sp["rel"].shape[-1]                                             # village's 11 x 11 grid inside canvas's 16 x 16
    a = (dp["rel"].shape[-1] - k) // 2
    assert torch.equal(dp["rel"][:, a:a + k, a:a + k], sp["rel"])
    assert dp["act.weight"].shape[0] == cv.N_ACTIONS                    # the adapters stay the world's own
    cells, glob = cv.Canvas(0).obs()
    cl, gl, _ = dst(torch.from_numpy(cells)[None], torch.from_numpy(glob)[None], torch.tensor([0]))
    assert cl[0].shape == (1, cv.N * cv.N, cv.CELL_VOCAB[0])
