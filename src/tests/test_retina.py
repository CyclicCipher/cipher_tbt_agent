"""The sensorimotor retina (tbt.retina): a raw frame -> recurring (feature, pose) observations + the
exogenous-attention salience channel. This is the front-end the live games proved necessary — global frames never
recur, but local receptive fields recur ~99%, so the column must sense locally. Pure stdlib; no live API."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.retina import Retina, dominant_region, salient_cells  # noqa: E402


def _frame(n=8, fill=0):
    return [[fill] * n for _ in range(n)]


def test_same_patch_same_feature_novel_gets_new_id():
    """Recurrence + label-free discovery: identical local content gets the SAME feature, new content a NEW id."""
    r = Retina(rf=3)
    g = _frame(8)
    g[1][1] = 5                                               # one structured 3x3 at the origin
    f1, p1 = r.sense(g, 0, 0)
    f2, _ = r.sense(g, 4, 4)                                  # a different (all-zero) patch
    f3, _ = r.sense(g, 0, 0)                                  # the SAME patch again -> same id
    assert f1 == f3 and f1 != f2
    assert p1 == (0, 0)
    assert len(r.codebook) == 2


def test_perceive_tiles_the_frame():
    r = Retina(rf=4, stride=4)
    obs = r.perceive(_frame(8))                               # 8x8 stride 4 -> a 2x2 tiling = 4 RFs
    assert len(obs) == 4
    assert all(isinstance(f, int) and len(pose) == 2 for f, pose in obs)


def test_codebook_recurs_not_grows_unbounded():
    """A repeating pattern yields a tiny vocabulary regardless of frame size — the recurrence the column needs
    (the opposite of the global frame, which is unique every time)."""
    r = Retina(rf=2, stride=2)
    checker = [[(x + y) % 2 for x in range(20)] for y in range(20)]
    r.perceive(checker)
    assert len(r.codebook) <= 4                               # only a handful of distinct 2x2 patches exist


def test_salience_finds_the_coherent_moving_object():
    """Exogenous attention picks out the changed cells and foveates the largest connected moving region."""
    prev = _frame(10)
    cur = _frame(10)
    for (x, y) in [(3, 3), (4, 3), (3, 4), (4, 4)]:          # a 2x2 object appears
        cur[y][x] = 7
    cells = salient_cells(prev, cur)
    assert cells == {(3, 3), (4, 3), (3, 4), (4, 4)}
    comp, centroid = dominant_region(cells)
    assert len(comp) == 4 and centroid == (3.5, 3.5)


def test_no_change_no_salience():
    g = _frame(6, fill=2)
    assert salient_cells(g, g) == set()
    assert dominant_region(set()) == (set(), None)
