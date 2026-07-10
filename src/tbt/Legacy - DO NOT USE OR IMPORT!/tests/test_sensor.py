"""The retina + sensor after the symbolic-path retirement: the retina delivers raw feature-at-locations, reports what
CHANGED, and computes a bottom-up SALIENCE peak (center-surround pop-out). Object segmentation/tracking is gone — the
column owns it. The end-to-end sensorimotor loop is covered by the game tests (test_path_integration / test_live_loop)."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.retina import background, salient_cells, salient_targets  # noqa: E402
from tbt.sensor import Sensor  # noqa: E402


def test_salient_cells_reports_what_changed():
    prev = [[0, 0], [0, 7]]
    cur = [[0, 7], [0, 0]]
    assert salient_cells(prev, cur) == {(1, 0), (1, 1)}


def test_background_is_the_dominant_value():
    frame = [[0, 0, 0], [0, 7, 0], [0, 0, 3]]
    assert background(frame) == 0


def test_salience_peak_pops_out_the_distinct_marker_not_the_block_interior():
    """Center-surround pop-out: an isolated distinct cell (a marker) beats a uniform block's interior, and the tracked
    self is excluded — so a static marker becomes the cued-discovery target."""
    frame = [[0] * 8 for _ in range(8)]
    for dx in (0, 1):                                          # a 2x2 block (colour 7) — its interior does NOT pop out
        for dy in (0, 1):
            frame[1 + dy][1 + dx] = 7
    frame[5][6] = 3                                            # an isolated distinct marker (colour 3) — full contrast
    peaks = salient_targets(frame, exclude=[(1, 1), (2, 1), (1, 2), (2, 2)], k=1)
    assert peaks == [(6, 5)], peaks


def test_sensor_read_delivers_change_without_a_column():
    """Standalone (no column): the sensor still reports the change stream between frames."""
    s = Sensor(local=True, integrate=True)
    a = [[0, 0], [0, 7]]
    b = [[0, 7], [0, 0]]
    _s0, ch0 = s.read(a)
    assert ch0 == set()                                       # no previous frame -> no change yet
    _s1, ch1 = s.read(b)
    assert ch1 == {(1, 0), (1, 1)}
