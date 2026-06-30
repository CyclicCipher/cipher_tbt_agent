"""Egocentric local sensing (step 7b) -- the recurrence fix. Global per-pixel states never recur on real 64x64 frames
(measured live: cn04 0.04, ls20 0.01), so the loop is starved. Sensing a WINDOW around the dynamic-residual fovea makes
LOCAL views recur (measured on the real captured frames: 0.59 cn04 / 0.66 ls20, up from ~0). Here that is reproduced on
a synthetic noisy scene: the global encoding is unique every frame, the egocentric one collapses to the few local views
that actually recur -- the representation the column needs to learn."""

from __future__ import annotations

import os
import random
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.sensor import Sensor  # noqa: E402

N = 40


def _noisy_frames(n=24):
    """A 2x2 mover OSCILLATING between two spots, with a clump of cells in the FAR corner toggling colour every frame
    (a HUD / animation -- the 'noisy TV'). Globally every frame differs; locally (around the mover) only two views."""
    rng = random.Random(0)
    frames = []
    for i in range(n):
        g = [[0] * N for _ in range(N)]
        mx = 10 if i % 2 == 0 else 12                        # the mover oscillates between two positions
        for dx in (0, 1):
            for dy in (0, 1):
                g[10 + dy][mx + dx] = 7
        for k in range(5):                                   # toggling noise in the far corner (never in the mover's window)
            g[34][30 + k] = rng.choice([1, 2, 3])
        frames.append(g)
    return frames


def test_egocentric_recurs_where_global_does_not():
    """The global config-state is unique nearly every frame (the noise churns it); the egocentric window around the
    mover collapses to the handful of local views that recur -- exactly the encoding that lets the column learn."""
    frames = _noisy_frames(24)
    g = Sensor(local=False)
    glob = [g.read(f)[0] for f in frames]
    e = Sensor(local=True, window=7)
    ego = [e.read(f)[0] for f in frames]
    assert len(set(glob)) >= 0.7 * len(frames), f"global unexpectedly recurred: {len(set(glob))}/{len(frames)}"
    assert len(set(ego)) <= 4, f"egocentric did not recur: {len(set(ego))} distinct (expected the 2 mover views)"


def test_egocentric_fovea_tracks_the_moving_residual():
    """The fovea follows the dynamic residual (the controllable change), so the window is centred on the mover -- not
    on the autonomous corner noise (the largest CONNECTED change wins)."""
    s = Sensor(local=True, window=7)
    frames = _noisy_frames(4)
    s.read(frames[0])
    s.read(frames[1])
    fx, fy = s._fovea
    assert 9 <= fx <= 14 and 9 <= fy <= 12, f"fovea {s._fovea} not on the mover"


def test_feature_at_location_encodes_the_patch_and_preserves_recurrence():
    """The L4 seam (Phase 4a): with the column's L4.encode wired in, the egocentric state becomes (feature_id,
    position) -- a FEATURE-at-location, not a raw pixel tuple. The relabeling is INJECTIVE, so recurrence is
    identical to the raw-patch state (same local view -> same feature id), keeping the 7b/7c gains intact."""
    from tbt.column import CorticalColumn
    frames = _noisy_frames(24)
    raw = Sensor(local=True, window=7, integrate=True)
    raw_states = [raw.read(f)[0] for f in frames]
    col = CorticalColumn(n_entities=16, seed=0)
    enc = Sensor(local=True, window=7, integrate=True, encode=col.L4.encode)
    enc_states = [enc.read(f)[0] for f in frames]
    assert all(isinstance(s[0], int) for s in enc_states), enc_states[:3]   # state[0] is now an L4 feature id
    assert len(set(enc_states)) == len(set(raw_states))                     # injective -> identical recurrence
