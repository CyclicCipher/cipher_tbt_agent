"""test_modality.py -- a sensory modality is JUST PARAMETERS, and the factory builds a wired column from them.

The point of the modality bundle: the column + connections are modality-invariant (the TBT thesis), so adding a sense is a
four-field spec. This checks the two specs we ship (vision transducer-only; touch with a body-frame column), that the factory
builds touch's column, and that the built column binds contact-at-face cleanly through the same `GridEncoder` the spatial
columns use -- no per-modality encoder.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.encoders import SDR, CategoryEncoder        # noqa: E402
from tbt.modality import build_column, touch, vision  # noqa: E402
from tasks.games.push import Push                    # noqa: E402


def test_vision_is_transducer_only():
    """VISION declares no column yet (its recognising column is the Gap-3 paydown) -- an honest depth, not dead weight."""
    v = vision()
    assert not v.has_column() and build_column(v, 64, 0) is None


def test_touch_is_a_body_frame_column_spec():
    """TOUCH declares a feature space + a BODY frame, so the factory gives it a column; it rides the body (efference-coupled)."""
    t = touch()
    assert t.has_column() and t.pose_source == "body"
    assert build_column(t, 64, 0) is not None


def test_touch_transduces_the_skin():
    """The touch spec's transducer IS the skin sense: standing left of the block, the east face reads colour 6."""
    game = Push(); game.load_level(0); game.agent = (1, 2)
    contact = touch().transduce(game.render()[0], {game.agent})
    assert contact[((1, 2), (1, 0))] == 6, "east face presses into the block"


def test_factory_column_binds_contact_at_face():
    """The factory's touch column (built from the spec's body-frame GridEncoder) binds each face's contact and recalls it --
    the reused grid code encompasses the somatotopic body frame, no new encoder."""
    t = touch()
    col = build_column(t, 64, 0)
    faces = {(0, -1): 1, (0, 1): 3, (1, 0): 6, (-1, 0): 7}
    for _ in range(5):
        for off, c in faces.items():
            col.locate(off); col.sense_at(t.feature.encode(c))
    for off, c in faces.items():
        col.locate(off)
        cols = col.predict_feature()
        assert cols and t.feature.decode(SDR(t.feature.n, cols)) == c, f"face {off} should recall contact {c}"
