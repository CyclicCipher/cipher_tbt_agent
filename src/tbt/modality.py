"""modality.py -- a SENSORY MODALITY as a parameter bundle, and the factory that turns it into a wired sensory column.

Why a modality is *just parameters* is a THEOREM of TBT, not a lucky pattern from two examples: the cortical COLUMN and the
thalamo-cortical CONNECTIONS are modality-INVARIANT. The same feature-at-location algorithm runs whether the sensor is retina or
skin, and the wiring (efference copy in, recognised content up to the shared scene column, cross-column voting) is identical for
every sensory column. So everything that distinguishes a sense collapses to `(transduce, feature, location, pose_source)`, and
`Agent(modalities=[...])` builds it all from that list -- adding proprioception or audio later is one more spec, zero new wiring.

Every modality's `location` is one `GridEncoder` class, differing only in its FRAME parameters: the board (`bounds 0..63`) for
vision, a small local BODY frame for touch. A body face is a POSITION in the body's own 2-D coordinates -- the somatotopic map
IS a 2-D body-surface map -- so the grid code encompasses it (verified: 4/4 clean contact-at-face binding). No per-modality
encoder. `GridEncoder`'s modular metric aliases at that small scale, but that only degrades recognise-BY-touch generalisation
(deferred), not the identity binding the dynamics needs (`reference_sdr_regime_and_phase_codes`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .encoders import CategoryEncoder, GridEncoder
from .perceive import segment
from .touch import sense_contact


@dataclass
class Modality:
    """The parameters that distinguish one sense from another; the column and its connections are shared. `location`/`feature`
    are the column's L6a frame + feature space -- both None means a TRANSDUCER-ONLY modality (no column yet: the honest state of
    vision until its recognising column is built, so it is not dead weight but a declared depth). `pose_source` is the efference
    coupling: 'world' (a fixed allocentric view) or 'body' (rides the self, so the efference copy moves this sensor's frame)."""

    name: str
    transduce: Callable                 # (observation, body_cells) -> the modality's readings
    feature: Optional[object] = None    # feature-space encoder (sets the column's feature width)
    location: Optional[object] = None   # L6a frame encoder (a GridEncoder in THIS modality's frame)
    pose_source: str = "world"

    def has_column(self) -> bool:
        """A modality gets a sensory column iff it declares both a frame and a feature space; else it is transducer-only."""
        return self.location is not None and self.feature is not None


def build_column(modality: Modality, n_cols: int, seed: int):
    """The FACTORY: a `Column` in the modality's frame, or None for a transducer-only modality. `order=2` so L4 fires
    feature-AT-location (the location-specific cell), exactly as the nav/scene spatial columns are built -- one column class,
    the modality supplies only the frame. The feature is fed later via `sense_at(modality.feature.encode(v))`."""
    if not modality.has_column():
        return None
    from .column import Column
    return Column(sensory_n=1, n_cols=n_cols, order=2, seed=seed, location=modality.location)


def vision(palette: int = 16) -> Modality:
    """VISION -- the retina. `segment` groups the frame into colour objects (the transducer). Transducer-only for now; the
    recognising vision COLUMN (identity by STRUCTURE, not colour) is the Gap-3 paydown and slots in here as feature/location when
    built. Allocentric board => pose_source 'world' (the eye does not ride the body)."""
    return Modality("vision", transduce=lambda obs, body: segment(obs), pose_source="world")


def touch(palette: int = 16) -> Modality:
    """TOUCH -- the skin over the BODY surface. `sense_contact` reports the colour pressed against each body face (the
    transducer); the touch column models that contact at the faces in the BODY frame (a small local `GridEncoder`, faces as
    positions). Feature = the contact colour (the object's felt surface); pose_source 'body' (it rides the self)."""
    return Modality(
        "touch",
        transduce=lambda obs, body: sense_contact(obs, body),
        feature=CategoryEncoder(range(palette), w=8, capacity=palette),
        location=GridEncoder(scales=(3, 4, 5), dims=2, mw=1, bounds=[(-2, 2), (-2, 2)]),
        pose_source="body",
    )
