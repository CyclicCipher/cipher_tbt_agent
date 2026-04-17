"""Cortical messaging protocol.

A CorticalMessage is the only thing a cortical column sends to
other columns — laterally (for voting) or upward (as input to a
higher region).

What it contains:
  location   — where the column currently IS in its reference frame
               (a hashable tuple from ReferenceFrame.position_key())
  feature    — what the column currently SEES at that location. 
               Must be an SDR, if it is not an SDR then the code is wrong.
  confidence — how well the column's prediction matched reality [0, 1]
               (1.0 = perfect prediction, 0.0 = never seen before)

What it does NOT contain:
  - Any string label or object name. Object identity is never a named
    thing inside any column. It emerges from the pattern of which
    columns are simultaneously active with consistent (feature, location)
    pairs.
  - Raw pixel values or raw sensory input.
  - Any explicit 'object_id'. Recognition = voting consensus, not a field.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorticalMessage:
    """Immutable message passed between cortical columns."""

    location: tuple
    """Position key from the sending column's reference frame."""

    feature: str
    """Feature key at this location (SDR identifier string)."""

    confidence: float
    """Prediction accuracy for this step, in [0, 1]."""
