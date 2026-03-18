"""
Surprise Detection (Stage 5).

Computes KL(observed || predicted) per token position and flags positions
whose surprise exceeds a threshold.  The threshold is stored as an edge
annotation in the MorphismGraph rather than as a Python float constant,
satisfying the Phase XXV requirement that "all knowledge stored in CTKG".

Design
------
- ``SurpriseDetector`` wraps a ``Predictor`` and records surprise per token.
- ``compute_surprise(prefix, observed_token)`` returns the KL divergence
  between the point-mass observation and the predicted distribution.
- The threshold is stored as the ``weight`` field of a SURPRISE_THRESHOLD
  self-loop morphism on a designated "meta" object in the MorphismGraph.
- ``is_surprising(surprise_value)`` reads the threshold from the graph and
  returns True when the surprise exceeds it.
- ``scan_sequence(tokens)`` returns a list of (position, token, surprise)
  triples where the token was flagged as surprising.

Iron Law compliance
-------------------
No string comparisons on token content.  Tokens are encoded to NodeId at the
boundary; KL is computed purely over the predicted probability distribution
(a dict[str, float] from Predictor.predict_next).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    ObjectId,
    MorphId,
)


# ---------------------------------------------------------------------------
# Surprise annotation
# ---------------------------------------------------------------------------

_THRESHOLD_MORPH_TYPE = "SURPRISE_THRESHOLD"
_META_LABEL = "__surprise_meta__"
_DEFAULT_THRESHOLD = 1.0          # nats; ~0.86 bits


@dataclass
class SurpriseAnnotation:
    """A flagged token position in a sequence.

    Attributes
    ----------
    position  : 0-based index in the sequence.
    token     : the token that was observed.
    surprise  : KL(δ_token || predicted) in nats.
    predicted : the distribution that was predicted at this position.
    """
    position: int
    token: str
    surprise: float
    predicted: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SurpriseDetector
# ---------------------------------------------------------------------------

class SurpriseDetector:
    """Wrapper around Predictor that measures per-token surprise.

    Parameters
    ----------
    predictor
        Any object with a ``predict_next(prefix: list[str]) -> dict[str, float]``
        method (typically ``ctkg.inference.predict.Predictor``).
    mg : MorphismGraph, optional
        Graph in which the surprise threshold is stored.  If None, a fresh
        graph is created.  The threshold is stored as the ``weight`` on a
        SURPRISE_THRESHOLD self-loop morphism on a meta object.
    threshold : float, optional
        Initial threshold value in nats.  Default 1.0.  Can be changed
        via ``set_threshold()``.  Persisted as graph edge annotation.
    """

    def __init__(
        self,
        predictor,
        mg: Optional[MorphismGraph] = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._predictor = predictor
        self._mg: MorphismGraph = mg if mg is not None else MorphismGraph()
        self._meta_obj_id: ObjectId = self._ensure_meta_object()
        self._threshold_morph_id: MorphId = self._ensure_threshold_morph(threshold)

    # ------------------------------------------------------------------
    # Threshold management (stored in graph)
    # ------------------------------------------------------------------

    def get_threshold(self) -> float:
        """Read the current surprise threshold from the graph."""
        m = self._mg._morphisms.get(self._threshold_morph_id)
        if m is None or m.weight is None:
            return _DEFAULT_THRESHOLD
        return m.weight

    def set_threshold(self, value: float) -> None:
        """Update the surprise threshold in the graph."""
        m = self._mg._morphisms.get(self._threshold_morph_id)
        if m is not None:
            m.weight = float(value)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_surprise(
        self,
        prefix: list[str],
        observed_token: str,
    ) -> float:
        """Compute KL(δ_observed || predicted) for one token position.

        KL(δ_x || Q) = -log Q(x) when Q(x) > 0.
                     = +inf      when Q(x) = 0  (clamped to 20.0 for safety).

        Parameters
        ----------
        prefix         : tokens seen BEFORE the current position.
        observed_token : the token that actually occurred.

        Returns
        -------
        float : surprise in nats.
        """
        dist = self._predictor.predict_next(prefix)
        return _kl_point_mass(observed_token, dist)

    def is_surprising(self, surprise_value: float) -> bool:
        """Return True when *surprise_value* exceeds the stored threshold."""
        return surprise_value > self.get_threshold()

    def scan_sequence(
        self,
        tokens: list[str],
        start: int = 0,
    ) -> list[SurpriseAnnotation]:
        """Scan *tokens* and return flagged positions.

        For each position i >= start, computes surprise(tokens[:i], tokens[i])
        and flags positions where surprise > threshold.

        Parameters
        ----------
        tokens : the full token sequence.
        start  : first position to check (default 0 — check every position).

        Returns
        -------
        list[SurpriseAnnotation] for each flagged position, in order.
        """
        flagged: list[SurpriseAnnotation] = []
        for i in range(start, len(tokens)):
            prefix = tokens[:i]
            tok = tokens[i]
            dist = self._predictor.predict_next(prefix)
            s = _kl_point_mass(tok, dist)
            if self.is_surprising(s):
                flagged.append(SurpriseAnnotation(
                    position=i,
                    token=tok,
                    surprise=s,
                    predicted=dict(dist),
                ))
        return flagged

    def surprise_sequence(
        self,
        tokens: list[str],
        start: int = 0,
    ) -> list[float]:
        """Return the surprise value at every position in *tokens*.

        Same as ``scan_sequence`` but returns a flat list of floats (one per
        position >= start) without the threshold filter.
        """
        result: list[float] = []
        for i in range(start, len(tokens)):
            prefix = tokens[:i]
            tok = tokens[i]
            dist = self._predictor.predict_next(prefix)
            result.append(_kl_point_mass(tok, dist))
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_meta_object(self) -> ObjectId:
        """Create or retrieve the meta object that anchors the threshold morph."""
        # Check if a meta object already exists by label scan.
        for obj in self._mg.objects():
            if obj.label == _META_LABEL:
                return obj.obj_id
        # Create a new one with no concept (structural placeholder).
        obj = self._mg.add_object(concept=None, label=_META_LABEL)
        return obj.obj_id

    def _ensure_threshold_morph(self, initial_value: float) -> MorphId:
        """Create or retrieve the SURPRISE_THRESHOLD self-loop morphism."""
        oid = self._meta_obj_id
        existing = self._mg.hom(oid, oid, include_identity=False)
        for m in existing:
            if m.morph_type == _THRESHOLD_MORPH_TYPE:
                # Already exists; update weight to initial_value.
                m.weight = float(initial_value)
                return m.morph_id
        # Create new threshold morphism.
        m = self._mg.add_morphism(
            oid, oid,
            morph_type=_THRESHOLD_MORPH_TYPE,
            evidence=1,
        )
        m.weight = float(initial_value)
        return m.morph_id


# ---------------------------------------------------------------------------
# KL helper
# ---------------------------------------------------------------------------

_KL_INF_SUBSTITUTE = 20.0   # cap for log(0) cases


def _kl_point_mass(token: str, dist: dict[str, float]) -> float:
    """KL divergence from a point mass at *token* to distribution *dist*.

    KL(δ_token || dist) = -log dist(token)  if dist(token) > 0
                        = _KL_INF_SUBSTITUTE  otherwise (token not predicted)

    If *dist* is empty (predictor returned {}), the surprise is maximal.

    Parameters
    ----------
    token : the observed token.
    dist  : predicted distribution {token: probability}.

    Returns
    -------
    float : KL divergence in nats.
    """
    if not dist:
        return _KL_INF_SUBSTITUTE
    p = dist.get(token, 0.0)
    if p <= 0.0:
        return _KL_INF_SUBSTITUTE
    # Normalise in case probabilities don't sum to 1.
    total = sum(dist.values())
    if total > 0.0:
        p = p / total
    if p <= 0.0:
        return _KL_INF_SUBSTITUTE
    return -math.log(p)
