"""Symbolic cortical column — TBT feature-location binding with SDRs.

A column stores (feature_SDR, location) bindings that form object models.
Recognition = the current (feature, location) observations overlap with
a stored model. No labels stored — identity emerges from which model
has the best overlap with current observations.

Each column maintains:
- A set of learned object models (each = set of feature-location bindings)
- A reference frame (current location, updated by displacement)
- SDR-based similarity matching for recognition

This is domain-general: the same column handles visual patches,
auditory features, or abstract tokens. Only the SDR encoding differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from sdr import SDR


MAX_MEMORY = 256  # max bindings per object model


@dataclass
class CorticalMessage:
    """TBT Cortical Messaging Protocol message."""
    object_id: str | None   # recognized object (or None if ambiguous)
    location: tuple          # position in reference frame
    feature: SDR | None      # observed feature at this location
    confidence: float = 0.0


class ObjectModel:
    """A learned object: set of (feature, location) bindings.

    The model IS the set of bindings. "Coffee cup" = {
        (smooth_texture, (0,0)),
        (handle_curve, (10,5)),
        (rough_ceramic, (-5,8)),
    }

    Recognition: how many of the current observations match bindings
    in this model (SDR overlap at each location).
    """

    def __init__(self, object_id: str, max_bindings: int = MAX_MEMORY):
        self.object_id = object_id
        self.max_bindings = max_bindings
        # Bindings: list of (location_tuple, feature_SDR)
        self.bindings: list[tuple[tuple, SDR]] = []

    def add_binding(self, location: tuple, feature: SDR):
        """Store a feature-location binding. Evict oldest if full."""
        # Check for duplicate location (update feature if exists).
        for i, (loc, _) in enumerate(self.bindings):
            if loc == location:
                self.bindings[i] = (location, feature)
                return
        if len(self.bindings) >= self.max_bindings:
            self.bindings.pop(0)  # evict oldest
        self.bindings.append((location, feature))

    def match_score(self, observations: list[tuple[tuple, SDR]],
                    threshold: float = 0.3) -> float:
        """Score how well current observations match this model.

        For each observation (location, feature), find the best
        matching binding in the model at that location. Score =
        average SDR similarity across matching locations.
        """
        if not self.bindings or not observations:
            return 0.0

        # Build location → feature map for fast lookup.
        model_map: dict[tuple, SDR] = {loc: feat for loc, feat in self.bindings}

        total_sim = 0.0
        n_matched = 0
        for obs_loc, obs_feat in observations:
            stored_feat = model_map.get(obs_loc)
            if stored_feat is not None:
                sim = obs_feat.similarity(stored_feat)
                if sim >= threshold:
                    total_sim += sim
                    n_matched += 1

        if n_matched == 0:
            return 0.0
        return total_sim / max(n_matched, 1)

    @property
    def n_bindings(self) -> int:
        return len(self.bindings)

    def __repr__(self):
        return f"ObjectModel({self.object_id}, {self.n_bindings} bindings)"


class SymbolicColumn:
    """A cortical column that stores object models via feature-location binding.

    The column:
    1. Receives a feature (SDR) at its retinal position
    2. Transforms retinal position → object-centered location (via reference frame)
    3. Stores (feature, location) bindings for known objects
    4. Recognizes objects by matching current observations against stored models
    5. Reports its best match via CMP message

    No labels — identity = which stored model has best overlap.
    """

    def __init__(self, name: str, receptive_field: Any = None,
                 position: tuple = (0.0, 0.0),
                 max_models: int = 32,
                 max_bindings: int = MAX_MEMORY):
        self.name = name
        self.receptive_field = receptive_field
        self.position = position  # retinotopic position
        self.max_models = max_models
        self.max_bindings = max_bindings

        # Object models: the column's learned knowledge.
        self.models: dict[str, ObjectModel] = {}

        # Reference frame state.
        self.location: tuple = (0.0, 0.0)  # current location in object-centered coords

        # Current observation.
        self.current_feature: SDR | None = None
        self.recognized: str | None = None
        self.confidence: float = 0.0

    def set_location(self, location: tuple):
        """Set current location in the reference frame."""
        self.location = location

    def displace(self, dx: float, dy: float):
        """Update location by displacement (efference copy)."""
        self.location = (self.location[0] + dx, self.location[1] + dy)

    def observe(self, feature: SDR):
        """Observe a feature at the current location. Try to recognize."""
        self.current_feature = feature
        self._recognize()

    def learn(self, object_id: str, feature: SDR | None = None,
              location: tuple | None = None):
        """Store a (feature, location) binding for an object.

        Uses current feature and location if not specified.
        Creates the object model if it doesn't exist.
        """
        feat = feature if feature is not None else self.current_feature
        loc = location if location is not None else self.location
        if feat is None:
            return

        if object_id not in self.models:
            if len(self.models) >= self.max_models:
                # Evict the model with fewest bindings.
                worst = min(self.models, key=lambda k: self.models[k].n_bindings)
                del self.models[worst]
            self.models[object_id] = ObjectModel(object_id, self.max_bindings)

        self.models[object_id].add_binding(loc, feat)

    def _recognize(self):
        """Match current observation against all stored models."""
        if self.current_feature is None:
            self.recognized = None
            self.confidence = 0.0
            return

        obs = [(self.location, self.current_feature)]
        best_id = None
        best_score = 0.0

        for obj_id, model in self.models.items():
            score = model.match_score(obs)
            if score > best_score:
                best_score = score
                best_id = obj_id

        self.recognized = best_id
        self.confidence = best_score

    def message(self) -> CorticalMessage:
        """Produce a CMP message."""
        return CorticalMessage(
            object_id=self.recognized,
            location=self.location,
            feature=self.current_feature,
            confidence=self.confidence,
        )

    def reset(self):
        """Reset transient state. Models preserved."""
        self.location = (0.0, 0.0)
        self.current_feature = None
        self.recognized = None
        self.confidence = 0.0

    def total_bindings(self) -> int:
        return sum(m.n_bindings for m in self.models.values())

    def __repr__(self):
        return (f"SymbolicColumn({self.name}, {len(self.models)} models, "
                f"{self.total_bindings()} bindings)")


# -----------------------------------------------------------------------
# Succession Engine (unchanged — token domain, not SDR)
# -----------------------------------------------------------------------

class SuccessionEngine:
    """Z/10Z successor morphism for multi-digit succession."""

    SUCC = {}
    for _d in range(10):
        for _c in (False, True):
            _val = _d + 1 + (1 if _c else 0)
            SUCC[(_d, _c)] = (_val % 10, _val >= 10)

    @staticmethod
    def successor(number_str: str) -> str:
        digits = [int(d) for d in number_str]
        result = []
        carry = False
        for i in range(len(digits) - 1, -1, -1):
            d = digits[i]
            if i == len(digits) - 1:
                out_d, carry = SuccessionEngine.SUCC[(d, False)]
            else:
                if carry:
                    out_d, carry = SuccessionEngine.SUCC[(d, False)]
                else:
                    out_d, carry = d, False
            result.append(str(out_d))
        if carry:
            result.append("1")
        result.reverse()
        return ''.join(result)
