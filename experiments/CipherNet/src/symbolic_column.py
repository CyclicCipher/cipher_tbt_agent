"""Symbolic cortical column with receptive fields and hierarchical stacking.

Each column has a POSITION (from wiring) and learns FEATURES at that
position. Memory accumulates votes. Prediction returns the mode.

HIERARCHY: each level uses the same SymbolicColumn protocol. A higher
level's input features ARE the lower level's output votes + positions.
This is TBT's mechanism: V1 outputs "edge_H at (3,2)" → V2 treats
that as a feature and learns "edge_H@(3,2) + edge_V@(4,2) = corner".

The Cortical Messaging Protocol (CMP): columns exchange
(object_id, position, confidence). Same format at every level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


# -----------------------------------------------------------------------
# Cortical Messaging Protocol
# -----------------------------------------------------------------------

@dataclass
class CorticalMessage:
    """Message exchanged between columns (TBT's CMP).

    object_id: what the column thinks it's seeing (its vote)
    position: where on the cortical sheet this column sits
    confidence: how sure the column is (vote_count / total_votes)
    """
    object_id: str | None
    position: tuple
    confidence: float = 0.0


# -----------------------------------------------------------------------
# Symbolic Column
# -----------------------------------------------------------------------

MAX_MEMORY = 256  # biological minicolumn capacity (~100-500 patterns)


class SymbolicColumn:
    """A cortical column with capped memory (256 entries max).

    When memory is full, the least confident entry is evicted.
    This forces selectivity: columns keep only the most useful
    associations, like biological columns developing orientation
    or object selectivity through competitive learning.
    """

    def __init__(self, name: str, receptive_field: Any = None,
                 position: tuple = (0.0, 0.0),
                 max_memory: int = MAX_MEMORY):
        self.name = name
        self.receptive_field = receptive_field
        self.position = position
        self.max_memory = max_memory
        self.memory: dict[str, dict[str, int]] = {}
        self.current_input: str | None = None
        self.prediction: str | None = None
        self.confidence: float = 0.0

    def observe(self, feature: str):
        """Set current input and compute prediction."""
        self.current_input = feature
        self.prediction = self.predict()

    def teach(self, feature: str, target: str):
        """Accumulate: feature → target gets +1 vote. Evict if full."""
        if feature not in self.memory:
            if len(self.memory) >= self.max_memory:
                self._evict()
            self.memory[feature] = {}
        counts = self.memory[feature]
        counts[target] = counts.get(target, 0) + 1

    def _evict(self):
        """Remove the least confident memory entry.

        Confidence = max_votes / total_votes for that feature.
        Low confidence = ambiguous feature (maps to many targets
        with similar frequency). Evicting it loses the least.
        """
        worst_key = None
        worst_conf = float('inf')
        for feat, counts in self.memory.items():
            total = sum(counts.values())
            if total == 0:
                worst_key = feat
                break
            best = max(counts.values())
            conf = best / total
            # Tiebreak: prefer evicting features with fewer total votes.
            score = conf + 0.001 * total  # slight bias toward keeping frequent
            if score < worst_conf:
                worst_conf = score
                worst_key = feat
        if worst_key is not None:
            del self.memory[worst_key]

    def predict(self) -> str | None:
        """Return mode (most frequent target) for current input."""
        if self.current_input is None:
            return None
        counts = self.memory.get(self.current_input)
        if not counts:
            return None
        total = sum(counts.values())
        winner = max(counts, key=counts.get)
        self.confidence = counts[winner] / total
        return winner

    def message(self) -> CorticalMessage:
        """Produce a CMP message (vote + position + confidence)."""
        return CorticalMessage(
            object_id=self.prediction,
            position=self.position,
            confidence=self.confidence,
        )

    def reset(self):
        self.current_input = None
        self.prediction = None
        self.confidence = 0.0

    def __repr__(self):
        return f"SymbolicColumn({self.name}, {len(self.memory)} features)"


# -----------------------------------------------------------------------
# Column Sheet (2D grid of columns = one cortical area)
# -----------------------------------------------------------------------

class ColumnSheet:
    """A 2D grid of symbolic columns (one cortical area).

    Each column covers a rectangular receptive field of the input.
    For images: pixel patches. For higher levels: patches of lower-level
    columns' outputs.

    The same class is used at every hierarchy level. Only the input
    type changes (pixels → edge codes → shape codes → object codes).
    """

    def __init__(self, name: str, grid_h: int, grid_w: int,
                 rf_size: tuple[int, int] | None = None,
                 stride: tuple[int, int] | None = None):
        self.name = name
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.rf_size = rf_size  # (patch_h, patch_w) in input coordinates
        self.stride = stride    # (stride_h, stride_w)

        self.columns: list[list[SymbolicColumn]] = []
        for gy in range(grid_h):
            row = []
            for gx in range(grid_w):
                rf = None
                if rf_size and stride:
                    y0, x0 = gy * stride[0], gx * stride[1]
                    rf = (y0, x0, y0 + rf_size[0], x0 + rf_size[1])
                col = SymbolicColumn(
                    name=f"{name}:L{gy},{gx}",
                    receptive_field=rf,
                    position=(float(gx), float(gy)),
                )
                row.append(col)
            self.columns.append(row)

    def n_columns(self) -> int:
        return self.grid_h * self.grid_w

    def all_columns(self):
        for row in self.columns:
            yield from row

    def get_messages(self) -> list[list[CorticalMessage]]:
        """Collect CMP messages from all columns (2D grid of messages)."""
        return [[col.message() for col in row] for row in self.columns]

    def __repr__(self):
        return f"ColumnSheet({self.name}, {self.grid_h}x{self.grid_w})"


# -----------------------------------------------------------------------
# Hierarchical Column Stack (N levels, same protocol)
# -----------------------------------------------------------------------

class ColumnHierarchy:
    """A stack of ColumnSheets forming a cortical hierarchy.

    Level 0: receives raw features (pixel codes from codebook).
    Level N>0: receives MESSAGES from level N-1 as features.

    Each level uses the same SymbolicColumn with the same teach/predict.
    The only difference is what constitutes a "feature" at each level:
    - Level 0: "v42" (VQ codebook entry for a pixel patch)
    - Level 1: "v42@(0,0)+v17@(1,0)+v3@(0,1)+v55@(1,1)" (combination
      of lower-level codes at relative positions)
    - Level 2: combination of Level 1 outputs... etc.

    This is TBT: lower column's object_id becomes higher column's feature.
    """

    def __init__(self):
        self.levels: list[ColumnSheet] = []
        self._level_pool: list[tuple[int, int]] = []  # pooling factor per level

    def add_level(self, sheet: ColumnSheet, pool_h: int = 2, pool_w: int = 2):
        """Add a cortical area to the hierarchy.

        pool_h, pool_w: how many lower-level columns each higher-level
        column's receptive field covers. E.g., pool 2×2 means each
        higher column sees a 2×2 patch of lower columns' outputs.
        """
        self.levels.append(sheet)
        self._level_pool.append((pool_h, pool_w))

    def n_levels(self) -> int:
        return len(self.levels)

    @staticmethod
    def compose_feature(messages: list[CorticalMessage]) -> str:
        """Compose a higher-level feature from lower-level messages.

        Concatenates (object_id @ relative_position) for each message.
        This IS the feature-at-location binding from TBT.

        E.g., messages from a 2×2 patch of lower columns:
          [("edge_H", (0,0)), ("edge_V", (1,0)), (None, (0,1)), ("edge_H", (1,1))]
        → "edge_H@0,0|edge_V@1,0|_@0,1|edge_H@1,1"
        """
        parts = []
        if not messages:
            return "_empty"
        # Use relative positions (offset from first message's position).
        base_x, base_y = messages[0].position
        for msg in messages:
            obj = msg.object_id or "_"
            rx = msg.position[0] - base_x
            ry = msg.position[1] - base_y
            parts.append(f"{obj}@{int(rx)},{int(ry)}")
        return "|".join(sorted(parts))  # sorted for position-invariant key

    def propagate(self):
        """Propagate messages from level 0 upward through the hierarchy.

        At each level above 0:
        1. Collect messages from the level below
        2. For each higher-level column, gather the lower-level messages
           within its receptive field (pooling region)
        3. Compose them into a single feature string
        4. Feed that feature to the higher-level column
        """
        for lev in range(1, len(self.levels)):
            lower = self.levels[lev - 1]
            upper = self.levels[lev]
            pool_h, pool_w = self._level_pool[lev]

            lower_msgs = lower.get_messages()

            for gy in range(upper.grid_h):
                for gx in range(upper.grid_w):
                    # Gather messages from the lower-level patch.
                    patch_msgs = []
                    for dy in range(pool_h):
                        for dx in range(pool_w):
                            ly = gy * pool_h + dy
                            lx = gx * pool_w + dx
                            if ly < lower.grid_h and lx < lower.grid_w:
                                patch_msgs.append(lower_msgs[ly][lx])

                    # Compose into a higher-level feature.
                    feature = self.compose_feature(patch_msgs)
                    upper.columns[gy][gx].observe(feature)

    def teach_all(self, target: str):
        """Teach the TARGET label to all columns at all levels.

        Every column at every level learns: my_current_feature → target.
        This is weak supervision: each column independently associates
        its local (possibly abstract) feature with the global label.
        """
        for sheet in self.levels:
            for col in sheet.all_columns():
                if col.current_input is not None:
                    col.teach(col.current_input, target)

    def vote(self) -> tuple[str | None, dict[str, float]]:
        """Collect votes from ALL levels. Return (winner, vote_scores).

        Higher levels get more weight (they see larger context).
        Level weight = 2^level (level 0 = 1, level 1 = 2, level 2 = 4).
        """
        scores: dict[str, float] = {}
        for lev, sheet in enumerate(self.levels):
            weight = 2.0 ** lev  # higher levels count more
            for col in sheet.all_columns():
                pred = col.prediction
                if pred is not None:
                    scores[pred] = scores.get(pred, 0.0) + weight * col.confidence
        if not scores:
            return None, scores
        winner = max(scores, key=scores.get)
        return winner, scores

    def reset(self):
        for sheet in self.levels:
            for col in sheet.all_columns():
                col.reset()

    def __repr__(self):
        parts = [f"L{i}: {s}" for i, s in enumerate(self.levels)]
        return f"ColumnHierarchy({', '.join(parts)})"


# -----------------------------------------------------------------------
# Succession Engine (unchanged)
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
