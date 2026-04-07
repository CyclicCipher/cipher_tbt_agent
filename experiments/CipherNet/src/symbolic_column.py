"""Symbolic cortical column with receptive fields.

Each column has a POSITION (from wiring — which input it receives)
and learns FEATURES at that position (from signal content). Position
and feature are pre-separated by the topographic mapping, just like
the brain's retinotopic/tonotopic/somatotopic maps.

Key insight: position is NOT learned or computed. Position is a
structural property of the wiring — WHICH column receives WHICH input.
The column's job is to learn feature associations AT its position.

Receptive fields overlap between neighbors, enabling the system to
know that adjacent columns represent adjacent positions (not isolated
modalities). This overlap is how the cortical map discovers local
structure.

The reference frame is the topographic map itself. Displacement =
shifting which column is active (attention moves across the map).
"""
from __future__ import annotations

from typing import Any


class SymbolicColumn:
    """A cortical column with a fixed receptive field.

    Position = which input positions this column receives from (receptive field).
    Feature = what signal content arrives at those positions.
    Memory = {feature_pattern → predicted_next_feature} (one-shot learned).
    """

    def __init__(self, name: str, receptive_field: tuple | None = None,
                 position: tuple = (0.0, 0.0)):
        self.name = name
        # Receptive field: which input positions this column covers.
        # For token sequences: a range of sequence positions.
        # For vision: a patch of pixel coordinates.
        # None = receives from all positions (global column).
        self.receptive_field = receptive_field
        # Position on the cortical sheet (for broadcast/neighbor discovery).
        self.position = position

        # Memory: maps observed patterns to predictions.
        # Key = what was observed. Value = what comes next (or associated feature).
        self.memory: dict[str, str] = {}

        # Current state (transient, reset between episodes).
        self.current_input: str | None = None
        self.prediction: str | None = None
        self.error: bool = False
        self.vote: str | None = None

    def observe(self, feature: str) -> bool:
        """Observe a feature. Compare against prediction. Learn if surprised."""
        self.current_input = feature
        self.prediction = self.memory.get(feature)
        self.error = (self.prediction is None)  # surprised if never seen before
        self.vote = feature
        return self.error

    def teach(self, feature: str, target: str):
        """One-shot: associate feature → target at this column's position."""
        self.memory[feature] = target

    def predict(self) -> str | None:
        """Predict: what is associated with the current input?"""
        if self.current_input is None:
            return None
        return self.memory.get(self.current_input)

    def reset(self):
        """Reset transient state. Memory preserved."""
        self.current_input = None
        self.prediction = None
        self.error = False
        self.vote = None

    def __repr__(self):
        return (f"SymbolicColumn({self.name}, rf={self.receptive_field}, "
                f"mem={len(self.memory)})")


class ColumnSheet:
    """A sheet of symbolic columns with topographic mapping.

    The sheet is a 1D or 2D array of columns, each with a receptive field
    that overlaps slightly with its neighbors. This is the cortical map.

    For token sequences: a 1D sheet where each column covers one sequence
    position, with overlap to adjacent positions.

    For vision: a 2D sheet where each column covers a patch of pixels,
    with overlap to adjacent patches.
    """

    def __init__(self, name: str, n_columns: int, overlap: int = 1):
        self.name = name
        self.columns: list[SymbolicColumn] = []
        self.overlap = overlap

        for i in range(n_columns):
            # Receptive field: positions [i-overlap, i+overlap] (clamped).
            rf_start = max(0, i - overlap)
            rf_end = i + overlap  # inclusive
            col = SymbolicColumn(
                name=f"{name}:{i}",
                receptive_field=(rf_start, rf_end),
                position=(float(i), 0.0),
            )
            self.columns.append(col)

    def feed(self, tokens: list[str]):
        """Feed a sequence of tokens to the sheet.

        Each column receives the token at its primary position.
        Overlap columns also see adjacent tokens.
        """
        for col in self.columns:
            rf_start, rf_end = col.receptive_field
            primary_pos = self.columns.index(col)
            if primary_pos < len(tokens):
                col.observe(tokens[primary_pos])

    def teach_succession(self, tokens: list[str], targets: list[str]):
        """Teach: for each position, token[i] → target[i]."""
        for i, col in enumerate(self.columns):
            if i < len(tokens) and i < len(targets):
                col.teach(tokens[i], targets[i])

    def predict_all(self) -> list[str | None]:
        """Get predictions from all columns."""
        return [col.predict() for col in self.columns]

    def get_column_at(self, position: int) -> SymbolicColumn | None:
        """Get the column whose primary position matches."""
        if 0 <= position < len(self.columns):
            return self.columns[position]
        return None

    def __repr__(self):
        return f"ColumnSheet({self.name}, {len(self.columns)} columns)"


class SuccessionEngine:
    """Handles multi-digit succession using the Z/10Z morphism.

    This is NOT a column — it's the computational rule that columns
    at different positions coordinate through. Each digit position
    is handled by a column in the sheet. The carry propagation
    is the displacement morphism (+1 mod 10 with carry).

    The engine applies the successor morphism position-by-position,
    right-to-left, propagating carry. This is the algorithm that
    a trained system would discover from the regularity of the
    Z/10Z group structure.
    """

    # Z/10Z successor morphism (innate counting ability).
    # {(digit, carry_in) → (output_digit, carry_out)}
    SUCC = {}
    for _d in range(10):
        for _c in (False, True):
            _val = _d + 1 + (1 if _c else 0)
            SUCC[(_d, _c)] = (_val % 10, _val >= 10)

    @staticmethod
    def successor(number_str: str) -> str:
        """Compute successor of a number string using Z/10Z morphism."""
        digits = [int(d) for d in number_str]
        result = []
        carry = False

        # Right-to-left: ones first.
        # Ones position always gets +1 (the succession operation).
        for i in range(len(digits) - 1, -1, -1):
            d = digits[i]
            if i == len(digits) - 1:
                # Ones position: apply successor (+1)
                out_d, carry = SuccessionEngine.SUCC[(d, False)]
            else:
                # Higher positions: echo unless carry
                if carry:
                    out_d, carry = SuccessionEngine.SUCC[(d, False)]
                    # Note: carry_in is handled by doing +1 via SUCC
                    # This is equivalent to (d + carry_in) mod 10
                else:
                    out_d = d
                    carry = False
            result.append(str(out_d))

        if carry:
            result.append("1")

        result.reverse()
        return ''.join(result)
