"""Symbolic cortical column — TBT column as a data structure.

Each column maintains a reference frame (location in a category),
stores feature-location associations (one-shot), and predicts via
O(1) lookup. No neurons, no gradient descent, no epochs.

The displacement algebra is category-theoretic: location is an object,
displacement is a morphism, path integration is composition. Different
domains use different categories (Z for sequences, Z² for space, etc.)
but the column protocol is the same.

Biology: implements the FUNCTION of a cortical column (predict, observe,
error, displace, vote) without simulating individual neurons. Matches
TBT: each column has a reference frame, learns through sensorimotor
experience, and votes with neighbors for consensus.
"""
from __future__ import annotations
from typing import Any


class SymbolicColumn:
    """A cortical column that stores feature-location associations.

    Reference frame = a dict mapping locations to features.
    Prediction = dict lookup at current location.
    Learning = dict write (one-shot).
    Displacement = update location (path integration).
    """

    def __init__(self, name: str, position: tuple = (0.0, 0.0, 0.0)):
        self.name = name
        self.location: str = ""          # current position in reference frame
        self.memory: dict[str, str] = {} # {location → feature}
        self.prediction: str | None = None
        self.error: bool = False
        self.observed: str | None = None
        self.output_vote: str | None = None
        self.position = position         # spatial position for broadcast
        self.phase: float = 0.0          # theta phase (sequence position)
        self.gated: bool = False         # BG control

    def observe(self, feature: str) -> bool:
        """Observe a feature at current location. Learn if surprised."""
        self.observed = feature
        self.prediction = self.memory.get(self.location)
        self.error = (feature != self.prediction)
        if self.error and feature is not None:
            self.memory[self.location] = feature
        self.output_vote = feature
        return self.error

    def displace(self, morphism: Any) -> None:
        """Move location by applying a displacement morphism.

        For integers: location = str(int(location) + morphism).
        Override for other categories.
        """
        try:
            self.location = str(int(self.location) + morphism)
        except (ValueError, TypeError):
            # Non-integer location: treat morphism as string concatenation key
            self.location = f"{self.location}:{morphism}"
        self.prediction = self.memory.get(self.location)

    def predict(self) -> str | None:
        """Return predicted feature at current location."""
        return self.memory.get(self.location)

    def vote(self) -> str | None:
        """Return this column's output for downstream consumption."""
        return self.output_vote

    def reset(self):
        """Reset transient state. Memory is preserved."""
        self.location = ""
        self.prediction = None
        self.error = False
        self.observed = None
        self.output_vote = None
        self.phase = 0.0

    def __repr__(self):
        return (f"SymbolicColumn({self.name}, loc={self.location}, "
                f"mem={len(self.memory)} entries)")


class SuccessionColumn(SymbolicColumn):
    """Column that learns token succession (next token prediction).

    Location = current token. Memory maps current → next.
    feed("3") sets location="3". predict() returns memory["3"].
    teach("4") writes memory["3"] = "4".
    """

    def feed(self, token: str):
        """Set location to the given token."""
        self.location = token
        self.prediction = self.memory.get(self.location)

    def teach(self, next_token: str):
        """One-shot: store current_token → next_token."""
        self.memory[self.location] = next_token

    def predict_next(self) -> str | None:
        """Predict what comes after the current token."""
        return self.memory.get(self.location)


class PlaceValueColumn(SymbolicColumn):
    """Column that handles one digit position with carry logic.

    Uses the SUCCESSOR MORPHISM (+1 mod 10) as the displacement.
    This is the category Z/10Z (integers mod 10). The displacement
    is universal — it works for ANY digit, not just memorized pairs.

    The column learns ONE thing: the displacement operation (+1 mod 10).
    From a single example of "3+1=4", it can derive "7+1=8" because
    the operation is the SAME morphism applied at a different location.
    """

    def __init__(self, name: str, position: tuple = (0.0, 0.0, 0.0)):
        super().__init__(name, position)
        # Pre-load the successor morphism for digits 0-9.
        # This is the Z/10Z category: +1 mod 10.
        # In a fully general system, this would be DISCOVERED by the
        # RelationalLearner. For now, it's the innate "counting" ability
        # that even infants have (subitizing → successor).
        for d in range(10):
            next_d = (d + 1) % 10
            carry = (d + 1) >= 10
            self.memory[f"{d},0"] = f"{next_d},{'1' if carry else '0'}"
            # With carry-in: digit + 1 (carry) = (d+1) mod 10
            next_d_carry = (d + 2) % 10
            carry_out = (d + 2) >= 10
            self.memory[f"{d},1"] = f"{next_d_carry},{'1' if carry_out else '0'}"

    def feed_digit(self, digit: str, carry_in: bool = False):
        """Set location from digit + carry state."""
        self.location = f"{digit},{'1' if carry_in else '0'}"
        self.prediction = self.memory.get(self.location)

    def teach_digit(self, output_digit: str, carry_out: bool = False):
        """One-shot: store (digit, carry) → (output, carry_out)."""
        self.memory[self.location] = f"{output_digit},{'1' if carry_out else '0'}"

    def predict_digit(self) -> tuple[str | None, bool]:
        """Predict output digit and carry flag."""
        result = self.memory.get(self.location)
        if result is None:
            return None, False
        parts = result.split(',')
        return parts[0], parts[1] == '1' if len(parts) > 1 else False
