"""Symbolic Brain — cortical columns as data structures.

Wraps the existing Graph (for output cortex WTA and BG gating)
with symbolic columns that compute via dict lookup instead of
neural simulation. Learning is one-shot, prediction is O(1).

The old neural Brain is preserved — this is additive, not destructive.

Future: the RelationalLearner (experiments/symbolic_ai/) discovers
the category structure (reference frame morphisms) from raw data.
Currently the column types are manually specified.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph import Graph
from prior_loader import load_priors
from token_io import TokenIO
from symbolic_column import (
    SymbolicColumn, SuccessionColumn, PlaceValueColumn,
)


class SymbolicBrain:
    """Brain with symbolic cortical columns.

    Symbolic columns handle prediction and learning (one-shot dicts).
    The Graph handles output cortex WTA and BG gating (kept from priors).
    """

    def __init__(self):
        # Load only output_cortex and basal_ganglia from priors.
        # The rest is handled by symbolic columns.
        self.graph, self.priors = load_priors()
        self.tio = TokenIO(self.graph, self.priors)
        self.columns: dict[str, SymbolicColumn] = {}

        # Build output token map for fast lookup.
        self._output_nodes: dict[str, int] = {}
        for key, nid in self.priors.get('output_cortex', {}).items():
            node = self.graph.get_node(nid)
            if node and node.meta.get('token'):
                self._output_nodes[node.meta['token']] = nid

    def add_column(self, name: str, column: SymbolicColumn):
        """Register a symbolic column."""
        self.columns[name] = column

    def reset(self):
        """Reset all columns' transient state (keep memory)."""
        for col in self.columns.values():
            col.reset()

    # ----- Succession interface -----

    def feed(self, token: str):
        """Feed a token to all succession columns."""
        for col in self.columns.values():
            if isinstance(col, SuccessionColumn):
                col.feed(token)

    def teach(self, next_token: str):
        """Teach all succession columns: current → next."""
        for col in self.columns.values():
            if isinstance(col, SuccessionColumn):
                col.teach(next_token)

    def predict(self) -> str | None:
        """Get the succession prediction from the first succession column."""
        for col in self.columns.values():
            if isinstance(col, SuccessionColumn):
                return col.predict_next()
        return None

    def read_output(self) -> tuple[str | None, bool]:
        """Drive the output cortex with the column prediction.

        Activates the predicted output node, runs WTA, returns winner.
        """
        prediction = self.predict()
        if prediction is None:
            return None, False

        # Clear output cortex activations.
        self.tio.clear_output()

        # Activate the predicted output node.
        nid = self._output_nodes.get(prediction)
        if nid is not None:
            self.graph.activate(nid, 1.0)
            # Run one step for WTA resolution.
            self.graph.step()

        # Read the winner.
        token, act = self.tio.read_output()
        return token, act > 0.01

    # ----- Multi-digit interface -----

    def feed_number(self, number_str: str):
        """Feed a multi-digit number for succession prediction.

        Succession = +1 to the whole number. This means:
        - Ones position: always gets +1 (the successor operation)
        - Other positions: ECHO (pass through) unless carry arrives
        - If carry propagates past the most significant digit: prepend "1"

        Uses PlaceValueColumn for ones (applies +1 mod 10).
        Other positions: simple echo with carry addition.
        """
        digits = list(number_str)
        self._digit_results = []

        # Process right-to-left.
        # Ones position (i=0): apply successor (+1 via PlaceValueColumn)
        ones_col_name = "place_0"
        ones_col = self.columns.get(ones_col_name)
        if ones_col is None:
            ones_col = PlaceValueColumn(ones_col_name)
            self.add_column(ones_col_name, ones_col)

        ones_digit = digits[-1]
        ones_col.feed_digit(ones_digit, carry_in=False)
        out_digit, carry = ones_col.predict_digit()
        self._digit_results.append(out_digit)

        # Higher positions: echo + carry propagation.
        for i in range(len(digits) - 2, -1, -1):
            d = int(digits[i])
            if carry:
                d += 1
                carry = d >= 10
                d = d % 10
            self._digit_results.append(str(d))

        # Leading carry: prepend "1"
        if carry:
            self._digit_results.append("1")

        # Reverse to left-to-right order.
        self._digit_results.reverse()

    def teach_number(self, input_str: str, output_str: str):
        """Teach multi-digit succession with carry.

        The PlaceValueColumn already knows the Z/10Z successor morphism
        (+1 mod 10 with carry) from its pre-loaded memory. This method
        is only needed if the column encounters novel mappings outside
        the standard successor. For standard succession, the pre-loaded
        memory handles everything — this is a no-op.
        """
        # The PlaceValueColumn.__init__ pre-loads all (digit, carry) → (output, carry_out)
        # mappings. No additional teaching needed for standard succession.
        # The columns auto-create in feed_number when needed.
        pass

    def read_number(self) -> str:
        """Read the multi-digit prediction."""
        if not hasattr(self, '_digit_results'):
            return "?"
        result = ''.join(d if d else '?' for d in self._digit_results)
        # Strip leading zeros
        result = result.lstrip('0') or '0'
        return result


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    brain = SymbolicBrain()

    # Add a succession column
    succ = SuccessionColumn("succession")
    brain.add_column("succession", succ)

    # Train: one pass through 9 examples
    print("=== Training succession (1 pass) ===")
    for d in range(9):
        brain.feed(str(d))
        brain.teach(str(d + 1))
    print(f"  Memory: {succ.memory}")

    # Test
    print("\n=== Testing succession ===")
    correct = 0
    for d in range(9):
        brain.feed(str(d))
        token, ok = brain.read_output()
        expected = str(d + 1)
        tag = "OK" if token == expected else "X"
        print(f"  {d} -> {token} (expected {expected}) [{tag}]")
        if token == expected:
            correct += 1
    print(f"  Result: {correct}/9")
