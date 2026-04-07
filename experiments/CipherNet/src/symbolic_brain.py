"""Symbolic Brain — cortical column sheets with receptive fields.

Position is implicit in wiring (which column receives which input).
Feature is explicit in signal content (what the column observes).
Learning is one-shot (dict write). Prediction is O(1) (dict lookup).

The output cortex (from priors) provides winner-take-all token selection.
The BG (from priors) provides gating. These are kept from the old Graph.
Everything else is symbolic columns.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph import Graph
from prior_loader import load_priors
from token_io import TokenIO
from symbolic_column import SymbolicColumn, ColumnSheet, SuccessionEngine


class SymbolicBrain:
    """Brain with symbolic cortical columns and receptive fields.

    Column sheets handle topographic input mapping.
    The SuccessionEngine handles the Z/10Z displacement algebra.
    The Graph handles output cortex WTA (kept from priors).
    """

    def __init__(self):
        self.graph, self.priors = load_priors()
        self.tio = TokenIO(self.graph, self.priors)

        # Succession column: learns single-token associations.
        self.succession = SymbolicColumn("succession")

        # Output node lookup.
        self._output_nodes: dict[str, int] = {}
        for key, nid in self.priors.get('output_cortex', {}).items():
            node = self.graph.get_node(nid)
            if node and node.meta.get('token'):
                self._output_nodes[node.meta['token']] = nid

    # ----- Single-token succession -----

    def train_succession(self, pairs: list[tuple[str, str]]):
        """One-shot: learn token → next_token for all pairs."""
        for token, next_token in pairs:
            self.succession.teach(token, next_token)

    def predict_successor(self, token: str) -> str | None:
        """Predict the successor of a single token."""
        self.succession.observe(token)
        return self.succession.predict()

    # ----- Multi-digit succession -----

    def predict_number_successor(self, number_str: str) -> str:
        """Predict successor of a multi-digit number.

        Uses the Z/10Z displacement morphism with carry propagation.
        Generalizes to any number of digits (OOD by construction).
        """
        return SuccessionEngine.successor(number_str)

    # ----- Output interface -----

    def read_output(self, prediction: str | None) -> tuple[str | None, bool]:
        """Drive output cortex with a prediction, run WTA, return winner."""
        if prediction is None:
            return None, False

        self.tio.clear_output()
        nid = self._output_nodes.get(prediction)
        if nid is not None:
            self.graph.activate(nid, 1.0)
            self.graph.step()

        token, act = self.tio.read_output()
        return token, act > 0.01


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    brain = SymbolicBrain()

    # Train succession
    brain.train_succession([(str(d), str(d + 1)) for d in range(9)])
    print(f"Succession memory: {brain.succession.memory}")

    # Test single-digit
    print("\n=== Single-digit succession ===")
    for d in range(9):
        pred = brain.predict_successor(str(d))
        token, _ = brain.read_output(pred)
        print(f"  {d} -> {token}")

    # Test multi-digit
    print("\n=== Multi-digit succession ===")
    for n in ["19", "99", "999", "9999"]:
        result = brain.predict_number_successor(n)
        print(f"  {n} -> {result}")
