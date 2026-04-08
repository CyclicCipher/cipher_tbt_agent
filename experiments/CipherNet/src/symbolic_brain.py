"""Symbolic Brain — domain-general cortical column system.

Succession: single column with token associations.
Vision: RetinotopicV1 + FovealExplorer (in visual_cortex.py).
Both use the same SymbolicColumn with SDR feature-location bindings.

The Brain handles the output cortex (WTA for final token selection)
and coordinates between modules.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph import Graph
from prior_loader import load_priors
from token_io import TokenIO
from symbolic_column import SymbolicColumn, SuccessionEngine


class SymbolicBrain:
    """Brain with symbolic cortical columns."""

    def __init__(self):
        self.graph, self.priors = load_priors()
        self.tio = TokenIO(self.graph, self.priors)
        self.succession = SymbolicColumn("succession")

        self._output_nodes: dict[str, int] = {}
        for key, nid in self.priors.get('output_cortex', {}).items():
            node = self.graph.get_node(nid)
            if node and node.meta.get('token'):
                self._output_nodes[node.meta['token']] = nid

    def train_succession(self, pairs: list[tuple[str, str]]):
        for token, next_token in pairs:
            self.succession.learn(next_token, feature=None, location=None)
            # For succession: store token → next association.
            # Using the column's model system: object_id = next_token,
            # feature = SDR(token), location = origin.
            from sdr import SDR, SDREncoder
            enc = getattr(self, '_succ_enc', None)
            if enc is None:
                enc = SDREncoder(n=128, w=8)
                self._succ_enc = enc
            feat = enc.encode(token)
            self.succession.observe(feat)
            self.succession.learn(next_token)

    def predict_successor(self, token: str) -> str | None:
        enc = getattr(self, '_succ_enc', None)
        if enc is None:
            return None
        feat = enc.encode(token)
        self.succession.observe(feat)
        return self.succession.recognized

    def predict_number_successor(self, number_str: str) -> str:
        return SuccessionEngine.successor(number_str)

    def read_output(self, prediction: str | None) -> tuple[str | None, bool]:
        if prediction is None:
            return None, False
        self.tio.clear_output()
        nid = self._output_nodes.get(prediction)
        if nid is not None:
            self.graph.activate(nid, 1.0)
            self.graph.step()
        token, act = self.tio.read_output()
        return token, act > 0.01
