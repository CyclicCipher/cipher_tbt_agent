"""Character-level observation parser — Stage 3.

Converts raw token sequences into NodeId edges in the MorphismGraph.
This is the foundational observation intake entry point.

Iron Law compliance
-------------------
No string comparisons are performed on token content anywhere in this file.
All tokens are encoded to NodeId at the entry point (``observe_sequence`` /
``encode_sequence``) and only NodeIds flow downstream.  The encoder
``TOKEN_GRAPH.encode(str)`` is the ONLY place a string is consumed.

Design
------
- ``ObsParser`` wraps a MorphismGraph and a NodeId→ObjectId registry.
- Each unique token is mapped to exactly one CTKGObject (lazily created on
  first observation).
- Sequential observations (tok_i, tok_{i+1}) are added as OBS_SEQ morphisms:
  if the edge already exists its evidence_count is incremented; otherwise a
  new morphism is created with evidence=1.
- Multi-token sequences produce N-1 bigram edges (sliding window of 2).
- ``encode_sequence`` / ``decode_sequence`` give pure encoding utilities
  without side-effects on the graph.

Replaces RelationStore.update_batch()
--------------------------------------
In the full Phase XXV architecture this module is the entry point for all
incoming observation data.  RelationStore / HankelCount / other learning
components sit on top and receive NodeId sequences rather than raw strings.

For backward compatibility, the existing RelationStore.update_batch() API
continues to work; this module is the preferred new entry point.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH, NodeId
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph, ObjectId
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import DistributionalConcept


# ---------------------------------------------------------------------------
# Minimal concept factory for observed tokens
# ---------------------------------------------------------------------------

_CENTROID_DIM: int = 4  # minimal fixed dimension for obs-layer concepts


def _token_concept(node_id: NodeId) -> DistributionalConcept:
    """Create a minimal DistributionalConcept for an observed token.

    The concept has a zero centroid (observation layer concepts are not
    distributional in the FCA sense — they are structural placeholders).
    Support = 0.0 initially; incremented externally via MorphismGraph.observe().
    """
    label = TOKEN_GRAPH.decode(node_id)
    return DistributionalConcept(
        concept_id=node_id,          # NodeId doubles as concept_id for obs layer
        centroid_vector=np.zeros(_CENTROID_DIM),
        extent_weights={label: 1.0},
        intent_weights={label: 1.0},
        support=0.0,
    )


# ---------------------------------------------------------------------------
# ObsParser
# ---------------------------------------------------------------------------

class ObsParser:
    """Character-level observation parser (Stage 3).

    Encodes token sequences as NodeId sequences and writes sequential bigram
    edges into the associated MorphismGraph.  No token type is special-cased —
    all tokens are treated identically.

    Parameters
    ----------
    mg : MorphismGraph, optional
        Graph to write edges into.  A fresh empty graph is created when None.

    Attributes
    ----------
    graph : MorphismGraph
        The underlying MorphismGraph (read-only reference).
    """

    def __init__(self, mg: Optional[MorphismGraph] = None) -> None:
        self._mg: MorphismGraph = mg if mg is not None else MorphismGraph()
        # NodeId → ObjectId registry (avoids rescanning mg.objects())
        self._node_to_obj: dict[NodeId, ObjectId] = {}

    @property
    def graph(self) -> MorphismGraph:
        """The underlying MorphismGraph."""
        return self._mg

    # ------------------------------------------------------------------
    # Primary observation interface
    # ------------------------------------------------------------------

    def observe_sequence(self, tokens: list[str]) -> list[NodeId]:
        """Encode *tokens* as NodeIds and add bigram edges to the graph.

        Each consecutive pair (tok_i, tok_{i+1}) creates or increments an
        OBS_SEQ morphism from tok_i's object to tok_{i+1}'s object.

        Parameters
        ----------
        tokens : list of string tokens — NOT special-cased by type.

        Returns
        -------
        list[NodeId] — one NodeId per input token, in order.
        """
        node_ids = TOKEN_GRAPH.encode_seq(tokens)
        for i in range(len(node_ids) - 1):
            src_nid = node_ids[i]
            tgt_nid = node_ids[i + 1]
            src_obj = self._ensure_object(src_nid)
            tgt_obj = self._ensure_object(tgt_nid)
            # Increment evidence on the existing edge, or create a new one.
            existing = self._mg.hom(src_obj, tgt_obj, include_identity=False)
            if existing:
                self._mg.observe(existing[0].morph_id)
            else:
                self._mg.add_morphism(src_obj, tgt_obj,
                                      morph_type="OBS_SEQ", evidence=1)
        # Ensure isolated tokens (no neighbours) also get objects registered.
        if len(node_ids) == 1:
            self._ensure_object(node_ids[0])
        return node_ids

    def observe_batch(self, sequences: list[list[str]]) -> list[list[NodeId]]:
        """Encode and add edges for every sequence in *sequences*.

        Parameters
        ----------
        sequences : list of token sequences.

        Returns
        -------
        list of NodeId lists, one per input sequence.
        """
        return [self.observe_sequence(seq) for seq in sequences]

    # ------------------------------------------------------------------
    # Pure encoding utilities (no graph side-effects)
    # ------------------------------------------------------------------

    def encode_sequence(self, tokens: list[str]) -> list[NodeId]:
        """Encode *tokens* to NodeIds without touching the graph."""
        return TOKEN_GRAPH.encode_seq(tokens)

    def decode_sequence(self, node_ids: list[NodeId]) -> list[str]:
        """Decode NodeIds back to string tokens."""
        return TOKEN_GRAPH.decode_seq(node_ids)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def seen_node_ids(self) -> frozenset[NodeId]:
        """Return the set of NodeIds that have been observed so far."""
        return frozenset(self._node_to_obj.keys())

    def object_for(self, token: str) -> Optional[ObjectId]:
        """Return the ObjectId for *token*, or None if not yet observed."""
        nid = TOKEN_GRAPH.encode(token)
        return self._node_to_obj.get(nid)

    def n_tokens(self) -> int:
        """Number of distinct token types observed."""
        return len(self._node_to_obj)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_object(self, node_id: NodeId) -> ObjectId:
        """Return the ObjectId for *node_id*, creating a new object if absent."""
        cached = self._node_to_obj.get(node_id)
        if cached is not None:
            return cached
        concept = _token_concept(node_id)
        label = TOKEN_GRAPH.decode(node_id)
        obj = self._mg.add_object(concept, label=label)
        self._node_to_obj[node_id] = obj.obj_id
        return obj.obj_id
