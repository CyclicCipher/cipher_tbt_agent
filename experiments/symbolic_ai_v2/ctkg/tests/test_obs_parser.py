"""Tests for ctkg/core/obs_parser.py (Stage 3 — Observation Parser)."""

from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from experiments.symbolic_ai_v2.ctkg.core.obs_parser import ObsParser
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_creates_empty_graph(self):
        p = ObsParser()
        assert len(p.graph.objects()) == 0
        assert len(p.graph.morphisms()) == 0

    def test_accepts_existing_graph(self):
        mg = MorphismGraph()
        p = ObsParser(mg)
        assert p.graph is mg


# ---------------------------------------------------------------------------
# encode / decode (no graph side-effects)
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_encode_sequence(self):
        p = ObsParser()
        nids = p.encode_sequence(['succ', '5', 'eq'])
        assert len(nids) == 3
        assert all(isinstance(n, int) for n in nids)
        # Graph should be untouched
        assert p.n_tokens() == 0

    def test_decode_roundtrip(self):
        p = ObsParser()
        tokens = ['succ', '5', 'eq', '6']
        nids = p.encode_sequence(tokens)
        recovered = p.decode_sequence(nids)
        assert recovered == tokens

    def test_single_char_tokens_are_ascii_nodeids(self):
        p = ObsParser()
        nids = p.encode_sequence(['5'])
        assert nids[0] == ord('5')

    def test_encode_empty(self):
        p = ObsParser()
        assert p.encode_sequence([]) == []

    def test_decode_empty(self):
        p = ObsParser()
        assert p.decode_sequence([]) == []


# ---------------------------------------------------------------------------
# observe_sequence — graph side-effects
# ---------------------------------------------------------------------------

class TestObserveSequence:
    def test_returns_nodeids(self):
        p = ObsParser()
        nids = p.observe_sequence(['a', 'b', 'c'])
        assert len(nids) == 3

    def test_creates_objects_for_each_token(self):
        p = ObsParser()
        p.observe_sequence(['succ', '5', 'eq'])
        assert p.n_tokens() == 3
        assert len(p.graph.objects()) == 3

    def test_creates_bigram_edges(self):
        p = ObsParser()
        p.observe_sequence(['succ', '5', 'eq'])
        ms = p.graph.morphisms(include_identity=False)
        # Expect 2 OBS_SEQ edges: succ→5, 5→eq
        obs_edges = [m for m in ms if m.morph_type == "OBS_SEQ"]
        assert len(obs_edges) == 2

    def test_empty_sequence_no_edges(self):
        p = ObsParser()
        p.observe_sequence([])
        assert p.graph.morphisms(include_identity=False) == []

    def test_single_token_creates_object_no_edges(self):
        p = ObsParser()
        p.observe_sequence(['succ'])
        assert p.n_tokens() == 1
        assert p.graph.morphisms(include_identity=False) == []

    def test_repeated_bigram_increments_evidence(self):
        p = ObsParser()
        p.observe_sequence(['a', 'b'])
        p.observe_sequence(['a', 'b'])
        ms = p.graph.morphisms(include_identity=False)
        obs = [m for m in ms if m.morph_type == "OBS_SEQ"]
        assert len(obs) == 1  # only one edge (a→b)
        assert obs[0].evidence_count == 2

    def test_different_bigrams_are_separate_edges(self):
        p = ObsParser()
        p.observe_sequence(['a', 'b'])
        p.observe_sequence(['a', 'c'])
        ms = p.graph.morphisms(include_identity=False)
        obs = [m for m in ms if m.morph_type == "OBS_SEQ"]
        assert len(obs) == 2  # a→b and a→c

    def test_no_special_casing_eq_token(self):
        """'eq' is NOT special — it gets the same treatment as any other token."""
        p = ObsParser()
        nids1 = p.observe_sequence(['eq'])
        nids2 = p.observe_sequence(['succ'])
        # Both produce exactly one object each, with the same code path
        assert p.n_tokens() == 2
        # NodeIds differ but both are valid
        assert nids1[0] != nids2[0]

    def test_no_special_casing_digit_tokens(self):
        """Digit characters are treated identically to any other token."""
        p = ObsParser()
        p.observe_sequence(['1', '2', '3'])
        obs = [m for m in p.graph.morphisms(include_identity=False)
               if m.morph_type == "OBS_SEQ"]
        assert len(obs) == 2

    def test_anonymous_unicode_tokens(self):
        """Anonymous Unicode symbols (from I/D/A benchmark) are handled."""
        p = ObsParser()
        syms = ['\u2200', '\u2203', '\u2202']  # ∀, ∃, ∂
        nids = p.observe_sequence(syms)
        assert len(nids) == 3
        assert p.decode_sequence(nids) == syms


# ---------------------------------------------------------------------------
# observe_batch
# ---------------------------------------------------------------------------

class TestObserveBatch:
    def test_returns_nested_nodeids(self):
        p = ObsParser()
        result = p.observe_batch([['a', 'b'], ['c', 'd']])
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_accumulates_across_sequences(self):
        p = ObsParser()
        p.observe_batch([['a', 'b'], ['b', 'c']])
        obs = [m for m in p.graph.morphisms(include_identity=False)
               if m.morph_type == "OBS_SEQ"]
        assert len(obs) == 2  # a→b and b→c (no duplicate)

    def test_empty_batch(self):
        p = ObsParser()
        result = p.observe_batch([])
        assert result == []


# ---------------------------------------------------------------------------
# Inspection API
# ---------------------------------------------------------------------------

class TestInspection:
    def test_seen_node_ids_empty_initially(self):
        p = ObsParser()
        assert p.seen_node_ids() == frozenset()

    def test_seen_node_ids_after_observation(self):
        p = ObsParser()
        p.observe_sequence(['succ', '5'])
        seen = p.seen_node_ids()
        assert TOKEN_GRAPH.encode('succ') in seen
        assert TOKEN_GRAPH.encode('5') in seen

    def test_object_for_seen_token(self):
        p = ObsParser()
        p.observe_sequence(['succ'])
        oid = p.object_for('succ')
        assert oid is not None
        assert isinstance(oid, int)

    def test_object_for_unseen_token(self):
        p = ObsParser()
        assert p.object_for('succ') is None

    def test_n_tokens_counts_distinct(self):
        p = ObsParser()
        p.observe_sequence(['a', 'b', 'a'])  # 'a' appears twice
        assert p.n_tokens() == 2  # only 'a' and 'b'


# ---------------------------------------------------------------------------
# Graph isolation
# ---------------------------------------------------------------------------

class TestGraphIsolation:
    def test_two_parsers_on_different_graphs(self):
        p1 = ObsParser()
        p2 = ObsParser()
        p1.observe_sequence(['x', 'y'])
        assert p2.n_tokens() == 0  # p2 graph untouched

    def test_shared_graph_sees_all_observations(self):
        mg = MorphismGraph()
        p1 = ObsParser(mg)
        p2 = ObsParser(mg)
        p1.observe_sequence(['a', 'b'])
        p2.observe_sequence(['c', 'd'])
        # Both should contribute to the shared graph
        assert len(mg.objects()) == 4
