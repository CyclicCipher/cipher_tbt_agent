"""
AgenticLoop — the universal engine.

The only door between the environment and the knowledge graph. Every
environment, test, and benchmark interfaces through this class.

observe() takes tokens from the environment (via tokenizer).
step() spreads activation and learns from prediction error.
act() selects the most activated action candidate.

There are no pairs, no Graph objects, no InputOutputTopology.
Tokens fire nodes. Co-occurrence creates edges. That's it.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, COOCCURRENCE, TRANSITION,
    ACTIVATION_THRESHOLD,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


class AgenticLoop:
    """
    The universal observe → spread → learn → act engine.

    Parameters
    ----------
    kg : KnowledgeGraph
        The one graph.
    tokenizer : optional
        If provided, called on raw strings to get opaque IDs.
        If None, values are used as-is (already opaque).
    """

    # Default: consolidate every 100 observe() calls. Set to 0 to disable.
    CONSOLIDATION_INTERVAL: int = 100

    def __init__(self, kg: KnowledgeGraph, tokenizer=None) -> None:
        self.kg = kg
        self.tokenizer = tokenizer
        self.hippo = Hippocampus(max_episodes=5000)
        self._step_count: int = 0
        self._pending_prediction: dict[NodeId, float] = {}
        self._last_actual: dict[NodeId, float] = {}
        self._last_positions: dict[NodeId, int] = {}
        self._last_surprise: float = 0.0
        self._prev_active: dict[NodeId, float] = {}
        self._last_action_nid: NodeId | None = None
        self._last_action_context: dict[NodeId, float] = {}
        self._last_consolidation_step: int = 0
        self._last_consolidation_snapshot: int = 0
        self._suppress_observation: bool = False
        # Sliding window of recent token NodeIds for n-gram context creation.
        self._recent_tokens: list[NodeId] = []
        self.NGRAM_SIZE: int = 4  # context window size

    # -------------------------------------------------------------------
    # Homeostatic priors
    # -------------------------------------------------------------------

    def set_preferred(self, token: str, level: float) -> None:
        """Set a homeostatic prior for a token."""
        if self.tokenizer is not None:
            tok_id = self.tokenizer.encode_token(token)
            nid = self.kg.get_or_create(tok_id)
        else:
            nid = self.kg.get_or_create(token)
        self.kg.set_preferred(nid, level)

    # -------------------------------------------------------------------
    # Observe
    # -------------------------------------------------------------------

    def observe(self, tokens: list[str], edge_types: list[int | None] | None = None) -> None:
        """
        Feed one timestep of observation into the graph.

        Parameters
        ----------
        tokens : list[str]
            Raw token strings from the environment.
        edge_types : list[int | None], optional
            Per-token edge type: None (start), 0 (extero), 1 (intero), 2 (action).
        """
        if edge_types is None:
            edge_types = [None] + [COOCCURRENCE] * (len(tokens) - 1)

        # Step 1: decay
        self.kg.decay()

        # Step 2–3: resolve tokens to nodes, activate them.
        nodes = self.kg._nodes
        current_nids: list[NodeId] = []
        actual: dict[NodeId, float] = {}
        tokenizer = self.tokenizer
        for tok in tokens:
            if tokenizer is not None:
                tok_id = tokenizer.encode_token(tok)
                nid = self.kg.get_or_create(tok_id)
            else:
                nid = self.kg.get_or_create(tok)
            node = nodes[nid]
            node.activation = 1.0
            node.resting = min(1.0, node.resting + 0.01)
            current_nids.append(nid)
            actual[nid] = 1.0

        # Step 3.5: activate context nodes (layer 1).
        # A context node fires when ALL its constituent identity nodes are
        # active. Only check __context__ pattern nodes (not chunks/ngrams).
        # Uses _node_to_value for O(1) value lookup.
        from experiments.symbolic_ai_v2.ctkg.logic.graph import CONTEXT, IDENTITY, ACTIVATION_THRESHOLD as _AT
        active_ids = set(
            nid for nid, node in nodes.items()
            if node.activation >= _AT and node.layer == IDENTITY
        )
        if active_ids:
            for ctx_nid in self.kg.nodes_by_layer(CONTEXT):
                val = self.kg._node_to_value.get(ctx_nid)
                if not isinstance(val, tuple) or len(val) < 2:
                    continue
                if val[0] != "__context__":
                    continue
                pattern = val[1]
                if isinstance(pattern, frozenset) and pattern.issubset(active_ids):
                    nodes[ctx_nid].activation = 1.0

        # Step 4: co-occurrence edges (causal masking: forward-only).
        all_nids = list(dict.fromkeys(current_nids))
        COOCCUR_RATE = 0.15
        unique_nids = list(dict.fromkeys(all_nids))
        n = len(unique_nids)
        for i in range(n):
            for j in range(i + 1, n):
                src, tgt = unique_nids[i], unique_nids[j]
                if src == tgt:
                    continue
                edge = self.kg.get_or_create_edge(src, tgt, role=COOCCURRENCE)
                edge.strengthen(COOCCUR_RATE)
                edge.observe_distance(j - i)

        # Step 4.5: N-gram context learning.
        # If recent_tokens has enough history, create an n-gram context node
        # and a transition edge from that context to the current token.
        # recent_tokens is managed by read() (which decides chunking),
        # NOT by observe() directly. When observe() is called outside of
        # read(), recent_tokens accumulates identity nodes.
        if not self._suppress_observation:
            # Outside read(): append identity nids to recent_tokens.
            for nid in current_nids:
                self._recent_tokens.append(nid)
            if len(self._recent_tokens) > self.NGRAM_SIZE:
                self._recent_tokens = self._recent_tokens[-self.NGRAM_SIZE:]

        # Create/strengthen n-gram context → current token transition.
        if len(self._recent_tokens) >= self.NGRAM_SIZE:
            ctx_key = ("__ngram__", tuple(self._recent_tokens[-(self.NGRAM_SIZE-1):]))
            ctx_nid = self.kg.get_or_create(ctx_key, layer=CONTEXT)
            ctx_node = self.kg.node(ctx_nid)
            if ctx_node is not None:
                ctx_node.activation = 1.0
            # The current token is what follows this context.
            for nid in current_nids:
                edge = self.kg.get_or_create_edge(ctx_nid, nid, role=TRANSITION)
                edge.strengthen(0.15)

        # Step 5: transition edges from previous timestep.
        is_action_obs = edge_types and len(edge_types) > 0 and edge_types[0] == 2
        if self._prev_active and not is_action_obs:
            cur_set = set(current_nids)
            prev_set = set(self._prev_active)
            new_nids = cur_set - prev_set
            for prev_nid in self._prev_active:
                for cur_nid in new_nids:
                    edge = self.kg.get_or_create_edge(prev_nid, cur_nid, role=TRANSITION)
                    edge.strengthen(0.1)

        # Step 6: learn from the PENDING prediction.
        if self._pending_prediction:
            self._last_surprise = self.kg.learn(
                self._pending_prediction, actual,
                prev_active=self._prev_active if self._prev_active else None,
            )

        if is_action_obs and self._last_actual:
            merged = dict(self._last_actual)
            merged.update(actual)
            self._last_actual = merged
        else:
            self._last_actual = actual

        # PoPE: record positions.
        is_action_obs = any(et == 2 for et in edge_types if et is not None)
        if is_action_obs and self._last_positions:
            next_pos = max(self._last_positions.values()) + 1
            for i, nid in enumerate(current_nids):
                self._last_positions[nid] = next_pos + i
        else:
            self._last_positions = {nid: i for i, nid in enumerate(current_nids)}

        # Step 6b: retrospective action credit via predictive coding.
        if self._last_action_nid is not None and self._last_action_context:
            preferred = self.kg.preferred_nodes()
            credit = 0.0
            for nid in current_nids:
                pref = preferred.get(nid, 0.0)
                credit += pref

            if credit != 0.0:
                action_nid = self._last_action_nid
                strength = min(abs(credit), 3.0) * 0.1

                for ctx_nid in self._last_action_context:
                    edge = self.kg.edge(ctx_nid, action_nid)
                    if edge is not None and edge.role == TRANSITION:
                        if credit > 0:
                            edge.strengthen(strength)
                        else:
                            edge.weaken(strength)

                back_strength = strength * 0.3
                for ctx_nid in self._last_action_context:
                    for edge in self.kg.edges_to(ctx_nid):
                        if edge.role != TRANSITION:
                            continue
                        if edge.effective_weight < 0.01:
                            continue
                        if credit > 0:
                            edge.strengthen(back_strength)
                        else:
                            edge.weaken(back_strength)

        # Step 7a: co-occurrence spread (activate associated nodes).
        cooccur_spread = self.kg.spread(role_filter=COOCCURRENCE)
        for nid, level in cooccur_spread.items():
            if level > 0:
                node = nodes.get(nid)
                if node is not None:
                    node.activation = max(node.activation, min(1.0, level * 0.5))

        # Step 7b: spread TRANSITION edges (= predict what comes NEXT).
        self._pending_prediction = self.kg.spread(role_filter=TRANSITION)
        for nid in current_nids:
            self._pending_prediction.pop(nid, None)

        # Step 8: store snapshot + observation record.
        # During read(), observation storage is suppressed (read() stores
        # one observation for the full sentence at the end).
        if self._suppress_observation:
            self.hippo.store(self.kg.active_nodes(), observed_nids=None)
        else:
            self.hippo.store(self.kg.active_nodes(), observed_nids=current_nids)

        # Include the most recently created n-gram context in prev_active.
        # Only ONE context node (the current n-gram), not all active contexts.
        # This keeps transition edge creation O(1) per context.
        prev = dict(actual)
        if len(self._recent_tokens) >= self.NGRAM_SIZE - 1:
            ctx_key = ("__ngram__", tuple(self._recent_tokens[-(self.NGRAM_SIZE-1):]))
            ctx_nid = self.kg._value_to_node.get(ctx_key)
            if ctx_nid is not None:
                prev[ctx_nid] = 1.0
        self._prev_active = prev
        self._step_count += 1

        # Step 9: periodic consolidation.
        if (self.CONSOLIDATION_INTERVAL > 0
                and self._step_count - self._last_consolidation_step
                    >= self.CONSOLIDATION_INTERVAL):
            self.consolidate()

    # -------------------------------------------------------------------
    # Act
    # -------------------------------------------------------------------

    def act(self, candidates: list[str]) -> str | None:
        """Select an action via co-occurrence attention.

        Returns the best candidate, or None if no candidates exist.
        """
        if not candidates:
            return None

        candidate_nids: list[NodeId] = []
        nid_to_token: dict[NodeId, str] = {}
        for tok in candidates:
            if self.tokenizer is not None:
                tok_id = self.tokenizer.encode_token(tok)
                nid = self.kg.get_or_create(tok_id)
            else:
                nid = self.kg.get_or_create(tok)
            candidate_nids.append(nid)
            nid_to_token[nid] = tok

        fixed_context = dict(self._last_actual) if self._last_actual else None
        context_positions = dict(self._last_positions) if self._last_positions else None
        answer_pos = None
        if context_positions:
            answer_pos = max(context_positions.values()) + 1

        best_nid = self.kg.select_action(
            candidate_nids,
            fixed_context=fixed_context,
            context_positions=context_positions,
            answer_position=answer_pos,
        )
        if best_nid is None:
            return None

        self._last_action_nid = best_nid
        self._last_action_context = dict(self._last_actual) if self._last_actual else {}
        return nid_to_token.get(best_nid)

    # -------------------------------------------------------------------
    # Read (adaptive fixation — processes a character sequence)
    # -------------------------------------------------------------------

    # Fixation parameters.
    MIN_FIXATION: int = 1
    MAX_FIXATION: int = 15       # perceptual span ~15 chars to the right
    SURPRISE_THRESHOLD: float = 2.0  # above this → shrink fixation

    def read(self, characters: list[str]) -> None:
        """Process a character sequence with adaptive fixation.

        Mimics human reading: each fixation takes in a variable number
        of characters. Fixation size starts at MIN_FIXATION and adapts
        based on prediction surprise:
        - Low surprise (familiar/predicted content) → grow fixation
        - High surprise (novel content) → shrink fixation

        Each fixation stores a SNAPSHOT (activation pattern for context
        node discovery) but NOT an observation record. After the full
        sequence, ONE observation record is stored for the complete
        token list (for FCA and other observation-based discovery).

        Snapshots per fixation = fine-grained activation patterns.
        One observation per sentence = the full episode.
        """
        pos = 0
        fixation_size = self.MIN_FIXATION
        n = len(characters)

        all_nids: list[NodeId] = []
        self._suppress_observation = True

        while pos < n:
            end = min(pos + fixation_size, n)
            chunk = characters[pos:end]

            self.observe(chunk)

            # Resolve chunk characters to NodeIds.
            chunk_nids: list[NodeId] = []
            for tok in chunk:
                if self.tokenizer is not None:
                    tok_id = self.tokenizer.encode_token(tok)
                    nid = self.kg.get_or_create(tok_id)
                else:
                    nid = self.kg.get_or_create(tok)
                chunk_nids.append(nid)
                all_nids.append(nid)

            # Adapt fixation size based on surprise.
            surprise = self._last_surprise

            # Online chunk recognition: when the fixation covered multiple
            # characters with low surprise, those characters form a chunk.
            # The chunk node replaces the individual characters in
            # recent_tokens, giving hierarchical context.
            if len(chunk_nids) > 1 and surprise < self.SURPRISE_THRESHOLD:
                # Low surprise multi-char fixation = recognized chunk.
                chunk_key = ("__ngram__", tuple(chunk_nids))
                from experiments.symbolic_ai_v2.ctkg.logic.graph import CONTEXT as _CTX
                chunk_nid = self.kg.get_or_create(chunk_key, layer=_CTX)
                chunk_node = self.kg.node(chunk_nid)
                if chunk_node is not None:
                    chunk_node.activation = 1.0
                self._recent_tokens.append(chunk_nid)
            else:
                # High surprise or single char: add individual tokens.
                self._recent_tokens.extend(chunk_nids)

            # Keep recent_tokens bounded.
            if len(self._recent_tokens) > self.NGRAM_SIZE * 2:
                self._recent_tokens = self._recent_tokens[-self.NGRAM_SIZE * 2:]

            if surprise > self.SURPRISE_THRESHOLD:
                fixation_size = max(self.MIN_FIXATION, fixation_size - 1)
            else:
                fixation_size = min(self.MAX_FIXATION, fixation_size + 1)

            pos = end

        # Restore observation storage and store ONE record for the full read.
        self._suppress_observation = False
        self.hippo.store(self.kg.active_nodes(), observed_nids=all_nids)

    # -------------------------------------------------------------------
    # Predict next token from n-gram context
    # -------------------------------------------------------------------

    def predict_next(self) -> str | None:
        """Predict the next character from the n-gram context.

        Looks up the current n-gram context (last N-1 tokens in
        recent_tokens, which may include chunk nodes). Finds the
        transition edge from that context with the highest weight
        to an identity node. Returns the predicted character, or None.
        """
        from experiments.symbolic_ai_v2.ctkg.logic.graph import TRANSITION, IDENTITY

        if len(self._recent_tokens) < self.NGRAM_SIZE - 1:
            return None

        # Try progressively shorter context windows.
        for ctx_len in range(self.NGRAM_SIZE - 1, 0, -1):
            ctx_key = ("__ngram__", tuple(self._recent_tokens[-ctx_len:]))
            ctx_nid = self.kg._value_to_node.get(ctx_key)
            if ctx_nid is None:
                continue

            # Find best transition to an identity node.
            best_nid = None
            best_w = 0.0
            for edge in self.kg._outgoing.get(ctx_nid, []):
                if edge.role != TRANSITION:
                    continue
                if edge.weight <= best_w:
                    continue
                tgt = edge.target
                if tgt in self.kg._nodes and self.kg._nodes[tgt].layer == IDENTITY:
                    best_w = edge.weight
                    best_nid = tgt

            if best_nid is not None:
                return self.kg.label_for_node(best_nid)

        return None

    # -------------------------------------------------------------------
    # Consolidation (the slow path, callable through the loop)
    # -------------------------------------------------------------------

    def consolidate(self, replay_passes: int = 1) -> dict:
        """Run consolidation: replay → prune → colimits → morphisms.

        Incremental: only processes snapshots/observations since the
        last consolidation.
        """
        from experiments.symbolic_ai_v2.ctkg.logic import Consolidation as consolidation
        since = self._last_consolidation_snapshot
        stats = consolidation.consolidate(
            self.kg, self.hippo, replay_passes=replay_passes,
            since_index=since,
        )
        self._last_consolidation_step = self._step_count
        self._last_consolidation_snapshot = len(self.hippo.all_snapshots())
        return stats

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_surprise(self) -> float:
        return self._last_surprise

    @property
    def last_predicted(self) -> dict[NodeId, float]:
        return dict(self._pending_prediction)

    @property
    def last_actual(self) -> dict[NodeId, float]:
        return dict(self._last_actual)
