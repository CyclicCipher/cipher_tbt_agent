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
    ACTIVATION_THRESHOLD, IDENTITY, CONTEXT,
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
    # Surprise threshold for working memory entry.
    WM_SURPRISE_THRESHOLD: float = 0.5

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
        self._prev_context_nid: NodeId | None = None  # last n-gram context node
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
        current_nids: list[NodeId] = []
        actual: dict[NodeId, float] = {}
        tokenizer = self.tokenizer
        for tok in tokens:
            if tokenizer is not None:
                tok_id = tokenizer.encode_token(tok)
                nid = self.kg.get_or_create(tok_id)
            else:
                nid = self.kg.get_or_create(tok)
            self.kg.activate(nid, 1.0)
            current_nids.append(nid)
            actual[nid] = 1.0

        # Step 3.5: context is the n-gram — no pattern scan needed.
        # The n-gram context node (created below in step 4.5) IS the context.
        # No O(N) scan of __context__ pattern nodes. Context = hash lookup.

        # Step 4: co-occurrence edges — context node → current tokens only.
        # NOT all-pairs. The n-gram context captures the sequential context;
        # individual identity tokens don't need pairwise edges.
        # (Co-occurrence edges between tokens in the same fixation are created
        # only for consecutive pairs, not all pairs.)
        unique_nids = list(dict.fromkeys(current_nids))
        COOCCUR_RATE = 0.15
        for i in range(len(unique_nids) - 1):
            src, tgt = unique_nids[i], unique_nids[i + 1]
            if src != tgt:
                edge = self.kg.get_or_create_edge(src, tgt, role=COOCCURRENCE)
                edge.strengthen(COOCCUR_RATE)
                edge.observe_distance(1)

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
        # This is the ONLY source of transition edges per timestep.
        # The context node IS the sequential context — one edge per
        # (context, next_token) pair, not all-pairs from prev_active.
        ctx_nid_current = None
        if len(self._recent_tokens) >= self.NGRAM_SIZE:
            ctx_key = ("__ngram__", tuple(self._recent_tokens[-(self.NGRAM_SIZE-1):]))
            ctx_nid_current = self.kg.get_or_create(ctx_key, layer=CONTEXT)
            self.kg.activate(ctx_nid_current, 1.0)
            # The current token is what follows this context.
            for nid in current_nids:
                edge = self.kg.get_or_create_edge(ctx_nid_current, nid, role=TRANSITION)
                edge.strengthen(0.15)

        # Step 5: transition edge from previous context to current context.
        # Instead of all-prev_active × all-new_nids (O(N²)), create ONE
        # edge: prev_context → current_context (context chain).
        # Plus: prev_context → current tokens (for cross-context prediction).
        is_action_obs = edge_types and len(edge_types) > 0 and edge_types[0] == 2
        if self._prev_context_nid is not None and not is_action_obs:
            if ctx_nid_current is not None and ctx_nid_current != self._prev_context_nid:
                # Context-to-context transition (displacement).
                edge = self.kg.get_or_create_edge(
                    self._prev_context_nid, ctx_nid_current, role=TRANSITION)
                edge.strengthen(0.1)
            # Previous context → current identity tokens.
            for nid in current_nids:
                edge = self.kg.get_or_create_edge(
                    self._prev_context_nid, nid, role=TRANSITION)
                edge.strengthen(0.1)

        # Step 6: learn from the PENDING prediction.
        if self._pending_prediction:
            self._last_surprise = self.kg.learn(
                self._pending_prediction, actual,
                prev_active=self._prev_active if self._prev_active else None,
            )

        # Step 6a: surprise-gated working memory.
        # High-surprise tokens (unpredictable content like date digits) get
        # held in WM so their activation survives across long template spans.
        # Low-surprise tokens (predictable template text) decay normally.
        if self._last_surprise > self.WM_SURPRISE_THRESHOLD:
            for nid in current_nids:
                node = self.kg.node(nid)
                if node is not None and node.layer == IDENTITY:
                    self.kg.wm_hold(nid)

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
                node = self.kg.node(nid)
                if node is not None:
                    new_act = max(node.activation, min(1.0, level * 0.5))
                    self.kg.activate(nid, new_act)

        # Step 7b: spread TRANSITION edges (= predict what comes NEXT).
        self._pending_prediction = self.kg.spread(role_filter=TRANSITION)
        for nid in current_nids:
            self._pending_prediction.pop(nid, None)

        # Step 8: store snapshot + observation record.
        if self._suppress_observation:
            self.hippo.store(self.kg.active_nodes(), observed_nids=None)
        else:
            self.hippo.store(self.kg.active_nodes(), observed_nids=current_nids)

        # Track the current context node for next timestep's transitions.
        self._prev_context_nid = ctx_nid_current
        self._prev_active = dict(actual)
        if ctx_nid_current is not None:
            self._prev_active[ctx_nid_current] = 1.0
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
        # Release WM from previous sentence — each sentence is independent.
        self.kg.wm_release_all()

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
                chunk_nid = self.kg.get_or_create(chunk_key, layer=CONTEXT)
                self.kg.activate(chunk_nid, 1.0)
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
