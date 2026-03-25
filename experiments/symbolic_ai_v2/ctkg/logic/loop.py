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
        self.hippo = Hippocampus()
        self._step_count: int = 0
        # Prediction from the PREVIOUS spread, to be compared against
        # the NEXT observation. This is the temporal credit assignment:
        # spread at time t predicts what tokens appear at time t+1.
        self._pending_prediction: dict[NodeId, float] = {}
        self._last_actual: dict[NodeId, float] = {}
        self._last_positions: dict[NodeId, int] = {}  # PoPE: token positions
        self._last_surprise: float = 0.0
        # Previous timestep's active nodes (for transition edge creation).
        self._prev_active: dict[NodeId, float] = {}
        # Retrospective action credit: track the last action and its context
        # so FEEDBACK tokens can retroactively update the edges.
        self._last_action_nid: NodeId | None = None
        self._last_action_context: dict[NodeId, float] = {}
        # Product position tracking: how many "digit-like" tokens were in
        # the last non-action observation, and how many output digits have
        # been emitted since then. Used to compute position_from_end.
        self._input_digit_count: int = 0
        self._output_digits_emitted: int = 0
        # The positions of tokens from the ORIGINAL observation (not emitted actions).
        # Stored as a LIST to preserve duplicate positions (e.g., 5000 has three 0s
        # at positions 1, 2, 3 — all the same NodeId but different positions).
        self._original_position_list: list[tuple[NodeId, int]] = []
        # Consolidation tracking.
        self._last_consolidation_step: int = 0

    # -------------------------------------------------------------------
    # Homeostatic priors
    # -------------------------------------------------------------------

    def set_preferred(self, token: str, level: float) -> None:
        """Set a homeostatic prior for a token.

        This is called by the environment adapter to seed metabolic
        preferences. It's the ONE place where domain knowledge enters
        the system: the agent prefers certain metabolic states.

        level > 0: agent prefers this token active (e.g., ENERGY_sated)
        level < 0: agent prefers this token inactive (e.g., ENERGY_starving)
        """
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
            Raw token strings from the environment. Each becomes a node.
        edge_types : list[int | None], optional
            Per-token edge type from the environment:
              None — sequence start (first token)
              0    — extero (world-state)
              1    — intero (self-state)
              2    — action
            If not provided, all tokens are treated as co-occurring.

        The process:
        1. Decay all activations.
        2. Resolve each token to a node via tokenizer → kg.get_or_create().
        3. Activate each node.
        4. Create/strengthen co-occurrence edges between tokens in the same
           observation.
        5. Create/strengthen transition edges from the previous timestep's
           active nodes to the current timestep's active nodes.
        6. Spread activation (= predict).
        7. Learn from prediction error (= update).
        8. Store activation snapshot in Hippocampus.
        """
        if edge_types is None:
            edge_types = [None] + [COOCCURRENCE] * (len(tokens) - 1)

        # Step 1: decay
        self.kg.decay()

        # Step 2–3: resolve tokens to nodes, activate them.
        # Inlined for performance: build current_nids AND actual dict in one pass,
        # avoiding separate dict comprehension that calls node() per nid.
        nodes = self.kg._nodes  # direct access, skip method call overhead
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
            node.activation = 1.0  # inline activate — skip min/SPREAD_CAP for perf
            node.resting = min(1.0, node.resting + 0.01)  # RESTING_GROWTH
            current_nids.append(nid)
            actual[nid] = 1.0

        # Step 4: co-occurrence edges within this observation.
        # Connect ALL pairs of tokens with the same edge type — not just
        # consecutive ones. "Fire together, wire together" applies to all
        # neurons active in the same event, regardless of temporal order.
        # This creates direct 3↔4 edges from counting [3, next_is, 4]
        # instead of requiring 2-hop traversal through next_is.
        # Strength diminishes with distance to avoid O(n²) saturation.
        # All tokens in one group for co-occurrence.
        # The normalised attention in select_action handles non-discriminating
        # context tokens (PHASE_test co-occurs with all digits → each gets
        # 1/10 → non-discriminating → doesn't affect ranking).
        all_nids = list(dict.fromkeys(current_nids))
        for etype, nids in {0: all_nids}.items():
            # CAUSAL MASKING: forward-only co-occurrence edges.
            # Earlier tokens predict later tokens, not the reverse.
            # In [6, next_is, 7]: 6→7 exists but 7→6 does not.
            # This is exactly causal masking in transformers — each
            # position only attends to positions that came before it.
            #
            # Without causal masking, training [5, succ, 6] creates
            # 6→5 (backward), making 6 predict 5 instead of 7 on tests.
            unique_nids = list(dict.fromkeys(nids))
            n = len(unique_nids)
            COOCCUR_STRENGTH = 0.5
            for i in range(n):
                for j in range(i + 1, n):
                    src, tgt = unique_nids[i], unique_nids[j]
                    if src == tgt:
                        continue
                    edge = self.kg.get_or_create_edge(src, tgt, role=COOCCURRENCE)
                    edge.observe_present(COOCCUR_STRENGTH)
                    # PoPE: record relative position for what-where decoupling.
                    edge.observe_distance(j - i)
                    # NO reverse edge — causal masking.

        # Step 5: transition edges from previous timestep.
        # CRITICAL: only create transitions when the CURRENT observation
        # is NOT an action (edge_type 2). Transitions from context→action
        # conflate "I was in this context" with "this action is correct."
        # That creates a degenerate attractor where wrong actions get
        # reinforced by their own occurrence.
        #
        # Instead: context→next_observation transitions form normally.
        # Action→next_observation transitions form normally (efference copy).
        # Context→action transitions are ONLY created/updated by
        # retrospective credit (step 6b) based on FEEDBACK.
        is_action_obs = edge_types and len(edge_types) > 0 and edge_types[0] == 2
        if self._prev_active and not is_action_obs:
            cur_set = set(current_nids)
            prev_set = set(self._prev_active)
            new_nids = cur_set - prev_set
            for prev_nid in self._prev_active:
                for cur_nid in new_nids:
                    edge = self.kg.get_or_create_edge(prev_nid, cur_nid, role=TRANSITION)
                    edge.observe_present()

        # Step 6: learn from the PENDING prediction.
        if self._pending_prediction:
            self._last_surprise = self.kg.learn(
                self._pending_prediction, actual,
                prev_active=self._prev_active if self._prev_active else None,
            )
        # For action observations, MERGE with previous context (don't replace).
        # This preserves the original observation tokens for autoregressive
        # digit-by-digit production: the second digit needs to see the original
        # input digits, not just the just-emitted action.
        if is_action_obs and self._last_actual:
            merged = dict(self._last_actual)
            merged.update(actual)
            self._last_actual = merged
        else:
            self._last_actual = actual
        # PoPE: record positions of tokens in this observation.
        is_action_obs = any(et == 2 for et in edge_types if et is not None)
        if is_action_obs and self._last_positions:
            next_pos = max(self._last_positions.values()) + 1
            for i, nid in enumerate(current_nids):
                self._last_positions[nid] = next_pos + i
            # Track that we've emitted an output digit (for product position).
            self._output_digits_emitted += len(current_nids)
        else:
            self._last_positions = {nid: i for i, nid in enumerate(current_nids)}
            # Save the original observation positions as a LIST (preserves duplicates).
            self._original_position_list = [(nid, i) for i, nid in enumerate(current_nids)]
            # Count digit tokens in this non-action observation.
            digit_count = 0
            for nid in current_nids:
                val = self.kg.value_for_node(nid)
                if isinstance(val, str) and len(val) == 1 and val.isdigit():
                    digit_count += 1
            self._input_digit_count = digit_count
            self._output_digits_emitted = 0

        # Step 6b: retrospective action credit via predictive coding.
        # When FEEDBACK arrives, propagate the error backward through
        # the action chain: feedback → action → context → context's sources.
        # This is the predictive coding error propagation applied to
        # the action-selection pathway.
        #
        # Level 0: feedback → action (did this action lead to good/bad?)
        # Level 1: action → context (which context tokens selected this action?)
        # Level 2: context → context's sources (which edges activated the context?)
        if self._last_action_nid is not None and self._last_action_context:
            preferred = self.kg.preferred_nodes()
            credit = 0.0
            for nid in current_nids:
                pref = preferred.get(nid, 0.0)
                credit += pref

            if credit != 0.0:
                action_nid = self._last_action_nid
                strength = min(abs(credit), 3.0)

                # Level 0: update context → action edges.
                for ctx_nid in self._last_action_context:
                    edge = self.kg.edge(ctx_nid, action_nid)
                    if edge is not None and edge.role == TRANSITION:
                        if credit > 0:
                            edge.observe_present(strength)
                        else:
                            edge.observe_absent(strength)

                # Level 1: propagate credit backward from context to its
                # incoming transition edges. This is the predictive coding
                # error propagation — the error at the action level
                # propagates to the edges that activated the context.
                # Reduced strength (0.3x) to prevent runaway.
                back_strength = strength * 0.3
                for ctx_nid in self._last_action_context:
                    for edge in self.kg.edges_to(ctx_nid):
                        if edge.role != TRANSITION:
                            continue
                        if edge.effective_weight < 0.01:
                            continue
                        if credit > 0:
                            edge.observe_present(back_strength)
                        else:
                            edge.observe_absent(back_strength)

        # Step 7a: iterative co-occurrence spread + sigma.
        # Each round: spread co-occurrence, then compute sigma on edges
        # between co-active nodes. Multiple rounds let the activation
        # cascade: round 1 partially activates the successor digit,
        # round 2 boosts sigma on edges to that digit, round 3 activates
        # it more strongly. This is the neural oscillation that forms
        # a coherent assembly — it takes multiple cycles to settle.
        COOCCUR_ROUNDS = 3
        self.kg.reset_sigma()
        for _round in range(COOCCUR_ROUNDS):
            cooccur_spread = self.kg.spread(role_filter=COOCCURRENCE)
            for nid, level in cooccur_spread.items():
                if level > 0:
                    node = nodes.get(nid)
                    if node is not None:
                        node.activation = max(node.activation, min(1.0, level * 0.5))

            active_nids_full = set(
                nid for nid, n in nodes.items()
                if n.activation >= ACTIVATION_THRESHOLD
            )
            self.kg.compute_sigma(active_nids_full)

        # Step 7b: spread from THIS observation (= predict what comes NEXT).
        # Only follow TRANSITION edges for temporal prediction.
        self._pending_prediction = self.kg.spread(role_filter=TRANSITION)
        # Remove current observation nodes from the prediction. They are the
        # present, not the future. Without this filter, corridor→go_west
        # predicts go_west DURING the go_west observation, and then at the
        # NEXT observation (closet), go_west is "predicted but not observed"
        # — wrongly penalising a correct edge.
        for nid in current_nids:
            self._pending_prediction.pop(nid, None)

        # Step 8: store snapshot + observation record.
        # The snapshot captures ALL active nodes (including decay residuals)
        # for replay. The observation record captures ONLY this observe()
        # call's tokens for NT discovery (no ghosts).
        self.hippo.store(self.kg.active_nodes(), observed_nids=current_nids)

        # Remember ONLY this observation's nodes for next timestep's transitions.
        # Inlined: reuse `actual` dict instead of rebuilding via node() calls.
        self._prev_active = actual
        self._step_count += 1

        # Step 9: periodic consolidation (the slow path).
        if (self.CONSOLIDATION_INTERVAL > 0
                and self._step_count - self._last_consolidation_step
                    >= self.CONSOLIDATION_INTERVAL):
            self.consolidate()

    # -------------------------------------------------------------------
    # Act
    # -------------------------------------------------------------------

    def act(self, candidates: list[str]) -> str | None:
        """
        Select an action via Expected Free Energy minimisation.

        Each candidate is a token string. Resolved to a NodeId, then
        scored by select_action() which combines forward spread (context
        consistency), backward spread (pragmatic value toward preferences),
        and epistemic value (exploration bonus for novel candidates).

        Includes epsilon-greedy exploration to break rich-get-richer
        attractors. Without exploration, the agent locks onto the most-
        connected candidate and never discovers that other candidates
        lead to correct outcomes in different contexts.

        Epsilon decays as 1/(1 + step/100), starting at ~50% exploration
        and declining to ~10% after 1000 steps.

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

        # Product position: compute position-from-end for the current output
        # digit. If we know the input had N digits and we've emitted K so far,
        # we're producing position N-1-K from the end (0 = units, 1 = tens, etc.).
        # This is a heuristic that assumes output length = input length (correct
        # for all non-carry cases; carry cases need special handling later).
        pos_from_end = None
        if self._input_digit_count > 0 and self._output_digits_emitted < self._input_digit_count:
            pos_from_end = self._input_digit_count - 1 - self._output_digits_emitted

        best_nid = self.kg.select_action(
            candidate_nids,
            fixed_context=fixed_context,
            context_positions=context_positions,
            answer_position=answer_pos,
            position_from_end=pos_from_end,
            input_digit_count=self._input_digit_count,
            original_position_list=list(self._original_position_list) if self._original_position_list else None,
        )
        if best_nid is None:
            return None

        # Store for retrospective credit: when FEEDBACK arrives, we need to
        # know which action was taken and what context it was taken in.
        self._last_action_nid = best_nid
        self._last_action_context = dict(self._last_actual) if self._last_actual else {}

        return nid_to_token.get(best_nid)

    # -------------------------------------------------------------------
    # Consolidation (the slow path, callable through the loop)
    # -------------------------------------------------------------------

    def consolidate(self, replay_passes: int = 2) -> dict:
        """Run consolidation: replay → prune → colimits.

        This is the only way to trigger consolidation. Tests and
        environments call this method, not consolidation.py directly.
        """
        from experiments.symbolic_ai_v2.ctkg.logic import consolidation
        stats = consolidation.consolidate(
            self.kg, self.hippo, replay_passes=replay_passes,
        )
        self._last_consolidation_step = self._step_count
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
