"""
KnowledgeGraph — the one graph.

Nodes are tokens. Edges are connections. Activation is working memory.
Spread is prediction. Hebbian update is learning. There is no separate
observation format, no pair extraction, no InputOutputTopology.

Every node carries an activation level. Every edge carries a weight.
Working memory = {nodes where activation > threshold}.

The single process each timestep: spread → compare → update.
  - Spread (deduction): active nodes propagate along weighted edges.
  - Compare: divergence between predicted and actual activation.
  - Update (induction + abduction): strengthen confirmed edges, weaken
    wrong ones, create edges to explain surprises.

Tokenization happens upstream (tokenizer.py). This module never sees raw
strings — only opaque integer node IDs.

Performance: adjacency index (_outgoing) gives O(degree) edge iteration
instead of O(E). __slots__ on Node/Edge reduces memory ~40%.
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

NodeId = int

# Edge roles: the only two kinds.
COOCCURRENCE = 0   # A and B appeared in the same observation
TRANSITION   = 1   # A preceded B across a timestep boundary (action between)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """A single node in the knowledge graph."""
    __slots__ = ('id', 'activation', 'resting', 'preferred', 'label')

    def __init__(self, id: NodeId, activation: float = 0.0,
                 resting: float = 0.0, preferred: float = 0.0,
                 label: str | None = None):
        self.id = id
        self.activation = activation
        self.resting = resting
        self.preferred = preferred
        self.label = label


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class Edge:
    """A directed edge with Beta-posterior weight.

    The edge weight is a Bayesian posterior, not a fixed number.
    alpha counts evidence that this edge's target APPEARS when the source
    is active. beta counts evidence that it DOESN'T.

    weight = (alpha - beta) / (alpha + beta)   ∈ [-1, 1]
    confidence = alpha + beta                   (total evidence)

    One update rule covers all four cases:
      target appeared  → alpha += 1  (more positive / less negative)
      target absent    → beta  += 1  (more negative / less positive)

    New edges start at alpha=1, beta=1 (uniform prior, weight=0).
    Learning rate is implicit: low confidence → large updates,
    high confidence → small updates. No tuning parameter.
    """
    __slots__ = ('source', 'target', 'alpha', 'beta', 'role', '_w', 'sigma',
                 '_dist_sum', '_dist_sq_sum', '_dist_count')

    def __init__(self, source: NodeId, target: NodeId,
                 alpha: float = 1.0, beta: float = 1.0,
                 role: int = COOCCURRENCE):
        self.source = source
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.role = role
        self._w = (alpha - beta) / (alpha + beta) if (alpha + beta) > 0 else 0.0
        self.sigma = 0.0  # transient context-dependent state (BDH / sheaf)
        # PoPE-inspired positional statistics: decouple "what" from "where".
        # Tracks the distribution of relative positions at which these two
        # tokens co-occur. Used to compute position_match in attention.
        self._dist_sum = 0.0     # sum of observed distances
        self._dist_sq_sum = 0.0  # sum of squared distances
        self._dist_count = 0     # number of distance observations

    @property
    def weight(self) -> float:
        """Posterior mean, cached in _w. Range [-1, 1].

        Cached value is updated by observe_present/observe_absent.
        If alpha/beta are set directly (e.g., in tests), call _recalc().
        """
        return self._w

    @weight.setter
    def weight(self, val: float) -> None:
        """Set weight by adjusting alpha/beta symmetrically."""
        total = self.alpha + self.beta
        if total == 0:
            total = 2.0
        self.alpha = total * (1.0 + max(-1.0, min(1.0, val))) / 2.0
        self.beta = total - self.alpha
        self._w = (self.alpha - self.beta) / (self.alpha + self.beta)

    def _recalc(self) -> None:
        """Recompute cached weight after direct alpha/beta modification."""
        total = self.alpha + self.beta
        self._w = (self.alpha - self.beta) / total if total > 0 else 0.0

    @property
    def effective_weight(self) -> float:
        """Context-dependent effective weight.

        CO-OCCURRENCE edges: permanent weight + sigma. Co-occurrence is a
        global fact ("3 and 4 appear together") modulated by context.

        TRANSITION edges: sigma ONLY. The permanent weight gates existence
        (the transition was observed at least once) but does NOT contribute
        to strength. The strength of a transition in any given context is
        entirely determined by the current co-activation pattern (sigma).

        This mirrors cortical synapses: the effective postsynaptic response
        depends on the local dendritic computation (what else is active),
        not just the synapse's long-term potentiation weight. A synapse
        that was strengthened in context A doesn't fire strongly in context B
        just because it was potentiated — the dendritic context gates it.

        The permanent alpha/beta on transition edges records that the
        transition EXISTS (has been observed). Whether it FIRES depends
        on sigma (the current assembly pattern).
        """
        if self.role == COOCCURRENCE:
            return self._w + self.sigma
        else:
            # TRANSITION: permanent weight as prior, modulated by sigma.
            # The permanent alpha/beta captures accumulated evidence
            # (20 eat observations vs 3 go observations). Sigma adds
            # context-dependent boost from current co-activation.
            if self.alpha + self.beta <= 2.01:  # uniform prior = never observed
                return 0.0
            return self._w + self.sigma

    @property
    def confidence(self) -> float:
        """Total evidence count. Higher = more stable."""
        return self.alpha + self.beta

    def observe_present(self, strength: float = 1.0) -> None:
        """Target appeared. Bayesian update: alpha += strength."""
        self.alpha += strength
        total = self.alpha + self.beta
        self._w = (self.alpha - self.beta) / total

    def observe_absent(self, strength: float = 1.0) -> None:
        """Target did not appear. Bayesian update: beta += strength."""
        self.beta += strength
        total = self.alpha + self.beta
        self._w = (self.alpha - self.beta) / total

    def observe_distance(self, dist: int) -> None:
        """Record the relative position at which source→target co-occurred.

        PoPE-inspired: tracks the distribution of distances so that
        attention can decouple "what" (content association) from "where"
        (positional pattern).
        """
        self._dist_sum += dist
        self._dist_sq_sum += dist * dist
        self._dist_count += 1

    @property
    def mean_distance(self) -> float:
        """Mean observed relative position."""
        if self._dist_count == 0:
            return 0.0
        return self._dist_sum / self._dist_count

    @property
    def var_distance(self) -> float:
        """Variance of observed relative positions."""
        if self._dist_count < 2:
            return 1.0  # high variance prior when few observations
        mu = self._dist_sum / self._dist_count
        return max(0.1, self._dist_sq_sum / self._dist_count - mu * mu)

    def position_match(self, query_dist: int) -> float:
        """How well does query_dist match this edge's typical distance?

        Returns a value in (0, 1]: 1.0 = perfect match, decays as a
        Gaussian away from the mean distance. Broad variance = tolerant
        of position mismatch. Narrow variance = specific position required.

        When no distance observations exist, returns 1.0 (no position bias).
        """
        import math
        if self._dist_count == 0:
            return 1.0  # no positional information — content-only
        mu = self.mean_distance
        var = self.var_distance
        return math.exp(-0.5 * (query_dist - mu) ** 2 / var)

    def __repr__(self) -> str:
        return (f"Edge(source={self.source}, target={self.target}, "
                f"alpha={self.alpha}, beta={self.beta}, role={self.role})")


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

# Tuning constants
DECAY_FACTOR       = 0.85     # per-step activation decay
ACTIVATION_THRESHOLD = 0.05   # below this, node is not "active"
EDGE_CREATION_THRESHOLD = 0.3 # minimum source activation to create a new edge
SPREAD_THRESHOLD       = 0.5  # minimum source activation to contribute to spread
RESTING_GROWTH     = 0.01     # how much resting potential grows per activation
SPREAD_CAP         = 1.0      # maximum activation after spread


class KnowledgeGraph:
    """
    The one graph. Nodes, edges, activation dynamics, learning.

    Uses an adjacency index (_outgoing) for O(degree) edge lookups
    instead of O(E) full scans.
    """

    def __init__(self) -> None:
        self._nodes: dict[NodeId, Node] = {}
        # Edges keyed by (source, target) — at most one edge per directed pair.
        self._edges: dict[tuple[NodeId, NodeId], Edge] = {}
        # Adjacency indices: NodeId → list of Edges.
        # _outgoing: edges FROM this node. _incoming: edges TO this node.
        # Both maintained in sync with _edges by get_or_create_edge.
        self._outgoing: dict[NodeId, list[Edge]] = {}
        self._incoming: dict[NodeId, list[Edge]] = {}
        self._next_id: int = 0
        # Reverse index: value → NodeId (for get_or_create)
        self._value_to_node: dict[Any, NodeId] = {}
        # Cache: sum of positive outgoing co-occurrence effective weights per node.
        # Invalidated when sigma changes (reset_sigma clears it).
        self._cooccur_out_sum: dict[NodeId, float] = {}
        # Product projections: discovered by consolidation, stored separately
        # from edges because (src, tgt) may already have a co-occurrence edge.
        # Maps (src_nid, tgt_nid, position_from_end) → weight.
        self._product_projections: dict[tuple[NodeId, NodeId, int], float] = {}

    # -------------------------------------------------------------------
    # Node creation
    # -------------------------------------------------------------------

    def get_or_create(self, value: Any) -> NodeId:
        """Return the NodeId for a value, creating the node if needed.

        Two observations of the same value share ONE node — identity.
        """
        if value in self._value_to_node:
            return self._value_to_node[value]
        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = Node(id=nid, label=str(value) if value is not None else None)
        self._value_to_node[value] = nid
        return nid

    def node(self, nid: NodeId) -> Node | None:
        return self._nodes.get(nid)

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    # -------------------------------------------------------------------
    # Activation
    # -------------------------------------------------------------------

    def activate(self, nid: NodeId, level: float = 1.0) -> None:
        """Set a node's activation to the given level."""
        node = self._nodes.get(nid)
        if node is None:
            return
        node.activation = min(SPREAD_CAP, level)
        # Exposure builds resting potential.
        node.resting = min(1.0, node.resting + RESTING_GROWTH)

    def decay(self) -> None:
        """Decay all activations toward resting potential."""
        for node in self._nodes.values():
            node.activation *= DECAY_FACTOR
            if node.activation < ACTIVATION_THRESHOLD:
                node.activation = 0.0

    def active_nodes(self) -> dict[NodeId, float]:
        """Return {node_id: activation} for all active nodes."""
        return {
            nid: n.activation
            for nid, n in self._nodes.items()
            if n.activation >= ACTIVATION_THRESHOLD
        }

    # -------------------------------------------------------------------
    # Edges (with adjacency index)
    # -------------------------------------------------------------------

    def get_or_create_edge(self, src: NodeId, tgt: NodeId, role: int = COOCCURRENCE) -> Edge:
        """Get or create a directed edge. Role is set on creation only.

        New edges start with uniform Beta(1, 1) prior — weight = 0, no bias.
        Maintains both _outgoing and _incoming adjacency indices.
        """
        key = (src, tgt)
        existing = self._edges.get(key)
        if existing is not None:
            return existing
        edge = Edge(source=src, target=tgt, alpha=1.0, beta=1.0, role=role)
        self._edges[key] = edge
        # Maintain outgoing index.
        out = self._outgoing.get(src)
        if out is None:
            self._outgoing[src] = [edge]
        else:
            out.append(edge)
        # Maintain incoming index.
        inc = self._incoming.get(tgt)
        if inc is None:
            self._incoming[tgt] = [edge]
        else:
            inc.append(edge)
        return edge

    def edges_to(self, nid: NodeId) -> list[Edge]:
        """All edges pointing TO this node."""
        return self._incoming.get(nid, [])

    def edge(self, src: NodeId, tgt: NodeId) -> Edge | None:
        return self._edges.get((src, tgt))

    def edges_from(self, nid: NodeId) -> list[Edge]:
        """All outgoing edges from a node. O(degree) via adjacency index."""
        return self._outgoing.get(nid, [])

    def edges_to(self, nid: NodeId) -> list[Edge]:
        """All incoming edges to a node. O(E) — no incoming index."""
        return [e for (_, t), e in self._edges.items() if t == nid]

    def remove_edge(self, src: NodeId, tgt: NodeId) -> bool:
        """Remove an edge. Returns True if found and removed."""
        key = (src, tgt)
        edge = self._edges.pop(key, None)
        if edge is None:
            return False
        out = self._outgoing.get(src)
        if out is not None:
            try:
                out.remove(edge)
            except ValueError:
                pass
        return True

    # -------------------------------------------------------------------
    # Harmonic extension (sheaf Laplacian energy minimisation)
    # -------------------------------------------------------------------

    def harmonic_extend(
        self,
        fixed: dict[NodeId, float],
        free: set[NodeId] | None = None,
        iterations: int = 5,
        damping: float = 0.5,
    ) -> dict[NodeId, float]:
        """Find activations on free nodes that minimise sheaf Laplacian energy.

        Given fixed activations (the known observation) and free nodes (the
        unknowns to be filled in), iteratively relax free node activations
        toward consistency with the edge structure.

        This is the sheaf-theoretic answer to "what value should the ? node
        take?" — the answer is the harmonic extension of the partial section.

        The sheaf Laplacian energy for edge (A→B) with effective weight w is:
            E = |w| * (act_A - act_B)² if w > 0  (excitatory: agreement)
            E = |w| * act_A * act_B     if w < 0  (inhibitory: mutual exclusion)

        Minimising E for excitatory edges pulls connected activations together.
        Minimising E for inhibitory edges pushes them apart.

        Parameters
        ----------
        fixed : {NodeId: activation} — nodes whose activation is known (clamped).
        free : set of NodeIds whose activation should be solved for.
            If None, all nodes NOT in fixed are free.
        iterations : number of relaxation iterations.
        damping : how much to move toward the Laplacian-optimal value each step.
            0 = no change, 1 = full step.

        Returns {NodeId: settled_activation} for the free nodes.
        """
        if free is None:
            free = set(self._nodes.keys()) - set(fixed.keys())

        if not free:
            return {}

        # Initialise free nodes from current activation.
        current: dict[NodeId, float] = {}
        for nid in free:
            node = self._nodes.get(nid)
            current[nid] = node.activation if node else 0.0

        # Iterative relaxation.
        for _ in range(iterations):
            updates: dict[NodeId, float] = {}

            for nid in free:
                # Compute the Laplacian-optimal activation for this node:
                # weighted average of neighbor activations.
                numerator = 0.0
                denominator = 0.0

                # Incoming edges: neighbors that point TO this node.
                for (src, tgt), edge in self._edges.items():
                    if tgt != nid:
                        continue
                    w = edge.effective_weight
                    if w <= 0:
                        continue  # only excitatory edges pull
                    # Source activation: use fixed if clamped, else current.
                    src_act = fixed.get(src, current.get(src, 0.0))
                    numerator += w * src_act
                    denominator += w

                # Outgoing edges: neighbors this node points TO (bidirectional pull).
                for edge in self._outgoing.get(nid, ()):
                    w = edge.effective_weight
                    if w <= 0:
                        continue
                    tgt_act = fixed.get(edge.target, current.get(edge.target, 0.0))
                    numerator += w * tgt_act
                    denominator += w

                if denominator > 0:
                    optimal = numerator / denominator
                    updates[nid] = current[nid] + damping * (optimal - current[nid])
                else:
                    updates[nid] = current[nid]

            # Apply updates.
            for nid, val in updates.items():
                current[nid] = max(0.0, min(SPREAD_CAP, val))

        return current

    # -------------------------------------------------------------------
    # Dynamic sigma (BDH-style transient state / sheaf local section)
    # -------------------------------------------------------------------

    def reset_sigma(self) -> None:
        """Reset all transient edge state to zero.

        Called at the start of each observation. The sigma field captures
        context-dependent modulation — it lives for one inference pass only.
        """
        for edge in self._edges.values():
            edge.sigma = 0.0
        self._cooccur_out_sum.clear()  # invalidate selectivity cache

    def compute_sigma(
        self,
        active_nids: set[NodeId],
        idf: dict[NodeId, float] | None = None,
    ) -> None:
        """QKV-style sigma with IDF weighting for discriminating context.

        The transformer's QKV attention, adapted to the graph:
        - Q (query) = the current activation pattern, IDF-weighted
        - K (key) = each target node's co-occurrence profile
        - sigma(A→B) = activation_A × activation_B × (Q · K_B) × SCALE

        IDF weighting: context tokens that appear in every observation
        (succ, PHASE_test) have low IDF and contribute little to Q·K.
        Tokens that vary between observations (the question digit) have
        high IDF and dominate Q·K. This is the TF-IDF insight: common
        tokens are not informative.

        Without IDF, the Q·K for all digits is dominated by their
        co-occurrence with `succ` (which appears in every question),
        making all digits score similarly. With IDF, only the question
        digit's co-occurrence matters, correctly selecting the successor.

        Parameters
        ----------
        active_nids : set of currently active NodeIds
        idf : {NodeId: idf_weight} — inverse document frequency weights.
            If None, all context tokens weighted equally (falls back to
            plain QKV). Computed from hippocampus observation counts.
        """
        SIGMA_SCALE = 0.3

        # Step 1: build the IDF-weighted context vector Q.
        active_acts: dict[NodeId, float] = {}
        for nid in active_nids:
            node = self._nodes.get(nid)
            if node is not None and node.activation > 0:
                # Weight activation by IDF: rare tokens contribute more.
                idf_w = idf.get(nid, 1.0) if idf else 1.0
                active_acts[nid] = node.activation * idf_w

        # Step 2: precompute Q · K for each active target node.
        qk: dict[NodeId, float] = {}
        for tgt_nid in active_nids:
            support = 0.0
            for edge in self._incoming.get(tgt_nid, ()):
                if edge.role != COOCCURRENCE:
                    continue
                c_nid = edge.source
                c_act = active_acts.get(c_nid, 0.0)
                if c_act <= 0:
                    continue
                cw = edge._w
                if cw > 0:
                    support += c_act * cw
            qk[tgt_nid] = support

        # Step 3: set sigma on each edge using precomputed Q·K.
        for src_nid in active_nids:
            src_act = active_acts.get(src_nid, 0.0)
            if src_act <= 0:
                continue
            for edge in self._outgoing.get(src_nid, ()):
                tgt_nid = edge.target
                if tgt_nid not in active_nids:
                    continue
                tgt_act = active_acts.get(tgt_nid, 0.0)
                if tgt_act <= 0:
                    continue
                support = qk.get(tgt_nid, 0.0)
                edge.sigma += src_act * tgt_act * (1.0 + support) * SIGMA_SCALE

    # -------------------------------------------------------------------
    # Spread (= prediction = deduction)
    # -------------------------------------------------------------------

    def spread(self, role_filter: int | None = None) -> dict[NodeId, float]:
        """Propagate activation along weighted edges.

        Two normalisation modes based on edge role:

        TRANSITION edges: normalised per source (stochastic kernel).
        Each source's positive outgoing transitions form a probability
        distribution. This is correct: "given state S, what's the
        probability of each next state?"

        CO-OCCURRENCE edges: RAW weight, not normalised. Each association
        activates independently. "Node A co-occurs with B" and "A co-occurs
        with C" are independent facts — B and C don't compete. Normalisation
        would dilute the number line signal (6→7) across all of 6's
        co-occurrence neighbors (succ, space, next_is, etc.).

        Inhibitory edges (negative weight): raw weight in both modes
        (threshold, not normalised).

        Returns {node_id: predicted_level}. Does NOT modify activations.
        """
        predicted: dict[NodeId, float] = {}

        for nid, node in self._nodes.items():
            if node.activation < SPREAD_THRESHOLD:
                continue
            out = self._outgoing.get(nid)
            if not out:
                continue

            src_act = node.activation

            # Separate edges by role and sign.
            trans_pos: list[tuple[Edge, float]] = []
            trans_pos_total = 0.0
            cooccur_pos: list[tuple[Edge, float]] = []
            neg_edges: list[tuple[Edge, float]] = []

            for edge in out:
                if role_filter is not None and edge.role != role_filter:
                    continue
                w = edge.effective_weight
                if w > 0:
                    if edge.role == TRANSITION:
                        trans_pos.append((edge, w))
                        trans_pos_total += w
                    else:
                        cooccur_pos.append((edge, w))
                elif w < 0:
                    neg_edges.append((edge, w))

            # Transition: normalised (stochastic kernel).
            if trans_pos_total > 0:
                for e, w in trans_pos:
                    contribution = src_act * (w / trans_pos_total)
                    predicted[e.target] = predicted.get(e.target, 0.0) + contribution

            # Co-occurrence: raw weight (independent associations).
            for e, w in cooccur_pos:
                predicted[e.target] = predicted.get(e.target, 0.0) + src_act * w

            # Inhibitory: raw weight (threshold).
            for e, w in neg_edges:
                predicted[e.target] = predicted.get(e.target, 0.0) + src_act * w

        # Clamp to [-SPREAD_CAP, +SPREAD_CAP].
        for nid in predicted:
            v = predicted[nid]
            if v > SPREAD_CAP:
                predicted[nid] = SPREAD_CAP
            elif v < -SPREAD_CAP:
                predicted[nid] = -SPREAD_CAP
        return predicted

    # -------------------------------------------------------------------
    # Multi-hop co-occurrence spread (substrate for functor discovery)
    # -------------------------------------------------------------------

    def spread_cooccurrence(
        self,
        seeds: dict[NodeId, float],
        hops: int = 4,
        decay: float = 0.7,
    ) -> dict[NodeId, float]:
        """Iterative co-occurrence spread from seed nodes.

        Each hop: for each currently active node, spread activation along
        positive-weight FORWARD co-occurrence edges (causal masking respected).
        Activation at each hop is multiplied by `decay` to prevent saturation.

        Returns {NodeId: activation} for all reachable nodes.

        This is the substrate for multi-hop chain traversal. By itself it
        doesn't know when to stop (that's the functor's job). It just makes
        distant nodes reachable so that higher-level structure can use them.

        Parameters
        ----------
        seeds : initial activations {NodeId: level}
        hops : number of spread iterations
        decay : multiplicative decay per hop (prevents saturation)
        """
        current = dict(seeds)
        accumulated = dict(seeds)

        for hop in range(hops):
            next_act: dict[NodeId, float] = {}
            for nid, act in current.items():
                if act < 0.01:
                    continue
                for edge in self._outgoing.get(nid, ()):
                    if edge.role != COOCCURRENCE:
                        continue
                    ew = edge.effective_weight
                    if ew <= 0:
                        continue
                    contrib = act * ew * decay
                    if contrib > 0.01:
                        next_act[edge.target] = max(
                            next_act.get(edge.target, 0.0), contrib
                        )
            current = next_act
            for nid, act in current.items():
                accumulated[nid] = max(accumulated.get(nid, 0.0), act)

        return accumulated

    # -------------------------------------------------------------------
    # Learn (= Hebbian update = induction + abduction)
    # -------------------------------------------------------------------

    def learn(
        self,
        predicted: dict[NodeId, float],
        actual: dict[NodeId, float],
        prev_active: dict[NodeId, float] | None = None,
    ) -> float:
        """Predictive coding: settle activations, then update edge weights.

        Two phases, following Rao & Ballard (1999) and Song et al. (2024):

        1. INFERENCE PHASE: clamp observed nodes, propagate prediction errors
           backward through the graph, settle to a prospective configuration.
           This distributes credit across the full chain, not just one hop.

        2. WEIGHT UPDATE PHASE: update edge alpha/beta based on the settled
           error pattern. Edges that contributed to correct predictions get
           observe_present(); edges that contributed to errors get observe_absent().

        Returns total surprise.
        """
        surprise = 0.0
        actual_set = set(actual.keys())
        predicted_set = set(predicted.keys())

        # --- Phase 1: Inference (settle prediction errors) ---
        # Compute initial prediction error at each node.
        # error > 0 means "observed but underpredicted" (surprise)
        # error < 0 means "predicted but not observed" (false alarm)
        errors: dict[NodeId, float] = {}

        for nid in predicted_set | actual_set:
            p = predicted.get(nid, 0.0)
            a = actual.get(nid, 0.0)
            err = a - p
            if abs(err) > 0.01:
                errors[nid] = err

        # Propagate errors backward along edges for several iterations.
        # Each iteration: for each node with error, propagate a fraction
        # of that error backward to its sources (via _incoming edges).
        # This is the predictive coding error propagation.
        PC_ITERATIONS = 3
        PC_ERROR_RATE = 0.3  # fraction of error propagated per iteration

        # Collect which edges participated in the prediction (for phase 2).
        edge_credit: dict[tuple[NodeId, NodeId], float] = {}

        for iteration in range(PC_ITERATIONS):
            new_errors: dict[NodeId, float] = {}
            for tgt_nid, err in errors.items():
                if abs(err) < 0.01:
                    continue
                # Propagate error backward to sources.
                for edge in self._incoming.get(tgt_nid, ()):
                    if edge.role != TRANSITION:
                        continue
                    src_nid = edge.source
                    # Only propagate to nodes that were active (contributed
                    # to the prediction). Don't propagate to inactive nodes.
                    if prev_active and src_nid not in prev_active:
                        continue
                    # The error at the source is proportional to the edge
                    # weight (how much this edge contributed to the prediction)
                    # times the target error.
                    w = edge.effective_weight
                    if abs(w) < 0.01:
                        continue
                    backprop = err * w * PC_ERROR_RATE
                    new_errors[src_nid] = new_errors.get(src_nid, 0.0) + backprop

                    # Accumulate edge credit for phase 2.
                    key = (src_nid, tgt_nid)
                    edge_credit[key] = edge_credit.get(key, 0.0) + err

            # Add propagated errors to existing errors.
            for nid, err in new_errors.items():
                errors[nid] = errors.get(nid, 0.0) + err

        # --- Phase 2: Weight update (Bayesian) ---
        # Use the settled error pattern to update edges.
        # An edge gets credit from BOTH the direct error at its target
        # AND the propagated error from further downstream.

        # Restrict to prev_active sources. If not provided, use all active
        # nodes as sources (backward compatibility with direct learn() calls).
        if prev_active:
            contributing_sources = set(prev_active.keys())
        else:
            contributing_sources = set(
                nid for nid, node in self._nodes.items()
                if node.activation >= ACTIVATION_THRESHOLD
            )

        for src_nid in contributing_sources:
            for edge in self._outgoing.get(src_nid, ()):
                tgt_nid = edge.target
                tgt_predicted = predicted.get(tgt_nid)
                if tgt_predicted is None or tgt_predicted == 0.0:
                    continue

                tgt_observed = tgt_nid in actual_set

                # Base credit from direct observation.
                credit = edge_credit.get((src_nid, tgt_nid), 0.0)

                if tgt_observed:
                    # Correctly predicted or underpredicted: strengthen.
                    # Strength proportional to how much credit this edge gets.
                    strength = 1.0 + min(abs(credit), 2.0) * 0.5
                    edge.observe_present(strength)
                    p = max(abs(tgt_predicted), 0.01)
                    surprise += -math.log(min(p, 1.0))
                else:
                    # Predicted but not observed: weaken.
                    if edge.role == TRANSITION:
                        strength = 1.0 + min(abs(credit), 2.0) * 0.5
                        edge.observe_absent(strength)
                    if tgt_predicted > 0:
                        p = max(tgt_predicted, 0.01)
                        surprise += -math.log(min(p, 1.0))

        # --- Abduction: observed but not predicted → create edges ---
        active_sources = self.active_nodes()
        abduct_role = TRANSITION if prev_active else COOCCURRENCE
        for tgt_nid in actual_set - predicted_set:
            best_src = None
            best_act = 0.0
            for src_nid, src_act in active_sources.items():
                if src_act > best_act and src_nid != tgt_nid:
                    if self.edge(src_nid, tgt_nid) is None:
                        best_src = src_nid
                        best_act = src_act
            if best_src is not None and best_act >= EDGE_CREATION_THRESHOLD:
                edge = self.get_or_create_edge(best_src, tgt_nid, role=abduct_role)
                edge.alpha = 2.0
                edge.beta = 1.0
                edge._recalc()
            surprise += 1.0

        return surprise

    # -------------------------------------------------------------------
    # Probability queries
    # -------------------------------------------------------------------

    def transition_distribution(self, source: NodeId) -> dict[NodeId, float]:
        """Return the normalised probability distribution P(target | source).

        Only considers positive-weight TRANSITION edges from source.
        O(degree) via adjacency index.
        """
        total = 0.0
        edges: list[Edge] = []
        for e in self._outgoing.get(source, ()):
            if e.role == TRANSITION and e.weight > 0:
                edges.append(e)
                total += e.weight
        if total <= 0:
            return {}
        return {e.target: e.weight / total for e in edges}

    def edge_entropy(self, source: NodeId, role: int = TRANSITION) -> float:
        """Shannon entropy of the outgoing distribution from source.

        H = -sum(p * log2(p)). O(degree) via adjacency index.
        """
        total = 0.0
        weights: list[float] = []
        for e in self._outgoing.get(source, ()):
            if e.role == role and e.weight > 0:
                w = e.weight
                weights.append(w)
                total += w
        if total <= 0:
            return 0.0
        entropy = 0.0
        for w in weights:
            p = w / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def edge_confidence(self, src: NodeId, tgt: NodeId) -> float:
        """Return the confidence (alpha + beta) of a specific edge."""
        e = self.edge(src, tgt)
        return e.confidence if e is not None else 0.0

    # -------------------------------------------------------------------
    # Homeostatic priors
    # -------------------------------------------------------------------

    def set_preferred(self, nid: NodeId, level: float) -> None:
        """Set the homeostatic prior for a node."""
        node = self._nodes.get(nid)
        if node is not None:
            node.preferred = level

    def preferred_nodes(self) -> dict[NodeId, float]:
        """Return {nid: preferred_level} for all nodes with preferences."""
        return {
            nid: n.preferred
            for nid, n in self._nodes.items()
            if n.preferred != 0.0
        }

    # -------------------------------------------------------------------
    # Action selection: attention + active inference
    # -------------------------------------------------------------------

    PRODUCT_ROLE = 3  # edge role for product projections

    PRODUCT_ROLE = 3  # edge role for product projections

    def select_action(
        self,
        candidates: list[NodeId],
        fixed_context: dict[NodeId, float] | None = None,
        context_positions: dict[NodeId, int] | None = None,
        answer_position: int | None = None,
        position_from_end: int | None = None,
        input_digit_count: int = 0,
        original_position_list: list[tuple[NodeId, int]] | None = None,
    ) -> NodeId | None:
        """Select an action via attention with product-aware position matching.

        Two layers of attention, combined:

        1. **Product attention** (position-specific): if position_from_end is
           provided, consult PRODUCT edges (role=3) from context tokens whose
           INPUT position-from-end matches the OUTPUT position-from-end.
           Only the context digit at the MATCHING input position contributes
           product signal. This prevents the units digit's product edges from
           influencing the tens output and vice versa.

        2. **Co-occurrence attention** (content): normalised co-occurrence
           weight from each context token to each candidate.

        Parameters
        ----------
        candidates : list of candidate action NodeIds.
        fixed_context : {NodeId: activation} for the current observation.
        context_positions : {NodeId: position_index} for each context token.
        answer_position : the position index where the answer would appear.
        position_from_end : which digit position we're producing (0 = units,
            1 = tens, 2 = hundreds, etc.).
        input_digit_count : total number of digit tokens in the input. Used
            to compute each context token's position-from-end so we only
            consult product edges from the matching input position.
        """
        if not candidates:
            return None

        context = fixed_context if fixed_context is not None else self.active_nodes()
        candidate_set = set(candidates)

        if context_positions and answer_position is None:
            answer_position = max(context_positions.values()) + 1

        logits: dict[NodeId, float] = {nid: 0.0 for nid in candidates}

        # --- Layer 1: Product attention (position-specific projections) ---
        # Use original_position_list (preserves duplicates) to find which
        # input digit is at the matching position-from-end. For 5000 with
        # positions [(5,0), (0,1), (0,2), (0,3), (succ,4)]:
        #   position_from_end=3 → input digit at obs_pos 0 → node 5
        #   position_from_end=2 → input digit at obs_pos 1 → node 0
        #   position_from_end=1 → input digit at obs_pos 2 → node 0
        #   position_from_end=0 → input digit at obs_pos 3 → node 0
        if position_from_end is not None and original_position_list and input_digit_count > 0:
            # Extract input digit positions: the first input_digit_count
            # entries in the position list that have product projections.
            input_digit_entries: list[tuple[NodeId, int]] = []
            for nid, pos in original_position_list:
                if len(input_digit_entries) >= input_digit_count:
                    break
                # Check if this node appears as source in any product projection.
                has_product = any(
                    k[0] == nid for k in self._product_projections
                )
                if has_product:
                    input_digit_entries.append((nid, pos))

            # The matching input digit: count from the RIGHT.
            target_idx = len(input_digit_entries) - 1 - position_from_end
            if 0 <= target_idx < len(input_digit_entries):
                match_nid = input_digit_entries[target_idx][0]
                for cand_nid in candidates:
                    key = (match_nid, cand_nid, position_from_end)
                    weight = self._product_projections.get(key, 0.0)
                    if weight > 0:
                        logits[cand_nid] += weight * 2.0

        # --- Layer 2: Co-occurrence attention (content) ---
        for ctx_nid, ctx_act in context.items():
            if ctx_act <= 0:
                continue

            profile: list[tuple[NodeId, Edge]] = []
            total_ew = 0.0
            for edge in self._outgoing.get(ctx_nid, ()):
                if edge.role != COOCCURRENCE:
                    continue
                if edge.target not in candidate_set:
                    continue
                ew = edge.effective_weight
                if ew > 0:
                    profile.append((edge.target, edge))
                    total_ew += ew

            if total_ew <= 0:
                continue

            if context_positions and answer_position is not None:
                ctx_pos = context_positions.get(ctx_nid)
                if ctx_pos is not None:
                    query_dist = answer_position - ctx_pos
                else:
                    query_dist = None
            else:
                query_dist = None

            for cand_nid, edge in profile:
                content_w = edge.effective_weight / total_ew

                if query_dist is not None:
                    pos_match = edge.position_match(query_dist)
                else:
                    pos_match = 1.0

                logits[cand_nid] += ctx_act * content_w * pos_match

        if not logits or all(v == 0 for v in logits.values()):
            return candidates[0] if candidates else None

        return max(logits, key=logits.get)

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def value_for_node(self, nid: NodeId) -> Any:
        """Reverse lookup: NodeId → original value."""
        for val, n in self._value_to_node.items():
            if n == nid:
                return val
        return None

    def label_for_node(self, nid: NodeId) -> str:
        """Human-readable label."""
        node = self._nodes.get(nid)
        if node is not None and node.label is not None:
            return node.label
        return f"node_{nid}"
