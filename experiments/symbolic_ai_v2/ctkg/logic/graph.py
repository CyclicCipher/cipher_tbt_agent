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
    """A directed edge with Hebbian weight.

    Simple additive learning: strengthen() nudges toward +1,
    weaken() nudges toward -1. Weight stays in [-1, 1] via
    asymptotic update. No alpha/beta, no posterior, no confidence.
    """
    __slots__ = ('source', 'target', 'weight', 'role', 'count',
                 '_dist_sum', '_dist_sq_sum', '_dist_count')

    def __init__(self, source: NodeId, target: NodeId,
                 weight: float = 0.0, role: int = COOCCURRENCE):
        self.source = source
        self.target = target
        self.weight = weight
        self.role = role
        self.count = 0  # number of observations (for diagnostics)
        self._dist_sum = 0.0
        self._dist_sq_sum = 0.0
        self._dist_count = 0

    @property
    def effective_weight(self) -> float:
        """Edge weight. Same as .weight (sigma removed)."""
        return self.weight

    def strengthen(self, rate: float = 0.1) -> None:
        """Nudge weight toward +1. Asymptotic: large weights change less."""
        self.weight += rate * (1.0 - self.weight)
        self.count += 1

    def weaken(self, rate: float = 0.1) -> None:
        """Nudge weight toward -1. Asymptotic: large negative weights change less."""
        self.weight -= rate * (1.0 + self.weight)
        self.count += 1

    def observe_distance(self, dist: int) -> None:
        """Record the relative position at which source→target co-occurred."""
        self._dist_sum += dist
        self._dist_sq_sum += dist * dist
        self._dist_count += 1

    @property
    def mean_distance(self) -> float:
        if self._dist_count == 0:
            return 0.0
        return self._dist_sum / self._dist_count

    @property
    def var_distance(self) -> float:
        if self._dist_count < 2:
            return 1.0
        mu = self._dist_sum / self._dist_count
        return max(0.1, self._dist_sq_sum / self._dist_count - mu * mu)

    def position_match(self, query_dist: int) -> float:
        """How well does query_dist match this edge's typical distance?

        Returns (0, 1]: 1.0 = perfect match, Gaussian decay.
        No observations → 1.0 (no position bias).
        """
        if self._dist_count == 0:
            return 1.0
        mu = self.mean_distance
        var = self.var_distance
        return math.exp(-0.5 * (query_dist - mu) ** 2 / var)

    def __repr__(self) -> str:
        return (f"Edge(source={self.source}, target={self.target}, "
                f"weight={self.weight:.3f}, role={self.role})")


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
        # PMI cache: computed during consolidation, used by select_action.
        # Maps (src, tgt) → PMI score. Positive = specifically associated.
        self._pmi: dict[tuple[NodeId, NodeId], float] = {}
        # IDF cache: inverse document frequency per node.
        # High IDF = rare token (informative). Low IDF = common (noise).
        self._idf: dict[NodeId, float] = {}
        # Discovered successor map: populated by initial algebra discovery.
        self._discovered_succ: dict[NodeId, NodeId] = {}
        # Concept embeddings: continuous membership vectors.
        # _embeddings[nid] = list[float] of length |concepts|.
        # _embedding_concepts = list of concept extents (dimension labels).
        self._embeddings: dict[NodeId, list[float]] = {}
        self._embedding_concepts: list[frozenset[NodeId]] = []

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

        New edges start at weight=0.
        Maintains both _outgoing and _incoming adjacency indices.
        """
        key = (src, tgt)
        existing = self._edges.get(key)
        if existing is not None:
            return existing
        edge = Edge(source=src, target=tgt, weight=0.0, role=role)
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

    # edges_to already defined above using _incoming index — O(degree)

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
                for edge in self._incoming.get(nid, ()):
                    w = edge.effective_weight
                    if w <= 0:
                        continue
                    src_act = fixed.get(edge.source, current.get(edge.source, 0.0))
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
    # Multi-hop co-occurrence spread
    # -------------------------------------------------------------------

    def spread_cooccurrence(
        self,
        seeds: dict[NodeId, float],
        hops: int = 4,
        decay: float = 0.7,
    ) -> dict[NodeId, float]:
        """Iterative co-occurrence spread from seed nodes.

        Each hop: spread along positive-weight FORWARD co-occurrence edges.
        Returns {NodeId: activation} for all reachable nodes.
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
    # Enriched spread: PMI × position_match (the enriched Kan extension)
    # -------------------------------------------------------------------

    def spread_enriched(
        self,
        seeds: dict[NodeId, float],
        candidates: set[NodeId] | None = None,
        hops: int = 3,
        decay: float = 0.8,
    ) -> dict[NodeId, float]:
        """Multi-hop spread weighted by PMI and positional distance.

        This is the enriched Kan extension: for each target, compute
        the weighted colimit over all paths from seeds to target, where
        the weight of each edge is PMI × position_match.

        PMI measures specific association (filters hubs). Position_match
        measures whether the positional distance along this edge matches
        the typical distance (filters wrong-distance co-occurrences).

        Each hop tracks cumulative distance from the seed. At hop k,
        the expected distance from seed is k+1 (adjacent tokens are
        distance 1, two hops = distance 2, etc.).

        Parameters
        ----------
        seeds : {NodeId: activation} — starting points.
        candidates : if provided, only accumulate scores for these nodes.
        hops : number of spread iterations.
        decay : per-hop decay (prevents distant paths from dominating).

        Returns {NodeId: score} for all reachable nodes (or candidates only).
        """
        # Current frontier: {NodeId: (score, cumulative_distance)}
        current: dict[NodeId, tuple[float, int]] = {
            nid: (act, 0) for nid, act in seeds.items()
        }
        accumulated: dict[NodeId, float] = {}

        for hop in range(hops):
            next_frontier: dict[NodeId, tuple[float, int]] = {}
            expected_dist = hop + 1  # distance from seed at this hop

            for nid, (act, cum_dist) in current.items():
                if act < 0.01:
                    continue

                for edge in self._outgoing.get(nid, ()):
                    if edge.role != COOCCURRENCE:
                        continue

                    tgt = edge.target

                    # PMI gate: only follow specifically associated edges.
                    pmi_score = self._pmi.get((nid, tgt), 0.0)
                    if pmi_score <= 0:
                        continue

                    # Position gate: does this edge's typical distance
                    # match what we expect at this hop?
                    pos_match = edge.position_match(expected_dist)

                    # Combined weight: PMI × position_match × decay.
                    contrib = act * pmi_score * pos_match * decay

                    if contrib > 0.01:
                        existing = next_frontier.get(tgt)
                        if existing is None or contrib > existing[0]:
                            next_frontier[tgt] = (contrib, expected_dist)

            current = next_frontier

            # Accumulate: for each reached node, keep the max score
            # across all hops (earlier hops = shorter paths = typically better).
            for nid, (score, _) in current.items():
                if candidates is not None and nid not in candidates:
                    continue
                accumulated[nid] = max(accumulated.get(nid, 0.0), score)

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

        2. WEIGHT UPDATE PHASE: update edge weights based on the settled
           error pattern. Edges that contributed to correct predictions get
           strengthened; edges that contributed to errors get weakened.

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

        # --- Phase 2: Weight update (Hebbian) ---
        # Use the settled error pattern to update edges.

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
                credit = edge_credit.get((src_nid, tgt_nid), 0.0)

                if tgt_observed:
                    rate = 0.1 * (1.0 + min(abs(credit), 2.0) * 0.5)
                    edge.strengthen(rate)
                    p = max(abs(tgt_predicted), 0.01)
                    surprise += -math.log(min(p, 1.0))
                else:
                    if edge.role == TRANSITION:
                        rate = 0.1 * (1.0 + min(abs(credit), 2.0) * 0.5)
                        edge.weaken(rate)
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
                edge.strengthen(0.3)
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

    def edge_observation_count(self, src: NodeId, tgt: NodeId) -> int:
        """Return how many times this edge has been observed."""
        e = self.edge(src, tgt)
        return e.count if e is not None else 0

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
    # Action selection: attention-based
    # -------------------------------------------------------------------

    def select_action(
        self,
        candidates: list[NodeId],
        fixed_context: dict[NodeId, float] | None = None,
        context_positions: dict[NodeId, int] | None = None,
        answer_position: int | None = None,
    ) -> NodeId | None:
        """Select an action via discovered structure, enriched spread, or fallback.

        Four layers, in priority order:
        1. **Successor computation**: direct chain walk (1 hop). If exactly
           one context token's successor is a candidate, return it.
        2. **Enriched spread**: multi-hop PMI × position_match spread from
           context to candidates. This is the enriched Kan extension — it
           finds the best candidate by following paths through intermediate
           nodes, weighted by specific association and positional match.
           Handles cases like succ(0) where the direct successor is a
           structural token (next_is) but the 2-hop path through next_is
           reaches the correct digit (1).
        3. **PMI attention**: discriminativeness-weighted direct PMI.
        4. **Co-occurrence fallback**: raw edge weight, normalized.

        Returns the highest-scoring candidate, or None.
        """
        if not candidates:
            return None

        context = fixed_context if fixed_context is not None else self.active_nodes()
        candidate_set = set(candidates)

        # --- Layer 0: Successor computation (1-hop chain walk) ---
        if self._discovered_succ:
            succ_hits: list[NodeId] = []
            for ctx_nid in context:
                succ_nid = self._discovered_succ.get(ctx_nid)
                if succ_nid is not None and succ_nid in candidate_set:
                    succ_hits.append(succ_nid)
            unique_hits = list(dict.fromkeys(succ_hits))
            if len(unique_hits) == 1:
                return unique_hits[0]

        # Layer 0.5 (embedding-based successor) is implemented in
        # embedding.py but not wired in here yet — the 189-dimensional
        # concept vectors are too noisy for reliable displacement matching.
        # Needs dimensionality reduction or concept selection first.

        # --- Layer 1: Enriched spread (multi-hop PMI × position) ---
        # Only fire if the best candidate is clearly better than the rest.
        # Margin threshold prevents noisy early-training PMI from dominating.
        if len(self._pmi) > 0:
            enriched_scores = self.spread_enriched(
                context, candidates=candidate_set, hops=3,
            )
            if enriched_scores:
                sorted_scores = sorted(enriched_scores.values(), reverse=True)
                best_score = sorted_scores[0]
                second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
                # Require the best to be at least 2x the second.
                if best_score > 0.05 and best_score > second_score * 2:
                    best = max(enriched_scores, key=enriched_scores.get)
                    return best

        # --- Layer 2: PMI attention (discriminativeness-weighted) ---
        if context_positions and answer_position is None:
            answer_position = max(context_positions.values()) + 1

        logits: dict[NodeId, float] = {nid: 0.0 for nid in candidates}

        use_pmi = len(self._pmi) > 0

        for ctx_nid, ctx_act in context.items():
            if ctx_act <= 0:
                continue

            if use_pmi:
                n_positive = sum(
                    1 for cand_nid in candidates
                    if self._pmi.get((ctx_nid, cand_nid), 0.0) > 0
                )
                if n_positive == 0:
                    continue
                disc_weight = 1.0 / n_positive

                for cand_nid in candidates:
                    pmi_score = self._pmi.get((ctx_nid, cand_nid), 0.0)
                    if pmi_score > 0:
                        logits[cand_nid] += ctx_act * pmi_score * disc_weight
            else:
                # Fallback: raw co-occurrence weight, normalised.
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
                    query_dist = (answer_position - ctx_pos) if ctx_pos is not None else None
                else:
                    query_dist = None

                for cand_nid, edge in profile:
                    content_w = edge.effective_weight / total_ew
                    pos_match = edge.position_match(query_dist) if query_dist is not None else 1.0
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
