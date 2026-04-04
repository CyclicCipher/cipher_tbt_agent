"""CipherNet core graph — predictive coding on cortical columns.

Every node computes prediction error: the difference between bottom-up
sensory input and top-down predictions. Learning minimizes prediction
error by adjusting edge weights. Credit assignment is automatic —
only edges whose target has non-zero prediction error get updated.

Brain oscillations: different layers operate at different frequencies.
L2/3 (gamma, fast) carries feedforward errors. L5/L6 (beta, slow)
carries feedback predictions. PFC (theta, very slow) maintains WM.
Frequency-dependent decay creates this spectral separation.

Prospective configuration: before learning, activations settle to the
state that minimizes total prediction error with clamped I/O.

Active inference: goals are predictions (clamped output nodes). The
system acts to minimize prediction error between goals and reality.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

SPATIAL = 0    # undirected, metric structure (A <-> B)
TEMPORAL = 1   # directed, transitions/causality (A -> B)
BINDING = 2    # directed, stimulus -> latent position
GATE = 3       # directed, gate -> target (controls retention vs update)

# ---------------------------------------------------------------------------
# Frequency bands (decay rates per brain oscillation band)
# ---------------------------------------------------------------------------
# Gamma: fast response, carries feedforward prediction errors (L2/3, L4)
# Beta:  slow response, carries feedback predictions (L5, L6)
# Theta: very slow, working memory maintenance (PFC)
# Each band's decay determines how quickly a node responds to new input
# vs retains old state. Low decay = fast response. High decay = persistent.

FREQ_DECAY = {
    'gamma': 0.3,   # fast: 70% new input per step
    'beta':  0.7,   # slow: 30% new input per step
    'theta': 0.95,  # very persistent: 5% new input per step
}

# Layer -> frequency band mapping
LAYER_FREQ = {
    4:  'gamma',   # L4: error layer, fast response
    23: 'gamma',   # L2/3: superficial pyramidal, fast errors
    5:  'beta',    # L5: deep pyramidal, slow predictions
    6:  'beta',    # L6: feedback/prediction, slow
}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A position in the latent space with prediction error."""
    id: int
    activation: float = 0.0
    error: float = 0.0         # prediction error (sensory - prediction)
    label: str | None = None
    subgraph: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """A connection between two nodes."""
    source: int
    target: int
    edge_type: int = SPATIAL
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_spatial(self) -> bool:
        return self.edge_type == SPATIAL

    @property
    def is_temporal(self) -> bool:
        return self.edge_type == TEMPORAL

    @property
    def is_binding(self) -> bool:
        return self.edge_type == BINDING


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class Graph:
    """Predictive coding graph.

    step() computes prediction error at each node by splitting incoming
    temporal edges into sensory (feedforward) and prediction (feedback).
    Feedback edges come from nodes with role='feedback' (L6 in columns).

    learn() adjusts weights proportional to target prediction error
    times source activation. Zero error = zero weight change.

    settle() runs step() repeatedly with clamped nodes to find the
    activation state that minimizes total prediction error (prospective
    configuration).
    """

    def __init__(self):
        self._nodes: dict[int, Node] = {}
        self._edges: dict[tuple[int, int, int], Edge] = {}
        self._outgoing: dict[int, list[Edge]] = defaultdict(list)
        self._incoming: dict[int, list[Edge]] = defaultdict(list)
        self._next_id: int = 0
        self._label_to_id: dict[str, int] = {}
        self._subgraphs: dict[str, set[int]] = defaultdict(set)

    # -------------------------------------------------------------------
    # Node operations
    # -------------------------------------------------------------------

    def add_node(self, label: str | None = None,
                 subgraph: str | None = None,
                 **meta) -> int:
        nid = self._next_id
        self._next_id += 1
        node = Node(id=nid, label=label, subgraph=subgraph, meta=meta)
        self._nodes[nid] = node
        if label is not None:
            self._label_to_id[label] = nid
        if subgraph is not None:
            self._subgraphs[subgraph].add(nid)
        return nid

    def get_node(self, nid: int) -> Node | None:
        return self._nodes.get(nid)

    def get_by_label(self, label: str) -> int | None:
        return self._label_to_id.get(label)

    def node_count(self) -> int:
        return len(self._nodes)

    def nodes_in_subgraph(self, name: str) -> set[int]:
        return set(self._subgraphs.get(name, set()))

    def all_subgraphs(self) -> list[str]:
        return list(self._subgraphs.keys())

    # -------------------------------------------------------------------
    # Edge operations
    # -------------------------------------------------------------------

    def add_edge(self, source: int, target: int,
                 edge_type: int = SPATIAL,
                 weight: float = 1.0,
                 **meta) -> Edge:
        key = (source, target, edge_type)
        existing = self._edges.get(key)
        if existing is not None:
            existing.weight = weight
            existing.meta.update(meta)
            return existing

        edge = Edge(source=source, target=target, edge_type=edge_type,
                    weight=weight, meta=meta)
        self._edges[key] = edge
        self._outgoing[source].append(edge)
        self._incoming[target].append(edge)

        if edge_type == SPATIAL and source != target:
            rev_key = (target, source, edge_type)
            if rev_key not in self._edges:
                rev = Edge(source=target, target=source, edge_type=edge_type,
                           weight=weight, meta=meta)
                self._edges[rev_key] = rev
                self._outgoing[target].append(rev)
                self._incoming[source].append(rev)

        return edge

    def get_edge(self, source: int, target: int, edge_type: int = SPATIAL) -> Edge | None:
        return self._edges.get((source, target, edge_type))

    def edges_from(self, nid: int, edge_type: int | None = None) -> list[Edge]:
        edges = self._outgoing.get(nid, [])
        if edge_type is not None:
            return [e for e in edges if e.edge_type == edge_type]
        return list(edges)

    def edges_to(self, nid: int, edge_type: int | None = None) -> list[Edge]:
        edges = self._incoming.get(nid, [])
        if edge_type is not None:
            return [e for e in edges if e.edge_type == edge_type]
        return list(edges)

    def edge_count(self) -> int:
        return len(self._edges)

    def _remove_edge(self, key: tuple[int, int, int]):
        edge = self._edges.pop(key, None)
        if edge is None:
            return
        src, tgt, etype = key
        out_list = self._outgoing.get(src, [])
        self._outgoing[src] = [e for e in out_list if e is not edge]
        in_list = self._incoming.get(tgt, [])
        self._incoming[tgt] = [e for e in in_list if e is not edge]

    # -------------------------------------------------------------------
    # Graph distance
    # -------------------------------------------------------------------

    def spatial_distance(self, a: int, b: int, max_dist: int = 1000) -> int | None:
        if a == b:
            return 0
        visited = {a}
        frontier = [a]
        dist = 0
        while frontier and dist < max_dist:
            dist += 1
            next_frontier = []
            for nid in frontier:
                for edge in self.edges_from(nid, edge_type=SPATIAL):
                    tgt = edge.target
                    if tgt == b:
                        return dist
                    if tgt not in visited:
                        visited.add(tgt)
                        next_frontier.append(tgt)
            frontier = next_frontier
        return None

    # -------------------------------------------------------------------
    # Activation dynamics — predictive coding
    # -------------------------------------------------------------------

    def activate(self, nid: int, level: float = 1.0):
        node = self._nodes.get(nid)
        if node is not None:
            node.activation = min(1.0, level)

    def reset_activations(self):
        for node in self._nodes.values():
            node.activation = 0.0
            node.error = 0.0

    def active_nodes(self, threshold: float = 0.01) -> dict[int, float]:
        return {nid: n.activation for nid, n in self._nodes.items()
                if n.activation >= threshold}

    def total_error(self) -> float:
        """Sum of squared prediction errors across all nodes."""
        return sum(n.error ** 2 for n in self._nodes.values())

    def step(self, default_decay: float = 0.85, threshold: float = 0.01):
        """One timestep with prediction error computation.

        For each node:
        1. GATE: compute gate signal from incoming GATE edges.
        2. Split incoming TEMPORAL into SENSORY vs PREDICTION:
           - Edges from nodes with role='feedback' are PREDICTIONS
           - All other temporal edges are SENSORY
        3. PREDICTION ERROR: error = sensory - prediction
        4. UPDATE: new_act = retain * old + (1-retain) * sensory
           (activation tracks sensory input, NOT prediction)
        5. INHIBITION: negative spatial edges.

        The error field is stored on each node for learn() to use.
        All nodes update simultaneously.
        """
        new_activations: dict[int, float] = {}
        new_errors: dict[int, float] = {}

        for nid, node in self._nodes.items():
            old_act = node.activation

            # 1. GATE signal.
            gate_signal = 0.0
            has_gate = False
            for edge in self._incoming.get(nid, []):
                if edge.edge_type == GATE:
                    src = self._nodes.get(edge.source)
                    if src is not None:
                        gate_signal += edge.weight * src.activation
                        has_gate = True

            if has_gate:
                gate_signal = max(0.0, min(1.0, gate_signal))
                retain = 1.0 - gate_signal
            else:
                # Frequency-dependent decay: layer determines band.
                layer = node.meta.get('layer')
                freq = LAYER_FREQ.get(layer)
                retain = FREQ_DECAY.get(freq, default_decay) if freq else default_decay

            # 2. Split incoming temporal into sensory vs prediction.
            sensory = 0.0
            prediction = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.edge_type != TEMPORAL:
                    continue
                if edge.source == nid:
                    continue  # self-loop handled separately
                src = self._nodes.get(edge.source)
                if src is None or src.activation < 0.001:
                    continue
                signal = edge.weight * src.activation
                # Feedback (L6) edges carry top-down predictions.
                if src.meta.get('role') == 'feedback':
                    prediction += signal
                else:
                    sensory += signal

            # 3. Prediction error.
            error = sensory - prediction

            # 4. Recurrence: activation tracks SENSORY input.
            self_loop_weight = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.edge_type == TEMPORAL and edge.source == nid:
                    self_loop_weight = edge.weight
                    break

            if self_loop_weight > 0 and has_gate and gate_signal < 0.5:
                effective_retain = min(1.0, retain * self_loop_weight + (1.0 - retain) * self_loop_weight)
                new_act = effective_retain * old_act + (1.0 - effective_retain) * sensory
            elif self_loop_weight > 0 and not has_gate:
                new_act = self_loop_weight * old_act + (1.0 - self_loop_weight) * sensory
            else:
                new_act = retain * old_act + (1.0 - retain) * sensory

            # 5. Inhibition.
            inhibition = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.edge_type == SPATIAL and edge.weight < 0:
                    src = self._nodes.get(edge.source)
                    if src is not None and src.activation > 0.001:
                        inhibition += abs(edge.weight) * src.activation

            new_act = max(0.0, new_act - inhibition)
            new_act = min(1.0, new_act)
            if new_act < threshold:
                new_act = 0.0

            new_activations[nid] = new_act
            new_errors[nid] = error

        # Apply synchronously.
        for nid, act in new_activations.items():
            self._nodes[nid].activation = act
            self._nodes[nid].error = new_errors.get(nid, 0.0)

    # -------------------------------------------------------------------
    # Settle — prospective configuration
    # -------------------------------------------------------------------

    def settle(self, n_steps: int = 20, clamp: dict[int, float] | None = None,
               default_decay: float = 0.85, threshold: float = 0.01,
               learn_rate: float = 0.0):
        """Run inference steps to minimize prediction error.

        Clamped nodes (input + desired output) keep their activation
        fixed. Their prediction error is set to (desired - computed),
        which is the TEACHING SIGNAL.

        If learn_rate > 0, weights adjust at EVERY step (simultaneous
        inference and learning, like biological predictive coding).
        Error propagates one hop per step, and weights adjust to reduce
        it. Over N steps, error propagates N hops deep.

        Args:
            n_steps: number of inference iterations
            clamp: {node_id: activation_value} for fixed nodes
            default_decay: retention for non-gated nodes
            threshold: activation cutoff
            learn_rate: if > 0, adjust weights each step (online learning)
        """
        for _ in range(n_steps):
            self.step(default_decay=default_decay, threshold=threshold)
            # Set teaching error at clamped nodes.
            if clamp:
                for nid, val in clamp.items():
                    node = self._nodes.get(nid)
                    if node is not None:
                        node.error = val - node.activation
                        node.activation = val
            # Online learning: adjust weights at every step.
            if learn_rate > 0:
                self.learn(learning_rate=learn_rate, synaptogenesis=False)

    # -------------------------------------------------------------------
    # Learning — error-driven weight updates
    # -------------------------------------------------------------------

    def learn(self, learning_rate: float = 0.01,
              edge_types: set[int] | None = None,
              synaptogenesis: bool = True,
              synapse_threshold: float = 0.3,
              synapse_weight: float = 0.05,
              prune_threshold: float = 0.001):
        """Predictive coding weight update.

        delta_w = learning_rate * target.error * source.activation

        - If target has positive error (surprised by input):
          strengthen edges from active sources (they provide useful signal).
        - If target has negative error (over-predicted):
          weaken edges from active sources (they're over-contributing).
        - If target has zero error (prediction was correct):
          NO weight change. This prevents catastrophic forgetting.

        Synaptogenesis: when a node has high prediction error AND a
        potential source is strongly active, create an edge. The node
        "needs" more input, and this source can provide it.
        """
        if edge_types is None:
            edge_types = {TEMPORAL}

        # --- 1. Error-driven weight adjustment ---
        for (src, tgt, etype), edge in list(self._edges.items()):
            if etype not in edge_types:
                continue
            src_node = self._nodes.get(src)
            tgt_node = self._nodes.get(tgt)
            if src_node is None or tgt_node is None:
                continue
            # Only update if source is active AND target has error.
            if abs(tgt_node.error) < 0.001 or src_node.activation < 0.001:
                continue

            # Error-driven: delta = lr * target_error * source_activation
            delta = learning_rate * tgt_node.error * src_node.activation
            edge.weight += delta
            edge.weight = max(-2.0, min(2.0, edge.weight))

        # --- 2. Synaptogenesis: create edges to high-error nodes ---
        if synaptogenesis:
            high_error = [(nid, n) for nid, n in self._nodes.items()
                          if abs(n.error) >= synapse_threshold]
            high_active = [(nid, n) for nid, n in self._nodes.items()
                           if n.activation >= synapse_threshold]

            for tgt_id, tgt_node in high_error:
                for src_id, src_node in high_active:
                    if src_id == tgt_id:
                        continue
                    # Skip same subgraph.
                    if (src_node.subgraph is not None and
                            src_node.subgraph == tgt_node.subgraph):
                        continue
                    # Create edge from active source to error-ful target.
                    if self.get_edge(src_id, tgt_id, TEMPORAL) is None:
                        w = synapse_weight * abs(tgt_node.error) * src_node.activation
                        self.add_edge(src_id, tgt_id, edge_type=TEMPORAL, weight=w)

        # --- 3. Pruning ---
        to_remove = []
        for key, edge in self._edges.items():
            if edge.edge_type in edge_types and abs(edge.weight) < prune_threshold:
                to_remove.append(key)
        for key in to_remove:
            self._remove_edge(key)

    # -------------------------------------------------------------------
    # Subgraph operations
    # -------------------------------------------------------------------

    def create_subgraph(self, name: str) -> str:
        if name not in self._subgraphs:
            self._subgraphs[name] = set()
        return name

    def merge_subgraph(self, source_graph: 'Graph', source_subgraph: str,
                       target_subgraph: str | None = None) -> dict[int, int]:
        id_map: dict[int, int] = {}
        target_sg = target_subgraph or source_subgraph

        for old_id in source_graph.nodes_in_subgraph(source_subgraph):
            old_node = source_graph.get_node(old_id)
            if old_node is None:
                continue
            new_id = self.add_node(
                label=old_node.label,
                subgraph=target_sg,
                **old_node.meta,
            )
            id_map[old_id] = new_id

        for (src, tgt, etype), edge in source_graph._edges.items():
            if src in id_map and tgt in id_map:
                self.add_edge(
                    id_map[src], id_map[tgt],
                    edge_type=etype,
                    weight=edge.weight,
                    **edge.meta,
                )

        return id_map

    # -------------------------------------------------------------------
    # Column factory
    # -------------------------------------------------------------------

    def create_column(self, name: str,
                      self_loop_weight: float = 0.0,
                      create_relay: bool = True) -> dict[str, int]:
        """Create a predictive coding cortical column.

        Microcircuit (Bastos et al. 2012):
        - L4 (gamma): ERROR layer. Receives feedforward input + L6
          prediction. Computes error = input - prediction.
        - L2/3 (gamma): SUPERFICIAL PYRAMIDAL. Encodes prediction errors.
          Sends errors FORWARD (up) to L4 of next higher area.
        - L5 (beta): DEEP PYRAMIDAL. Encodes conditional expectations.
          Sends predictions BACKWARD (down) to L2/3 of next lower area.
          NOTE: predictions skip L4 (the error layer).
        - L6 (beta): FEEDBACK. Generates intra-column prediction for L4.

        Feedforward relay: L2/3 -> relay -> next area's L4 (errors go UP)
        Feedback relay: receives from higher area's L5 -> this L2/3
        (predictions come DOWN, skip L4)
        """
        sg = f"column:{name}"
        self.create_subgraph(sg)

        l4 = self.add_node(label=f"{name}:L4", subgraph=sg, layer=4, role="input")
        l23 = self.add_node(label=f"{name}:L23", subgraph=sg, layer=23, role="process")
        l5 = self.add_node(label=f"{name}:L5", subgraph=sg, layer=5, role="output")
        l6 = self.add_node(label=f"{name}:L6", subgraph=sg, layer=6, role="feedback")

        # Internal wiring:
        # L4 (error) -> L2/3 (error encoding)
        self.add_edge(l4, l23, edge_type=TEMPORAL, weight=1.0)
        # L2/3 (error) -> L5 (deep pyramidal takes error, forms prediction)
        self.add_edge(l23, l5, edge_type=TEMPORAL, weight=1.0)
        # L5 (prediction) -> L6 (feedback generator)
        self.add_edge(l5, l6, edge_type=TEMPORAL, weight=1.0)
        # L6 (prediction) -> L4 (top-down prediction for error computation)
        # This is the PREDICTION edge: role='feedback' on source means
        # step() treats this as prediction, not sensory input.
        self.add_edge(l6, l4, edge_type=TEMPORAL, weight=0.5)

        # Self-loop on L2/3 for persistence within the gamma band.
        if self_loop_weight > 0:
            self.add_edge(l23, l23, edge_type=TEMPORAL, weight=self_loop_weight)

        result = {"L4": l4, "L23": l23, "L5": l5, "L6": l6, "name": name}

        if create_relay:
            self.create_subgraph("thalamus")
            # Feedforward relay: carries ERRORS up.
            # L2/3 -> relay -> next area's L4
            relay = self.add_node(
                label=f"thalamus:relay:{name}",
                subgraph="thalamus",
                role="relay",
                column=name,
            )
            # L2/3 sends errors to relay (not L5 — L5 sends predictions DOWN)
            self.add_edge(l23, relay, edge_type=TEMPORAL, weight=1.0)
            # Relay feeds next area's L4 (feedforward target)
            self.add_edge(relay, l4, edge_type=TEMPORAL, weight=1.0)
            # L5 still connects to relay for backward prediction routing
            # through thalamus to lower areas' L2/3
            self.add_edge(l5, relay, edge_type=TEMPORAL, weight=0.5)
            result["relay"] = relay

        return result

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "subgraphs": {name: len(nids) for name, nids in self._subgraphs.items()},
        }
