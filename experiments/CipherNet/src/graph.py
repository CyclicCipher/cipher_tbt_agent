"""CipherNet core graph — the learned latent space.

Nodes are positions. Edges encode spatial (metric), temporal (causal),
and binding (stimulus grounding) relationships.

The graph is the analog of grid cells in cortical columns: it provides
the coordinate system in which manifolds (rules) are embedded.

Subgraphs are named, self-contained regions that can be created
independently (fast local training) and integrated later (slow
global consolidation).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

SPATIAL = 0    # undirected, metric structure (A ↔ B)
TEMPORAL = 1   # directed, transitions/causality (A → B)
BINDING = 2    # directed, stimulus → latent position
GATE = 3       # directed, gate → target (controls retention vs update)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A position in the latent space."""
    id: int
    activation: float = 0.0
    label: str | None = None
    # Subgraph membership.
    subgraph: str | None = None
    # Arbitrary metadata.
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
    """The core graph data structure.

    Supports:
    - Node creation with optional subgraph membership
    - Three edge types: spatial, temporal, binding
    - Adjacency indices for O(degree) traversal
    - Subgraph isolation and integration
    - Activation dynamics (spread, decay)
    """

    def __init__(self):
        self._nodes: dict[int, Node] = {}
        self._edges: dict[tuple[int, int, int], Edge] = {}  # (src, tgt, type) -> Edge
        self._outgoing: dict[int, list[Edge]] = defaultdict(list)
        self._incoming: dict[int, list[Edge]] = defaultdict(list)
        self._next_id: int = 0

        # Label index: label -> node id (for fast lookup by name).
        self._label_to_id: dict[str, int] = {}

        # Subgraph index: subgraph_name -> set of node ids.
        self._subgraphs: dict[str, set[int]] = defaultdict(set)

    # -------------------------------------------------------------------
    # Node operations
    # -------------------------------------------------------------------

    def add_node(self, label: str | None = None,
                 subgraph: str | None = None,
                 **meta) -> int:
        """Create a node. Returns its id."""
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
        """Create or update an edge. For spatial edges, also creates
        the reverse direction (undirected)."""
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

        # Spatial edges are undirected: add reverse.
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

    # -------------------------------------------------------------------
    # Graph distance (shortest path via spatial edges)
    # -------------------------------------------------------------------

    def spatial_distance(self, a: int, b: int, max_dist: int = 1000) -> int | None:
        """BFS shortest path via spatial edges. Returns step count or None."""
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
    # Activation dynamics
    # -------------------------------------------------------------------

    def activate(self, nid: int, level: float = 1.0):
        node = self._nodes.get(nid)
        if node is not None:
            node.activation = min(1.0, level)

    def decay(self, factor: float = 0.85, threshold: float = 0.01):
        """Simple decay (legacy). Prefer step() for full dynamics."""
        for node in self._nodes.values():
            node.activation *= factor
            if node.activation < threshold:
                node.activation = 0.0

    def active_nodes(self, threshold: float = 0.01) -> dict[int, float]:
        return {nid: n.activation for nid, n in self._nodes.items()
                if n.activation >= threshold}

    def spread(self, edge_type: int = SPATIAL,
               decay: float = 0.5) -> dict[int, float]:
        """Simple spread (legacy). Prefer step() for full dynamics."""
        predictions: dict[int, float] = defaultdict(float)
        for nid, node in self._nodes.items():
            if node.activation < 0.01:
                continue
            for edge in self.edges_from(nid, edge_type=edge_type):
                predictions[edge.target] += node.activation * edge.weight * decay
        return dict(predictions)

    def step(self, default_decay: float = 0.85, threshold: float = 0.01):
        """One timestep of the full update rule.

        For each node:
        1. GATE: compute gate signal from incoming GATE edges.
           Gate signal > 0 → allow update (retain = 1 - gate_signal).
           No gate edges → use default_decay.
        2. WRITE: sum weighted activations from incoming TEMPORAL edges
           (excluding self-loops and gate edges).
        3. RECURRENCE: blend old state with new input based on gate.
           activation = retain * old + (1 - retain) * new_input
        4. INHIBITION: subtract from competing nodes via negative SPATIAL edges.
        5. SELF-LOOP: if node has a self-loop TEMPORAL edge, apply its weight
           as an ADDITIONAL retention factor (strengthens persistence).

        All nodes update simultaneously (synchronous, like one gamma cycle).
        """
        # Compute new activations for all nodes BEFORE applying any.
        new_activations: dict[int, float] = {}

        for nid, node in self._nodes.items():
            old_act = node.activation

            # 1. GATE signal from incoming GATE edges.
            gate_signal = 0.0
            has_gate = False
            for edge in self._incoming.get(nid, []):
                if edge.edge_type == GATE:
                    src = self._nodes.get(edge.source)
                    if src is not None:
                        gate_signal += edge.weight * src.activation
                        has_gate = True

            # Compute retention factor.
            if has_gate:
                gate_signal = max(0.0, min(1.0, gate_signal))
                retain = 1.0 - gate_signal
            else:
                # No gate edges: use default decay for non-gated nodes.
                retain = default_decay

            # 2. WRITE: new input from incoming TEMPORAL edges.
            new_input = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.edge_type != TEMPORAL:
                    continue
                if edge.source == nid:
                    continue  # skip self-loops (handled separately)
                if edge.edge_type == GATE:
                    continue
                src = self._nodes.get(edge.source)
                if src is not None and src.activation > 0.001:
                    new_input += edge.weight * src.activation

            # 3. RECURRENCE: blend old and new.
            # Self-loop TEMPORAL edge provides additional retention boost.
            self_loop_weight = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.edge_type == TEMPORAL and edge.source == nid:
                    self_loop_weight = edge.weight
                    break

            if self_loop_weight > 0 and has_gate and gate_signal < 0.5:
                # Gate mostly closed + self-loop: strong bistable hold.
                # Self-loop weight amplifies retention.
                effective_retain = min(1.0, retain * self_loop_weight + (1.0 - retain) * self_loop_weight)
                new_act = effective_retain * old_act + (1.0 - effective_retain) * new_input
            elif self_loop_weight > 0 and not has_gate:
                # Self-loop but no gate: persistent node with slow decay.
                # Activation decays by (1 - self_loop_weight) per step.
                new_act = self_loop_weight * old_act + (1.0 - self_loop_weight) * new_input
            else:
                # Standard gated update.
                new_act = retain * old_act + (1.0 - retain) * new_input

            # 4. INHIBITION: negative spatial edges reduce activation.
            inhibition = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.edge_type == SPATIAL and edge.weight < 0:
                    src = self._nodes.get(edge.source)
                    if src is not None and src.activation > 0.001:
                        inhibition += abs(edge.weight) * src.activation

            new_act = max(0.0, new_act - inhibition)

            # Clamp and threshold.
            new_act = min(1.0, new_act)
            if new_act < threshold:
                new_act = 0.0

            new_activations[nid] = new_act

        # Apply all updates simultaneously (synchronous).
        for nid, act in new_activations.items():
            self._nodes[nid].activation = act

    # -------------------------------------------------------------------
    # Subgraph operations
    # -------------------------------------------------------------------

    def create_subgraph(self, name: str) -> str:
        """Register a named subgraph. Returns the name."""
        if name not in self._subgraphs:
            self._subgraphs[name] = set()
        return name

    def merge_subgraph(self, source_graph: 'Graph', source_subgraph: str,
                       target_subgraph: str | None = None) -> dict[int, int]:
        """Import a subgraph from another graph into this one.

        Returns a mapping from source node ids to new node ids in this graph.
        """
        id_map: dict[int, int] = {}
        target_sg = target_subgraph or source_subgraph

        # Copy nodes.
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

        # Copy edges between subgraph nodes.
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
        """Create a new cortical column with standard layer structure.

        Every column gets:
          - L4 (input): receives from thalamus
          - L2/3 (processing): recurrent, learns representations
          - L5 (output): broadcasts to other columns
          - L6 (feedback): prediction error

        If create_relay=True, also creates a thalamic relay node
        and wires it to the column (every column needs a relay to
        connect to the rest of the system).

        Args:
            name: unique name for this column
            self_loop_weight: persistence of L2/3 (0 = no persistence,
                0.95 = strong WM-like persistence)
            create_relay: whether to create a thalamic relay

        Returns:
            dict with node IDs: {"L4": id, "L23": id, "L5": id, "L6": id,
                                  "relay": id (if created)}
        """
        sg = f"column:{name}"
        self.create_subgraph(sg)

        l4 = self.add_node(label=f"{name}:L4", subgraph=sg, layer=4, role="input")
        l23 = self.add_node(label=f"{name}:L23", subgraph=sg, layer=23, role="process")
        l5 = self.add_node(label=f"{name}:L5", subgraph=sg, layer=5, role="output")
        l6 = self.add_node(label=f"{name}:L6", subgraph=sg, layer=6, role="feedback")

        # Standard internal wiring.
        self.add_edge(l4, l23, edge_type=TEMPORAL, weight=1.0)
        self.add_edge(l23, l5, edge_type=TEMPORAL, weight=1.0)
        self.add_edge(l5, l6, edge_type=TEMPORAL, weight=1.0)
        self.add_edge(l6, l4, edge_type=TEMPORAL, weight=0.5)

        # Self-loop for persistence (if requested).
        if self_loop_weight > 0:
            self.add_edge(l23, l23, edge_type=TEMPORAL, weight=self_loop_weight)

        result = {"L4": l4, "L23": l23, "L5": l5, "L6": l6, "name": name}

        # Create thalamic relay.
        if create_relay:
            # The relay goes in the thalamus subgraph.
            self.create_subgraph("thalamus")
            relay = self.add_node(
                label=f"thalamus:relay:{name}",
                subgraph="thalamus",
                role="relay",
                column=name,
            )
            # Relay → column L4 (feedforward, gated by BG→relay).
            self.add_edge(relay, l4, edge_type=TEMPORAL, weight=1.0)
            # Column L5 → relay (output to relay for routing to other columns).
            self.add_edge(l5, relay, edge_type=TEMPORAL, weight=1.0)
            result["relay"] = relay

        return result

    # -------------------------------------------------------------------
    # Edge weight learning (dopamine-modulated Hebbian)
    # -------------------------------------------------------------------

    def learn(self, reward: float, learning_rate: float = 0.1,
              edge_types: set[int] | None = None):
        """Adjust edge weights based on reward signal.

        Dopamine-modulated Hebbian: strengthen edges between co-active
        nodes when reward is positive, weaken when negative.

        delta_w = learning_rate * reward * source_activation * target_activation

        This is how the BG learns when to gate: positive reward after
        correct computation strengthens the Go/NoGo decisions that led to it.

        Args:
            reward: positive (correct) or negative (incorrect) signal
            learning_rate: how much to adjust weights
            edge_types: which edge types to modify (default: TEMPORAL only)
        """
        if edge_types is None:
            edge_types = {TEMPORAL}

        for (src, tgt, etype), edge in self._edges.items():
            if etype not in edge_types:
                continue
            src_node = self._nodes.get(src)
            tgt_node = self._nodes.get(tgt)
            if src_node is None or tgt_node is None:
                continue
            if src_node.activation < 0.01 and tgt_node.activation < 0.01:
                continue

            # Hebbian: co-active nodes' edge is modified by reward.
            delta = learning_rate * reward * src_node.activation * tgt_node.activation
            edge.weight += delta

            # Clamp weights to reasonable range.
            edge.weight = max(-2.0, min(2.0, edge.weight))

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "subgraphs": {name: len(nids) for name, nids in self._subgraphs.items()},
        }
