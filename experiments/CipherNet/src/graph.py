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

import math
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
# Synaptic weight quantization (8-bit precision)
# ---------------------------------------------------------------------------
# Biology: ~4.7 bits per synapse (24 distinguishable sizes via AMPA receptor
# count). We use 8 bits for computational convenience (256 levels).
# Excitatory: [0, 1.0] in 255 steps. Inhibitory: [-1.0, 0] in 255 steps.
# Weight changes smaller than one quantum have no effect (like inserting
# fewer than one receptor). This provides natural regularization.

WEIGHT_QUANTA = 255  # 8-bit: 0..255 levels
WEIGHT_STEP = 1.0 / WEIGHT_QUANTA  # ≈ 0.00392


def quantize_weight(w: float) -> float:
    """Quantize a weight to 8-bit precision in [-1.0, 1.0]."""
    w = max(-1.0, min(1.0, w))
    # Round to nearest quantum.
    return round(w / WEIGHT_STEP) * WEIGHT_STEP

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
    """A connection between two nodes, assigned to a dendritic segment.

    Dendritic computation:
    - Edges on the SAME segment are multiplicative (AND-like).
      All must be active for the segment to fire.
    - Edges on DIFFERENT segments are additive (OR-like).
      Any segment can fire the node.

    Default: each edge gets a unique auto-incrementing segment ID,
    making the computation purely additive (backwards compatible).
    Learning merges segments when conjunction is useful.
    """
    source: int
    target: int
    edge_type: int = SPATIAL
    weight: float = 1.0
    segment: int = -1          # dendritic segment (-1 = auto-assign unique)
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
        self._next_segment: int = 0  # auto-increment for unique segments
        self._label_to_id: dict[str, int] = {}
        self._subgraphs: dict[str, set[int]] = defaultdict(set)
        # Co-activation counter for dendritic segment merging.
        # Key: frozenset of two edge keys (src,tgt,type).
        # Value: count of episodes where both sources were active.
        self._coactivation: dict[frozenset, int] = defaultdict(int)

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
                 segment: int = -1,
                 **meta) -> Edge:
        key = (source, target, edge_type)
        existing = self._edges.get(key)
        if existing is not None:
            existing.weight = weight
            existing.meta.update(meta)
            return existing

        edge = Edge(source=source, target=target, edge_type=edge_type,
                    weight=quantize_weight(weight), segment=segment, meta=meta)
        # Auto-assign unique segment if not specified.
        if edge.segment < 0:
            edge.segment = self._next_segment
            self._next_segment += 1
        self._edges[key] = edge
        self._outgoing[source].append(edge)
        self._incoming[target].append(edge)

        if edge_type == SPATIAL and source != target:
            rev_key = (target, source, edge_type)
            if rev_key not in self._edges:
                rev = Edge(source=target, target=source, edge_type=edge_type,
                           weight=weight, meta=meta)
                if rev.segment < 0:
                    rev.segment = self._next_segment
                    self._next_segment += 1
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

    def step(self, default_decay: float = 0.85, threshold: float = 0.01,
             inference: bool = False):
        """One timestep: Mamba accumulation (feed) or PC inference (settle).

        For each node:
        1. GATE: compute gate signal from incoming GATE edges.
        2. Split incoming TEMPORAL into SENSORY vs PREDICTION
           (dendritic segments: AND within, OR across).
        3. PREDICTION ERROR: error = sensory - prediction
        4. UPDATE (Mamba-style): new_act = decay * old + input
           Input is ADDED to decayed state, NOT blended. This
           preserves signal strength across multiple hops.
           (Contrast with old rule: retain*old + (1-retain)*input
           which kills signal exponentially.)
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

            # 2. Dendritic computation: group incoming temporal edges
            #    by segment, compute multiplicatively within segments
            #    (AND-like), sum across segments (OR-like).
            #    Split into sensory vs prediction streams.
            #    ALL edges in a segment are included (even inactive ones
            #    contribute 0), so AND requires ALL inputs active.
            sensory_segments: dict[int, list[float]] = defaultdict(list)
            prediction_total = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.edge_type != TEMPORAL:
                    continue
                if edge.source == nid:
                    continue  # self-loop handled separately
                src = self._nodes.get(edge.source)
                if src is None:
                    continue
                # Skip negative temporal edges (handled as inhibition in step 5).
                if edge.weight < 0:
                    continue
                # Include ALL edges, even inactive (signal=0 for AND).
                signal = edge.weight * src.activation if src.activation > 0.001 else 0.0
                # Feedback (L6) edges carry top-down predictions
                # (not dendritic — predictions are always summed).
                if src.meta.get('role') == 'feedback':
                    prediction_total += signal
                else:
                    sensory_segments[edge.segment].append(signal)

            # Compute sensory input: dendritic AND within segments,
            # OR across segments.
            sensory = 0.0
            for seg_id, signals in sensory_segments.items():
                if len(signals) == 1:
                    # Single-edge segment: pass through
                    sensory += max(0.0, signals[0])
                else:
                    # Multi-edge segment: AND-like (geometric mean)
                    # If ANY input is 0, the whole segment output is 0.
                    product = 1.0
                    all_positive = True
                    for s in signals:
                        if s <= 0.0:
                            all_positive = False
                            break
                        product *= s
                    if all_positive:
                        sensory += product ** (1.0 / len(signals))
                    # else: segment output = 0 (AND not satisfied)

            # 3. Prediction error (clamped to prevent explosion).
            error = sensory - prediction_total
            error = max(-1.0, min(1.0, error))

            # 4. Update: two modes.

            self_loop_weight = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.edge_type == TEMPORAL and edge.source == nid:
                    self_loop_weight = edge.weight
                    break

            if has_gate:
                decay = retain
            elif self_loop_weight > 0:
                decay = self_loop_weight
            else:
                decay = retain

            if inference:
                # PC INFERENCE MODE (used during settle):
                # dμ/dt = -ε_local + Σ(W · ε_downstream)
                #
                # Pure gradient descent on prediction error energy.
                # No sensory accumulation — the error already captures
                # the mismatch between sensory input and prediction.
                # Each step propagates credit one hop deeper.
                downstream_error = 0.0
                for edge in self._outgoing.get(nid, []):
                    if edge.edge_type not in (TEMPORAL, BINDING):
                        continue
                    if edge.weight < 0:
                        continue
                    if edge.source == nid:
                        continue
                    tgt = self._nodes.get(edge.target)
                    if tgt is not None and abs(tgt.error) > 0.001:
                        downstream_error += edge.weight * tgt.error

                # PC value update: adjust to minimize total error.
                # +error: increase activation when getting unexpected
                #   input (positive error = sensory > prediction).
                #   This EXPLAINS the input by increasing the representation.
                # +downstream: increase when downstream needs more signal.
                inference_rate = 0.1
                new_act = old_act + inference_rate * (error + downstream_error)
            else:
                # FEED MODE (used during token input):
                # Mamba-style accumulation. Signal ADDS to state.
                new_act = decay * old_act + sensory

            # 5. Inhibition from negative SPATIAL edges AND negative
            #    TEMPORAL edges. Negative temporal = directed inhibition
            #    (inhibitory interneurons). Negative spatial = lateral
            #    inhibition (undirected competition).
            inhibition = 0.0
            for edge in self._incoming.get(nid, []):
                if edge.weight < 0 and edge.edge_type in (SPATIAL, TEMPORAL):
                    if edge.source == nid:
                        continue  # skip negative self-loops
                    src = self._nodes.get(edge.source)
                    if src is not None and src.activation > 0.001:
                        inhibition += abs(edge.weight) * src.activation

            new_act = max(0.0, new_act - inhibition)
            # Smooth compression (tanh): preserves relative differences
            # at high activation instead of hard-clamping to [0,1].
            # Like a real neuron's firing rate saturation curve.
            if new_act > 1.0:
                new_act = math.tanh(new_act)
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
            self.step(default_decay=default_decay, threshold=threshold,
                      inference=True)
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
    # Learning — local predictive coding weight updates
    # -------------------------------------------------------------------

    def learn(self, learning_rate: float = 0.01,
              edge_types: set[int] | None = None,
              synaptogenesis: bool = True,
              synapse_threshold: float = 0.3,
              synapse_weight: float = 0.05,
              prune_threshold: float = 0.001,
              weight_decay: float = 0.001):
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
            # Protect intra-subgraph structural edges from learning.
            # Column internal wiring (L4→L23→L5→L6) must not be modified.
            if (src_node.subgraph is not None and
                    src_node.subgraph == tgt_node.subgraph):
                continue

            # Error-driven: delta = lr * target_error * source_activation
            delta = learning_rate * tgt_node.error * src_node.activation
            edge.weight *= (1.0 - weight_decay)
            edge.weight += delta
            # Quantize to 8-bit precision, clamped to [-1, 1].
            edge.weight = quantize_weight(edge.weight)

        # --- 2. Dendritic segment merging via co-activation tracking ---
        # Based on Bhatt et al. (2015): synapses that repeatedly co-fire
        # within a learning window migrate onto the same dendritic branch.
        # We track co-activation counts for each edge pair to the same
        # target. When the count exceeds merge_threshold, we merge their
        # segments (making them conjunctive/AND).
        merge_threshold = 10  # co-activations needed before merging
        coact_threshold = 0.05  # minimum activation to count as "active"

        # Group active incoming edges by target node.
        target_active_edges: dict[int, list[tuple]] = defaultdict(list)
        for (src, tgt, etype), edge in self._edges.items():
            if etype not in edge_types:
                continue
            src_node = self._nodes.get(src)
            tgt_node = self._nodes.get(tgt)
            if src_node is None or tgt_node is None:
                continue
            if src_node.activation < coact_threshold:
                continue
            if abs(tgt_node.error) < 0.01:
                continue  # only track at nodes with prediction error
            target_active_edges[tgt].append((src, tgt, etype))

        # For each target with 2+ active incoming edges, update counts.
        for tgt_id, edge_keys in target_active_edges.items():
            if len(edge_keys) < 2:
                continue
            # Update co-activation counts for all pairs.
            for i in range(len(edge_keys)):
                for j in range(i + 1, len(edge_keys)):
                    pair_key = frozenset((edge_keys[i], edge_keys[j]))
                    self._coactivation[pair_key] += 1
                    # Check if ready to merge.
                    if self._coactivation[pair_key] >= merge_threshold:
                        edge_a = self._edges.get(edge_keys[i])
                        edge_b = self._edges.get(edge_keys[j])
                        if edge_a and edge_b and edge_a.segment != edge_b.segment:
                            # Merge: move B onto A's segment.
                            old_seg = edge_b.segment
                            edge_b.segment = edge_a.segment
                            # Reset counter (don't merge again immediately).
                            self._coactivation[pair_key] = 0

        # --- 3. Synaptogenesis: create edges to high-error nodes ---
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
