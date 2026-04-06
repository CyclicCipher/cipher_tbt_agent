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

# Oscillatory timing: how many gamma cycles per phase
BETA_PERIOD = 3     # L5/L6 update every 3 gamma cycles
THETA_PERIOD = 8    # PFC/WM update every 8 gamma cycles
ALPHA_PERIOD = 10   # thalamic relay cycle every 10 gamma cycles

# Segment calcium dynamics
CALCIUM_INCREMENT = 0.05  # calcium added per NMDA spike (segment fires)
CALCIUM_DECAY = 0.8       # calcium multiplier per beta cycle
CALCIUM_MAX = 0.5         # cap: threshold can't exceed half max signal
CALCIUM_THRESHOLD_SCALE = 0.5  # how much calcium raises the segment threshold

# Eligibility trace dynamics (three-factor learning)
ELIGIBILITY_DECAY = 0.9   # trace decays per step (~10 step half-life)
ELIGIBILITY_THRESHOLD = 0.3  # minimum trace for merge consideration

# BAC firing threshold
BAC_APICAL_THRESHOLD = 0.2  # minimum apical activation for burst
BAC_BASAL_THRESHOLD = 0.2   # minimum basal activation for burst
BAC_AMPLIFICATION = 1.5     # burst multiplier when BAC fires


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A position in the latent space with two dendritic compartments.

    Basal: receives feedforward input. Drives the neuron.
           Dendritic segments (AND-gates) live here.
    Apical: receives feedback/context. Amplifies the basal signal.
            Does NOT drive the neuron alone (Larkum 1999).

    Output = basal * (1 + apical_gain)  [apical amplification]
    Burst = basal > threshold AND apical > threshold  [BAC firing]
    """
    id: int
    activation: float = 0.0
    error: float = 0.0         # prediction error (sensory - prediction)
    basal: float = 0.0         # feedforward compartment
    apical: float = 0.0        # feedback/context compartment
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
    eligibility: float = 0.0   # eligibility trace (decaying flag for 3-factor learning)
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
        # Oscillatory clock: increments every step (each step = 1 gamma cycle)
        self._clock: int = 0

        # Segment state: per-segment calcium for metaplasticity.
        # segment_id -> calcium float (0.0 = resting, higher = recently active)
        self._segment_calcium: dict[int, float] = defaultdict(float)

        # Parallel edges: multiple synapses from the same presynaptic
        # neuron to different dendritic branches of the same postsynaptic
        # neuron. Stored separately from _edges (which is keyed by
        # (src, tgt, type) and allows only one). Parallel edges are in
        # the adjacency lists so step() and learn() see them.
        self._parallel_edges: list[Edge] = []

        # Eligibility traces are stored on edges directly (Edge.eligibility).
        # No separate counters needed — traces decay naturally.

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

    def add_parallel_edge(self, source: int, target: int,
                          edge_type: int = TEMPORAL,
                          weight: float = 0.5,
                          segment: int = -1) -> Edge | None:
        """Create a parallel synapse (same src/tgt, different branch).

        Biology: one presynaptic neuron can make multiple synaptic
        contacts on different dendritic branches of the same
        postsynaptic neuron. This enables the same input to
        participate in multiple AND-gates independently.

        Returns None if a parallel edge already exists from this
        source to this target on the requested segment.
        """
        # Check for duplicate: don't create if one already exists
        # from same source to same target on the same segment.
        for pe in self._parallel_edges:
            if (pe.source == source and pe.target == target
                    and pe.segment == segment):
                return None  # already exists
        edge = Edge(source=source, target=target, edge_type=edge_type,
                    weight=quantize_weight(weight), segment=segment)
        if edge.segment < 0:
            edge.segment = self._next_segment
            self._next_segment += 1
        self._parallel_edges.append(edge)
        self._outgoing[source].append(edge)
        self._incoming[target].append(edge)
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
        return len(self._edges) + len(self._parallel_edges)

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
            node.basal = 0.0
            node.apical = 0.0
        # Reset eligibility traces and segment calcium between examples.
        for edge in self._edges.values():
            edge.eligibility = 0.0
        for edge in self._parallel_edges:
            edge.eligibility = 0.0
        self._segment_calcium.clear()

    def active_nodes(self, threshold: float = 0.01) -> dict[int, float]:
        return {nid: n.activation for nid, n in self._nodes.items()
                if n.activation >= threshold}

    def total_error(self) -> float:
        """Sum of squared prediction errors across all nodes."""
        return sum(n.error ** 2 for n in self._nodes.values())

    def _node_freq(self, node: Node) -> str | None:
        """Get the frequency band for a node based on its layer."""
        return LAYER_FREQ.get(node.meta.get('layer'))

    def _should_update(self, node: Node) -> bool:
        """Check if a node should update this clock cycle."""
        freq = self._node_freq(node)
        if freq == 'beta':
            return self._clock % BETA_PERIOD == 0
        if freq == 'theta':
            return self._clock % THETA_PERIOD == 0
        return True  # gamma and untagged: every step

    def _compute_sensory(self, nid: int, node: Node) -> tuple[float, float, float]:
        """Two-compartment dendritic computation.

        BASAL: feedforward input from processing nodes (role='process').
               Dendritic segments with AND-gates + calcium thresholds.
               This DRIVES the neuron.

        APICAL: feedback/context from feedback nodes (role='feedback')
                and from non-column sources (output nodes, etc.).
                This AMPLIFIES the basal signal (gain modulation).

        Returns (basal, apical, error).
        Output = basal * (1 + apical) via BAC-like amplification.
        """
        # BASAL compartment: feedforward, with dendritic segments.
        basal_segments: dict[int, list[float]] = defaultdict(list)
        # APICAL compartment: feedback/context, simple sum.
        apical_total = 0.0

        for edge in self._incoming.get(nid, []):
            if edge.edge_type != TEMPORAL or edge.source == nid or edge.weight < 0:
                continue
            src = self._nodes.get(edge.source)
            if src is None:
                continue
            # Include ALL edges, even inactive (signal=0 for AND-gate).
            # Inactive sources contribute 0 — this is critical for
            # AND-gates where one missing input must kill the segment.
            signal = edge.weight * src.activation if src.activation > 0.001 else 0.0

            # Route to compartment based on source type.
            # ONLY explicit feedback (L6, role='feedback') → APICAL
            # EVERYTHING else → BASAL (feedforward by default)
            if src.meta.get('role') == 'feedback':
                apical_total += signal
            else:
                basal_segments[edge.segment].append(signal)

        # Compute basal: dendritic AND within segments, OR across.
        basal = 0.0
        for seg_id, signals in basal_segments.items():
            calcium = self._segment_calcium.get(seg_id, 0.0)
            seg_threshold = CALCIUM_THRESHOLD_SCALE * calcium
            if len(signals) == 1:
                val = max(0.0, signals[0])
                if val > seg_threshold:
                    basal += val
                    self._segment_calcium[seg_id] = min(
                        CALCIUM_MAX, calcium + CALCIUM_INCREMENT)
            else:
                product = 1.0
                all_positive = True
                for s in signals:
                    if s <= 0.0:
                        all_positive = False
                        break
                    product *= s
                if all_positive:
                    val = product ** (1.0 / len(signals))
                    if val > seg_threshold:
                        basal += val
                        self._segment_calcium[seg_id] = min(
                            CALCIUM_MAX, calcium + CALCIUM_INCREMENT)

        # Store compartment values on the node.
        node.basal = basal
        node.apical = apical_total

        # Apical amplification (Phillips & Larkum 2024):
        # output = basal * (1 + gain). Apical AMPLIFIES, doesn't drive.
        # BAC firing: if both basal AND apical exceed threshold, BURST.
        if basal > BAC_BASAL_THRESHOLD and apical_total > BAC_APICAL_THRESHOLD:
            # BAC burst: amplified output
            sensory = basal * BAC_AMPLIFICATION
        else:
            # Normal: basal drives, apical provides mild gain
            sensory = basal * (1.0 + 0.5 * max(0.0, apical_total))

        # Prediction error: what L6 feedback predicted vs what arrived.
        # L6 feedback is in the apical stream.
        error = max(-1.0, min(1.0, basal - apical_total))
        return sensory, apical_total, error

    def step(self, default_decay: float = 0.85, threshold: float = 0.01,
             inference: bool = False):
        """One gamma cycle of the physics engine.

        Oscillatory: nodes only update when their band's phase arrives.
        Gamma (every step): L4/L23. Beta (every 3): L5/L6. Theta (every 8): PFC.
        PV gamma reset clears L23 after each cycle (prevents accumulation).
        Segment calcium decays at beta rate.

        For each updating node:
        1. GATE: compute gate signal from incoming GATE edges.
        2. Dendritic sensory + prediction error (with calcium thresholds).
        3. UPDATE: PC inference (gradient descent) or Mamba accumulation.
        4. UPDATE (Mamba-style): new_act = decay * old + input
           Input is ADDED to decayed state, NOT blended. This
           preserves signal strength across multiple hops.
           (Contrast with old rule: retain*old + (1-retain)*input
           which kills signal exponentially.)
        5. INHIBITION: negative spatial edges.

        The error field is stored on each node for learn() to use.
        All nodes update simultaneously.
        """
        self._clock += 1
        new_activations: dict[int, float] = {}
        new_errors: dict[int, float] = {}

        for nid, node in self._nodes.items():
            # Oscillatory gating: skip nodes not in this phase.
            if not self._should_update(node):
                continue

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
                # Biology: GPi INHIBITS thalamus. GPi active = gate CLOSED.
                # retain = gate_signal: high GPi = high retention = closed.
                # D1 Go inhibits GPi → gate_signal drops → retain drops → OPEN.
                retain = gate_signal
            else:
                # Frequency-dependent decay: layer determines band.
                layer = node.meta.get('layer')
                freq = LAYER_FREQ.get(layer)
                retain = FREQ_DECAY.get(freq, default_decay) if freq else default_decay

            # 2-3. Two-compartment dendritic computation.
            sensory, prediction, error = self._compute_sensory(nid, node)

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
                # Larkum's principle: apical (downstream) cannot drive
                # without basal (feedforward). If no basal input,
                # downstream error is ignored — nothing to amplify.
                if node.basal < 0.01:
                    downstream_error = 0.0
                inference_rate = 0.1
                new_act = old_act + inference_rate * (error + downstream_error)
            else:
                # FEED MODE (used during token input).
                if has_gate:
                    # GATED nodes: blend old and new based on gate openness.
                    # retain high (GPi active, closed): hold old state.
                    # retain low (GPi inhibited, open): accept new input.
                    new_act = retain * old_act + (1.0 - retain) * sensory
                else:
                    # NON-GATED nodes: Mamba accumulation.
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

            # Set eligibility traces on edges into D1/D2 MSNs.
            # Biology: when cortex activates D1/D2 (gating decision),
            # the synapse is "tagged" for ~2s. Dopamine arriving
            # later converts the tag to plasticity.
            node_role = node.meta.get('role')
            if node_role in ('d1_msn', 'd2_msn') and new_act > 0.1:
                for edge in self._incoming.get(nid, []):
                    if edge.edge_type == TEMPORAL and edge.weight > 0:
                        src = self._nodes.get(edge.source)
                        if src and src.activation > 0.1:
                            edge.eligibility = max(edge.eligibility, 1.0)

        # Apply synchronously.
        for nid, act in new_activations.items():
            self._nodes[nid].activation = act
            self._nodes[nid].error = new_errors.get(nid, 0.0)

        # === Phase-specific operations ===

        # PV GAMMA RESET: partial clear of L23 each gamma cycle.
        # Prevents accumulation/saturation. Only in feed mode.
        if not inference:
            for nid, node in self._nodes.items():
                if (node.meta.get('layer') == 23
                        and node.meta.get('role') == 'process'):
                    # Spare WM-like nodes with strong self-loops.
                    has_strong_loop = any(
                        e.edge_type == TEMPORAL and e.source == nid
                        and e.weight > 0.5
                        for e in self._incoming.get(nid, []))
                    if not has_strong_loop:
                        node.activation *= 0.3

        # BETA PHASE: segment calcium decay.
        if self._clock % BETA_PERIOD == 0:
            for seg_id in list(self._segment_calcium.keys()):
                self._segment_calcium[seg_id] *= CALCIUM_DECAY
                if self._segment_calcium[seg_id] < 0.001:
                    del self._segment_calcium[seg_id]

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
        # Track the MAXIMUM teaching error at each clamped node across
        # all settle steps. The first few steps have the largest error
        # (before the network converges to the clamped state). This
        # error is what learning should use — not the final near-zero
        # error after convergence.
        max_clamp_errors: dict[int, float] = {}

        for i in range(n_steps):
            self.step(default_decay=default_decay, threshold=threshold,
                      inference=True)
            # Set teaching error at clamped nodes.
            if clamp:
                for nid, val in clamp.items():
                    node = self._nodes.get(nid)
                    if node is not None:
                        teaching_err = val - node.activation
                        node.error = teaching_err
                        node.activation = val
                        # Track max error magnitude for learning.
                        if abs(teaching_err) > abs(max_clamp_errors.get(nid, 0)):
                            max_clamp_errors[nid] = teaching_err
            # Online learning: adjust weights at every step.
            if learn_rate > 0:
                self.learn(learning_rate=learn_rate, synaptogenesis=False)

        # Restore the max teaching errors at clamped nodes for learn().
        # The final settle errors are near-zero (network converged),
        # but the INITIAL errors encode what the network couldn't produce.
        for nid, err in max_clamp_errors.items():
            node = self._nodes.get(nid)
            if node is not None:
                node.error = err

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
        # Include both primary and parallel edges.
        all_edges = list(self._edges.items()) + [
            ((e.source, e.target, e.edge_type), e)
            for e in self._parallel_edges]
        for (src, tgt, etype), edge in all_edges:
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

        # --- 2. Eligibility-trace-based dendritic segment merging ---
        #
        # Biology (Gerstner et al. 2018): three-factor learning.
        # 1. Pre+post co-fire → set eligibility trace (flag, not weight change)
        # 2. Trace decays over ~10 steps
        # 3. Neuromodulatory signal (error) arrives → trace converts to change
        #
        # For segment merging: edges that are eligible (recently active
        # during error) and co-eligible with a partner get merged.
        # ONLY feedforward (basal) edges participate — apical excluded.

        # Decay all eligibility traces.
        for (src, tgt, etype), edge in self._edges.items():
            edge.eligibility *= ELIGIBILITY_DECAY
        for edge in self._parallel_edges:
            edge.eligibility *= ELIGIBILITY_DECAY

        # Set eligibility for active FEEDFORWARD edges at error targets.
        for (src, tgt, etype), edge in self._edges.items():
            if etype not in edge_types or edge.weight < 0:
                continue
            src_node = self._nodes.get(src)
            tgt_node = self._nodes.get(tgt)
            if src_node is None or tgt_node is None:
                continue
            if src_node.activation < 0.1:
                continue
            if src_node.subgraph and src_node.subgraph == tgt_node.subgraph:
                continue
            # ONLY basal (feedforward from L23) edges.
            if src_node.meta.get('role') != 'process':
                continue

            if tgt_node.error < -0.05:
                # Negative error: this edge is causing a false positive.
                # Set negative eligibility (wants AND protection).
                edge.eligibility = min(edge.eligibility, -1.0)
            elif tgt_node.error > 0.05:
                # Positive error: this edge is helping but not enough.
                # Set positive eligibility (co-firing is useful).
                edge.eligibility = max(edge.eligibility, 1.0)

        # Check for merge: pairs of edges to the same target where
        # one has negative eligibility (conflict) and both have
        # positive eligibility in some recent step (co-success).
        # The eligibility trace naturally time-windows this.
        target_eligible: dict[int, list] = defaultdict(list)
        for (src, tgt, etype), edge in self._edges.items():
            if abs(edge.eligibility) > ELIGIBILITY_THRESHOLD:
                src_node = self._nodes.get(src)
                if src_node and src_node.meta.get('role') == 'process':
                    target_eligible[tgt].append(((src, tgt, etype), edge))

        for tgt_id, eligible_list in target_eligible.items():
            if len(eligible_list) < 2:
                continue
            # Find pairs where BOTH have positive eligibility
            # (both co-fired during a positive-error example).
            # At least one must ALSO have had recent negative eligibility
            # (solo conflict from a negative example).
            # Since eligibility resets between examples, positive traces
            # only come from the CURRENT example.
            pos_edges = [(k, e) for k, e in eligible_list
                         if e.eligibility > ELIGIBILITY_THRESHOLD]
            if len(pos_edges) < 2:
                continue

            # All pairs of positively eligible edges are merge candidates.
            for i in range(len(pos_edges)):
                for j in range(i + 1, len(pos_edges)):
                    ek_a, ea = pos_edges[i]
                    ek_b, eb = pos_edges[j]
                    if ea.segment == eb.segment:
                        continue

                    a_seg_size = sum(1 for e in self._incoming.get(tgt_id, [])
                                    if e.segment == ea.segment and e is not ea)
                    b_seg_size = sum(1 for e in self._incoming.get(tgt_id, [])
                                    if e.segment == eb.segment and e is not eb)

                    if a_seg_size > 0 and b_seg_size == 0:
                        self.add_parallel_edge(
                            ea.source, ea.target, ea.edge_type,
                            weight=ea.weight, segment=eb.segment)
                    elif b_seg_size > 0 and a_seg_size == 0:
                        self.add_parallel_edge(
                            eb.source, eb.target, eb.edge_type,
                            weight=eb.weight, segment=ea.segment)
                    elif a_seg_size > 0 and b_seg_size > 0:
                        if a_seg_size <= b_seg_size:
                            self.add_parallel_edge(
                                ea.source, ea.target, ea.edge_type,
                                weight=ea.weight, segment=eb.segment)
                        else:
                            self.add_parallel_edge(
                                eb.source, eb.target, eb.edge_type,
                                weight=eb.weight, segment=ea.segment)
                    else:
                        eb.segment = ea.segment
                    ea.eligibility = 0.0
                    eb.eligibility = 0.0

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
