"""Active inference: free energy as MDL description length.

Free energy (BLUEPRINT.md §"Active inference"):
  F = description_length(observations | model) + description_length(model)
    = -log2 P(observations | edges) + |edges| + |rules|

Minimising F simultaneously drives:
  - Perception:    update edge counts to better predict observations
  - Compression:   dissolve rules used only once (rule utility invariant)
  - Action (agent): choose next observation to maximise expected info gain

Functions:
  prediction_error(mg, context_id, etype, tgt_id)  — -log2 P(tgt | ctx, etype)
  free_energy(mg)                                   — MDL cost of current model
  expected_info_gain(mg, context_id, etype)         — EIG for choosing an edge

Classes:
  ActiveInferenceTracker  — wraps MorphismGraph; registers an on_observe hook
                            so PE is computed BEFORE every model update.
                            Works even when code calls mg.observe() directly
                            (not through the tracker), because the hook fires
                            inside MorphismGraph.observe().

Usage:
    topo = sequence_1d()
    mg   = MorphismGraph()
    ait  = ActiveInferenceTracker(mg)
    for text in corpus:
        ait.observe_sequence(text, topo)
        ait.flush()
    print(ait.log_line())
    print(f"free_energy = {free_energy(mg):.1f} bits")
"""

from __future__ import annotations

import math
from typing import Optional

from ..core.morphism import MorphismGraph
from ..core.predict import predict, _marginal_dist


# ── Prediction error ──────────────────────────────────────────────────────────

def prediction_error(
    mg: MorphismGraph,
    context_id: int,
    etype: int,
    tgt_id: int,
) -> float:
    """Return -log2 P(tgt_id | context_id, etype) in bits.

    This is the surprisal of the observation tgt_id given the current model.
    High surprisal → this triple is a candidate for a new composition (if it
    recurs, it will trigger create_composition automatically in observe()).
    """
    dist = mg.predict_dist(context_id, etype)
    if not dist:
        dist = _marginal_dist(mg, etype)

    p = dist.get(tgt_id, 0.0)
    if p <= 0.0:
        # Unseen transition: assign a conservative floor probability
        n_known = max(len(dist) + 1, 1)
        p = 1.0 / (n_known * 10)

    return -math.log2(p)


# ── Free energy / MDL cost ────────────────────────────────────────────────────

def free_energy(mg: MorphismGraph) -> float:
    """Compute the MDL description length of the current model in bits.

    F = observation_cost + model_cost

    observation_cost: total bits needed to encode all observed edges under
      the current model.  Approximated as sum of -log2(edge_freq/total_edges)
      for each distinct (src, etype, tgt).

    model_cost: number of distinct rules × average bits per rule.
      Each rule encodes (left_id, etype, right_id): approximately
      log2(n_atoms + n_compositions) × 2 + log2(n_edge_types) bits.

    Lower is better.  Decreasing F means the model is compressing better.
    """
    total_edges = sum(mg.edges.values())
    if total_edges == 0:
        return 0.0

    # Observation cost: negative log-likelihood of edge distribution
    obs_cost = 0.0
    for cnt in mg.edges.values():
        p = cnt / total_edges
        obs_cost += cnt * (-math.log2(p))

    # Model cost: description length of the grammar rules
    n_syms = mg.n_symbols()
    n_etypes = max(
        1,
        max((et for (_, et, _) in mg.edges), default=0) + 1
    )
    bits_per_id    = math.log2(max(n_syms, 2))
    bits_per_etype = math.log2(max(n_etypes, 2))
    bits_per_rule  = 2 * bits_per_id + bits_per_etype
    model_cost = mg.n_compositions() * bits_per_rule

    return obs_cost + model_cost


# ── Expected information gain (action selection) ──────────────────────────────

def expected_info_gain(
    mg: MorphismGraph,
    context_id: int,
    etype: int,
) -> float:
    """Expected reduction in entropy if the agent follows edge etype from context.

    EIG(etype from context) = H(prediction) - E[H(prediction after observing)]

    The first term is the current entropy of P(next | context, etype).
    The second term is the expected post-observation entropy, approximated
    as zero (we will know the outcome with certainty once observed).
    So EIG ≈ H(prediction) = entropy of the current distribution.

    The agent should choose the edge type with the highest EIG to maximise
    information gain (epistemic foraging).  This implements the epistemic
    component of active inference.
    """
    dist = mg.predict_dist(context_id, etype)
    if not dist:
        dist = _marginal_dist(mg, etype)
    if not dist:
        # No information available → maximum uncertainty, estimate as log2(n_atoms)
        return math.log2(max(mg.n_atoms(), 2))

    # Shannon entropy of the current predictive distribution
    entropy = 0.0
    for p in dist.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def best_action(
    mg: MorphismGraph,
    context_id: int,
    candidate_etypes: list[int],
) -> Optional[int]:
    """Return the edge type with the highest expected information gain.

    Returns None if candidate_etypes is empty.
    """
    if not candidate_etypes:
        return None
    return max(candidate_etypes, key=lambda et: expected_info_gain(mg, context_id, et))


# ── Active-inference wrapper ───────────────────────────────────────────────────

class ActiveInferenceTracker:
    """Attaches an on_observe hook to a MorphismGraph.

    Registers itself via mg.on_observe() so prediction_error is computed
    BEFORE every model update — even when callers invoke mg.observe()
    directly, bypassing this wrapper.  This makes active-inference tracking
    structural rather than opt-in.

    Tracks:
      - running mean prediction error (bits/token)
      - total observations processed
      - list of recent PE values (bounded to _MAX_HIST entries)

    The observe() / observe_sequence() / flush() convenience methods forward
    to the underlying graph with the same signatures, so existing code that
    wraps mg through the tracker continues to work unchanged.
    """

    _MAX_HIST = 10_000   # cap on history length to avoid unbounded memory

    def __init__(self, mg: MorphismGraph) -> None:
        self.mg = mg

        # Running statistics
        self._pe_sum:   float = 0.0
        self._pe_count: int   = 0
        self._pe_last:  float = 0.0
        self._pe_hist:  list[float] = []   # recent surprisal values

        # Register the hook — fires BEFORE every mg.observe() model update
        mg.on_observe(self._hook)

    def _hook(self, ctx_id: Optional[int], etype: Optional[int], tgt_id: int, graph: MorphismGraph) -> None:
        """Pre-observation hook: compute PE from the prior state of graph.

        ctx_id is None on the first token (no incoming context edge), in which
        case we record PE = 0.0 (no prediction was possible).
        etype is None for the same reason; both None checks are equivalent.
        """
        if ctx_id is None or etype is None:
            pe = 0.0
        else:
            pe = prediction_error(graph, ctx_id, etype, tgt_id)

        self._pe_last   = pe
        self._pe_sum   += pe
        self._pe_count += 1
        if len(self._pe_hist) < self._MAX_HIST:
            self._pe_hist.append(pe)

    # ── Convenience API (mirrors MorphismGraph) ───────────────────────────────

    def observe(self, value: object, etype: int) -> float:
        """Forward to mg.observe(); hook fires automatically.

        Returns the surprisal (bits) that the hook recorded for this token.
        """
        self.mg.observe(value, etype)
        return self._pe_last

    def observe_sequence(self, data: object, topology) -> None:
        """Forward to mg.observe_sequence(); hook fires for each token."""
        self.mg.observe_sequence(data, topology)

    def flush(self) -> None:
        """Flush the underlying MorphismGraph buffer."""
        self.mg.flush()

    def process_corpus(self, sequences: list, topology) -> None:
        """Process a list of sequences with flush between each."""
        for seq in sequences:
            self.observe_sequence(seq, topology)
            self.flush()

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def pe_mean(self) -> float:
        """Running mean prediction error in bits/token."""
        return self._pe_sum / max(self._pe_count, 1)

    @property
    def pe_last(self) -> float:
        """Surprisal of the most recent observation."""
        return self._pe_last

    @property
    def n_observations(self) -> int:
        """Total tokens processed."""
        return self._pe_count

    def free_energy(self) -> float:
        """MDL cost of the current model (delegates to module-level function)."""
        return free_energy(self.mg)

    def log_line(self) -> str:
        """One-line summary suitable for printing during training."""
        return (
            f"ActiveInference("
            f"n={self._pe_count}, "
            f"pe_mean={self.pe_mean:.3f} bits, "
            f"pe_last={self._pe_last:.3f} bits, "
            f"F={self.free_energy():.1f} bits, "
            f"compositions={self.mg.n_compositions()}, "
            f"pairs={self.mg.n_pairs()})"
        )

    def __repr__(self) -> str:
        return self.log_line()
