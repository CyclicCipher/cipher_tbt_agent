"""CipherNet Brain — predictive coding on a graph.

No Python calculators. No Python parsers. No Python state machines.
The graph IS the computation. If it can't be done by graph.step()
propagating activations and computing prediction errors through
edges, it doesn't exist yet.

What remains:
- Load priors (graph structure from JSON)
- Feed tokens (activate a column)
- Run steps (graph.step() — the ONE update rule)
- Settle (graph.settle() — prospective configuration)
- Read output (most active output cortex node)
- Learn (graph.learn() — error-driven weight updates)
"""
from __future__ import annotations

try:
    from .graph import Graph, TEMPORAL
    from .prior_loader import load_priors
    from .token_io import TokenIO
except ImportError:
    from graph import Graph, TEMPORAL
    from prior_loader import load_priors
    from token_io import TokenIO


class Brain:
    """The CipherNet brain. Tokens in, graph dynamics, tokens out."""

    def __init__(self, default_decay: float = 0.5):
        self.graph, self.priors = load_priors()
        self.tio = TokenIO(self.graph, self.priors)
        self.default_decay = default_decay

    def feed(self, token: str, n_steps: int = 2):
        """Feed one token. Activates its column, runs graph steps."""
        col = self.tio.get_or_create_input_column(token)
        self.graph.activate(col["L4"], 1.0)
        for _ in range(n_steps):
            self.graph.step(default_decay=self.default_decay)

    def step(self, n: int = 1):
        """Run n graph update steps."""
        for _ in range(n):
            self.graph.step(default_decay=self.default_decay)

    def settle(self, n_steps: int = 20, clamp: dict[int, float] | None = None,
               learn_rate: float = 0.0):
        """Prospective configuration: settle activations with clamped I/O.

        If learn_rate > 0, weights adjust at every step (simultaneous
        inference and learning). Error propagates deeper each step.
        """
        self.graph.settle(n_steps=n_steps, clamp=clamp,
                          default_decay=self.default_decay,
                          learn_rate=learn_rate)

    def reward(self, value: float, learning_rate: float = 0.01):
        """Deliver reward via dopamine (three-factor learning).

        Biology (Yagishita et al. 2014): the silent eligibility trace.
        1. Cortex→D1 synapses that were recently active have an
           eligibility trace (set during gating decisions).
        2. Dopamine arrives ~2s later (reward signal).
        3. ONLY synapses with active traces get potentiated.
           Synapses without traces are unaffected.

        Positive RPE: dopamine burst → D1 edges with traces get LTP.
        Negative RPE: dopamine dip → D1 edges get LTD (or D2 LTP).

        The eligibility trace window (~20 gamma cycles) bridges the
        delay between the gating action and the reward signal.
        """
        da_key = self.priors.get('basal_ganglia', {}).get('dopamine')
        if da_key is None:
            return

        if value > 0:
            # Positive RPE: dopamine burst.
            self.graph.activate(da_key, min(1.0, value))
        else:
            # Negative RPE: dopamine dip.
            self.graph.activate(da_key, 0.0)

        # Propagate dopamine to D1/D2 (one step for immediate effect).
        self.step(1)

        # Three-factor learning: adjust edges that have BOTH
        # active source AND active eligibility trace.
        # The eligibility traces were set during earlier gating
        # decisions. Dopamine converts them to weight changes.
        from graph import TEMPORAL, quantize_weight, ELIGIBILITY_THRESHOLD
        bg_nodes = self.priors.get('basal_ganglia', {})
        for edge_key, edge in list(self.graph._edges.items()):
            if edge.edge_type != TEMPORAL:
                continue
            # Only modify edges INTO D1/D2 MSNs (the gating decision edges).
            tgt_node = self.graph.get_node(edge.target)
            if tgt_node is None:
                continue
            tgt_role = tgt_node.meta.get('role')
            if tgt_role not in ('d1_msn', 'd2_msn'):
                continue
            # Three factors: eligibility × dopamine_value × source_activation
            if abs(edge.eligibility) < ELIGIBILITY_THRESHOLD:
                continue  # no trace → no change (temporal specificity)
            src_node = self.graph.get_node(edge.source)
            if src_node is None or src_node.activation < 0.01:
                continue
            # D1: positive dopamine → strengthen (LTP)
            # D2: positive dopamine → weaken (LTD)
            if tgt_role == 'd1_msn':
                delta = learning_rate * value * edge.eligibility * src_node.activation
            else:  # d2_msn
                delta = -learning_rate * value * edge.eligibility * src_node.activation
            edge.weight += delta
            edge.weight = quantize_weight(edge.weight)
            edge.eligibility = 0.0  # trace consumed

    def attend(self, level: float = 1.0):
        """Set acetylcholine attentional gain.

        Biology: ACh from nucleus basalis of Meynert.
        - level > 0: enhance thalamocortical, suppress lateral
        - level = 0: no attentional modulation (default state)
        - Sharpens representation: boost direct input, suppress noise
        """
        self.graph._ach_level = max(0.0, min(1.0, level))

    def read_output(self) -> tuple[str | None, float]:
        """Read the winning output token."""
        return self.tio.read_output()

    def clear_output(self):
        """Clear output cortex activations."""
        self.tio.clear_output()

    def status(self) -> dict:
        """Current graph state."""
        return self.graph.summary()


if __name__ == "__main__":
    brain = Brain()
    print(f"Brain: {brain.status()}")
    print(f"Priors: {list(brain.priors.keys())}")
