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

    def propagate_errors(self, n_passes: int = 3,
                          clamp_errors: dict[int, float] | None = None):
        """Backward error sweep for full credit assignment."""
        self.graph.propagate_errors_backward(
            n_passes=n_passes, clamp_errors=clamp_errors)

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
