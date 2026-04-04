"""Token I/O — input tokens sequentially, output tokens autoregressively.

Input:  tokens arrive one at a time, activate their column.
Output: the output cortex produces one token at a time.
        The winner of the competition IS the output.
        Fed back as input for autoregressive generation.

This is the LLM-like interface to CipherNet.
"""
from __future__ import annotations

from typing import Any

try:
    from .graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
    from .prior_loader import load_priors
except ImportError:
    from graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
    from prior_loader import load_priors


class TokenIO:
    """Token-level I/O interface to the CipherNet brain.

    Input: feed_token(char) — activates the corresponding column.
    Output: read_output() — reads the winning output token.
    Generate: produce_tokens(n) — autoregressively generate n tokens.
    """

    def __init__(self, graph: Graph, priors: dict):
        self.graph = graph
        self.priors = priors
        self.output_cortex = priors.get("output_cortex", {})

        # Build token → output node mapping.
        self._output_token_map: dict[str, int] = {}
        self._output_node_tokens: dict[int, str] = {}
        for node_key, node_id in self.output_cortex.items():
            if node_key.startswith("out:"):
                node = graph.get_node(node_id)
                if node and node.meta.get("token"):
                    token = node.meta["token"]
                    self._output_token_map[token] = node_id
                    self._output_node_tokens[node_id] = token

        # Input columns: created dynamically as characters are observed.
        self._input_columns: dict[str, dict[str, int]] = {}

    def get_or_create_input_column(self, char: str) -> dict[str, int]:
        """Get or create a column for an input character."""
        if char in self._input_columns:
            return self._input_columns[char]

        col = self.graph.create_column(f"char:{char}")
        col["token"] = char
        self._input_columns[char] = col
        return col

    def feed_token(self, char: str, n_steps: int = 2):
        """Feed one input token. Activates its column and runs graph steps."""
        col = self.get_or_create_input_column(char)
        self.graph.activate(col["L4"], 1.0)

        for _ in range(n_steps):
            self.graph.step()

    def read_output(self) -> tuple[str | None, float]:
        """Read the current output — the most active output token.

        Returns (token, activation) or (None, 0.0) if no output is active.
        """
        best_token = None
        best_act = 0.0

        for node_id, token in self._output_node_tokens.items():
            node = self.graph.get_node(node_id)
            if node and node.activation > best_act:
                best_act = node.activation
                best_token = token

        return best_token, best_act

    def clear_output(self):
        """Clear all output node activations."""
        for node_id in self._output_node_tokens:
            node = self.graph.get_node(node_id)
            if node:
                node.activation = 0.0

    def drive_output(self, token: str, strength: float = 1.0):
        """Directly drive an output token's activation.

        Used during training to teach the system what output to produce.
        The Hebbian learning rule then strengthens the edges that led
        to this output being active.
        """
        node_id = self._output_token_map.get(token)
        if node_id is not None:
            self.graph.activate(node_id, strength)

    def produce_token(self, n_deliberation_steps: int = 5) -> str | None:
        """Let the graph deliberate and produce one output token.

        Runs several graph steps to let activations settle, then
        reads the winning output token.
        """
        for _ in range(n_deliberation_steps):
            self.graph.step()

        token, activation = self.read_output()
        return token if activation > 0.01 else None

    def generate(self, max_tokens: int = 20,
                 n_deliberation_steps: int = 5) -> list[str]:
        """Autoregressively generate tokens.

        Produce one token, feed it back as input, repeat.
        Stops at <EOS> or max_tokens.
        """
        output_tokens = []

        for _ in range(max_tokens):
            token = self.produce_token(n_deliberation_steps)
            if token is None or token == "<EOS>":
                break
            output_tokens.append(token)

            # Feed the produced token back as input (autoregressive).
            self.clear_output()
            self.feed_token(token)

        return output_tokens

    def connect_input_to_output(self, input_char: str, output_char: str,
                                 weight: float = 0.5):
        """Create a learned edge from an input column to an output token.

        This is how the system learns "when I see '=' after '3+4',
        output '7'." The edge goes from the input column's L5 to
        the output token's node.
        """
        input_col = self.get_or_create_input_column(input_char)
        output_node = self._output_token_map.get(output_char)
        if output_node is not None:
            self.graph.add_edge(input_col["L5"], output_node,
                                edge_type=TEMPORAL, weight=weight)

    def connect_column_to_output(self, column: dict[str, int],
                                  output_char: str, weight: float = 0.5):
        """Connect any column's L5 output to an output token node."""
        output_node = self._output_token_map.get(output_char)
        if output_node is not None:
            self.graph.add_edge(column["L5"], output_node,
                                edge_type=TEMPORAL, weight=weight)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Token I/O System")
    print("=" * 60)

    graph, priors = load_priors()
    tio = TokenIO(graph, priors)
    print(f"Graph after priors: {graph.summary()}")
    print(f"Output tokens available: {len(tio._output_token_map)}")

    # Create input columns for digits and operators.
    for char in "0123456789+-*=":
        tio.get_or_create_input_column(char)
    print(f"After input columns: {graph.summary()}")

    # Test: manually wire 3+4 → output 7.
    # In the real system, this wiring is LEARNED. Here we set it up
    # to verify the output mechanism works.
    print("\n--- Test: manual wiring 3+4 = 7 ---")

    # Feed tokens.
    tio.feed_token('3')
    tio.feed_token('+')
    tio.feed_token('4')
    tio.feed_token('=')

    # At this point, columns for 3, +, 4, = are all active.
    # We need to drive the output. In the trained system, the graph
    # dynamics would drive the correct output. For now, test the
    # output reading mechanism.
    tio.drive_output('7', 0.9)
    tio.drive_output('3', 0.2)  # competitor
    tio.drive_output('4', 0.3)  # competitor

    # Run a few steps to let inhibition settle.
    for _ in range(3):
        graph.step()

    token, act = tio.read_output()
    print(f"  Output: '{token}' (activation: {act:.3f})")
    print(f"  Expected: '7'")
    print(f"  Correct: {token == '7'}")

    # Test: autoregressive generation with manual driving.
    print("\n--- Test: token read/clear cycle ---")
    tio.clear_output()
    t1, a1 = tio.read_output()
    print(f"  After clear: token='{t1}', act={a1:.3f} (should be None/0)")

    tio.drive_output('1', 0.8)
    t2, a2 = tio.read_output()
    print(f"  After drive '1': token='{t2}', act={a2:.3f}")

    print(f"\nFinal graph: {graph.summary()}")
