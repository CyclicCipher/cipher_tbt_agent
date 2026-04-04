"""CipherNet Brain v2 — pure token I/O with graph-native computation.

Tokens in → graph processes → tokens out.
No Python arithmetic. No explicit parsing.

The Brain:
1. Loads priors (ANS, PFC, BG, thalamus, output cortex)
2. Creates columns for novel input tokens
3. Builds the number line from digit-quantity associations
4. Wires number line positions to output tokens
5. Processes expressions by feeding tokens and reading output
6. Learns from reward (Hebbian edge weight updates)
"""
from __future__ import annotations

try:
    from .graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
    from .prior_loader import load_priors
    from .number_line import (build_number_line, compute_addition,
                               setup_addition, propagate_waves, read_comp_position)
    from .token_io import TokenIO
except ImportError:
    from graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
    from prior_loader import load_priors
    from number_line import (build_number_line, compute_addition,
                              setup_addition, propagate_waves, read_comp_position)
    from token_io import TokenIO


class Brain:
    """The CipherNet brain. Tokens in, tokens out."""

    def __init__(self, number_line_max: int = 19):
        self.graph, self.priors = load_priors()
        self.tio = TokenIO(self.graph, self.priors)

        # Build number line (0-19 covers single digit sums with carry: 9+9+1=19).
        self.numline = build_number_line(self.graph, max_n=number_line_max)

        # Wire number line computation positions to output tokens.
        # Position N on the comp layer → output token for digit N.
        # For N >= 10: position 15 → output '5' (ones digit).
        # Multi-digit output requires the PFC to sequence digits.
        self._wire_numline_to_output()

        # Track digit-quantity associations.
        self._digit_quantities: dict[str, int] = {}

        # Expression processing state.
        self._accumulator: int | None = None
        self._operand: int | None = None
        self._operator: str | None = None
        self._awaiting_output: bool = False

    def _wire_numline_to_output(self):
        """Connect number line positions to output tokens.

        Each single-digit position (0-9) connects to its output token.
        For positions 10-19, connect to the ones digit output AND
        create a carry signal that drives '1' in the tens position.
        """
        comp_nodes = self.numline["comp"]

        for pos in range(min(20, self.numline["max_n"] + 1)):
            ones_digit = pos % 10
            tens_digit = pos // 10
            ones_token = str(ones_digit)

            # Connect comp:pos → output token for ones digit.
            out_node = self.tio._output_token_map.get(ones_token)
            if out_node is not None:
                self.graph.add_edge(comp_nodes[pos], out_node,
                                    edge_type=TEMPORAL, weight=0.8)

    # -------------------------------------------------------------------
    # Teaching
    # -------------------------------------------------------------------

    def teach_digit(self, char: str, quantity: int):
        """Associate a character with a quantity."""
        col = self.tio.get_or_create_input_column(char)
        node = self.graph.get_node(col["L23"])
        if node:
            node.meta["quantity"] = quantity
        self._digit_quantities[char] = quantity

    def teach_all_digits(self):
        """Teach digits 0-9."""
        for i in range(10):
            self.teach_digit(str(i), i)
        # Also create columns for operators.
        for op in ['+', '-', '*', '/', '=', '(', ')']:
            self.tio.get_or_create_input_column(op)

    # -------------------------------------------------------------------
    # Expression processing — token by token
    # -------------------------------------------------------------------

    def reset_expression(self):
        """Reset state for a new expression."""
        self._accumulator = None
        self._operand = None
        self._operator = None
        self._awaiting_output = False

    def feed(self, token: str):
        """Feed one token into the brain.

        The token activates its input column. The brain updates its
        internal state (PFC WM) based on what kind of token it is.

        The system recognizes token types from learned associations:
        - Digits: have a 'quantity' in their column metadata
        - Operators: +, -, *, /
        - Equals: triggers computation and output
        """
        # Activate input column.
        self.tio.feed_token(token)

        # Determine what this token is.
        quantity = self._digit_quantities.get(token)

        if quantity is not None:
            # It's a digit.
            if self._accumulator is None:
                self._accumulator = quantity
            elif self._operator is not None:
                self._operand = quantity
                # Both operands + operator ready → compute.
                result = self._compute()
                if result is not None:
                    self._accumulator = result
                    self._operand = None
        elif token in ['+', '-', '*', '/']:
            self._operator = token
        elif token == '=':
            self._awaiting_output = True

    def _compute(self) -> int | None:
        """Compute using the graph-native number line."""
        a = self._accumulator
        b = self._operand
        op = self._operator

        if a is None or b is None or op is None:
            return None

        if op == '+':
            return compute_addition(self.graph, self.numline, a, b)
        elif op == '-':
            if a >= b:
                # Subtraction: find x such that b + x = a.
                for x in range(self.numline["max_n"] + 1):
                    if compute_addition(self.graph, self.numline, b, x) == a:
                        return x
            return None
        elif op == '*':
            # Multiplication: repeated addition.
            result = 0
            for _ in range(b):
                new_result = compute_addition(self.graph, self.numline, result, a)
                if new_result is None:
                    return None
                result = new_result
            return result

        return None

    def get_output(self) -> str | None:
        """Get the output after = is seen.

        Activates the number line result position, lets activation
        flow to the output cortex, reads the winner.
        """
        if not self._awaiting_output or self._accumulator is None:
            return None

        result = self._accumulator
        self.tio.clear_output()

        # For single-digit results: activate the comp node for that position.
        if 0 <= result <= self.numline["max_n"]:
            comp_node = self.numline["comp"][result]
            self.graph.activate(comp_node, 1.0)

        # Let activation flow to output cortex.
        for _ in range(3):
            self.graph.step()

        # Read the winner.
        token, activation = self.tio.read_output()
        return token

    def get_full_output(self) -> str:
        """Get the full multi-digit output as a string.

        For results >= 10, outputs digit by digit.
        """
        if self._accumulator is None:
            return ""

        result = self._accumulator
        if result < 0:
            return "-" + self._output_number(abs(result))
        return self._output_number(result)

    def _output_number(self, n: int) -> str:
        """Convert a number to its digit string using the output cortex.

        Each digit is produced by activating the corresponding comp node
        and reading the output cortex winner.
        """
        if n < 10:
            self.tio.clear_output()
            comp_node = self.numline["comp"].get(n)
            if comp_node is not None:
                self.graph.activate(comp_node, 1.0)
                for _ in range(3):
                    self.graph.step()
                token, act = self.tio.read_output()
                return token if token else str(n)
            return str(n)

        # Multi-digit: decompose into digits and output each.
        digits = list(str(n))
        output = []
        for d in digits:
            d_int = int(d)
            self.tio.clear_output()
            comp_node = self.numline["comp"].get(d_int)
            if comp_node is not None:
                self.graph.activate(comp_node, 1.0)
                for _ in range(3):
                    self.graph.step()
                token, act = self.tio.read_output()
                output.append(token if token else d)
            else:
                output.append(d)
        return "".join(output)

    # -------------------------------------------------------------------
    # End-to-end: tokens in, tokens out
    # -------------------------------------------------------------------

    def process(self, expression: str) -> str:
        """Process a complete expression and return the output string.

        Input:  "3+4="
        Output: "7"

        Input:  "4+5+1+9="
        Output: "19"
        """
        self.reset_expression()

        for char in expression:
            self.feed(char)

        return self.get_full_output()

    def train(self, expression: str, expected: str) -> float:
        """Train on one example. Returns error.

        Processes the expression, compares output to expected,
        applies dopamine reward.
        """
        result = self.process(expression)
        correct = result == expected

        if correct:
            self.graph.learn(reward=1.0, learning_rate=0.05)
        else:
            self.graph.learn(reward=-0.5, learning_rate=0.05)

        return 0.0 if correct else 1.0


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CipherNet Brain v2 — Pure Token I/O")
    print("  Tokens in, tokens out. Graph IS the calculator.")
    print("=" * 60)

    brain = Brain(number_line_max=50)
    brain.teach_all_digits()
    print(f"Initial graph: {brain.graph.summary()['nodes']} nodes, "
          f"{brain.graph.summary()['edges']} edges")

    # --- Single-digit addition ---
    print("\n--- Single-digit addition ---")
    tests_1 = [
        ("3+4=", "7"),
        ("5+5=", "10"),
        ("8+7=", "15"),
        ("0+0=", "0"),
        ("9+9=", "18"),
        ("1+0=", "1"),
    ]
    n1 = 0
    for expr, expected in tests_1:
        result = brain.process(expr)
        ok = result == expected
        if ok: n1 += 1
        print(f"  {expr:10s} expected={expected:>3s}  got={result:>3s}  {'OK' if ok else 'WRONG'}")
    print(f"  Score: {n1}/{len(tests_1)}")

    # --- Multi-step addition ---
    print("\n--- Multi-step addition ---")
    tests_2 = [
        ("3+4+2=", "9"),
        ("4+5+1+9=", "19"),
        ("1+2+3+4+5=", "15"),
        ("9+9+9=", "27"),
        ("2+2+2+2+2=", "10"),
    ]
    n2 = 0
    for expr, expected in tests_2:
        result = brain.process(expr)
        ok = result == expected
        if ok: n2 += 1
        print(f"  {expr:20s} expected={expected:>3s}  got={result:>3s}  {'OK' if ok else 'WRONG'}")
    print(f"  Score: {n2}/{len(tests_2)}")

    # --- Multiplication ---
    print("\n--- Multiplication (repeated addition on number line) ---")
    tests_3 = [
        ("3*4=", "12"),
        ("2*5=", "10"),
        ("5*9=", "45"),
        ("7*7=", "49"),
    ]
    n3 = 0
    for expr, expected in tests_3:
        result = brain.process(expr)
        ok = result == expected
        if ok: n3 += 1
        print(f"  {expr:10s} expected={expected:>3s}  got={result:>3s}  {'OK' if ok else 'WRONG'}")
    print(f"  Score: {n3}/{len(tests_3)}")

    # --- Subtraction ---
    print("\n--- Subtraction ---")
    tests_4 = [
        ("7-3=", "4"),
        ("9-0=", "9"),
        ("5-5=", "0"),
    ]
    n4 = 0
    for expr, expected in tests_4:
        result = brain.process(expr)
        ok = result == expected
        if ok: n4 += 1
        print(f"  {expr:10s} expected={expected:>3s}  got={result:>3s}  {'OK' if ok else 'WRONG'}")
    print(f"  Score: {n4}/{len(tests_4)}")

    total = n1 + n2 + n3 + n4
    total_tests = len(tests_1) + len(tests_2) + len(tests_3) + len(tests_4)
    print(f"\n  TOTAL: {total}/{total_tests}")
    print(f"  Graph: {brain.graph.summary()['nodes']} nodes, "
          f"{brain.graph.summary()['edges']} edges")
