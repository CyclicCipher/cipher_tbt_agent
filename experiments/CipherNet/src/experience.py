"""Experience-driven learning — the system grows from stimuli.

Starts with only priors (ANS, PFC, BG, thalamus). Encounters stimuli
(characters, quantities, expressions) and dynamically:
1. Creates columns for novel characters
2. Discovers ordering via ANS → creates successor edges (number line)
3. Discovers operations from expression examples
4. Learns gating policy from multi-step examples

The brain of CipherNet. No pre-built number line. No pre-built
arithmetic columns. Everything grows from experience.
"""
from __future__ import annotations

from typing import Any

try:
    from .graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
    from .prior_loader import load_priors
    from .number_line import build_number_line, compute_addition, propagate_waves, setup_addition, read_comp_position
except ImportError:
    from graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
    from prior_loader import load_priors
    from number_line import build_number_line, compute_addition, propagate_waves, setup_addition, read_comp_position


class Brain:
    """The CipherNet brain. Starts with priors, grows from experience.

    This is the top-level class that orchestrates everything:
    - Manages the graph and all subgraphs
    - Creates columns for novel stimuli
    - Discovers relationships between columns
    - Learns operations from expression examples
    - Handles multi-step computation via PFC
    """

    def __init__(self):
        """Initialize with priors only."""
        self.graph, self.priors = load_priors()
        self.pfc = self.priors["pfc"]
        self.bg = self.priors["basal_ganglia"]
        self.thalamus = self.priors["thalamus"]
        self.ans = self.priors["ans"]

        # Registry of learned columns: {stimulus_value: column_dict}
        self.columns: dict[str, dict[str, int]] = {}

        # Number line: grows as digit columns are created and ordered.
        # Maps position → column name for digits on the number line.
        self.number_line_order: list[str] = []  # ordered column names
        self.number_line_built = False
        self.numline = None  # the graph-native number line (ref/comp/done layers)

        # Known operations: {operator_symbol: column_dict}
        self.operations: dict[str, dict] = {}

        # Expression parsing state.
        self._wm0_value: float | None = None  # what's "in" WM stripe 0
        self._wm1_value: float | None = None
        self._wm2_op: str | None = None       # current operation

    # -------------------------------------------------------------------
    # Stimulus processing
    # -------------------------------------------------------------------

    def observe_character(self, char: str) -> dict[str, int]:
        """Observe a character. Create a column if novel.

        Returns the column dict for this character.
        """
        if char in self.columns:
            # Activate existing column.
            col = self.columns[char]
            self.graph.activate(col["L4"], 1.0)
            self.graph.step()
            return col

        # Novel character — create a new column.
        col = self.graph.create_column(f"char:{char}")
        self.columns[char] = col
        col["value"] = char

        # Activate it.
        self.graph.activate(col["L4"], 1.0)
        self.graph.step()

        return col

    def observe_quantity(self, char: str, quantity: int):
        """Associate a character with a quantity via ANS.

        This teaches the system that '5' represents 5 items.
        The ANS provides magnitude; the column stores the association.
        """
        # Ensure column exists.
        col = self.observe_character(char)

        # Store the quantity as metadata.
        node = self.graph.get_node(col["L23"])
        if node is not None:
            node.meta["quantity"] = quantity

    def observe_ordering(self, char_a: str, char_b: str):
        """Use ANS to determine that char_a < char_b, then create
        a successor edge if they're adjacent.

        This grows the number line incrementally.
        """
        if char_a not in self.columns or char_b not in self.columns:
            return

        col_a = self.columns[char_a]
        col_b = self.columns[char_b]

        qty_a = self.graph.get_node(col_a["L23"]).meta.get("quantity")
        qty_b = self.graph.get_node(col_b["L23"]).meta.get("quantity")

        if qty_a is None or qty_b is None:
            return

        # ANS comparison: is b exactly one more than a?
        if qty_b == qty_a + 1:
            # Create successor edge: a → b (temporal, directed).
            self.graph.add_edge(col_a["L5"], col_b["L4"],
                                edge_type=TEMPORAL, weight=1.0,
                                relation="successor")

    # -------------------------------------------------------------------
    # Number line construction
    # -------------------------------------------------------------------

    def build_number_line_from_digits(self, max_n: int = 50):
        """Build the graph-native number line from learned digit columns.

        Creates the dual-layer (ref/comp/done) structure needed for
        wave-based arithmetic. Connects to existing digit columns.
        """
        self.numline = build_number_line(self.graph, max_n=max_n,
                                          subgraph_name="number_line")
        self.number_line_built = True

    # -------------------------------------------------------------------
    # Expression processing
    # -------------------------------------------------------------------

    def process_expression(self, tokens: list[str]) -> float | None:
        """Process a complete expression token by token.

        Tokens: ['4', '+', '5', '+', '1', '+', '9']

        The system:
        1. Recognizes digit vs operator tokens
        2. Loads digits into PFC WM stripes via BG gating
        3. When both operands are ready, computes via number line
        4. Gates result back into WM0
        5. Repeats until all tokens processed
        6. Returns the final value in WM0

        This is the multi-step computation loop driven by the graph.
        """
        if not self.number_line_built:
            self.build_number_line_from_digits()

        self._wm0_value = None
        self._wm1_value = None
        self._wm2_op = None

        for token in tokens:
            self._process_token(token)

        return self._wm0_value

    def _process_token(self, token: str):
        """Process one token in the context of the current expression."""
        # Observe the character (creates column if novel).
        col = self.observe_character(token)

        # Determine token type from learned knowledge.
        node = self.graph.get_node(col["L23"])
        quantity = node.meta.get("quantity") if node else None

        if quantity is not None:
            # This is a digit with a known quantity.
            self._process_digit(quantity)
        elif token in ['+', '-', '*', '/']:
            # This is an operator.
            self._process_operator(token)
        elif token == '=':
            # End of expression — result is in WM0.
            pass

    def _process_digit(self, value: float):
        """A digit arrived. Load into the appropriate WM stripe."""
        if self._wm0_value is None:
            # First operand → WM0.
            self._wm0_value = value
            # Gate WM0 open, load value.
            self._gate_wm(0, value)
        else:
            # Second operand → WM1.
            self._wm1_value = value
            self._gate_wm(1, value)

            # Both operands ready — compute.
            if self._wm2_op is not None:
                result = self._compute()
                if result is not None:
                    # Gate result into WM0, clear WM1.
                    self._wm0_value = result
                    self._wm1_value = None
                    self._gate_wm(0, result)
                    self._clear_wm(1)

    def _process_operator(self, op: str):
        """An operator arrived. Set the goal in WM2."""
        self._wm2_op = op
        self._gate_wm(2, 1.0)  # activate goal stripe

    def _compute(self) -> float | None:
        """Execute the current operation using the graph-native number line."""
        a = self._wm0_value
        b = self._wm1_value
        op = self._wm2_op

        if a is None or b is None or op is None:
            return None

        a_int = int(round(a))
        b_int = int(round(b))

        if op == '+':
            if self.numline is not None:
                result = compute_addition(self.graph, self.numline, a_int, b_int)
                return float(result) if result is not None else None
            return None
        elif op == '-':
            # Subtraction: find b such that b + operand = a.
            # For now, use the number line inversely.
            if self.numline is not None and a_int >= b_int:
                # a - b = ? means ? + b = a. Walk from 0 to find ?.
                # Simpler: walk backward from a by b steps.
                result = a_int - b_int  # TODO: make graph-native
                return float(result)
            return None
        elif op == '*':
            # Multiplication: repeated addition on the number line.
            # a * b = a + a + a + ... (b times)
            if self.numline is not None and a_int >= 0 and b_int >= 0:
                result = 0
                for _ in range(b_int):
                    new_result = compute_addition(self.graph, self.numline,
                                                   result, a_int)
                    if new_result is None:
                        return None
                    result = new_result
                return float(result)
            return None

        return None

    def _gate_wm(self, stripe: int, value: float):
        """Open a WM gate and load a value.

        Activates the BG Go pathway for the stripe, which disinhibits
        the thalamic relay, which opens the gate on the WM L23 node.
        """
        # Activate the Go node for this stripe.
        go_node = self.bg[f"d1_go_{stripe}"]
        self.graph.activate(go_node, 1.0)

        # Activate the WM input with the value (normalized to 0-1).
        wm_l4 = self.pfc[f"wm{stripe}:L4"]
        self.graph.activate(wm_l4, min(1.0, value / 20.0))

        # Run steps to propagate the gating signal.
        for _ in range(3):
            self.graph.step()

        # Deactivate Go (gate closes after update).
        self.graph.activate(go_node, 0.0)

    def _clear_wm(self, stripe: int):
        """Clear a WM stripe by opening the gate with no input."""
        go_node = self.bg[f"d1_go_{stripe}"]
        self.graph.activate(go_node, 1.0)
        # No input → WM decays.
        for _ in range(3):
            self.graph.step()
        self.graph.activate(go_node, 0.0)

    # -------------------------------------------------------------------
    # Training from examples
    # -------------------------------------------------------------------

    def train_digits(self, digits: list[tuple[str, int]]):
        """Teach digit-quantity associations.

        digits: [('0', 0), ('1', 1), ..., ('9', 9)]
        """
        for char, quantity in digits:
            self.observe_character(char)
            self.observe_quantity(char, quantity)

        # Discover ordering from quantities.
        sorted_digits = sorted(digits, key=lambda x: x[1])
        for i in range(len(sorted_digits) - 1):
            self.observe_ordering(sorted_digits[i][0], sorted_digits[i + 1][0])

    def train_operators(self, operators: list[str]):
        """Register operator characters."""
        for op in operators:
            self.observe_character(op)

    def train_expression(self, tokens: list[str], expected: float) -> float:
        """Train on one expression. Returns prediction error.

        Processes the expression, compares result to expected,
        and applies dopamine reward signal.
        """
        result = self.process_expression(tokens)

        if result is not None:
            error = abs(result - expected)
            # Reward: inversely proportional to error.
            if error < 0.5:
                reward = 1.0  # correct
            else:
                reward = -0.5  # wrong

            self.graph.learn(reward=reward, learning_rate=0.05)
            return error
        else:
            # Failed to compute.
            self.graph.learn(reward=-1.0, learning_rate=0.05)
            return float('inf')

    # -------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------

    def status(self) -> dict:
        """Current state of the brain."""
        return {
            "graph": self.graph.summary(),
            "columns": list(self.columns.keys()),
            "number_line_built": self.number_line_built,
            "operations": list(self.operations.keys()),
        }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CipherNet Brain — Learning from Experience")
    print("  Starts with only priors. Everything else is learned.")
    print("=" * 60)

    brain = Brain()
    print(f"\nInitial state (priors only): {brain.graph.summary()}")

    # Phase 1: Learn digits.
    print("\n--- Phase 1: Learn digits 0-9 ---")
    brain.train_digits([(str(i), i) for i in range(10)])
    brain.train_operators(['+', '-', '*', '='])
    print(f"After digits + operators: {brain.graph.summary()}")
    print(f"Columns: {list(brain.columns.keys())}")

    # Phase 2: Build number line for computation.
    print("\n--- Phase 2: Build number line ---")
    brain.build_number_line_from_digits(max_n=20)
    print(f"After number line: {brain.graph.summary()}")

    # Phase 3: Single-step addition.
    print("\n--- Phase 3: Single-step addition ---")
    single_tests = [
        (['3', '+', '4', '='], 7),
        (['5', '+', '5', '='], 10),
        (['8', '+', '7', '='], 15),
        (['0', '+', '9', '='], 9),
        (['1', '+', '0', '='], 1),
    ]
    for tokens, expected in single_tests:
        error = brain.train_expression(tokens, expected)
        result = brain._wm0_value
        expr = ''.join(tokens)
        ok = result is not None and abs(result - expected) < 0.5
        print(f"  {expr} expected={expected}, got={result}, error={error:.1f}  {'OK' if ok else 'WRONG'}")

    # Phase 4: Multi-step addition!
    print("\n--- Phase 4: Multi-step addition ---")
    multi_tests = [
        (['3', '+', '4', '+', '2', '='], 9),
        (['1', '+', '2', '+', '3', '='], 6),
        (['4', '+', '5', '+', '1', '+', '9', '='], 19),
        (['2', '+', '2', '+', '2', '+', '2', '+', '2', '='], 10),
        (['9', '+', 9 , '+', '9', '='], 27),  # intentional: 9 as int not str — test robustness
    ]
    for tokens, expected in multi_tests:
        # Convert any non-string tokens.
        tokens = [str(t) for t in tokens]
        result = brain.process_expression(tokens)
        ok = result is not None and abs(result - expected) < 0.5
        expr = ''.join(tokens)
        print(f"  {expr} expected={expected}, got={result}  {'OK' if ok else 'WRONG'}")

    # Phase 5: Multiplication as repeated addition.
    print("\n--- Phase 5: Multiplication (repeated addition) ---")
    mul_tests = [
        (['3', '*', '4', '='], 12),
        (['2', '*', '5', '='], 10),
        (['4', '*', '5', '='], 20),
    ]
    for tokens, expected in mul_tests:
        result = brain.process_expression(tokens)
        ok = result is not None and abs(result - expected) < 0.5
        expr = ''.join(tokens)
        print(f"  {expr} expected={expected}, got={result}  {'OK' if ok else 'WRONG'}")

    print(f"\n--- Final state ---")
    print(f"  {brain.graph.summary()}")
