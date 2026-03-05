"""SymbolicAI engine — the main interface for the symbolic reasoning system.

Ties together interpreter, memory, and synthesis.

The engine distinguishes two kinds of knowledge:

    Built-in knowledge: concepts with pre-defined process expressions
    (loaded from a .ctkg file).  These execute immediately via the
    interpreter.  They represent "instincts" — pre-loaded capability
    that doesn't need training.

    Learned knowledge: concepts without a process expression.  The system
    accumulates (inputs, outputs) examples via teach(), then consolidates
    them into a process rule via consolidate().  Before consolidation,
    queries are answered by exact-match example lookup.  After
    consolidation, queries use the discovered rule.

KL divergence (from memory.py) acts as the consolidation signal:
    high KL → the current rule (or absence of a rule) is surprising
              given the stored examples — consolidation is needed.
    low KL  → the rule explains the examples — consolidated.
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, Optional

from interpreter import ProcessInterpreter
from memory import ExampleStore
from synthesis import Synthesizer


class SymbolicAI:
    """Symbolic AI backed by a CTKG knowledge graph.

    Usage:
        import sys; sys.path.insert(0, '..')
        from ctkg.parser import parse_file
        graph = parse_file('ctkg/domains/arithmetic.ctkg')

        ai = SymbolicAI(graph)

        # Built-in ops work immediately:
        ai.ask('successor', (4,))          # → (5,)
        ai.ask('comparison', (7, 3))       # → ('GT',)

        # Clear the pre-loaded process to simulate "not yet learned":
        ai.clear_process('single_digit_addition')

        # Teach some examples:
        ai.teach('single_digit_addition', (3, 'ADD', 4), (0, 7))
        ai.teach('single_digit_addition', (8, 'ADD', 5), (1, 3))
        ...

        # Consolidate examples → discovers the process rule:
        rule = ai.consolidate('single_digit_addition')
        # rule = ['result = fold(a, b, succ)', ..., 'emit(c, ones)']

        # Now generalizes perfectly:
        ai.ask('single_digit_addition', (9, 'ADD', 1))  # → (1, 0)
    """

    def __init__(self, graph) -> None:
        self.graph = graph
        self.stores: Dict[str, ExampleStore] = {}
        self._synthesizer = Synthesizer()
        self._interp = ProcessInterpreter(
            engine_ask=self.ask,
            concept_names=frozenset(graph.concepts.keys()),
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def teach(
        self,
        concept_name: str,
        inputs: tuple,
        outputs: tuple,
    ) -> None:
        """Record one (inputs, outputs) training example for a concept."""
        if concept_name not in self.stores:
            self.stores[concept_name] = ExampleStore(concept_name)
        self.stores[concept_name].add(tuple(inputs), tuple(outputs))

    def ask(
        self,
        concept_name: str,
        inputs: tuple,
    ) -> Optional[tuple]:
        """Answer a query about a concept given inputs.

        Priority:
            1. If the concept has a process expression, execute it.
            2. If there are stored examples, do exact-match lookup.
            3. Return None (cannot answer).
        """
        concept = self.graph.concepts.get(concept_name)
        if concept is None:
            return None

        # Execute process expression if one exists (built-in or consolidated).
        if concept.process:
            try:
                return self._interp.run(
                    concept.process, tuple(inputs), concept.input_type
                )
            except Exception:
                # Process failed (e.g. wrong input type) — fall through.
                pass

        # Exact-match lookup in example store.
        store = self.stores.get(concept_name)
        if store is not None:
            for stored_inputs, stored_outputs in store.examples:
                if stored_inputs == tuple(inputs):
                    return stored_outputs

        return None

    def consolidate(self, concept_name: str) -> Optional[List[str]]:
        """Discover a process rule from stored examples (consolidation).

        Runs the synthesizer to find the shortest process consistent with
        all stored examples.  On success, the discovered process is stored
        in the concept so that future ask() calls use it.

        Returns the process lines on success, None on failure.

        Failure means: insufficient examples, missing prerequisites, or
        ambiguous examples that don't uniquely determine a rule.
        """
        store = self.stores.get(concept_name)
        if store is None or len(store) == 0:
            return None

        process = self._synthesizer.synthesize(
            concept_name=concept_name,
            store=store,
            graph=self.graph,
            interpreter=self._interp,
            engine_ask=self.ask,
        )

        if process is not None:
            self.graph.concepts[concept_name].process = process

        return process

    # ------------------------------------------------------------------
    # KL divergence / consolidation trigger
    # ------------------------------------------------------------------

    def kl(self, concept_name: str) -> float:
        """KL divergence of stored examples from the current model (in bits).

        KL ≈ 0:   the current rule (or example cache) explains everything.
        KL → inf: no rule, or rule is wrong.
        """
        store = self.stores.get(concept_name)
        if store is None or len(store) == 0:
            return float('inf')
        return store.kl_divergence(lambda inp: self.ask(concept_name, inp))

    def should_consolidate(
        self,
        concept_name: str,
        threshold: float = 0.1,
    ) -> bool:
        """True if KL exceeds threshold — consolidation is recommended."""
        return self.kl(concept_name) > threshold

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_process(self, concept_name: str) -> None:
        """Clear a concept's process expression (makes it 'not yet learned').

        Used in experiments to simulate learning from scratch even when
        the .ctkg file contains a pre-defined process.
        """
        concept = self.graph.concepts.get(concept_name)
        if concept is not None:
            concept.process = []

    def reset_concept(self, concept_name: str) -> None:
        """Clear both examples and process for a concept.

        Used in minimum-examples sweep experiments where the concept
        is repeatedly tested with different training set sizes.
        """
        self.stores.pop(concept_name, None)
        concept = self.graph.concepts.get(concept_name)
        if concept is not None:
            concept.process = []

    def example_count(self, concept_name: str) -> int:
        store = self.stores.get(concept_name)
        return len(store) if store else 0

    def has_process(self, concept_name: str) -> bool:
        concept = self.graph.concepts.get(concept_name)
        return bool(concept and concept.process)
