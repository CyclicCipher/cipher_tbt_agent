"""Template-based program synthesis (consolidation).

Given stored examples of a concept and the CTKG prerequisite graph,
finds the shortest process expression consistent with all examples.

The hypothesis space is constrained by the prerequisite graph:
only operations available from ancestor concepts are considered.
This is the structural guarantee: you cannot learn addition before
knowing successor, because the fold+succ template is not generated
unless 'successor' is in the concept's transitive ancestors.

Algorithm:
    1. Compute available_ops = {primitives from ancestor concepts}
    2. Generate all templates that only use available_ops
    3. Sort by complexity (fewer lines = preferred)
    4. Return the first template consistent with ALL stored examples

Template levels:
    Level 1 (1 line)  — single primitive output
    Level 2 (2 lines) — fold output (no carry)
    Level 2 (4 lines) — fold output with carry/borrow detection

Concept → primitive mapping:
    'successor'   → 'succ'    (enables fold-succ templates)
    'predecessor' → 'pred'    (enables fold-pred templates)
    'comparison'  → 'compare' (enables compare templates; implied by fold-carry)
"""

from __future__ import annotations

from itertools import permutations
from typing import Callable, Dict, List, Optional, Set, Tuple

from interpreter import ProcessInterpreter
from memory import ExampleStore


# ---------------------------------------------------------------------------
# Concept → primitive mapping
# ---------------------------------------------------------------------------

# Maps concept name to the primitive operation it provides to dependents.
CONCEPT_TO_PRIM: Dict[str, str] = {
    'successor':   'succ',
    'predecessor': 'pred',
    'comparison':  'compare',
}


# ---------------------------------------------------------------------------
# Input variable naming (must match interpreter._make_env)
# ---------------------------------------------------------------------------

def _input_vars(input_type: List[str]) -> List[str]:
    """Assign variable names to inputs, matching ProcessInterpreter._make_env."""
    names: List[str] = []
    letter_names = 'abcdefgh'
    digit_idx = 0
    for type_name in input_type:
        if type_name == 'op':
            names.append('op')
        else:
            names.append(letter_names[digit_idx])
            digit_idx += 1
    return names


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

def _generate_templates(
    input_type: List[str],
    available_ops: Set[str],
) -> List[List[str]]:
    """Generate candidate process templates, sorted by complexity (fewest lines first).

    Only templates that use operations in available_ops are generated.
    This enforces the prerequisite-graph constraint at synthesis time.

    Args:
        input_type:     Concept.input_type (e.g. ['digit', 'op', 'digit'])
        available_ops:  Set of primitive names (succ, pred, compare) derived
                        from the concept's transitive ancestors.

    Returns:
        Sorted list of process templates (each template is List[str]).
    """
    vars_ = _input_vars(input_type)
    digit_vars = [v for v, t in zip(vars_, input_type) if t != 'op']

    templates: List[List[str]] = []

    # ------------------------------------------------------------------
    # Level 1: single-primitive emit (1 line)
    # ------------------------------------------------------------------

    if 'succ' in available_ops:
        for v in digit_vars:
            templates.append([f'emit(succ({v}))'])

    if 'pred' in available_ops:
        for v in digit_vars:
            templates.append([f'emit(pred({v}))'])

    if 'compare' in available_ops and len(digit_vars) >= 2:
        for v1, v2 in permutations(digit_vars, 2):
            templates.append([f'emit(compare({v1}, {v2}))'])

    # ------------------------------------------------------------------
    # Level 2: simple fold — emit a single intermediate value (2 lines)
    # Useful for concepts like modular arithmetic or counting.
    # ------------------------------------------------------------------

    if 'succ' in available_ops and len(digit_vars) >= 2:
        for v1, v2 in permutations(digit_vars, 2):
            templates.append([
                f'result = fold({v1}, {v2}, succ)',
                'emit(result)',
            ])

    if 'pred' in available_ops and len(digit_vars) >= 2:
        for v1, v2 in permutations(digit_vars, 2):
            templates.append([
                f'result = fold({v1}, {v2}, pred)',
                'emit(result)',
            ])

    # ------------------------------------------------------------------
    # Level 2: fold with carry detection — emit (carry, ones) (4 lines)
    # This is the template for single-digit addition.
    # ------------------------------------------------------------------

    if 'succ' in available_ops and len(digit_vars) >= 2:
        for v1, v2 in permutations(digit_vars, 2):
            templates.append([
                f'result = fold({v1}, {v2}, succ)',
                'c = if(compare(result, 9) == GT, 1, 0)',
                'ones = if(c == 1, result - 10, result)',
                'emit(c, ones)',
            ])

    # ------------------------------------------------------------------
    # Level 2: fold with borrow detection — emit (borrow, ones) (4 lines)
    # This is the template for single-digit subtraction.
    # ------------------------------------------------------------------

    if 'pred' in available_ops and len(digit_vars) >= 2:
        for v1, v2 in permutations(digit_vars, 2):
            templates.append([
                f'result = fold({v1}, {v2}, pred)',
                'borrow = if(compare(result, 0) == LT, 1, 0)',
                'ones = if(borrow == 1, result + 10, result)',
                'emit(borrow, ones)',
            ])

    # Sort by line count (fewest lines = simplest = preferred by MDL)
    templates.sort(key=len)

    return templates


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class Synthesizer:
    """Find the shortest process expression consistent with all examples.

    The synthesizer is the consolidation engine: it converts extensional
    knowledge (stored examples) into intensional knowledge (a process rule).
    """

    def synthesize(
        self,
        concept_name: str,
        store: ExampleStore,
        graph,                          # KnowledgeGraph
        interpreter: ProcessInterpreter,
        engine_ask: Callable,
    ) -> Optional[List[str]]:
        """Find the shortest process consistent with all stored examples.

        Returns process lines if successful, None if synthesis fails.

        Failure modes:
          - No examples stored.
          - Concept not in graph.
          - No consistent template found (ambiguous or missing prerequisite).
        """
        if not store.examples:
            return None

        concept = graph.concepts.get(concept_name)
        if concept is None:
            return None

        # Determine which primitive operations are available via ancestors.
        available_ops: Set[str] = set()
        for ancestor_name in graph.ancestors(concept_name):
            prim = CONCEPT_TO_PRIM.get(ancestor_name)
            if prim is not None:
                available_ops.add(prim)

        if not available_ops:
            # No primitives available — synthesis cannot proceed.
            # This is the structural guarantee: the prerequisite graph
            # correctly blocks synthesis when prerequisites are absent.
            return None

        templates = _generate_templates(concept.input_type, available_ops)

        for template in templates:
            if _test_template(
                template, store, concept.input_type, interpreter, engine_ask
            ):
                return template

        return None  # No consistent template found


# ---------------------------------------------------------------------------
# Template testing
# ---------------------------------------------------------------------------

def _test_template(
    template: List[str],
    store: ExampleStore,
    input_type: List[str],
    interpreter: ProcessInterpreter,
    engine_ask: Callable,
) -> bool:
    """Return True iff template correctly predicts every stored example."""
    old_ask = interpreter.engine_ask
    interpreter.engine_ask = engine_ask
    try:
        for inputs, expected in store.examples:
            try:
                actual = interpreter.run(template, inputs, input_type)
            except Exception:
                return False
            if actual != expected:
                return False
        return True
    finally:
        interpreter.engine_ask = old_ask
