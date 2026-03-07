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
    Level 3 (2 lines) — double fold via fn  (multiplication: a*b)
    Level 4 (2 lines) — triple fold via fn  (exponentiation: a^b)
    Level 5 (3 lines) — fold_until with pair state (division: a//b)

Concept → primitive mapping:
    'successor'   → 'succ'    (enables fold-succ templates L1–L4)
    'predecessor' → 'pred'    (enables fold-pred templates L1–L2, L5)
    'comparison'  → 'compare' (enables compare templates; implied by fold-carry)

fn(param, body) is not a separate CONCEPT_TO_PRIM entry — it is a feature of
the process language (interpreter.py) always available as a special form.
Level 3 and 4 templates use fn to build closures inside fold, enabling
multiplication and exponentiation without any new primitive beyond succ.

fold_until(max_steps, init, step_fn, stop_pred) is always bounded — it cannot
loop forever.  max_steps is always set to one of the input variables, which
for digit-domain inputs guarantees termination in at most max_steps steps.

Key derivations:
    addition:       fold(b, a, succ)                           [from succ]
    multiplication: fold(b, 0, fn(k, fold(a, k, succ)))       [from succ]
    exponentiation: fold(b, 1, fn(acc, fold(a, 0, fn(k, fold(acc, k, succ))))) [succ]
    division:       fold_until(a, pair(a,0), step, stop)       [from pred + compare]
                    where step = fn(s, pair(fold(b,first(s),pred), second(s)+1))
                          stop = fn(s, compare(first(s),b) == LT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import random as _random_mod
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from interpreter import DryRunError, ProcessInterpreter
from memory import ExampleStore


# ---------------------------------------------------------------------------
# Learned template record
# ---------------------------------------------------------------------------

@dataclass
class _LearnedTemplate:
    """A process that successfully synthesized some concept in the past.

    Stored in Synthesizer._learned so that future synthesis attempts can
    try proven patterns before falling back to the fixed template library.

    Fields:
        process_lines:  The actual process (List[str]).
        n_digit_inputs: Number of non-'op' inputs (shapes the template
                        to the target concept's arity).
        required_ops:   Primitive operations the template uses.  A learned
                        template is only tried when required_ops ⊆ available_ops.
        success_count:  Number of distinct concept syntheses that used this
                        template.  Higher count = tried earlier.
        source_concept: Name of the first concept that produced this template
                        (for debugging).
    """
    process_lines: List[str]
    n_digit_inputs: int
    required_ops: FrozenSet[str]
    success_count: int = 1
    source_concept: str = ''


# ---------------------------------------------------------------------------
# Concept → primitive mapping
# ---------------------------------------------------------------------------

# Maps concept name to the primitive operation it provides to dependents.
CONCEPT_TO_PRIM: Dict[str, str] = {
    'successor':          'succ',
    'predecessor':        'pred',
    'comparison':         'compare',
    'visual_perception':  'vision',   # enables visual templates
    # Phase F: Minecraft motor and observation concepts unlock 'minecraft'
    # templates (visual-condition patterns, inventory-delta rules, etc.)
    'motor_forward':      'minecraft',
    'motor_attack':       'minecraft',
    'motor_use':          'minecraft',
    'observe_rgb':        'minecraft',
    'observe_inventory':  'minecraft',
    'log_detection':      'minecraft',
}

# Threshold values tried by approximate synthesis (float literals in templates).
_VISUAL_THRESHOLDS: List[str] = [
    '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.40',
    '0.50', '0.60', '0.70', '0.80', '0.90',
]


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

    # ------------------------------------------------------------------
    # Level 3: double fold via fn — multiplication: a * b (2 lines)
    # fold(b, 0, fn(k, fold(a, k, succ))) counts a up, b times.
    # Equivalent to: result = 0; for _ in range(b): result += a
    # ------------------------------------------------------------------

    if 'succ' in available_ops and len(digit_vars) >= 2:
        for v1, v2 in permutations(digit_vars, 2):
            templates.append([
                f'result = fold({v2}, 0, fn(k, fold({v1}, k, succ)))',
                'emit(result)',
            ])

    # ------------------------------------------------------------------
    # Level 4: triple fold via fn — exponentiation: a ^ b (2 lines)
    # fold(b, 1, fn(acc, fold(a, 0, fn(k, fold(acc, k, succ)))))
    # Outer fold: accumulator starts at 1, multiplied by a b times.
    # Inner double-fold: multiply acc by a via counting-up.
    # ------------------------------------------------------------------

    if 'succ' in available_ops and len(digit_vars) >= 2:
        for v1, v2 in permutations(digit_vars, 2):
            templates.append([
                f'result = fold({v2}, 1, fn(acc, fold({v1}, 0, fn(k, fold(acc, k, succ)))))',
                'emit(result)',
            ])

    # ------------------------------------------------------------------
    # Level 5: fold_until with pair state — division: a // b (3 lines)
    #
    # Counts how many times b can be subtracted from a before the
    # remainder drops below b.  The fold_until hard-caps at v1 steps,
    # which is a safe upper bound because floor(a/b) <= a for b >= 1.
    #
    # state = pair(remaining, count)
    # step  = fn(s, pair(fold(b, first(s), pred), second(s) + 1))
    #           i.e., subtract b from remaining and increment count
    # stop  = fn(s, compare(first(s), b) == LT)
    #           i.e., stop when remaining < b (can't subtract b anymore)
    #
    # Requires 'pred' (for subtraction via fold) and 'compare'.
    # Generates both orderings (v1//v2 and v2//v1).
    # ------------------------------------------------------------------

    if 'pred' in available_ops and 'compare' in available_ops and len(digit_vars) >= 2:
        for v1, v2 in permutations(digit_vars, 2):
            templates.append([
                f'state = fold_until({v1}, pair({v1}, 0), fn(s, pair(fold({v2}, first(s), pred), second(s) + 1)), fn(s, compare(first(s), {v2}) == LT))',
                'result = second(state)',
                'emit(result)',
            ])

    # Phase F: Minecraft causal templates (pure observation patterns)
    templates.extend(_generate_minecraft_templates(input_type, available_ops))

    # Sort by line count (fewest lines = simplest = preferred by MDL)
    templates.sort(key=len)

    return templates


def _generate_visual_templates(
    input_type: List[str],
) -> List[List[str]]:
    """Generate visual concept templates for image-type inputs.

    Called by synthesize_approx() when 'vision' is in available_ops.
    Templates cover a range of biologically-grounded features:
      - Brightness (mean luminance)
      - Texture roughness (std dev)
      - Edge presence (DoG response)
      - Oriented edge energy (single-direction Gabor)
      - Total edge energy (sum over 4 orientations)
      - Face presence (innate face schematic score)

    Each feature is thresholded at multiple values (both polarities).
    Templates are ordered roughly easiest → hardest to compute.

    Output label convention:
        1 = positive class, 0 = negative class.
    """
    vars_ = _input_vars(input_type)
    image_vars = [v for v, t in zip(vars_, input_type) if t == 'image']
    if not image_vars:
        return []

    templates: List[List[str]] = []

    for img in image_vars:
        for t in _VISUAL_THRESHOLDS:
            # ── Brightness ──────────────────────────────────────────────
            templates.append([
                f'gray = img_to_gray({img})',
                f'score = img_mean(gray)',
                f'emit(if(score > {t}, 1, 0))',
            ])
            templates.append([
                f'gray = img_to_gray({img})',
                f'score = img_mean(gray)',
                f'emit(if(score > {t}, 0, 1))',
            ])

            # ── Texture roughness (std) ──────────────────────────────────
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'score = img_std(gray)',
                f'emit(if(score > {t}, 1, 0))',
            ])
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'score = img_std(gray)',
                f'emit(if(score > {t}, 0, 1))',
            ])

            # ── Edge presence (DoG std — higher = more edges) ────────────
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'dog = img_dog(gray, 1.0, 2.0)',
                f'score = img_std(dog)',
                f'emit(if(score > {t}, 1, 0))',
            ])
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'dog = img_dog(gray, 1.0, 2.0)',
                f'score = img_std(dog)',
                f'emit(if(score > {t}, 0, 1))',
            ])

            # ── Horizontal Gabor energy (structural horizontal edges) ─────
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'ge = img_gabor_energy(gray, 0.0, 2.0, 0.2)',
                f'score = img_mean(ge)',
                f'emit(if(score > {t}, 1, 0))',
            ])

            # ── Vertical Gabor energy ─────────────────────────────────────
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'ge = img_gabor_energy(gray, 1.5708, 2.0, 0.2)',
                f'score = img_mean(ge)',
                f'emit(if(score > {t}, 1, 0))',
            ])

            # ── Total Gabor energy (all 4 cardinal orientations) ──────────
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'ge0 = img_mean(img_gabor_energy(gray, 0.0, 2.0, 0.2))',
                f'ge1 = img_mean(img_gabor_energy(gray, 0.7854, 2.0, 0.2))',
                f'ge2 = img_mean(img_gabor_energy(gray, 1.5708, 2.0, 0.2))',
                f'ge3 = img_mean(img_gabor_energy(gray, 2.3562, 2.0, 0.2))',
                f'score = float_add(float_add(ge0, ge1), float_add(ge2, ge3))',
                f'emit(if(score > {t}, 1, 0))',
            ])
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'ge0 = img_mean(img_gabor_energy(gray, 0.0, 2.0, 0.2))',
                f'ge1 = img_mean(img_gabor_energy(gray, 0.7854, 2.0, 0.2))',
                f'ge2 = img_mean(img_gabor_energy(gray, 1.5708, 2.0, 0.2))',
                f'ge3 = img_mean(img_gabor_energy(gray, 2.3562, 2.0, 0.2))',
                f'score = float_add(float_add(ge0, ge1), float_add(ge2, ge3))',
                f'emit(if(score > {t}, 0, 1))',
            ])

            # ── Top-half vs bottom-half DoG std (face structural prior) ─────
            # More edge activity in upper half → face-like arrangement.
            # Directly implements the Goren (1975) logic in a template;
            # the same computation lives in vision.ctkg:face_schematic as
            # a process expression callable via lookup().
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'dog = img_dog(gray, 1.0, 3.0)',
                f'top = img_crop_rel(dog, 0.0, 0.0, 0.5, 1.0)',
                f'bot = img_crop_rel(dog, 0.5, 0.0, 1.0, 1.0)',
                f'score = float_sub(img_std(top), img_std(bot))',
                f'emit(if(score > {t}, 1, 0))',
            ])
            templates.append([
                f'gray = img_normalize(img_to_gray({img}))',
                f'dog = img_dog(gray, 1.0, 3.0)',
                f'top = img_crop_rel(dog, 0.0, 0.0, 0.5, 1.0)',
                f'bot = img_crop_rel(dog, 0.5, 0.0, 1.0, 1.0)',
                f'score = float_sub(img_std(top), img_std(bot))',
                f'emit(if(score > {t}, 0, 1))',
            ])

    return templates


def _generate_prereq_lookup_templates(
    concept_name: str,
    input_type: List[str],
    available_ops: Set[str],
    graph,
) -> List[List[str]]:
    """Generate templates that call prerequisite concepts via lookup().

    For each ancestor concept that has a process block defined, generates
    threshold templates that obtain the feature score via lookup(prereq, img).

    This is the key architectural mechanism: face understanding lives in
    the CTKG graph as a process expression (vision.ctkg:face_schematic),
    not as hardcoded Python.  The synthesizer discovers how to USE it
    (what threshold works) from examples — exactly as it discovers fold+succ
    for addition from arithmetic examples.

    Generated template shapes:
        (a) Single-lookup threshold — each polarity
        (b) Lookup + horizontal Gabor energy (face + fur)
        (c) Lookup + total Gabor energy all 4 orientations (face + texture)
    """
    vars_   = _input_vars(input_type)
    img_vars = [v for v, t in zip(vars_, input_type) if t == 'image']
    if not img_vars:
        return []

    # Ancestor concepts that have processes defined (can be called via lookup).
    prereqs_with_process: List[str] = []
    for ancestor_name in sorted(graph.ancestors(concept_name)):
        concept = graph.concepts.get(ancestor_name)
        if concept is not None and concept.process:
            prereqs_with_process.append(ancestor_name)

    if not prereqs_with_process:
        return []

    templates: List[List[str]] = []

    for img in img_vars:
        for prereq in prereqs_with_process:
            for t in _VISUAL_THRESHOLDS:
                # (a) Single lookup threshold
                templates.append([
                    f'score = first(lookup({prereq}, {img}))',
                    f'emit(if(score > {t}, 1, 0))',
                ])
                templates.append([
                    f'score = first(lookup({prereq}, {img}))',
                    f'emit(if(score > {t}, 0, 1))',
                ])

                # (b) Lookup + horizontal Gabor energy (face score + fur texture)
                templates.append([
                    f'face_s = first(lookup({prereq}, {img}))',
                    f'gray = img_normalize(img_to_gray({img}))',
                    f'tex = img_mean(img_gabor_energy(gray, 0.0, 2.0, 0.2))',
                    f'score = float_add(face_s, tex)',
                    f'emit(if(score > {t}, 1, 0))',
                ])
                templates.append([
                    f'face_s = first(lookup({prereq}, {img}))',
                    f'gray = img_normalize(img_to_gray({img}))',
                    f'tex = img_mean(img_gabor_energy(gray, 0.0, 2.0, 0.2))',
                    f'score = float_add(face_s, tex)',
                    f'emit(if(score > {t}, 0, 1))',
                ])

                # (c) Lookup + total Gabor energy (face score + multi-orientation texture)
                templates.append([
                    f'face_s = first(lookup({prereq}, {img}))',
                    f'gray = img_normalize(img_to_gray({img}))',
                    f'ge0 = img_mean(img_gabor_energy(gray, 0.0, 2.0, 0.2))',
                    f'ge1 = img_mean(img_gabor_energy(gray, 0.7854, 2.0, 0.2))',
                    f'ge2 = img_mean(img_gabor_energy(gray, 1.5708, 2.0, 0.2))',
                    f'ge3 = img_mean(img_gabor_energy(gray, 2.3562, 2.0, 0.2))',
                    f'tex = float_add(float_add(ge0, ge1), float_add(ge2, ge3))',
                    f'score = float_add(face_s, tex)',
                    f'emit(if(score > {t}, 1, 0))',
                ])
                templates.append([
                    f'face_s = first(lookup({prereq}, {img}))',
                    f'gray = img_normalize(img_to_gray({img}))',
                    f'ge0 = img_mean(img_gabor_energy(gray, 0.0, 2.0, 0.2))',
                    f'ge1 = img_mean(img_gabor_energy(gray, 0.7854, 2.0, 0.2))',
                    f'ge2 = img_mean(img_gabor_energy(gray, 1.5708, 2.0, 0.2))',
                    f'ge3 = img_mean(img_gabor_energy(gray, 2.3562, 2.0, 0.2))',
                    f'tex = float_add(float_add(ge0, ge1), float_add(ge2, ge3))',
                    f'score = float_add(face_s, tex)',
                    f'emit(if(score > {t}, 0, 1))',
                ])

    return templates


# ---------------------------------------------------------------------------
# Approximate template testing
# ---------------------------------------------------------------------------

def _test_template_approx(
    template: List[str],
    store,          # ExampleStore (uses only .examples)
    input_type: List[str],
    interpreter: ProcessInterpreter,
    engine_ask: Callable,
    subsample: Optional[int] = None,
) -> float:
    """Return classification accuracy of template on (a subsample of) examples.

    Unlike _test_template (which requires exact match on all examples),
    this returns a fraction in [0.0, 1.0].  Used for statistical / visual
    concepts where no template achieves 100% accuracy.

    Args:
        subsample: if given, evaluate on at most this many examples
                   (random subsample for speed during synthesis).
    """
    import random as _random
    examples = store.examples
    if subsample and len(examples) > subsample:
        examples = _random.sample(examples, subsample)

    old_ask = interpreter.engine_ask
    interpreter.engine_ask = engine_ask
    correct = 0
    total   = 0
    try:
        for inputs, expected in examples:
            try:
                actual = interpreter.run(template, inputs, input_type)
                if actual == expected:
                    correct += 1
            except Exception:
                pass
            total += 1
        return correct / total if total > 0 else 0.0
    finally:
        interpreter.engine_ask = old_ask


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class Synthesizer:
    """Find the shortest process expression consistent with all examples.

    The synthesizer is the consolidation engine: it converts extensional
    knowledge (stored examples) into intensional knowledge (a process rule).

    Template learning:
        After each successful synthesis, the discovered process is stored in
        _learned (a dynamic library).  On future synthesis calls, learned
        templates that match the arity and available_ops of the target concept
        are tried AFTER the fixed templates, sorted by success_count (most
        successful first).  This allows the system to re-use patterns it has
        already discovered without hard-coding them.

        Limitation: learned templates are re-parameterised by n_digit_inputs
        only — the variable names (a, b, c, ...) must match.  This is correct
        because _input_vars() assigns names deterministically from input_type,
        so two concepts with the same n_digit_inputs will share the same
        variable names and can exchange templates safely.
    """

    def __init__(self) -> None:
        self._learned: List[_LearnedTemplate] = []

    def register_success(
        self,
        concept_name: str,
        process_lines: List[str],
        input_type: List[str],
        available_ops: Set[str],
    ) -> None:
        """Record a successful synthesis so future attempts can re-use it.

        If the exact same process_lines already exists in _learned, increment
        its success_count.  Otherwise append a new entry.
        """
        n = sum(1 for t in input_type if t != 'op')
        for tmpl in self._learned:
            if tmpl.process_lines == process_lines and tmpl.n_digit_inputs == n:
                tmpl.success_count += 1
                return
        self._learned.append(_LearnedTemplate(
            process_lines=list(process_lines),
            n_digit_inputs=n,
            required_ops=frozenset(available_ops),
            success_count=1,
            source_concept=concept_name,
        ))

    def synthesize(
        self,
        concept_name: str,
        store: ExampleStore,
        graph,                          # KnowledgeGraph
        interpreter: ProcessInterpreter,
        engine_ask: Callable,
    ) -> Optional[List[str]]:
        """Find the shortest process consistent with all stored examples.

        Search order:
          1. Fixed templates (sorted by line count — MDL preference).
          2. Learned templates not already in fixed list, sorted by
             success_count descending (most-proven patterns first).

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

        fixed_templates = _generate_templates(concept.input_type, available_ops)
        fixed_set = {tuple(t) for t in fixed_templates}

        # Learned templates: matching arity, required_ops ⊆ available_ops,
        # not already in fixed list, sorted by success_count descending.
        n_digit = sum(1 for t in concept.input_type if t != 'op')
        learned_candidates = [
            tmpl for tmpl in self._learned
            if (tmpl.n_digit_inputs == n_digit
                and tmpl.required_ops.issubset(available_ops)
                and tuple(tmpl.process_lines) not in fixed_set)
        ]
        learned_candidates.sort(key=lambda t: -t.success_count)
        learned_templates = [t.process_lines for t in learned_candidates]

        all_templates = fixed_templates + learned_templates

        for template in all_templates:
            if _test_template(
                template, store, concept.input_type, interpreter, engine_ask
            ):
                return template

        return None  # No consistent template found

    def synthesize_approx(
        self,
        concept_name: str,
        store: ExampleStore,
        graph,                          # KnowledgeGraph
        interpreter: ProcessInterpreter,
        engine_ask: Callable,
        accuracy_threshold: float = 0.85,
        subsample: Optional[int] = 300,
        verbose: bool = False,
    ) -> Optional[Tuple[List[str], float]]:
        """Find a process that achieves >= accuracy_threshold on stored examples.

        Used for statistical / visual concepts where no template achieves 100%.
        Returns (process_lines, accuracy) on success, None on failure.

        Search order:
          1. Learned templates (highest success_count first) — often fastest win.
          2. Visual templates from _generate_visual_templates() if 'vision' in
             available_ops — covers brightness, DoG, Gabor, top/bottom split.
          3. Lookup-based templates from _generate_prereq_lookup_templates() —
             calls CTKG concepts with processes (e.g. face_schematic) via lookup().

        Args:
            accuracy_threshold: Minimum fraction of examples correctly predicted.
            subsample: Evaluate each template on at most this many examples
                       for speed; then re-verify the winner on ALL examples.
            verbose: Print per-template accuracy while searching.

        Failure modes:
          - No examples stored.
          - Concept not in graph.
          - No template achieves accuracy_threshold.
        """
        if not store.examples:
            return None

        concept = graph.concepts.get(concept_name)
        if concept is None:
            return None

        available_ops: Set[str] = set()
        for ancestor_name in graph.ancestors(concept_name):
            prim = CONCEPT_TO_PRIM.get(ancestor_name)
            if prim is not None:
                available_ops.add(prim)

        # Candidates: learned templates first (proven patterns), then visual
        n_digit = sum(1 for t in concept.input_type if t != 'op')
        learned_candidates = sorted(
            [tmpl for tmpl in self._learned
             if tmpl.n_digit_inputs == n_digit
             and tmpl.required_ops.issubset(available_ops)],
            key=lambda t: -t.success_count,
        )
        candidates = [t.process_lines for t in learned_candidates]

        if 'vision' in available_ops:
            candidates += _generate_visual_templates(concept.input_type)
            # Lookup-based templates: call prerequisite CTKG process expressions.
            # face_schematic is the prime example — knowledge in the graph, not Python.
            candidates += _generate_prereq_lookup_templates(
                concept_name, concept.input_type, available_ops, graph
            )

        if not candidates:
            return None

        best_template: Optional[List[str]] = None
        best_acc = 0.0

        for i, template in enumerate(candidates):
            acc = _test_template_approx(
                template, store, concept.input_type,
                interpreter, engine_ask, subsample=subsample,
            )
            if verbose and i % 20 == 0:
                print(f'  [approx synthesis] {i}/{len(candidates)} templates, '
                      f'best so far: {best_acc:.2%}', end='\r')
            if acc > best_acc:
                best_acc    = acc
                best_template = template
            if acc >= accuracy_threshold:
                # Fast path: re-verify on ALL examples before accepting
                full_acc = _test_template_approx(
                    template, store, concept.input_type,
                    interpreter, engine_ask, subsample=None,
                )
                if full_acc >= accuracy_threshold:
                    if verbose:
                        print(f'\n  Found: {full_acc:.2%} accuracy')
                    return template, full_acc

        if verbose:
            print(f'\n  Best accuracy: {best_acc:.2%} (threshold {accuracy_threshold:.0%})')

        # Return best even if below threshold (caller decides what to do)
        if best_template is not None and best_acc > 0:
            full_acc = _test_template_approx(
                best_template, store, concept.input_type,
                interpreter, engine_ask, subsample=None,
            )
            return best_template, full_acc

        return None


# ---------------------------------------------------------------------------
# Template testing
# ---------------------------------------------------------------------------

def _generate_minecraft_templates(
    input_type: List[str],
    available_ops: Set[str],
) -> List[List[str]]:
    """Phase F: Minecraft causal synthesis templates.

    These are PURE templates — they read observation data (frames,
    inventory counts) and return predicted conditions or counts.
    Motor actions are the agent's BEHAVIOR (Phase G policy); these
    templates express the agent's KNOWLEDGE about what conditions
    lead to what observable outcomes.

    Generated when 'minecraft' in available_ops (i.e. the concept's
    ancestors include at least one Minecraft motor or observation primitive).

    Template families:
      MC-1  Visual condition — log_detection from frame
      MC-2  Inventory threshold — item count exceeds N
      MC-3  Frame-difference threshold — motion intensity signal
      MC-4  Causal delta — inventory before vs after (requires compare)
      MC-5  Day/night predictor from time ticks
    """
    if 'minecraft' not in available_ops:
        return []

    vars_      = _input_vars(input_type)
    frame_vars = [v for v, t in zip(vars_, input_type) if t == 'mc_frame']
    count_vars = [v for v, t in zip(vars_, input_type) if t == 'item_count']
    time_vars  = [v for v, t in zip(vars_, input_type) if t == 'time_ticks']

    templates: List[List[str]] = []

    # MC-1: Visual log detection (pure wrapper: returns bool from frame)
    for fv in frame_vars:
        templates.append([
            f'detected = log_detection({fv})',
            'emit(detected,)',
        ])

    # MC-2: Inventory threshold (does the agent have >= N of this item?)
    for cv in count_vars:
        for threshold in (1, 2, 4, 8, 16):
            templates.append([
                f'emit(if(compare({cv}, {threshold}) == LT, 0, 1))',
            ])
        # Direct pass-through (inventory count as output)
        templates.append([f'emit({cv},)'])

    # MC-3: Frame-difference magnitude (motion intensity from temporal buffer)
    for fv in frame_vars:
        templates.append([
            f'diff = img_mean(img_dog({fv}, 1.0, 2.0))',
            f'emit(diff,)',
        ])

    # MC-4: Causal delta — "did count increase?"  (requires compare)
    if 'compare' in available_ops and len(count_vars) >= 2:
        for i, cv1 in enumerate(count_vars):
            for cv2 in count_vars[i + 1:]:
                templates.append([
                    f'delta = compare({cv2}, {cv1})',
                    'emit(if(delta == GT, 1, 0))',
                ])

    # MC-5: Day/night predictor
    for tv in time_vars:
        # Night if 13000 <= time <= 23000
        templates.append([
            f'is_day1 = compare({tv}, 13000)',
            f'is_day2 = compare({tv}, 23000)',
            f'is_night = if(is_day1 == LT, 0, if(is_day2 == GT, 0, 1))',
            'emit(is_night,)',
        ])

    return templates


# ---------------------------------------------------------------------------
# Phase O: Unsupervised category discovery (distributional hypothesis)
# ---------------------------------------------------------------------------

def _kmeans_cluster(
    matrix:      'np.ndarray',
    k:           int,
    n_iter:      int = 30,
    random_seed: int = 42,
) -> List[int]:
    """K-means clustering on rows of `matrix` using cosine distance.

    Uses K-means++ initialisation for reproducible, quality centroids.
    Row vectors are L2-normalised so dot-product == cosine similarity.

    Args:
        matrix:      (n, d) float32 array — one probability vector per row.
        k:           Number of clusters.
        n_iter:      Maximum EM iterations (early stop on convergence).
        random_seed: Seed for K-means++ initialisation.

    Returns:
        List[int] of length n with cluster assignments in 0..k-1.
    """
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float32)
    n, d   = matrix.shape
    if k >= n:
        return list(range(n))

    rng = np.random.RandomState(random_seed)

    # L2-normalise rows so dot-product gives cosine similarity.
    # Replace zero-norm rows with a uniform direction to avoid NaN.
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-12, 1.0, norms)
    normed = matrix / norms                                    # (n, d)

    # ---- K-means++ initialisation ----------------------------------------
    centres_idx: List[int] = [int(rng.randint(n))]
    for _ in range(1, k):
        c       = normed[centres_idx]                          # (c, d)
        sims    = normed @ c.T                                 # (n, c)
        # Distance = 1 - max_cosine_similarity (clipped to [0,2]).
        max_sim = sims.max(axis=1)                             # (n,)
        dist2   = np.maximum(0.0, 1.0 - max_sim) ** 2         # (n,)
        total   = dist2.sum()
        if total < 1e-12:
            centres_idx.append(int(rng.randint(n)))
        else:
            centres_idx.append(int(rng.choice(n, p=dist2 / total)))

    centroids   = normed[centres_idx].copy()                   # (k, d)
    assignments = np.zeros(n, dtype=np.int32)

    # ---- EM iterations ---------------------------------------------------
    for _ in range(n_iter):
        sims            = normed @ centroids.T                 # (n, k)
        new_assignments = sims.argmax(axis=1).astype(np.int32)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        # Update centroids as the unit-normalised mean of each cluster.
        for ci in range(k):
            mask = assignments == ci
            if not mask.any():
                continue                                       # empty cluster: keep old centroid
            mean = normed[mask].mean(axis=0)
            nm   = np.linalg.norm(mean)
            centroids[ci] = mean / nm if nm > 1e-12 else mean

    return assignments.tolist()


def discover_categories(
    store:        ExampleStore,
    n_clusters:   int            = 9,
    min_examples: int            = 1,
    vocab_size:   Optional[int]  = None,
    method:       str            = 'auto',
) -> Dict[tuple, int]:
    """Discover latent input categories by JS-divergence / cosine clustering.

    Implements the distributional hypothesis (Firth 1957):
    inputs with similar output distributions belong to the same category.

    Given a store with (word,) -> (next_word,) examples (one per sequential
    step), groups input tokens by what tends to follow them.  The resulting
    clusters are POS-like without POS being pre-specified:

      DET cluster:  the, a, an     -- all precede NOUN/ADJ tokens
      NOUN cluster: cat, dog, mat  -- all precede VERB/PREP tokens
      VERB cluster: sat, ran, is   -- all precede NOUN/PREP/ADJ tokens

    The same algorithm discovers action categories in Minecraft, harmonic
    roles in music, or motor primitives in motor control -- no domain
    knowledge required.  Only sequential data.

    Args:
        store:        ExampleStore with (word,) -> (next_word,) format.
                      Each input is a single-element tuple.
        n_clusters:   Target number of categories to discover.
        min_examples: Minimum observations of a word to include it.
        vocab_size:   Cap the output vocabulary to the top-N most frequent
                      output tokens.  Distributions are re-normalised over
                      the reduced vocabulary.  None = use all observed tokens.
                      Recommended for large corpora (e.g. 2000 for WikiText-2).
        method:       Clustering algorithm to use.
                        'auto'          -- agglomerative for n<=200,
                                          k-means (numpy) otherwise.
                        'agglomerative' -- bottom-up hierarchical (O(n^3 * V)).
                                          Best quality; impractical for n>1000.
                        'kmeans'        -- K-means with cosine distance.
                                          Requires numpy; scales to n~10000.

    Returns:
        Dict mapping each included input tuple -> cluster_id (0..k-1).
        Words with fewer than min_examples observations are excluded.

    Complexity:
        agglomerative: O(n^3 * V) -- fine for n<=200 (built-in corpus).
        kmeans:        O(n * k * V * T) with T~20 -- fine for n~5000 (WikiText-2).
    """
    import math        as _math
    import collections as _col

    # ------------------------------------------------------------------
    # Step 1: Filter by minimum observation count.
    # ------------------------------------------------------------------
    input_counts: Dict[tuple, int] = _col.Counter(
        inp for inp, _ in store.examples
    )
    eligible: List[tuple] = [
        inp for inp, cnt in input_counts.items() if cnt >= min_examples
    ]
    if not eligible:
        return {}

    n_clusters = min(n_clusters, len(eligible))
    if n_clusters <= 1:
        return {inp: 0 for inp in eligible}

    # ------------------------------------------------------------------
    # Step 2: Build empirical output distributions.
    # ------------------------------------------------------------------
    freq_table = store.build_full_freq_table()
    dists: Dict[tuple, Dict[tuple, float]] = {
        inp: freq_table[inp] for inp in eligible if inp in freq_table
    }
    eligible   = list(dists.keys())
    if not eligible:
        return {}

    n_clusters = min(n_clusters, len(eligible))

    # ------------------------------------------------------------------
    # Step 3: Build shared output vocabulary (optionally capped).
    # ------------------------------------------------------------------
    if vocab_size is not None:
        # Weight output tokens by the count of their source input word.
        output_freq: Dict[tuple, float] = _col.Counter()
        for inp, dist in dists.items():
            w = input_counts[inp]
            for out, p in dist.items():
                output_freq[out] += p * w
        top_outputs  = [out for out, _ in output_freq.most_common(vocab_size)]
        all_outputs  = sorted(top_outputs)

        # Re-normalise each distribution over the restricted vocabulary.
        renormed: Dict[tuple, Dict[tuple, float]] = {}
        for inp, dist in dists.items():
            mass = sum(dist.get(out, 0.0) for out in all_outputs)
            if mass < 1e-12:
                renormed[inp] = {out: 1.0 / len(all_outputs) for out in all_outputs}
            else:
                renormed[inp] = {out: dist.get(out, 0.0) / mass
                                 for out in all_outputs}
        dists = renormed
    else:
        all_outputs = sorted(
            {out for dist in dists.values() for out in dist}
        )

    out_idx: Dict[tuple, int] = {out: i for i, out in enumerate(all_outputs)}
    V = len(all_outputs)

    def dist_to_vec(dist: Dict[tuple, float]) -> List[float]:
        vec = [0.0] * V
        for out, p in dist.items():
            idx = out_idx.get(out)
            if idx is not None:
                vec[idx] = p
        return vec

    vecs_list: List[List[float]] = [dist_to_vec(dists[inp]) for inp in eligible]

    # ------------------------------------------------------------------
    # Step 4: Cluster.
    # ------------------------------------------------------------------
    n_eligible = len(eligible)
    want_kmeans = (
        method == 'kmeans'
        or (method == 'auto' and n_eligible > 200)
    )

    kmeans_ok = False
    raw_assignments: List[int] = []

    if want_kmeans:
        try:
            import numpy as np
            matrix          = np.array(vecs_list, dtype=np.float32)
            raw_assignments = _kmeans_cluster(matrix, n_clusters)
            kmeans_ok       = True
        except ImportError:
            pass  # Fall back to agglomerative below.

    if kmeans_ok:
        result: Dict[tuple, int] = {}
        for inp, cid in zip(eligible, raw_assignments):
            result[inp] = int(cid)
        return result

    # ---- Agglomerative clustering (fallback / explicit choice) ----------
    def jsd(p: List[float], q: List[float]) -> float:
        """Jensen-Shannon divergence (symmetric, bounded in [0, 1])."""
        result = 0.0
        for pi, qi in zip(p, q):
            mi = 0.5 * (pi + qi)
            if pi > 1e-12 and mi > 1e-12:
                result += 0.5 * pi * _math.log2(pi / mi)
            if qi > 1e-12 and mi > 1e-12:
                result += 0.5 * qi * _math.log2(qi / mi)
        return max(0.0, result)

    vecs:    List[List[float]]  = vecs_list
    members: List[List[tuple]] = [[inp] for inp in eligible]
    sizes:   List[int]         = [1] * n_eligible
    active:  List[int]         = list(range(n_eligible))

    while len(active) > n_clusters:
        best_dist = float('inf')
        best_i = best_j = -1
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                ci, cj = active[ii], active[jj]
                d = jsd(vecs[ci], vecs[cj])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = ci, cj
        if best_i < 0:
            break

        # Weighted average merge.
        ni, nj = sizes[best_i], sizes[best_j]
        total  = ni + nj
        vi, vj = vecs[best_i], vecs[best_j]
        vecs[best_i] = [(ni * a + nj * b) / total for a, b in zip(vi, vj)]
        sizes[best_i] = total
        members[best_i].extend(members[best_j])
        active.remove(best_j)

    result_agg: Dict[tuple, int] = {}
    for new_id, ci in enumerate(active):
        for inp in members[ci]:
            result_agg[inp] = new_id
    return result_agg


def _test_template(
    template: List[str],
    store: ExampleStore,
    input_type: List[str],
    interpreter: ProcessInterpreter,
    engine_ask: Callable,
) -> bool:
    """Return True iff template correctly predicts every stored example.

    Always runs in dry_run=True mode so that effectful primitives
    (e.g. mc_attack) raise DryRunError instead of sending game commands.
    DryRunError → False (template cannot be evaluated without live env).
    """
    old_ask = interpreter.engine_ask
    interpreter.engine_ask = engine_ask
    try:
        for inputs, expected in store.examples:
            try:
                actual = interpreter.run(
                    template, inputs, input_type, dry_run=True
                )
            except DryRunError:
                return False  # Template needs live env — skip it
            except Exception:
                return False
            if actual != expected:
                return False
        return True
    finally:
        interpreter.engine_ask = old_ask
