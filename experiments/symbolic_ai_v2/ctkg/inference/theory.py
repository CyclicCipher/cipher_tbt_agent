"""
Theory Compartments — Phase 4 of the Einstein Roadmap.

TheoryManager
-------------
Manages named, competing theories inside a MorphismGraph.  Each theory is a
labelled set of morphisms (stored as THEORY_MEMBER self-loops on a theory
object, following the Belief-Layer convention from MorphismGraph.add_theory).

Public API
----------
  register_theory(name, morph_ids)   → TheoryId
  assign_morphism(morph_id, theory_id)          (idempotent)
  predict_under_theory(theory_id, bindings, ctx) → Optional[float]
  consistency_check(theory_a, theory_b, bindings, ctx) → ConsistencyResult
  blame_theory(candidate_theories, bindings, observed, ctx) → Optional[BlameResult]

Iron Law compliance
-------------------
All dispatch is by TheoryId (an ObjectId = int) and MorphId.  Theory names are
stored purely as metadata for diagnostics — no dispatch is on the name string.
predict_under_theory uses FITTED_LAW morphism payloads, calling
predict_continuous() which itself dispatches on NodeId operator keys.

Defect probe (blame locality)
------------------------------
blame_theory MUST return a BlameResult.morph_id — the specific morphism
responsible for the anomaly — NOT just the TheoryId.  A theory with 10
morphisms, exactly 1 wrong, must cause blame_theory to return that one
morphism.  Any implementation returning only the theory id fails the probe.

Bitter Lesson compliance
------------------------
TheoryManager does not hard-code any physical law names.  A Newtonian theory
registered with anonymous operator symbols (⊕, ⊗ …) must behave identically
to one registered with 'mul', 'add' strings — the cage tests verify this across
10 symbol-table seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    MorphId,
    ObjectId,
)
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    predict_continuous,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext

TheoryId = ObjectId

_FITTED_LAW_TYPE = "FITTED_LAW"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    """Outcome of comparing two theories on a single observable.

    Attributes
    ----------
    consistent : True iff |pred_a - pred_b| ≤ tolerance.
    theory_a_id, theory_b_id : the two theories compared.
    pred_a, pred_b : predictions from each theory (None if the theory has
        no FITTED_LAW morphisms for this input).
    gap : |pred_a - pred_b|, or float('inf') if either prediction is None.
    """
    consistent:   bool
    theory_a_id:  TheoryId
    theory_b_id:  TheoryId
    pred_a:       Optional[float]
    pred_b:       Optional[float]
    gap:          float


@dataclass
class BlameResult:
    """The single morphism most responsible for an anomaly.

    Attributes
    ----------
    theory_id : the theory the blamed morphism belongs to.
    morph_id  : THE specific morphism id — not just the theory id.
    pred      : what that morphism predicted (None if evaluation failed).
    error     : |pred - observed|, or float('inf') if pred is None.
    """
    theory_id: TheoryId
    morph_id:  MorphId
    pred:      Optional[float]
    error:     float


@dataclass
class SymmetryCheckResult:
    """Whether a coordinate transform preserves a theory's predictions.

    Attributes
    ----------
    invariant     : True if all tested inputs yield |pred_orig - pred_transformed|
                    ≤ tolerance.
    theory_id     : the theory tested.
    max_deviation : largest deviation across all test inputs.
    n_tested      : number of test inputs evaluated (may be < len(test_inputs) if
                    either prediction returned None).
    """
    invariant:     bool
    theory_id:     "TheoryId"
    max_deviation: float
    n_tested:      int


@dataclass
class CrossTheoryPrediction:
    """Combined prediction from two theory compartments.

    Attributes
    ----------
    prediction    : composition_fn(pred_a, pred_b).
    pred_a        : prediction from theory_a (None if unavailable).
    pred_b        : prediction from theory_b (None if unavailable).
    theory_a_id, theory_b_id : the two theories.
    """
    prediction:   float
    pred_a:       Optional[float]
    pred_b:       Optional[float]
    theory_a_id:  "TheoryId"
    theory_b_id:  "TheoryId"


# ---------------------------------------------------------------------------
# TheoryManager
# ---------------------------------------------------------------------------

class TheoryManager:
    """Manage competing theories stored in a MorphismGraph.

    Parameters
    ----------
    mg : the MorphismGraph that holds all morphisms and theory objects.
    """

    def __init__(self, mg: MorphismGraph) -> None:
        self._mg = mg
        self._names: dict[TheoryId, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_theory(
        self,
        name: str,
        morph_ids: Optional[list[MorphId]] = None,
    ) -> TheoryId:
        """Create a named theory in the graph and return its TheoryId.

        Parameters
        ----------
        name      : human-readable label (metadata only; not used for dispatch).
        morph_ids : initial set of morphisms.  Defaults to empty list.

        Returns
        -------
        TheoryId (an ObjectId integer).
        """
        morph_ids = morph_ids or []
        theory_id = self._mg.add_theory(morph_ids)
        self._names[theory_id] = name
        return theory_id

    def assign_morphism(self, morph_id: MorphId, theory_id: TheoryId) -> None:
        """Add *morph_id* to *theory_id* (idempotent).

        Raises
        ------
        KeyError if *theory_id* is not a registered theory or *morph_id* is
        not present in the graph.
        """
        self._mg.add_theory_member(theory_id, morph_id)

    def theory_name(self, theory_id: TheoryId) -> Optional[str]:
        """Return the human-readable label for a theory, or None."""
        return self._names.get(theory_id)

    def all_theories(self) -> list[tuple[TheoryId, str]]:
        """Return (theory_id, name) for every registered theory."""
        return [
            (tid, self._names.get(tid, ""))
            for tid in (o.obj_id for o in self._mg.theories())
            if tid in self._names
        ]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_under_theory(
        self,
        theory_id: TheoryId,
        input_bindings: dict[str, float],
        ctx: EvalContext,
    ) -> Optional[float]:
        """Predict the output using FITTED_LAW morphisms in *theory_id*.

        Evaluates every FITTED_LAW morphism in the theory.  If the theory
        contains exactly one FITTED_LAW morphism, returns its prediction
        directly.  If there are multiple, returns their simple mean.
        Returns None if the theory contains no FITTED_LAW morphisms.

        Parameters
        ----------
        theory_id     : the theory to query.
        input_bindings: variable-name → float bindings for the formula.
        ctx           : EvalContext supplying operator callables.
        """
        preds = list(self._iter_predictions(theory_id, input_bindings, ctx))
        if not preds:
            return None
        return sum(v for _, v in preds) / len(preds)

    def _iter_predictions(
        self,
        theory_id: TheoryId,
        input_bindings: dict[str, float],
        ctx: EvalContext,
    ):
        """Yield (morph_id, prediction) for each FITTED_LAW morphism in theory.

        Silently skips morphisms whose evaluation raises ValueError.
        """
        member_ids = self._mg.theory_members(theory_id)
        for mid in member_ids:
            m = self._mg.morphism_by_id(mid)
            if m is None or m.morph_type != _FITTED_LAW_TYPE:
                continue
            law: FittedLaw = m.payload
            try:
                val = predict_continuous(law, input_bindings, ctx)
                yield mid, val
            except (ValueError, KeyError, TypeError):
                pass  # binding mismatch — this morphism can't answer

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def consistency_check(
        self,
        theory_a: TheoryId,
        theory_b: TheoryId,
        input_bindings: dict[str, float],
        ctx: EvalContext,
        tolerance: float = 1e-6,
    ) -> ConsistencyResult:
        """Check whether two theories give compatible predictions on an input.

        Parameters
        ----------
        theory_a, theory_b : the two theories to compare.
        input_bindings     : variable → float for the test point.
        ctx                : EvalContext.
        tolerance          : maximum |pred_a - pred_b| for 'consistent'.

        Returns
        -------
        ConsistencyResult with .consistent, .pred_a, .pred_b, .gap.
        """
        pred_a = self.predict_under_theory(theory_a, input_bindings, ctx)
        pred_b = self.predict_under_theory(theory_b, input_bindings, ctx)

        if pred_a is None or pred_b is None:
            gap = float("inf")
            consistent = False
        else:
            gap = abs(pred_a - pred_b)
            consistent = gap <= tolerance

        return ConsistencyResult(
            consistent=consistent,
            theory_a_id=theory_a,
            theory_b_id=theory_b,
            pred_a=pred_a,
            pred_b=pred_b,
            gap=gap,
        )

    # ------------------------------------------------------------------
    # Blame
    # ------------------------------------------------------------------

    def blame_theory(
        self,
        candidate_theories: list[TheoryId],
        input_bindings: dict[str, float],
        observed: float,
        ctx: EvalContext,
    ) -> Optional[BlameResult]:
        """Find the specific morphism most responsible for an anomaly.

        For each FITTED_LAW morphism in every candidate theory, computes
        |prediction - observed|.  Returns a BlameResult pointing at the
        morphism with the largest error — the most egregiously wrong one.

        Returns None if no theory contains any evaluable FITTED_LAW morphism.

        Parameters
        ----------
        candidate_theories : list of TheoryIds to inspect.
        input_bindings     : variable → float for the anomalous observation.
        observed           : the observed (ground-truth) value.
        ctx                : EvalContext.

        Returns
        -------
        BlameResult with .theory_id, .morph_id, .pred, .error.
        The .morph_id is the SPECIFIC morphism, not just the theory.
        """
        worst: Optional[BlameResult] = None

        for theory_id in candidate_theories:
            for mid, pred in self._iter_predictions(theory_id, input_bindings, ctx):
                error = abs(pred - observed)
                if worst is None or error > worst.error:
                    worst = BlameResult(
                        theory_id=theory_id,
                        morph_id=mid,
                        pred=pred,
                        error=error,
                    )

        return worst

    def symmetry_group_check(
        self,
        theory_id: "TheoryId",
        transform_fn: "Callable[[dict[str, float]], dict[str, float]]",
        ctx: EvalContext,
        test_inputs: "list[dict[str, float]]",
        tolerance: float = 1e-6,
    ) -> SymmetryCheckResult:
        """Test whether a coordinate transform leaves theory predictions invariant.

        For each test_input x, computes:
          pred_orig        = predict_under_theory(theory_id, x, ctx)
          pred_transformed = predict_under_theory(theory_id, transform_fn(x), ctx)

        If |pred_orig - pred_transformed| ≤ tolerance for ALL test_inputs,
        returns invariant=True (the theory has this symmetry).

        Parameters
        ----------
        theory_id    : the theory to test.
        transform_fn : coordinate transform x → x' (e.g. Galilean boost).
        ctx          : EvalContext.
        test_inputs  : list of input binding dicts to test.
        tolerance    : max allowed deviation for 'invariant'.

        Returns
        -------
        SymmetryCheckResult.
        """
        max_dev = 0.0
        n_tested = 0

        for inp in test_inputs:
            pred_orig = self.predict_under_theory(theory_id, inp, ctx)
            pred_trans = self.predict_under_theory(theory_id, transform_fn(inp), ctx)

            if pred_orig is None or pred_trans is None:
                continue

            dev = abs(pred_orig - pred_trans)
            if dev > max_dev:
                max_dev = dev
            n_tested += 1

        invariant = (n_tested > 0) and (max_dev <= tolerance)
        return SymmetryCheckResult(
            invariant=invariant,
            theory_id=theory_id,
            max_deviation=max_dev,
            n_tested=n_tested,
        )

    def cross_theory_inference(
        self,
        theory_a: "TheoryId",
        theory_b: "TheoryId",
        bindings_a: "dict[str, float]",
        bindings_b: "dict[str, float]",
        composition_fn: "Callable[[float, float], float]",
        ctx: EvalContext,
    ) -> "Optional[CrossTheoryPrediction]":
        """Combine predictions from two theories via a composition function.

        Evaluates theory_a on bindings_a and theory_b on bindings_b, then applies
        composition_fn(pred_a, pred_b) to produce a combined prediction.

        Typical use: pred_a = relativistic correction (SR theory)
                     pred_b = Newtonian orbital prediction (Newton theory)
                     composition_fn = lambda sr, newton: newton + sr
                     → combined perihelion prediction.

        Returns None if either theory cannot make a prediction.

        Parameters
        ----------
        theory_a, theory_b  : the two theories.
        bindings_a          : input bindings for theory_a.
        bindings_b          : input bindings for theory_b.
        composition_fn      : f(pred_a, pred_b) → combined prediction.
        ctx                 : EvalContext.

        Returns
        -------
        CrossTheoryPrediction, or None if either prediction is unavailable.
        """
        pred_a = self.predict_under_theory(theory_a, bindings_a, ctx)
        pred_b = self.predict_under_theory(theory_b, bindings_b, ctx)

        if pred_a is None or pred_b is None:
            return None

        combined = composition_fn(pred_a, pred_b)
        return CrossTheoryPrediction(
            prediction=combined,
            pred_a=pred_a,
            pred_b=pred_b,
            theory_a_id=theory_a,
            theory_b_id=theory_b,
        )

    def blame_theory_all(
        self,
        candidate_theories: list[TheoryId],
        input_bindings: dict[str, float],
        observed: float,
        ctx: EvalContext,
    ) -> list[BlameResult]:
        """Return all BlameResults sorted by error descending.

        Useful for diagnostics when you want the full ranking of morphisms.
        """
        results: list[BlameResult] = []
        for theory_id in candidate_theories:
            for mid, pred in self._iter_predictions(theory_id, input_bindings, ctx):
                error = abs(pred - observed)
                results.append(BlameResult(
                    theory_id=theory_id,
                    morph_id=mid,
                    pred=pred,
                    error=error,
                ))
        results.sort(key=lambda r: r.error, reverse=True)
        return results
