"""
Theory Revision with Preservation Guarantee — Phase 9 of the Einstein Roadmap.

Implements A-7: theory revision that never breaks phenomena the current theory
already explains correctly.

Problem statement
-----------------
The RetractEngine (Phase 6) can retract a wrong morphism and add a replacement.
But there is no guarantee that the replacement doesn't also break phenomena that
were already correctly predicted (collateral damage).

Concretely: if theory T correctly predicts class A (k=5) AND fails on class C
(k=50), a careless replacement might fit a new law optimised for class C but
break class A.  Phase 6's `propose_replacement` has a `correct_examples`
parameter that partially mitigates this — but it requires the caller to explicitly
supply all the correct examples.  In practice the caller does not always know
which prior observations are at risk.

A-7 solution: PredictionLedger
-------------------------------
A PredictionLedger records every observation that the theory has answered
correctly.  Before applying any revision, the `apply_with_preservation` method
checks that ALL ledger entries still pass.  If any break, the revision is
REJECTED and the theory is restored (the new morphism is vetoed and the old one
is un-vetoed).

Design
------
PredictionLedger
    records   : list[LedgerEntry] — all observations confirmed correct so far.
    record(theory_id, input_bindings, observed, predicted)
        Store a confirmed-correct observation.
    correct_examples(theory_id) → list[tuple[dict, float]]
        Return all stored examples for a theory.

PreservationResult
    passed           : True iff all ledger entries still pass after revision.
    n_preserved      : number of entries that still pass.
    n_broken         : number of entries that broke.
    broken_examples  : the entries that broke (up to max_report).
    reason           : human-readable string.

check_preservation(theory_id, examples, mg, tm, ctx, tolerance)
    For each (input, observed) in examples, compute theory's current prediction.
    Return PreservationResult.

apply_with_preservation(candidate, ctx, mg, tm, reviser, ledger, theory_id,
                        label, tolerance) → Optional[RevisionResult]
    1. Apply the replacement.
    2. Check preservation against ledger.
    3. If preservation fails: undo (veto new, un-veto old).
    4. Return None if preservation failed; RevisionResult otherwise.

Iron Law compliance
-------------------
No token strings used in logic.  All dispatch uses ObjectId / MorphId.
Prediction is delegated to TheoryManager._iter_predictions, which uses
EvalContext keyed by NodeId (opaque int).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    MorphId,
    ObjectId,
)
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId
from experiments.symbolic_ai_v2.ctkg.inference.retract import (
    RetractEngine,
    ReplacementCandidate,
)
from experiments.symbolic_ai_v2.ctkg.inference.revision import (
    ClosedLoopReviser,
    RevisionResult,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext

_DEFAULT_TOLERANCE = 0.05   # relative error below which a prediction "passes"


# ---------------------------------------------------------------------------
# Ledger types
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    """One confirmed-correct observation in the prediction ledger.

    Attributes
    ----------
    theory_id      : the theory that made the correct prediction.
    input_bindings : the input variable bindings ({str: float}).
    observed       : the observed output value.
    predicted      : the predicted value at the time of recording.
    """
    theory_id:      TheoryId
    input_bindings: dict
    observed:       float
    predicted:      float


class PredictionLedger:
    """Records confirmed-correct predictions for one or more theories.

    Maintains a growing list of (input, observed) pairs that the theory
    has answered correctly.  Used by `apply_with_preservation` to check
    that a proposed revision does not break any prior correct prediction.
    """

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    # --- write ---------------------------------------------------------------

    def record(
        self,
        theory_id:      TheoryId,
        input_bindings: dict,
        observed:       float,
        predicted:      float,
    ) -> None:
        """Record a confirmed-correct prediction."""
        self._entries.append(LedgerEntry(
            theory_id=theory_id,
            input_bindings=dict(input_bindings),
            observed=observed,
            predicted=predicted,
        ))

    # --- read ----------------------------------------------------------------

    def all_entries(self, theory_id: Optional[TheoryId] = None) -> list[LedgerEntry]:
        """Return all entries, optionally filtered by theory_id."""
        if theory_id is None:
            return list(self._entries)
        return [e for e in self._entries if e.theory_id == theory_id]

    def correct_examples(
        self,
        theory_id: TheoryId,
    ) -> list[tuple[dict, float]]:
        """Return all stored (input_bindings, observed) pairs for theory_id."""
        return [(e.input_bindings, e.observed)
                for e in self._entries if e.theory_id == theory_id]

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Preservation check
# ---------------------------------------------------------------------------

@dataclass
class PreservationResult:
    """Result of a preservation check.

    Attributes
    ----------
    passed          : True iff ALL examples still pass (n_broken == 0).
    n_preserved     : number of examples still predicted correctly.
    n_broken        : number of examples that broke.
    broken_examples : the (input_bindings, observed) pairs that broke.
    reason          : human-readable explanation.
    """
    passed:          bool
    n_preserved:     int
    n_broken:        int
    broken_examples: list[tuple[dict, float]]
    reason:          str


def check_preservation(
    theory_id: TheoryId,
    examples:  list[tuple[dict, float]],
    mg:        MorphismGraph,
    tm:        TheoryManager,
    ctx:       EvalContext,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> PreservationResult:
    """Check that theory_id still predicts all examples within tolerance.

    For each (input_bindings, observed) pair, the theory's current prediction
    is computed.  If the relative error exceeds tolerance, the example is
    counted as "broken".

    Parameters
    ----------
    theory_id : the theory to check.
    examples  : list of (input_bindings, observed) pairs to verify.
    mg, tm    : graph and theory manager.
    ctx       : EvalContext.
    tolerance : maximum relative error to consider "preserved" (default 0.05).

    Returns
    -------
    PreservationResult.
    """
    n_preserved   = 0
    n_broken      = 0
    broken: list[tuple[dict, float]] = []

    for inp, obs in examples:
        pred = tm.predict_under_theory(theory_id, inp, ctx)
        if pred is None:
            n_broken += 1
            broken.append((inp, obs))
            continue
        denom    = abs(obs) + 1e-12
        rel_err  = abs(pred - obs) / denom
        if rel_err <= tolerance:
            n_preserved += 1
        else:
            n_broken += 1
            broken.append((inp, obs))

    passed = (n_broken == 0)
    reason = (
        "OK — all examples preserved"
        if passed
        else f"{n_broken} example(s) broken by revision"
    )
    return PreservationResult(
        passed=passed,
        n_preserved=n_preserved,
        n_broken=n_broken,
        broken_examples=broken,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Preservation-aware apply
# ---------------------------------------------------------------------------

def apply_with_preservation(
    candidate:  ReplacementCandidate,
    ctx:        EvalContext,
    mg:         MorphismGraph,
    tm:         TheoryManager,
    reviser:    ClosedLoopReviser,
    eng:        RetractEngine,
    ledger:     PredictionLedger,
    label:      str = "__pres_rev__",
    tolerance:  float = _DEFAULT_TOLERANCE,
) -> Optional[RevisionResult]:
    """Apply a replacement and verify no prior correct prediction is broken.

    Algorithm
    ---------
    1. Record the candidate's retract_id as the morphism being replaced.
    2. Apply the replacement via eng.apply_replacement().
    3. Check preservation: run all ledger entries through the updated theory.
    4. If any entry breaks:
       a. Undo: add the new morphism to reviser._vetoed.
       b. Undo: remove the retracted morphism from reviser._vetoed.
       c. Return None.
    5. If all entries pass, return the RevisionResult.

    Parameters
    ----------
    candidate  : ReplacementCandidate from propose_replacement.
    ctx        : EvalContext.
    mg, tm     : graph and theory manager.
    reviser    : ClosedLoopReviser (holds _vetoed set).
    eng        : RetractEngine.
    ledger     : PredictionLedger — contains all prior correct predictions.
    label      : morphism label for the new law.
    tolerance  : relative error threshold for "preserved" (default 0.05).

    Returns
    -------
    RevisionResult if the revision passes preservation, None otherwise.
    """
    examples = ledger.correct_examples(candidate.theory_id)

    # Step 1: apply the replacement
    result = eng.apply_replacement(candidate, ctx, label=label)

    # Step 2: preservation check
    pres = check_preservation(
        candidate.theory_id, examples, mg, tm, ctx, tolerance=tolerance,
    )

    if pres.passed:
        return result

    # Step 3: preservation failed — undo the revision
    # Undo: veto the new morphism (if it was added to the graph)
    if result.candidate.morph_id != -1:
        reviser._vetoed.add(result.candidate.morph_id)
    # Undo: un-veto the retracted morphism
    reviser._vetoed.discard(candidate.retract_id)

    return None


# ---------------------------------------------------------------------------
# Convenience: preservation-aware replacement pipeline
# ---------------------------------------------------------------------------

def propose_and_apply_safe(
    eng:       RetractEngine,
    reviser:   ClosedLoopReviser,
    tm:        TheoryManager,
    mg:        MorphismGraph,
    ledger:    PredictionLedger,
    theory_id: TheoryId,
    anomalies: list[tuple[dict, float]],
    ctx:       EvalContext,
    schema,    # SchematicLaw
    label:     str   = "__safe_rev__",
    tolerance: float = _DEFAULT_TOLERANCE,
) -> Optional[RevisionResult]:
    """Propose + apply a replacement with automatic preservation guarantee.

    1. Uses ledger's correct_examples as the `correct_examples` argument to
       propose_replacement (Phase 6 preservation check during proposal).
    2. After applying, re-checks preservation against the FULL ledger
       (apply_with_preservation).

    Parameters
    ----------
    eng, reviser, tm, mg : the usual components.
    ledger               : PredictionLedger for theory_id.
    theory_id            : theory to revise.
    anomalies            : observations that triggered the revision.
    ctx                  : EvalContext.
    schema               : SchematicLaw template for the replacement law.
    label                : morphism label for the new law.
    tolerance            : relative error threshold.

    Returns
    -------
    RevisionResult if the revision succeeds, None if it is rejected.
    """
    correct_examples = ledger.correct_examples(theory_id)
    candidate = eng.propose_replacement(
        theory_id, anomalies, correct_examples, ctx, schema,
    )
    if candidate is None:
        return None

    return apply_with_preservation(
        candidate, ctx, mg, tm, reviser, eng, ledger,
        label=label, tolerance=tolerance,
    )
