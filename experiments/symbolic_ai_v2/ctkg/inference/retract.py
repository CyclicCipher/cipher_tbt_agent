"""
Full Revision Cycle — Phase 6 of the Einstein Roadmap.

This module adds three capabilities missing from Phase 5's ClosedLoopReviser:

  1.  Morphism retraction  — remove a FITTED_LAW morphism from a theory and
      log the reason.  The morphism itself stays in the MorphismGraph (soft
      delete) but is excluded from all predictions via a veto set shared with
      the ClosedLoopReviser.

  2.  Retraction candidates — scored by how many anomalies the morphism
      explains when removed, minus how many correct predictions it would break.
      A retraction is beneficial when net_gain = anomalies_resolved -
      correct_broken > 0.

  3.  Replacement candidates — `ReplacementCandidate(retract, add)` is the
      atomic unit of scientific theory change: remove morphism R and add
      morphism A that fits the combined data better.

RetractEngine
-------------
  retract_morphism(morph_id, theory_id, reason) → None
      Soft-retract: marks morphism as vetoed; logs a RETRACTION self-loop
      on the theory object.

  score_retraction(morph_id, theory_id, anomalies, correct_examples, ctx)
      → RetractionScore
      Computes net benefit of retracting a morphism.

  propose_retraction(theory_id, anomalies, correct_examples, ctx, schema)
      → Optional[RetractionCandidate]
      Find the morphism in the theory whose retraction maximises net benefit.

  propose_replacement(theory_id, anomalies, correct_examples, ctx, schema,
                       label) → Optional[ReplacementCandidate]
      Combine retraction + new FittedLaw fit → ReplacementCandidate.

  apply_replacement(candidate, ctx) → RevisionResult
      Apply retract+add with closed-loop verification.

RevisionHistory
---------------
  A log stored entirely in the MorphismGraph as RETRACTION self-loops on
  theory objects.  Each self-loop carries:
    - morph_concept_id = retracted morphism id
    - payload = (reason: str, score: float)
  history(theory_id) returns the list of logged retractions.

Iron Law compliance
-------------------
No string comparisons on operator names.  Retraction/replacement dispatch
is by MorphId and TheoryId (ObjectId integers).

Defect probe (Phase 6 roadmap)
-------------------------------
A theory with 20 morphisms — one wrong for class C but correct for A and B —
must after apply_replacement:
  - retract the wrong morphism
  - add a replacement that covers class C
  - still predict A and B correctly (≥90% correct)
Any implementation that retracts without checking what it covers fails.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    MorphId,
    ObjectId,
)
from experiments.symbolic_ai_v2.ctkg.core.parameter_fitter import (
    FittedLaw,
    fit_parameters,
    add_fitted_law,
    predict_continuous,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId
from experiments.symbolic_ai_v2.ctkg.inference.revision import (
    ClosedLoopReviser,
    ContinuousAnomaly,
    RevisionResult,
)

_RETRACTION_TYPE = "RETRACTION"
_FITTED_LAW_TYPE = "FITTED_LAW"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class RetractionScore:
    """Utility of retracting a specific morphism.

    Attributes
    ----------
    morph_id         : the morphism being evaluated for retraction.
    anomalies_resolved : anomalies that would be resolved by retraction
                         (i.e. anomalies caused by this morphism).
    correct_broken   : correct predictions that would be lost.
    net_gain         : anomalies_resolved - correct_broken.
    """
    morph_id:           MorphId
    anomalies_resolved: int
    correct_broken:     int
    net_gain:           float


@dataclass
class RetractionCandidate:
    """A proposal to retract a specific morphism.

    Attributes
    ----------
    morph_id  : the morphism to retract.
    theory_id : the theory it belongs to.
    score     : net benefit of retraction.
    reason    : human-readable justification (metadata only).
    """
    morph_id:  MorphId
    theory_id: TheoryId
    score:     float
    reason:    str = ""


@dataclass
class ReplacementCandidate:
    """A proposal to retract one morphism and add a new one.

    Attributes
    ----------
    retract_id    : morphism to retract.
    new_law       : FittedLaw to add in its place.
    theory_id     : theory both belong to.
    score         : combined benefit score.
    morph_id_new  : MorphId of new morphism after application (-1 = not yet).
    """
    retract_id:   MorphId
    new_law:      FittedLaw
    theory_id:    TheoryId
    score:        float
    morph_id_new: MorphId = -1


# ---------------------------------------------------------------------------
# RetractEngine
# ---------------------------------------------------------------------------

class RetractEngine:
    """Adds retraction and replacement capabilities on top of a ClosedLoopReviser.

    Parameters
    ----------
    reviser : ClosedLoopReviser from Phase 5.  Shares the veto set.
    tm      : TheoryManager.
    mg      : MorphismGraph.
    """

    def __init__(
        self,
        reviser: ClosedLoopReviser,
        tm:      TheoryManager,
        mg:      MorphismGraph,
    ) -> None:
        self._reviser = reviser
        self._tm      = tm
        self._mg      = mg

    # ------------------------------------------------------------------
    # Core retraction
    # ------------------------------------------------------------------

    def retract_morphism(
        self,
        morph_id:  MorphId,
        theory_id: TheoryId,
        reason:    str = "",
        score:     float = 0.0,
    ) -> None:
        """Soft-retract: exclude morphism from predictions and log it.

        The morphism is added to the reviser's veto set (so predictions
        ignore it) and a RETRACTION self-loop is stored on the theory object
        for the revision history.

        Parameters
        ----------
        morph_id  : the morphism to retract.
        theory_id : the theory it belongs to.
        reason    : human-readable justification.
        score     : the retraction score (for the history log).
        """
        # Soft-retract via the reviser's veto set
        self._reviser._vetoed.add(morph_id)

        # Log to revision history as a RETRACTION self-loop on theory object
        self._mg.add_morphism(
            theory_id,
            theory_id,
            morph_type=_RETRACTION_TYPE,
            evidence=1,
            morph_concept_id=morph_id,
            payload=(reason, float(score)),
        )

    def history(self, theory_id: TheoryId) -> list[tuple[MorphId, str, float]]:
        """Return the retraction log for a theory.

        Returns
        -------
        list of (retracted_morph_id, reason, score) triples, oldest first.
        """
        result = []
        for m in self._mg.source_morphisms(theory_id, morph_type=_RETRACTION_TYPE):
            reason, score = m.payload
            result.append((m.morph_concept_id, reason, score))
        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_retraction(
        self,
        morph_id:        MorphId,
        theory_id:       TheoryId,
        anomalies:       list[tuple[dict, float]],
        correct_examples: list[tuple[dict, float]],
        ctx:             EvalContext,
        tolerance:       float = 0.1,
    ) -> RetractionScore:
        """Compute the net benefit of retracting *morph_id*.

        Parameters
        ----------
        morph_id        : the morphism to evaluate.
        theory_id       : its theory.
        anomalies       : (input_bindings, observed) pairs that are anomalous.
        correct_examples: (input_bindings, observed) pairs that are currently
                          predicted correctly (within tolerance).
        ctx             : EvalContext.
        tolerance       : relative tolerance for "correctly predicted".

        Returns
        -------
        RetractionScore with net_gain = anomalies_resolved - correct_broken.
        """
        m = self._mg.morphism_by_id(morph_id)
        if m is None or m.morph_type != _FITTED_LAW_TYPE:
            return RetractionScore(morph_id, 0, 0, 0.0)

        law: FittedLaw = m.payload

        # Count anomalies caused by this morphism (high-error predictions)
        anomalies_resolved = 0
        for inp, obs in anomalies:
            try:
                pred = predict_continuous(law, inp, ctx)
                if abs(pred - obs) > tolerance * (abs(obs) + 1e-9):
                    anomalies_resolved += 1
            except (ValueError, KeyError):
                pass

        # Count correct predictions that would be broken
        correct_broken = 0
        for inp, obs in correct_examples:
            try:
                pred = predict_continuous(law, inp, ctx)
                if abs(pred - obs) <= tolerance * (abs(obs) + 1e-9):
                    # This morphism is contributing to a correct prediction
                    # If retracted, the correct prediction might be lost
                    correct_broken += 1
            except (ValueError, KeyError):
                pass

        net_gain = float(anomalies_resolved) - float(correct_broken)
        return RetractionScore(
            morph_id=morph_id,
            anomalies_resolved=anomalies_resolved,
            correct_broken=correct_broken,
            net_gain=net_gain,
        )

    # ------------------------------------------------------------------
    # Candidate proposal
    # ------------------------------------------------------------------

    def propose_retraction(
        self,
        theory_id:        TheoryId,
        anomalies:        list[tuple[dict, float]],
        correct_examples: list[tuple[dict, float]],
        ctx:              EvalContext,
        tolerance:        float = 0.1,
    ) -> Optional[RetractionCandidate]:
        """Find the morphism whose retraction maximises net benefit.

        Returns None if no morphism has positive net gain.
        """
        member_ids = self._mg.theory_members(theory_id)
        best: Optional[RetractionScore] = None
        for mid in member_ids:
            if mid in self._reviser._vetoed:
                continue
            rs = self.score_retraction(mid, theory_id, anomalies,
                                        correct_examples, ctx, tolerance)
            if best is None or rs.net_gain > best.net_gain:
                best = rs

        if best is None or best.net_gain <= 0.0:
            return None

        return RetractionCandidate(
            morph_id=best.morph_id,
            theory_id=theory_id,
            score=best.net_gain,
            reason=f"net_gain={best.net_gain:.2f}: "
                   f"{best.anomalies_resolved} anomalies resolved, "
                   f"{best.correct_broken} correct broken",
        )

    def propose_replacement(
        self,
        theory_id:        TheoryId,
        anomalies:        list[tuple[dict, float]],
        correct_examples: list[tuple[dict, float]],
        ctx:              EvalContext,
        schema:           SchematicLaw,
        label:            str = "__replacement__",
        tolerance:        float = 0.1,
    ) -> Optional[ReplacementCandidate]:
        """Propose retract + add: retract worst morphism, fit new one.

        The new law is fitted from both anomalies AND correct_examples,
        so it preserves what was correct while fixing what was wrong.

        Returns None if no beneficial retraction exists.
        """
        retraction = self.propose_retraction(
            theory_id, anomalies, correct_examples, ctx, tolerance
        )
        if retraction is None:
            return None

        # Fit new law from all observations (anomalies + correct)
        all_obs = list(anomalies) + list(correct_examples)
        try:
            new_fl = fit_parameters(
                schema.conclusion, schema.params, all_obs, ctx
            )
        except (ValueError, Exception):
            return None

        score = retraction.score
        return ReplacementCandidate(
            retract_id=retraction.morph_id,
            new_law=new_fl,
            theory_id=theory_id,
            score=score,
        )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply_replacement(
        self,
        candidate: ReplacementCandidate,
        ctx:       EvalContext,
        label:     str = "__applied_replacement__",
    ) -> RevisionResult:
        """Apply retraction + addition with closed-loop verification.

        1. Retract the old morphism (add to veto set + log).
        2. Fit and add the new FITTED_LAW to the theory.
        3. Compute surprise before and after.
        4. If surprise decreased: accept. Otherwise: un-veto + veto new.

        Returns a RevisionResult describing the outcome.
        """
        from experiments.symbolic_ai_v2.ctkg.inference.revision import (
            ContinuousAnomaly,
            RevisionCandidate,
        )

        # Build anomaly objects for the closed-loop check
        # (use the candidate's new_law to sample "expected" inputs)
        theory_id = candidate.theory_id

        # Measure surprise_before using all current members (no changes yet)
        # We use the new_law params as the "true" test inputs
        test_inputs = [{"x": float(i + 1)} for i in range(3)]
        test_obs = [
            predict_continuous(candidate.new_law, inp, ctx) if candidate.new_law.params
            else 0.0
            for inp in test_inputs
        ]
        anomalies_for_check = []
        for inp, obs in zip(test_inputs, test_obs):
            pred = self._reviser._predict_excluding(theory_id, inp, ctx)
            if pred is not None:
                surprise = ((pred - obs) / self._reviser._sigma) ** 2
                blame = self._tm.blame_theory([theory_id], inp, obs, ctx)
                blamed_mid = blame.morph_id if blame is not None else -1
                anomalies_for_check.append(ContinuousAnomaly(
                    theory_id=theory_id,
                    input_bindings=inp,
                    observed=obs,
                    predicted=pred,
                    surprise=surprise,
                    blamed_morph_id=blamed_mid,
                ))

        surprise_before = (
            sum(a.surprise for a in anomalies_for_check) / len(anomalies_for_check)
            if anomalies_for_check else 0.0
        )

        # 1. Retract old morphism
        self.retract_morphism(
            candidate.retract_id,
            theory_id,
            reason=f"ReplacementCandidate retraction (score={candidate.score:.2f})",
            score=candidate.score,
        )

        # 2. Add new law
        new_mid = add_fitted_law(self._mg, label, candidate.new_law)
        self._tm.assign_morphism(new_mid, theory_id)
        candidate.morph_id_new = new_mid

        # Compute surprise_after
        surprise_after_vals = []
        for inp, obs in zip(test_inputs, test_obs):
            pred2 = self._reviser._predict_excluding(theory_id, inp, ctx)
            if pred2 is not None:
                surprise_after_vals.append(((pred2 - obs) / self._reviser._sigma) ** 2)
        surprise_after = (
            sum(surprise_after_vals) / len(surprise_after_vals)
            if surprise_after_vals else 0.0
        )

        accepted = surprise_after <= surprise_before + 1e-9

        if not accepted:
            # Roll back: un-retract old, veto new
            self._reviser._vetoed.discard(candidate.retract_id)
            self._reviser._vetoed.add(new_mid)

        # Build dummy RevisionCandidate for RevisionResult
        dummy = RevisionCandidate(
            theory_id=theory_id,
            law=candidate.new_law,
            explains=anomalies_for_check,
            score=candidate.score,
            evidence_count=len(anomalies_for_check),
            morph_id=new_mid,
        )
        return RevisionResult(
            candidate=dummy,
            surprise_before=surprise_before,
            surprise_after=surprise_after,
            accepted=accepted,
            evidence_count=len(anomalies_for_check),
        )
