"""
Abduction Orchestrator — Phase 11 of the Einstein Roadmap.

Chains all A-track phases (4–10) into a single decision procedure.
Given an anomaly stream and a current theory, the orchestrator runs a
cascading sequence of abductive strategies from simplest to most radical:

  Level 1 — Theory revision (Phases 5 + 6)
      Retract the offending morphism; replace with a new fitted law.
      Blocked by preservation check (Phase 9) if it breaks correct examples.

  Level 2 — Latent variable abduction (Phase 7)
      Hypothesise a hidden intermediate quantity f = h ∘ g that explains
      the anomaly.  Preferred if it reduces residual below threshold.

  Level 3 — Multi-anomaly coverage abduction (Phase 8)
      When multiple anomaly sets share a common generating process, find
      the single latent hypothesis that covers the most.

  Level 4 — Paradigm shift (Phase 10)
      If levels 1–3 all fail (revision blocked by preservation, latent
      abduction coverage too low), propose a new theory cluster.

Decision
--------
The orchestrator tries each level in order, stopping at the first that
succeeds (returns a non-None result with coverage ≥ min_coverage).

AbductionDecision records:
  - which level was reached
  - the result at each level (for audit/explanation)
  - the active theory after abduction (old theory if shift happened, or
    new theory id if paradigm shift occurred)

Einstein test mapping
---------------------
Michelson-Morley scenario:
  Level 1: retract "ether drift" morphism — blocked (preservation: Newtonian
           mechanics must still explain terrestrial observations).
  Level 2: try latent variable — does not reduce surprise sufficiently.
  Level 3: multiple anomalies (multiple orientations) — same null result.
           Coverage of "null result" hypothesis = 1.0, but the explanation
           requires a NEW concept: constant light speed.
  Level 4: paradigm shift — create "Special Relativity" theory cluster.

Mercury precession scenario:
  Level 1: retract Newtonian gravity morphism — blocked (preservation:
           must explain Earth/Venus orbits).
  Level 2: latent variable — add GR correction term as hidden quantity.
           This IS the correct level for Mercury: GR adds a 1/r³ correction
           to the 1/r² gravity law, which is a latent quantity.

Iron Law compliance
-------------------
No token names used.  All routing uses morph_type strings and ObjectId ints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    MorphId,
    ObjectId,
)
from experiments.symbolic_ai_v2.ctkg.core.quantity import EvalContext
from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.inference.theory import TheoryManager, TheoryId
from experiments.symbolic_ai_v2.ctkg.inference.revision import (
    ClosedLoopReviser,
    RevisionResult,
)
from experiments.symbolic_ai_v2.ctkg.inference.retract import RetractEngine
from experiments.symbolic_ai_v2.ctkg.inference.latent import (
    LatentHypothesis,
    hypothesise_latent_mdl_select,
)
from experiments.symbolic_ai_v2.ctkg.inference.coverage import (
    multi_anomaly_abduction,
    score_coverage,
)
from experiments.symbolic_ai_v2.ctkg.inference.preservation import (
    PredictionLedger,
    propose_and_apply_safe,
)
from experiments.symbolic_ai_v2.ctkg.inference.paradigm import (
    ParadigmShiftResult,
    propose_paradigm_shift,
)

_LEVEL_NAMES = {
    0: "no_action",
    1: "revision",
    2: "latent",
    3: "coverage",
    4: "paradigm_shift",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AbductionDecision:
    """The result of running the full A-track abduction pipeline.

    Attributes
    ----------
    level_reached    : which level succeeded (0 = no anomaly or nothing worked).
                       1=revision, 2=latent, 3=coverage, 4=paradigm_shift.
    level_name       : human-readable name for level_reached.
    anomaly_count    : number of anomaly observations processed.
    revision_result  : RevisionResult from level 1, or None.
    latent_hyp       : LatentHypothesis from level 2, or None.
    coverage_hyp     : LatentHypothesis from level 3, or None.
    paradigm_shift   : ParadigmShiftResult from level 4, or None.
    active_theory_id : the theory that now accounts for the observations.
                       May differ from input theory_id after a paradigm shift.
    success          : True if any level succeeded (level_reached > 0).
    """
    level_reached:    int
    level_name:       str
    anomaly_count:    int
    revision_result:  Optional[RevisionResult]
    latent_hyp:       Optional[LatentHypothesis]
    coverage_hyp:     Optional[LatentHypothesis]
    paradigm_shift:   Optional[ParadigmShiftResult]
    active_theory_id: TheoryId
    success:          bool


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AbductionOrchestrator:
    """Chains all A-track phases into a single abductive decision procedure.

    Usage
    -----
    orch = AbductionOrchestrator(mg, tm, reviser, eng, ledger, ctx)
    decision = orch.run(
        theory_id     = newton_theory_id,
        anomaly_sets  = [[obs1, obs2, ...], [obs3, obs4, ...]],
        schema_g_list = [schema_g1, schema_g2],
        schema_h      = schema_h,
    )

    Parameters (constructor)
    -----------------------
    mg, tm        : MorphismGraph and TheoryManager.
    reviser       : ClosedLoopReviser (Phase 5 — must be pre-constructed).
    eng           : RetractEngine (Phase 6 — must be pre-constructed).
    ledger        : PredictionLedger (Phase 9 — carries prior correct examples).
    ctx           : EvalContext.

    Parameters (run)
    ----------------
    theory_id        : the theory to revise / extend.
    anomaly_sets     : list of anomaly observation sets.
    schema_g_list    : schema candidates for latent g.
    schema_h         : schema for latent h.
    schema_flat      : Optional[SchematicLaw] — flat schema for level-1 revision.
                       If None, level 1 is skipped.
    min_coverage     : minimum coverage for level 3/4 to succeed.
    revision_tol     : relative error tolerance for preservation check.
    latent_tol       : relative error tolerance for latent coverage scoring.
    new_theory_name  : name for the new theory in case of paradigm shift.
    label_prefix     : prefix for all morphism labels.
    """

    def __init__(
        self,
        mg:      MorphismGraph,
        tm:      TheoryManager,
        reviser: ClosedLoopReviser,
        eng:     RetractEngine,
        ledger:  PredictionLedger,
        ctx:     EvalContext,
    ) -> None:
        self._mg      = mg
        self._tm      = tm
        self._reviser = reviser
        self._eng     = eng
        self._ledger  = ledger
        self._ctx     = ctx

    # ------------------------------------------------------------------

    def run(
        self,
        theory_id:       TheoryId,
        anomaly_sets:    list[list[tuple[dict, float]]],
        schema_g_list:   list[SchematicLaw],
        schema_h:        SchematicLaw,
        schema_flat:     Optional[SchematicLaw] = None,
        min_coverage:    float = 0.5,
        revision_tol:    float = 0.05,
        latent_tol:      float = 0.10,
        new_theory_name: str   = "__paradigm__",
        label_prefix:    str   = "__orch__",
    ) -> AbductionDecision:
        """Run the cascading abduction pipeline.

        Returns an AbductionDecision at the first level that succeeds,
        or a level_reached=0 decision if nothing works.
        """
        combined = [obs for obs_set in anomaly_sets for obs in obs_set]
        n_anomaly = len(combined)

        # Defaults for the decision
        rev_result:    Optional[RevisionResult]      = None
        latent_hyp:    Optional[LatentHypothesis]    = None
        coverage_hyp:  Optional[LatentHypothesis]    = None
        pshift:        Optional[ParadigmShiftResult] = None
        active_theory: TheoryId = theory_id

        # ---- Level 1: Revision (Phases 5+6+9) ---------------------------
        if schema_flat is not None and combined:
            rev_result = propose_and_apply_safe(
                self._eng, self._reviser, self._tm, self._mg,
                self._ledger, theory_id, combined, self._ctx, schema_flat,
                label=f"{label_prefix}_L1",
                tolerance=revision_tol,
            )
            if rev_result is not None:
                return AbductionDecision(
                    level_reached=1,
                    level_name=_LEVEL_NAMES[1],
                    anomaly_count=n_anomaly,
                    revision_result=rev_result,
                    latent_hyp=None,
                    coverage_hyp=None,
                    paradigm_shift=None,
                    active_theory_id=theory_id,
                    success=True,
                )

        # ---- Level 2: Latent abduction (Phase 7) -------------------------
        if combined and schema_g_list:
            latent_hyp = hypothesise_latent_mdl_select(
                combined, schema_g_list, schema_h, self._ctx,
                self._mg, self._tm, theory_id,
                label_prefix=f"{label_prefix}_L2",
            )
            if latent_hyp is not None:
                cov = score_coverage(
                    latent_hyp, anomaly_sets, self._ctx, tolerance=latent_tol,
                )
                if cov.coverage >= min_coverage:
                    return AbductionDecision(
                        level_reached=2,
                        level_name=_LEVEL_NAMES[2],
                        anomaly_count=n_anomaly,
                        revision_result=None,
                        latent_hyp=latent_hyp,
                        coverage_hyp=None,
                        paradigm_shift=None,
                        active_theory_id=theory_id,
                        success=True,
                    )

        # ---- Level 3: Multi-anomaly coverage (Phase 8) ------------------
        if len(anomaly_sets) >= 2 and schema_g_list:
            coverage_hyp = multi_anomaly_abduction(
                anomaly_sets, schema_g_list, schema_h, self._ctx,
                self._mg, self._tm, theory_id,
                tolerance=latent_tol,
                label_prefix=f"{label_prefix}_L3",
            )
            if coverage_hyp is not None:
                cov = score_coverage(
                    coverage_hyp, anomaly_sets, self._ctx, tolerance=latent_tol,
                )
                if cov.coverage >= min_coverage:
                    return AbductionDecision(
                        level_reached=3,
                        level_name=_LEVEL_NAMES[3],
                        anomaly_count=n_anomaly,
                        revision_result=None,
                        latent_hyp=None,
                        coverage_hyp=coverage_hyp,
                        paradigm_shift=None,
                        active_theory_id=theory_id,
                        success=True,
                    )

        # ---- Level 4: Paradigm shift (Phase 10) -------------------------
        if anomaly_sets and schema_g_list:
            pshift = propose_paradigm_shift(
                theory_id, anomaly_sets, schema_g_list, schema_h, self._ctx,
                self._mg, self._tm,
                new_theory_name=new_theory_name,
                label_prefix=f"{label_prefix}_L4",
                min_coverage=min_coverage,
                tolerance=latent_tol,
            )
            if pshift is not None:
                return AbductionDecision(
                    level_reached=4,
                    level_name=_LEVEL_NAMES[4],
                    anomaly_count=n_anomaly,
                    revision_result=None,
                    latent_hyp=None,
                    coverage_hyp=None,
                    paradigm_shift=pshift,
                    active_theory_id=pshift.new_theory_id,
                    success=True,
                )

        # Nothing worked
        return AbductionDecision(
            level_reached=0,
            level_name=_LEVEL_NAMES[0],
            anomaly_count=n_anomaly,
            revision_result=None,
            latent_hyp=None,
            coverage_hyp=None,
            paradigm_shift=None,
            active_theory_id=theory_id,
            success=False,
        )
