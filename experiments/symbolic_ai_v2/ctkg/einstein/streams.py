"""
Einstein Test Observation Streams — Phase 12 of the Einstein Roadmap.

Defines four synthetic observation streams that structurally mirror the
four historical scenarios that led from Newtonian mechanics to General Relativity.

Within the constraint that our current latent-variable infrastructure only
handles linear h∘g compositions (k*x), each scenario is modelled as a
linear process.  The categorical structure (which abduction level is triggered,
whether the old theory is preserved, whether a new cluster is needed) is exact.

Scenario 1 — Newtonian Verification
--------------------------------------
Observations: f(x) = k_N * x   (k_N ≈ 10: "mass × acceleration")
Expected outcome:
  - Theory correctly predicts these observations.
  - Orchestrator should find no anomaly or accept at level 1 easily.
  - Represents: system CONFIRMS the existing theory.

Scenario 2 — Michelson-Morley Null Result
-------------------------------------------
Observations: f(x) = k_MM * x  (k_MM ≈ 0.0001: near-zero "ether drift")
Expected outcome:
  - Theory (Newtonian, k_N=10) predicts 10*x; observations show ≈0.
  - Class-A ledger (Newtonian) has strict preservation.
  - Orchestrator: level 1 (revision) blocked by preservation.
  - Level 2/3/4 must explain the null result with a new concept.
  Expected level: ≥ 2 (ideally 4: paradigm shift to a new "constant c" theory).
  Represents: Michelson-Morley → Special Relativity.

Scenario 3 — Mercury Perihelion Precession
---------------------------------------------
Observations: f(x) = k_Hg * x  (k_Hg = k_N + δ: tiny Newtonian deviation)
Expected outcome:
  - Theory (k_N=10) is slightly off: 10.3 vs 10.0.
  - No strict preservation (ledger empty or loose tolerance).
  - Level 1 revision succeeds: retract k=10 law, add k≈10.3 law.
  Expected level: 1 (small revision accepted).
  Represents: Mercury precession → GR correction term.

Scenario 4 — Maxwell Electromagnetic Anomalies
-------------------------------------------------
Observations: THREE sets from f(x) = k_EM * x  (k_EM ≈ 100: "light speed scale")
Expected outcome:
  - Three independent measurement sets from k=100 process.
  - Multi-anomaly coverage (level 3) finds shared k=100 explanation.
  Expected level: ≥ 2 (latent/coverage finds shared EM law).
  Represents: Maxwell's equations → unification of electricity/magnetism.

EinsteinScenario dataclass
    name             : short identifier.
    description      : one-line description.
    observation_sets : list of anomaly observation sets.
    ledger_examples  : (input, observed) pairs for the preservation ledger.
    expected_min_level: minimum abduction level expected from orchestrator.
    revision_tol     : preservation tolerance for this scenario.
    latent_tol       : latent coverage tolerance for this scenario.

Factory functions (one per scenario)
    All take `nid: int` (an anonymous NodeId) and `ctx: EvalContext` to remain
    symbol-invariant (Iron Law compliance).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from experiments.symbolic_ai_v2.ctkg.core.schematic_law import SchematicLaw
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import var, Expr


# ---------------------------------------------------------------------------
# Scenario descriptor
# ---------------------------------------------------------------------------

@dataclass
class EinsteinScenario:
    """A synthetic observation stream scenario for Einstein-test evaluation.

    Attributes
    ----------
    name              : short identifier (e.g. "michelson_morley").
    description       : one-line description of what this scenario models.
    observation_sets  : K lists of (input_bindings, observed) anomaly pairs.
    ledger_examples   : (input, observed) pairs for the preservation ledger;
                        these represent the "old correct predictions" that
                        must not be broken by revision.
    expected_min_level: minimum orchestrator level expected.
    revision_tol      : relative error tolerance for preservation check.
    latent_tol        : relative error tolerance for latent/coverage scoring.
    schema_g_list     : schema candidates for latent g.
    schema_h          : schema for latent h.
    """
    name:               str
    description:        str
    observation_sets:   list[list[tuple[dict, float]]]
    ledger_examples:    list[tuple[dict, float]]
    expected_min_level: int
    revision_tol:       float
    latent_tol:         float
    schema_g_list:      list[SchematicLaw]
    schema_h:           SchematicLaw


# ---------------------------------------------------------------------------
# Helper: build schemas
# ---------------------------------------------------------------------------

def _make_schema_g(nid: int) -> SchematicLaw:
    formula = Expr(head=nid, args=(var("k"), var("x")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["k"]), variables=frozenset(["x"]), evidence=1,
    )


def _make_schema_h(nid: int) -> SchematicLaw:
    formula = Expr(head=nid, args=(var("a"), var("z")))
    return SchematicLaw(
        pattern=formula, conclusion=formula,
        params=frozenset(["a"]), variables=frozenset(["z"]), evidence=1,
    )


def _obs(k: float, n: int = 5) -> list[tuple[dict, float]]:
    return [({  "x": float(i + 1)}, k * (i + 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Scenario factories
# ---------------------------------------------------------------------------

def newtonian_scenario(nid: int) -> EinsteinScenario:
    """Scenario 1: Newtonian verification — small revision or no anomaly.

    Observation stream: f(x) = 10.2*x  (slight drift from k=10 Newtonian law)
    Theory: k=10 morphism.
    Ledger: empty (no strict preservation constraint).
    Expected: level 1 (revision accepted with loose tolerance).
    """
    return EinsteinScenario(
        name="newtonian",
        description="Newtonian verification: slight drift in F=ma",
        observation_sets=[_obs(10.2, n=6), _obs(10.2, n=4)],
        ledger_examples=[],
        expected_min_level=1,
        revision_tol=0.5,    # loose: small k deviation is OK
        latent_tol=0.15,
        schema_g_list=[_make_schema_g(nid)],
        schema_h=_make_schema_h(nid),
    )


def michelson_morley_scenario(nid: int) -> EinsteinScenario:
    """Scenario 2: Michelson-Morley null result → paradigm shift.

    Observation stream: f(x) ≈ 0.0001 * x  (near-zero ether drift)
    Theory: Newtonian k=10 morphism.
    Ledger: 4 class-A observations (k=10, strict preservation).
    Expected: level ≥ 2 (revision blocked by preservation, escalate).
    """
    return EinsteinScenario(
        name="michelson_morley",
        description="MM null result: near-zero ether drift vs k=10 Newtonian",
        observation_sets=[
            _obs(0.0001, n=5),   # near-zero "ether drift"
            _obs(0.0001, n=5),
            _obs(0.0001, n=5),
        ],
        ledger_examples=[
            ({"x": float(i + 1)}, 10.0 * (i + 1)) for i in range(4)
        ],
        expected_min_level=2,
        revision_tol=0.05,   # strict: k=0.0001 must NOT replace k=10
        latent_tol=0.20,
        schema_g_list=[_make_schema_g(nid)],
        schema_h=_make_schema_h(nid),
    )


def mercury_precession_scenario(nid: int) -> EinsteinScenario:
    """Scenario 3: Mercury perihelion precession → small revision.

    Observation stream: f(x) = 10.3*x  (tiny GR correction over k=10)
    Theory: k=10 Newtonian morphism.
    Ledger: empty (no preservation constraint).
    Expected: level 1 (small revision accepted).
    """
    return EinsteinScenario(
        name="mercury_precession",
        description="Mercury precession: k=10.3 GR correction over k=10 Newton",
        observation_sets=[_obs(10.3, n=6)],
        ledger_examples=[],
        expected_min_level=1,
        revision_tol=0.5,    # loose
        latent_tol=0.15,
        schema_g_list=[_make_schema_g(nid)],
        schema_h=_make_schema_h(nid),
    )


def maxwell_em_scenario(nid: int) -> EinsteinScenario:
    """Scenario 4: Maxwell EM unification → multi-anomaly coverage.

    Three independent anomaly sets from k=100 (electromagnetic scale).
    Theory: Newtonian k=10 morphism.
    Ledger: class-A Newtonian observations (strict preservation).
    Expected: level ≥ 2 (revision blocked by preservation; latent/coverage
    unifies the three EM anomaly sets at level 2 or 3).
    """
    return EinsteinScenario(
        name="maxwell_em",
        description="Maxwell EM: 3 independent k=100 field observation sets",
        observation_sets=[_obs(100.0, n=5), _obs(100.0, n=5), _obs(100.0, n=5)],
        ledger_examples=[
            ({"x": float(i + 1)}, 10.0 * (i + 1)) for i in range(4)
        ],
        expected_min_level=2,
        revision_tol=0.05,   # strict: k=100 revision must not break k=10 class-A
        latent_tol=0.10,
        schema_g_list=[_make_schema_g(nid)],
        schema_h=_make_schema_h(nid),
    )


def all_scenarios(nid: int) -> list[EinsteinScenario]:
    """Return all four Einstein test scenarios."""
    return [
        newtonian_scenario(nid),
        michelson_morley_scenario(nid),
        mercury_precession_scenario(nid),
        maxwell_em_scenario(nid),
    ]
