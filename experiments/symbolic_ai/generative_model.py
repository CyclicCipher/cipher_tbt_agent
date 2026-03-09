"""generative_model.py — Active Inference Generative Model.

Architecture
============
The generative model is the agent's internal model of the world.  It has three
components, following the standard Active Inference formulation (Friston 2010):

  P(o)          Prior preference distribution.  Encodes what observations the
                agent *expects* when all drives are satisfied.  High log P(o)
                = preferred; low log P(o) = aversive or surprising.

  P(s'|s, a)    Transition model.  How actions change hidden world states.
                Learned from observed (prev_state, action, next_state) triples.

  P(o|s)        Likelihood model.  How hidden states produce observations.
                In the current implementation: identity (o ≈ s), because the
                state dict IS the observation.  Will be extended for visual
                observations (Phase R4+).

Expected Free Energy
====================
For a policy π = [a₁, a₂, ..., aₖ], the Expected Free Energy G(π) is::

    G(π) = Σ_τ [ pragmatic_value(o_τ) + epistemic_value(s_τ, a_τ) ]

where the sum is over future time steps τ under the policy.

Minimising G(π) trades off:
  - Pragmatic value:  -log P(predicted_obs) — prefer observations matching priors.
  - Epistemic value:  KL[q(s|o) || q(s)] — prefer actions that reduce uncertainty.

In the reactive limit (horizon=1, no epistemic term), this recovers the
priority-weighted goal selection of DecisionEngine.  With horizon>1 and the
epistemic term, the agent explores intelligently and plans ahead.

Drive → PreferenceFactor conversion
=====================================
The existing Drive class (planning.py) encodes the same information as a
PreferenceFactor, in a different form::

    Drive(name, measure, setpoint=1.0, urgency=0.8)
    ↔
    PreferenceFactor(name, urgency=0.8,
                     log_pref_fn = lambda s: -(setpoint - measure(s)) × scale)

Conversion is exact: Drive.deficit(s) == PreferenceFactor.prediction_error(s)/urgency.
Use GenerativeModel.from_drives(drives) to convert a Drive list automatically.

Design principle
================
**THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST BE GENERAL.**

This file contains zero game-specific logic.  Preferences are provided by the
caller (adapter layer).  The engine works on any state dict.

See AIF_ROADMAP.md for the full Phase R implementation plan.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PreferenceFactor  (replaces Drive in the AIF generative model)
# ---------------------------------------------------------------------------

@dataclass
class PreferenceFactor:
    """One factor in the agent's prior preference distribution P(o).

    Maps observations to log-preference density.  High log_pref → preferred
    observation (agent "expects" this when satisfied).  Low log_pref →
    surprising or aversive observation.

    Replacing Drive
    ---------------
    A Drive encodes "how far am I from my set-point?".  A PreferenceFactor
    encodes the same information as a log-probability: log P(o) is high when
    the drive is satisfied, low when it is depleted.  The two are interchangeable
    via ``PreferenceFactor.from_drive()``.

    The AIF advantage: log P(o) composes correctly across factors via addition
    (independent factors multiply in probability space → add in log space).
    Drive deficits do not compose this cleanly.

    Parameters
    ----------
    name         Human-readable label.
    urgency      Weight of this factor in the total log preference.
                 Corresponds to Drive.urgency.
    log_pref_fn  Callable ``(state: dict) -> float`` returning the unnormalised
                 log preference density.  Should return 0.0 when the factor is
                 fully satisfied; return large negative when violated.
    """

    name:        str
    urgency:     float
    log_pref_fn: Callable[[dict], float]

    def log_preference(self, state: dict) -> float:
        """Return urgency-weighted log preference for this observation."""
        try:
            raw = float(self.log_pref_fn(state))
        except Exception:
            raw = 0.0
        return self.urgency * raw

    def prediction_error(self, state: dict) -> float:
        """Prediction error = -log P(o).  High → observation is not preferred."""
        return -self.log_preference(state)

    @classmethod
    def from_drive(
        cls,
        name:     str,
        measure:  Callable[[dict], float],
        setpoint: float = 1.0,
        urgency:  float = 1.0,
        scale:    float = 5.0,
    ) -> 'PreferenceFactor':
        """Convert a Drive(name, measure, setpoint, urgency) to a PreferenceFactor.

        The log_pref_fn is constructed so that:
          log_pref = 0              when measure(s) >= setpoint (satisfied)
          log_pref = -scale × deficit  when measure(s) < setpoint (unsatisfied)

        With scale=5.0 and urgency=0.8:
          prediction_error ≈ 0.8 × 5 × deficit = 4 × deficit
        which creates a strong gradient toward satisfaction.

        Parameters
        ----------
        scale   How steeply log preference falls per unit of deficit.
                Higher = sharper preference boundary.
        """
        def _log_pref(s: dict) -> float:
            try:
                current = float(measure(s))
            except Exception:
                current = 0.0
            deficit = max(0.0, setpoint - current)
            return -scale * deficit  # 0 when satisfied; negative when depleted

        return cls(name=name, urgency=urgency, log_pref_fn=_log_pref)


# ---------------------------------------------------------------------------
# TransitionModel  (learned P(s'|s, a))
# ---------------------------------------------------------------------------

class TransitionModel:
    """Learned transition model: P(s' | s, a).

    Stores observed (prev_state, action) → next_state transitions and uses
    them to predict future states for policy evaluation.

    Representation
    --------------
    A table keyed by (location, action) → list of observed state deltas.
    For each key, the model records what changed between prev_state and
    next_state.  Prediction returns the most-frequently-observed delta
    (mode) along with a confidence score based on sample count.

    Confidence
    ----------
    confidence(loc, action) = min(1.0, n_observations / saturation)
    where saturation = 5 (default).  Confidence saturates after 5 observations.

    Low confidence → the epistemic value of this action is high (exploring it
    will reduce uncertainty).

    Phase R3 limitation
    -------------------
    This is a tabular transition model: it only generalises within observed
    (location, action) pairs.  Phase R8 will replace this with a parametric
    model that generalises across similar states and actions.
    """

    def __init__(
        self,
        saturation:  int                           = 5,
        context_fn:  Optional[Callable[[dict], str]] = None,
    ) -> None:
        # (context, action) → list of observed deltas {key: new_value}
        self._deltas: Dict[Tuple[str, str], List[Dict[str, Any]]] = (
            collections.defaultdict(list)
        )
        # (context, action) → observation count
        self._counts: collections.Counter = collections.Counter()
        self._saturation = saturation
        # Domain adapter: extracts context identifier from state dict.
        # Default: '' (all transitions share one context, suitable for
        # non-navigational domains or single-room games).
        self._context_fn: Callable[[dict], str] = context_fn or (lambda s: '')

    # ------------------------------------------------------------------

    def update(
        self,
        prev_state: dict,
        action:     str,
        next_state: dict,
        reward:     float = 0.0,
    ) -> None:
        """Record an observed transition (prev_state, action) → next_state.

        Parameters
        ----------
        prev_state   State dict before the action.
        action       String action (or action identifier).
        next_state   State dict after the action.
        reward       Optional scalar reward signal (for future model improvements).
        """
        try:
            loc = str(self._context_fn(prev_state))
        except Exception:
            loc = ''
        key   = (loc, action)
        delta = self._delta(prev_state, next_state)
        self._deltas[key].append(delta)
        self._counts[key] += 1

    @staticmethod
    def _delta(prev: dict, nxt: dict) -> Dict[str, Any]:
        """Compute the diff dict: keys whose values changed."""
        result: Dict[str, Any] = {}
        all_keys = set(prev) | set(nxt)
        for k in all_keys:
            pv = prev.get(k)
            nv = nxt.get(k)
            # Compare only simple types; skip large structures for speed.
            if isinstance(pv, (str, int, float, bool)) or pv is None:
                if pv != nv:
                    result[k] = nv
            elif isinstance(pv, list) and isinstance(nv, list):
                if set(pv) != set(nv):
                    result[k] = nv
            elif isinstance(pv, set) and isinstance(nv, set):
                if pv != nv:
                    result[k] = nv
        return result

    def predict(
        self,
        state:  dict,
        action: str,
    ) -> Tuple[dict, float]:
        """Predict next state and confidence given (state, action).

        Returns
        -------
        predicted_state  A copy of state with the most-common delta applied.
        confidence       float in [0, 1].  0 = never seen; 1 = saturated.

        If the (location, action) pair has never been observed, returns
        (state unchanged, confidence=0.0), indicating maximum uncertainty.
        """
        try:
            loc = str(self._context_fn(state))
        except Exception:
            loc = ''
        key = (loc, action)
        n   = self._counts[key]

        if n == 0:
            return dict(state), 0.0

        confidence = min(1.0, n / self._saturation)

        # Mode delta: the delta that occurred most frequently.
        # For now: use the most recent observation (simple, adequate for Phase R3).
        # Phase R8 will use the mode / weighted average.
        mode_delta = self._deltas[key][-1]

        predicted = dict(state)
        predicted.update(mode_delta)
        return predicted, confidence

    def confidence(self, state: dict, action: str) -> float:
        """Return confidence in our model of (context, action) ∈ [0, 1]."""
        try:
            loc = str(self._context_fn(state))
        except Exception:
            loc = ''
        return min(1.0, self._counts[(loc, action)] / self._saturation)

    def is_novel(self, state: dict, action: str) -> bool:
        """True if we have never observed this (context, action) pair."""
        try:
            loc = str(self._context_fn(state))
        except Exception:
            loc = ''
        return self._counts[(loc, action)] == 0


# ---------------------------------------------------------------------------
# GenerativeModel
# ---------------------------------------------------------------------------

class GenerativeModel:
    """The agent's internal model of the world for Active Inference.

    Combines:
      preferences  — P(o): prior expectations over preferred observations.
      transition   — P(s'|s,a): learned dynamics model.

    Usage in the AIFEngine loop::

        # Build from Drive list (backward-compatible):
        model = GenerativeModel.from_drives(my_drives)

        # Or directly:
        model = GenerativeModel([
            PreferenceFactor('hunger', urgency=0.8,
                             log_pref_fn=lambda s: -5*max(0,1-s.get('food',0))),
        ])

        # Evaluate a policy:
        G = model.expected_free_energy(['go north', 'take apple'], current_state)
        # Lower G → preferred policy

        # Update from observed transition:
        model.transition.update(prev_state, action, next_state)

    Parameters
    ----------
    preferences  List of PreferenceFactor objects.  Can be empty (→ epistemic
                 agent only; explores without preferences).
    """

    def __init__(self, preferences: Optional[List[PreferenceFactor]] = None) -> None:
        self.preferences: List[PreferenceFactor] = preferences or []
        self.transition:  TransitionModel        = TransitionModel()

    # ------------------------------------------------------------------
    # Factory

    @classmethod
    def from_drives(
        cls,
        drives: List[Any],   # List[Drive] from planning.py
        scale:  float = 5.0,
    ) -> 'GenerativeModel':
        """Convert a list of Drive objects to a GenerativeModel.

        Each Drive becomes a PreferenceFactor with equivalent semantics.
        ``Drive.deficit(s)`` maps to ``PreferenceFactor.prediction_error(s)``.

        Parameters
        ----------
        drives  List of planning.py Drive instances.
        scale   Log-preference slope per unit deficit.  Default 5.0.
        """
        factors = []
        for d in drives:
            # Drive has: name, measure, setpoint, urgency
            f = PreferenceFactor.from_drive(
                name     = d.name,
                measure  = d.measure,
                setpoint = d.setpoint,
                urgency  = d.urgency,
                scale    = scale,
            )
            factors.append(f)
        return cls(preferences=factors)

    # ------------------------------------------------------------------
    # Core quantities

    def log_preference(self, state: dict) -> float:
        """Total log preference for this observation: Σ_i log P_i(o).

        High (close to 0) → all drives satisfied.
        Low (large negative) → one or more drives severely depleted.
        """
        return sum(p.log_preference(state) for p in self.preferences)

    def pragmatic_value(self, state: dict) -> float:
        """Pragmatic value of a state: -log P(o).

        High → observation is NOT preferred (penalty).
        Low  → observation is preferred (reward).

        In the AIF framework, agents minimise G(π) which includes pragmatic_value.
        So lower pragmatic_value states are actively sought.
        """
        return -self.log_preference(state)

    def epistemic_value(
        self,
        state:  dict,
        action: str,
        weight: float = 0.5,
    ) -> float:
        """Epistemic value (information gain) estimate for (state, action).

        Rewards actions that reduce uncertainty about the world.  Novel actions
        (never observed) have maximum epistemic value.  Well-known actions have
        low epistemic value (we already know what will happen).

        In the AIF framework, epistemic value reduces G(π), making exploratory
        actions preferred when pragmatic values are similar.

        Returns a value in [0, weight].  0 = fully known; weight = completely novel.

        Parameters
        ----------
        weight   Maximum epistemic bonus.  Default 0.5.  Tune to balance
                 exploration vs. exploitation (higher = more explorative).
        """
        conf = self.transition.confidence(state, action)
        # Information gain ∝ 1 - confidence.
        # Novel action (conf=0): full epistemic bonus → subtract from G.
        # Known action (conf=1): zero epistemic bonus.
        info_gain = (1.0 - conf) * weight
        return -info_gain  # subtract from G (lower G = preferred)

    def expected_free_energy(
        self,
        policy:  List[str],
        state:   dict,
        horizon: int = 1,
    ) -> float:
        """Expected Free Energy G(π) for a policy (sequence of actions).

        G(π) = Σ_τ [ pragmatic_value(o_τ) + epistemic_value(s_τ, a_τ) ]

        Lower G → policy is preferred.  The agent selects argmin_π G(π).

        Current implementation (Phase R3): single-step look-ahead (horizon=1).
        Multi-step rollouts are implemented in Phase R8.

        Parameters
        ----------
        policy   List of action strings.
        state    Current state dict.
        horizon  Maximum look-ahead depth.  Phase R3: always 1.

        Returns
        -------
        G  Float.  Lower is better.  The action selection rule is argmin G.
        """
        if not policy:
            return float('inf')

        G = 0.0
        current = dict(state)

        for action in policy[:max(1, horizon)]:
            # Predict next state under this action.
            predicted, conf = self.transition.predict(current, action)

            # Pragmatic term: how much do we prefer the predicted outcome?
            G += self.pragmatic_value(predicted)

            # Epistemic term: how much uncertainty does this action resolve?
            G += self.epistemic_value(current, action)

            current = predicted

        return G

    # ------------------------------------------------------------------
    # Inspection

    def preference_breakdown(self, state: dict) -> List[Tuple[str, float]]:
        """Return per-factor (name, prediction_error) for debugging."""
        return [
            (p.name, p.prediction_error(state))
            for p in self.preferences
        ]

    def total_prediction_error(self, state: dict) -> float:
        """Sum of all factor prediction errors = -log P(o).  Alias for pragmatic_value."""
        return self.pragmatic_value(state)

    def __repr__(self) -> str:
        names = [p.name for p in self.preferences]
        return (
            f'GenerativeModel(preferences={names}, '
            f'n_transitions={sum(self.transition._counts.values())})'
        )


# ---------------------------------------------------------------------------
# Phase R8: Beam search over multi-step policies
# ---------------------------------------------------------------------------

def beam_search(
    state:       dict,
    model:       'GenerativeModel',
    candidates:  List[str],
    horizon:     int = 3,
    beam_width:  int = 5,
) -> Tuple[List[str], float]:
    """Find the best k-step policy by beam search over Expected Free Energy.

    Implements the Phase R8 multi-step policy rollout.  Each policy is a
    sequence of actions evaluated by cumulative G(π) = Σ_τ G(a_τ, s_τ).
    The argmin policy is returned together with its G value.

    Parameters
    ----------
    state       Current state dict (starting point for rollout).
    model       ``GenerativeModel`` providing pragmatic + epistemic values
                and the transition model for state prediction.
    candidates  List of candidate action strings to consider at each step.
                Typically ``state['admissible']`` or the engine's default set.
    horizon     Number of steps to look ahead.  Default 3.
    beam_width  Maximum number of candidate policies to keep at each step.
                Default 5.  Larger values → better quality, higher cost.
                Complexity: O(beam_width × |candidates| × horizon).

    Returns
    -------
    (best_policy, best_G)
        ``best_policy`` — list of action strings (length ≤ ``horizon``).
        ``best_G``      — cumulative Expected Free Energy of that policy.

    Notes
    -----
    When ``candidates`` is empty, returns ([], inf).

    When ``model.transition`` has no observations for a (state, action) pair,
    ``predict()`` returns (state, 0.0) — low confidence — so the epistemic
    bonus fully applies (novel actions are preferred during exploration).

    The first element of ``best_policy`` is the action to execute now.
    Subsequent elements represent the intended future actions, but they are
    NOT executed yet — they are re-evaluated each step as new observations
    arrive.  This is Model Predictive Control (MPC) style execution.
    """
    if not candidates or horizon < 1:
        return [], float('inf')

    # Beam: list of (cumulative_G, policy_so_far, predicted_state)
    beams: List[Tuple[float, List[str], dict]] = [(0.0, [], dict(state))]

    for _step in range(horizon):
        next_beams: List[Tuple[float, List[str], dict]] = []

        for cum_G, policy, current in beams:
            for action in candidates:
                # Predict next state under this action.
                predicted, _conf = model.transition.predict(current, action)

                # Accumulate G for this step.
                step_G = (
                    model.pragmatic_value(predicted)
                    + model.epistemic_value(current, action)
                )
                next_beams.append((cum_G + step_G, policy + [action], predicted))

        if not next_beams:
            break

        # Keep the top-beam_width policies by cumulative G (lower = better).
        next_beams.sort(key=lambda b: b[0])
        beams = next_beams[:beam_width]

    if not beams:
        return [], float('inf')

    best_G, best_policy, _ = beams[0]
    return best_policy, best_G
