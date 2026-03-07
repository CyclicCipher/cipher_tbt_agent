"""perception.py — Variational Bayesian belief state for Active Inference.

Architecture
============
In Active Inference, perception is variational inference:

    q(s) ← argmin_q F(q, o)

where F is the variational free energy and q(s) is the agent's approximate
posterior over hidden world states given all observations so far.

In the brain, this corresponds to the predictive coding hierarchy: higher
layers predict activity in lower layers; prediction errors propagate upward
to update the higher-layer representation.

Implementation
==============
We use a **factored (mean-field) approximation**::

    q(s) ≈ ∏_i q_i(s_i)

where each factor q_i is a categorical distribution over variable s_i's domain.
This is exact when variables are conditionally independent given observations,
and approximate otherwise.  It is tractable and interpretable.

For each state variable (e.g., 'location', 'inv:apple', 'score'), the belief
is a probability distribution over possible values:

    q('location') = {'kitchen': 0.9, 'living_room': 0.1}
    q('inv:apple') = {True: 0.97, False: 0.03}

Observations sharply update the relevant factor; temporal decay spreads
probability toward the uniform prior when the variable is not recently observed.

Relationship to existing BeliefState
======================================
``BeliefState`` (planning.py) tracks only P(item | location).
``VariationalBelief`` is strictly more general: it tracks any state variable.
``BeliefState`` is a special case where variables are 'item_loc:<item>'.

Both classes coexist.  The AIFEngine uses VariationalBelief.  DecisionEngine
continues to use BeliefState.  They can be run in parallel during the transition
period (Phase R3).

Design principle
================
**THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST BE GENERAL.**

This file contains zero game-specific logic.  Variable names, value sets, and
update patterns are all provided by the caller at construction or update time.

See AIF_ROADMAP.md for the full Phase R implementation plan.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# VariationalBelief
# ---------------------------------------------------------------------------

class VariationalBelief:
    """Factored Bayesian belief state: q(s) = ∏_i q_i(s_i).

    Each factor q_i is a categorical distribution over variable s_i's domain.
    Beliefs are updated by sharp Bayesian observation updates and decay toward
    the uniform prior over time (temporal uncertainty).

    Parameters
    ----------
    variables
        Optional initial specification:
        ``{variable_name: [possible_value_1, possible_value_2, ...]}``
        New variables are created automatically on first observation.
    decay
        Per-step decay rate toward uniform prior.  0.95 = gentle decay
        (beliefs persist for ~19 steps before approaching uniform).
        0.50 = fast decay (half-life of 1 step).
    observation_sharpness
        How strongly a direct observation updates the belief.
        0.97 = place 97% mass on observed value; 3% residual.
        1.00 = deterministic (certainty).

    Usage
    -----
    ::

        belief = VariationalBelief()

        # Observe: agent is in 'kitchen' (direct observation)
        belief.observe('location', 'kitchen')

        # Observe: apple is in inventory
        belief.observe('inv:apple', True)

        # Bulk update from obs dict (all observed values)
        belief.update_from_obs({'location': 'kitchen', 'score': 2})

        # Step: decay beliefs one time step
        belief.decay()

        # Query: what location does the agent believe it's in?
        loc, prob = belief.most_likely('location')

        # Prediction error: how surprising is this observation?
        err = belief.prediction_error('location', 'kitchen')
    """

    def __init__(
        self,
        variables:              Optional[Dict[str, List[Any]]] = None,
        decay:                  float = 0.95,
        observation_sharpness:  float = 0.97,
    ) -> None:
        # {variable: {value: probability}}
        self._beliefs: Dict[str, Dict[Any, float]] = {}
        # {variable: [known_values]} — known domain (grows on first observation)
        self._domains: Dict[str, List[Any]] = {}
        self._decay  = decay
        self._sharp  = observation_sharpness

        # Pre-initialise provided variables with uniform priors.
        if variables:
            for var, values in variables.items():
                self._init_variable(var, values)

    # ------------------------------------------------------------------
    # Internal helpers

    def _init_variable(self, var: str, values: List[Any]) -> None:
        """Initialise a variable with uniform prior over its domain."""
        n = max(len(values), 1)
        self._domains[var]  = list(values)
        self._beliefs[var]  = {v: 1.0 / n for v in values}

    def _ensure(self, var: str, value: Any) -> None:
        """Ensure a variable exists; add value to its domain if new."""
        if var not in self._beliefs:
            # Brand new variable: two-value domain [value, _unknown]
            self._init_variable(var, [value, '_unknown'])
        elif value not in self._beliefs[var]:
            # Known variable, new value: add with small prior mass.
            n   = len(self._beliefs[var])
            new_mass = 1.0 / (n + 1)
            # Renormalise existing values to make room for new.
            scale = 1.0 - new_mass
            for v in self._beliefs[var]:
                self._beliefs[var][v] *= scale
            self._beliefs[var][value] = new_mass
            self._domains.setdefault(var, []).append(value)

    @staticmethod
    def _normalise(b: Dict[Any, float]) -> None:
        total = sum(b.values()) or 1e-12
        for k in b:
            b[k] /= total

    # ------------------------------------------------------------------
    # Observation updates

    def observe(
        self,
        variable:   str,
        value:      Any,
        certainty:  Optional[float] = None,
    ) -> None:
        """Sharp Bayesian update: concentrate belief on observed value.

        After this call, P(variable = value) ≈ certainty (default 0.97).
        The residual (1 - certainty) is distributed proportionally over
        other values, preserving their relative ordering.

        Parameters
        ----------
        variable    Name of the state variable.
        value       Observed value.
        certainty   Confidence in the observation.  None → use __init__ default.
        """
        sharp = certainty if certainty is not None else self._sharp
        self._ensure(var=variable, value=value)
        b = self._beliefs[variable]

        # Concentrate mass on observed value.
        other_total = max(sum(v for k, v in b.items() if k != value), 1e-12)
        for k in b:
            if k == value:
                b[k] = sharp
            else:
                b[k] = (1.0 - sharp) * (b[k] / other_total)
        self._normalise(b)

    def unobserve(self, variable: str, value: Any) -> None:
        """Soft update: observed that value is NOT the current state.

        Redistributes this value's mass to other values proportionally.
        """
        if variable not in self._beliefs:
            return
        b = self._beliefs[variable]
        if value not in b:
            return
        mass = b[value]
        b[value] = 1e-6
        other_total = max(sum(v for k, v in b.items() if k != value), 1e-12)
        for k in b:
            if k != value:
                b[k] += mass * (b[k] / other_total)
        self._normalise(b)

    def update_from_obs(
        self,
        obs_dict:  dict,
        certainty: Optional[float] = None,
        skip_keys: Optional[List[str]] = None,
    ) -> None:
        """Bulk update from an observation dictionary.

        For each (key, value) in obs_dict, calls ``observe(key, value)``.
        Large/complex values (lists, dicts) that cannot form categorical
        distributions are handled specially:
          - Lists → observed as a frozenset (unordered membership).
          - Dicts → skipped by default (too structured for categorical belief).

        Parameters
        ----------
        obs_dict   Observation dict (e.g., the state dict from build_state()).
        certainty  Override sharpness for all updates in this call.
        skip_keys  Variable names to ignore (e.g., 'admissible', 'description').
        """
        skip = set(skip_keys or ['admissible', 'description', 'text',
                                  'info', 'raw_obs'])
        for key, val in obs_dict.items():
            if key in skip:
                continue
            if isinstance(val, dict):
                continue  # dict values not representable as categorical
            if isinstance(val, (list, set, frozenset)):
                # For collections: observe membership of each element.
                collection = list(val)
                for item in collection:
                    self.observe(f'{key}:{item}', True, certainty=certainty)
                # Observe absence of items NOT in the collection?
                # (Omitted in Phase R3 for efficiency; add in Phase R4 if needed.)
            elif isinstance(val, (str, int, float, bool)) or val is None:
                self.observe(key, val, certainty=certainty)

    # ------------------------------------------------------------------
    # Temporal dynamics

    def decay(self, rate: Optional[float] = None) -> None:
        """Decay all beliefs toward uniform prior (one time step passes).

        Implements temporal uncertainty: the world may have changed since
        the last observation.  Beliefs drift toward maximum entropy (uniform)
        at rate (1 - decay) per step.

        Parameters
        ----------
        rate    Override per-step decay rate.  None → use __init__ default.
        """
        d = rate if rate is not None else self._decay
        for var, b in self._beliefs.items():
            n       = max(len(b), 1)
            uniform = 1.0 / n
            for k in b:
                b[k] = d * b[k] + (1.0 - d) * uniform
            self._normalise(b)

    # ------------------------------------------------------------------
    # Query

    def most_likely(self, variable: str) -> Tuple[Any, float]:
        """Return (most_likely_value, probability) for a variable.

        Returns ('_unknown', 0.0) if the variable has never been observed.
        """
        if variable not in self._beliefs:
            return '_unknown', 0.0
        b = self._beliefs[variable]
        if not b:
            return '_unknown', 0.0
        best = max(b, key=b.__getitem__)
        return best, b[best]

    def probability(self, variable: str, value: Any) -> float:
        """Return P(variable = value).  0.0 if never observed."""
        if variable not in self._beliefs:
            return 0.0
        return self._beliefs[variable].get(value, 0.0)

    def entropy(self, variable: str) -> float:
        """Shannon entropy of P(variable) in bits.  0 = certain; log₂(n) = uniform."""
        if variable not in self._beliefs:
            return 0.0
        return -sum(
            p * math.log2(p + 1e-12)
            for p in self._beliefs[variable].values()
            if p > 0
        )

    def total_entropy(self) -> float:
        """Sum of per-variable entropies: total world-state uncertainty in bits."""
        return sum(self.entropy(var) for var in self._beliefs)

    def prediction_error(self, variable: str, value: Any) -> float:
        """Perceptual prediction error for observing value for variable.

        = -log P(variable = value)  [nats]

        High → observation is surprising under current belief.
        0    → observation was fully expected (probability 1).
        """
        p = self.probability(variable, value)
        return -math.log(max(p, 1e-12))

    def total_prediction_error(self, obs_dict: dict) -> float:
        """Sum of prediction errors across all observed variables.

        Equivalent to the perceptual free energy F_perception:
          F = Σ_i -log q(s_i = observed_i)

        High F_perception → current beliefs are inconsistent with observations;
        strong prediction errors drive belief updating.
        """
        total = 0.0
        for key, val in obs_dict.items():
            if isinstance(val, (str, int, float, bool)) or val is None:
                total += self.prediction_error(key, val)
        return total

    def sample(self) -> Dict[str, Any]:
        """Sample one world state from the factored belief q(s).

        Returns a state dict with one sampled value per known variable.
        """
        import random
        result: Dict[str, Any] = {}
        for var, b in self._beliefs.items():
            vals  = list(b.keys())
            probs = list(b.values())
            # Weighted random choice.
            cumul = 0.0
            r     = random.random()
            chosen = vals[-1]
            for v, p in zip(vals, probs):
                cumul += p
                if r < cumul:
                    chosen = v
                    break
            result[var] = chosen
        return result

    # ------------------------------------------------------------------
    # Inspection

    def known_variables(self) -> List[str]:
        """Return list of all variables with non-trivial beliefs."""
        return list(self._beliefs.keys())

    def summary(self, top_variables: int = 20, min_entropy: float = 0.01) -> str:
        """Human-readable summary of the belief state.

        Shows variables ordered by entropy (most uncertain first), skipping
        low-entropy variables below min_entropy threshold.

        Parameters
        ----------
        top_variables   Maximum number of variables to show.
        min_entropy     Variables with entropy below this are omitted (too certain).
        """
        rows: List[Tuple[float, str]] = []
        for var in self._beliefs:
            ent  = self.entropy(var)
            if ent < min_entropy:
                continue
            val, prob = self.most_likely(var)
            rows.append((ent, f'    {var:<30} P({val!r}) = {prob:.2f}  '
                              f'H = {ent:.2f} bits'))
        rows.sort(key=lambda x: -x[0])
        lines = [r for _, r in rows[:top_variables]]
        total_h = self.total_entropy()
        header  = f'    total H = {total_h:.2f} bits  '
        header += f'({len(self._beliefs)} variables known)'
        return header + ('\n' + '\n'.join(lines) if lines else '')

    def __repr__(self) -> str:
        return (
            f'VariationalBelief('
            f'vars={len(self._beliefs)}, '
            f'H_total={self.total_entropy():.2f} bits)'
        )
