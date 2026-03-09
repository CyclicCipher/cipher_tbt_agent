"""goal_learning.py — Learn preference factors from causal experience.

Architecture
============
Phase R5 of the Active Inference roadmap.  Implements ``discover_goals()``:
an unsupervised algorithm that extracts preference factors (what the agent
should want) from its own causal experience — without manual specification.

Design
======
In the current system (DecisionEngine / early AIFEngine), goals are
*specified* by the programmer::

    eat_goal = FEPGoal('eat', drives=[hunger_drive], condition=has_food, ...)

This is a form of prior knowledge injection.  It works for small domains
where we know in advance what goals matter.  For a new complex game with
hundreds of possible goals, manual specification is impractical.

``discover_goals()`` replaces this with unsupervised goal inference:

    1. Observe: the agent records (s, a, s', reward) tuples during exploration.
    2. Discover: for each Drive D, find transitions where D.deficit decreased.
       These are empirically good outcomes for drive D.
    3. Extract: find state features that *predict* when an action will reduce D.
       A feature with high conditional probability P(feature=v | D reduced) is
       informative about goal-relevant states.
    4. Generate: emit a PreferenceFactor for each informative feature-drive pair.
       The preference concentrates log P(o) on states with that feature.

This is analogous to Phase O (unsupervised POS category discovery from text)
but applied to causal experience rather than linguistic statistics:

    Phase O:          word distributions → syntactic categories
    discover_goals(): state-action distributions → goal categories

Output
======
``discover_goals()`` returns a list of ``DiscoveredGoal`` objects — structured
descriptions of what was learned.  The caller (adapter / AIFEngine) can:
  - Convert to ``PreferenceFactor`` objects and add to a ``GenerativeModel``.
  - Inspect the evidence (which transitions supported the discovery).
  - Filter by confidence before accepting.

The ``to_preference_factor()`` method on each ``DiscoveredGoal`` performs
the conversion automatically.

Design principle
================
**THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST BE GENERAL.**

This file contains zero game-specific logic.  Drive objects, state dicts,
and causal history are provided by the caller.

See AIF_ROADMAP.md for the full Phase R implementation plan.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DiscoveredGoal  (structured discovery output)
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredGoal:
    """One goal preference discovered from causal experience.

    Attributes
    ----------
    drive_name
        Name of the Drive this goal serves (e.g. 'hunger').
    feature_key
        State dict key of the discovered predictive feature (e.g. 'inv:apple').
    feature_value
        Value of the feature in drive-reducing states (e.g. True).
    confidence
        Conditional probability P(feature=value | drive reduced).
        Range [0, 1].  Higher = more reliable predictor.
    support
        Number of drive-reducing transitions that supported this discovery.
    base_rate
        Unconditional P(feature=value) across all transitions.
        Used to compute the lift: confidence / base_rate.
    drive_urgency
        Urgency of the parent Drive (used to weight the PreferenceFactor).
    evidence
        List of (s, a, s') tuples that contributed to this discovery.
        Kept for debugging / auditing; truncated to at most 10 entries.
    """

    drive_name:    str
    feature_key:   str
    feature_value: Any
    confidence:    float
    support:       int
    base_rate:     float
    drive_urgency: float
    evidence:      List[Tuple[dict, str, dict]] = field(default_factory=list)

    @property
    def lift(self) -> float:
        """Lift = confidence / base_rate.  > 1 means predictive (above chance)."""
        return self.confidence / max(self.base_rate, 1e-8)

    @property
    def information_gain(self) -> float:
        """Information gain in bits for this feature-drive association.

        IG = H(drive_reduced) - H(drive_reduced | feature=value)
           = -log2(p) + p*log2(p) + (1-p)*log2(1-p)  (rough approximation)
        Higher = feature is more predictive of drive reduction.
        """
        p = self.confidence
        b = self.base_rate
        if p <= 0 or p >= 1 or b <= 0 or b >= 1:
            return 0.0
        # KL divergence: how much does knowing this feature help?
        try:
            kl = p * math.log2(p / b) + (1 - p) * math.log2((1 - p) / (1 - b))
        except (ValueError, ZeroDivisionError):
            kl = 0.0
        return max(0.0, kl)

    def to_preference_factor(self, scale: float = 5.0) -> Any:
        """Convert to a PreferenceFactor for use in a GenerativeModel.

        Parameters
        ----------
        scale   Log-preference slope per unit of feature mismatch.
                Passed through to PreferenceFactor.from_drive().

        Returns
        -------
        PreferenceFactor from generative_model.py.
        """
        from generative_model import PreferenceFactor

        key   = self.feature_key
        value = self.feature_value

        # The preference: log P(o) is high when state[key] == value.
        # Implemented as a Drive-like measure:
        #   measure(s) = 1.0 if s.get(key) == value else 0.0
        # Then PreferenceFactor.from_drive converts this to log P(o).
        measure  = lambda s, k=key, v=value: 1.0 if s.get(k) == v else 0.0  # noqa
        urgency  = self.drive_urgency * self.confidence   # discount by confidence
        name     = f'{self.drive_name}:{key}={value!r}'

        return PreferenceFactor.from_drive(
            name     = name,
            measure  = measure,
            setpoint = 1.0,
            urgency  = urgency,
            scale    = scale,
        )

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f'drive={self.drive_name:<12} '
            f'feature={self.feature_key}={self.feature_value!r:<8} '
            f'conf={self.confidence:.2f}  '
            f'lift={self.lift:.2f}x  '
            f'IG={self.information_gain:.2f}bit  '
            f'n={self.support}'
        )


# ---------------------------------------------------------------------------
# discover_goals
# ---------------------------------------------------------------------------

def discover_goals(
    causal_history:       List[Tuple[dict, str, dict, float]],
    drives:               List[Any],              # List[Drive] from planning.py
    min_support:          int   = 3,
    min_confidence:       float = 0.5,
    min_lift:             float = 1.2,
    max_goals_per_drive:  int   = 5,
    skip_keys:            Optional[List[str]] = None,
) -> List[DiscoveredGoal]:
    """Discover goal-relevant preference factors from causal experience.

    Parameters
    ----------
    causal_history
        List of (prev_state, action, next_state, reward) tuples.
        Typically collected during the agent's exploration phase (Phase Q1 style).
        The more diverse the exploration, the more reliable the discoveries.

    drives
        List of ``Drive`` objects from planning.py.  Each drive defines a
        homeostatic urgency and a ``measure(state) -> float`` function.
        ``discover_goals`` finds which state features predict reduction in
        each drive's deficit.

    min_support
        Minimum number of drive-reducing transitions that must support
        a discovered feature for it to be reported.  Default 3.
        Increase for noisy environments.

    min_confidence
        Minimum conditional probability P(feature | drive reduced) required.
        Default 0.5.  Features below this threshold are not reported.

    min_lift
        Minimum lift (confidence / base_rate) required.  Default 1.2.
        Lift < 1.0 means the feature is anti-correlated with drive reduction.
        Lift = 1.0 means no predictive power (random).

    max_goals_per_drive
        Maximum number of goals reported per drive, sorted by information gain.
        Default 5.

    skip_keys
        State dict keys to ignore (e.g. 'admissible', 'description').
        Complex or high-cardinality keys that cannot form useful categorical
        features are automatically skipped.

    Returns
    -------
    List[DiscoveredGoal]
        Sorted by information_gain descending.  The caller should inspect
        these and decide which to add to the GenerativeModel.

    Algorithm
    ---------
    For each Drive D:

      1. Partition transitions into 'drive-reducing' and 'other':
         drive_reduced = D.deficit(prev_state) > D.deficit(next_state) + ε

      2. For each (key, value) pair in the state dicts:
         Compute:
           support     = |{(s,a,s') : drive_reduced AND s[key] == value}|
           confidence  = support / |drive_reduced transitions|
           base_rate   = P(s[key] == value) over all transitions
           lift        = confidence / base_rate

      3. Report (key, value) pairs with support >= min_support,
         confidence >= min_confidence, lift >= min_lift.

    Complexity: O(|history| × |drives| × |state_keys|).
    For typical runs (200 steps, 5 drives, 20 state keys): very fast (<1ms).
    """
    if not causal_history or not drives:
        return []

    skip = set(skip_keys or [])

    results: List[DiscoveredGoal] = []

    for drive in drives:
        drive_name    = getattr(drive, 'name', str(drive))
        drive_urgency = getattr(drive, 'urgency', 1.0)
        drive_measure = getattr(drive, 'measure', None)
        drive_setpoint = getattr(drive, 'setpoint', 1.0)
        if drive_measure is None:
            continue

        # --- Step 1: partition into reducing / non-reducing ----------------
        reducing: List[Tuple[dict, str, dict]] = []
        all_trans: List[Tuple[dict, str, dict]] = []

        for entry in causal_history:
            ps, a, ns, _reward = entry
            try:
                deficit_before = max(0.0, drive_setpoint - float(drive_measure(ps)))
                deficit_after  = max(0.0, drive_setpoint - float(drive_measure(ns)))
            except Exception:
                continue
            all_trans.append((ps, a, ns))
            if deficit_before - deficit_after > 0.01:   # ε = 0.01
                reducing.append((ps, a, ns))

        n_reducing = len(reducing)
        n_total    = len(all_trans)
        if n_reducing < min_support or n_total == 0:
            continue

        # --- Step 2: compute feature statistics ----------------------------
        # Collect all (key, value) pairs from prev_states.
        feature_counts_reducing: Dict[Tuple[str, Any], int] = defaultdict(int)
        feature_counts_total:    Dict[Tuple[str, Any], int] = defaultdict(int)
        evidence_map: Dict[Tuple[str, Any], List[Tuple[dict, str, dict]]] = defaultdict(list)

        for ps, a, ns in all_trans:
            for key, val in ps.items():
                if key in skip:
                    continue
                if not _is_simple(val):
                    continue
                fk = (key, val)
                feature_counts_total[fk] += 1

        for ps, a, ns in reducing:
            for key, val in ps.items():
                if key in skip:
                    continue
                if not _is_simple(val):
                    continue
                fk = (key, val)
                feature_counts_reducing[fk] += 1
                if len(evidence_map[fk]) < 10:
                    evidence_map[fk].append((ps, a, ns))

        # --- Step 3: compute confidence / lift / IG -------------------------
        discovered: List[DiscoveredGoal] = []
        for fk, support in feature_counts_reducing.items():
            if support < min_support:
                continue
            key, val = fk
            confidence = support / n_reducing
            base_rate  = feature_counts_total.get(fk, 0) / n_total
            if confidence < min_confidence:
                continue
            if base_rate < 1e-8:
                continue
            lift = confidence / base_rate
            if lift < min_lift:
                continue

            goal = DiscoveredGoal(
                drive_name    = drive_name,
                feature_key   = key,
                feature_value = val,
                confidence    = confidence,
                support       = support,
                base_rate     = base_rate,
                drive_urgency = drive_urgency,
                evidence      = evidence_map[fk],
            )
            discovered.append(goal)

        # Sort by information gain, return top-k.
        discovered.sort(key=lambda g: g.information_gain, reverse=True)
        results.extend(discovered[:max_goals_per_drive])

    # Final sort across all drives by information gain.
    results.sort(key=lambda g: g.information_gain, reverse=True)
    return results


# ---------------------------------------------------------------------------
# update_generative_model  (convenience: discover + inject)
# ---------------------------------------------------------------------------

def update_generative_model(
    generative_model: Any,           # GenerativeModel from generative_model.py
    causal_history:   List[Tuple[dict, str, dict, float]],
    drives:           List[Any],
    scale:            float = 5.0,
    verbose:          bool  = False,
    **kwargs,
) -> int:
    """Discover goals and add them to a GenerativeModel as PreferenceFactor objects.

    Convenience wrapper: ``discover_goals()`` → ``to_preference_factor()``
    → ``generative_model.preferences.append()``.

    Parameters
    ----------
    generative_model   GenerativeModel instance (modified in place).
    causal_history     Collected (s, a, s', reward) tuples.
    drives             Drive list to analyse.
    scale              PreferenceFactor log-preference scale.
    verbose            If True, print a summary of discoveries.
    **kwargs           Forwarded to ``discover_goals()``.

    Returns
    -------
    int  Number of new PreferenceFactor objects added to the model.
    """
    goals = discover_goals(causal_history, drives, **kwargs)
    if verbose:
        print(f'  discover_goals: {len(goals)} goals found from '
              f'{len(causal_history)} transitions × {len(drives)} drives')
        for g in goals:
            print(f'    {g.summary()}')

    n_added = 0
    for goal in goals:
        pf = goal.to_preference_factor(scale=scale)
        generative_model.preferences.append(pf)
        n_added += 1

    return n_added


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_simple(val: Any) -> bool:
    """Return True if val is a simple scalar suitable as a feature value."""
    return isinstance(val, (str, int, float, bool)) or val is None
