"""Theory of mind: belief functor B_alpha over other agents' action distributions.

The theory of mind module tracks a belief state for each observed agent
(NPC, opponent, collaborator) as a sliding-window frequency distribution over
that agent's actions.  After each observation of agent alpha taking action a,
the belief B_alpha is updated so that future predictions of alpha's behaviour
become better calibrated.

The belief state is a categorical distribution: each action maps to an
empirical count within a recent window.  The window prevents stale evidence
from dominating — if an NPC changes strategy mid-game, the new behaviour
is reflected within `window` observations.

This is a lightweight approximation of the full belief functor described in
CTKG_ARCHITECTURE.md §TheoryOfMind.  The functor interpretation: B_alpha
maps the CTKG's morphism space to a probability simplex; update_action
applies the Bayesian update step (here: exact count update with window eviction).

Categorical reference: Fritz & Klingler (2023), d-separation in categorical
probability.  The belief update is the finite stochastic matrix (FinStoch)
corresponding to a Markov kernel from observation sequences to agent models.

See ROADMAP.md Stage 5, Step 5.3 for design decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """Belief state for a single observed agent.

    Attributes
    ----------
    agent_id:
        Identifier string for the agent being modelled.
    action_counts:
        Raw count of each action observed within the current window.
    total:
        Total number of observations recorded (may exceed window if capped).
    """

    agent_id: str
    action_counts: dict[str, int] = field(default_factory=dict)
    total: int = 0

    def distribution(self) -> dict[str, float]:
        """Return normalised probability distribution over observed actions.

        Returns a uniform distribution over all observed actions if total == 0.
        """
        if self.total == 0 or not self.action_counts:
            n = len(self.action_counts)
            if n == 0:
                return {}
            return {a: 1.0 / n for a in self.action_counts}
        return {a: c / self.total for a, c in self.action_counts.items()}

    def most_likely(self) -> Optional[str]:
        """Return the action with the highest probability, or None if empty."""
        if not self.action_counts:
            return None
        return max(self.action_counts, key=lambda a: self.action_counts[a])

    def __repr__(self) -> str:
        top = self.most_likely()
        return (
            f"BeliefState(agent={self.agent_id!r}, "
            f"total={self.total}, "
            f"top_action={top!r})"
        )


# ---------------------------------------------------------------------------
# TheoryOfMind
# ---------------------------------------------------------------------------

class TheoryOfMind:
    """Sliding-window belief tracker for multiple observed agents.

    Maintains one BeliefState per agent.  Observations are added with
    observe_action(); the window evicts the oldest observations to prevent
    stale evidence from dominating.

    Parameters
    ----------
    window:
        Maximum number of recent observations to remember per agent.
        Default: 20 (about one room's worth of NPC interactions).
    """

    def __init__(self, window: int = 20) -> None:
        self._window = window
        self._beliefs: dict[str, BeliefState] = {}
        # Track per-agent action history for window eviction
        self._history: dict[str, deque[str]] = {}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def observe_action(self, agent_id: str, action: str) -> None:
        """Record that agent_id performed action.

        Parameters
        ----------
        agent_id:
            Identifier of the observed agent.
        action:
            The action string observed (e.g. 'go_north', 'take_key').
        """
        if agent_id not in self._beliefs:
            self._beliefs[agent_id] = BeliefState(agent_id=agent_id)
            self._history[agent_id] = deque()

        hist = self._history[agent_id]
        belief = self._beliefs[agent_id]

        # Increment count
        belief.action_counts[action] = belief.action_counts.get(action, 0) + 1
        belief.total += 1
        hist.append(action)

        # Evict oldest if over window
        if len(hist) > self._window:
            old_action = hist.popleft()
            if old_action in belief.action_counts:
                belief.action_counts[old_action] -= 1
                belief.total -= 1
                if belief.action_counts[old_action] <= 0:
                    del belief.action_counts[old_action]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def action_probs(self, agent_id: str) -> dict[str, float]:
        """Return normalised distribution over agent_id's next action.

        Returns an empty dict if agent_id has never been observed.

        Parameters
        ----------
        agent_id:
            Agent to query.

        Returns
        -------
        dict mapping action -> probability.  Probabilities sum to 1.0.
        """
        if agent_id not in self._beliefs:
            return {}
        return self._beliefs[agent_id].distribution()

    def predict_action(self, agent_id: str) -> str:
        """Return the most likely next action for agent_id.

        Returns '' if agent_id has never been observed.

        Parameters
        ----------
        agent_id:
            Agent to query.

        Returns
        -------
        str -- most likely action, or '' if unknown.
        """
        if agent_id not in self._beliefs:
            return ''
        top = self._beliefs[agent_id].most_likely()
        return top if top is not None else ''

    def get_belief(self, agent_id: str) -> Optional[BeliefState]:
        """Return the full BeliefState for agent_id, or None if unknown."""
        return self._beliefs.get(agent_id)

    def known_agents(self) -> list[str]:
        """Return sorted list of agent_ids with at least one observation."""
        return sorted(self._beliefs.keys())

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TheoryOfMind(agents={self.known_agents()}, "
            f"window={self._window})"
        )
