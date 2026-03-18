"""Abstract Environment base class.

Every concrete environment (TextWorldEnv, TiTS, Danganronpa, Minecraft) inherits
from this class and implements the four required methods.

Edge types returned by observe() match the topology registered via
agent_topology() in core/topology.py:
  None  — sequence start (first token only)
  0     — extero  (world-state)
  1     — intero  (self-state)
  2     — action  (fed by the AgentLoop between observations)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Environment(ABC):
    """Abstract perception-action environment.

    Concrete subclasses implement observe/act and maintain world state.
    The AgentLoop drives the interaction cycle:

        env.reset()
        while not (env.done or env.won):
            tokens = env.observe()
            action  = agent.select(tokens, env.available_actions())
            env.act(action)
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset world state to the initial configuration."""

    @abstractmethod
    def observe(self) -> list[tuple[str, Optional[int]]]:
        """Return the full observation for the current timestep.

        Returns
        -------
        List of (token, edge_type) pairs.  The first pair always has
        edge_type=None (sequence start).  Subsequent pairs use integer
        edge types from agent_topology() — 0 for extero, 1 for intero.
        """

    @abstractmethod
    def act(self, action: str) -> None:
        """Execute action and advance the world state.

        The consequence is visible in the NEXT call to observe().
        """

    @abstractmethod
    def available_actions(self) -> list[str]:
        """Return the list of legal actions from the current state."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """True when the episode has ended (failure or termination)."""

    @property
    @abstractmethod
    def won(self) -> bool:
        """True when the agent has achieved the goal."""
