"""The plug-and-play Environment contract for the thalamo-cortical agent.

Any experiment — number line, arithmetic, LockPath, … — implements `Environment`; the agentic wrapper
(`tbt/agent.py`) drives the column(s) + thalamus + basal ganglia + reward over it **unchanged**. Kept
deliberately tiny and domain-agnostic, and **torch-free** (so it imports without pulling in PyTorch):

  observation : whatever the agent must perceive (a symbol, a grid, …) — the agent decides how to read it.
  action      : an index into `actions`.
  reward      : a scalar; sparse, may be 0 (stage-1 structure learning uses none).
  done        : episode-end flag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Step:
    observation: Any
    reward: float
    done: bool


class Environment(ABC):
    @abstractmethod
    def reset(self) -> Any:
        """Start an episode; return the initial observation."""

    @abstractmethod
    def step(self, action: int) -> Step:
        """Apply an action; return the resulting (observation, reward, done)."""

    @property
    @abstractmethod
    def actions(self) -> List[int]:
        """The available action indices."""
