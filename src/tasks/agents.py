"""Agent interface and a baseline, plus an episode runner.

The Agent contract mirrors the real ARC-AGI-3 toolkit: an agent observes a frame
and chooses the next action (with coordinates iff the action requires them). No
task description, no win condition, no reward shaping is provided — the agent must
infer the goal from the pixels and the score signal alone.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from .core import Coordinates, FrameData, GameAction, GameState
from .harness import Environment, GameResult


class Agent(ABC):
    """Base class for an ARC-AGI-3 agent."""

    @abstractmethod
    def choose_action(
        self, frame: FrameData
    ) -> Tuple[GameAction, Optional[Coordinates]]:
        """Pick the next (action, coordinates). coordinates is None unless the
        action requires them."""

    def is_done(self, frame: FrameData) -> bool:
        """Stop early? Default: stop only when the game itself terminates."""
        return frame.is_terminal()

    def reset(self) -> None:
        """Clear any per-game internal state. Default: no-op."""


class RandomAgent(Agent):
    """Uniformly random over the currently-available actions.

    A sanity baseline and a fuzzer for the harness. After GAME_OVER it will pick
    the only legal action (RESET) and keep exploring until the action budget runs
    out.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def choose_action(
        self, frame: FrameData
    ) -> Tuple[GameAction, Optional[Coordinates]]:
        action = self._rng.choice(frame.available_actions)
        coords: Optional[Coordinates] = None
        if action.requires_coordinates:
            coords = (self._rng.randint(0, 63), self._rng.randint(0, 63))
        return action, coords


def run_episode(
    env: Environment,
    agent: Agent,
    max_actions: int = 1000,
) -> GameResult:
    """Play one game to termination or until the action budget is exhausted.

    Returns a GameResult capturing levels completed and total actions — the two
    quantities the Scorecard ranks on.
    """
    agent.reset()
    frame = env.reset()

    while frame.action_counter < max_actions:
        if agent.is_done(frame) and frame.state == GameState.WIN:
            break
        if frame.state == GameState.WIN:
            break
        action, coords = agent.choose_action(frame)
        frame = env.step(action, coords)

    return GameResult(
        game_id=frame.game_id,
        won=frame.state == GameState.WIN,
        levels_completed=frame.score,
        total_actions=frame.action_counter,
    )
