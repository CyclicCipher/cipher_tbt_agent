"""Present the ARC-AGI-3 harness through the domain-agnostic `tbt.env.Environment` contract.

`harness.Environment` is the rich, ARC-faithful environment (multi-level, `FrameData`, the coordinate click).
This thin adapter exposes it as the MINIMAL contract (`reset` / `step` / `actions` -> `Step`) so the thin agent
drives the SAME contract it would for any task — proving it is not coupled to ARC. Nothing about the harness or
its `FrameData` consumers (collect, oracle, the wm scorer) changes; this is additive.

Mapping: observation = the `FrameData` (perception reads it); reward = a level was just completed (score rose);
done = WIN (every level finished). GAME_OVER is deliberately NOT done — the ARC lifecycle has the agent RESET
and retry the current level — so a contract episode is the whole multi-level game.
"""

from __future__ import annotations

from typing import List, Optional

from tbt.env import Environment, Step

from .core import Coordinates, GameAction, GameState
from .game import Game
from .harness import Environment as Harness


class ContractEnv(Environment):
    def __init__(self, game: Game):
        self._h = Harness(game)
        self._score = 0

    def reset(self):
        f = self._h.reset()
        self._score = f.score
        return f

    def step(self, action: GameAction, coords: Optional[Coordinates] = None) -> Step:
        f = self._h.step(action, coords)
        reward = float(f.score - self._score)
        self._score = f.score
        return Step(observation=f, reward=reward, done=f.state == GameState.WIN)

    @property
    def actions(self) -> List[GameAction]:
        return list(self._h._available_actions())
