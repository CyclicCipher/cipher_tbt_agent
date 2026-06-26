"""The Environment harness and the Scorecard.

The Environment is generic: it drives any Game through the ARC-AGI-3 lifecycle,
gates actions, counts actions, advances levels, and emits FrameData. It is the
in-process analogue of the ARC-AGI-3 REST harness.

Lifecycle / scoring rules (faithful to the benchmark):
  - reset() starts (or restarts) the whole game at level 0, state NOT_FINISHED.
  - On completing a level, `score` increments. Completing the last level -> WIN.
  - Dying -> GAME_OVER; thereafter the only legal action is RESET, which restarts
    the *current* level (completed levels stay completed).
  - `action_counter` counts every submitted action (including RESET) and is the
    score tiebreaker, so wasted exploration is penalized.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .core import (
    ActionNotAvailable,
    Coordinates,
    FrameData,
    GameAction,
    GameState,
)
from .game import Game


class Environment:
    """Drives a single Game instance through the ARC-AGI-3 lifecycle."""

    def __init__(self, game: Game):
        self.game = game
        self.guid: str = ""
        self.state: GameState = GameState.NOT_PLAYED
        self.level: int = 0
        self.score: int = 0           # levels completed
        self.action_counter: int = 0

    # -- public API ---------------------------------------------------------

    def reset(self) -> FrameData:
        """Start a fresh game instance at level 0."""
        self.guid = uuid.uuid4().hex
        self.level = 0
        self.score = 0
        self.action_counter = 0
        self.game.load_level(0)
        self.state = GameState.NOT_FINISHED
        return self._frame()

    def step(
        self, action: GameAction, coordinates: Optional[Coordinates] = None
    ) -> FrameData:
        """Submit one action and return the resulting frame."""
        if self.state == GameState.NOT_PLAYED:
            raise ActionNotAvailable("call reset() before step()")

        if action not in self._available_actions():
            raise ActionNotAvailable(
                f"{action.name} is not available in state {self.state.value} "
                f"(available: {[a.name for a in self._available_actions()]})"
            )

        if action.requires_coordinates:
            _validate_coordinates(coordinates)
        else:
            coordinates = None

        self.action_counter += 1

        if action == GameAction.RESET:
            # Restart the current level; completed levels stay completed.
            self.game.load_level(self.level)
            self.state = GameState.NOT_FINISHED
            return self._frame()

        self.game.apply(action, coordinates)

        if self.game.is_dead():
            self.state = GameState.GAME_OVER
        elif self.game.level_complete():
            self.score += 1
            if self.level + 1 >= self.game.level_count:
                self.state = GameState.WIN
            else:
                self.level += 1
                self.game.load_level(self.level)
                self.state = GameState.NOT_FINISHED
        else:
            self.state = GameState.NOT_FINISHED

        return self._frame()

    # -- internals ----------------------------------------------------------

    def _available_actions(self) -> List[GameAction]:
        if self.state == GameState.GAME_OVER:
            return [GameAction.RESET]
        if self.state == GameState.WIN:
            return []
        # Mid-play: RESET is always allowed, plus whatever the game exposes.
        return [GameAction.RESET, *self.game.available_actions()]

    def _frame(self) -> FrameData:
        return FrameData(
            game_id=self.game.game_id,
            guid=self.guid,
            frame=self.game.render(),
            state=self.state,
            score=self.score,
            level=self.level,
            available_actions=self._available_actions(),
            action_counter=self.action_counter,
        )


@dataclass
class GameResult:
    """Outcome of playing one game to termination (or an action budget)."""

    game_id: str
    won: bool
    levels_completed: int
    total_actions: int


@dataclass
class Scorecard:
    """Aggregates results across one or more games.

    Primary metric: total levels completed. Tiebreaker: fewer total actions.
    This mirrors the benchmark's ranking and rewards skill-acquisition
    *efficiency*, not just eventual success.
    """

    results: Dict[str, GameResult] = field(default_factory=dict)

    def record(self, result: GameResult) -> None:
        self.results[result.game_id] = result

    @property
    def levels_completed(self) -> int:
        return sum(r.levels_completed for r in self.results.values())

    @property
    def total_actions(self) -> int:
        return sum(r.total_actions for r in self.results.values())

    @property
    def games_won(self) -> int:
        return sum(1 for r in self.results.values() if r.won)

    def summary(self) -> str:
        lines = [
            f"Scorecard: {self.games_won}/{len(self.results)} games won, "
            f"{self.levels_completed} levels, {self.total_actions} actions"
        ]
        for r in self.results.values():
            tag = "WIN" if r.won else "..."
            lines.append(
                f"  [{tag}] {r.game_id}: {r.levels_completed} levels, "
                f"{r.total_actions} actions"
            )
        return "\n".join(lines)


def _validate_coordinates(coordinates: Optional[Coordinates]) -> None:
    if coordinates is None:
        raise ActionNotAvailable("ACTION6 requires (x, y) coordinates")
    x, y = coordinates
    if not (0 <= x <= 63 and 0 <= y <= 63):
        raise ActionNotAvailable(f"coordinates {coordinates} out of range 0..63")
