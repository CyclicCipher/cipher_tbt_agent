"""The Game contract.

A Game owns its internal state and the mechanics of play. It knows nothing about
the GameState lifecycle, scoring, or action availability gating beyond declaring
which actions are *meaningful* to it — the Environment (harness.py) orchestrates
all of that. This separation mirrors the real benchmark: the harness is generic,
each game supplies only its world.

A game is a sequence of levels. Each level is a self-contained puzzle; the
benchmark convention is that each successive level introduces a new mechanic
(and later levels compose mechanics introduced earlier).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .core import Coordinates, Frame, GameAction


class Game(ABC):
    """Base class for an ARC-AGI-3-style game.

    Lifecycle (driven by the Environment):
        load_level(i)  -> set internal state to the start of level i
        apply(a, xy)   -> mutate internal state by one action
        render()       -> current observation (a frame = list of grids)
        level_complete -> has the current level's hidden win condition been met?
        is_dead        -> did the last action end the game (a GAME_OVER)?
    """

    #: Stable identifier, e.g. "lp01". Real game ids look like "ls20".
    game_id: str = "game"

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Total number of levels in this game."""

    @abstractmethod
    def load_level(self, level: int) -> None:
        """Initialize internal state to the start of `level` (0-based)."""

    @abstractmethod
    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        """Apply one action to the internal state.

        `action` is guaranteed to be one of `available_actions()` and never RESET
        (the harness handles RESET). `coordinates` is provided iff
        `action.requires_coordinates`.
        """

    @abstractmethod
    def render(self) -> Frame:
        """Return the current observation as a list of grids (color values 0..15)."""

    @abstractmethod
    def level_complete(self) -> bool:
        """True iff the current level's (hidden) win condition is satisfied."""

    def is_dead(self) -> bool:
        """True iff the last action put the game into an unrecoverable state.

        Default: games never die (levels are only completed). Override to add
        hazards / failure states that should trigger GAME_OVER.
        """
        return False

    def available_actions(self) -> List[GameAction]:
        """Actions meaningful to this game *right now* (excluding RESET).

        The Environment adds/removes RESET according to the lifecycle and
        re-advertises this set on every frame. Default: the four directions plus
        interact. Override to expose ACTION6/ACTION7 or to vary by state.
        """
        return [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]
