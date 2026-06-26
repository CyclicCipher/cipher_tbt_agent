"""Core object model for the ARC-AGI-3 replica.

Mirrors the public ARC-AGI-3 API (docs.arcprize.org) so an agent written against
this in-process environment ports to the real REST API with minimal change.

Fidelity notes (what is faithful to the real benchmark):
  - Observation is a *frame*: a list of 1..N grids, each up to 64x64, with integer
    color values in 0..15. Indexing is grid[y][x] with (0,0) at the top-left.
  - Action space: RESET, ACTION1..ACTION4 (up/down/left/right), ACTION5 (a
    game-specific "interact"), ACTION6 (complex, requires (x, y) in 0..63),
    ACTION7 (undo).
  - State machine: NOT_PLAYED -> NOT_FINISHED -> WIN | GAME_OVER. On GAME_OVER the
    only legal action is RESET.
  - Each frame advertises which actions are currently available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

# Grid conventions (faithful to ARC-AGI-3).
GRID_SIZE = 64          # maximum grid edge
NUM_COLORS = 16         # legal cell values are 0..15

# A single grid is rows of integer color values, indexed grid[y][x].
Grid = List[List[int]]
# A frame is one or more grids stacked (the real API returns 1..N grids).
Frame = List[Grid]
# Coordinates for ACTION6 are (x, y) with 0 <= x, y <= 63.
Coordinates = Tuple[int, int]


class GameAction(Enum):
    """The full ARC-AGI-3 action space.

    The directional/interact *semantics* below are the conventional mapping the
    benchmark documents; a given game is free to use any subset and to interpret
    ACTION5 ("interact") however it likes. ACTION6 is the only action that carries
    coordinates.
    """

    RESET = 0     # initialize / restart the current level
    ACTION1 = 1   # up
    ACTION2 = 2   # down
    ACTION3 = 3   # left
    ACTION4 = 4   # right
    ACTION5 = 5   # interact (game-specific: select / rotate / toggle / ...)
    ACTION6 = 6   # complex action; requires (x, y) coordinates in 0..63
    ACTION7 = 7   # undo

    @property
    def requires_coordinates(self) -> bool:
        return self is GameAction.ACTION6

    @property
    def is_movement(self) -> bool:
        return self in _MOVEMENT

    @property
    def delta(self) -> Coordinates:
        """(dx, dy) for the four directional actions; (0, 0) otherwise."""
        return _DELTAS.get(self, (0, 0))


_MOVEMENT = frozenset(
    {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
)

# Screen coordinates: +y is down (row index grows downward), matching grid[y][x].
_DELTAS = {
    GameAction.ACTION1: (0, -1),   # up
    GameAction.ACTION2: (0, 1),    # down
    GameAction.ACTION3: (-1, 0),   # left
    GameAction.ACTION4: (1, 0),    # right
}


class GameState(Enum):
    """Lifecycle of a game instance."""

    NOT_PLAYED = "NOT_PLAYED"      # instance created, no RESET yet
    NOT_FINISHED = "NOT_FINISHED"  # mid-play
    WIN = "WIN"                    # all levels completed
    GAME_OVER = "GAME_OVER"        # died; only RESET is legal


@dataclass
class FrameData:
    """One observation returned by the environment after reset() or step().

    Fields mirror the real API's frame object closely enough that agent code is
    portable. `frame` is the list of grids; `grid` is a convenience accessor for
    the primary (last) grid, which is what single-grid games return.
    """

    game_id: str
    guid: str                       # unique id for this game instance
    frame: Frame                    # 1..N grids of color values 0..15
    state: GameState
    score: int                      # number of levels completed so far
    level: int                      # current level index (0-based)
    available_actions: List[GameAction]
    action_counter: int = 0         # total actions taken (the score tiebreaker)

    @property
    def grid(self) -> Grid:
        """The primary grid (the last grid in the frame)."""
        return self.frame[-1]

    def is_terminal(self) -> bool:
        return self.state in (GameState.WIN, GameState.GAME_OVER)


class ActionNotAvailable(Exception):
    """Raised when an action not in `available_actions` is submitted.

    Mirrors the real API returning HTTP 400 for an illegal action (e.g. anything
    other than RESET once the game is in GAME_OVER).
    """
