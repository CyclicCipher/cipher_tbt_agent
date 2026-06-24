"""An offline replica of the ARC-AGI-3 interactive reasoning environment.

Faithful to the public ARC-AGI-3 object model (FrameData / GameAction / GameState
/ Scorecard, 64x64 color grids, action-availability gating, the
NOT_PLAYED->NOT_FINISHED->WIN|GAME_OVER lifecycle), with original games standing
in for the proprietary ones so the harness is runnable offline.

Quick start:

    from arc_agi_3 import Environment, RandomAgent, run_episode
    from arc_agi_3.games import LockPath

    env = Environment(LockPath())
    result = run_episode(env, RandomAgent(seed=0), max_actions=2000)
    print(result)
"""

from .core import (
    ActionNotAvailable,
    Coordinates,
    Frame,
    FrameData,
    GameAction,
    GameState,
    Grid,
    GRID_SIZE,
    NUM_COLORS,
)
from .game import Game
from .harness import Environment, GameResult, Scorecard
from .agents import Agent, RandomAgent, run_episode

__all__ = [
    "ActionNotAvailable",
    "Coordinates",
    "Frame",
    "FrameData",
    "GameAction",
    "GameState",
    "Grid",
    "GRID_SIZE",
    "NUM_COLORS",
    "Game",
    "Environment",
    "GameResult",
    "Scorecard",
    "Agent",
    "RandomAgent",
    "run_episode",
]
