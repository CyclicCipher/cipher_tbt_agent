"""ARC-AGI-3 object-model types (the live API contract): FrameData / GameAction / GameState / colour grids.

Pared down to the types the live integration (`arc_run` / `arc_sdk`) speaks. The offline replica games, harness, and
agents were removed in the reset to columns + API; rebuild any offline fixtures on the columns as needed.
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
]
