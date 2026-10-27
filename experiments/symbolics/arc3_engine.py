"""
arc3_engine.py
Core framework and API for ARC-AGI-3 interactive worlds.
Follows the official ARC-AGI-3 turn-based grid and action specification:
- 16-color palette (0-15)
- Standard action space: ACTION1..ACTION5 (discrete) + ACTION6 (complex/coordinates) + RESET
- Frame observation lifecycle: step(action, data=...) -> FrameData
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class GameAction(Enum):
    RESET = "RESET"
    ACTION1 = "ACTION1"  # Typically North / Up
    ACTION2 = "ACTION2"  # Typically South / Down
    ACTION3 = "ACTION3"  # Typically West  / Left
    ACTION4 = "ACTION4"  # Typically East  / Right
    ACTION5 = "ACTION5"  # Typically Interact / Toggle / Action
    ACTION6 = "ACTION6"  # Complex Action (requires {"x": int, "y": int})

    def is_complex(self) -> bool:
        return self == GameAction.ACTION6


class GameState(Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


# ARC 16-Color Palette ANSI mapping for terminal rendering
ARC_PALETTE_ANSI = {
    0: "\033[40m  \033[0m",   # Black (Background)
    1: "\033[44m  \033[0m",   # Blue
    2: "\033[41m  \033[0m",   # Red
    3: "\033[42m  \033[0m",   # Green
    4: "\033[43m  \033[0m",   # Yellow
    5: "\033[47m  \033[0m",   # Grey
    6: "\033[45m  \033[0m",   # Magenta
    7: "\033[48;5;208m  \033[0m",  # Orange
    8: "\033[46m  \033[0m",   # Cyan
    9: "\033[48;5;94m  \033[0m",   # Brown
    10: "\033[48;5;200m  \033[0m", # Pink
    11: "\033[48;5;28m  \033[0m",  # Dark Green
    12: "\033[48;5;17m  \033[0m",  # Dark Blue
    13: "\033[48;5;52m  \033[0m",  # Maroon
    14: "\033[48;5;238m  \033[0m", # Dark Grey
    15: "\033[48;5;255m  \033[0m", # White
}


class FrameData:
    """The observation payload returned after every step."""
    def __init__(
        self,
        grid: List[List[int]],
        state: GameState,
        step_count: int,
        available_actions: List[GameAction],
        levels_completed: int = 0
    ):
        self.grid = grid  # 2D array [height][width] with values 0..15
        self.state = state
        self.step_count = step_count
        self.available_actions = available_actions
        self.levels_completed = levels_completed

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def render_terminal(self, title: str = ""):
        """Pretty-prints the ARC grid to the terminal using ANSI colors."""
        header = f"--- {title} | Step: {self.step_count} | Status: {self.state.value} ---"
        print(header)
        border = "+-" + "--" * self.width + "-+"
        print(border)
        for row in self.grid:
            line = "| " + "".join(ARC_PALETTE_ANSI.get(c, "??") for c in row) + " |"
            print(line)
        print(border)


class ARCBaseWorld:
    """
    Abstract Base Class for all ARC-AGI-3 environments.
    Handles action lifecycle, grid allocation, and state transitions.
    """
    def __init__(self, width: int = 12, height: int = 12, max_steps: int = 100):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.step_count = 0
        self.state = GameState.NOT_STARTED
        self.grid: List[List[int]] = []
        self.levels_completed = 0

    @property
    def action_space(self) -> List[GameAction]:
        return [
            GameAction.RESET,
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
            GameAction.ACTION6,
        ]

    # --- Grid Manipulation Utilities for Subclasses ---
    def clear_grid(self, fill_color: int = 0):
        self.grid = [[fill_color for _ in range(self.width)] for _ in range(self.height)]

    def set_cell(self, x: int, y: int, color: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = color

    def get_cell(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return -1

    def draw_rect(self, x: int, y: int, w: int, h: int, color: int):
        for cy in range(y, y + h):
            for cx in range(x, x + w):
                self.set_cell(cx, cy, color)

    # --- ARC Lifecycle Methods ---
    def reset(self) -> FrameData:
        self.step_count = 0
        self.state = GameState.IN_PROGRESS
        self.setup_world()
        return self._make_frame()

    def step(self, action: GameAction, data: Optional[Dict[str, Any]] = None) -> FrameData:
        if action == GameAction.RESET:
            return self.reset()

        if self.state != GameState.IN_PROGRESS:
            return self._make_frame()

        self.step_count += 1
        self.update_world(action, data or {})

        # Default timeout
        if self.step_count >= self.max_steps and self.state == GameState.IN_PROGRESS:
            self.state = GameState.GAME_OVER

        return self._make_frame()

    def _make_frame(self) -> FrameData:
        # Return deep copy of current grid state
        grid_copy = [row[:] for row in self.grid]
        return FrameData(
            grid=grid_copy,
            state=self.state,
            step_count=self.step_count,
            available_actions=self.action_space,
            levels_completed=self.levels_completed
        )

    # Subclasses must implement these two methods:
    def setup_world(self):
        """Build initial level layout."""
        raise NotImplementedError

    def update_world(self, action: GameAction, data: Dict[str, Any]):
        """Execute tick logic for the chosen action."""
        raise NotImplementedError


# --- World Registry ---
_WORLD_REGISTRY: Dict[str, type] = {}

def register_world(name: str):
    def decorator(cls):
        _WORLD_REGISTRY[name] = cls
        return cls
    return decorator

def make_world(name: str, **kwargs) -> ARCBaseWorld:
    if name not in _WORLD_REGISTRY:
        raise ValueError(f"World '{name}' not found. Available: {list(_WORLD_REGISTRY.keys())}")
    return _WORLD_REGISTRY[name](**kwargs)

def list_worlds() -> List[str]:
    return list(_WORLD_REGISTRY.keys())
