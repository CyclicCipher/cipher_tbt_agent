"""1-D number line environment — arithmetic counterpart of MazeEnv.

Positions  : integers in [lo, hi]
Actions    : FORWARD (+1) or BACKWARD (-1)
Interface  : mirrors MazeEnv (step → (pos, hit_wall), valid_actions)
             so the same cortical column machinery applies unchanged.
"""
from __future__ import annotations

FORWARD  = 0   # +1 step along the number line
BACKWARD = 1   # -1 step along the number line

# Mirrors maze_env.DELTA but 1-D: action → displacement tuple
DELTA: dict[int, tuple[int]] = {
    FORWARD:  ( 1,),
    BACKWARD: (-1,),
}


class NumberLineEnv:
    """Discrete 1-D number line with hard boundaries.

    Parameters
    ----------
    lo, hi : inclusive integer bounds.
             Attempting to step outside returns hit_wall=True and leaves
             position unchanged (same semantics as a maze wall).
    """

    def __init__(self, lo: int = 0, hi: int = 9) -> None:
        self.lo  = lo
        self.hi  = hi
        self.pos = lo

    def reset_at(self, pos: int) -> None:
        """Hard-reset to pos."""
        if not (self.lo <= pos <= self.hi):
            raise ValueError(f"pos={pos} not in [{self.lo},{self.hi}]")
        self.pos = int(pos)

    def step(self, action: int) -> tuple[int, bool]:
        """Execute action.  Returns (new_pos, hit_wall)."""
        new_pos = self.pos + DELTA[action][0]
        if self.lo <= new_pos <= self.hi:
            self.pos = new_pos
            return self.pos, False
        return self.pos, True   # boundary hit; position unchanged

    def valid_actions(self) -> list[int]:
        actions: list[int] = []
        if self.pos < self.hi:
            actions.append(FORWARD)
        if self.pos > self.lo:
            actions.append(BACKWARD)
        return actions
