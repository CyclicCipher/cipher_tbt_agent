"""A BFS oracle for LockPath-style games.

This is a *teacher*, not an agent: it plans an optimal solution by searching over
the game's true internal state (it knows the win condition and simulates the
dynamics). Its job is to (a) verify procedurally-generated layouts are solvable
and (b) produce ground-truth action labels for behavior cloning. The learning
agent never uses it — the agent sees only frames.
"""

from __future__ import annotations

from collections import deque
from typing import FrozenSet, List, Optional, Tuple

from .core import GameAction

# The four directional actions LockPath responds to.
_DIRECTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
]

# The mutable part of a LockPath state, as a hashable key for BFS.
_Dyn = Tuple[Tuple[int, int], FrozenSet, FrozenSet, bool, bool]


def _capture(game) -> _Dyn:
    return (
        game.agent,
        frozenset(game.blocks),
        frozenset(game.keys),
        game.has_key,
        game._dead,
    )


def _restore(game, dyn: _Dyn) -> None:
    game.agent = dyn[0]
    game.blocks = set(dyn[1])
    game.keys = set(dyn[2])
    game.has_key = dyn[3]
    game._dead = dyn[4]


def solve_level(game) -> Optional[List[GameAction]]:
    """BFS for a shortest action sequence completing the game's current level.

    Mutates `game` while searching, then restores it to the level start so the
    caller can replay the returned path through a real Environment. Returns None
    if the level is unsolvable.
    """
    level = game._level
    start = _capture(game)
    seen = {start}
    queue: deque = deque([(start, [])])
    solution: Optional[List[GameAction]] = None
    while queue:
        state, path = queue.popleft()
        _restore(game, state)
        if game.level_complete():
            solution = path
            break
        for action in _DIRECTIONS:
            _restore(game, state)
            game.apply(action, None)
            if game._dead:
                continue
            nxt = _capture(game)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [action]))
    game.load_level(level)
    return solution


def is_solvable(game) -> bool:
    """True iff the game's current level has a solution."""
    return solve_level(game) is not None
