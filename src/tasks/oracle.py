"""A generic BFS oracle for ARC-AGI-3-style games.

This is a *teacher*, not an agent: it plans an optimal solution by searching over a game's true internal state
(it knows the win condition and simulates the dynamics). Its job is to (a) verify procedurally-generated layouts
are solvable and (b) generate winning experience for the agent's learning phase to observe. The learning agent
never uses it — the agent sees only frames.

Generic across games: each `Game` supplies `snapshot()`/`restore()` (its hashable mutable state) and
`available_actions()`; the oracle searches over the non-coordinate actions. (Coordinate/click actions would
make the branching factor the whole grid — out of scope for the teacher; movement/interact games only.)
"""

from __future__ import annotations

import sys
from collections import deque
from typing import List, Optional

from .core import GameAction

# A solve_level that explores more than this many distinct states bails (returns None) instead of hanging — a
# safety net for mechanics whose JOINT state is combinatorial (many independently-movable pieces: N blocks search
# ~positions x C(cells, N)). Callers treat None as "no oracle path" (collect falls back to random hints; scoring
# drops that level's baseline). Generous, so it never trips on a legitimately-solvable layout — it exists to keep
# a pathological / procedurally-generated layout from turning the per-level solve into an hours-long hang.
_DEFAULT_MAX_STATES = 2_000_000
_budget_warned = False


def _capture(game):
    return game.snapshot()


def _restore(game, snap) -> None:
    game.restore(snap)


def solve_level(game, max_states: int = _DEFAULT_MAX_STATES) -> Optional[List[GameAction]]:
    """BFS for a shortest action sequence completing the game's current level. Mutates `game` while searching,
    then restores it to the level start so the caller can replay the path through a real Environment. Returns
    None if unsolvable OR if the search exceeds `max_states` (the safety budget — it degrades gracefully instead
    of hanging on a combinatorial joint state)."""
    global _budget_warned
    level = game._level
    actions = [a for a in game.available_actions() if not getattr(a, "requires_coordinates", False)]
    start = _capture(game)
    seen = {start}
    queue: deque = deque([(start, [])])
    solution: Optional[List[GameAction]] = None
    while queue:
        if len(seen) > max_states:                       # safety budget: bail rather than hang
            if not _budget_warned:
                print(f"[oracle] solve_level exceeded {max_states:,} states - bailing (level treated as "
                      f"unsolved); reduce movable pieces or raise max_states.", file=sys.stderr)
                _budget_warned = True
            break
        state, path = queue.popleft()
        _restore(game, state)
        if game.level_complete():
            solution = path
            break
        for action in actions:
            _restore(game, state)
            game.apply(action, None)
            if game.is_dead():
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
