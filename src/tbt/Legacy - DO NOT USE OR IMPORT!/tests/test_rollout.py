"""R0 — validate SEARCH over the (ground-truth) model before any learned model exists (`src/tbt/ROLLOUT_PLAN.md` R0).

The linear-value read-off is proven insufficient (`test_sokoban_hold`), so planning must be deliberative SEARCH over a
forward model toward the goal. This validates the search MECHANISM on the TRUE model: given only `step(state, action)` and
the win predicate `is_goal` (NO hand-coded heuristic, no per-game structure), a forward-model search finds an executable
winning plan and it WINS — on Sokoban L0 and the multi-cell M0, so it is not L0-specific.

Uninformed (BFS) on purpose: it validates search CORRECTNESS with zero domain knowledge. The SAMPLED / efficient variant
(Gumbel + sequential halving + an SR value at the leaves) is R3 — needed only when the branching is too large to enumerate,
and where a value heuristic is required *because* sparse-reward sampling alone would never find the win. R0 proves the
model + goal are plannable; R3 makes the planner cheap.
"""

from __future__ import annotations

import os
import sys
from collections import deque

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tasks.games.sokoban import MULTICELL_LEVELS, Sokoban  # noqa: E402


def plan_forward_search(start, step, is_goal, actions, max_expand=200_000):
    """Model-agnostic forward-model search (the R0 search skeleton): BFS from `start` over `step(state, action) → state'`
    to the first `is_goal` state; returns the winning action tuple, or None. `state` must be hashable. R3 swaps the
    uninformed expansion here for SAMPLED (Gumbel) expansion + an SR value at the leaves; the interface is unchanged."""
    seen = {start}
    q = deque([(start, ())])
    expanded = 0
    while q and expanded < max_expand:
        s, path = q.popleft()
        if is_goal(s):
            return path
        expanded += 1
        for a in actions:
            s2 = step(s, a)
            if s2 not in seen:
                seen.add(s2)
                q.append((s2, path + (a,)))
    return None


def _model(game):
    """The ground-truth model as the (step, is_goal, actions) interface — the SAME interface a LEARNED model will implement."""
    def step(state, a):
        game.restore(state)
        game.apply(a, None)
        return game.snapshot()

    def is_goal(state):
        game.restore(state)
        return game.level_complete()

    return step, is_goal, game.available_actions()


def _search_solves(level, levels=None):
    game = Sokoban(levels) if levels is not None else Sokoban()
    game.load_level(level)
    step, is_goal, actions = _model(game)
    start = game.snapshot()
    plan = plan_forward_search(start, step, is_goal, actions)
    assert plan is not None, "forward-model search found no winning plan"
    game.restore(start)                                        # EXECUTE the found plan in the real game
    for a in plan:
        game.apply(a, None)
    assert game.level_complete(), f"the plan did not actually win (len={len(plan)})"


def test_forward_search_solves_sokoban_l0():
    """Deliberative forward-model search wins L0 from the model + win predicate alone — the value read-off could not."""
    _search_solves(0)


def test_forward_search_solves_multicell_m0():
    """...and the multi-cell domino M0, so the search is not L0-specific (no per-game code path)."""
    _search_solves(0, MULTICELL_LEVELS)
