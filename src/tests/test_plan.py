"""The planner (tbt.plan): one active-inference value that babbles to learn its operators, explores toward novelty,
and exploits a learned goal -- no epsilon, no separate explore branch, the harness dissolved. The explore/exploit
arbitration is emergent (pursue a reachable goal/untried action, else seek the nearest unvisited arrangement). Closed
loops drive a simulated self by its OWN learned operators; coverage is compared to a random walk on an unbounded grid
(where the operators are exact, isolating exploration from the parked wall problem). Pure stdlib."""

from __future__ import annotations

import os
import random
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.forward import ForwardModel       # noqa: E402
from tbt.goal import GoalModel             # noqa: E402
from tbt.plan import Planner               # noqa: E402

MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}     # up/down/left/right -- the env's actions (effects learned)


def _teach(fm, ops):
    for a, (dx, dy) in ops.items():
        fm.observe((0, 0), a, (dx, dy))


def test_babbling_learns_every_operator_cold():
    """Cold start: with no operators known, optimism-under-uncertainty makes each untried action the best target, so
    the agent tries each ONCE and learns the lot in |actions| steps (motor babbling) -- not a random flail."""
    fm, g = ForwardModel(), GoalModel()
    p = Planner(fm, g)
    pos, land = (5, 5), [("L", (0, 0))]
    for t in range(6):
        a = p.act(pos, land, actions=[0, 1, 2, 3])
        nxt = (pos[0] + MOVES[a][0], pos[1] + MOVES[a][1])
        fm.observe(pos, a, nxt)                            # the world reveals the effect; the operator is learned
        g.observe(pos, land, 0)
        pos = nxt
        if all(fm.delta(x) is not None for x in range(4)):
            assert t + 1 <= 4                              # all four operators learned in at most |actions| steps
            return
    assert False, "cold-start babbling did not learn all operators"


def test_navigates_to_a_learned_goal_through_unexplored_space():
    """A goal learned from one score increment is reached by a direct path -- the planner trusts its operators to
    traverse cells it has never personally visited (the action-cheap choice)."""
    fm, g = ForwardModel(), GoalModel()
    _teach(fm, MOVES)
    box = [("box", (10, 8))]
    g.observe((9, 8), box, 1)                              # scored one cell left of the box -> that arrangement is the goal
    p = Planner(fm, g)
    pos, steps = (3, 8), 0
    while not g.is_goal(pos, box) and steps < 20:
        pos = fm.predict(pos, p.act(pos, box, actions=[0, 1, 2, 3]))
        steps += 1
    assert pos == (9, 8) and steps <= 8                    # ~optimal (6), no detour for novelty


def test_uses_learned_operators_not_an_assumed_move_set():
    """No notion of unit moves: with (+2,0)/(-2,0) hop operators it reaches a goal six cells along in three hops."""
    fm, g = ForwardModel(), GoalModel()
    _teach(fm, {"hopR": (2, 0), "hopL": (-2, 0)})
    g.observe((6, 0), [("flag", (20, 0))], 1)
    p = Planner(fm, g)
    pos, others, seq = (0, 0), [("flag", (20, 0))], []
    while not g.is_goal(pos, others) and len(seq) < 10:
        a = p.act(pos, others, actions=["hopR", "hopL"])
        seq.append(a)
        pos = fm.predict(pos, a)
    assert pos == (6, 0) and seq == ["hopR", "hopR", "hopR"]


def test_seeks_the_nearest_unvisited_frontier_when_no_goal():
    """No goal, operators known: the agent routes to the nearest arrangement it has never visited (directed
    exploration), not a random move and not back into the visited interior."""
    fm, g = ForwardModel(), GoalModel()
    _teach(fm, MOVES)
    land = [("L", (50, 50))]
    for c in [(0, 0), (0, -1), (-1, 0), (1, 0)]:           # up / left / right already visited; DOWN (0,1) is the frontier
        g.observe(c, land, 0)
    p = Planner(fm, g)
    assert p.act((0, 0), land, actions=[0, 1, 2, 3]) == 1  # heads down, the only depth-1 unvisited neighbour


def test_exploits_a_reachable_goal_over_exploring():
    """With a reachable goal AND unvisited frontiers everywhere, the pragmatic plan outvalues novelty -> it pursues
    the goal (the RHAE-aligned choice: once you can score, score)."""
    fm, g = ForwardModel(), GoalModel()
    _teach(fm, MOVES)
    box = [("box", (10, 0))]
    g.observe((9, 0), box, 1)                              # goal: one cell left of the box
    p = Planner(fm, g)
    assert p.act((0, 0), box, actions=[0, 1, 2, 3]) == 3   # moves right toward the goal, not off to a frontier


def test_directed_exploration_covers_more_than_a_random_walk():
    """On an unbounded grid (operators exact, so no wall confound) frontier-seeking covers more distinct cells than a
    random walk -- the action-efficiency the cold start needs."""
    land = [("L", (99, 99))]

    def cover(directed, seed, steps=30):
        fm, g = ForwardModel(), GoalModel()
        _teach(fm, MOVES)
        p = Planner(fm, g, seed=seed)
        rng = random.Random(seed)
        pos, seen = (0, 0), {(0, 0)}
        for _ in range(steps):
            a = p.act(pos, land, actions=[0, 1, 2, 3]) if directed else rng.randrange(4)
            if directed:
                g.observe(pos, land, 0)
            pos = (pos[0] + MOVES[a][0], pos[1] + MOVES[a][1])
            seen.add(pos)
        return len(seen)

    directed = sum(cover(True, s) for s in range(5))
    random_walk = sum(cover(False, s) for s in range(5))
    assert directed > random_walk                          # directed exploration is strictly more efficient
