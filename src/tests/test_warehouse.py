"""test_warehouse.py — the SET-goal fixture, and the blocker it found UPSTREAM of the thing it was built to test.

`test_crates` showed a graded relational value is redundant wherever `goal_mem`'s `(mover, landmark)` PAIR goal names the
win, and asked for the case a pair CANNOT state. `tasks.games.warehouse` is that case, breaking the pair representation two
ways at once: the win is "EVERY pad covered" (a SET property — one crate on one pad is not a win, so a pair is true while
the level is half done), and every crate is one colour and every pad another, so `(6, 7)` names a COLOUR PAIR rather than a
crate and a pad, leaving the assignment — which crate to which pad — with nowhere to live.

WHAT IT ACTUALLY FOUND, which is not what it was aimed at. The pair goal does fail here, exactly as predicted
(`goal_mem.goals()` is empty). But so does everything else, for a reason that sits far upstream of goals and values:
`Agent._positions` is a `{feature: cell}` MAP — one cell per feature — which structurally cannot hold two instances of one
feature. Two identical 1×1 crates are distinguishable neither by colour nor by shape, so they are dropped as ambiguous;
`_movers` therefore stays EMPTY, the push is never learned, the scene holds only the wall, and no goal, value or subgoal
mechanism downstream ever sees a crate. The frames are perceived correctly — the retina segments `[1, 2, 6, 6, 7, 7]` — and
then the representation throws the duplicates away.

So this is the `project_representation_shortcut_lesson` shape for the fifth time: a convenient representation (index the
world by feature, since features were unique in every fixture so far) fails at the seam where the convenience stops holding.
The general state is a LIST OF INSTANCES with the feature as a read-out. Recorded here rather than fixed, because that is a
representational change reaching `_track_movers`, `_learn_dynamics`, `_credit_goal`, `_route_scene` and the world map — a
direction to take deliberately, not a repair to slip into a fixture commit.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                            # noqa: E402
from tasks.games.warehouse import Warehouse, _LEVELS   # noqa: E402
from tasks.harness import Environment                  # noqa: E402
from tasks.oracle import solve_level                   # noqa: E402

ORACLE = [9, 9, 8]
_PLAYED: list = []


def _play(budget: int = 70):
    """One memoised play — several tests read different facts off the same seeded trajectory."""
    if _PLAYED:
        return _PLAYED[0]
    env = Environment(Warehouse())
    fd = env.reset()
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    for _ in range(budget):
        action, coords = a.step(fd)
        fd = env.step(action, coords)
        if fd.is_terminal() or fd.is_win():
            break
    _PLAYED.append((a, fd))
    return _PLAYED[0]


def test_every_level_is_solvable():
    """The fixture must be honest before anything measured on it counts."""
    for level in range(len(_LEVELS)):
        game = Warehouse()
        game.load_level(level)
        plan = solve_level(game)
        assert plan and len(plan) == ORACLE[level], f"L{level} must cost {ORACLE[level]}, oracle says {plan and len(plan)}"


def test_the_win_is_a_SET_property_that_no_pair_can_state():
    """One crate on one pad is NOT a win. A pair goal is true from the first delivery onward, so it cannot tell half-done
    from done — which is the representational point of the whole fixture."""
    game = Warehouse()
    game.load_level(0)
    pad = sorted(game.pads)[0]
    game.crates = {pad} | {c for c in list(game.crates)[:1] if c != pad}
    assert not game.level_complete(), "one pad covered must not be a win"
    game.crates = set(game.pads)
    assert game.level_complete(), "every pad covered is"


def test_the_instances_are_indistinguishable_by_colour_AND_by_shape():
    """Why recognition cannot rescue this. Same-coloured objects are told apart by SHAPE, but two 1×1 crates have the same
    shape too — so there is genuinely nothing about either object that distinguishes it from the other. Only POSITION does,
    which is precisely what an index keyed on the feature has thrown away."""
    a, fd = _play()
    objects = a.transduce(fd.grid)
    crates = [o for o in objects if o.color == 6]
    assert len(crates) == 2, f"the level must present two crates, got {len(crates)}"
    shapes = {frozenset((x - o.anchor[0], y - o.anchor[1]) for x, y in o.cells) for o in crates}
    assert len(shapes) == 1, "and they must be the same shape, so shape cannot disambiguate them either"


def test_the_blocker_is_UPSTREAM_the_agent_never_even_sees_a_crate():
    """THE FINDING. The retina segments the crates correctly, and then `_positions` — a `{feature: cell}` map — drops them
    because two instances cannot share one key. Nothing downstream recovers: `_movers` stays empty, so the push is never
    learned; the scene holds only the wall; and the goal learner has nothing to credit. The failure is representational and
    sits well before any question about goals, values or subgoals."""
    a, fd = _play()
    objects = a.transduce(fd.grid)
    assert {o.color for o in objects} >= {6, 7}, "the crates and pads ARE perceived"
    assert 6 not in a._positions(objects), "but an ambiguous feature has no place in a feature-keyed index"
    assert not a._movers, "so the crates are never discovered to move, and the push is never learned"
    assert a.goal_mem.goals() == set(), "and the pair goal learns nothing, as the fixture predicted"


def test_the_relational_value_cannot_help_because_the_blocker_is_below_it():
    """Ablating the relational value changes nothing here either — but for a different reason than in `test_crates`. There
    it was redundant against a sharper mechanism; here it is starved, because the representation never delivers a crate for
    any relation to be about. A fixture built to discriminate two mechanisms discriminated neither, and said why."""
    def play(relational: bool):
        env = Environment(Warehouse())
        fd = env.reset()
        a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
        if not relational:
            a._task_value_of = lambda w: 0.0
        per, base = [], 0
        for _ in range(60):
            action, coords = a.step(fd)
            fd = env.step(action, coords)
            if fd.score > len(per):
                per.append(fd.action_counter - base)
                base = fd.action_counter
            if fd.is_terminal() or fd.is_win():
                break
        return per

    assert play(True) == play(False), "the ablation must be measured, not assumed"
