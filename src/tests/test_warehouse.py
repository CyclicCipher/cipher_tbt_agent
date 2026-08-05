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
from tbt.operator import norm, sub                     # noqa: E402
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


def test_the_crates_are_now_TRACKED_and_the_push_is_learned():
    """WHAT THE INDEXES FIXED. `_positions` still cannot hold two crates — a `{feature: cell}` map never will — but the
    scene is no longer keyed on appearance: every object gets an INDEX (`Column.track`), so both crates are held apart, one
    of them is seen to MOVE, and the kind it carries is learned as a mover. Before this the scene held only the wall and
    `_movers` was empty through an entire game.

    The pads are NOT movers, which is the part that took the work: a pointer that may jump to the nearest same-coloured
    thing will claim the other pad the moment a crate occludes its own, and that phantom displacement taught "pad moves"."""
    a, fd = _play()
    objects = a.transduce(fd.grid)
    assert 6 not in a._positions(objects), "the feature-keyed index still cannot hold two crates — that is not what fixed it"
    assert a._movers == {6}, f"the crate kind must be discovered to move, and ONLY it, got {a._movers}"
    scene = a._scene_col()
    crates = [i for i in scene.scene_snapshot() if scene.feature_of(i) == 6]
    assert len(crates) == 2, f"both crates must be held as separate things, got {len(crates)}"


def test_the_push_DELTA_is_learned_and_not_learned_as_zero():
    """THE BLOCKER UNDER THE BLOCKER. `_movers` knows the crate KIND moves — that is the test above — but the L5 transform
    has to know HOW MUCH it moves, and that is a different fact learned in a different place. Measured before the fix:
    `_dynamics_delta("of", 6, ...)` returned (0, 0). The agent had not failed to learn; it had actively learned that
    CRATES DO NOT MOVE, and every planner above it was then correct to conclude a push does nothing.

    The cause was that `_learn_dynamics` read displacement out of the FEATURE-keyed `_positions`, which holds nothing for a
    colour realised by two identical objects, and then taught the resulting zero as if it were an observation. Absence of
    evidence was being learned as evidence of absence — so the fix is not only to read motion off the INDEXES, but to
    decline the lesson when the pressed thing's displacement was not observable at all."""
    a, _fd = _play()
    for action, eff in ((None, (1.0, 0.0)), (None, (-1.0, 0.0)), (None, (0.0, 1.0)), (None, (0.0, -1.0))):
        delta = a._dynamics_delta("of", 6, None, eff)
        if norm(delta) > 1e-6:
            assert norm(sub(delta, eff)) < 1e-6, f"a pushed crate must move BY the press, got {delta} for {eff}"
            break
    else:
        raise AssertionError("the crate push delta is still (0, 0) in every direction — 'crates do not move' was learned")


def test_what_is_STILL_unsolved_is_the_set_goal_itself():
    """The fixture's original question, now ACTUALLY REACHED. This used to assert `goals() == set()` and passed for the
    wrong reason: `_winning_conditions` read the feature-keyed `_positions`, so on two identical crates it generated no
    mover condition at all and `goal_mem` had nothing to compete over. An empty goal set was being read as "no pair names
    this win" when the truth was "no candidate was ever offered".

    Read per INDEX, a pair goal IS discovered — `(6, 7)`, "a crate rests on a pad" — and it is the WRONG one for exactly
    the reason the fixture was built to expose: the condition set collapses two deliveries into one element, so it is
    already true when the level is half done and cannot tell half-done from done. That is the set goal, isolated at last,
    with perception, tracking, dynamics and candidate generation all out of the way."""
    a, fd = _play()
    assert a.goal_mem.goals() == {(6, 7)}, f"a pair goal must now be discovered at all, got {a.goal_mem.goals()}"
    game = Warehouse()
    game.load_level(0)
    half = {sorted(game.pads)[0]}
    game.crates = half | {c for c in sorted(game.crates) if c not in game.pads}
    assert not game.level_complete(), "the level is NOT won with one pad covered …"
    assert (6, 7) in a.goal_mem.goals(), "… yet the discovered pair goal is already satisfied there, which is the gap"
    assert fd.score < 3, "so the agent still does not finish the game"


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
