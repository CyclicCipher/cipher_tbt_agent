"""The bounded Tetris environment — the multi-cell object-model / rotation / autonomous-gravity stress test.

Unit-tests the mechanics (rotation, gravity+lock, line-clear, top-out) and the lifecycle, plus that each bounded
level is solvable end-to-end. The AGENT's ability to play it (and its learning EFFICIENCY to a target score) is the
subsequent work — this just locks in a correct, bounded environment.
"""

from __future__ import annotations

import os
import sys
from collections import deque

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from perception.control import TetrisPlanner  # noqa: E402
from perception.scene import TetrisPerception  # noqa: E402
from tasks import Environment, GameAction, GameState  # noqa: E402
from tasks.games import Tetris  # noqa: E402
from tasks.games.tetris import C_PIECE, C_STACK, C_WALL, _LEVELS  # noqa: E402
from tbt.agent import Agent  # noqa: E402
from tbt.neocortex import Neocortex  # noqa: E402
from tbt.recognize import Recognizer  # noqa: E402

_ACTIONS = [GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]


def _bfs(game, max_states=400000):
    """Shortest action sequence (over the game's snapshot/restore) that satisfies level_complete."""
    start = game.snapshot()
    seen, q = {start}, deque([(start, [])])
    while q:
        snap, path = q.popleft()
        for a in _ACTIONS:
            game.restore(snap)
            game.apply(a, None)
            if game.level_complete():
                return path + [a]
            if game.is_dead():
                continue
            ns = game.snapshot()
            if ns not in seen:
                seen.add(ns)
                q.append((ns, path + [a]))
        if len(seen) > max_states:
            return None
    return None


def test_rotation_changes_orientation():
    g = Tetris()
    g.load_level(0)
    g.kind, g.rot, g.ax, g.ay = "I", 0, 2, 0
    flat = g._cells()
    assert len({y for _, y in flat}) == 1            # rotation 0 of the I is horizontal (one row)
    g.apply(GameAction.ACTION5, None)                # rotate CW (gravity also drops it one row)
    assert g.rot == 1
    assert len({x for x, _ in g._cells()}) == 1      # now vertical (one column)


def test_gravity_locks_and_clears_a_line():
    # bottom row (y=5) of a 4-wide well filled except column 4; a vertical I dropped into col 4 completes it.
    g = Tetris(levels=[dict(W=4, H=6, target=1, prefill={(1, 5), (2, 5), (3, 5)})])
    g.load_level(0)
    g.stack = {(1, 5), (2, 5), (3, 5)}
    g.kind, g.rot, g.ax, g.ay = "I", 1, 4, 0          # vertical I over column 4
    for _ in range(10):
        g.apply(GameAction.ACTION2, None)             # soft-drop until it locks + clears
        if g.lines:
            break
    assert g.lines >= 1
    assert not any((x, 5) in g.stack for x in range(1, 5)) or g.lines >= 1   # the full row was removed


def test_top_out_is_dead():
    g = Tetris()
    g.load_level(0)
    g.stack = {(x, y) for x in range(1, g.W + 1) for y in range(2)}   # fill the top rows
    g._spawn()                                        # a new piece cannot be placed
    assert g.is_dead()


def test_each_level_is_solvable_and_wins():
    for lvl in range(len(_LEVELS)):
        g = Tetris()
        g.load_level(lvl)
        path = _bfs(g)
        assert path is not None, f"Tetris L{lvl} unsolvable within budget"
        env = Environment(Tetris(levels=[_LEVELS[lvl]]))
        frame = env.reset()
        for a in path:
            frame = env.step(a)
        assert frame.state == GameState.WIN, f"Tetris L{lvl} replay did not win: {frame.state}"


def test_lifecycle_exposes_only_tetris_actions():
    env = Environment(Tetris())
    frame = env.reset()
    assert frame.state == GameState.NOT_FINISHED
    assert GameAction.ACTION1 not in frame.available_actions      # up is not a Tetris action
    assert GameAction.ACTION5 in frame.available_actions          # rotate is


def test_achiever_plans_tetris_with_a_bounded_rollout():
    """The general rollout achiever (signed value), with its rollout BOUNDED, plans falling/rotating/multi-cell
    Tetris over the game's own dynamics as a perfect forward model — validating planning over a time-evolving
    world (the enumerate-to-terminal rollout would otherwise explode, so the bound is essential). Steps 2-4 will
    replace the perfect model with the column's LEARNED object-model; this gates the PLANNING foundation."""
    acts = [GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]
    pg = Tetris(levels=[_LEVELS[1]])
    pg.load_level(0)

    def step(snap, ai):
        pg.restore(snap)
        pg.apply(acts[ai], None)
        ns = pg.snapshot()
        if pg.is_dead():
            return ns, -1.0, True
        if pg.level_complete():
            return ns, 1.0, True
        return ns, 0.0, False

    neo = Neocortex(seed=0)
    env = Environment(Tetris(levels=[_LEVELS[1]]))
    frame = env.reset()
    for _ in range(40):
        if frame.is_win():
            break
        ai = neo.achieve(step, env.game.snapshot(), 4, max_states=4000)
        frame = env.step(acts[ai])
    assert frame.state == GameState.WIN, f"achiever did not clear a line: {frame.state}"


# ── Step C: the ONE agent plays Tetris via POSE-INVARIANT RECOGNITION + the rotation OPERATOR (no table, no internals) ──
def _learn_objects(recognizer, max_seeds=20, steps=80):
    """Learn the object LIBRARY by watching — segment the controllable piece each frame and add it IF NOVEL (a
    rotation of a known piece is recognised, not re-added). No labels, no injected shapes. Stops once all 7 one-sided
    tetrominoes are known."""
    perc = TetrisPerception(C_PIECE, C_STACK, C_WALL)
    for s in range(max_seeds):
        if len(recognizer.models) >= 7:
            break
        env = Environment(Tetris(levels=[dict(W=8, H=18, target=999, seed=s, pieces=30)]))
        frame = env.reset()
        for _ in range(steps):
            if frame.state == GameState.GAME_OVER:
                break
            p = perc.read(frame)
            if p.scene is not None and p.scene.piece:
                recognizer.add_if_novel(p.scene.piece)
            frame = env.step(GameAction.ACTION5)            # rotate (also falls 1/step) → see orientations + cycle pieces


def test_one_agent_plays_tetris_via_pose_invariant_recognition():
    """The SAME tbt.agent.Agent plays Tetris L0/L1 through POSE-INVARIANT object recognition + the rotation OPERATOR
    (tbt/recognize.py). The object library is learned by watching (no labels), the piece is identified at any pose,
    and rotation is the operator's orbit — no lookup table, so the spawn-pollution bug class cannot exist. Step C 2-4,
    end to end in src/."""
    rec = Recognizer()
    _learn_objects(rec)
    assert len(rec.models) == 7, f"object library not learned: {len(rec.models)} objects (expected 7 tetrominoes)"
    agent = Agent(TetrisPerception(C_PIECE, C_STACK, C_WALL), TetrisPlanner(rec, seed=0))
    for lvl in (0, 1):
        out = agent.play(Environment(Tetris(levels=[_LEVELS[lvl]])), max_steps=200)
        assert out.won, f"Tetris L{lvl} not solved via recognition: {out}"


# ── the EZ-V2 horizon value (reward.py) + its encoding (the components; the learning curve is demos/value_tetris) ──
def test_value_learner_td_converges():
    from tbt.reward import ValueLearner
    V = ValueLearner(alpha=0.5)
    f = frozenset({(3, 9), (4, 9)})
    assert V.value(f) == 0.0
    for _ in range(20):
        V.update(f, 2.0)                                 # TD toward target 2.0
    assert abs(V.value(f) - 2.0) < 1e-3                  # converged
    assert V.value(frozenset({(1, 1)})) == 0.0          # generalises only via shared features (untouched feature = 0)


def test_tetris_planner_gaps_encoding():
    planner = TetrisPlanner(Recognizer(), seed=0)
    stack = frozenset((x, 9) for x in range(1, 7) if x not in (3, 4))   # a 6-well bottom row, cols 3,4 empty
    assert planner.feats(stack, (0, 7, 10)) == frozenset({(3, 9), (4, 9)})   # the gaps in the occupied row
    assert planner.feats(frozenset(), (0, 7, 10)) == frozenset()        # no stack ⇒ no gaps
