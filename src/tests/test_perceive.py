"""test_perceive.py — the peripheral RETINA: a game frame → OBJECTS, and the DISCOVERED self (perceive.py; the game-loop bridge).

Step 1 of the thin-agent game loop. The bridge segments a raw colour frame into objects (Core-Knowledge objectness, no
semantics) and discovers the controllable ROOT from MOTION — never colour-as-self. Grounded on the real LockPath frame so the
build starts from what `render()` actually produces.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                      # noqa: E402
from tbt.perceive import segment                 # noqa: E402
from tasks.core import GameAction                # noqa: E402
from tasks.games.lockpath import LockPath        # noqa: E402
from tasks.harness import Environment            # noqa: E402


def test_segment_recovers_lockpath_objects():
    """On the real LockPath L0 frame, the retina recovers exactly the objects present — the wall boundary (one connected
    component), the agent cell, the goal cell — at their true positions, with NO semantics read."""
    game = LockPath()
    game.load_level(0)
    grid = game.render()[0]
    objs = segment(grid)
    by_color = {}
    for o in objs:
        by_color.setdefault(o.color, []).append(o)
    assert set(by_color) == {1, 2, 3}, f"L0 has walls(1), agent(2), goal(3) only, got {sorted(by_color)}"
    (agent_obj,) = by_color[2]
    (goal_obj,) = by_color[3]
    assert agent_obj.cells == frozenset({game.agent}) and agent_obj.anchor == game.agent, "agent = one cell at its position"
    assert goal_obj.cells == frozenset({game.goal}), "goal = one cell at its position"
    assert len(by_color[1]) == 1 and len(by_color[1][0].cells) > 4, "the wall boundary is ONE connected object"


def test_segment_groups_a_multicell_object():
    """A same-colour 4-connected region is ONE object (multi-cell), anchored at its top-left — what Sokoban blocks / Tetris
    pieces need."""
    grid = [[0, 0, 0, 0], [0, 6, 6, 0], [0, 6, 0, 0], [0, 0, 0, 0]]   # an L of colour 6
    (six,) = [o for o in segment(grid) if o.color == 6]
    assert six.cells == frozenset({(1, 1), (2, 1), (1, 2)}), f"the 4-connected region is one object, got {sorted(six.cells)}"
    assert six.anchor == (1, 1), "anchor = the top-left cell"


def test_self_is_discovered_from_motion_not_colour():
    """Drive the agent through LockPath and the retina discovers the controllable ROOT from MOTION: the object that moves with
    every action (colour 2 here) is the self — learned, never assumed from a colour value."""
    game = LockPath()
    env = Environment(game)
    fd = env.reset()
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    for act in (GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION2, GameAction.ACTION2):
        before = a.transduce(fd.grid)
        fd = env.step(act)
        after = a.transduce(fd.grid)
        a.observe_self(before, after, act)
    assert a.self_color() == 2, f"the object that moves with the actions is the self, got {a.self_color()}"


if __name__ == "__main__":
    g = LockPath(); g.load_level(2)
    for o in segment(g.render()[0]):
        print(f"colour {o.color}: {len(o.cells)} cell(s) at {o.anchor}")
