"""Object-affordance perception (tbt.affordance, GROUNDING_PLAN §3 G-A): detect MOVABLE objects (from motion) and
SALIENT MARKERS (rare static distinct cells, remembered) from the frame stream -- general, no colour ids."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tasks import games  # noqa: E402
from tasks.core import GameAction  # noqa: E402
from tbt.affordance import Affordances  # noqa: E402

C_AGENT, C_GOAL, C_BLOCK, C_PAD = 2, 3, 6, 7


def test_affordances_detect_movables_and_remembered_markers():
    """On Sokoban L0, after the agent pushes the block: the BLOCK and the AGENT are detected MOVABLE (they moved), and
    the PAD and GOAL are SALIENT MARKERS (rare, static) -- all from motion + salience, never a colour id."""
    game = games.Sokoban()
    game.load_level(0)
    aff = Affordances()
    aff.update(game.render()[-1])
    for a in (GameAction.ACTION2, GameAction.ACTION4):    # agent: down to (1,2), then right -> pushes the block right
        game.apply(a, None)
        aff.update(game.render()[-1])

    movable = aff.movable_colors()
    assert C_AGENT in movable and C_BLOCK in movable, movable        # both moved -> movable
    marker_colours = set(aff.markers().values())
    assert C_PAD in marker_colours and C_GOAL in marker_colours, marker_colours   # rare static -> markers
    assert C_BLOCK not in marker_colours and C_AGENT not in marker_colours        # movables are NOT markers


def test_marker_is_remembered_after_being_covered():
    """A pad COVERED by a block (the pad colour disappears under the block) stays a remembered marker -- the L7-A
    memory: an object/target seen then occluded is still mapped, so the GSG can still aim a movable at it."""
    game = games.Sokoban()
    game.load_level(0)                                     # block (2,2), pad (6,2): push the block right onto the pad
    aff = Affordances()
    aff.update(game.render()[-1])
    for a in (GameAction.ACTION2,) + (GameAction.ACTION4,) * 5:   # down, then right x5 (push block (2,2)->(6,2)=pad)
        game.apply(a, None)
        aff.update(game.render()[-1])
    assert (6, 2) in aff.markers(), aff.markers()          # the pad at (6,2) is remembered even though a block covers it
    assert aff.markers()[(6, 2)] == C_PAD
