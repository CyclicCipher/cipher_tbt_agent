"""The ARC-AGI-3 SDK conduit (`arc_sdk.TbtPolicy`): drive THE agent through the real SDK's
`choose_action`/`is_done` contract WITHOUT the SDK or an API key. Our replica `FrameData` is modelled on the real
one (verified against arcprize/ARC-AGI-3-Agents source), so the policy's duck-typed translation runs the SAME path
the live bridge will — proving the wiring end to end (and the lifecycle + field translation in isolation)."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_sdk import TbtPolicy, _primary_grid, _to_obs  # noqa: E402
from perception.control import NeocortexPlanner  # noqa: E402
from perception.scene import Perception, WorldModel  # noqa: E402
from tasks import Environment, GameAction, GameState  # noqa: E402
from tasks.games import Sokoban  # noqa: E402
from tasks.games.sokoban import C_AGENT, C_BLOCK, C_GOAL, C_PAD, C_WALL, MULTICELL_LEVELS  # noqa: E402
from tbt.agent import Agent  # noqa: E402
from tbt.column import CorticalColumn  # noqa: E402


def _sokoban_agent() -> Agent:
    world = WorldModel(body=C_AGENT, pushable={C_BLOCK}, blocking={C_WALL},
                       goal_colors={C_GOAL}, required_absent={C_PAD})
    dm = CorticalColumn(n_entities=1)
    dm.dyn_rules = []
    return Agent(Perception(world), NeocortexPlanner(world, dm, seed=0))


def _drive(env, policy, max_steps=1500):
    """Mimic the SDK `main()` loop: choose_action → step, stop on is_done (WIN) or budget. The action comes back
    BY NAME (as it would cross the SDK enum boundary) and is re-mapped to step the env — no agent.play()."""
    frame = env.reset()
    steps = 0
    while not policy.is_done([frame], frame) and steps < max_steps:
        name, coords = policy.choose_action([frame], frame)
        frame = env.step(GameAction[name], coords)
        steps += 1
    return frame


def test_policy_drives_agent_to_a_win_through_sdk_contract():
    """The full conduit: multi-cell Sokoban played to WIN via TbtPolicy.choose_action/is_done only — the exact
    calls the real SDK makes — confirming frame→perception→planner→action(name)→enum round-trips correctly."""
    env = Environment(Sokoban(levels=MULTICELL_LEVELS))
    frame = _drive(env, TbtPolicy(_sokoban_agent()))
    assert frame.state == GameState.WIN, f"conduit did not reach WIN: {frame.state}"


def test_not_played_yields_reset():
    """A NOT_PLAYED game must be RESET to start — the SDK lifecycle the policy owns (the agent never sees it)."""
    class _F:
        state = GameState.NOT_PLAYED
        frame = [[[0]]]
        levels_completed = 0

    name, coords = TbtPolicy(_sokoban_agent()).choose_action([], _F())
    assert name == "RESET" and coords is None


def test_is_done_only_on_win():
    """Stop on WIN; keep playing on GAME_OVER (the level reloads) — faithful to the SDK Random template."""
    pol = TbtPolicy(_sokoban_agent())

    class _F:
        def __init__(self, s):
            self.state = s

    assert pol.is_done([], _F(GameState.WIN))
    assert not pol.is_done([], _F(GameState.NOT_FINISHED))
    assert not pol.is_done([], _F(GameState.GAME_OVER))


def test_to_obs_translates_real_field_names():
    """`levels_completed` (the real SDK's field) maps to our perception's `level`, the primary grid is coerced,
    and the state maps onto our GameState — the duck-typing that lets one path serve both SDK and replica."""
    class _F:
        state = GameState.NOT_FINISHED
        levels_completed = 2
        frame = [[[0, 1], [2, 3]]]            # the real SDK frame: a list of grids (here one 2x2)

    obs = _to_obs(_F())
    assert obs.level == 2
    assert obs.grid == [[0, 1], [2, 3]]
    assert obs.state == GameState.NOT_FINISHED


def test_primary_grid_handles_single_grid_and_stack():
    """`frame` is 1..N grids; the primary is the last. Distinguish a bare 2-D grid from a stack of grids by depth."""
    assert _primary_grid([[0, 1], [2, 3]]) == [[0, 1], [2, 3]]        # a single 2-D grid
    assert _primary_grid([[[9, 9]], [[0, 1]]]) == [[0, 1]]            # a stack → take the last grid
