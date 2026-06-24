"""Phase-2 replica test — the prior-minimal agent (agent/wm2). It is the Phase-1 WorldModelAgent
with the SEEDED single-cell agency prior removed (agency discovered instead), reusing the rest of the
machinery. Regression: removing that prior must not break the win."""

from arc_agi_3 import Environment, GameState
from arc_agi_3.games import LockPath

from agent.wm2.agent import HierarchicalPlanAgent, VolumeAgent
from agent.wm2.world_model import DiscoveredWorldModel


def test_starts_with_no_seeded_agent():
    wm = DiscoveredWorldModel()
    assert wm.agent_color is None              # not seeded — must be discovered from transitions


def test_discovered_agency_still_wins_the_full_game():
    env = Environment(LockPath())
    agent = VolumeAgent(seed=6)
    f = env.reset()
    for _ in range(6000):
        a, c = agent.choose_action(f)
        f = env.step(a, c)
        if f.state == GameState.WIN:
            break
    assert f.state == GameState.WIN and f.level == 3
    assert agent.wm.agent_color == 2           # the agent colour was DISCOVERED, not given


def test_hierarchical_planner_can_win_by_composing_macros():
    """The hierarchical planner (macros = discovered edges, composed in prerequisite order, refined
    by navigation) wins when discovery + recovery cooperate — seed 2 solves the full game quickly.
    Locks in that the macro composition itself is correct; the residual gap is the control policy
    (when to plan / explore / recover), not the planner. See docs/phase2/REPLICA_TEST.md."""
    env = Environment(LockPath())
    agent = HierarchicalPlanAgent(seed=2)
    f = env.reset()
    for _ in range(4000):
        a, c = agent.choose_action(f)
        f = env.step(a, c)
        if f.state == GameState.WIN:
            break
    assert f.state == GameState.WIN and f.level == 3
