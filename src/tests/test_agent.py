"""The ONE game agent (tbt.agent.Agent) playing through perception + the Neocortex control-loop planner.

Validates RECONNECT S3 — the factored relational PUSH (egocentric ⊗ absolute, one mover at a time, never the
agent × all-blocks joint) — end to end through the single agent, on Sokoban (push N blocks onto N pads, then
reach the goal). The world-model roles are INJECTED here; F's cold-start that would DISCOVER them from the score
is RECONNECT S2 (next). This test locks in that the Neocortex, wired as the agent's planner, solves the multi-pad
cover loop through the one play(env) loop — no per-game harness.
"""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from perception.control import NeocortexPlanner  # noqa: E402
from perception.learn import WorldLearner  # noqa: E402
from perception.scene import Perception, WorldModel  # noqa: E402
from tasks import Environment  # noqa: E402
from tasks.games import CollectAll, LockPath, Sokoban, Toggle  # noqa: E402
from tasks.games.collectall import C_AGENT as CA_AGENT, C_ITEM  # noqa: E402
from tasks.games.collectall import C_WALL as CA_WALL  # noqa: E402
from tasks.games.collectall import _LEVELS as CA_LEVELS  # noqa: E402
from tasks.games.lockpath import C_DOOR, C_HAZARD, C_KEY  # noqa: E402
from tasks.games.lockpath import _LEVELS as LP_LEVELS  # noqa: E402
from tasks.games.sokoban import C_AGENT, C_BLOCK, C_GOAL, C_PAD, C_WALL, _LEVELS  # noqa: E402
from tasks.games.toggle import C_AGENT as TG_AGENT, C_DOOR as TG_DOOR, C_GOAL as TG_GOAL  # noqa: E402
from tasks.games.toggle import C_SWITCH, C_WALL as TG_WALL  # noqa: E402
from tasks.games.toggle import _LEVELS as TG_LEVELS  # noqa: E402
from tbt.agent import Agent  # noqa: E402
from tbt.column import CorticalColumn  # noqa: E402


def _dyn_from_world(world) -> CorticalColumn:
    """A dynamics column whose learned rules match the injected world — what F's cold-start would DISCOVER. The
    planner reads dynamics ONLY from this column's `predict_effect` (Step C1), never from the world's effects/adds:
    a contact that removes a colour → that colour 'gone'; a contact that BOTH removes and adds it → the
    context-conditioned TOGGLE (door present → gone, absent → appeared); a death colour → 'death'. Features are
    `(stepped_on,) + presence-bits[0..15]`, so a colour's presence sits at index 1+colour."""
    rules = []
    for c in sorted(world.death):
        rules.append((lambda f, c=c: f[0] == c, f"c0=={c}", "death"))
    for trig in sorted(world.effects):
        for col in sorted(world.effects[trig]):
            if col in world.adds.get(trig, set()):
                rules.append((lambda f, t=trig, i=1 + col: f[0] == t and f[i] == 1, "", f"color_{col}_gone"))
                rules.append((lambda f, t=trig, i=1 + col: f[0] == t and f[i] == 0, "", f"color_{col}_appeared"))
            else:
                rules.append((lambda f, t=trig: f[0] == t, f"c0=={trig}", f"color_{col}_gone"))
    dm = CorticalColumn(n_entities=1)
    dm.dyn_rules = rules
    return dm


def _sokoban_world() -> WorldModel:
    """Sokoban's roles (INJECTED — RECONNECT S2 will discover these from the sparse score): the agent body, the
    block as the only pushable, walls block, the goal colour is the reach target, an uncovered pad is the
    required-absent (cover) term. No effects/doors/hazards."""
    return WorldModel(
        body=C_AGENT, pushable={C_BLOCK}, blocking={C_WALL}, death=set(),
        effects={}, adds={}, harmful=set(), goal_colors={C_GOAL}, required_absent={C_PAD},
    )


def _agent() -> Agent:
    world = _sokoban_world()
    return Agent(Perception(world), NeocortexPlanner(world, _dyn_from_world(world), seed=0))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_neocortex_agent_solves_each_sokoban_level(level):
    """Each level in isolation (a one-level game), so a failure localises to that mechanic depth."""
    env = Environment(Sokoban(levels=[_LEVELS[level]]))
    out = _agent().play(env, max_steps=600)
    assert out.won, f"Sokoban L{level} not solved: {out}"


def test_neocortex_agent_solves_full_sokoban():
    """All three levels in sequence through the one play loop — win == cleared every level."""
    out = _agent().play(Environment(Sokoban()), max_steps=2000)
    assert out.won and out.levels == len(_LEVELS), out


# ── RECONNECT S2a: doors emerge as sub-goals (the affordance) ─────────────────────────────────────────────
def _lockpath_world() -> WorldModel:
    """LockPath's roles (INJECTED — S2's cold-start will discover them): + the key→door effect (reaching the key
    opens the door) and the hazard as death. The Neocortex turns the key→door effect into an affordance sub-goal
    (reach the key first), with no hardcoded key/door knowledge — it just sees 'this trigger clears that blocker'."""
    return WorldModel(
        body=C_AGENT, pushable={C_BLOCK}, blocking={C_WALL}, death={C_HAZARD},
        effects={C_KEY: {C_DOOR}}, adds={}, harmful=set(), goal_colors={C_GOAL}, required_absent={C_PAD},
    )


def _lp_agent() -> Agent:
    world = _lockpath_world()
    return Agent(Perception(world), NeocortexPlanner(world, _dyn_from_world(world), seed=0))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_neocortex_agent_solves_each_lockpath_level(level):
    """L0 navigation, L1 key+door (the affordance), L2 block+pad — each in isolation through the one agent."""
    env = Environment(LockPath(levels=[LP_LEVELS[level]]))
    out = _lp_agent().play(env, max_steps=800)
    assert out.won, f"LockPath L{level} not solved: {out}"


def test_neocortex_agent_solves_full_lockpath():
    """All four levels in sequence (L3 composes key+door AND block+pad with a hazard) through the one play loop."""
    out = _lp_agent().play(Environment(LockPath()), max_steps=3000)
    assert out.won and out.levels == len(LP_LEVELS), out


# ── RECONNECT S2b: F's cold-start — learn the roles, no injection ─────────────────────────────────────────
def test_cold_start_learns_goal_from_score_and_plans():
    """From an EMPTY world the agent learns the body (efference copy) and the GOAL colour (from the sparse score)
    by self-directed play, then PLANS to it — the core autonomy claim, NO injected roles. A tiny L0 budget keeps
    this fast and deterministic (seed=0); the full multi-mechanic convergence (LockPath 4/4, MultiKey 2/2) is the
    heavier `demos/cold_start.py`."""
    learner = WorldLearner()
    agent = Agent(Perception(learner.world), NeocortexPlanner(learner.world, learner.dm, seed=0))
    agent.explore_and_learn(Environment(LockPath(levels=[LP_LEVELS[0]])), learner,
                            episodes=20, max_steps=120, explore=0.3, refresh_every=20)
    assert learner.world.body == C_AGENT                  # learned the body by the efference copy
    assert C_GOAL in learner.world.goal_colors            # learned the goal from the score (the cold-start claim)
    out = agent.play(Environment(LockPath(levels=[LP_LEVELS[0]])), max_steps=400)
    assert out.won, out                                   # and now PLANS to the learned goal (not random wandering)


# ── Phase-2 Step B: a structurally different mechanic the TYPED planner could NOT do ───────────────────────
def _collectall_world() -> WorldModel:
    """CollectAll's roles (INJECTED — the cold-start discovers them): the item is a CONSUMABLE required-absent and
    there is NO goal cell. There is no mover, so the rollout achiever reaches each item with the AGENT (collect) —
    the same achiever as cover/reach, the consumable falling out of signed value, with no new sub-goal type. This
    is the game the old typed (cover/reach) planner could not express; it is the proof the rollout generalises."""
    return WorldModel(
        body=CA_AGENT, pushable=set(), blocking={CA_WALL}, death=set(),
        effects={}, adds={}, harmful=set(), goal_colors=set(), required_absent={C_ITEM},
    )


def _ca_agent() -> Agent:
    world = _collectall_world()
    return Agent(Perception(world), NeocortexPlanner(world, _dyn_from_world(world), seed=0))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_neocortex_agent_solves_each_collectall_level(level):
    """Each level in isolation — collect every item (a multi-target tour, no goal cell) through the one agent."""
    env = Environment(CollectAll(levels=[CA_LEVELS[level]]))
    out = _ca_agent().play(env, max_steps=800)
    assert out.won, f"CollectAll L{level} not solved: {out}"


def test_neocortex_agent_solves_full_collectall():
    """All three levels in sequence through the one play loop — win == every item collected on every level."""
    out = _ca_agent().play(Environment(CollectAll()), max_steps=2000)
    assert out.won and out.levels == len(CA_LEVELS), out


# ── Phase-2 Step B: the TOGGLE — a REVERSIBLE effect (the switch flips the door), the adversarial mechanic ──
def _toggle_world() -> WorldModel:
    """Toggle's roles (INJECTED — the cold-start learns exactly this: the switch BOTH removes AND adds the door
    colour, i.e. it FLIPS the door open↔closed). No `harmful` role is read by the planner — avoiding the switch
    EMERGES because the forward model rolls the flip and signed value sees that closing the door blocks the goal.
    The door starts open (invisible); its position is discovered when a toggle makes it appear, then the planner
    deliberately reopens + passes (no flailing). This is the mechanic that breaks a one-way `color_gone` model."""
    return WorldModel(
        body=TG_AGENT, pushable=set(), blocking={TG_WALL}, death=set(),
        effects={C_SWITCH: {TG_DOOR}}, adds={C_SWITCH: {TG_DOOR}}, harmful=set(),
        goal_colors={TG_GOAL}, required_absent=set(),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_neocortex_agent_solves_toggle(seed):
    """The switch flips the door; the agent must reason through the reversible effect (deterministically, no luck —
    a few seeds guard against a flailing 'win'). Solved by the rolled flip + signed value, not a 'harmful' role."""
    world = _toggle_world()
    agent = Agent(Perception(world), NeocortexPlanner(world, _dyn_from_world(world), seed=seed))
    out = agent.play(Environment(Toggle(levels=[TG_LEVELS[0]])), max_steps=400)
    assert out.won, f"Toggle not solved (seed {seed}): {out}"
