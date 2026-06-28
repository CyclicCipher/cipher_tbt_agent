"""F's cold-start — the ONE agent LEARNS its world model from self-directed play, then solves the game.

ARC-AGI-3 gives no goal and no win condition. Here the agent starts from an EMPTY world model and discovers it by
playing: the GOAL colour from the sparse score, the BODY from the efference copy, the PUSHABLE / BLOCKING roles
from motion, the conditional EFFECTS (a key opens a door) as a residual predicate, and the HAZARD by dying. No
injected roles, no oracle, no hand-coded mechanics.

It is the SAME agent as everywhere else — `tbt.agent.Agent.explore_and_learn` runs one online loop that ACTS
(choose_action) and LEARNS (a perception-side `WorldLearner`) from the same experience, re-decoding the roles into
the shared world IN PLACE so the agent plans with them live. ε-exploration anneals to 0 (discover → exploit). The
control loop is the multi-column Neocortex (doors become affordance sub-goals; blocks are pushed factored).

Run:  PYTHONPATH=src python -m demos.cold_start
"""

from __future__ import annotations

from perception.control import NeocortexPlanner
from perception.learn import WorldLearner
from perception.scene import Perception

from tasks import Environment
from tasks.games import LockPath, MultiKey

from tbt.agent import Agent


def cold_start(game_cls, episodes=80, max_steps=200, explore=0.25, seed=0):
    """Learn the world model from scratch by self-directed play, then evaluate the learned policy. Returns
    (learned world, levels-solved-per-episode, eval outcome)."""
    learner = WorldLearner()
    agent = Agent(Perception(learner.world), NeocortexPlanner(learner.world, seed=seed))
    history = agent.explore_and_learn(Environment(game_cls()), learner,
                                      episodes=episodes, max_steps=max_steps, explore=explore)
    outcome = agent.play(Environment(game_cls()), max_steps=3000)
    return learner.world, history, outcome


if __name__ == "__main__":
    print("F's cold-start — the ONE agent learns the world model from self-directed play (no injected roles, no")
    print("oracle), then solves the game. Every role is DISCOVERED: the goal from the sparse score; the body,")
    print("the pushable, the key->door effects and the hazard from watching.\n")
    for game_cls in (LockPath, MultiKey):
        world, history, outcome = cold_start(game_cls)
        n = game_cls().level_count
        print(f"=== {game_cls.__name__} ({game_cls.game_id}) — {n} levels ===")
        print(f"    learned roles: body={world.body}  pushable={world.pushable}  effects={world.effects}")
        print(f"                   goal={world.goal_colors}  required_absent={world.required_absent}  "
              f"death={world.death}")
        print(f"    convergence (levels solved per episode, last 10): {history[-10:]}")
        tag = "WIN" if outcome.won else "..."
        print(f"    eval with the LEARNED model: [{tag}] {outcome.levels}/{n} levels in {outcome.actions} "
              f"actions\n")
    print("the goal is found ONCE from the score, then the agent PLANS through the learned model to re-reach it")
    print("(no dense reward) — autonomous skill acquisition, the same agent for both games (no per-mechanic code).")
