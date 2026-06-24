"""RHAE-style scoring (proxy) — Relative Human Action Efficiency.

Real ARC-AGI-3 (docs.arcprize.org/methodology): per *completed* level, score
`(human_baseline / agent_actions)^2` (capped at 1), where the human baseline is the
*second-best* human's action count; failed levels score 0; mean over levels; and an
action budget of 5x the human *median* per level terminates a run.

We have no human data for the replica, so we use the BFS ORACLE's *optimal* action
count per level as the baseline proxy. This is HARSHER than the real metric (the
oracle is optimal; humans — like our agent — pay actions to learn the game on the
fly), so this number is a conservative lower bound on what RHAE-with-humans would give.
"""

from __future__ import annotations

from statistics import mean
from typing import Callable, List

from arc_agi_3 import Environment, GameState
from arc_agi_3.games import LockPath
from arc_agi_3.oracle import solve_level

from .agent import WorldModelAgent


def oracle_optimal(make_game: Callable) -> List:
    game = make_game()
    opt = []
    for lvl in range(game.level_count):
        game.load_level(lvl)
        path = solve_level(game)
        opt.append(len(path) if path else None)
    return opt


def per_level_actions(env: Environment, agent, max_actions: int):
    """Actions the agent spent on each level it completed (deaths/exploration included)."""
    agent.reset()
    f = env.reset()
    per, completed = {}, set()
    last, cur = 0, f.level
    for _ in range(max_actions):
        a, c = agent.choose_action(f)
        f = env.step(a, c)
        if f.level != cur:                               # finished level `cur`
            per[cur] = f.action_counter - last
            completed.add(cur)
            last, cur = f.action_counter, f.level
        if f.state == GameState.WIN:                     # finished the final level `cur`
            per[cur] = f.action_counter - last
            completed.add(cur)
            break
    return per, completed


def rhae(make_game: Callable, agent_factory: Callable, seeds, max_actions: int = 6000):
    opt = oracle_optimal(make_game)
    n = len(opt)
    per_seed = []
    for s in seeds:
        env = Environment(make_game())
        per, completed = per_level_actions(env, agent_factory(s), max_actions)
        lvl = [min(1.0, (opt[i] / per[i]) ** 2) if (i in completed and opt[i] and per.get(i))
               else 0.0 for i in range(n)]
        per_seed.append(mean(lvl))
    return per_seed, opt


if __name__ == "__main__":
    seeds = list(range(12))
    scores, opt = rhae(LockPath, lambda s: WorldModelAgent(seed=s), seeds)
    print("oracle-optimal actions / level:", opt)
    print("RHAE-proxy per seed (%):", [round(100 * x, 1) for x in scores])
    print(f"mean RHAE-proxy: {100 * mean(scores):.1f}%   "
          f"(proxy uses the optimal oracle as baseline — harsher than human)")
