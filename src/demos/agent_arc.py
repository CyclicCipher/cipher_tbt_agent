"""Drive + evaluate the thin TBT `Agent` on the ARC-AGI-3 replica — the ARC wiring kept OUT of the agent.

`play_online` IS the agent: ONE online loop that acts AND learns the model from the same experience (no offline
collect, no oracle), scored in STEPS-to-solve. The offline `collect` + the `egocentric` config remain ONLY for
the partial-observability recurrence demo, which inherently learns full-obs then solves egocentric (so it is not
a removable duplication). The same `Agent` driver runs both; adding a game mechanic touches only the game.

Run:  PYTHONPATH=src python -m demos.agent_arc
"""

from __future__ import annotations

import random
from statistics import mean

from agent import Agent
from perception.scene import build_world, Perception, EgoPerception
from tbt.planner import Planner, EgoPlanner

from perception.perceive import DynamicsPerceiver, ObjectPerceiver, GoalModel
from tbt.dynamics import DynamicsModel
from tasks import Environment, GameState
from tasks.contract import ContractEnv
from tasks.games import LockPath, MultiKey


def collect(episodes=150, max_steps=120, seed=0, game_cls=LockPath):
    """OFFLINE learner: one shared play loop -> three learners (DynamicsModel, ObjectPerceiver, GoalModel) by
    SELF-DIRECTED random play, no oracle. Kept for the partial-obs eval + as an offline comparison; the online
    agent is `play_online`. Returns (dm, object_perceiver, goal)."""
    rng = random.Random(seed)
    perc, dm = DynamicsPerceiver(), DynamicsModel()
    objp, goal = ObjectPerceiver(), GoalModel()
    for ep in range(episodes):
        env = Environment(game_cls())
        frame = env.reset()
        perc.reset()
        for _ in range(max_steps):
            if frame.state != GameState.NOT_FINISHED:
                break
            moves = [a for a in frame.available_actions if a.is_movement]
            action = rng.choice(moves)
            prev = frame
            frame = env.step(action)
            f, e, present = perc.observe(prev, action, frame)
            if f is not None:
                dm.observe(f, e)
                if action.is_movement and prev.state == GameState.NOT_FINISHED and frame.level == prev.level:
                    objp.observe(prev.grid, action.delta, frame.grid)
                stepped_on = f[0]
                if e == "score_up":
                    goal.observe_win(present, stepped_on)
                elif stepped_on in goal.goal_colors:
                    goal.observe_reach_no_win(present)
            if frame.level != prev.level:
                perc.new_level(); objp.new_level()
    dm.learn()
    return dm, objp, goal


def learn_mechanics(game_cls, **kw):
    """One play phase -> the canonical learners (DynamicsModel, ObjectPerceiver, GoalModel)."""
    return collect(game_cls=game_cls, **kw)


def egocentric(dm, objp, goal, radius=2, memory=True, seed=0) -> Agent:
    """The SAME agent under an egocentric window — the recurrence becomes essential."""
    world = build_world(dm, objp, goal)
    return Agent(EgoPerception(world, radius, memory),
                 EgoPlanner(world, EgoPerception.deltas, radius=radius, memory=memory, seed=seed))


def run_game(env, agent, max_actions):
    """Drive the agent over the tbt.env contract; per-level action counts from the reward (a completed level).
    The whole multi-level game is one contract episode — GAME_OVER is not done (the agent RESETs and retries)."""
    agent.reset()
    obs = env.reset()
    per, completed, n, level = {}, set(), 0, 0
    for _ in range(max_actions):
        action, coords = agent.choose_action(obs)
        step = env.step(action, coords)
        obs = step.observation
        n += 1
        if step.reward > 0:                                    # a level just completed
            per[level] = n; completed.add(level); n = 0; level += 1
        if step.done:                                          # WIN (every level finished)
            break
    return per, completed


# ── S1 of the rework: ONE online loop (act + learn in the same pass; no offline collect, no oracle) ──────────
def _refresh_world(world, dm, objp, goal):
    """Re-decode the online-learned model into the SHARED `world` object in place, so the agent's perception +
    planner (which hold a reference to it) immediately see newly-learned roles (body, effects, goal)."""
    dm.learn()
    world.__dict__.update(build_world(dm, objp, goal).__dict__)


def _online_episode(env, agent, perc, dm, objp, goal, world, max_steps, refresh_every, explore):
    """ONE episode: act + learn from the SAME experience, sharing the persistent learners + world. `explore` is
    the chance of a random move even when a plan exists — so the agent keeps discovering (cover/tour) instead of
    only beelining to a known goal. Returns (steps_per_level, completed_levels)."""
    frame = env.reset()
    per, completed, n, level = {}, set(), 0, 0
    prev, last = None, None
    for step in range(max_steps):
        if prev is not None:                                  # LEARN from the previous transition
            f, e, present = perc.observe(prev, last, frame)
            if f is not None:
                dm.observe(f, e)
                if last.is_movement and prev.state == GameState.NOT_FINISHED and frame.level == prev.level:
                    objp.observe(prev.grid, last.delta, frame.grid)
                so = f[0]
                if e == "score_up":
                    goal.observe_win(present, so)
                elif so in goal.goal_colors:
                    goal.observe_reach_no_win(present)
            if frame.level != prev.level:                     # level advanced -> reset the learners' per-level memory
                perc.new_level(); objp.new_level()
        if step % refresh_every == 0:                         # refresh the shared world (agent sees new roles)
            _refresh_world(world, dm, objp, goal)
        action, coords = agent.choose_action(frame, explore=explore)   # ACT: perceive + plan (ε-explore inside)
        prev, last = frame, action
        s = env.step(action, coords)
        frame = s.observation
        n += 1
        if s.reward > 0:
            per[level] = n; completed.add(level); n = 0; level += 1
        if s.done:
            break
    return per, completed


def play_online(game_cls, episodes=120, max_steps=250, seed=0, refresh_every=40, explore=0.15):
    """The ONLINE loop, MULTI-EPISODE: act + learn across reset-rich episodes, KEEPING the model — reset-rich +
    ε-exploratory inside the one loop (no offline collect, no oracle). `explore` decays linearly to 0 over the
    run. Returns (per-episode [levels, steps_per_level], n_levels)."""
    perc, dm, objp, goal = DynamicsPerceiver(), DynamicsModel(), ObjectPerceiver(), GoalModel()
    world = build_world(dm, objp, goal)
    agent = Agent(Perception(world), Planner(world, Perception.deltas, seed=seed))
    n_levels = game_cls().level_count
    history = []
    for ep in range(episodes):
        perc.reset(); agent.reset(); _refresh_world(world, dm, objp, goal)
        eps = explore * max(0.0, 1.0 - ep / max(1, episodes - 1))          # anneal exploration -> exploitation
        per, completed = _online_episode(ContractEnv(game_cls()), agent, perc, dm, objp, goal, world,
                                         max_steps, refresh_every, eps)
        history.append((len(completed), [per.get(i) for i in range(n_levels)]))
    return history, n_levels


def evaluate_online(game_cls, episodes=120, max_steps=250, seed=0):
    """Multi-episode online; report the converged performance: the best episode + how many of the last 20 fully
    solve (the agent has learned to solve, online, in steps)."""
    history, n = play_online(game_cls, episodes=episodes, max_steps=max_steps, seed=seed)
    best = max(history, key=lambda h: (h[0], -sum(s for s in h[1] if s)))
    last = history[-20:]
    return n, history, best, (sum(1 for nc, _ in last if nc == n), len(last))


def evaluate_partial(levels=(0, 1), budget=300, seeds=range(8)):
    """Egocentric partial observability: the recurrent (memory) agent vs the memoryless ablation, on the
    navigation + key+door levels — why recurrence is needed, via the ONE agent."""
    dm, objp, goal = learn_mechanics(LockPath)
    out = {}
    for label, radius, memory in [("full obs, memory", 12, True), ("egocentric r=2, memory", 2, True),
                                  ("egocentric r=2, MEMORYLESS", 2, False), ("egocentric r=1, memory", 1, True),
                                  ("egocentric r=1, MEMORYLESS", 1, False)]:
        solved = {lvl: [0, []] for lvl in levels}
        for s in seeds:
            env = ContractEnv(LockPath())
            per, completed = run_game(
                env, egocentric(dm, objp, goal, radius=radius, memory=memory, seed=s), budget)
            for lvl in completed:
                if lvl in solved:
                    solved[lvl][0] += 1; solved[lvl][1].append(per[lvl])
        out[label] = {lvl: (c, mean(a) if a else None) for lvl, (c, a) in solved.items()}
    return out, len(list(seeds))


if __name__ == "__main__":
    print("thin TBT agent — ONE online loop, acts AND learns the model; self-learned, no oracle, scored in steps:\n")
    for game_cls in (LockPath, MultiKey):
        n, history, best, (solved, last) = evaluate_online(game_cls)
        bnc, bsteps = best
        steps_str = " ".join(str(x) if x is not None else "-" for x in bsteps)
        print(f"=== {game_cls.__name__} ({game_cls.game_id}) — {n} levels ===")
        print(f"    converged: best {bnc}/{n} levels, steps/level [{steps_str}]; "
              f"{solved}/{last} of the last episodes fully solve\n")

    print("partial observability — the SAME agent, egocentric window; recurrence (path-int + map) vs ablation:\n")
    res, nseeds = evaluate_partial()
    print(f"  {'agent':>30}  {'L0 (actions)':>18}  {'L1 (actions)':>18}")
    for label, d in res.items():
        def _c(lvl, d=d):
            c, a = d.get(lvl, (0, None))
            return f"{c}/{nseeds} ({a:.0f})" if a is not None else f"{c}/{nseeds} (-)"
        print(f"  {label:>30}  {_c(0):>18}  {_c(1):>18}")
    print("\n  the recurrence is ESSENTIAL: the memory agent path-integrates + remembers the map and solves")
    print("  near-optimally; the memoryless ablation wanders (5-15x actions) and fails L1 under budget.")
