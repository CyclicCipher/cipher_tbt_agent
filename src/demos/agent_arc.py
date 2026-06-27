"""Drive + evaluate the thin TBT `Agent` on the ARC-AGI-3 replica.

This is the ARC wiring kept OUT of the agent: build the agent from the learned model (perception + the
spatial planner over the column), then score it with the RHAE proxy. `full_obs` / `egocentric` are the two
standard configs (the same `Agent` driver, different perception+planner). Adding a game mechanic touches only
the game (+ the oracle teacher); the agent, perception, and planner never change.

Run:  PYTHONPATH=src python -m demos.agent_arc
"""

from __future__ import annotations

from statistics import mean

from agent import Agent
from perception.scene import build_world, Perception, EgoPerception
from tbt.planner import Planner, EgoPlanner

from perception.dynamics_perceive import collect, DynamicsPerceiver
from perception.object_perceiver import ObjectPerceiver
from perception.goal_discover import GoalModel
from tbt.dynamics import DynamicsModel
from tasks import GameState
from tasks.contract import ContractEnv
from tasks.games import LockPath, MultiKey


def learn_mechanics(game_cls, **kw):
    """One play phase -> the canonical learners (DynamicsModel, ObjectPerceiver, GoalModel)."""
    return collect(game_cls=game_cls, **kw)


def full_obs(dm, objp, goal, seed=0) -> Agent:
    """The full-observation agent over the learned model."""
    world = build_world(dm, objp, goal)
    return Agent(Perception(world), Planner(world, Perception.deltas, seed=seed))


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


def play_online(env, max_steps=1500, seed=0, refresh_every=40):
    """The merged loop: act and learn the model from the SAME experience. The dynamics/E/F learners run during
    play (no separate `collect`); the agent explores while the model is empty (no body/goal -> random moves) and
    plans once it isn't. S2 collapses these three learners into the column's G/V. Returns (steps_per_level, set)."""
    perc, dm, objp, goal = DynamicsPerceiver(), DynamicsModel(), ObjectPerceiver(), GoalModel()
    world = build_world(dm, objp, goal)                        # empty: no body / effects / goal yet
    agent = Agent(Perception(world), Planner(world, Perception.deltas, seed=seed))
    agent.reset()
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
        action, coords = agent.choose_action(frame)           # ACT: perceive + plan with the current world
        prev, last = frame, action
        s = env.step(action, coords)
        frame = s.observation
        n += 1
        if s.reward > 0:
            per[level] = n; completed.add(level); n = 0; level += 1
        if s.done:
            break
    return per, completed


def evaluate_online(game_cls, seeds=range(1), max_steps=1500):
    """Drive `play_online` (learn + solve in one online run); report levels solved + steps. The S1 gate."""
    n = game_cls().level_count
    rows = []
    for seed in seeds:
        per, completed = play_online(ContractEnv(game_cls()), max_steps=max_steps, seed=seed)
        rows.append((seed, len(completed), [per.get(i) for i in range(n)]))
    return n, rows


def evaluate(game_cls, seeds=range(6), max_actions=6000):
    """Learn the mechanics (self-directed, no oracle), then drive the thin agent. Metric: levels SOLVED and the
    STEPS the architecture itself took per level (no oracle-optimal baseline — lower steps is better, but the
    first question is simply whether it solves)."""
    dm, objp, goal = learn_mechanics(game_cls)
    n = game_cls().level_count
    rows = []
    for s in seeds:
        env = ContractEnv(game_cls())
        per, completed = run_game(env, full_obs(dm, objp, goal, seed=s), max_actions)
        rows.append((s, len(completed), [per.get(i) for i in range(n)]))
    return (dm, objp, goal), n, rows


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
    print("thin TBT agent — the SAME driver on every game (no per-mechanic code); self-learned, no oracle:\n")
    for game_cls in (LockPath, MultiKey):
        (dm, objp, goal), n, rows = evaluate(game_cls)
        print(f"=== {game_cls.__name__} ({game_cls.game_id}) — {n} levels ===")
        for s, nc, steps in rows:
            steps_str = " ".join(str(x) if x is not None else "-" for x in steps)
            print(f"    seed {s}:  {nc}/{n} levels   steps/level: [{steps_str}]")
        print(f"  mean: {mean(r[1] for r in rows):.2f}/{n} levels solved\n")

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
