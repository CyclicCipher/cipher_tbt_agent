"""The TBT agent — THE one agent, a thin env-driver over a cortical-column planner.

Deliberately tiny and task-agnostic. The agent does ONLY the part the column should not know about: drive an
environment — read each observation through a `perception` (the senses), hand the resulting scene to a
`planner` (the brain: the SR-frame map + recurrence + the Neocortex control loop, all in `tbt/`), and emit the
action. Perception owns every grid/colour/action detail; the planner owns the model + planning. Both are
INJECTED, so the SAME agent runs any task whose perception + planner implement the contract:

  perception.read(obs) -> Percept(scene, new_level, terminal);  perception.to_action(move);  .reset_action
  planner.act(scene, explore) -> move index;  planner.reset() / new_level() / on_death()

If this agent would crash on a non-grid environment, it is wrong by construction (feedback_thin_shell_agent).
The `play(env)` loop lives here too, so this file — not a harness — is where the agent lives: a transformer's
capabilities live in the transformer, not a per-application shim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Outcome:
    """The result of playing one game: did we win, how many levels cleared, how many actions it cost."""
    won: bool
    levels: int
    actions: int


class Agent:
    def __init__(self, perception, planner):
        self.perception = perception
        self.planner = planner

    def reset(self):
        self.planner.reset()
        self.perception.reset()

    def choose_action(self, observation, explore=0.0):
        p = self.perception.read(observation)
        if p.terminal:                                         # GAME_OVER -> the env reloads the level
            self.planner.on_death()
            return self.perception.reset_action, None
        if p.new_level:
            self.planner.new_level()
        return self.perception.to_action(self.planner.act(p.scene, explore=explore)), None

    def play(self, env, max_steps=2000, explore=0.0):
        """THE play loop — drive `env` to a win (or the action budget), in the agent, NOT a harness. The agent
        self-heals GAME_OVER inside `choose_action` (it emits the reset action; the env reloads the level), so the
        loop stops only on a win or exhaustion. Uses a generic env protocol (reset/step + frame.is_win /
        .action_counter / .score) — no task import, so the same loop runs any environment perception can read."""
        self.reset()
        frame = env.reset()
        while frame.action_counter < max_steps and not frame.is_win():
            action, coords = self.choose_action(frame, explore=explore)
            frame = env.step(action, coords)
        return Outcome(won=frame.is_win(), levels=frame.score, actions=frame.action_counter)

    def explore_and_learn(self, env, learner, episodes=120, max_steps=250, explore=0.2, refresh_every=40):
        """THE act-and-learn play loop — cold-start the world model by self-directed play, no injected roles, no
        oracle. One online loop that ACTS (choose_action) and LEARNS (learner.observe) from the SAME experience;
        the learner re-decodes the roles into the shared world in place, so this agent plans with them live. ε
        anneals to 0 over the run (discover → exploit). Generic: only frame.score/level/is_win + the duck-typed
        learner — no task import. Returns the per-episode count of levels solved (convergence is it rising)."""
        history = []
        for ep in range(episodes):
            self.reset()
            learner.reset()
            learner.refresh()
            eps = explore * max(0.0, 1.0 - ep / max(1, episodes - 1))
            frame = env.reset()
            solved = set()
            for step in range(max_steps):
                if step % refresh_every == 0:                  # the agent sees newly-learned roles
                    learner.refresh()
                action, _ = self.choose_action(frame, explore=eps)   # ACT (ε-explore inside)
                nxt = env.step(action)
                learner.observe(frame, action, nxt)            # LEARN from THIS transition — incl. the WINNING one
                if nxt.score > frame.score:                    # (a win/level-up is where GoalModel learns the goal,
                    solved.add(frame.level)                    #  so it must be observed BEFORE the break below)
                if nxt.level != frame.level:                   # a level completed → reset per-level perceiver memory
                    learner.new_level()
                frame = nxt
                if frame.is_win():
                    break
            history.append(len(solved))
        return history
