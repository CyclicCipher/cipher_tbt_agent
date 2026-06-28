"""The TBT agent — THE one agent, a thin env-driver over a cortical-column planner.

Deliberately tiny and task-agnostic. The agent does ONLY the part the column should not know about: drive an
environment — read each observation through a `perception` (the senses), hand the resulting scene to a
`planner` (the brain: the SR-frame map + recurrence + the Neocortex control loop, all in `tbt/`), and emit the
action. Perception owns every grid/colour/action detail; the planner owns the model + planning. Both are
INJECTED, so the SAME agent runs full-observation ARC games, the egocentric partial-observation variant, or any
future task whose perception + planner implement the contract:

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
