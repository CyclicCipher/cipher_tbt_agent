"""The TBT agent — a THIN env-driver over a cortical column.

This file is deliberately tiny and imports nothing task-specific. The agent does ONLY the part the column
should not know about: drive an environment — read each observation through a `perception` (the senses), hand
the resulting scene to a `planner` (the brain: the SR-frame map + recurrence + subgoal value, all in `tbt/`),
and emit the action. Perception owns every grid/colour/action detail; the planner owns the model + planning.
Both are injected, so the SAME driver runs full-observation ARC games, the egocentric partial-observation
variant, or any future task whose perception+planner implement the contract:

  perception.read(obs) -> Percept(scene, new_level, terminal);  perception.to_action(move);  .reset_action
  planner.act(scene) -> move index;  planner.reset() / new_level() / on_death()

If this agent would crash on a non-grid environment, it is wrong by construction (feedback_thin_shell_agent).
The ARC wiring (build the perception+planner from the learned model, evaluate) lives in `demos/agent_arc.py`.
"""

from __future__ import annotations


class Agent:
    def __init__(self, perception, planner):
        self.perception = perception
        self.planner = planner

    def reset(self):
        self.planner.reset()
        self.perception.reset()

    def choose_action(self, observation):
        p = self.perception.read(observation)
        if p.terminal:                                         # GAME_OVER -> the env reloads the level
            self.planner.on_death()
            return self.perception.reset_action, None
        if p.new_level:
            self.planner.new_level()
        return self.perception.to_action(self.planner.act(p.scene)), None
