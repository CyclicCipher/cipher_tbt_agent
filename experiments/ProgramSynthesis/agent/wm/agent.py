"""WorldModelAgent — the v0 self-revising symbolic world-model agent.

Loop: perceive -> induce (from the last transition) -> infer goal from score ->
exploit (plan to goal) if the model is rich enough, else explore (resolve unknown
actions / cover least-visited ground). The schema is carried across levels
(transport); only a new level's residual drives fresh induction.

Honest scope (v0): models movement + blockers + a goal color. It will solve a
pure-navigation level by discovery and transport that to later levels, but it does
NOT yet model causal object interactions (key->door, push block->pad) — those are
exactly the residual the inducer must later learn to grow structure for.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

from arc_agi_3.agents import Agent
from arc_agi_3.core import Coordinates, FrameData, GameAction, GameState

from .perceptor import color_at, find_cells
from .planner import plan, plan_push_to, plan_to
from .world_model import WorldModel


class WorldModelAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.wm = WorldModel()
        self._prev_grid = None
        self._prev_action: Optional[GameAction] = None
        self._prev_score = 0
        self._prev_level: Optional[int] = None
        self._visited: dict = {}

    def _candidates(self, frame: FrameData):
        return [a for a in frame.available_actions
                if a != GameAction.RESET and not a.requires_coordinates]

    def _unknown_colors(self, grid):
        present = {c for row in grid for c in row}
        known = (self.wm.blocker_colors | self.wm.goal_colors | self.wm.contacted
                 | {self.wm.background, self.wm.agent_color})
        return [c for c in present if c not in known]

    def choose_action(self, frame: FrameData) -> Tuple[GameAction, Optional[Coordinates]]:
        grid = frame.grid
        if frame.state == GameState.GAME_OVER:
            if (self._prev_grid is not None and self._prev_action is not None
                    and self._prev_level == frame.level):
                self.wm.learn_death(self._prev_grid, self._prev_action)
            self._after(grid, GameAction.RESET, frame)
            return GameAction.RESET, None

        if (self._prev_grid is not None and self._prev_action is not None
                and self._prev_action != GameAction.RESET):   # RESET teleports; not a move
            if self._prev_level == frame.level:
                self.wm.update(self._prev_grid, self._prev_action, grid,
                               frame.score - self._prev_score)
            else:
                # Crossed a level boundary: the last action completed the prior level.
                if frame.score > self._prev_score:
                    self.wm.infer_goal(self._prev_grid, self._prev_action)
                self._visited = {}                       # new layout; keep the schema
                self.wm.dead_block_cells.clear()         # dead cells are per-layout

        action = self._decide(frame)
        self._after(grid, action, frame)
        return action, None

    def _decide(self, frame: FrameData) -> GameAction:
        cands = self._candidates(frame)
        if not cands:
            return GameAction.RESET

        # Exploit: plan to the goal — but only if reaching it would actually win. If the
        # goal is known-but-insufficient (its context condition isn't met), don't fixate;
        # fall through to exploration (the provisional-reward / un-fixation fix).
        if (self.wm.goal_colors and self.wm.move_model
                and self.wm.goal_sufficient(frame.grid)):
            path = plan(frame.grid, self.wm)
            if path and path[0] in frame.available_actions:
                return path[0]

        # Goal known but its context isn't met: try to *make it sufficient* by pushing a
        # block onto a coverable required-absent color (cover the pad), then experiment.
        if self.wm.goal_colors and self.wm.move_model and self.wm.required_absent():
            blocking = self.wm.required_absent() & self.wm.present_colors(frame.grid)
            coverable = [c for c in blocking
                         if self.wm.pushable_colors and c not in self.wm.pushable_colors]
            cover_targets = [cell for c in coverable for cell in find_cells(frame.grid, c)]
            for bc in self.wm.pushable_colors:
                path = plan_push_to(frame.grid, self.wm, bc, cover_targets)
                if path and path[0] in frame.available_actions:
                    return path[0]
            # No executable push right now. Distinguish a *true deadlock* (the block is
            # walled in by permanent blockers — the level can no longer be won) from "not
            # reachable yet" (a door to open, a color to investigate first). The optimistic
            # probe treats openable + unknown colors as passable: if even THAT can't route
            # the block to a target, it is genuinely stranded -> record the mistake (its
            # cells are dead ends) and RESET to retry, learning not to wander it there again.
            if cover_targets:
                openable = (set().union(*self.wm.contact_effect.values())
                            if self.wm.contact_effect else set())
                passthrough = frozenset(openable | set(self._unknown_colors(frame.grid)))
                reachable = any(
                    plan_push_to(frame.grid, self.wm, bc, cover_targets, passthrough=passthrough)
                    is not None for bc in self.wm.pushable_colors)
                if not reachable:
                    for bc in self.wm.pushable_colors:
                        self.wm.dead_block_cells.update(find_cells(frame.grid, bc))
                    return GameAction.RESET

        # Explore: first resolve actions whose effect we don't know yet.
        unresolved = [a for a in cands
                      if not self.wm.resolved(a) and not self._would_strand(frame.grid, a)]
        if unresolved:
            return self._rng.choice(unresolved)

        # Epistemic: deliberately go contact an object whose effect we don't know yet
        # (active-inference epistemic value) — this is what learns rules *before* needing them.
        unknown = self._unknown_colors(frame.grid)
        if unknown and self.wm.move_model:
            targets = [cell for col in unknown for cell in find_cells(frame.grid, col)]
            path = plan_to(frame.grid, self.wm, targets)
            if path and path[0] in frame.available_actions:
                return path[0]

        # Experiment (last resort): goal known but insufficient, nothing left to cover or
        # investigate -> reach the goal anyway to TEST the (possibly over-constrained)
        # condition. A win records the goal with the un-removable color present, which
        # self-corrects F_τ(C).
        if self.wm.goal_colors and self.wm.move_model and self.wm.required_absent():
            blocking = self.wm.required_absent() & self.wm.present_colors(frame.grid)
            coverable = [c for c in blocking
                         if self.wm.pushable_colors and c not in self.wm.pushable_colors]
            if blocking and not coverable:
                path = plan(frame.grid, self.wm)
                if path and path[0] in frame.available_actions:
                    return path[0]

        # Else cover the least-visited reachable cell (find the score signal / new ground).
        ap = self.wm.agent_pos(frame.grid)
        if ap is not None and self.wm.move_model:
            h, w = len(frame.grid), len(frame.grid[0])
            order = cands[:]
            self._rng.shuffle(order)
            best, best_v = None, None
            for a in order:
                d = self.wm.move_model.get(a)
                if not d or d == (0, 0):
                    continue
                nx, ny = ap[0] + d[0], ap[1] + d[1]
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if frame.grid[ny][nx] in self.wm.blocker_colors:
                    continue
                if self._would_strand(frame.grid, a):    # don't repeat a learned mistake
                    continue
                v = self._visited.get((nx, ny), 0)
                if best_v is None or v < best_v:
                    best, best_v = a, v
            if best is not None:
                return best
        safe = [a for a in cands if not self._would_strand(frame.grid, a)]
        return self._rng.choice(safe or cands)

    def _would_strand(self, grid, action: GameAction) -> bool:
        """True if `action` would shove a pushable block into a cell already discovered
        (by a prior deadlock + reset) to strand the level. The learned 'not that again' —
        only cells the agent has actually been burned by, never a blanket rule."""
        d = self.wm.move_model.get(action)
        ap = self.wm.agent_pos(grid)
        if not d or ap is None or not self.wm.dead_block_cells:
            return False
        nx, ny = ap[0] + d[0], ap[1] + d[1]
        if color_at(grid, nx, ny) in self.wm.pushable_colors:
            return (nx + d[0], ny + d[1]) in self.wm.dead_block_cells
        return False

    def _after(self, grid, action: GameAction, frame: FrameData) -> None:
        self._prev_grid = grid
        self._prev_action = action
        self._prev_score = frame.score
        self._prev_level = frame.level
        ap = self.wm.agent_pos(grid)
        if ap is not None:
            self._visited[ap] = self._visited.get(ap, 0) + 1
