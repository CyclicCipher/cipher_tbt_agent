"""Phase-2 prior-minimal agents.

`VolumeAgent` — the Phase-1 `WorldModelAgent` with its seeded priors swapped for discovered ones
(increment A: agency; increment B: contact rule-types), reusing the proven acting loop / typed
planner / exploration / deadlock learning. **This is the working prior-minimal agent: 12/12 seeds,
37.5% RHAE — equal to the baseline on a strictly smaller prior floor.**

`ForwardPlanAgent` — increment C (WIP): replaces the typed planners with ONE forward-simulating
search (`plan.plan_to_win`); cover/open/experiment emerge from search rather than being coded. It
generalizes *exploitation* but currently caps at ~2–4/12 on the full game — the flat forward-BFS is
the predicted scaling wall, and the gap to the baseline is exactly the reactive machinery (deadlock
recovery, hazard self-correction, directed discovery) that doesn't reduce to flat planning. The next
step is HIERARCHICAL planning over the discovered edges (Merge / topological order of relations,
refined by navigation). Kept here as the foundation for that, NOT as the production agent. See
`docs/phase2/REPLICA_TEST.md`.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from arc_agi_3.core import FrameData, GameAction

from agent.wm.agent import WorldModelAgent
from agent.wm.planner import plan_push_to

from .hplan import hplan_to_win
from .perceive import find_cells
from .plan import plan_to_win
from .world_model import DiscoveredWorldModel


class VolumeAgent(WorldModelAgent):
    """Prior-minimal induction (discovered agency + general residual effects), reusing the rest of
    the Phase-1 machinery — including its typed planner. The working agent (12/12)."""

    def reset(self) -> None:
        super().reset()
        self.wm = DiscoveredWorldModel()


class ForwardPlanAgent(VolumeAgent):
    """Increment C (WIP) — typed planners replaced by one forward-simulating search."""

    def reset(self) -> None:
        super().reset()
        self._plan = []                          # cached forward-sim plan, followed until exhausted
        self._plan_sig = None                    # model signature the plan was built against

    def _compute_plan(self, grid):
        """The plan-to-win used by `_decide`. Flat forward-simulation here; the hierarchical
        subclass overrides this with the macro planner."""
        return plan_to_win(grid, self.wm)

    def _model_sig(self):
        wm = self.wm
        return (frozenset(wm.move_model.items()), frozenset(wm.blocker_colors),
                frozenset(wm.pushable_colors), frozenset(wm.goal_colors),
                frozenset(wm.required_absent()),
                frozenset((k, frozenset(v)) for k, v in wm.contact_effect.items()))

    def _decide(self, frame: FrameData) -> GameAction:
        cands = self._candidates(frame)
        if not cands:
            return GameAction.RESET
        wm = self.wm
        if wm.move_model and wm.agent_color is not None:
            sig = self._model_sig()
            if sig != self._plan_sig:                             # the model changed → plan is stale
                self._plan = []
                self._plan_sig = sig
            if self._plan:                                        # follow the cached plan
                a = self._plan.pop(0)
                if a in frame.available_actions:
                    return a
                self._plan = []
            path = self._compute_plan(frame.grid)                 # plan to a win
            if path:
                self._plan = path
                a = self._plan.pop(0)
                if a in frame.available_actions:
                    return a
            step = self._explore(frame)                           # else learn more of the model
            if step is not None and step in frame.available_actions:
                return step
            if path is None:                                      # no reachable win, nothing to learn
                return GameAction.RESET                           # → general deadlock: retry the level
        return self._rng.choice(cands)                            # no model yet: gather data

    def _explore(self, frame: FrameData) -> Optional[GameAction]:
        grid = frame.grid
        ap = self.wm.agent_pos(grid)
        if ap is None:
            return None
        unresolved = [a for a in self._candidates(frame) if not self.wm.resolved(a)]
        if unresolved:
            return self._rng.choice(unresolved)

        unknown = self._unknown_cells(grid)                       # discover new colours' effects first
        if unknown:
            step = self._bfs_first(grid, ap, set(unknown))
            if step is not None:
                return step

        # Push the block onto a required-absent colour — to cover it, or to discover its effect (the
        # context-dependent block→pad rule is only learned by trying it). DEADLOCK TRIAD: if no such
        # target is reachable, the block is stranded → LEARN its cells as dead and signal a reset.
        present = {c for row in grid for c in row}
        push_targets = (self.wm.required_absent() & present) - self.wm.pushable_colors
        if push_targets and self.wm.pushable_colors:
            for bc in self.wm.pushable_colors:
                for color in push_targets:
                    path = plan_push_to(grid, self.wm, bc, find_cells(grid, color))
                    if path and path[0] in frame.available_actions:
                        return path[0]
            for bc in self.wm.pushable_colors:                    # stranded → learn the mistake
                self.wm.dead_block_cells.update(find_cells(grid, bc))
            return None                                           # → _decide resets (recover)

        return self._bfs_least_visited(grid, ap)

    def _unknown_cells(self, grid):
        known = (self.wm.blocker_colors | self.wm.goal_colors | self.wm.contacted
                 | {self.wm.background, self.wm.agent_color})
        return [(x, y) for y in range(len(grid)) for x in range(len(grid[0]))
                if grid[y][x] not in known]

    def _passable(self, grid, x, y):
        return (0 <= x < len(grid[0]) and 0 <= y < len(grid)
                and grid[y][x] not in self.wm.blocker_colors)

    def _bfs_first(self, grid, start, targets) -> Optional[GameAction]:
        seen = {start}
        q = deque([(start, None)])
        while q:
            (x, y), first = q.popleft()
            if (x, y) in targets and first is not None:
                return first
            for a, (dx, dy) in self.wm.move_model.items():
                if dx == 0 and dy == 0:
                    continue
                if first is None and self._would_strand(grid, a):   # don't re-strand the block
                    continue
                nxt = (x + dx, y + dy)
                if nxt in seen or not (self._passable(grid, *nxt) or nxt in targets):
                    continue
                seen.add(nxt)
                q.append((nxt, a if first is None else first))
        return None

    def _bfs_least_visited(self, grid, start) -> Optional[GameAction]:
        seen = {start}
        q = deque([(start, None)])
        best = None
        while q:
            (x, y), first = q.popleft()
            if first is not None:
                v = self._visited.get((x, y), 0)
                if best is None or v < best[1]:
                    best = (first, v)
                    if v == 0:
                        break
            for a, (dx, dy) in self.wm.move_model.items():
                if dx == 0 and dy == 0:
                    continue
                if first is None and self._would_strand(grid, a):   # don't re-strand the block
                    continue
                nxt = (x + dx, y + dy)
                if nxt in seen or not self._passable(grid, *nxt):
                    continue
                seen.add(nxt)
                q.append((nxt, a if first is None else first))
        return best[0] if best else None


class HierarchicalPlanAgent(ForwardPlanAgent):
    """The forward-plan agent with the flat search replaced by the HIERARCHICAL macro planner
    (`hplan.hplan_to_win`): plan over discovered-edge macros (reach-trigger / push-onto-target /
    reach-goal) composed in prerequisite order, each refined by a focused BFS. Collapses the scaling
    wall and turns deadlock into a high-level subgoal-infeasibility. Everything else (discovery,
    epistemic probe, plan cache) is inherited."""

    def _compute_plan(self, grid):
        return hplan_to_win(grid, self.wm)


class EFEAgent(HierarchicalPlanAgent):
    """Action selection by EXPECTED-FREE-ENERGY structure, replacing the hand-tuned reactive branches.

    Each step the agent prefers, in value order: (1) PRAGMATIC — execute a plan to a winning state
    (the hierarchical planner is this term); (2) EPISTEMIC — reduce model uncertainty: resolve unknown
    moves, contact unknown colours, push movables onto colours they have NOT yet occupied; (3) RECOVER
    — reset, only when *every* epistemic option is exhausted and no win is reachable (a genuine
    deadlock). The fix over the earlier reactive glue is honest uncertainty accounting:
    `_block_landings` tracks which colours a movable has actually been pushed onto, so "nothing left to
    learn" means it — exploration completes before a deadlock is ever concluded. (A discrete
    approximation of EFE minimisation; the continuous, commensurable-units version is the fuller form.)
    """

    def reset(self) -> None:
        super().reset()
        self._block_landings = set()             # colours a movable has occupied (tested pushes)

    def _after(self, grid, action, frame) -> None:
        if self._prev_grid is not None:          # record what any movable just landed on
            for m in self.wm.pushable_colors:
                for (x, y) in find_cells(grid, m):
                    prev = self._prev_grid[y][x]
                    if prev != m:
                        self._block_landings.add(prev)
        super()._after(grid, action, frame)

    def _decide(self, frame: FrameData) -> GameAction:
        cands = self._candidates(frame)
        if not cands:
            return GameAction.RESET
        wm = self.wm
        if not (wm.move_model and wm.agent_color is not None):
            return self._rng.choice(cands)                       # no model yet: learning the moves is epistemic

        sig = self._model_sig()                                  # follow a cached plan if still valid
        if sig != self._plan_sig:
            self._plan = []
            self._plan_sig = sig
        if self._plan:
            a = self._plan.pop(0)
            if a in frame.available_actions:
                return a
            self._plan = []

        plan = self._compute_plan(frame.grid)                    # (1) PRAGMATIC: exploit a win plan
        if plan:
            self._plan = plan
            a = self._plan.pop(0)
            if a in frame.available_actions:
                return a

        epi = self._epistemic_action(frame)                      # (2) EPISTEMIC: reduce uncertainty
        if epi is not None and epi in frame.available_actions:
            return epi

        return GameAction.RESET                                  # (3) RECOVER: deadlock → retry the level

    def _epistemic_action(self, frame: FrameData) -> Optional[GameAction]:
        grid = frame.grid
        ap = self.wm.agent_pos(grid)
        if ap is None:
            return None
        unresolved = [a for a in self._candidates(frame) if not self.wm.resolved(a)]
        if unresolved:
            return self._rng.choice(unresolved)                  # learn the unknown moves

        unknown = self._unknown_cells(grid)                      # contact colours we've never touched
        if unknown:
            step = self._bfs_first(grid, ap, set(unknown))
            if step is not None:
                return step

        present = {c for row in grid for c in row}               # push a movable onto an untested colour
        skip = (self._block_landings | self.wm.blocker_colors | self.wm.pushable_colors
                | self.wm.goal_colors | {self.wm.background, self.wm.agent_color})
        for m in self.wm.pushable_colors:
            for d in present - skip:
                path = plan_push_to(grid, self.wm, m, find_cells(grid, d))
                if path and path[0] in frame.available_actions:
                    return path[0]

        if not self.wm.goal_colors:                              # still hunting the score signal
            return self._bfs_least_visited(grid, ap)
        return None                                              # uncertainty exhausted → let _decide recover
