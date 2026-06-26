"""Multi-column LockPath agent — the task ⊕ space control loop on the real ARC replica (architecture §5/§6).

The flat `control_agent` BFS's the JOINT state (agent × all blocks × has_key) — the conjunctive explosion the
multi-column architecture replaces (validated additively on ConjGrid in RecurrentWorldModel/precursor/
control_loop.py). Ported here to LockPath, where — unlike ConjGrid's any-order switches — the subgoals carry
DEPENDENCIES that are LEARNED from the dynamics column (dynamics_perceive.py), not given:

  score_up needs all pads covered  ⟹  cover-pad ≺ reach-goal     (the conjunctive win)
  key removes the doors             ⟹  get-key   ≺ a door-blocked path
  hazard ends the game              ⟹  a navigation constraint (avoid), not a subgoal

So the TASK column emits an ORDERED plan — [get-key?] + [cover-pad_i…] + [reach-goal] — and the SPATIAL column
(a CorticalColumn over the walkable grid) navigates to ONE subgoal at a time: reach-a-cell over agent positions
(key/goal), or push-one-block over (agent, that block) for a pad — NEVER the joint state. The THALAMUS routes
the active subgoal → its goal-cell (the §5 top-down channel; `read_location`). Search is ADDITIVE: K+2 small
navigations sequenced by the task column.

Measured on the RHAE scorer (agent/wm/score.py) vs the flat control_agent (≈4/4 levels, 96.5%).

Run from ProgramSynthesis:  python -m agent.column.multicolumn_agent
"""

from __future__ import annotations

import os
import sys
from collections import deque
from statistics import mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "RecurrentWorldModel")))

from tbt.column import CorticalColumn                            # noqa: E402
from tbt.thalamus import Thalamus                               # noqa: E402

from arc_agi_3 import Environment, GameAction, GameState         # noqa: E402
from arc_agi_3.games import LockPath                             # noqa: E402

from ..wm.score import oracle_optimal, per_level_actions         # noqa: E402
from .control_agent import ControlLoopAgent, learn_roles         # reuse role-learning + perception  # noqa: E402
from .perceive import active_cells, modal_background             # noqa: E402

_MOVES = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]


def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _walkable(layout):
    """Every non-wall cell inside the level's bounding box — the spatial column's nodes (doors/pads/hazard
    cells are physically walkable; whether the agent MAY enter is a dynamic constraint applied at plan time)."""
    pts = set(layout["walls"]) | set(layout["doors"]) | set(layout["pads"]) | set(layout["blocks"]) | set(layout["hazards"])
    for k in ("key", "goal"):
        if layout[k] is not None:
            pts.add(layout[k])
    if not pts:
        return set()
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]
    return {(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)
            if (x, y) not in layout["walls"]}


def _bfs_reach(walk, start, target, blocked):
    """Navigate the agent to `target` — over AGENT POSITIONS only (N² not the joint). `blocked` = cells the
    agent may not enter (hazards, closed doors, blocks treated as obstacles)."""
    q, seen = deque([(start, [])]), {start}
    while q:
        p, path = q.popleft()
        if p == target:
            return path
        for a in _MOVES:
            dx, dy = a.delta
            nb = (p[0] + dx, p[1] + dy)
            if nb in walk and nb not in blocked and nb not in seen:
                seen.add(nb); q.append((nb, path + [a]))
    return []


def _bfs_push(walk, start, block, pad, blocked, other_blocks):
    """Push ONE block onto `pad` — BFS over (agent, block), not the joint of all blocks. `blocked` = cells
    neither agent nor block may enter (hazards/closed doors); `other_blocks` = immovable obstacles."""
    q, seen = deque([((start, block), [])]), {(start, block)}
    while q:
        (a, b), path = q.popleft()
        if b == pad:
            return path
        for act in _MOVES:
            dx, dy = act.delta
            na = (a[0] + dx, a[1] + dy)
            if na not in walk or na in blocked or na in other_blocks:
                continue
            if na == b:                                          # moving into the block pushes it
                nb = (b[0] + dx, b[1] + dy)
                if nb not in walk or nb in blocked or nb in other_blocks:
                    continue                                     # can't push (wall/door/another block beyond)
                state = (na, nb)
            else:
                state = (na, b)
            if state not in seen:
                seen.add(state); q.append((state, path + [act]))
    return []


class MultiColumnAgent(ControlLoopAgent):
    """task ⊕ space on LockPath: reuses the flat agent's calibration + perception, replaces its joint BFS."""

    def reset(self):
        super().reset()
        self._cache = None                                       # per-level: spatial column + thalamus routing

    # -- the spatial column (the map) + the thalamus binding of the level's landmarks -------------------
    def _spatial(self, layout):
        key = frozenset(layout["walls"])
        if self._cache and self._cache["key"] == key:
            return self._cache
        walk = _walkable(layout)
        cells = sorted(walk)
        cid = {c: i for i, c in enumerate(cells)}
        col = CorticalColumn(n_entities=max(1, len(cells)))
        for c in cells:                                          # learn the walkable grid → SR-eigenvector frame
            for j, a in enumerate(_MOVES):
                dx, dy = a.delta
                nb = (c[0] + dx, c[1] + dy)
                if nb in walk:
                    col.observe(cid[c], j, cid[nb])
        col.consolidate()
        landmarks = []                                           # the level's goal-states (key, pads, goal)
        if layout["key"] is not None:
            landmarks.append(("key", layout["key"]))
        for pad in sorted(layout["pads"]):
            landmarks.append(("pad", pad))
        if layout["goal"] is not None:
            landmarks.append(("goal", layout["goal"]))
        thal = Thalamus()
        task_col = CorticalColumn(n_entities=max(1, len(landmarks)))
        R = thal.bind(task_col, col, [(i, cid[cell]) for i, (_, cell) in enumerate(landmarks)])
        inv = {idx: c for c, idx in col.loc.items()}            # consolidate-index → cell-id
        routed = {}
        for i, (_, cell) in enumerate(landmarks):                # TOP-DOWN: thalamus routes each landmark → its cell
            idx = thal.read_location(R, task_col, col, i)
            routed[i] = cells[inv[idx]] if idx is not None and idx in inv else cell
        self._cache = dict(key=key, walk=walk, landmarks=landmarks, routed=routed,
                           route_ok=all(routed[i] == landmarks[i][1] for i in range(len(landmarks))))
        return self._cache

    # -- the TASK column: the next subgoal, ORDERED by the learned dependencies -------------------------
    def _active(self, layout, has_key, landmarks):
        key_i = next((i for i, (k, _) in enumerate(landmarks) if k == "key"), None)
        if layout["doors"] and key_i is not None and layout["key"] is not None and not has_key:
            return key_i                                         # get-key ≺ a door-blocked path
        if self.roles["needs_pads"]:                             # cover-pad ≺ reach-goal (the conjunctive win)
            for i, (k, cell) in enumerate(landmarks):
                if k == "pad" and cell not in layout["blocks"]:  # an uncovered pad
                    return i
        return next((i for i, (k, _) in enumerate(landmarks) if k == "goal"), None)

    def _plan_subgoal(self, layout, has_key, body):
        cache = self._spatial(layout)
        walk, landmarks, routed = cache["walk"], cache["landmarks"], cache["routed"]
        i = self._active(layout, has_key, landmarks)
        if i is None:
            return []
        kind, _ = landmarks[i]
        target = routed[i]                                       # the §5 goal-state, routed by the thalamus
        blocked = set(layout["hazards"]) | (set(layout["doors"]) if not has_key else set())
        if kind == "pad":                                        # push one free block onto the pad
            free = [b for b in layout["blocks"] if b not in layout["pads"]]
            if not free:
                return []
            block = min(free, key=lambda b: _manh(b, target))
            return _bfs_push(walk, body, block, target, blocked, set(layout["blocks"]) - {block})
        return _bfs_reach(walk, body, target, blocked | set(layout["blocks"]))   # reach: blocks are obstacles

    def choose_action(self, frame):
        if frame.state == GameState.GAME_OVER:
            self.plan = []
            self.prev_cells, self.prev_action = None, None
            return GameAction.RESET, None
        grid = frame.grid
        cells = active_cells(grid, modal_background(grid))
        self._calibrate(cells)
        cells, body, layout, has_key = self._perceive(grid)
        if body is None or self.body_color is None:              # still calibrating who I am → move for reafference
            action = self.rng.choice(_MOVES)
        else:
            if not self.plan:
                self.plan = self._plan_subgoal(layout, has_key, body)
            action = self.plan.pop(0) if self.plan else self.rng.choice(_MOVES)
        self.prev_cells, self.prev_action = cells, action
        return action, None


def run(roles, seeds=range(12), max_actions=6000):
    opt = oracle_optimal(LockPath)
    n = len(opt)
    rows = []
    for s in seeds:
        env = Environment(LockPath())
        per, completed = per_level_actions(env, MultiColumnAgent(roles, seed=s), max_actions)
        lvl = [min(1.0, (opt[i] / per[i]) ** 2) if (i in completed and opt[i] and per.get(i)) else 0.0
               for i in range(n)]
        rows.append((s, len(completed), mean(lvl)))
    return opt, rows


def check_routing(roles):
    """Confirm the thalamus is LOAD-BEARING: for each level, does read_location route every landmark to its
    correct cell (not the perceived-cell fallback)?"""
    game = LockPath()
    ag = MultiColumnAgent(roles); ag.body_color = 2                  # C_AGENT — skip efference calibration here
    out = []
    for lvl in range(game.level_count):
        game.load_level(lvl)
        ag._cache = None
        _, _, layout, _ = ag._perceive(game.render()[0])
        cache = ag._spatial(layout)
        out.append((lvl, len(cache["landmarks"]), cache["route_ok"]))
    return out


if __name__ == "__main__":
    print("multi-column LockPath agent — task ⊕ space, subgoal deps LEARNED from the dynamics column.\n")
    roles = learn_roles()
    print(f"  colour roles read off the learned dynamics: {roles}\n")
    print("  thalamus goal-state routing (read_location → landmark cell, per level):")
    for lvl, nlm, ok in check_routing(roles):
        print(f"      L{lvl}: {nlm} landmarks   route_ok={ok}")
    print()
    opt, rows = run(roles)
    for s, nc, sc in rows:
        print(f"  seed {s:2d}:  {nc}/{len(opt)} levels   RHAE {100 * sc:5.1f}%")
    print(f"\n  mean levels: {mean(r[1] for r in rows):.2f}/{len(opt)}    "
          f"mean RHAE-proxy: {100 * mean(r[2] for r in rows):.1f}%   (flat control_agent ~4/4, 96.5%)")
    print("  task ⊕ space: K+2 additive subgoals (get-key ≺ cover-pad ≺ reach-goal), never the joint state.")
