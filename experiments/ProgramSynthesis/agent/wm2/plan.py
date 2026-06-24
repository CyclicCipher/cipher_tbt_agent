"""Phase-2 increment C — a forward-simulating planner (replaces Phase-1's typed planners).

Phase 1 had three hand-written strategies — `plan_to` (route + open doors), `plan` (route around
pushables), `plan_push_to` (Sokoban) — plus a typed `_decide` (exploit / cover-the-pad / experiment).
Here there is ONE thing: a forward model `forward_step` that applies the *discovered* transitions
generically, and a BFS that searches over abstract states for a winning one. Push and open are no
longer strategies — they are ordinary transitions the search plans *through*, so "cover the pad, then
reach the goal" is discovered by search rather than coded.

Abstract state = (agent_pos, movable positions, opened colours). The base grid is the static layout
(agent + movables removed); a cell's effective colour is its movable's colour, else background if its
base colour has been opened, else its base colour.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

from .perceive import find_cells

Pos = Tuple[int, int]


def _removable(wm) -> FrozenSet[int]:
    """Colours something can make absent (any opening's target). The rest of a goal's
    context-condition is *un*-removable → tested by experiment (reach the goal anyway)."""
    out = set()
    for targets in wm.contact_effect.values():
        out |= set(targets)
    return frozenset(out)


def _eff(base, x, y, movables: Dict[Pos, int], opened: FrozenSet[int], bg) -> Optional[int]:
    if (x, y) in movables:
        return movables[(x, y)]
    if not (0 <= y < len(base) and 0 <= x < len(base[0])):
        return None
    c = base[y][x]
    return bg if c in opened else c


def forward_step(agent: Pos, movables: Dict[Pos, int], opened: FrozenSet[int],
                 action, base, wm) -> Optional[Tuple[Pos, Dict[Pos, int], FrozenSet[int]]]:
    """Apply one action to an abstract state via the discovered model. Returns the next state, or
    None if the action produces no change (blocked / off-grid)."""
    d = wm.move_model.get(action)
    if not d or d == (0, 0):
        return None
    bg = wm.background
    tx, ty = agent[0] + d[0], agent[1] + d[1]
    if not (0 <= ty < len(base) and 0 <= tx < len(base[0])):
        return None

    if (tx, ty) in movables:                                  # push a movable
        mcolor = movables[(tx, ty)]
        nx, ny = tx + d[0], ty + d[1]
        if not (0 <= ny < len(base) and 0 <= nx < len(base[0])) or (nx, ny) in movables:
            return None
        if _eff(base, nx, ny, movables, opened, bg) in wm.blocker_colors:
            return None                                       # can't shove it into a blocker
        nm = dict(movables); del nm[(tx, ty)]; nm[(nx, ny)] = mcolor
        no = set(opened)
        landing = base[ny][nx]                                # movable-triggered effect (block→pad)
        if landing in wm.contact_effect.get(mcolor, ()):
            no.add(landing)
        return (tx, ty), nm, frozenset(no)

    tc = _eff(base, tx, ty, movables, opened, bg)             # walk / contact
    if tc in wm.blocker_colors:                               # wall / closed door / hazard
        return None
    no = set(opened)
    for opened_color in wm.contact_effect.get(base[ty][tx], ()):   # agent-triggered effect (key→door)
        no.add(opened_color)
    return (tx, ty), dict(movables), frozenset(no)


def _is_win(base, agent: Pos, movables, opened, wm, removable) -> bool:
    """Winning state: the agent stands on a goal colour and the *removable* part of the goal's
    context-condition is satisfied. Un-removable required colours are ignored here — reaching the
    goal with them present is the experiment that refutes an over-constrained condition."""
    if agent in movables:
        return False
    bc = base[agent[1]][agent[0]]
    if bc not in wm.goal_colors or bc in opened:
        return False
    req = wm.required_absent()
    need = req & removable
    if req and not need:
        return False              # condition unmet and nothing removable yet → not a win, go discover
    return need <= opened


def plan_to_win(grid, wm, max_states: int = 40000) -> Optional[List]:
    """BFS over abstract states for a winning one; returns the FULL action sequence to it (the
    agent caches and follows it, re-planning only on surprise), [] if already winning, or None if
    no win is reachable under the current model."""
    if not wm.move_model or wm.agent_color is None:
        return None
    agent_cells = find_cells(grid, wm.agent_color)
    if len(agent_cells) != 1:
        return None
    agent = agent_cells[0]
    movables = {cell: mc for mc in wm.pushable_colors for cell in find_cells(grid, mc)}

    base = [row[:] for row in grid]                           # static layout = grid − agent − movables
    bg = wm.background
    for (x, y) in agent_cells:
        base[y][x] = bg
    for (x, y) in movables:
        base[y][x] = bg

    removable = _removable(wm)
    if _is_win(base, agent, movables, frozenset(), wm, removable):
        return []

    start_key = (agent, frozenset(movables.items()), frozenset())
    seen = {start_key}
    parent = {start_key: None}                                # state_key -> (prev_key, action)
    queue = deque([(agent, movables, frozenset())])
    states = 0
    goal_key = None
    while queue and states < max_states and goal_key is None:
        states += 1
        ag, mv, op = queue.popleft()
        cur_key = (ag, frozenset(mv.items()), op)
        for a in wm.move_model:
            res = forward_step(ag, mv, op, a, base, wm)
            if res is None:
                continue
            nag, nmv, nop = res
            key = (nag, frozenset(nmv.items()), nop)
            if key in seen:
                continue
            seen.add(key)
            parent[key] = (cur_key, a)
            if _is_win(base, nag, nmv, nop, wm, removable):
                goal_key = key
                break
            queue.append((nag, nmv, nop))
    if goal_key is None:
        return None
    path = []
    k = goal_key
    while parent[k] is not None:
        pk, a = parent[k]
        path.append(a)
        k = pk
    path.reverse()
    return path
