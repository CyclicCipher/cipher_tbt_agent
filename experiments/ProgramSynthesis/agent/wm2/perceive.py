"""Phase-2 prior-minimal perception & dynamics discovery (the start of the replica test).

The Phase-1 agent (`agent/wm/`) *seeded* its Core Knowledge: `detect_move` hardcoded "the single
cell that translates is the agent" (it required exactly one gained + one lost cell), and contact /
push / rule-types were engineered. **Here we remove those** and keep only the bare floor:

  - SENSORY interface: a frame is a grid of (x, y, color) cells; we also observe the action key
    (opaque — no assumed semantics) and, later, the score.
  - METRIC / temporal continuity: a colour that leaves one cell and appears at another *nearby* cell
    is the same unit moving (nearest-position matching); position is a metric.
  - COMPRESSION: a regularity is real only if it holds consistently across observations (modal effect
    with high agreement) — the MDL/prediction drive.

From that floor alone we DISCOVER agency: not "one cell = me", but *whichever colour the actions
actually control*. The agent is the action-consistent colour with the smallest presence (a localized
unit, not the pervasive substrate) — a far weaker, more general prior than the seeded single-cell one.
No contact, no push, no rule-type vocabulary is assumed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]
Delta = Tuple[int, int]


def changes(g0: Grid, g1: Grid) -> List[Tuple[int, int, int, int]]:
    """Cells that differ between two frames: (x, y, old_color, new_color)."""
    return [(x, y, g0[y][x], g1[y][x])
            for y in range(len(g0)) for x in range(len(g0[0])) if g0[y][x] != g1[y][x]]


def color_displacements(g0: Grid, g1: Grid) -> Dict[int, Delta]:
    """For each colour that moved as a single coherent unit (left exactly one cell, arrived at
    exactly one cell), its displacement Δ = arrival − departure. No background is assumed; every
    colour (including the substrate) is treated symmetrically and falls out by presence later."""
    lost: Dict[int, List[Pos]] = defaultdict(list)
    gained: Dict[int, List[Pos]] = defaultdict(list)
    for x, y, a, b in changes(g0, g1):
        lost[a].append((x, y))
        gained[b].append((x, y))
    disp: Dict[int, Delta] = {}
    for c in set(lost) | set(gained):
        L, G = lost.get(c, []), gained.get(c, [])
        if len(L) == 1 and len(G) == 1:          # one coherent displacement of this colour
            (px, py), (qx, qy) = L[0], G[0]
            disp[c] = (qx - px, qy - py)
    return disp


def find_cells(grid: Grid, color: int) -> List[Pos]:
    return [(x, y) for y in range(len(grid)) for x in range(len(grid[0])) if grid[y][x] == color]


def color_at(grid: Grid, x: int, y: int) -> Optional[int]:
    if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
        return grid[y][x]
    return None


def discover_blockers(observations: List[Tuple[Grid, object, Grid]],
                      agent_color: int, move_model: Dict[object, Delta]) -> set:
    """Discover which colours *stop* the agent. We predict the agent's next cell from the discovered
    move model; when the move fails (the agent stays put), the colour at the predicted cell is what
    blocked it — prediction + residual, no seeded blocker concept. A pure blocker is a colour the
    agent can NEVER enter (failures, zero successes); colours it sometimes enters and sometimes
    can't (e.g. a door) are *conditional* and fall to causal-edge discovery (increment 3)."""
    enter: Counter = Counter()
    blocked: Counter = Counter()
    for g0, action, g1 in observations:
        if action not in move_model:
            continue
        ap = find_cells(g0, agent_color)
        if len(ap) != 1:
            continue
        (px, py), (dx, dy) = ap[0], move_model[action]
        tc = color_at(g0, px + dx, py + dy)
        if tc is None:
            continue
        after = find_cells(g1, agent_color)
        if (px + dx, py + dy) in after:
            enter[tc] += 1
        elif (px, py) in after:
            blocked[tc] += 1
    return {c for c in blocked if enter[c] == 0}


def _mean_presence(grids: List[Grid]) -> Dict[int, float]:
    cnt: Counter = Counter()
    for g in grids:
        for row in g:
            cnt.update(row)
    return {c: cnt[c] / len(grids) for c in cnt}


def discover_dynamics(observations: List[Tuple[Grid, object, Grid]],
                      min_support: int = 3, agree: float = 0.6
                      ) -> Tuple[Optional[int], Dict[object, Delta]]:
    """From (grid, action, next_grid) transitions, DISCOVER the agent colour and its action→Δ move
    model — using only the floor priors above. Returns (agent_color or None, move_model).

    A colour is a candidate agent if each action it responds to has a *consistent* modal displacement
    (agreement ≥ `agree`) with enough support. Among candidates, the agent is the one with the
    smallest mean presence — a localized controllable unit rather than the pervasive substrate.
    """
    per: Dict[int, Dict[object, Counter]] = defaultdict(lambda: defaultdict(Counter))
    grids: List[Grid] = []
    for g0, action, g1 in observations:
        grids.append(g0)
        for c, d in color_displacements(g0, g1).items():
            if d != (0, 0):
                per[c][action][d] += 1
    presence = _mean_presence(grids)

    candidates: Dict[int, Dict[object, Delta]] = {}
    for c, amap in per.items():
        model, support, consistent = {}, 0, True
        for action, ctr in amap.items():
            d, n = ctr.most_common(1)[0]
            total = sum(ctr.values())
            support += total
            if n / total < agree:
                consistent = False
            model[action] = d
        if consistent and support >= min_support and model:
            candidates[c] = model

    if not candidates:
        return None, {}
    agent = min(candidates, key=lambda c: presence.get(c, float("inf")))
    return agent, candidates[agent]
