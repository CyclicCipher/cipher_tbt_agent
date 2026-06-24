"""ControlLoopAgent — the §5/§6 control loop: plan THROUGH the learned dynamics, don't react (ARC step 2).

The flat ColumnAgent reacts toward the goal colour and stalls on the conjunctive levels because it has no model
of the dynamics. This agent uses the dynamics column (dynamics_perceive.py) it learned from play: it reads off
the colour ROLES it discovered — which colour is the key (opens doors), the goal, the hazard, and that the win
is CONJUNCTIVE (on goal AND all pads covered) — and PLANS in that learned forward model (a search over the
perceived state) toward the learned win, avoiding the learned hazard. The subgoal order — cover the pad, get
the key, then reach the goal — falls out of the search; it is not hand-sequenced.

Scaffolded roles for this first cut (the loop closing, measured); the BG-emergent allocation is the follow-on.
Body identity is the efference copy (no colour prior); the colour ROLES are LEARNED, not given.

Run from ProgramSynthesis:  python -m agent.column.control_agent
"""

from __future__ import annotations

import os
import random
import re
import sys
from collections import deque
from statistics import mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "RecurrentWorldModel")))

from arc_agi_3 import Environment, GameAction, GameState     # noqa: E402
from arc_agi_3.games import LockPath                         # noqa: E402

from ..wm.score import oracle_optimal, per_level_actions     # noqa: E402
from .dynamics_perceive import collect                       # noqa: E402
from .perceive import active_cells, detect_motion, modal_background  # noqa: E402

_MOVES = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
_PAD, _BLOCK = 7, 6                                           # perception fixtures (the pushable + its target)


def learn_roles():
    """Run the dynamics column on play, then read the colour ROLES off the rules it discovered."""
    dm = collect()
    roles = {"goal": None, "key": None, "door": None, "hazard": None, "needs_pads": False}
    for _pred, desc, eff in dm.rules:
        m = re.search(r"c0==(\d+)", desc)                    # c0 is `stepped_on`
        if not m:
            continue
        v = int(m.group(1))
        if eff == "score_up":
            roles["goal"] = v
            roles["needs_pads"] = "c1==1" in desc            # c1 is `all_pads_covered`
        elif eff == "death":
            roles["hazard"] = v
        elif eff.startswith("color_") and eff.endswith("_gone"):
            roles["key"], roles["door"] = v, int(eff.split("_")[1])
    return roles


class ControlLoopAgent:
    def __init__(self, roles, seed: int = 0):
        self.roles = roles
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.body_color = None
        self.body_evidence = {}
        self.prev_cells = None
        self.prev_action = None
        self.plan: list = []

    # -- perception (body = efference copy; the static layout from the learned colour roles) ------------
    def _calibrate(self, cells):
        if self.prev_cells is not None and self.prev_action in _MOVES:
            for c, d in detect_motion(self.prev_cells, cells).items():
                if d == self.prev_action.delta:
                    self.body_evidence[c] = self.body_evidence.get(c, 0) + 1
            if self.body_evidence:
                self.body_color = max(self.body_evidence, key=self.body_evidence.get)

    def _perceive(self, grid):
        bg = modal_background(grid)
        cells = active_cells(grid, bg)
        body = next((p for p, c in cells.items() if c == self.body_color), None)
        r = self.roles
        known = {r["goal"], r["key"], r["door"], r["hazard"], _PAD, _BLOCK, self.body_color, bg}
        layout = {
            "walls": {p for p, c in cells.items() if c not in known},      # any other non-bg colour = obstacle
            "doors": {p for p, c in cells.items() if c == r["door"]},
            "pads": {p for p, c in cells.items() if c == _PAD},
            "blocks": {p for p, c in cells.items() if c == _BLOCK},
            "key": next((p for p, c in cells.items() if c == r["key"]), None),
            "goal": next((p for p, c in cells.items() if c == r["goal"]), None),
            "hazards": {p for p, c in cells.items() if c == r["hazard"]},
        }
        has_key = not layout["doors"]                        # doors gone (or none) => key already used
        return cells, body, layout, has_key

    # -- planning: BFS in the LEARNED forward model toward the LEARNED win, avoiding the hazard ----------
    def _plan(self, body, layout, has_key):
        W, D, P, B, K, G, H = (layout["walls"], layout["doors"], set(layout["pads"]),
                               layout["blocks"], layout["key"], layout["goal"], layout["hazards"])
        if G is None:
            return []
        # remembered pads: a pad currently under a block reads as a block, so union the two
        pads = P | {p for p in B if p in P}

        def step(state, dx, dy):
            (ax, ay), blocks, hk = state
            t = (ax + dx, ay + dy)
            if t in W or (t in D and not hk):
                return None
            blocks = set(blocks)
            if t in blocks:
                beyond = (t[0] + dx, t[1] + dy)
                if beyond in W or beyond in blocks or (beyond in D and not hk):
                    return None
                blocks.discard(t)
                blocks.add(beyond)
            if t in H:
                return None                                  # never plan INTO the learned hazard
            if t == K:
                hk = True
            return (t, frozenset(blocks), hk)

        start = (body, frozenset(B), has_key)
        seen = {start}
        q = deque([(start, [])])
        while q:
            state, path = q.popleft()
            (pos, blocks, hk) = state
            if pos == G and (not self.roles["needs_pads"] or pads <= set(blocks)):
                return path
            for a in _MOVES:
                dx, dy = a.delta
                nxt = step(state, dx, dy)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + [a]))
        return []

    # -- the agent contract --------------------------------------------------------------------------
    def choose_action(self, frame):
        if frame.state == GameState.GAME_OVER:
            self.plan = []
            self.prev_cells, self.prev_action = None, None
            return GameAction.RESET, None
        grid = frame.grid
        cells = active_cells(grid, modal_background(grid))
        self._calibrate(cells)
        cells, body, layout, has_key = self._perceive(grid)

        if body is None or self.body_color is None:          # still calibrating who I am -> move to generate reafference
            action = self.rng.choice(_MOVES)
        else:
            if not self.plan:
                self.plan = self._plan(body, layout, has_key)
            action = self.plan.pop(0) if self.plan else self.rng.choice(_MOVES)

        self.prev_cells, self.prev_action = cells, action
        return action, None


def run(roles, seeds=range(12), max_actions=6000):
    opt = oracle_optimal(LockPath)
    n = len(opt)
    rows = []
    for s in seeds:
        env = Environment(LockPath())
        per, completed = per_level_actions(env, ControlLoopAgent(roles, seed=s), max_actions)
        lvl = [min(1.0, (opt[i] / per[i]) ** 2) if (i in completed and opt[i] and per.get(i)) else 0.0
               for i in range(n)]
        rows.append((s, len(completed), mean(lvl)))
    return opt, rows


if __name__ == "__main__":
    print("ControlLoopAgent — plan through the LEARNED dynamics toward the LEARNED conjunctive win.\n")
    roles = learn_roles()
    print(f"  colour roles read off the learned dynamics: {roles}\n")
    opt, rows = run(roles)
    for s, nc, sc in rows:
        print(f"  seed {s:2d}:  {nc}/{len(opt)} levels   RHAE {100 * sc:5.1f}%")
    print(f"\n  mean levels completed: {mean(r[1] for r in rows):.2f}/{len(opt)}    "
          f"mean RHAE-proxy: {100 * mean(r[2] for r in rows):.1f}%   (flat ColumnAgent ~2.0/4, 3.3%)")
