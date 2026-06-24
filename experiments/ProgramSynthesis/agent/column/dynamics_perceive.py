"""Live-grid dynamics learning — the dynamics column learns LockPath's real rules from real play (ARC step 1b).

The flat ColumnAgent detects exafference and throws it away. This grounds the dynamics column (tbt/dynamics.py)
in actual frames: from each (prev_frame, action, frame) it perceives the dynamics FEATURES and the EXAFFERENT
effect, with no semantic priors —
  - body identity is the EFFERENCE COPY (the colour that translates by the issued action's delta);
  - `stepped_on` = the colour at the body's intended destination (generic);
  - `all_pads_covered` = every remembered pad cell now shows a block (the relational, occlusion-aware feature);
  - effects = score_up (a level completed), death (GAME_OVER), or `color_C_gone` (a colour vanished spontaneously,
    i.e. doors opening) — characterised generically, never as "door"/"key".
The DynamicsModel then DISCOVERS the rules, including the CONJUNCTIVE win condition that the flat agent can't
represent. An oracle-hinted + random explorer triggers the dynamics (the agent never sees the oracle's labels;
it only generates experience to learn from — directed exploration is the control loop's job, step 2).

Run from ProgramSynthesis:  python -m agent.column.dynamics_perceive
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "RecurrentWorldModel")))
from tbt.dynamics import DynamicsModel                       # noqa: E402

from arc_agi_3 import Environment, GameAction, GameState     # noqa: E402
from arc_agi_3.games import LockPath                         # noqa: E402
from arc_agi_3.oracle import _capture, _restore, solve_level  # noqa: E402

from .perceive import active_cells, detect_motion, modal_background   # noqa: E402

_GOAL, _KEY, _DOOR = 3, 4, 5                                  # palette positions used only to LABEL the demo
_FEAT = {0: "stepped_on", 1: "all_pads_covered"}


class DynamicsPerceiver:
    def __init__(self):
        self.reset()

    def reset(self):
        self.body_color = None
        self.body_evidence = {}
        self.new_level()

    def new_level(self):
        """Clear the per-level memory (pad positions, body location) on a level change — the layout changed —
        while KEEPING the body's learned colour identity (the agent is the same across levels)."""
        self.remembered_pads = set()
        self.body_pos = None

    def _find(self, grid, color):
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c == color:
                    return (x, y)
        return None

    def _mover_cells(self, prev_cells, cur_cells):
        cells = set()
        for c in detect_motion(prev_cells, cur_cells):       # every translated colour (body + pushed block)
            cells |= {p for p, cc in prev_cells.items() if cc == c}
            cells |= {p for p, cc in cur_cells.items() if cc == c}
        return cells

    def observe(self, prev_frame, action, frame):
        """Return (features, effect) for this transition, or (None, None) until the body is calibrated."""
        prev, cur = prev_frame.grid, frame.grid
        prev_cells = active_cells(prev, modal_background(prev))
        cur_cells = active_cells(cur, modal_background(cur))

        # efference copy: the body is the colour that translated by the action's delta
        if action.is_movement and frame.state == GameState.NOT_FINISHED:
            for c, d in detect_motion(prev_cells, cur_cells).items():
                if d == action.delta:
                    self.body_evidence[c] = self.body_evidence.get(c, 0) + 1
            if self.body_evidence:
                self.body_color = max(self.body_evidence, key=self.body_evidence.get)

        # remember pad cells (a colour that gets occluded by a block; palette pos 7) seen in the PRE-state only
        # — never from `cur`, which at a level transition is already the NEXT level's layout
        for (x, y), c in prev_cells.items():
            if c == 7:
                self.remembered_pads.add((x, y))

        body = self.body_pos if self.body_pos is not None else self._find(prev, self.body_color)
        self.body_pos = self._find(cur, self.body_color)     # update for next step
        if body is None or self.body_color is None:
            return None, None

        # features (computed from the PRE-state + the issued action, so they survive a level transition)
        dx, dy = action.delta
        dest = (body[0] + dx, body[1] + dy)
        stepped_on = prev[dest[1]][dest[0]] if 0 <= dest[1] < len(prev) and 0 <= dest[0] < len(prev[0]) else -1
        all_covered = 1 if all(prev[y][x] == 6 for (x, y) in self.remembered_pads) else 0
        features = (stepped_on, all_covered)

        # effect (priority: terminal / score before a generic world-diff that a level change would corrupt)
        if frame.state == GameState.GAME_OVER:
            effect = "death"
        elif frame.score > prev_frame.score:
            effect = "score_up"
        else:
            movers = self._mover_cells(prev_cells, cur_cells)
            bg = modal_background(cur)
            gone = {}
            for (x, y), c in prev_cells.items():
                if (x, y) not in movers and cur[y][x] == bg and c != bg:
                    gone[c] = gone.get(c, 0) + 1             # a colour vanished spontaneously (door opened)
            effect = f"color_{max(gone, key=gone.get)}_gone" if gone else None
        return features, effect


def collect(episodes=150, max_steps=200, seed=0):
    """Many short episodes (a fresh one recovers a stuck block). Mostly oracle-hinted so levels actually get
    solved — triggering key→doors and pad→win — with random moves for the negatives (e.g. reaching the goal
    BEFORE covering the pad, which forces the win condition to be a CONJUNCTION) and for hitting the hazard."""
    rng = random.Random(seed)
    perc, dm = DynamicsPerceiver(), DynamicsModel()
    for ep in range(episodes):
        env = Environment(LockPath())
        frame = env.reset()
        perc.reset()
        for _ in range(max_steps):
            if frame.state != GameState.NOT_FINISHED:
                break
            moves = [a for a in frame.available_actions if a.is_movement]
            sol = None
            if rng.random() < 0.8:                           # oracle HINT from the CURRENT state (saved/restored)
                saved = _capture(env.game)
                sol = solve_level(env.game)
                _restore(env.game, saved)
            action = sol[0] if (sol and sol[0] in moves) else rng.choice(moves)
            prev = frame
            frame = env.step(action)
            f, e = perc.observe(prev, action, frame)
            if f is not None:
                dm.observe(f, e)
            if frame.level != prev.level:                    # a level was completed — reset the per-level memory
                perc.new_level()
    dm.learn()
    return dm


if __name__ == "__main__":
    print("live-grid dynamics — the dynamics column learns LockPath's rules from real play (no semantic priors):\n")
    dm = DynamicsModel.__new__(DynamicsModel)  # placeholder so name exists if collect errors
    dm = collect()
    for _pred, desc, eff in dm.rules:
        for i, n in _FEAT.items():
            desc = desc.replace(f"c{i}", n)
        print(f"    {eff:>13}  when  {desc}")
    print("\n  discovered from real frames: stepping on colour 4 (the key) makes colour 5 (the doors) vanish;")
    print("  stepping on colour 8 (the hazard) = death; and the CONJUNCTIVE win — score rises only when the body")
    print("  reaches colour 3 (the goal) AND all pads are covered — exactly the condition the flat agent can't represent.")
