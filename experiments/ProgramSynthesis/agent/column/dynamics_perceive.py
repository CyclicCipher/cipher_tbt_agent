"""Live-grid learning — one shared play loop grounds the dynamics + perception (E) + value (F) in real play.

From each (prev_frame, action, frame) the agent perceives, with NO semantic priors:
  - body identity = the EFFERENCE COPY (the colour that translates by the issued action's delta);
  - `stepped_on` = the colour at the body's intended destination (generic);
  - the dynamics EFFECT = death (GAME_OVER), score_up, or `color_C_gone` (a colour vanished — doors opening),
    characterised generically, never as "door"/"key".
One shared loop drives THREE learners: the DynamicsModel (cause→effect rules, tbt/dynamics.py), the
ObjectPerceiver (E: body + the pushable piece, from motion — object_perceiver.py), and the GoalModel (F: the
goal + its conjunctive context / the pad, from the sparse score — goal_discover.py). The conjunctive win is no
longer a hand-coded pad=7 / block=6 feature; it FALLS OUT of the score (F). An oracle-hinted + random explorer
triggers the dynamics (the agent never sees the oracle's labels — it only generates experience to learn from).

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

from .goal_discover import GoalModel                         # noqa: E402  (F: goal + pad from the score)
from .object_perceiver import ObjectPerceiver               # noqa: E402  (E: body + pushable from motion)
from .perceive import active_cells, detect_motion, modal_background   # noqa: E402

_FEAT = {0: "stepped_on"}


class DynamicsPerceiver:
    def __init__(self):
        self.reset()

    def reset(self):
        self.body_color = None
        self.body_evidence = {}
        self.new_level()

    def new_level(self):
        """Clear the per-level body location on a level change (the layout changed) while KEEPING the body's
        learned colour identity (the agent is the same across levels)."""
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
        """Return (features, effect, present) for this transition, or (None, None, None) until calibrated.
        `present` = the non-background, non-body colours in the pre-state — the context F differences."""
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

        body = self.body_pos if self.body_pos is not None else self._find(prev, self.body_color)
        self.body_pos = self._find(cur, self.body_color)     # update for next step
        if body is None or self.body_color is None:
            return None, None, None

        # features (computed from the PRE-state + the issued action, so they survive a level transition)
        dx, dy = action.delta
        dest = (body[0] + dx, body[1] + dy)
        stepped_on = prev[dest[1]][dest[0]] if 0 <= dest[1] < len(prev) and 0 <= dest[0] < len(prev[0]) else -1
        features = (stepped_on,)
        present = {c for c in prev_cells.values()} - {self.body_color}

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
        return features, effect, present


def collect(episodes=150, max_steps=200, seed=0, game_cls=LockPath):
    """One shared play loop → three learners: the DynamicsModel (cause→effect), the ObjectPerceiver (E: body +
    pushable), the GoalModel (F: goal + pad). Oracle-hinted so levels get won (key→doors, pad→win) with random
    moves for the negatives (reaching the goal BEFORE covering a pad — what forces the conjunctive win) and for
    hitting the hazard. Generic over `game_cls` (any Game with a snapshot/restore for the oracle). Returns
    (dynamics_model, object_perceiver, goal_model)."""
    rng = random.Random(seed)
    perc, dm = DynamicsPerceiver(), DynamicsModel()
    objp, goal = ObjectPerceiver(), GoalModel()
    for ep in range(episodes):
        env = Environment(game_cls())
        frame = env.reset()
        perc.reset()
        plan, level = [], frame.level                        # cached oracle plan; replan only on a cache miss
        for _ in range(max_steps):
            if frame.state != GameState.NOT_FINISHED:
                break
            if frame.level != level:                         # the layout changed -> the cached plan is stale
                plan, level = [], frame.level
            moves = [a for a in frame.available_actions if a.is_movement]
            hint = rng.random() < 0.8
            if hint and not plan:                            # (re)solve ONLY on a cache miss, not every step:
                saved = _capture(env.game)                   # following the cached optimal plan is the same
                plan = solve_level(env.game) or []           # hint without re-running the BFS each step (which
                _restore(env.game, saved)                    # explodes with the number of movable pieces)
            action = plan[0] if (hint and plan and plan[0] in moves) else rng.choice(moves)
            plan = plan[1:] if (plan and action == plan[0]) else []   # advance along / drop the cached plan
            prev = frame
            frame = env.step(action)
            f, e, present = perc.observe(prev, action, frame)
            if f is not None:
                dm.observe(f, e)                             # the cause→effect rules (key→doors, hazard→death)
                if action.is_movement and prev.state == GameState.NOT_FINISHED and frame.level == prev.level:
                    objp.observe(prev.grid, action.delta, frame.grid)        # E: body + pushable, from motion
                stepped_on = f[0]                                            # F: goal + pad, from the score
                if e == "score_up":
                    goal.observe_win(present, stepped_on)
                elif stepped_on in goal.goal_colors:
                    goal.observe_reach_no_win(present)
            if frame.level != prev.level:                    # a level was completed — reset the per-level memory
                perc.new_level()
                objp.new_level()
    dm.learn()
    return dm, objp, goal


if __name__ == "__main__":
    print("live-grid learning: one shared play loop grounds the dynamics + E (body/pushable) + F (goal/pad):\n")
    dm, objp, goal = collect()
    for _pred, desc, eff in dm.rules:
        for i, n in _FEAT.items():
            desc = desc.replace(f"c{i}", n)
        print(f"    {eff:>13}  when  {desc}")
    print(f"\n  E (perception): body colour {objp.body_color}, pushable {sorted(objp.pushable)}")
    print(f"  F (value):      goal {sorted(goal.goal_colors)}, required-absent / pad {sorted(goal.required_absent())}")
    print("\n  all from real frames + the sparse score, no colour priors: key->doors (color_gone) and hazard->death")
    print("  from the dynamics; body+pushable from motion (E); the conjunctive win goal+pad from the score (F).")
