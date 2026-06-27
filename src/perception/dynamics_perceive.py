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

Run:  PYTHONPATH=src python -m perception.dynamics_perceive
"""

from __future__ import annotations

import os
import random
import sys

from tbt.dynamics import DynamicsModel                       # noqa: E402

from tasks import Environment, GameAction, GameState     # noqa: E402
from tasks.games import LockPath                         # noqa: E402

from .goal_discover import GoalModel                         # noqa: E402  (F: goal + pad from the score)
from .object_perceiver import ObjectPerceiver               # noqa: E402  (E: body + pushable from motion)
from .perceive import active_cells, detect_motion, modal_background   # noqa: E402

_FEAT = {0: "stepped_on"}
_PALETTE = 16                                                 # ARC's colour count: presence-context = present(0..15)


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
        prev_bg, cur_bg = modal_background(prev), modal_background(cur)
        prev_cells = active_cells(prev, prev_bg)
        cur_cells = active_cells(cur, cur_bg)

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
        present = {c for c in prev_cells.values()} - {self.body_color}
        features = (stepped_on,) + tuple(1 if c in present else 0 for c in range(_PALETTE))   # + presence-context:
        #   the per-colour presence makes a CONDITIONAL effect expressible (a switch flips a door only if present)

        # effect (terminal / score first, then a generic world-diff a level change would corrupt). The diff is now
        # SYMMETRIC: a colour can vanish (door opens) OR appear (door closes); both are occlusion-safe (a mover or
        # the body never counts), so the toggle's switch<->door is two conditional rules, not effects={}.
        if frame.state == GameState.GAME_OVER:
            effect = "death"
        elif frame.score > prev_frame.score:
            effect = "score_up"
        else:
            movers = self._mover_cells(prev_cells, cur_cells)
            gone = {}
            for (x, y), c in prev_cells.items():
                if (x, y) not in movers and cur[y][x] == cur_bg and c != cur_bg:
                    gone[c] = gone.get(c, 0) + 1             # a colour vanished where it was (door opened)
            appeared = {}
            for (x, y), c in cur_cells.items():
                if (x, y) not in movers and prev[y][x] == prev_bg and c != self.body_color:
                    appeared[c] = appeared.get(c, 0) + 1     # a colour materialised on bare background (door closed)
            effect = (f"color_{max(gone, key=gone.get)}_gone" if gone else
                      f"color_{max(appeared, key=appeared.get)}_appeared" if appeared else None)
        return features, effect, present


def collect(episodes=150, max_steps=120, seed=0, game_cls=LockPath):
    """One shared play loop → three learners: the DynamicsModel (cause→effect), the ObjectPerceiver (E: body +
    pushable), the GoalModel (F: goal + context). Exploration is SELF-DIRECTED random play — NO oracle teacher
    (it was both the speed bottleneck — a per-divergence BFS that explodes with movable pieces — and a crutch
    that demonstrated wins instead of letting the architecture discover them). The agent generates its own
    experience; the only signals are the env's frames + sparse score. Effects are triggered by exploration; F's
    wins/negatives come from STUMBLING on the goal — so a mechanic whose win needs a precise act (cover a pad,
    tour every item) is a genuine exploration challenge, not a solved one. Returns (dm, object_perceiver, goal)."""
    rng = random.Random(seed)
    perc, dm = DynamicsPerceiver(), DynamicsModel()
    objp, goal = ObjectPerceiver(), GoalModel()
    for ep in range(episodes):
        env = Environment(game_cls())
        frame = env.reset()
        perc.reset()
        for _ in range(max_steps):
            if frame.state != GameState.NOT_FINISHED:
                break
            moves = [a for a in frame.available_actions if a.is_movement]
            action = rng.choice(moves)                       # self-directed random exploration (no teacher)
            prev = frame
            frame = env.step(action)
            f, e, present = perc.observe(prev, action, frame)
            if f is not None:
                dm.observe(f, e)                             # the cause→effect rules (key→doors, hazard→death)
                if action.is_movement and prev.state == GameState.NOT_FINISHED and frame.level == prev.level:
                    objp.observe(prev.grid, action.delta, frame.grid)        # E: body + pushable, from motion
                stepped_on = f[0]                                            # F: goal + context, from the score
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
