"""The arc_agent, as one game-playing object: `ArcAgent(game_id).act(frame_data) -> (action id, data)`.

A thin driver: it hands each transition to the brain, tells the executive whether a WIN happened, asks the executive
for the next action, and keeps the engine's rules (RESET after GAME_OVER; never RESET before acting in a level, since
the engine would restart the whole game).
"""
import json
import os
import random
from collections import Counter

import numpy as np

from arc_agent.blueprint import DEFAULT
from arc_agent.brain import Brain, size_bucket
from arc_agent.executive import Executive

RESET, CLICK = 0, 6
PRIORS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "priors.json")


def shipped_priors():
    """The goal-type draws from warm-up play (warmup.py), if they were shipped."""
    try:
        return json.load(open(PRIORS))["draws"]
    except (OSError, ValueError, KeyError):
        return []


def region_size(grid, x, y, cap=200):
    c, seen, q = grid[y, x], {(x, y)}, [(x, y)]
    while q and len(seen) < cap:
        px, py = q.pop()
        for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
            if (nx, ny) not in seen and 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0] and grid[ny, nx] == c:
                seen.add((nx, ny))
                q.append((nx, ny))
    return len(seen)


class ArcAgent:
    def __init__(self, game_id="", blueprint=DEFAULT, seed=0, type_draws=None):
        self.game_id = game_id
        self.brain = Brain(blueprint)
        self.executive = Executive(self.brain, shipped_priors() if type_draws is None else type_draws)
        self.level, self.prev, self.acted = None, None, 0
        self.expected = None
        self.rng = random.Random(seed)
        self.notes = Counter()

    def kind(self, action, grid):
        aid, x, y = action
        if aid != CLICK:
            return ("key", aid)
        return ("click", int(grid[y, x]), size_bucket(region_size(grid, x, y)))

    def act(self, fd):
        state = getattr(fd.state, "name", str(fd.state))
        grid = np.array(fd.frame[-1], dtype=np.int8, copy=True) if len(fd.frame) else None
        if state in ("NOT_PLAYED", "GAME_OVER") or grid is None:
            self.prev, self.acted, self.executive.plan = None, 0, []
            self.notes["reset (game over)" if state == "GAME_OVER" else "reset (start)"] += 1
            return RESET, None
        if self.level != fd.levels_completed:          # a new level: the last action won the previous one
            if self.prev is not None and self.level is not None and fd.levels_completed > self.level:
                pgrid, paction, _ = self.prev
                states, _ = self.brain.model_step(dict(self.brain.joint()), paction, self.brain.static_scene())
                self.executive.observe(win_states=states)
                self.notes["win"] += 1
            self.level = fd.levels_completed
            self.executive.since = 0
            self.brain.start(grid, new_level=True)
            self.executive.plan, self.prev, self.acted = [], None, 0
            self.executive.start_grid = grid.copy()
        elif self.prev is None:                         # the level restarted
            self.brain.start(grid, new_level=False)
            self.executive.plan = []
        else:
            pgrid, paction, pkind = self.prev
            report = self.brain.observe(pgrid, paction, grid, pkind)
            self.executive.observe()
            if report["recruited"] or report["lost"]:
                self.executive.plan = []
            else:
                self.executive.surprise(self.expected)
            self.notes["recruited"] += len(report["recruited"])
        self.executive.can_reset = self.acted > 0         # the engine restarts the whole game otherwise
        choice = self.executive.decide(list(fd.available_actions))
        if choice is not None and choice[0][0] == RESET:  # a plan that uses RESET (its effect is in the model)
            self.prev, self.acted, self.executive.plan = None, 0, []
            self.notes["reset (planned)"] += 1
            return RESET, None
        if choice is None:
            choice = self.executive.fallback(list(fd.available_actions))
            if choice is not None:
                self.notes["fallback"] += 1
        if choice is None:
            avail = [a for a in fd.available_actions if a not in (RESET, CLICK)]
            if avail:
                action = (self.rng.choice(avail), None, None)
            else:
                ys, xs = np.nonzero(grid != self.brain.v1.ground)
                i = self.rng.randrange(len(xs)) if len(xs) else 0
                action = (CLICK, int(xs[i]) if len(xs) else 32, int(ys[i]) if len(ys) else 32)
            self.expected = None
            self.notes["random"] += 1
        else:
            action, self.expected = choice
            self.notes[self.executive.kind_of_plan] += 1
        self.prev = (grid, action, self.kind(action, grid))
        self.acted += 1
        self.executive.since += 1
        aid, x, y = action
        return aid, ({"x": x, "y": y} if aid == CLICK else None)
