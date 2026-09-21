"""ZipLearner plays the ARC-AGI-3 replica games (src/tasks/games): the interface, and the first agent behind it.

The interface (DESIGN.md §16 applied to a game): a frame is the state; the game's available actions are the actions;
the score and WIN / GAME_OVER signals are the only feedback; nothing about any game is known in advance. Everything
the agent learns is a LOCAL RULE learned by counting with the rate price -- the same core loop as everywhere:

  world model   one description per action: "the new colour of a cell is a function of the window around it" -- a
                table over windows, the same table at every cell (§5's tied blocks: translation invariance). Two
                window sizes (radius 1 and 2) compete on price, so the description that compresses best decides how
                much context the rule needs (a push needs more than a step). A window never seen predicts "unchanged"
                and is counted as UNKNOWN -- generalise, then correct.
  goal model    the windows present around the cells that changed at a transition that raised the score (win keys,
                per action) or ended the game (death keys). A (frame, action) is predicted to win / kill when a
                window of the frame is a known win / death key for that action.
  planner       breadth-first search over PREDICTED frames to a (frame, action) predicted to win, avoiding predicted
                deaths; with no known goal, to the nearest frame that still has unknown windows (bits to learn) or
                was never visited (novelty) -- exploration as the price of ignorance, §16.
  loop          plan, execute one step, observe (the transition trains the world and goal models), re-plan when the
                observation differs from the prediction.

No coordinates (ACTION6) yet: the games here do not use it. The frame is cropped to the bounding box of its non-zero
cells plus a margin, fixed per level -- a constant background costs nothing and is not looked at.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from ziplearner import flag_bits   # noqa: E402
from tasks.core import GameAction, GameState   # noqa: E402

V = 16                              # ARC colours
BORDER = -1


class LocalRule:
    """One action's effect as a table from windows to the centre cell's new colour, shared by every cell."""

    def __init__(self, radius):
        self.r = radius
        self.table = {}                                 # window -> {colour: count}
        self.stats = {}                                 # window -> [n_right, n_wrong]
        self.cost = 0.0

    def windows(self, frame):
        r = self.r
        H, W = frame.shape
        pad = np.full((H + 2 * r, W + 2 * r), BORDER, dtype=np.int16)
        pad[r:r + H, r:r + W] = frame
        out = {}
        for i in range(H):
            for j in range(W):
                out[(i, j)] = pad[i:i + 2 * r + 1, j:j + 2 * r + 1].tobytes()
        return out

    def observe(self, before, after):
        win = self.windows(before)
        for (i, j), k in win.items():
            y = int(after[i, j])
            e = self.table.get(k)
            if e is None:                               # a new window: its outcome is a parameter
                self.cost += math.log2(V)
                self.table[k] = {y: 1}
                self.stats[k] = [0, 0]
                continue
            pred = max(e, key=e.get)
            st = self.stats[k]
            if pred == y:
                self.cost += flag_bits(False, st[0], st[1])
                st[0] += 1
            else:
                self.cost += flag_bits(True, st[0], st[1]) + math.log2(V - 1)
                st[1] += 1
            e[y] = e.get(y, 0) + 1

    def predict(self, frame):
        """The predicted frame and the number of cells whose window was never seen (left unchanged)."""
        win = self.windows(frame)
        out = frame.copy()
        unknown = 0
        for (i, j), k in win.items():
            e = self.table.get(k)
            if e is None:
                unknown += 1
            else:
                out[i, j] = max(e, key=e.get)
        return out, unknown


class ActionModel:
    """Two window sizes compete on price; the cheapest description predicts."""

    def __init__(self, radii=(1, 2)):
        self.rules = {r: LocalRule(r) for r in radii}
        self.n_obs = 0

    def observe(self, before, after):
        for rule in self.rules.values():
            rule.observe(before, after)
        self.n_obs += 1

    def best(self):
        return min(self.rules.values(), key=lambda rule: (round(rule.cost + math.log2(len(self.rules)), 9), rule.r))

    def predict(self, frame):
        if self.n_obs == 0:
            return None, None
        return self.best().predict(frame)


class GoalModel:
    """Win and death keys: the windows around the cells that changed at a scoring / fatal transition."""

    def __init__(self, radius=1):
        self.r = radius
        self.win, self.death = {}, {}                   # action -> set of windows
        self.probe = LocalRule(radius)

    def record(self, action, before, after, outcome, recent=()):
        """Keys = the windows (in the frame before the move) around the cells that changed -- or, when the after-frame
        is not observable (a win moves to the next level), around the cells the model predicted to change and the
        cells that changed in the PREVIOUS transition: where things were happening. Never every window: a goal key
        that matches everywhere is a goal that fires everywhere (the first run's failure)."""
        changed = set(zip(*np.nonzero(before != after))) | set(recent)
        if not changed:
            return
        win = self.probe.windows(before)
        keys = {win[(int(i), int(j))] for i, j in changed}
        target = self.win if outcome == "win" else self.death
        target.setdefault(action, set()).update(keys)

    def predicts(self, which, action, frame):
        keys = (self.win if which == "win" else self.death).get(action)
        if not keys:
            return False
        return any(k in keys for k in self.probe.windows(frame).values())

    def known(self):
        return any(self.win.values())


class Player:
    """The loop of §16 on a game: describe, plan, act, learn from what happened."""

    def __init__(self, actions, max_depth=14, max_nodes=4000):
        self.actions = list(actions)
        self.models = {a: ActionModel() for a in self.actions}
        self.goal = GoalModel()
        self.max_depth, self.max_nodes = max_depth, max_nodes
        self.visited = set()
        self.last_changed = set()
        self.plan = []
        self.stats = dict(predictions=0, correct=0, unknown_cells=0, cells=0, replans=0, explore_plans=0, goal_plans=0)

    def frame_key(self, frame):
        return frame.tobytes()

    def observe(self, action, before, after, outcome):
        if outcome != "win":                                  # a winning move's after-frame is the next level: not seen
            self.models[action].observe(before, after)
        if outcome in ("win", "death"):
            self.goal.record(action, before, after, outcome, recent=self.last_changed)
        self.last_changed = set(zip(*np.nonzero(before != after)))
        self.visited.add(self.frame_key(after))

    def predict(self, action, frame):
        pred, unknown = self.models[action].predict(frame)
        return pred, unknown

    def search(self, frame, want_goal):
        """Breadth-first over predicted frames. want_goal: a (frame, action) the goal model predicts to win; otherwise
        the nearest frame with unknown windows under some action, or never visited."""
        start_key = self.frame_key(frame)
        seen = {start_key: []}
        frontier = deque([(frame, [])])
        nodes = 0
        while frontier and nodes < self.max_nodes:
            f, path = frontier.popleft()
            if len(path) >= self.max_depth:
                continue
            for a in self.actions:
                nodes += 1
                if self.goal.predicts("death", a, f):
                    continue
                if want_goal and self.goal.predicts("win", a, f):
                    return path + [a], "goal"
                pred, unknown = self.predict(a, f)
                if pred is None or unknown:                      # an effect not yet described: bits to learn
                    if not want_goal:
                        return path + [a], "explore"
                    if pred is None:
                        continue
                k = self.frame_key(pred)
                if k in seen:
                    continue
                seen[k] = path + [a]
                if not want_goal and k not in self.visited:      # a frame never visited: novelty
                    return path + [a], "explore"
                frontier.append((pred, path + [a]))
        return None, None

    def choose(self, frame):
        if self.plan:
            return self.plan.pop(0)
        plan, kind = self.search(frame, want_goal=self.goal.known())
        if plan is None:
            plan, kind = self.search(frame, want_goal=False)
        if plan is None:                                             # nothing known and nothing new: any action
            plan, kind = [self.actions[len(self.visited) % len(self.actions)]], "blind"
        self.stats["goal_plans" if kind == "goal" else "explore_plans"] += 1
        self.plan = plan[1:]
        return plan[0]


def crop_box(frame):
    nz = np.argwhere(frame != 0)
    if len(nz) == 0:
        return (0, 0, frame.shape[0], frame.shape[1])
    (i0, j0), (i1, j1) = nz.min(0), nz.max(0)
    m = 1
    return (max(0, i0 - m), max(0, j0 - m), min(frame.shape[0], i1 + 1 + m), min(frame.shape[1], j1 + 1 + m))


def play(env, budget_per_level=200, max_levels=None, seed=0, verbose=False):
    """Drive one Environment with a Player. Returns per-level results and the player's statistics."""
    fd = env.reset()
    actions = [a for a in env.game.available_actions()]
    player = Player(actions)
    results = []
    level = env.level
    used = 0
    box = crop_box(np.array(fd.grid))
    n_levels = env.game.level_count if max_levels is None else min(max_levels, env.game.level_count)
    while env.state not in (GameState.WIN,) and level < n_levels:
        frame = np.array(fd.grid, dtype=np.int16)[box[0]:box[2], box[1]:box[3]]
        a = player.choose(frame)
        pred, _unknown = player.predict(a, frame)
        fd = env.step(a)
        used += 1
        after_full = np.array(fd.grid, dtype=np.int16)
        if env.level != level or env.state == GameState.WIN:              # the level was completed by this action;
            guess = pred if pred is not None else frame                    # the next frame is another level, so the
            player.observe(a, frame, guess, "win")                          # changed cells are the model's prediction
            results.append(dict(level=level, solved=True, actions=used))
            if verbose:
                print(f"   level {level} solved in {used} actions")
            level, used = env.level, 0
            player.plan, player.visited = [], set()
            if env.state != GameState.WIN:
                box = crop_box(after_full)
            continue
        after = after_full[box[0]:box[2], box[1]:box[3]]
        if after.shape != frame.shape:                                      # the playfield grew: re-crop next step
            box = crop_box(after_full)
            after = after_full[box[0]:box[2], box[1]:box[3]]
        if env.state == GameState.GAME_OVER:
            player.observe(a, frame, after, "death")
            fd = env.step(GameAction.RESET)
            used += 1
            player.plan = []
            continue
        player.observe(a, frame, after, "none")
        if pred is not None and pred.shape == after.shape:
            player.stats["predictions"] += 1
            player.stats["correct"] += int(np.array_equal(pred, after))
            player.stats["cells"] += int(after.size)
            player.stats["unknown_cells"] += int(_unknown or 0)
            if not np.array_equal(pred, after):
                player.plan = []
                player.stats["replans"] += 1
        if used >= budget_per_level:
            results.append(dict(level=level, solved=False, actions=used))
            if verbose:
                print(f"   level {level} NOT solved in {used} actions")
            break
    for r in results:
        r["model"] = {a.name: {"radius": player.models[a].best().r if player.models[a].n_obs else None,
                               "windows": len(player.models[a].best().table) if player.models[a].n_obs else 0}
                      for a in player.actions}
    return results, player.stats
