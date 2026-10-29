"""Goal worlds for testing planning (docs/AGENT.md part 29): hand-built games with levels and a WIN, played by the
whole ArcAgent the way the engine drives it (`frame_data()` mirrors the engine's FrameData: state, frame,
levels_completed, available_actions). A room of 3-pixel cells (13 x 13 with its walls) at the top left of the screen.

G1 push           the avatar (colour 9) pushes a box (12) by walking into it; a level is won when the box covers the
                  target cell (3). Walls (5) block both. Every layout is checked solvable when it is made.
G2 lock and exit  a wall of doors (8) splits the room; ACTION5 arms a hidden lock and ACTION7 opens it if armed
                  (W12's lock: 5 then 7); a level is won when the avatar stands on the exit cell (3), behind the doors.

The agent sees only the frames and the level count. `stuck()` (G1: no sequence of pushes, ignoring where the avatar
can go, brings the box onto the target) is measurement, never shown to the agent.
"""
from collections import deque

import numpy as np

N, C, LEVELS = 13, 3, 2
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
GOAL_WORLDS = ("G1 push", "G2 lock and exit")


class State:
    def __init__(self, name):
        self.name = name


class FrameData:
    def __init__(self, state, frame, levels, actions):
        self.state, self.frame, self.levels_completed, self.available_actions = State(state), [frame], levels, actions


class GoalWorld:
    def __init__(self, name, seed):
        self.name, self.tag, self.rng = name, name.split()[0], np.random.default_rng(seed)
        self.layouts = [self.layout() for _ in range(LEVELS)]
        self.level, self.started, self.won = 0, False, False
        self.load()

    # ---- layouts ------------------------------------------------------------------------------------------------
    def layout(self):
        while True:
            walls = {(x, y) for x in range(N) for y in range(N) if x in (0, N - 1) or y in (0, N - 1)}
            if self.tag == "G1":
                walls |= {(int(x), int(y)) for x, y in self.rng.integers(2, N - 2, (6, 2))}
                free = [(x, y) for x in range(1, N - 1) for y in range(1, N - 1) if (x, y) not in walls]
                i = self.rng.choice(len(free), 3, replace=False)
                avatar, box, target = (free[j] for j in i)
                lay = dict(walls=walls, avatar=avatar, box=box, target=target, doors=set())
                if self.solvable(lay):
                    return lay
            else:
                mid = N // 2
                doors = {(mid, y) for y in range(1, N - 1)}
                left = [(x, y) for x in range(1, mid) for y in range(1, N - 1)]
                right = [(x, y) for x in range(mid + 1, N - 1) for y in range(1, N - 1)]
                avatar = left[self.rng.integers(len(left))]
                target = right[self.rng.integers(len(right))]
                return dict(walls=walls, avatar=avatar, box=None, target=target, doors=doors)

    def pushes(self, lay, avatar, box, a):
        """G1's dynamics: (avatar, box) after action a."""
        dx, dy = MOVES[a]
        nxt = (avatar[0] + dx, avatar[1] + dy)
        if nxt in lay["walls"]:
            return avatar, box
        if nxt == box:
            nb = (box[0] + dx, box[1] + dy)
            if nb in lay["walls"]:
                return avatar, box
            return nxt, nb
        return nxt, box

    def solvable(self, lay, limit=200):
        start = (lay["avatar"], lay["box"])
        seen, q = {start: 0}, deque([start])
        while q:
            s = q.popleft()
            if s[1] == lay["target"]:
                return 4 <= seen[s] <= limit
            for a in MOVES:
                n = self.pushes(lay, *s, a)
                if n not in seen:
                    seen[n] = seen[s] + 1
                    q.append(n)
        return False

    def stuck(self):
        """G1: no sequence of pushes brings the box onto the target, wherever the avatar is (measurement only)."""
        if self.tag != "G1" or self.won:
            return False
        lay, seen, q = self.lay, {self.box}, deque([self.box])
        while q:
            b = q.popleft()
            if b == lay["target"]:
                return False
            for dx, dy in MOVES.values():
                behind, ahead = (b[0] - dx, b[1] - dy), (b[0] + dx, b[1] + dy)
                if behind not in lay["walls"] and ahead not in lay["walls"] and ahead not in seen:
                    seen.add(ahead)
                    q.append(ahead)
        return True

    # ---- play ---------------------------------------------------------------------------------------------------
    def load(self):
        self.lay = self.layouts[self.level]
        self.avatar, self.box, self.q = self.lay["avatar"], self.lay["box"], 0

    def actions(self):
        return [1, 2, 3, 4, 5, 7] if self.tag == "G2" else [1, 2, 3, 4]

    def frame(self):
        g = np.zeros((64, 64), np.int8)
        cell = lambda p, c: g.__setitem__((slice(p[1] * C, p[1] * C + C), slice(p[0] * C, p[0] * C + C)), c)
        for p in self.lay["walls"]:
            cell(p, 5)
        for p in self.lay["doors"]:
            cell(p, 8)
        cell(self.lay["target"], 3)
        if self.box is not None:
            cell(self.box, 12)
        cell(self.avatar, 9)
        return g

    def frame_data(self):
        state = "WIN" if self.won else ("NOT_FINISHED" if self.started else "NOT_PLAYED")
        return FrameData(state, self.frame(), self.level, self.actions())

    def step(self, a):
        """One engine step: 0 is RESET (restart this level); otherwise an action. Returns the new FrameData."""
        if a == 0:
            self.started = True
            self.load()
            return self.frame_data()
        if self.tag == "G1" and a in MOVES:
            self.avatar, self.box = self.pushes(self.lay, self.avatar, self.box, a)
            done = self.box == self.lay["target"]
        elif self.tag == "G2":
            if a in MOVES:
                dx, dy = MOVES[a]
                nxt = (self.avatar[0] + dx, self.avatar[1] + dy)
                if nxt not in self.lay["walls"] and (nxt not in self.lay["doors"] or self.q == 2):
                    self.avatar = nxt
            elif a in (5, 7):
                self.q = 1 if a == 5 else (2 if self.q == 1 else 0)
            done = self.avatar == self.lay["target"]
        else:
            done = False
        if done:
            self.level += 1
            if self.level == LEVELS:
                self.won = True
            else:
                self.load()
        return self.frame_data()
