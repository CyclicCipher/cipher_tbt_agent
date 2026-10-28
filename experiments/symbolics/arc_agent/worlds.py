"""Hand-built worlds where an action's effect depends on context (docs/AGENT.md part 20). Each is a 64 x 64 screen of
3-pixel cells (a 21 x 21 board, background 0). The avatar, a 3 x 3 block of colour 9, moves one cell per key
(1 up, 2 down, 3 left, 4 right); ACTION5 toggles the switch where there is one.

W1 walls        colour-5 cells block
W2 inverted     colour-5 cells are passable decoration; cells holding a small colour-14 dot block
W3 torus        leaving one side of the board re-enters at the opposite side
W4 ice          stepping onto ice (colour 11) slides the avatar 2 cells instead of 1
W5 one-way      colour-12 tiles can be entered only moving right
W6 speed switch while the switch (top-right corner: colour 10 off, 14 on) is on, every move is 2 cells
W7 switch door  colour-8 doors block unless the switch is on (a conjunction: door ahead AND switch off)
W8 rotating     an L-shaped piece: keys 1-4 move it one cell, ACTION5 turns it 90 degrees about its own pivot (part 12)
W9 ladders      "up" works only where the avatar's column x is 0 mod 3; nothing on screen marks it (a quotient, part 10)
W10 sliding     an ice board: the avatar slides in the key's direction until the cell before the first wall (a numeric
                relation: the slide is the distance to the wall)

Hidden state (part 27; docs/HIDDEN_STATE.md): nothing on the screen shows it.
W11 hidden toggle   colour-8 doors block unless a hidden switch is on; ACTION5 toggles it
W12 lock            ACTION5 arms (and closes), then ACTION7 opens if armed (closes otherwise): doors open only after 5 then 7
W13 mode            ACTION5 advances a hidden mode A -> B -> C -> A, ACTION7 returns it to A; doors are open in B and C
W14 replay          a second block (colour 12) repeats the avatar's move from 3 steps earlier (no tiles)
W15 clock door      colour-8 doors are open while (steps since the start) mod 8 < 4 (part 28: a clock, period 8)
W16 fuse            colour-8 doors are open for the 4 actions after each ACTION5 (part 28: a timer)

Leaving the board otherwise fails (the avatar stays), except in W3.
"""
import numpy as np

N, C = 21, 3                                            # board cells, pixels per cell
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
WORLDS = ("W1 walls", "W2 inverted", "W3 torus", "W4 ice", "W5 one-way", "W6 speed switch", "W7 switch door",
          "W8 rotating", "W9 ladders", "W10 sliding")
HIDDEN_WORLDS = ("W11 hidden toggle", "W12 lock", "W13 mode", "W14 replay", "W15 clock door", "W16 fuse")
PIECE = [(0, -1), (0, 0), (0, 1), (1, 1)]               # W8's L, around its pivot (0, 0)


class World:
    def __init__(self, name, rng):
        self.name, self.rng = name, rng
        self.tiles = {}                                 # (x, y) -> colour
        self.tag = name.split()[0]
        density = {"W1": 0.2, "W2": 0.15, "W4": 0.25, "W5": 0.25, "W7": 0.2, "W10": 0.15, "W11": 0.2, "W12": 0.2,
                   "W13": 0.2, "W15": 0.2, "W16": 0.2}.get(self.tag, 0.0)
        colour = {"W1": 5, "W2": 14, "W4": 11, "W5": 12, "W7": 8, "W10": 5, "W11": 8, "W12": 8, "W13": 8, "W15": 8,
                  "W16": 8}.get(self.tag)
        for x in range(N):
            for y in range(N):
                if (x, y) != (10, 10) and colour is not None and rng.random() < density:
                    self.tiles[(x, y)] = colour
        if name.startswith("W2"):                       # passable decoration in the colour W1 uses for walls
            for x in range(N):
                for y in range(N):
                    if (x, y) not in self.tiles and (x, y) != (10, 10) and rng.random() < 0.3:
                        self.tiles[(x, y)] = 5
        self.switched = self.tag in ("W6", "W7")
        self.turn = 0                                   # W8: the piece's quarter turns
        if self.switched:
            self.tiles.pop((N - 1, 0), None)
        self.avatar, self.on = (10, 10), False
        self.q, self.follower, self.past = 0, (4, 4), []   # W11-W13: the hidden machine; W14: the second block
        self.steps, self.fuse = 0, 0                      # W15: the steps taken; W16: actions left on the fuse

    def actions(self):
        if self.tag in ("W12", "W13"):
            return [1, 2, 3, 4, 5, 7]
        return [1, 2, 3, 4, 5] if (self.switched or self.tag in ("W8", "W11", "W16")) else [1, 2, 3, 4]

    def piece(self, at=None, turn=None):
        ax, ay = self.avatar if at is None else at
        t = self.turn if turn is None else turn
        out = []
        for x, y in PIECE:
            for _ in range(t % 4):
                x, y = -y, x
            out.append((ax + x, ay + y))
        return out

    def frame(self):
        g = np.zeros((64, 64), np.int8)
        for (x, y), c in self.tiles.items():
            if c == 14:                                 # a small dot in the middle of the cell
                g[y * C + 1, x * C + 1] = 14
            else:
                g[y * C:y * C + C, x * C:x * C + C] = c
        if self.switched:
            g[0:C, (N - 1) * C:N * C] = 14 if self.on else 10
        if self.tag == "W10":                           # the ice board under everything else
            ice = g == 0
            ice[N * C:, :] = False
            ice[:, N * C:] = False
            g[ice] = 11
        if self.tag == "W14":
            fx, fy = self.follower
            g[fy * C:fy * C + C, fx * C:fx * C + C] = 12
        for ax, ay in (self.piece() if self.tag == "W8" else [self.avatar]):
            g[ay * C:ay * C + C, ax * C:ax * C + C] = 9
        return g

    def open(self):
        """W11-W13: whether the hidden machine lets doors pass."""
        return {"W11": self.q == 1, "W12": self.q == 2, "W13": self.q in (1, 2), "W15": self.steps % 8 < 4,
                "W16": self.fuse > 0}[self.tag]

    def passable(self, cell, a):
        x, y = cell
        if self.switched and cell == (N - 1, 0):
            return False                                # the switch's cell
        t = self.tiles.get(cell)
        if self.tag == "W1":                            # (W10-W14 also start with "W1")
            return t != 5
        if self.name.startswith("W2"):
            return t != 14
        if self.name.startswith("W5"):
            return t != 12 or a == 4
        if self.name.startswith("W7"):
            return t != 8 or self.on
        if self.tag in ("W11", "W12", "W13", "W15", "W16"):
            return t != 8 or self.open()
        if self.tag == "W14" and cell == self.follower:
            return False
        return True

    def step(self, a):
        out = self._step(a)
        self.steps += 1
        if self.tag == "W16":                           # the fuse burns one action per action, and ACTION5 relights it
            self.fuse = 4 if a == 5 else max(0, self.fuse - 1)
        return out

    def _step(self, a):
        if self.tag == "W8":
            at, turn = self.avatar, self.turn
            if a == 5:
                turn = (turn + 1) % 4
            else:
                at = (at[0] + MOVES[a][0], at[1] + MOVES[a][1])
            if all(0 <= x < N and 0 <= y < N for x, y in self.piece(at, turn)):
                self.avatar, self.turn = at, turn
            return self.frame()
        if self.tag == "W9":
            dx, dy = MOVES[a] if a in MOVES else (0, 0)
            x, y = self.avatar
            if a == 1 and x % 3 != 0:                   # an invisible ladder only every third column
                return self.frame()
            if 0 <= x + dx < N and 0 <= y + dy < N:
                self.avatar = (x + dx, y + dy)
            return self.frame()
        if self.tag == "W10":
            dx, dy = MOVES[a]
            x, y = self.avatar
            while 0 <= x + dx < N and 0 <= y + dy < N and self.tiles.get((x + dx, y + dy)) != 5:
                x, y = x + dx, y + dy
            self.avatar = (x, y)
            return self.frame()
        if self.tag == "W14":                           # the second block repeats the move from 3 steps earlier
            lag = self.past[-3] if len(self.past) >= 3 else None
            self.past.append(a)
            if lag is not None:
                fx, fy = self.follower[0] + MOVES[lag][0], self.follower[1] + MOVES[lag][1]
                if 0 <= fx < N and 0 <= fy < N and (fx, fy) != self.avatar:
                    self.follower = (fx, fy)
        if a in (5, 7) and self.tag in ("W11", "W12", "W13"):
            if self.tag == "W11" and a == 5:
                self.q = 1 - self.q
            elif self.tag == "W12":
                self.q = 1 if a == 5 else (2 if self.q == 1 else 0)
            elif self.tag == "W13":
                self.q = (self.q + 1) % 3 if a == 5 else 0
            return self.frame()
        if a == 5:
            if self.switched:
                self.on = not self.on
            return self.frame()
        dx, dy = MOVES[a]
        x, y = self.avatar
        steps = 2 if (self.name.startswith("W6") and self.on) else 1
        if self.name.startswith("W4") and self.tiles.get((x + dx, y + dy)) == 11:
            steps = 2
        target = (x + steps * dx, y + steps * dy)
        if self.name.startswith("W3"):
            target = (target[0] % N, target[1] % N)
        if not (0 <= target[0] < N and 0 <= target[1] < N):
            if steps == 2 and self.name.startswith("W4"):   # sliding off the board: the slide stops on the ice
                target = (x + dx, y + dy)
            else:
                return self.frame()
        if all(self.passable((x + k * dx, y + k * dy) if not self.name.startswith("W3") else target, a)
               for k in range(1, steps + 1)):
            self.avatar = target
        return self.frame()
