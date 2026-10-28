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

Leaving the board otherwise fails (the avatar stays), except in W3.
"""
import numpy as np

N, C = 21, 3                                            # board cells, pixels per cell
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
WORLDS = ("W1 walls", "W2 inverted", "W3 torus", "W4 ice", "W5 one-way", "W6 speed switch", "W7 switch door")


class World:
    def __init__(self, name, rng):
        self.name, self.rng = name, rng
        self.tiles = {}                                 # (x, y) -> colour
        density = {"W1": 0.2, "W2": 0.15, "W4": 0.25, "W5": 0.25, "W7": 0.2}.get(name[:2], 0.0)
        colour = {"W1": 5, "W2": 14, "W4": 11, "W5": 12, "W7": 8}.get(name[:2])
        for x in range(N):
            for y in range(N):
                if (x, y) != (10, 10) and colour is not None and rng.random() < density:
                    self.tiles[(x, y)] = colour
        if name.startswith("W2"):                       # passable decoration in the colour W1 uses for walls
            for x in range(N):
                for y in range(N):
                    if (x, y) not in self.tiles and (x, y) != (10, 10) and rng.random() < 0.3:
                        self.tiles[(x, y)] = 5
        self.switched = name[:2] in ("W6", "W7")
        if self.switched:
            self.tiles.pop((N - 1, 0), None)
        self.avatar, self.on = (10, 10), False

    def actions(self):
        return [1, 2, 3, 4, 5] if self.switched else [1, 2, 3, 4]

    def frame(self):
        g = np.zeros((64, 64), np.int8)
        for (x, y), c in self.tiles.items():
            if c == 14:                                 # a small dot in the middle of the cell
                g[y * C + 1, x * C + 1] = 14
            else:
                g[y * C:y * C + C, x * C:x * C + C] = c
        if self.switched:
            g[0:C, (N - 1) * C:N * C] = 14 if self.on else 10
        ax, ay = self.avatar
        g[ay * C:ay * C + C, ax * C:ax * C + C] = 9
        return g

    def passable(self, cell, a):
        x, y = cell
        if self.switched and cell == (N - 1, 0):
            return False                                # the switch's cell
        t = self.tiles.get(cell)
        if self.name.startswith("W1"):
            return t != 5
        if self.name.startswith("W2"):
            return t != 14
        if self.name.startswith("W5"):
            return t != 12 or a == 4
        if self.name.startswith("W7"):
            return t != 8 or self.on
        return True

    def step(self, a):
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
