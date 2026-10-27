"""
arc3_worlds.py
Collection of ARC-AGI-3 test worlds.
Contains:
1. 'nav_simple'   : Spatial movement, obstacle avoidance, and goal portal discovery.
2. 'push_block'   : Dynamic object interaction (pushing a block onto a designated target).
3. 'click_repair' : Complex coordinate action (ACTION6) to find and fix broken grid symmetry.
"""

from arc3_engine import ARCBaseWorld, GameAction, GameState, register_world, make_world, list_worlds
from typing import Dict, Any, Tuple


# =====================================================================
# World 1: nav_simple (Navigation & Goal Induction)
# Core prior: Spatial translation, obstacle barriers, and target absorption.
# =====================================================================
@register_world("nav_simple")
class NavSimpleWorld(ARCBaseWorld):
    def __init__(self, width=10, height=10):
        super().__init__(width=width, height=height, max_steps=40)
        self.player_pos = [1, 1]
        self.goal_pos = [8, 8]
        self.walls = set()

    def setup_world(self):
        self.clear_grid(fill_color=0)  # Black background
        self.player_pos = [1, 1]
        self.goal_pos = [self.width - 2, self.height - 2]

        # Outer walls (Grey = 5)
        self.walls = set()
        for x in range(self.width):
            self.walls.add((x, 0))
            self.walls.add((x, self.height - 1))
        for y in range(self.height):
            self.walls.add((0, y))
            self.walls.add((self.width - 1, y))

        # An internal obstacle barrier
        for y in range(2, 7):
            self.walls.add((5, y))

        self._render()

    def _render(self):
        self.clear_grid(0)
        for wx, wy in self.walls:
            self.set_cell(wx, wy, 5)  # Walls: Grey (5)
        self.set_cell(self.goal_pos[0], self.goal_pos[1], 3)  # Goal: Green (3)
        self.set_cell(self.player_pos[0], self.player_pos[1], 1)  # Player: Blue (1)

    def update_world(self, action: GameAction, data: Dict[str, Any]):
        dx, dy = 0, 0
        if action == GameAction.ACTION1:    # UP
            dy = -1
        elif action == GameAction.ACTION2:  # DOWN
            dy = 1
        elif action == GameAction.ACTION3:  # LEFT
            dx = -1
        elif action == GameAction.ACTION4:  # RIGHT
            dx = 1

        new_x = self.player_pos[0] + dx
        new_y = self.player_pos[1] + dy

        # Check collision with walls
        if (new_x, new_y) not in self.walls:
            self.player_pos = [new_x, new_y]

        self._render()

        # Check win condition
        if self.player_pos == self.goal_pos:
            self.state = GameState.WIN
            self.levels_completed += 1


# =====================================================================
# World 2: push_block (Object Mechanics & Causal Induction)
# Core prior: Movable entities, contact physics, and state transformations.
# =====================================================================
@register_world("push_block")
class PushBlockWorld(ARCBaseWorld):
    def __init__(self, width=9, height=9):
        super().__init__(width=width, height=height, max_steps=50)
        self.player = [2, 4]
        self.box = [4, 4]
        self.target = [6, 4]

    def setup_world(self):
        self.clear_grid(0)
        self.player = [2, 4]
        self.box = [4, 4]
        self.target = [6, 4]
        self._render()

    def _render(self):
        self.clear_grid(0)
        # Target slot: Red (2)
        self.set_cell(self.target[0], self.target[1], 2)
        # Movable Box: Yellow (4)
        self.set_cell(self.box[0], self.box[1], 4)
        # Agent: Blue (1)
        self.set_cell(self.player[0], self.player[1], 1)

    def update_world(self, action: GameAction, data: Dict[str, Any]):
        dx, dy = 0, 0
        if action == GameAction.ACTION1: dy = -1
        elif action == GameAction.ACTION2: dy = 1
        elif action == GameAction.ACTION3: dx = -1
        elif action == GameAction.ACTION4: dx = 1
        else: return

        nx, ny = self.player[0] + dx, self.player[1] + dy

        # Boundary check
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return

        # Pushing mechanic
        if [nx, ny] == self.box:
            bx, by = self.box[0] + dx, self.box[1] + dy
            if 0 <= bx < self.width and 0 <= by < self.height:
                self.box = [bx, by]
                self.player = [nx, ny]
        else:
            self.player = [nx, ny]

        self._render()

        # Win condition: box placed on target
        if self.box == self.target:
            self.state = GameState.WIN
            self.levels_completed += 1


# =====================================================================
# World 3: click_repair (Pointer / Symmetry Repair via ACTION6)
# Core prior: Abstract geometric symmetry & coordinate interaction.
# =====================================================================
@register_world("click_repair")
class ClickRepairWorld(ARCBaseWorld):
    def __init__(self, size=7):
        super().__init__(width=size, height=size, max_steps=20)
        self.broken_coord = (5, 1)  # The corrupted pixel breaking vertical symmetry

    def setup_world(self):
        self.clear_grid(0)
        # Draw symmetric Cyan (8) frame
        for i in range(1, 6):
            self.set_cell(1, i, 8)
            self.set_cell(5, i, 8)
            self.set_cell(i, 1, 8)
            self.set_cell(i, 5, 8)

        # Corrupt one point (break symmetry with 0)
        self.set_cell(self.broken_coord[0], self.broken_coord[1], 0)

    def update_world(self, action: GameAction, data: Dict[str, Any]):
        # Requires ACTION6 with {"x": int, "y": int}
        if action == GameAction.ACTION6 and data:
            cx, cy = data.get("x", -1), data.get("y", -1)
            # If clicked on the broken coordinate, fix it to Cyan (8)
            if (cx, cy) == self.broken_coord:
                self.set_cell(cx, cy, 8)
                self.state = GameState.WIN
                self.levels_completed += 1
            else:
                # Misclick penalty: draw Red (2) error spot
                self.set_cell(cx, cy, 2)


# =====================================================================
# Self-Test Runner
# =====================================================================
if __name__ == "__main__":
    print(f"[*] Available ARC-AGI-3 Worlds: {list_worlds()}\n")

    # Demo 1: Run World 1 ('nav_simple') with manual scripted actions
    world = make_world("nav_simple")
    frame = world.reset()
    frame.render_terminal("nav_simple - Initial State")

    print("\nExecuting test actions towards goal...")
    actions = [GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION2, GameAction.ACTION4]
    for act in actions:
        frame = world.step(act)
    frame.render_terminal("nav_simple - After 4 moves")

    # Demo 2: Test World 3 ('click_repair') using ACTION6 coordinate click
    print("\nTesting World 3 (click_repair) via complex coordinate action ACTION6...")
    repair_world = make_world("click_repair")
    frame = repair_world.reset()
    frame.render_terminal("click_repair - Broken Symmetry")

    # Send repair action: ACTION6 at coordinate (x=5, y=1)
    frame = repair_world.step(GameAction.ACTION6, data={"x": 5, "y": 1})
    frame.render_terminal("click_repair - Repaired Result")
    assert frame.state == GameState.WIN
    print("[+] Test completed: ACTION6 successfully triggered WIN state on symmetry repair!")
