"""
hippocampus.py
Episodic Spatial Memory based on Percepta's Spotlight Architecture.

Core Tenets of Spotlight:
1. Memory is structured as a 2D lattice of addressable recurrent cells.
2. Keys write to a local neighborhood around their 2D address.
3. Queries read from a local neighborhood around their 2D address.
4. Intelligence is separated from memory: memory capacity grows dynamically
   without altering any column or model weights.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from arc3_engine import GameAction


class MemoryCell:
    """A single recurrent cell on the 2D Spotlight lattice."""
    def __init__(self, u: int, v: int):
        self.coords = (u, v)
        self.visit_count = 0.0
        # Action-transition history: action -> {dest_color: count}
        self.action_outcomes: Dict[GameAction, Dict[int, float]] = {}
        # Trace vector for custom embeddings/affordances
        self.state_trace: List[float] = [0.0] * 8

    def update(self, action: GameAction, dest_color: int, weight: float = 1.0):
        self.visit_count += weight
        if action not in self.action_outcomes:
            self.action_outcomes[action] = {}
        self.action_outcomes[action][dest_color] = (
            self.action_outcomes[action].get(dest_color, 0.0) + weight
        )


class SpotlightHippocampus:
    """
    The Spotlight Episodic Memory Engine.
    Maintains a 2D lattice of dynamically allocated memory cells.
    """
    def __init__(self, write_radius: float = 1.5, decay: float = 0.99):
        self.write_radius = write_radius
        self.decay = decay
        # Sparse 2D lattice: (u, v) -> MemoryCell
        self.lattice: Dict[Tuple[int, int], MemoryCell] = {}
        self.trajectory_history: List[Tuple[int, int, GameAction]] = []

    def _get_or_create_cell(self, u: int, v: int) -> MemoryCell:
        coord = (int(round(u)), int(round(v)))
        if coord not in self.lattice:
            self.lattice[coord] = MemoryCell(*coord)
        return self.lattice[coord]

    def _get_neighborhood(self, u: float, v: float) -> List[Tuple[MemoryCell, float]]:
        """
        Spotlight Neighborhood: finds cells within write_radius and computes 
        Gaussian radial weights.
        """
        center_u, center_v = int(round(u)), int(round(v))
        radius_int = int(math.ceil(self.write_radius))
        neighbors = []

        for du in range(-radius_int, radius_int + 1):
            for dv in range(-radius_int, radius_int + 1):
                dist = math.sqrt(du**2 + dv**2)
                if dist <= self.write_radius:
                    # Gaussian falloff kernel
                    weight = math.exp(- (dist**2) / (2.0 * (0.8**2)))
                    cell = self._get_or_create_cell(center_u + du, center_v + dv)
                    neighbors.append((cell, weight))

        return neighbors

    def write_event(self, x: float, y: float, action: GameAction, dest_color: int):
        """
        Spotlight Neighborhood Write:
        Writes the episodic experience to a spatial patch on the 2D lattice.
        """
        self.trajectory_history.append((int(x), int(y), action))
        neighborhood = self._get_neighborhood(x, y)

        for cell, weight in neighborhood:
            cell.update(action, dest_color, weight=weight)

    def read_familiarity(self, x: float, y: float) -> float:
        """
        Spotlight Neighborhood Read (Curiosity / Novelty Meter):
        Returns the aggregate familiarity of the (x, y) region.
        Low values indicate high novelty (unexplored space).
        """
        neighborhood = self._get_neighborhood(x, y)
        if not neighborhood:
            return 0.0

        total_familiarity = sum(cell.visit_count * weight for cell, weight in neighborhood)
        return total_familiarity

    def get_loop_penalty(self, x: float, y: float) -> float:
        """
        Ant-colony / Anti-oscillation metric:
        Heavily penalizes coordinates that were visited in the recent trajectory.
        """
        penalty = 0.0
        recent = self.trajectory_history[-6:]  # Last 6 steps
        for past_x, past_y, _ in recent:
            if past_x == int(x) and past_y == int(y):
                penalty += 5.0
        return penalty

    def reset_episode(self):
        """Clears transient short-term trajectory while preserving long-term spatial cells."""
        self.trajectory_history.clear()

    def print_lattice_stats(self):
        print(f"[*] Spotlight Lattice: {len(self.lattice)} active cells allocated.")


# =====================================================================
# Self-Test Runner
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SPOTLIGHT HIPPOCAMPUS LATTICE")
    print("=" * 60)

    hippo = SpotlightHippocampus(write_radius=1.5)

    # 1. Initial State: Coordinate (2, 2) is completely novel
    fam_before = hippo.read_familiarity(2, 2)
    print(f"Familiarity at (2, 2) BEFORE visit: {fam_before:.4f} (High Novelty)")
    assert fam_before == 0.0

    # 2. Write an episodic event at (2, 2)
    print("\nWriting event: Agent visited (2, 2) and pressed ACTION4 (Right)...")
    hippo.write_event(x=2, y=2, action=GameAction.ACTION4, dest_color=0)

    # 3. Read familiarity after visit
    fam_center = hippo.read_familiarity(2, 2)
    fam_adjacent = hippo.read_familiarity(2, 3)  # Nearby neighbor in the Spotlight beam
    fam_far = hippo.read_familiarity(8, 8)       # Far away

    print(f"Familiarity at center (2, 2)    : {fam_center:.4f}")
    print(f"Familiarity at adjacent (2, 3)  : {fam_adjacent:.4f} (Neighbor diffused activation)")
    print(f"Familiarity at distant (8, 8)   : {fam_far:.4f}")

    assert fam_center > 0.0
    assert fam_adjacent > 0.0
    assert fam_far == 0.0

    # 4. Loop Detection Check
    loop_pen = hippo.get_loop_penalty(2, 2)
    print(f"\nLoop Penalty for stepping back onto (2, 2): {loop_pen:.2f}")
    assert loop_pen > 0.0

    hippo.print_lattice_stats()
    print("\n[+] Spotlight Hippocampus unit test passed successfully!")
