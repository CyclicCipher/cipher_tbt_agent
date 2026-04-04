"""Toy protein folding world simulator.

A simplified 2D lattice model of protein folding:
- A chain of residues on a 2D grid
- Two residue types: H (hydrophobic) and P (polar/hydrophilic)
- The chain is connected: each residue is adjacent to the next
- Energy: each H-H contact (non-bonded adjacent) reduces energy by -1
- Goal: find the conformation with minimum energy
- The protein starts as a straight chain and must fold

This is the HP lattice model (Dill, 1985) — the simplest model that
captures the essential physics of protein folding: hydrophobic collapse.

Hidden rules the learner must discover:
- H-H contacts lower energy (hydrophobic effect)
- The chain can't cross itself (excluded volume)
- Energy is minimized at the folded state
- Hydrophobic residues cluster in the interior
"""
from __future__ import annotations

import math
import random as stdlib_random

import sys, os
_CATPLAN_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _CATPLAN_DIR not in sys.path:
    sys.path.insert(0, _CATPLAN_DIR)

from engine.continuous import (
    HybridState, ContinuousAction, OperadicInteraction,
    ground_continuous_action, applicable_continuous,
    apply_continuous, apply_interactions, simulated_annealing,
)
from worlds.base import World, Observation


# Directions on 2D lattice.
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIR_NAMES = ["right", "left", "up", "down"]


class ProteinWorld:
    """HP lattice model of protein folding on a 2D grid.

    The protein is a chain of residues, each placed on a grid cell.
    Consecutive residues must be adjacent. No two residues can
    occupy the same cell.
    """

    def __init__(self, sequence: str):
        """sequence: string of 'H' and 'P' characters."""
        self.sequence = sequence
        self.n = len(sequence)
        # Place as a straight horizontal chain.
        self.positions: list[tuple[int, int]] = [(i, 0) for i in range(self.n)]

    def reset(self):
        self.positions = [(i, 0) for i in range(self.n)]

    def energy(self) -> float:
        """Compute the energy: -1 per H-H non-bonded adjacent contact."""
        occupied = {pos: i for i, pos in enumerate(self.positions)}
        e = 0.0
        for i in range(self.n):
            if self.sequence[i] != 'H':
                continue
            x, y = self.positions[i]
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                j = occupied.get((nx, ny))
                if j is not None and j != i and abs(j - i) > 1:
                    if self.sequence[j] == 'H':
                        e -= 0.5  # each contact counted twice (i→j and j→i), so -0.5 each
        return e

    def is_valid(self) -> bool:
        """Check that chain is connected and no overlaps."""
        # No overlaps.
        if len(set(self.positions)) != self.n:
            return False
        # Connected: consecutive residues are adjacent.
        for i in range(self.n - 1):
            x1, y1 = self.positions[i]
            x2, y2 = self.positions[i + 1]
            if abs(x1 - x2) + abs(y1 - y2) != 1:
                return False
        return True

    def observe(self) -> Observation:
        """Return observation of current conformation."""
        facts = set()
        for i in range(self.n):
            name = f"r{i}"
            x, y = self.positions[i]
            facts.add(("residue", (name,), True))
            facts.add(("res_type", (name,), self.sequence[i]))
            facts.add(("pos_x", (name,), float(x)))
            facts.add(("pos_y", (name,), float(y)))
            facts.add(("index", (name,), i))

            # Chain connectivity.
            if i < self.n - 1:
                facts.add(("bonded", (name, f"r{i+1}"), True))

        # Non-bonded H-H contacts.
        occupied = {pos: i for i, pos in enumerate(self.positions)}
        n_contacts = 0
        for i in range(self.n):
            if self.sequence[i] != 'H':
                continue
            x, y = self.positions[i]
            for dx, dy in DIRS:
                j = occupied.get((x + dx, y + dy))
                if j is not None and abs(j - i) > 1 and self.sequence[j] == 'H':
                    n_contacts += 1
        n_contacts //= 2  # each counted twice

        facts.add(("energy", (), self.energy()))
        facts.add(("hh_contacts", (), n_contacts))
        facts.add(("is_valid", (), self.is_valid()))

        return Observation(facts=frozenset(facts))

    def available_moves(self) -> list[tuple[int, tuple[int, int]]]:
        """Return all valid moves.

        Uses a simple but ergodic move set: for each residue, try
        placing it at each empty position adjacent to its chain neighbor(s).
        The whole tail (or head) of the chain then "pulls" along to
        maintain connectivity.

        Returns list of (residue_index, new_position).
        """
        moves = []
        occupied = set(self.positions)

        for i in range(self.n):
            x, y = self.positions[i]
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if (nx, ny) in occupied:
                    continue

                # Try placing residue i at (nx, ny) and pulling the chain.
                new_positions = self._try_pull(i, (nx, ny))
                if new_positions is not None:
                    moves.append((i, (nx, ny)))

        return moves

    def _try_pull(self, idx: int, new_pos: tuple[int, int]) -> list[tuple[int, int]] | None:
        """Try a pull move: place residue idx at new_pos, pull the chain.

        Returns new positions list if valid, None otherwise.
        """
        new_positions = list(self.positions)
        new_positions[idx] = new_pos

        # Pull the chain from idx toward both ends.
        # Forward: for j = idx+1, idx+2, ..., check if j is still adjacent
        # to j-1. If not, move j to a position adjacent to j-1.
        for j in range(idx + 1, self.n):
            prev = new_positions[j - 1]
            curr = new_positions[j]
            if abs(prev[0] - curr[0]) + abs(prev[1] - curr[1]) == 1:
                break  # still connected, chain is fine from here
            # Need to move j to be adjacent to j-1.
            # Try the old position of j-1 (before it moved).
            old_prev = self.positions[j - 1]
            if old_prev not in set(new_positions[:j] + new_positions[j+1:]):
                new_positions[j] = old_prev
            else:
                return None  # can't find a valid pull

        # Backward: for j = idx-1, idx-2, ..., same logic.
        for j in range(idx - 1, -1, -1):
            nxt = new_positions[j + 1]
            curr = new_positions[j]
            if abs(nxt[0] - curr[0]) + abs(nxt[1] - curr[1]) == 1:
                break
            old_nxt = self.positions[j + 1]
            if old_nxt not in set(new_positions[:j] + new_positions[j+1:]):
                new_positions[j] = old_nxt
            else:
                return None

        # Check validity.
        if len(set(new_positions)) != self.n:
            return None  # overlap
        for j in range(self.n - 1):
            p1, p2 = new_positions[j], new_positions[j + 1]
            if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
                return None  # disconnected
        return new_positions

    def execute_move(self, residue_idx: int, new_pos: tuple[int, int]) -> bool:
        """Execute a pull move. Returns True if successful."""
        new_positions = self._try_pull(residue_idx, new_pos)
        if new_positions is None:
            return False
        self.positions = new_positions
        return True


# ---------------------------------------------------------------------------
# Build CatPlan continuous actions for the protein world
# ---------------------------------------------------------------------------

def build_protein_actions(protein: ProteinWorld) -> list[ContinuousAction]:
    """Build ContinuousAction objects for the protein's move set."""
    actions = []
    # One action per (residue, direction) — parameterized.
    # But for simulated annealing, we pre-ground them.
    for i in range(protein.n):
        for d_name in DIR_NAMES:
            action = ContinuousAction(
                name=f"move_r{i}_{d_name}",
                param_types=[],
                # No formal preconditions — validity checked by the world.
            )
            actions.append(action)
    return actions


def build_protein_state(protein: ProteinWorld) -> HybridState:
    """Convert protein conformation to HybridState."""
    obs = protein.observe()
    return HybridState.from_observation(obs)


def protein_energy_fn(state: HybridState) -> float:
    """Extract energy from a hybrid state."""
    return state.get_float("energy", ())


# ---------------------------------------------------------------------------
# Simulated annealing for protein folding
# ---------------------------------------------------------------------------

def fold_protein(
    protein: ProteinWorld,
    max_steps: int = 50000,
    initial_temp: float = 5.0,
    cooling_rate: float = 0.9995,
    seed: int = 42,
) -> tuple[float, int]:
    """Fold a protein using simulated annealing on the lattice model.

    Returns (final_energy, n_accepted_moves).
    """
    rng = stdlib_random.Random(seed)

    current_energy = protein.energy()
    best_energy = current_energy
    best_positions = list(protein.positions)
    n_accepted = 0

    temp = initial_temp

    for step in range(max_steps):
        moves = protein.available_moves()
        if not moves:
            break

        residue_idx, new_pos = rng.choice(moves)
        old_positions = list(protein.positions)

        success = protein.execute_move(residue_idx, new_pos)
        if not success:
            continue

        new_energy = protein.energy()
        delta_e = new_energy - current_energy

        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 0.001)):
            current_energy = new_energy
            n_accepted += 1
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = list(protein.positions)
        else:
            protein.positions = old_positions

        temp *= cooling_rate

    protein.positions = best_positions
    return best_energy, n_accepted


def visualize_protein(protein: ProteinWorld) -> str:
    """ASCII visualization of the protein on the lattice."""
    if not protein.positions:
        return "(empty)"

    xs = [p[0] for p in protein.positions]
    ys = [p[1] for p in protein.positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Build grid.
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    grid = [['.' for _ in range(width)] for _ in range(height)]

    pos_to_idx = {pos: i for i, pos in enumerate(protein.positions)}

    for i, (x, y) in enumerate(protein.positions):
        gx = x - min_x
        gy = max_y - y  # flip y for display
        char = protein.sequence[i]
        grid[gy][gx] = char

    lines = ['  ' + ''.join(row) for row in grid]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Demo generation for domain learning
# ---------------------------------------------------------------------------

def generate_folding_demonstrations(
    sequence: str,
    n_demos: int = 50,
    n_steps_per_demo: int = 200,
    seed: int = 42,
) -> list:
    """Generate demonstrations of random folding attempts."""
    from worlds.base import Transition, Demonstration

    rng = stdlib_random.Random(seed)
    demos = []

    for d in range(n_demos):
        protein = ProteinWorld(sequence)
        transitions = []

        for _ in range(n_steps_per_demo):
            moves = protein.available_moves()
            if not moves:
                break
            before = protein.observe()
            idx, new_pos = rng.choice(moves)
            protein.execute_move(idx, new_pos)
            after = protein.observe()
            transitions.append(Transition(
                before=before,
                action="move",
                action_args=(f"r{idx}", str(new_pos[0]), str(new_pos[1])),
                after=after,
            ))

        demos.append(Demonstration(transitions=transitions))

    return demos


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A classic HP model test sequence.
    # HHPHPPHPPHH — known to have optimal energy of -4 on a 2D lattice.
    sequence = "HPHPPHHPHPH"

    print(f"Sequence: {sequence} (length {len(sequence)})")
    print(f"H count: {sequence.count('H')}, P count: {sequence.count('P')}")

    protein = ProteinWorld(sequence)
    print(f"\nInitial (straight chain):")
    print(visualize_protein(protein))
    print(f"Energy: {protein.energy()}")

    print(f"\nFolding via simulated annealing...")
    energy, n_moves = fold_protein(protein, max_steps=50000, seed=42)

    print(f"\nFolded:")
    print(visualize_protein(protein))
    print(f"Energy: {energy}")
    print(f"Accepted moves: {n_moves}")
    print(f"Valid: {protein.is_valid()}")

    # Generate demonstrations for domain learning.
    print(f"\n--- Domain Learning Test ---")
    print("Generating demonstrations from random folding...")
    demos = generate_folding_demonstrations(sequence, n_demos=50, n_steps_per_demo=100, seed=42)
    print(f"Generated {len(demos)} demos, {sum(len(d.transitions) for d in demos)} transitions")

    # Learn domain.
    from engine.domain_learner import learn_domain, domain_to_catplan
    domain, types_map = learn_domain(demos, "ProteinFolding")
    print(f"\nDiscovered domain:")
    print(f"  Types: {list(domain.types.keys())}")
    print(f"  Predicates: {list(domain.predicates.keys())}")
    print(f"  Actions: {list(domain.actions.keys())}")
    print(f"  Invariants: {[inv.description for inv in domain.invariants]}")

    # Now fold a NOVEL sequence the system has never seen.
    print(f"\n--- Novel Protein Folding ---")
    novel_sequence = "HHHPPHPHPHHH"
    print(f"Novel sequence: {novel_sequence} (NEVER seen in training)")
    novel = ProteinWorld(novel_sequence)
    print(f"Initial energy: {novel.energy()}")
    print(visualize_protein(novel))

    novel_energy, novel_n_moves = fold_protein(novel, max_steps=100000, seed=99)
    print(f"\nFolded energy: {novel_energy}")
    print(f"Accepted moves: {novel_n_moves}")
    print(visualize_protein(novel))
    print(f"Valid: {novel.is_valid()}")

    # Count H-H contacts.
    obs = novel.observe()
    contacts = obs.get("hh_contacts", ())
    print(f"H-H contacts: {contacts}")
