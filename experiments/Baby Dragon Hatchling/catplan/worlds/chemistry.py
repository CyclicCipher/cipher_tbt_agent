"""Chemistry world simulator.

Hidden rules (the learner must discover these):
- Atoms have valences (H=1, O=2, C=4, N=3)
- Bonds consume valence: bonding A-B reduces both A and B's free valence by 1
- An atom can't bond if its free valence is 0
- Atoms are conserved: bond and break don't create or destroy atoms
- Charge is conserved

The learner sees: which atoms exist, which bonds exist, free valences.
It must discover: valence constraints, conservation laws, valid reactions.
"""
from __future__ import annotations

import random

from .base import World, Observation


# Valence rules (hidden from learner).
VALENCES = {"H": 1, "O": 2, "C": 4, "N": 3, "Cl": 1, "Na": 1}


class ChemistryWorld(World):
    """A world of atoms and bonds."""

    def __init__(self, atoms: list[tuple[str, str]] | None = None):
        """atoms: list of (name, element) pairs. e.g., [("h1", "H"), ("o1", "O")]"""
        super().__init__()
        if atoms is None:
            atoms = [
                ("h1", "H"), ("h2", "H"), ("h3", "H"), ("h4", "H"),
                ("o1", "O"), ("o2", "O"),
                ("c1", "C"),
            ]
        self._atoms: dict[str, str] = {name: elem for name, elem in atoms}
        self._bonds: set[frozenset[str]] = set()  # each bond is frozenset({a, b})
        self._initial_atoms = list(atoms)

    def reset(self):
        self._atoms = {name: elem for name, elem in self._initial_atoms}
        self._bonds.clear()
        self._history.clear()

    def _free_valence(self, atom_name: str) -> int:
        """How many more bonds this atom can form."""
        elem = self._atoms[atom_name]
        max_val = VALENCES.get(elem, 0)
        current_bonds = sum(1 for b in self._bonds if atom_name in b)
        return max_val - current_bonds

    def observe(self) -> Observation:
        facts = set()
        # Atom existence and element type.
        for name, elem in self._atoms.items():
            facts.add(("atom", (name,), True))
            facts.add(("element", (name, elem), True))
            facts.add(("free_valence", (name,), self._free_valence(name)))

        # Bonds.
        for bond in self._bonds:
            a, b = sorted(bond)
            facts.add(("bonded", (a, b), True))

        # Total atom counts per element (conservation observable).
        from collections import Counter
        counts = Counter(self._atoms.values())
        for elem, count in counts.items():
            facts.add(("atom_count", (elem,), count))

        return Observation(facts=frozenset(facts))

    def available_actions(self) -> list[tuple[str, tuple[str, ...]]]:
        actions = []
        atom_names = sorted(self._atoms.keys())

        # Bond: any two unbonded atoms with free valence > 0.
        for i, a in enumerate(atom_names):
            for b in atom_names[i+1:]:
                if frozenset({a, b}) not in self._bonds:
                    if self._free_valence(a) > 0 and self._free_valence(b) > 0:
                        actions.append(("bond", (a, b)))

        # Break: any existing bond.
        for bond in self._bonds:
            a, b = sorted(bond)
            actions.append(("break_bond", (a, b)))

        return actions

    def _execute_impl(self, action: str, args: tuple[str, ...]):
        if action == "bond":
            a, b = args
            if self._free_valence(a) > 0 and self._free_valence(b) > 0:
                self._bonds.add(frozenset({a, b}))
        elif action == "break_bond":
            a, b = args
            self._bonds.discard(frozenset({a, b}))


class WaterFormation(ChemistryWorld):
    """Tier 1: form water (H2O) from free atoms."""

    def __init__(self):
        super().__init__([("h1", "H"), ("h2", "H"), ("o1", "O")])

    def is_goal(self) -> bool:
        """Goal: h1-o1 and h2-o1 bonded."""
        return (frozenset({"h1", "o1"}) in self._bonds and
                frozenset({"h2", "o1"}) in self._bonds)


class ReactionBalancing(ChemistryWorld):
    """Tier 2: balance CH4 + 2O2 -> CO2 + 2H2O.

    Start with C, 4H, 4O. Goal: C bonded to 2 O (double bonds),
    each H bonded to an O.
    """

    def __init__(self):
        super().__init__([
            ("c1", "C"),
            ("h1", "H"), ("h2", "H"), ("h3", "H"), ("h4", "H"),
            ("o1", "O"), ("o2", "O"), ("o3", "O"), ("o4", "O"),
        ])


if __name__ == "__main__":
    print("=== Chemistry World: Water Formation ===")
    w = WaterFormation()
    obs = w.observe()
    print(f"Initial: {len(obs.facts)} facts")
    for p, a, v in sorted(obs.facts):
        if v is True or isinstance(v, int):
            print(f"  {p}({', '.join(a)}) = {v}")

    print(f"\nAvailable actions: {w.available_actions()}")

    # Form water.
    w.execute("bond", ("h1", "o1"))
    w.execute("bond", ("h2", "o1"))
    print(f"\nAfter bonding H2O: goal={w.is_goal()}")
    obs = w.observe()
    for p, a, v in sorted(obs.facts):
        if p == "bonded":
            print(f"  {p}({', '.join(a)})")

    # Generate demonstrations.
    print(f"\n=== Demonstrations ===")
    demos = w.generate_demonstrations(n=5, n_steps=5)
    for i, demo in enumerate(demos):
        print(f"Demo {i}: {len(demo.transitions)} transitions")
        for t in demo.transitions:
            print(f"  {t.action}({', '.join(t.action_args)})")
