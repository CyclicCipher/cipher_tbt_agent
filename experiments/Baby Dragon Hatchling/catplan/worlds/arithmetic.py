"""Arithmetic world simulator.

Hidden rules (the learner must discover these):
- add(a, b) produces a result equal to a + b
- sub(a, b) produces a result equal to a - b (when a >= b)
- mul(a, b) produces a result equal to a * b

The learner sees: numbers, operation names, results.
It must discover: what each operation does, and solve novel problems.
"""
from __future__ import annotations

import random as stdlib_random

import sys, os
_CATPLAN_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _CATPLAN_DIR not in sys.path:
    sys.path.insert(0, _CATPLAN_DIR)

from worlds.base import World, Observation, Transition, Demonstration


class ArithmeticWorld(World):
    """A world where you can perform arithmetic operations on numbers."""

    def __init__(self, max_num: int = 20):
        super().__init__()
        self.max_num = max_num
        self._numbers: set[str] = {str(i) for i in range(max_num + 1)}
        self._results: list[tuple[str, str, str, str]] = []  # (op, a, b, result)

    def reset(self):
        self._results.clear()
        self._history.clear()

    def observe(self) -> Observation:
        facts = set()
        # All numbers exist.
        for n in self._numbers:
            facts.add(("number", (n,), True))
            facts.add(("value", (n,), int(n)))

        # Successor relationships (the learner can discover ordering).
        for i in range(self.max_num):
            facts.add(("succ", (str(i), str(i + 1)), True))

        # Results of operations performed so far.
        for op, a, b, result in self._results:
            facts.add((f"result_{op}", (a, b, result), True))

        return Observation(facts=frozenset(facts))

    def available_actions(self) -> list[tuple[str, tuple[str, ...]]]:
        actions = []
        nums = sorted(self._numbers, key=int)
        for a in nums:
            for b in nums:
                ai, bi = int(a), int(b)
                # add
                if ai + bi <= self.max_num:
                    actions.append(("add", (a, b)))
                # sub (only when a >= b)
                if ai >= bi:
                    actions.append(("sub", (a, b)))
                # mul
                if ai * bi <= self.max_num:
                    actions.append(("mul", (a, b)))
        return actions

    def _execute_impl(self, action: str, args: tuple[str, ...]):
        a, b = args
        ai, bi = int(a), int(b)
        if action == "add":
            result = str(ai + bi)
        elif action == "sub":
            result = str(ai - bi)
        elif action == "mul":
            result = str(ai * bi)
        else:
            return
        self._results.append((action, a, b, result))
        self._numbers.add(result)

    def generate_demonstrations(self, n: int = 50, n_steps: int = 10,
                                 seed: int = 42) -> list[Demonstration]:
        """Generate demonstrations of random arithmetic operations."""
        rng = stdlib_random.Random(seed)
        demos = []

        for d in range(n):
            self.reset()
            transitions = []

            for _ in range(n_steps):
                # Pick a random operation on small numbers (to keep it tractable).
                a = rng.randint(0, min(9, self.max_num))
                b = rng.randint(0, min(9, self.max_num))
                op = rng.choice(["add", "sub", "mul"])

                # Ensure validity.
                if op == "add" and a + b > self.max_num:
                    op = "sub"
                if op == "sub" and a < b:
                    a, b = b, a
                if op == "mul" and a * b > self.max_num:
                    op = "add"
                    if a + b > self.max_num:
                        continue

                before = self.observe()
                self.execute(op, (str(a), str(b)))
                after = self.observe()

                transitions.append(Transition(
                    before=before,
                    action=op,
                    action_args=(str(a), str(b)),
                    after=after,
                ))

            demos.append(Demonstration(transitions=transitions))

        return demos


if __name__ == "__main__":
    from engine.domain_learner import learn_domain, domain_to_catplan

    print("=" * 60)
    print("  ARITHMETIC DOMAIN LEARNING")
    print("=" * 60)

    # Phase 1: Generate training demonstrations.
    print("\n--- Phase 1: Training demos (numbers 0-9) ---")
    world = ArithmeticWorld(max_num=20)
    demos = world.generate_demonstrations(n=100, n_steps=15, seed=42)
    n_transitions = sum(len(d.transitions) for d in demos)
    print(f"Generated {len(demos)} demos, {n_transitions} transitions")

    # Show a few.
    for t in demos[0].transitions[:5]:
        print(f"  {t.action}({', '.join(t.action_args)})")

    # Phase 2: Learn domain.
    print("\n--- Phase 2: Domain learning ---")
    domain, types_map = learn_domain(demos, "Arithmetic")
    print(f"Types: {list(domain.types.keys())}")
    print(f"Predicates: {list(domain.predicates.keys())}")
    print(f"Actions: {list(domain.actions.keys())}")
    print(f"Invariants: {[inv.description for inv in domain.invariants]}")

    for a in domain.actions.values():
        print(f"\n  {a.name}({', '.join(p.name + ':' + p.type_name for p in a.params)}):")
        print(f"    preconditions: {[str(c) for c in a.preconditions]}")
        print(f"    effects: {[str(e) for e in a.effects]}")

    # Phase 3: Solve NOVEL problems.
    print("\n--- Phase 3: Novel problems ---")
    from engine.types import Problem, ObjectDecl, GroundAtom
    from engine.planner import plan_advanced, build_objects_by_type, recompute_derived
    from engine.heuristic_selector import select_heuristic

    # Novel problem 1: What is 7 + 8?
    # The system must find that add(7, 8) produces result_add(7, 8, 15).
    print("\nProblem 1: What is 7 + 8?")
    obj_to_type = {}
    for tn, objs in types_map.items():
        for o in objs:
            obj_to_type[o] = tn

    objects = {}
    for n in range(21):
        s = str(n)
        t = obj_to_type.get(s, list(domain.types.keys())[0])
        objects[s] = ObjectDecl(name=s, type_name=t)

    init_atoms = set()
    for n in range(21):
        init_atoms.add(GroundAtom(predicate="number", args=(str(n),)))
    for n in range(20):
        init_atoms.add(GroundAtom(predicate="succ", args=(str(n), str(n+1))))

    problem = Problem(
        name="add_7_8",
        domain_name="Arithmetic",
        objects=objects,
        init=init_atoms,
        goal={GroundAtom(predicate="result_add", args=("7", "8", "15"))},
    )

    h = select_heuristic(domain, problem)
    result, stats = plan_advanced(domain, problem, heuristic_mode=h, max_expansions=5000)
    if result:
        print(f"  SOLVED in {len(result)} steps ({stats['expansions']} expansions): {[str(a) for a in result]}")
    else:
        print(f"  FAILED ({stats['expansions']} expansions)")

    # Novel problem 2: What is 6 * 3?
    print("\nProblem 2: What is 6 * 3?")
    problem2 = Problem(
        name="mul_6_3",
        domain_name="Arithmetic",
        objects=objects,
        init=init_atoms,
        goal={GroundAtom(predicate="result_mul", args=("6", "3", "18"))},
    )
    result2, stats2 = plan_advanced(domain, problem2, heuristic_mode=h, max_expansions=5000)
    if result2:
        print(f"  SOLVED in {len(result2)} steps ({stats2['expansions']} expansions): {[str(a) for a in result2]}")
    else:
        print(f"  FAILED ({stats2['expansions']} expansions)")

    # Novel problem 3: What is 15 - 7?
    print("\nProblem 3: What is 15 - 7?")
    problem3 = Problem(
        name="sub_15_7",
        domain_name="Arithmetic",
        objects=objects,
        init=init_atoms,
        goal={GroundAtom(predicate="result_sub", args=("15", "7", "8"))},
    )
    result3, stats3 = plan_advanced(domain, problem3, heuristic_mode=h, max_expansions=5000)
    if result3:
        print(f"  SOLVED in {len(result3)} steps ({stats3['expansions']} expansions): {[str(a) for a in result3]}")
    else:
        print(f"  FAILED ({stats3['expansions']} expansions)")

    # Novel problem 4: Multi-step: compute (3 + 4) * 2
    # This requires two steps: first add(3, 4) -> 7, then mul(7, 2) -> 14.
    print("\nProblem 4: Compute (3 + 4) * 2 = 14 (multi-step)")
    problem4 = Problem(
        name="compound",
        domain_name="Arithmetic",
        objects=objects,
        init=init_atoms,
        goal={
            GroundAtom(predicate="result_add", args=("3", "4", "7")),
            GroundAtom(predicate="result_mul", args=("7", "2", "14")),
        },
    )
    result4, stats4 = plan_advanced(domain, problem4, heuristic_mode=h, max_expansions=5000)
    if result4:
        print(f"  SOLVED in {len(result4)} steps ({stats4['expansions']} expansions): {[str(a) for a in result4]}")
    else:
        print(f"  FAILED ({stats4['expansions']} expansions)")
