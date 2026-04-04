"""Base class for CatPlan world simulators.

A world simulator:
1. Maintains ground truth state (hidden from the learner)
2. Executes actions and returns observations
3. Generates demonstration trajectories for domain learning

The learner sees observations (sets of predicate-value tuples).
It does NOT see the world's internal implementation — it must
discover the domain structure from observations alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    """What the learner sees: a set of (predicate, args, value) tuples.

    For boolean predicates, value is True/False.
    For numeric predicates, value is a float.
    """
    facts: frozenset[tuple[str, tuple[str, ...], Any]]

    def get(self, predicate: str, args: tuple[str, ...]) -> Any:
        """Get the value of a specific predicate, or None."""
        for p, a, v in self.facts:
            if p == predicate and a == args:
                return v
        return None

    def all_true(self) -> list[tuple[str, tuple[str, ...]]]:
        """Get all predicates that are true (boolean)."""
        return [(p, a) for p, a, v in self.facts if v is True]

    def predicates(self) -> set[str]:
        """Get all predicate names in this observation."""
        return {p for p, _, _ in self.facts}


@dataclass
class Transition:
    """One step: (state_before, action, state_after)."""
    before: Observation
    action: str
    action_args: tuple[str, ...]
    after: Observation


@dataclass
class Demonstration:
    """A full demonstration trajectory."""
    transitions: list[Transition]
    metadata: dict[str, Any] = field(default_factory=dict)


class World:
    """Base class for world simulators."""

    def __init__(self):
        self._history: list[Transition] = []

    def observe(self) -> Observation:
        """Return the current state as an observation."""
        raise NotImplementedError

    def available_actions(self) -> list[tuple[str, tuple[str, ...]]]:
        """Return all actions available in the current state.

        Each action is (action_name, (arg1, arg2, ...)).
        """
        raise NotImplementedError

    def execute(self, action: str, args: tuple[str, ...]) -> Observation:
        """Execute an action and return the new observation.

        Records the transition in history.
        """
        before = self.observe()
        self._execute_impl(action, args)
        after = self.observe()
        self._history.append(Transition(
            before=before, action=action, action_args=args, after=after,
        ))
        return after

    def _execute_impl(self, action: str, args: tuple[str, ...]):
        """Subclass implements: actually change the world state."""
        raise NotImplementedError

    def reset(self):
        """Reset to initial state."""
        raise NotImplementedError

    def get_history(self) -> list[Transition]:
        return list(self._history)

    def generate_demonstration(self, n_steps: int = 20,
                                strategy: str = "random") -> Demonstration:
        """Generate a demonstration trajectory.

        strategy: "random" = random valid actions.
        """
        import random
        self.reset()
        self._history.clear()

        for _ in range(n_steps):
            actions = self.available_actions()
            if not actions:
                break
            if strategy == "random":
                action_name, action_args = random.choice(actions)
            else:
                action_name, action_args = actions[0]
            self.execute(action_name, action_args)

        return Demonstration(
            transitions=list(self._history),
            metadata={"strategy": strategy, "n_steps": len(self._history)},
        )

    def generate_demonstrations(self, n: int = 50,
                                 n_steps: int = 20,
                                 seed: int = 42) -> list[Demonstration]:
        """Generate multiple demonstrations."""
        import random
        rng = random.Random(seed)
        old_state = random.getstate()

        demos = []
        for i in range(n):
            random.seed(rng.randint(0, 2**31))
            demos.append(self.generate_demonstration(n_steps=n_steps))

        random.setstate(old_state)
        return demos
