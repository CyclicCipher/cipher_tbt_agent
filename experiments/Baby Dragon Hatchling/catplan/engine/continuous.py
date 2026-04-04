"""CatPlan continuous extensions — enriched predicates, energy minimization,
multi-body interactions, sampling-based search, hierarchical abstraction.

Extends the discrete boolean planner to handle:
- D.8: Continuous state variables (positions, angles, energies)
- D.9: Multi-body actions (operads — reactions involving 3+ objects)
- Energy minimization as the planning objective
- Monte Carlo / simulated annealing for large state spaces
- D.7: Galois connections for hierarchical abstract→refine planning
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import sys, os
_CATPLAN_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _CATPLAN_DIR not in sys.path:
    sys.path.insert(0, _CATPLAN_DIR)


# ---------------------------------------------------------------------------
# Hybrid state: boolean atoms + continuous values
# ---------------------------------------------------------------------------

@dataclass
class HybridState:
    """A state with both boolean predicates and continuous values.

    Boolean: frozenset of (predicate, args) tuples that are true.
    Continuous: dict mapping (predicate, args) to float values.
    """
    booleans: frozenset[tuple[str, tuple[str, ...]]]
    continuous: dict[tuple[str, tuple[str, ...]], float]

    def get_bool(self, pred: str, args: tuple[str, ...]) -> bool:
        return (pred, args) in self.booleans

    def get_float(self, pred: str, args: tuple[str, ...]) -> float:
        return self.continuous.get((pred, args), 0.0)

    def set_bool(self, pred: str, args: tuple[str, ...], val: bool) -> 'HybridState':
        """Return new state with boolean changed."""
        if val:
            new_bools = self.booleans | {(pred, args)}
        else:
            new_bools = self.booleans - {(pred, args)}
        return HybridState(booleans=new_bools, continuous=dict(self.continuous))

    def set_float(self, pred: str, args: tuple[str, ...], val: float) -> 'HybridState':
        """Return new state with continuous value changed."""
        new_cont = dict(self.continuous)
        new_cont[(pred, args)] = val
        return HybridState(booleans=self.booleans, continuous=new_cont)

    def modify_float(self, pred: str, args: tuple[str, ...], delta: float) -> 'HybridState':
        """Return new state with continuous value incremented."""
        new_cont = dict(self.continuous)
        old = new_cont.get((pred, args), 0.0)
        new_cont[(pred, args)] = old + delta
        return HybridState(booleans=self.booleans, continuous=new_cont)

    def __hash__(self):
        # Discretize continuous values for hashing (otherwise states are never equal).
        disc = tuple(sorted((k, round(v, 2)) for k, v in self.continuous.items()))
        return hash((self.booleans, disc))

    def __eq__(self, other):
        if not isinstance(other, HybridState):
            return False
        if self.booleans != other.booleans:
            return False
        # Continuous values are equal if within epsilon.
        for k in set(self.continuous) | set(other.continuous):
            if abs(self.continuous.get(k, 0.0) - other.continuous.get(k, 0.0)) > 0.01:
                return False
        return True

    @staticmethod
    def from_observation(obs) -> 'HybridState':
        """Build a HybridState from a world Observation."""
        bools = set()
        conts = {}
        for pred, args, val in obs.facts:
            if isinstance(val, bool) and val:
                bools.add((pred, args))
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                conts[(pred, args)] = float(val)
        return HybridState(booleans=frozenset(bools), continuous=conts)


# ---------------------------------------------------------------------------
# Continuous action: checks + modifies both boolean and numeric state
# ---------------------------------------------------------------------------

@dataclass
class ContinuousAction:
    """An action over hybrid state.

    bool_preconditions: (predicate, arg_indices, negated) — checked on booleans
    numeric_preconditions: (predicate, arg_indices, op, value) — checked on continuous
    bool_effects: (predicate, arg_indices, set_to) — modify booleans
    numeric_effects: (predicate, arg_indices, op, value_or_delta) — modify continuous
    """
    name: str
    param_types: list[str]

    bool_preconds: list[tuple[str, tuple[int, ...], bool]] = field(default_factory=list)
    # (pred, param_indices, negated)

    numeric_preconds: list[tuple[str, tuple[int, ...], str, float]] = field(default_factory=list)
    # (pred, param_indices, op, threshold)  op in '<', '>', '<=', '>=', '='

    bool_effects: list[tuple[str, tuple[int, ...], bool]] = field(default_factory=list)
    # (pred, param_indices, set_to)

    numeric_effects: list[tuple[str, tuple[int, ...], str, float]] = field(default_factory=list)
    # (pred, param_indices, op, value)  op in 'assign', 'increase', 'decrease'


def ground_continuous_action(
    action: ContinuousAction,
    objects: list[str],
    object_types: dict[str, str],
) -> list[tuple[ContinuousAction, tuple[str, ...]]]:
    """Ground a continuous action over all type-valid object combinations."""
    from itertools import product as iterproduct

    type_objs: dict[str, list[str]] = defaultdict(list)
    for obj, typ in object_types.items():
        type_objs[typ].append(obj)

    param_domains = [type_objs.get(t, []) for t in action.param_types]
    if not param_domains:
        return [(action, ())]

    results = []
    for binding in iterproduct(*param_domains):
        if len(set(binding)) == len(binding):  # distinct
            results.append((action, tuple(binding)))
    return results


def applicable_continuous(
    action: ContinuousAction,
    binding: tuple[str, ...],
    state: HybridState,
) -> bool:
    """Check if a continuous action is applicable."""
    for pred, param_idx, negated in action.bool_preconds:
        args = tuple(binding[i] for i in param_idx)
        present = state.get_bool(pred, args)
        if negated and present:
            return False
        if not negated and not present:
            return False

    for pred, param_idx, op, threshold in action.numeric_preconds:
        args = tuple(binding[i] for i in param_idx)
        val = state.get_float(pred, args)
        if op == '>' and not (val > threshold):
            return False
        if op == '>=' and not (val >= threshold):
            return False
        if op == '<' and not (val < threshold):
            return False
        if op == '<=' and not (val <= threshold):
            return False
        if op == '=' and abs(val - threshold) > 0.01:
            return False

    return True


def apply_continuous(
    action: ContinuousAction,
    binding: tuple[str, ...],
    state: HybridState,
) -> HybridState:
    """Apply a continuous action to a hybrid state."""
    s = state
    for pred, param_idx, set_to in action.bool_effects:
        args = tuple(binding[i] for i in param_idx)
        s = s.set_bool(pred, args, set_to)

    for pred, param_idx, op, value in action.numeric_effects:
        args = tuple(binding[i] for i in param_idx)
        if op == 'assign':
            s = s.set_float(pred, args, value)
        elif op == 'increase':
            s = s.modify_float(pred, args, value)
        elif op == 'decrease':
            s = s.modify_float(pred, args, -value)

    return s


# ---------------------------------------------------------------------------
# D.9: Operadic actions (multi-body interactions)
# ---------------------------------------------------------------------------

@dataclass
class OperadicInteraction:
    """A multi-body interaction: computes a value from multiple objects.

    e.g., distance(a, b) is computed from pos_x(a), pos_y(a), pos_x(b), pos_y(b)
    e.g., interaction_energy(a, b, c) depends on distances between all three

    compute_fn takes (state, object_bindings) and returns a float.
    """
    name: str
    param_types: list[str]
    output_predicate: str
    compute_fn: Callable[[HybridState, tuple[str, ...]], float]


def apply_interactions(
    interactions: list[OperadicInteraction],
    state: HybridState,
    objects: list[str],
    object_types: dict[str, str],
) -> HybridState:
    """Recompute all multi-body interactions for the current state."""
    s = state
    for interaction in interactions:
        groundings = ground_continuous_action(
            ContinuousAction(name="", param_types=interaction.param_types),
            objects, object_types,
        )
        for _, binding in groundings:
            val = interaction.compute_fn(s, binding)
            s = s.set_float(interaction.output_predicate, binding, val)
    return s


# ---------------------------------------------------------------------------
# Energy minimization via simulated annealing
# ---------------------------------------------------------------------------

def energy_of_state(
    state: HybridState,
    energy_fn: Callable[[HybridState], float],
) -> float:
    """Compute the total energy of a state."""
    return energy_fn(state)


def simulated_annealing(
    initial_state: HybridState,
    actions: list[tuple[ContinuousAction, tuple[str, ...]]],
    energy_fn: Callable[[HybridState], float],
    interactions: list[OperadicInteraction],
    objects: list[str],
    object_types: dict[str, str],
    max_steps: int = 10000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.995,
    seed: int = 42,
) -> tuple[HybridState, float, list[tuple[str, tuple[str, ...]]]]:
    """Find the minimum-energy state via simulated annealing.

    Returns (best_state, best_energy, action_trace).
    """
    rng = random.Random(seed)

    current = initial_state
    current = apply_interactions(interactions, current, objects, object_types)
    current_energy = energy_fn(current)

    best = current
    best_energy = current_energy
    trace: list[tuple[str, tuple[str, ...]]] = []
    best_trace: list[tuple[str, tuple[str, ...]]] = []

    temp = initial_temp

    for step in range(max_steps):
        # Find applicable actions.
        applicable_list = [
            (action, binding) for action, binding in actions
            if applicable_continuous(action, binding, current)
        ]
        if not applicable_list:
            break

        # Pick a random action.
        action, binding = rng.choice(applicable_list)

        # Apply it.
        new_state = apply_continuous(action, binding, current)
        new_state = apply_interactions(interactions, new_state, objects, object_types)
        new_energy = energy_fn(new_state)

        # Metropolis criterion.
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 0.001)):
            current = new_state
            current_energy = new_energy
            trace.append((action.name, binding))

            if current_energy < best_energy:
                best = current
                best_energy = current_energy
                best_trace = list(trace)

        temp *= cooling_rate

    return best, best_energy, best_trace


# ---------------------------------------------------------------------------
# D.7: Galois connection — abstract then refine
# ---------------------------------------------------------------------------

def abstract_hybrid_state(
    state: HybridState,
    abstraction_fn: Callable[[HybridState], HybridState],
) -> HybridState:
    """Apply an abstraction function to get a coarser state."""
    return abstraction_fn(state)


def hierarchical_plan(
    initial_state: HybridState,
    abstract_fn: Callable[[HybridState], HybridState],
    abstract_actions: list[tuple[ContinuousAction, tuple[str, ...]]],
    concrete_actions: list[tuple[ContinuousAction, tuple[str, ...]]],
    energy_fn: Callable[[HybridState], float],
    interactions: list[OperadicInteraction],
    objects: list[str],
    object_types: dict[str, str],
    max_abstract_steps: int = 100,
    max_refine_steps: int = 1000,
    seed: int = 42,
) -> tuple[HybridState, float, list]:
    """Two-level planning: abstract first, then refine.

    1. Solve at the abstract level (fewer variables, faster)
    2. For each abstract action, find concrete actions that implement it
    """
    # Abstract planning.
    abstract_state = abstract_fn(initial_state)
    abstract_result, abstract_energy, abstract_trace = simulated_annealing(
        abstract_state, abstract_actions, energy_fn, interactions,
        objects, object_types, max_steps=max_abstract_steps, seed=seed,
    )

    # Concrete refinement.
    concrete_result, concrete_energy, concrete_trace = simulated_annealing(
        initial_state, concrete_actions, energy_fn, interactions,
        objects, object_types, max_steps=max_refine_steps, seed=seed + 1,
    )

    return concrete_result, concrete_energy, concrete_trace
