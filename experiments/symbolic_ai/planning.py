"""planning.py — Domain-agnostic goal-directed planning.

Architecture
============
  Goal            A (priority, condition, achieve) triple.  Static priority.
  Drive           Homeostatic drive: measures current level vs. desired set-point.
                  deficit(state) = max(0, setpoint - measure(state)) * urgency
  FEPGoal         Drive-based variant of Goal.  Priority computed each step from
                  the sum of drive deficits, grounding goal urgency in state.
  GoalStack       Hierarchical sub-goal stack.  DecisionEngine pursues the stack
                  top before falling through to the flat goal list.
  EpisodicBuffer  Rolling episodic memory (capacity-limited; evicts by surprise).
  BeliefState     Bayesian P(item | room) with observation updates + temporal decay.
  AffordanceModel Tracks action outcomes; infers missing preconditions from
                  failure by comparing failed states to historical successes.
  DecisionEngine  Iterates active goals; records feedback; clears resolved blocks.

Design principles
=================
  0. **THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST
     BE GENERAL.**  This is the inviolable principle from which all others follow.
     Consequence: Drive.measure lambdas and Goal.condition callables must read
     from the dynamic ``state`` dict (``s.get(key, default)``), NEVER from static
     variables captured in closures at construction time.  Static closures encode
     assumptions about a specific environment and silently break when the
     environment changes.  The engine itself knows nothing about any domain.
  1. Goals are contextual: ``condition(state) -> bool`` determines applicability.
  2. Safety goals (``is_safety=True``) always participate and cannot be
     suppressed.  They simply sit at the top of the priority order.
  3. AffordanceModel uses feature-based comparison to infer WHAT was missing
     when an action fails, enabling the engine to route around the block and
     automatically unblock once the missing item/condition is satisfied.
  4. All domain knowledge lives in Goal.condition / Goal.achieve callables and
     in the ``state`` dict the caller builds.  The engine itself is pure logic.
  5. FEPGoal replaces static priority with drive-computed effective_priority(state).
     Goals become active in proportion to their drive deficit, not hardcoded floats.
     This grounds goal selection in the Free Energy Principle: acting to reduce
     prediction error between desired state (drive set-point) and current state.
  6. GoalStack allows hierarchical decomposition: a high-level goal can push
     sub-goals that are pursued before returning to the parent goal.  Achieve
     functions receive ``stack=<GoalStack>`` as a keyword argument and may push
     sub-goals before returning an action (e.g. cook pushes eat; unlock pushes
     enter).  This enables sequential commitment without task-specific hacks.
  7. EpisodicBuffer retains surprising experiences longer; neutral steps decay.
  8. BeliefState maintains P(item|room) with Bayesian updates and temporal decay,
     enabling the agent to act on uncertain knowledge about item locations.
  9. RANDOM (the fallback achieve function) must never issue destructive actions.
     Specifically: never ``drop`` an item.  A general agent does not undo its own
     inventory possession without explicit goal direction.  Drops belong to
     domain-specific goals (e.g. put_in_container), never to stochastic fallback.

Failure learning
================
  When a navigation action fails (location unchanged after 'go X'):
    1. Record (location, action) -> FAIL in AffordanceModel.
    2. Compare current state features with features of historical successes for
       the same (location, action) pair.
    3. Report the intersection-minus-current as inferred missing preconditions,
       e.g. {'inv:brass_key'} meaning "brass key was in inventory on success".
    4. When the agent later acquires 'brass_key', on_acquired('brass_key')
       automatically clears all blocks whose missing feature was 'inv:brass_key'.

FEP goal priority
=================
  Each Drive.deficit(state) measures urgency on a [0, 1] scale.
  FEPGoal.effective_priority(state) = min(1, sum(d.deficit(state) for d in drives)).
  When all drives are satisfied (deficit ≈ 0), the goal naturally becomes
  lowest-priority without any special-casing.  This is biologically grounded:
  hunger drive priority emerges from blood glucose deficit, not a hardcoded float.

  Example Drives for a TextWorld agent::

      hunger_drive = Drive('hunger',
                           measure  = lambda s: s['food_level'] / 20.0,
                           setpoint = 1.0,
                           urgency  = 0.8)
      eat_goal = FEPGoal('eat', drives=[hunger_drive], condition=has_food, achieve=eat_fn)
      # effective_priority = 0.8 when starving, ≈0 when sated

GoalStack usage
===============
  Sub-goals push themselves onto the stack via the engine's goal_stack attribute.
  DecisionEngine.decide() tries the stack top first; pops on failure/completion::

      # In an achieve function, to defer to a sub-goal:
      engine.goal_stack.push(NavigateGoal(target_room))
      return navigate_step(state, aff, rng)

  The engine's feedback() can also auto-push acquire-item sub-goals from
  AffordanceModel.infer_missing() (when a nav fails with 'inv:X' missing).

Safety goals with contextual exceptions
========================================
  Set ``is_safety=True`` and priority=1.0 for inviolable constraints.
  For contextual exceptions, the achieve fn itself can return None when an
  exception condition applies, allowing lower-priority goals to act.

  Example::

      def _safety_fn(state, aff, rng):
          if state.get('fire_emergency'):
              return None  # exception: ignore 'do not touch fire' in emergency
          harm = [c for c in state['admissible'] if _is_harmful(c)]
          if harm:
              safe = [c for c in state['admissible'] if c not in harm]
              return (rng.choice(safe), 'SAFETY') if safe else ('look', 'SAFETY')
          return None  # no harmful actions present; let other goals act

      Goal('safety', 1.0, cond=lambda s: True, achieve=_safety_fn, is_safety=True)

Usage
=====
    # 1. Define goals (domain-specific — lives in the caller's module)
    goals = [
        Goal('eat',     0.8, cond=has_edible, achieve=eat_action),
        Goal('explore', 0.3, cond=lambda s: True, achieve=explore_action),
    ]
    # OR use FEPGoal for drive-based dynamic priority:
    goals = [
        FEPGoal('eat', drives=[hunger_drive], condition=has_edible, achieve=eat_action),
    ]

    # 2. Build affordance model (optionally seeded with prior observations)
    aff = AffordanceModel(causal_stores)   # causal_stores from Phase Q1

    # 3. Create engine (with optional FEP memory components)
    episodic = EpisodicBuffer(capacity=30)
    belief   = BeliefState(rooms=known_rooms, items=known_items)
    engine   = DecisionEngine(goals, aff, episodic=episodic, belief=belief)

    # 4. Planning loop
    while not done:
        state  = build_state(obs, world)        # caller builds; domain-specific
        action, reason = engine.decide(state, rng)
        obs    = env.step(action)
        events = env.get_events()
        new_state = build_state(obs, world)
        engine.feedback(prev_state, action, new_state, events)
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple,
)


# ---------------------------------------------------------------------------
# Goal  (static priority)
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """A contextual, prioritized goal with a static priority float.

    Attributes
    ----------
    name      Unique identifier used in trace output and debugging.
    priority  Float in [0, 1].  Higher values are attempted first.
    condition Callable ``(state: dict) -> bool``.  Goal fires only when True.
    achieve   Callable ``(state: dict, aff: AffordanceModel, rng) ->
              (action_str, reason_str) | None``.
              Return None if the goal cannot act right now; the engine
              will try the next lower-priority goal.
    is_safety If True, the goal always participates regardless of context.
              Use for inviolable safety constraints.
    """
    name:      str
    priority:  float
    condition: Callable[[dict], bool]
    achieve:   Callable[..., Optional[Tuple[str, str]]]
    is_safety: bool = False


# ---------------------------------------------------------------------------
# Drive  (homeostatic urgency signal)
# ---------------------------------------------------------------------------

@dataclass
class Drive:
    """Homeostatic drive: a scalar urgency signal that modulates goal priority.

    The drive computes how far the agent is from its set-point, yielding a
    value in [0, urgency] that sums into the parent FEPGoal's effective_priority.

    Parameters
    ----------
    name      Human-readable label for debugging and tracing.
    measure   Callable ``(state: dict) -> float`` returning the current level
              in [0, 1] (0 = depleted/critical, 1 = full/satisfied).
    setpoint  Desired level (default 1.0 = fully satisfied).
    urgency   Multiplier on deficit.  Total contribution = deficit × urgency.
              Use larger values for drives that should dominate when active.

    Example
    -------
    >>> hunger = Drive('hunger', measure=lambda s: s['food']/20, urgency=0.8)
    >>> hunger.deficit({'food': 0})     # starving
    0.8
    >>> hunger.deficit({'food': 20})    # sated
    0.0
    """
    name:     str
    measure:  Callable[[dict], float]
    setpoint: float = 1.0
    urgency:  float = 1.0

    def deficit(self, state: dict) -> float:
        """Return urgency-scaled deficit in [0, urgency]: max(0, setpoint-measure)*urgency."""
        try:
            current = float(self.measure(state))
        except Exception:
            current = 0.0
        return max(0.0, self.setpoint - current) * self.urgency


# ---------------------------------------------------------------------------
# FEPGoal  (drive-based dynamic priority)
# ---------------------------------------------------------------------------

@dataclass
class FEPGoal:
    """A goal whose priority is computed each step from homeostatic drive deficits.

    Biologically grounded: goal urgency = sum of relevant drive deficits.
    Eating becomes urgent when food deficit is high; safety always overrides.
    When all drives are satisfied, effective_priority ≈ 0 and the goal yields
    to higher-priority goals — no special-casing required.

    This implements the Free Energy Principle intuition: the agent acts to
    minimise prediction error between desired state (drive set-point) and
    observed state (drive.measure(state)).

    Parameters
    ----------
    name          Unique identifier.
    drives        List of Drive objects whose deficits sum into effective priority.
    condition     Same semantics as Goal.condition.
    achieve       Same semantics as Goal.achieve.
    is_safety     If True, effective_priority always returns 1.0.
    base_priority Priority to use when drives list is empty (default 0.1).

    Notes
    -----
    FEPGoal exposes a ``priority`` property for compatibility with DecisionEngine's
    initial sort.  The true dynamic priority is ``effective_priority(state)``
    which is re-computed each step in ``decide()``.
    """
    name:          str
    drives:        List[Drive]
    condition:     Callable[[dict], bool]
    achieve:       Callable[..., Optional[Tuple[str, str]]]
    is_safety:     bool  = False
    base_priority: float = 0.1

    @property
    def priority(self) -> float:
        """Static upper bound on effective_priority (used for initial sort only)."""
        if self.is_safety:
            return 1.0
        if not self.drives:
            return self.base_priority
        # Upper bound: if all drives are at max deficit.
        return min(1.0, sum(d.urgency for d in self.drives))

    def effective_priority(self, state: dict) -> float:
        """Compute actual priority from drive deficits evaluated against state."""
        if self.is_safety:
            return 1.0
        if not self.drives:
            return self.base_priority
        return min(1.0, sum(d.deficit(state) for d in self.drives))


# ---------------------------------------------------------------------------
# GoalStack  (hierarchical sub-goal decomposition)
# ---------------------------------------------------------------------------

class GoalStack:
    """Hierarchical sub-goal stack with push/pop semantics.

    When a high-level goal (e.g. 'enter bedroom') cannot be achieved directly
    because a precondition is unmet, it can push a sub-goal ('acquire brass key')
    onto the stack.  DecisionEngine.decide() pursues the stack-top goal before
    falling through to the flat goal list, implementing goal-directed sub-task
    decomposition analogous to PFC-mediated goal hierarchies in the brain.

    The stack is automatically cleared between episodes via ``clear()``.
    Goals pop themselves when their condition becomes False or achieve returns None.
    """

    def __init__(self) -> None:
        self._stack: List[Any] = []  # stack of Goal | FEPGoal

    def push(self, goal: Any) -> None:
        """Push a sub-goal onto the top of the stack."""
        self._stack.append(goal)

    def pop(self) -> Optional[Any]:
        """Pop and return the top sub-goal, or None if stack is empty."""
        return self._stack.pop() if self._stack else None

    def current(self) -> Optional[Any]:
        """Return the top sub-goal without removing it."""
        return self._stack[-1] if self._stack else None

    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def depth(self) -> int:
        return len(self._stack)

    def clear(self) -> None:
        """Clear all sub-goals (call between episodes)."""
        self._stack.clear()

    def __repr__(self) -> str:
        names = [getattr(g, 'name', repr(g)) for g in reversed(self._stack)]
        return f'GoalStack([{", ".join(names)}])'


# ---------------------------------------------------------------------------
# EpisodicBuffer  (capacity-limited rolling episodic memory)
# ---------------------------------------------------------------------------

@dataclass
class EpisodicEntry:
    """One step of episodic memory.

    Parameters
    ----------
    step      Monotone step counter (within engine lifetime).
    location  Room name at the start of this step.
    action    Command issued.
    outcome   'success', 'failure', or 'neutral'.
    delta     Dict of what changed: acquired, lost, score_delta, moved_to.
    surprise  Salience weight in [0, 1].  High = memorable.
    """
    step:     int
    location: str
    action:   str
    outcome:  str
    delta:    Dict[str, Any]
    surprise: float = 0.0


class EpisodicBuffer:
    """Rolling episodic buffer weighted by recency × surprise.

    Biologically inspired: surprising or rewarding events are retained longer
    (analogous to hippocampal novelty gating and dopamine-modulated consolidation).
    Boring, neutral steps (no score change, no items acquired) decay fastest.

    Capacity is enforced by evicting the entry with the lowest recency-weighted
    surprise when the buffer is full::

        weight(entry) = entry.surprise × decay^(now - entry.step)

    Parameters
    ----------
    capacity  Maximum number of entries retained (default 30).
    decay     Per-step recency weight, in (0, 1).  0.92 ≈ 12-step half-life.
    """

    def __init__(self, capacity: int = 30, decay: float = 0.92) -> None:
        self._entries: List[EpisodicEntry] = []
        self.capacity = capacity
        self.decay    = decay
        self._now:    int = 0

    def add(self, entry: EpisodicEntry) -> None:
        """Add an entry; evict the least-salient entry if at capacity."""
        self._now = max(self._now, entry.step)
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            weights = [
                e.surprise * (self.decay ** max(0, self._now - e.step))
                for e in self._entries
            ]
            self._entries.pop(int(weights.index(min(weights))))

    def recent_failures(self, location: str, action: str) -> int:
        """Count how many times (location, action) ended in failure in buffer."""
        return sum(
            1 for e in self._entries
            if e.location == location
            and e.action  == action
            and e.outcome == 'failure'
        )

    def recent_outcomes(self, n: int = 5) -> List[EpisodicEntry]:
        """Return the N most recent entries in chronological order."""
        return sorted(self._entries, key=lambda e: e.step)[-n:]

    def summary(self, n: int = 5) -> str:
        """Human-readable summary of the n most surprising entries."""
        top = sorted(self._entries, key=lambda e: e.surprise, reverse=True)[:n]
        lines = []
        for e in top:
            lines.append(
                f'    step {e.step:>3}: {e.action!r:30s} '
                f'@ {e.location!r:18s} -> {e.outcome}  '
                f'(surprise={e.surprise:.2f})'
            )
        return '\n'.join(lines) if lines else '    (empty)'


# ---------------------------------------------------------------------------
# BeliefState  (Bayesian P(item | location) with temporal decay)
# ---------------------------------------------------------------------------

class BeliefState:
    """Bayesian belief distribution over item locations.

    Maintains P(item in location) for each (item, location) pair.  Beliefs
    decay toward the uniform prior with each time step — items may have moved
    since the last observation.  Direct observations (item seen / not seen)
    perform sharp Bayesian updates.

    Biologically: analogous to hippocampal-prefrontal spatial memory that fades
    without rehearsal and updates sharply on observation.

    Parameters
    ----------
    rooms   Iterable of known room names.
    items   Iterable of items to track.
    decay   Per-step decay toward uniform prior.
            0.95 = gentle (19-step half-life toward uniform).
            0.50 = fast (1-step half-life).

    Locations
    ---------
    In addition to named rooms, the belief tracks two virtual locations:
    ``'_inv'``     : item is currently in inventory.
    ``'_unknown'`` : item location not yet observed.
    """

    _VIRTUAL = ('_inv', '_unknown')

    def __init__(
        self,
        rooms: List[str],
        items: List[str],
        decay: float = 0.95,
    ) -> None:
        self._rooms  = list(rooms)
        self._decay  = decay
        all_locs     = self._rooms + list(self._VIRTUAL)
        n            = max(len(all_locs), 1)
        self._beliefs: Dict[str, Dict[str, float]] = {
            item: {loc: 1.0 / n for loc in all_locs}
            for item in items
        }

    def _ensure(self, item: str) -> None:
        """Initialise uniform prior for a previously unseen item."""
        if item not in self._beliefs:
            all_locs = self._rooms + list(self._VIRTUAL)
            n        = max(len(all_locs), 1)
            self._beliefs[item] = {loc: 1.0 / n for loc in all_locs}

    def _normalise(self, b: Dict[str, float]) -> None:
        total = sum(b.values()) or 1e-12
        for k in b:
            b[k] /= total

    def observe(self, item: str, location: str, present: bool) -> None:
        """Update P(item|location) on a direct observation.

        Parameters
        ----------
        item      Item name.
        location  Room name, '_inv' for inventory, etc.
        present   True if item observed to be at this location; False if absent.
        """
        self._ensure(item)
        b = self._beliefs[item]
        if location not in b:
            b[location] = 0.0

        if present:
            # Sharp update: concentrate mass on observed location.
            total_other = max(sum(v for k, v in b.items() if k != location), 1e-12)
            for k in b:
                b[k] = 0.97 if k == location else 0.03 * (b[k] / total_other)
        else:
            # Soft update: redistribute this location's mass to others.
            mass = b.get(location, 0.0)
            b[location] = 1e-4
            total_other = max(sum(v for k, v in b.items() if k != location), 1e-12)
            for k in b:
                if k != location:
                    b[k] += mass * (b[k] / total_other)

        self._normalise(b)

    def step(self) -> None:
        """Decay all beliefs toward the uniform prior (one time step passes)."""
        for b in self._beliefs.values():
            n       = max(len(b), 1)
            uniform = 1.0 / n
            for loc in b:
                b[loc] = self._decay * b[loc] + (1.0 - self._decay) * uniform
            self._normalise(b)

    def most_likely(self, item: str) -> Tuple[str, float]:
        """Return (most_likely_location, probability)."""
        self._ensure(item)
        b = self._beliefs[item]
        if not b:
            return '_unknown', 0.0
        loc = max(b, key=b.__getitem__)
        return loc, b[loc]

    def entropy(self, item: str) -> float:
        """Shannon entropy of P(location | item) in bits.  0 = certain; log2(n) = uniform."""
        self._ensure(item)
        return -sum(
            p * math.log2(p + 1e-12)
            for p in self._beliefs[item].values() if p > 0
        )

    def total_entropy(self) -> float:
        """Sum of per-item entropies: overall location uncertainty in bits."""
        return sum(self.entropy(item) for item in self._beliefs)

    def summary(self) -> str:
        """Human-readable table of item beliefs."""
        lines = []
        for item in sorted(self._beliefs):
            loc, prob = self.most_likely(item)
            ent = self.entropy(item)
            lines.append(
                f'    {item:<22} P({loc}) = {prob:.2f}  '
                f'entropy = {ent:.2f} bits'
            )
        return '\n'.join(lines) if lines else '    (empty)'


# ---------------------------------------------------------------------------
# AffordanceModel
# ---------------------------------------------------------------------------

def _featurise(location: str, inventory: FrozenSet[str]) -> FrozenSet[str]:
    """Convert (location, inventory) to a frozenset of boolean feature strings.

    Examples
    --------
    >>> _featurise('living room', frozenset({'brass key'}))
    frozenset({'loc:living_room', 'inv:brass_key'})
    """
    feats: Set[str] = {f'loc:{location.replace(" ", "_")}'}
    feats.update(f'inv:{item.replace(" ", "_")}' for item in inventory)
    if not inventory:
        feats.add('inv_empty')
    return frozenset(feats)


class AffordanceModel:
    """Tracks action preconditions learned from success/failure history.

    State representation
    --------------------
    A ``frozenset`` of boolean feature strings derived from (location,
    inventory) via ``_featurise()``.  Example::

        {'loc:living_room', 'inv:brass_key'}

    Precondition inference
    ----------------------
    On failure for ``(location, action)``:

      1. Collect feature sets from ALL historical successes of the same
         ``(location, action)`` pair (including those pre-loaded from Q1).
      2. Compute the intersection — features present in *all* successes.
      3. ``missing = intersection − current_features``.
      4. Store ``missing`` keyed by ``(location, action)`` for lookup.

    Automatic unblocking
    --------------------
    When the agent acquires item ``X``, ``on_acquired(X)`` clears any
    ``(location, action)`` block whose inferred missing set contained
    ``'inv:<X>'``.  This allows routing to resume through previously
    blocked exits without explicit replanning.

    Parameters
    ----------
    causal_stores
        Optional dict from Phase Q1 ``run_phase_q1()``.  If it contains
        key ``'navigable_ctx'``, each entry ``((room, cmd, inv_fs), (to,))``
        is used to pre-populate success features so the engine can infer
        preconditions from the very first episode.
    """

    def __init__(self, causal_stores: Optional[dict] = None) -> None:
        # (location, action) -> list of feature sets at success time
        self._success_feats: Dict[Tuple[str, str], List[FrozenSet[str]]] = (
            collections.defaultdict(list)
        )
        # (location, action) -> count of unresolved failures
        self._fail_count: collections.Counter = collections.Counter()
        # (location, action) -> frozenset of inferred missing features
        self._missing: Dict[Tuple[str, str], FrozenSet[str]] = {}

        # Pre-populate from Q1 causal observations so the engine can
        # immediately infer what was different between Q1 successes and
        # Q4 failures without needing to re-observe the success case.
        if causal_stores:
            for entry, _ in causal_stores.get('navigable_ctx', []):
                room, cmd, inv_fs = entry
                feats = _featurise(room, inv_fs)
                self._success_feats[(room, cmd)].append(feats)

    # ------------------------------------------------------------------

    def record(self, state: dict, action: str, success: bool) -> None:
        """Record one action attempt.

        Call this after every environment step via
        ``DecisionEngine.feedback()``.
        """
        loc   = state.get('location', '')
        inv   = frozenset(state.get('inventory', []))
        feats = _featurise(loc, inv)
        key   = (loc, action)

        if success:
            self._success_feats[key].append(feats)
            # Success resolves the failure for this (loc, action).
            self._fail_count.pop(key, None)
            self._missing.pop(key, None)
        else:
            self._fail_count[key] += 1
            # Infer missing: present in ALL successes, absent from current.
            suc_lists = self._success_feats.get(key, [])
            if suc_lists:
                common: Set[str] = set(suc_lists[0])
                for fs in suc_lists[1:]:
                    common &= set(fs)
                self._missing[key] = frozenset(common - feats)

    def is_blocked(self, location: str, action: str) -> bool:
        """True if ``(location, action)`` has at least one unresolved failure."""
        return self._fail_count[(location, action)] > 0

    def infer_missing(
        self, location: str, action: str,
    ) -> FrozenSet[str]:
        """Return inferred missing features for a failed ``(location, action)``."""
        return self._missing.get((location, action), frozenset())

    def on_acquired(self, item: str) -> None:
        """Called when the agent acquires ``item``.

        Clears any blocks whose inferred missing feature set contained
        ``'inv:<item>'``.  This automatically unblocks navigation through
        doors / exits that require the item as a precondition.
        """
        feat = f'inv:{item.replace(" ", "_")}'
        for key in list(self._missing):
            if feat in self._missing[key]:
                self._fail_count.pop(key, None)
                self._missing.pop(key, None)

    def missing_summary(self) -> str:
        """Human-readable summary of active failure inferences."""
        lines = []
        for (loc, action), feats in sorted(self._missing.items()):
            lines.append(f'    {loc!r} + {action!r}:  missing {sorted(feats)}')
        return '\n'.join(lines) if lines else '    (none)'


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Domain-agnostic goal-directed planning engine.

    ``decide(state, rng) -> (action, reason)``
        1. Checks GoalStack top first (sub-goals pushed by prior steps).
        2. Falls through to the flat goal list, dynamically sorted by
           effective_priority(state) for FEPGoal objects; static priority
           for plain Goal objects.
        Returns ``('look', 'FALLBACK')`` if all goals decline.

    ``feedback(prev_state, action, new_state, events=[]) -> None``
        Records action outcome.  Navigation failure is detected when location
        is unchanged after a 'go' command.  Updates EpisodicBuffer and
        BeliefState if provided.  Item acquisition events clear AffordanceModel
        blocks automatically.

    Parameters
    ----------
    goals
        List of ``Goal`` or ``FEPGoal`` objects.  Sorted by priority at
        construction; FEPGoal priorities are re-sorted dynamically each step.
    affordances
        ``AffordanceModel`` instance; shared across all episodes so knowledge
        accumulates over the agent's lifetime.
    goal_stack
        Optional ``GoalStack`` for hierarchical sub-goal decomposition.
        Created automatically if not provided (always non-None after __init__).
    episodic
        Optional ``EpisodicBuffer`` for rolling episodic memory.
    belief
        Optional ``BeliefState`` for probabilistic item location tracking.
    """

    def __init__(
        self,
        goals:        List[Any],
        affordances:  AffordanceModel,
        goal_stack:   Optional[GoalStack]      = None,
        episodic:     Optional[EpisodicBuffer]  = None,
        belief:       Optional[BeliefState]    = None,
    ) -> None:
        # Initial static sort (safety first, then by static priority upper bound).
        self.goals = sorted(
            goals,
            key=lambda g: (-g.priority - (0.001 if g.is_safety else 0.0)),
        )
        self.affordances = affordances
        self.goal_stack  = goal_stack if goal_stack is not None else GoalStack()
        self.episodic    = episodic
        self.belief      = belief
        self._step:      int = 0

    # ------------------------------------------------------------------

    def _effective_priority(self, goal: Any, state: dict) -> float:
        """Return goal priority: state-dependent for FEPGoal, static for Goal."""
        if isinstance(goal, FEPGoal):
            return goal.effective_priority(state)
        return goal.priority

    def decide(self, state: dict, rng: Any) -> Tuple[str, str]:
        """Return ``(action, reason)`` from the highest-priority applicable goal.

        Checks the GoalStack first, then falls through to the flat goal list
        sorted by effective_priority(state).  Pops stack entries that decline.

        Achieve functions receive ``stack=self.goal_stack`` as a keyword argument.
        They may push sub-goals onto the stack before returning an action, enabling
        sequential commitment (e.g. cook pushes eat, unlock pushes enter).
        Achieve functions that do not need the stack should accept ``**kwargs``.
        """
        # 1. Pursue active sub-goal from stack.
        while not self.goal_stack.is_empty():
            current = self.goal_stack.current()
            try:
                if current.condition(state):
                    result = current.achieve(
                        state, self.affordances, rng, stack=self.goal_stack)
                    if result is not None:
                        return result
            except Exception:
                pass
            # Stack top declined or errored — pop it and continue.
            self.goal_stack.pop()

        # 2. Flat goal list, dynamically sorted by effective_priority(state).
        sorted_goals = sorted(
            self.goals,
            key=lambda g: (
                -self._effective_priority(g, state)
                - (0.001 if g.is_safety else 0.0)
            ),
        )
        for goal in sorted_goals:
            try:
                if not goal.condition(state):
                    continue
                result = goal.achieve(
                    state, self.affordances, rng, stack=self.goal_stack)
                if result is not None:
                    return result
            except Exception:
                pass  # goal errored; skip to next

        return 'look', 'FALLBACK'

    def feedback(
        self,
        prev_state: dict,
        action:     str,
        new_state:  dict,
        events:     Optional[List[dict]] = None,
    ) -> None:
        """Record the outcome of the last action; update all memory systems.

        Navigation failure heuristic
        ----------------------------
        If the action started with 'go ' (or was a bare direction word) AND
        the location is unchanged between ``prev_state`` and ``new_state``,
        the action is considered to have failed.

        Episodic surprise
        -----------------
        surprise = |score_delta| × 0.5 + |acquired| × 0.3 + |lost| × 0.3
                   + 0.4 × (1 if nav_failure)
        High-surprise entries are retained longer in the EpisodicBuffer.

        Belief state
        ------------
        Decays item beliefs toward uniform each step.  Acquisition events
        update P(item | '_inv') ≈ 1.  Loss events redistribute back to
        the room the item was last seen in.

        Item acquisitions
        -----------------
        Any event ``{'type': 'acquired', 'item': X}`` triggers
        ``AffordanceModel.on_acquired(X)`` AND ``BeliefState.observe(X, '_inv', True)``,
        automatically clearing navigation blocks that required X.
        """
        if not action:
            return

        self._step += 1

        # Detect navigation failure.
        _NAV_WORDS = {'north', 'south', 'east', 'west', 'up', 'down'}
        is_nav  = action.startswith('go ') or action in _NAV_WORDS
        success = True
        if is_nav:
            success = prev_state.get('location') != new_state.get('location')

        self.affordances.record(prev_state, action, success)

        # --- EpisodicBuffer update ----------------------------------------
        if self.episodic is not None:
            prev_score  = float(prev_state.get('score', 0) or 0)
            new_score   = float(new_state.get('score', 0) or 0)
            score_delta = new_score - prev_score
            prev_inv    = set(prev_state.get('inventory', []))
            new_inv     = set(new_state.get('inventory', []))
            acquired    = list(new_inv - prev_inv)
            lost        = list(prev_inv - new_inv)
            delta: Dict[str, Any] = {
                'acquired':    acquired,
                'lost':        lost,
                'score_delta': score_delta,
                'moved_to':    new_state.get('location') if success and is_nav else None,
            }
            surprise = min(1.0,
                abs(score_delta) * 0.5
                + len(acquired)  * 0.3
                + len(lost)      * 0.3
                + (0.4 if is_nav and not success else 0.0)
            )
            if success and not acquired and not lost and score_delta == 0:
                outcome = 'neutral'
            elif is_nav and not success:
                outcome = 'failure'
            else:
                outcome = 'success'
            self.episodic.add(EpisodicEntry(
                step=self._step,
                location=prev_state.get('location', ''),
                action=action,
                outcome=outcome,
                delta=delta,
                surprise=surprise,
            ))

        # --- BeliefState: temporal decay -----------------------------------
        if self.belief is not None:
            self.belief.step()

        # --- Item acquisition / loss events --------------------------------
        for ev in (events or []):
            if ev.get('type') == 'acquired':
                item = ev['item']
                self.affordances.on_acquired(item)
                if self.belief is not None:
                    self.belief.observe(item, '_inv', True)
            elif ev.get('type') == 'lost':
                item = ev.get('item', '')
                if item and self.belief is not None:
                    prev_loc = prev_state.get('location', '_unknown')
                    self.belief.observe(item, '_inv', False)
                    self.belief.observe(item, prev_loc, True)
