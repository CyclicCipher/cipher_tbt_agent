"""The neocortex — the multi-column control loop (task ⊕ space), columns coordinating by CMP messages.

This is the §5/§6 control loop made real: a TASK column sequences sub-goals and a SPATIAL column navigates to
them, joined ONLY by messages through the thalamus — never fused into one fat column (that is the 2^K
conjunctive explosion, architecture doc §2). Every inter-column message is in the **Cortical Messaging
Protocol** (RESEARCH.md R9 / Thousand Brains Project): a `(content, pose, confidence)` triple — *what*, *where
in the receiver's reference frame*, and *how sure*. Columns exchange BELIEFS, never raw input.

  top-down  (task → space):  the active sub-goal's content → the goal-state (a node) bound to it  [thalamus.read_location]
  bottom-up (space → task):  the reached node → which sub-goal (if any) is bound there  [thalamus.read]

The spatial column navigates by SR prioritized-replay (`reward.py`, Mattar–Daw gain×need) — value propagated
FROM the goal-state, the architecture's vector-nav — NOT an exhaustive search. K sub-goals sequenced additively.

Everything domain-specific is OPAQUE: `subgoals` are `(content, node)` pairs and `T`/`preds` a transition graph,
both handed in by perception. There is no grid / colour / door / push here. Lateral VOTING (the egocentric ⊗
absolute bind) + the BG gate as the learned sub-goal selector are the next stages; the spine is here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from tbt.reward import RewardModel
from tbt.thalamus import Thalamus


@dataclass
class CMP:
    """A Cortical Messaging Protocol message — the one currency between columns (R9)."""
    content: object              # WHAT — a sub-goal id / object label
    pose: object                 # WHERE — a node in the receiver column's reference frame
    confidence: float = 1.0


class Neocortex:
    """Task ⊕ space columns, joined by CMP messages through the thalamus; spatial nav by SR prioritized-replay."""

    def __init__(self, space_col, task_col, gamma: float = 0.9, seed: int = 0):
        self.space, self.task, self.thal = space_col, task_col, Thalamus()
        self.rm = RewardModel(1, gamma=gamma, beta=0.0, prioritized=True, optimistic=False)   # the SR value
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.R, self._sub, self._done, self._vgoal = None, [], set(), object()

    def bind(self, subgoals):
        """`subgoals = [(content, goal_node), …]`, ordered (any terminal goal last). Bind each content ⊗ its
        goal-state into the thalamus register — the shared blackboard the two columns read both ways."""
        self._sub, self._done, self._vgoal = list(subgoals), set(), object()
        self.R = self.thal.bind(self.task, self.space, subgoals) if subgoals else None

    # ---- the CMP channels (through the thalamus) ----------------------------------------------------------
    def _goal_node(self, content):
        """Top-down: the goal-state (a spatial node) the task content is bound to."""
        return self.thal.read_location(self.R, self.task, self.space, content)

    def _achieved(self, node):
        """Bottom-up: which sub-goal content (if any) is bound at the reached node."""
        return None if self.R is None else self.thal.read(self.R, self.task, self.space, node)

    # ---- the spatial column's navigation: SR prioritized-replay (not BFS) ---------------------------------
    def _navigate(self, T, preds, start, goal):
        """Greedy on the SR value field, value propagated from `goal` by prioritized sweeping (re-swept only
        when the goal-state changes — a sub-goal advancing — so it is one value field per sub-goal, not a
        search per step)."""
        if goal != self._vgoal:
            self.rm.R_ext = {goal: 1.0}
            self.rm.V.clear(); self.rm.queue.clear()
            self.rm._push(goal, 1.0)
            self.rm.budget = max(64, 3 * len(T))
            self.rm.plan(T, preds, start)
            self._vgoal = goal
        vals = [self.rm.V[nx] for nx in T[start]]
        m = max(vals)
        return self.rng.choice([a for a, v in enumerate(vals) if v == m])

    # ---- one step of the loop: acknowledge, sequence, route a goal-state, navigate ------------------------
    def act(self, current, T, preds):
        ack = self._achieved(current)                          # bottom-up CMP: did we reach a bound sub-goal?
        if ack is not None:
            self._done.add(ack)
        remaining = [(c, n) for c, n in self._sub if c not in self._done]
        if not remaining:
            return self.rng.randrange(len(T[current]))
        active, _ = remaining[0]                                # ordered reveal: the first unmet sub-goal (terminal last)
        return self._navigate(T, preds, current, self._goal_node(active))   # top-down goal-state → navigate
