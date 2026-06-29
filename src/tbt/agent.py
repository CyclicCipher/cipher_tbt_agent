"""The agent — the active-inference loop over ONE column + reward (value). Thin by construction.

Predict-then-compare each turn (HTM / the reafference principle): the column enters a PREDICTIVE STATE (predicts the
next state given the chosen action), and next turn the actual state is compared to it -- the mismatch (`surprised`) is
the learning signal. The column learns its transition GRAPH online (the world model); `reward` learns the goal from the
sparse score, unified with novelty into ONE value (epistemic + pragmatic, R-MAX optimism); the value-maximising action
is taken. A GOAL is a prediction the loop acts to fulfil.

The column supplies the transition model (its learned graph) + path integration; `reward` is the non-column value
"exception". The basal-ganglia gate over GSG-proposed goal-states is step 3b. Pure stdlib + the column.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .column import CorticalColumn
from .reward import RewardModel

_GOAL = ("\x00GOAL",)                                                          # the single terminal a completing transition leads to


class Agent:
    """One column + one reward model, driven by the predict-then-compare loop. `step(state, score_delta)` consumes the
    state sensed this turn (and the score gained entering it) and returns the next action. The model persists across
    episodes; `new_episode()` only resets the per-episode prediction/linkage (do not path-integrate across a reset)."""

    def __init__(self, n_actions: int, n_entities: int = 256, gamma: float = 0.9, beta: float = 0.3, seed: int = 0,
                 epistemic: str = "progress"):
        self.actions = list(range(n_actions))
        self.rng = random.Random(seed)
        self.col = CorticalColumn(n_entities=n_entities, seed=seed)           # the world model (graph + SR), learned online
        self.reward = RewardModel(16, gamma=gamma, beta=beta, optimistic=True, epistemic=epistemic)  # value: score + LEARNING-PROGRESS
        self.tried: set = set()                                              # (state, action) ATTEMPTED -- persists across episodes
        self.new_episode()

    def new_episode(self):
        self._prev = None                                                    # (state, action) of the previous turn
        self._pred = None                                                    # the predictive state (predicted current state)
        self.surprised = False

    def complete(self, score_delta: float = 1.0):
        """A level completed: the PREVIOUS (state, action) was the completing transition. Record it as leading to a
        single terminal GOAL sentinel and reward THAT -- so the agent learns to TAKE the completing action (value
        flows back through that one edge), not to PARK on the state before the goal (which crediting the state would
        teach). One GOAL across all levels = ARC's 'completion is the goal', and it transfers. Then end the episode
        (the level boundary). For the live loop, where the next observed frame is already the next level."""
        if self._prev is not None:
            ps, pa = self._prev
            self.col.observe(ps, pa, _GOAL)                                  # the completing transition -> the goal
            self.tried.add((ps, pa))
        self.reward.observe(_GOAL, max(score_delta, 1.0))                    # the goal is the rewarding terminal
        self.new_episode()

    def step(self, state, score_delta: float = 0.0, blocked=()):
        """One turn: COMPARE last turn's prediction to the actual `state`, LEARN the transition (column + reward), PLAN
        value over the column's learned transitions, CHOOSE the value-maximising action, then PREDICT the next state.
        `blocked` = actions the WORLD MODEL predicts lead into a recognised barrier (from the object-behaviour faculty);
        they lose R-MAX optimism, so the agent routes around a known barrier WITHOUT bumping it -- and because the
        prediction is keyed on the recognised object, it generalises to a NEVER-bumped instance."""
        self.surprised = self._pred is not None and state != self._pred       # predict-then-compare = the learning signal
        if self._prev is not None:
            ps, pa = self._prev
            self.col.observe(ps, pa, state)                                  # learn the transition online (graph + SR)
            self.tried.add((ps, pa))                                         # attempted (even if blocked -> no edge)
            self.reward.observe_error(ps, 1.0 if self.surprised else 0.0)    # LEARNING-PROGRESS signal: error at the state left
        self.reward.observe(state, score_delta)                             # value: reward where the score rose + novelty
        a = self._choose(state, blocked)
        self._pred = self.col.predict(state, a)                             # enter the predictive state
        self._prev = (state, a)
        return self.col.motor(a)                                            # the action enacted via L5's motor output

    def _transitions(self):
        """The transition model the value planner reads -- the column's FULL predictive model: from each visited state,
        `col.predict` returns the observed edge where there is one and the position-invariant DISPLACEMENT elsewhere, so
        value propagates through predicted-but-UNVISITED states (the generalization the bare graph cannot do). This is
        the column supplying T; `reward` never sees the world directly."""
        T, preds = {}, defaultdict(list)
        for s in list(self.col.graph):
            row = [self.col.predict(s, a) for a in self.actions]
            T[s] = row
            for nxt in row:
                preds[nxt].append(s)
        return T, preds

    def _choose(self, state, blocked=()):
        """The value-maximising action: plan value over the learned transitions, then pick the action whose predicted
        next state has the highest value -- an UNTRIED action is valued optimistically (R-MAX), so the frontier
        attracts (epistemic) until the score teaches a goal (pragmatic). A `blocked` action (a recognised barrier
        ahead) instead takes its predicted-STAY value -- NOT the optimistic Vmax -- so the agent does not waste a bump
        confirming a barrier it already recognises (the avoidance is value-driven, not hardcoded, and generalises)."""
        T, preds = self._transitions()
        if state in T:                                                       # plan only from a state with observed edges
            self.reward.plan(T, preds, state)                               # (an unvisited state has only R-MAX values)
        vals = []
        for a in self.actions:
            if a in blocked:                                                # predicted barrier: a DISCOUNTED stay -- strictly
                nxt = self.col.graph.get(state, {}).get(a, state)           # below the optimistic frontier, so a KNOWN barrier
                vals.append(self.reward.gamma * self.reward.V[nxt])         # is avoided even at a fresh state (no false optimism)
            elif (state, a) not in self.tried:                              # never ATTEMPTED -> optimistic (R-MAX frontier)
                vals.append(self.reward.Vmax)
            else:                                                           # attempted: value its outcome (self-loop if blocked)
                vals.append(self.reward.V[self.col.graph.get(state, {}).get(a, state)])
        best = max(vals)
        return self.rng.choice([a for a in self.actions if vals[a] == best])
