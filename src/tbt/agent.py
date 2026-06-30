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
                 epistemic: str = "progress", rmax: bool = False, effort: float = 0.0):
        self.actions = list(range(n_actions))
        self.rng = random.Random(seed)
        self.col = CorticalColumn(n_entities=n_entities, seed=seed)           # the world model (graph + SR), learned online
        # `effort` (the per-action efficiency cost) is plumbed but DEFAULTS OFF: at gamma=0.9 a flat per-step cost
        # fragily fights FAR goals (effort*Sum(gamma^t) vs gamma^D), so it is DEFERRED to Stage 2 -- once the SR-read
        # gives robust shortest-path value it becomes a tie-breaker among goal-reaching paths, not a force that abandons them.
        self.reward = RewardModel(16, gamma=gamma, beta=beta, optimistic=True, epistemic=epistemic, rmax=rmax,
                                  effort=(0.0 if rmax else effort))           # EFE value (R-MAX = ablation)
        self.tried: set = set()                                              # (state, action) ATTEMPTED -- persists across episodes
        self.plan_depth = 1                                                  # forward-model epistemic rollout depth (FM3; 1 = sound default)
        self.new_episode()

    def new_episode(self):
        self._prev = None                                                    # (state, action) of the previous turn
        self._pred = None                                                    # the predictive state (predicted current state)
        self.surprised = False
        self._prev_field = None                                              # L4 feature-field at the previous turn (FM2)
        self._pred_field = None                                              # the predicted next field (the efference at field grain)
        self.field_error = 0.0                                               # dense forward-model prediction error last turn (on CHANGED cells)

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

    def step(self, state, score_delta: float = 0.0, blocked=(), frame=None):
        """One turn: COMPARE last turn's prediction to the actual `state`, LEARN the transition (column + reward), PLAN
        value over the column's learned transitions, CHOOSE the value-maximising action, then PREDICT the next state.
        `blocked` = actions the WORLD MODEL predicts lead into a recognised barrier (from the object-behaviour faculty);
        they lose R-MAX optimism, so the agent routes around a known barrier WITHOUT bumping it -- and because the
        prediction is keyed on the recognised object, it generalises to a NEVER-bumped instance.

        With `frame` (FM2): the column ALSO does dense predict-then-compare in L4's feature-FIELD -- L5's per-location
        forward model predicts the next field, compared per location to the actual one. The fraction of CHANGED cells
        it mispredicts is a DENSE, CONTINUOUS learning-progress signal (replacing the opaque binary state-surprise),
        so the agent is drawn to where the DYNAMICS are still learnable (the structured-dynamics games H1 found). The
        field rule is learned online; it persists across episodes (the same mechanic everywhere)."""
        self.surprised = self._pred is not None and state != self._pred       # predict-then-compare = the learning signal
        field = self.col.feature_field(frame) if frame is not None else None
        if field is not None and self._prev_field is not None and self._prev is not None:
            self.field_error = self._field_err(self._pred_field, field, self._prev_field)   # dense error of last turn's field prediction
            self.col.observe_field(self._prev_field, self._prev[1], field)   # learn the per-location rule online (L5)
        if self._prev is not None:
            ps, pa = self._prev
            self.col.observe(ps, pa, state)                                  # learn the transition online (graph + SR)
            self.tried.add((ps, pa))                                         # attempted (even if blocked -> no edge)
            err = self.field_error if field is not None else (1.0 if self.surprised else 0.0)
            self.reward.observe_error(ps, err)                              # LEARNING-PROGRESS: dense field error (or binary fallback)
        self.reward.observe(state, score_delta)                             # value: reward where the score rose + novelty
        a = self._choose(state, blocked, field=field)
        self._pred = self.col.predict(state, a)                             # enter the predictive state
        if field is not None:
            self._pred_field = self.col.predict_field(field, a)             # the efference copy at field grain (the next predicted field)
        self._prev = (state, a)
        self._prev_field = field
        return self.col.motor(a)                                            # the action enacted via L5's motor output

    @staticmethod
    def _field_err(pred_field, actual_field, prev_field) -> float:
        """The forward model's error = fraction of the cells that ACTUALLY CHANGED (actual != prev) that the prediction
        got wrong. Scoring on changed cells (not all cells, which the static background trivially inflates) makes it
        the dynamics-learning signal -- 0 when the change is nailed, 1 when missed; -> 0 as the rule is mastered."""
        if pred_field is None:
            return 0.0
        chg = wrong = 0
        for row_p, row_a, row_v in zip(pred_field, actual_field, prev_field):
            for p, a, v in zip(row_p, row_a, row_v):
                if a != v:
                    chg += 1
                    wrong += (p != a)
        return wrong / chg if chg else 0.0

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

    def _choose(self, state, blocked=(), field=None):
        """Plan the EFE value over the learned transitions, then let the COLUMN's inverse-model motor (`col.act`)
        select the action that best achieves the highest-value next-state -- selection is seated in the column (the
        motor), not here. Untried actions take the bounded frontier optimism (epistemic frontier); a `blocked`
        action takes its discounted stay value (a recognised barrier, avoided -- value-driven, generalising).

        With `field` (FM3): the FORWARD MODEL contributes a per-action EPISTEMIC bonus -- the learning potential of
        each action -- so in a structured-dynamics game (where the tabular value is flat: states never recur) the
        agent is DRIVEN to the action whose effect it understands least, and the drive winds down as each action's
        rule is pinned (handing off to the pragmatic value)."""
        T, preds = self._transitions()
        if state in T:                                                       # plan only from a state with observed edges
            self.reward.plan(T, preds, state)                               # (an unvisited state has only frontier values)
        bonus = None
        if field is not None:
            bonus = {a: self.reward.beta * v for a, v in self._field_plan(field, self.plan_depth).items()}
        return self.col.act(state, self.actions, value=lambda s: self.reward.V[s], explore=self.reward.explore,
                            gamma=self.reward.gamma, tried=self.tried, blocked=blocked, rng=self.rng, bonus=bonus)

    def _field_plan(self, field, depth=1):
        """Per-action EPISTEMIC value via the forward model: each action's LEARNING POTENTIAL = how unsure the model
        is about its effect (`1 - field_confidence`). `depth > 1` adds the discounted best potential reachable after
        it (a shallow rollout via `predict_field`). The forward model DRIVING exploration -- try the action whose
        dynamics you understand least; it winds to 0 as each rule is pinned. depth-1 is the sound default (a deep
        epistemic rollout leans on unseen->identity predictions); deeper rollout is reserved for the PRAGMATIC goal
        (FM4), where the model is confident along the planned path."""
        out = {}
        for a in self.actions:
            epi = 1.0 - self.col.L5.field_confidence(field, a)
            if depth > 1:
                nxt = self.col.predict_field(field, a)
                out_next = self._field_plan(nxt, depth - 1)
                epi += self.reward.gamma * (max(out_next.values()) if out_next else 0.0)
            out[a] = epi
        return out
