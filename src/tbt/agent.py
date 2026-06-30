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
from collections import Counter, defaultdict

from .column import CorticalColumn
from .reward import RewardModel, ValueLearner

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
        self.plan_depth = 1                                                  # forward-model rollout depth (FM3/FM4; 1 = sound default)
        self.field_value = ValueLearner(alpha=0.25)                         # FM4: a GENERALISING value over field FEATURES (the goal in feature space)
        self.field_bin = 8                                                  # per-colour count binning for field features (coarse -> generalises)
        self.new_episode()

    def new_episode(self):
        self._prev = None                                                    # (state, action) of the previous turn
        self._pred = None                                                    # the predictive state (predicted current state)
        self.surprised = False
        self._prev_field = None                                              # L4 feature-field at the previous turn (FM2)
        self._pred_field = None                                              # the predicted next field (the efference at field grain)
        self.field_error = 0.0                                               # dense forward-model prediction error last turn (on CHANGED cells)
        self._prev_feats = None                                              # field FEATURES at the previous turn (FM4 TD target)
        self._tab_spread = 0.0                                               # last turn's tabular value spread (the arbitration signal; 0 = engage the forward model)

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
        if self._prev_feats is not None:                                     # FM4: the field CONFIG that led to completion is
            self.field_value.update(self._prev_feats, max(score_delta, 1.0))  # valuable -> plan toward it next level (goal in feature space)
        self.reward.observe(_GOAL, max(score_delta, 1.0))                    # the goal is the rewarding terminal
        self.new_episode()

    def step(self, state, score_delta: float = 0.0, frame=None):
        """One turn: COMPARE last turn's prediction to the actual `state`, LEARN the transition (column + reward), PLAN
        value over the column's learned transitions, CHOOSE the value-maximising action, then PREDICT the next state.

        With `frame` (FM2): the column ALSO does dense predict-then-compare in L4's feature-FIELD -- L5's per-location
        forward model predicts the next field, compared per location to the actual one. The fraction of CHANGED cells
        it mispredicts is a DENSE, CONTINUOUS learning-progress signal (replacing the opaque binary state-surprise),
        so the agent is drawn to where the DYNAMICS are still learnable (the structured-dynamics games H1 found). The
        field rule is learned online; it persists across episodes (the same mechanic everywhere)."""
        self.surprised = self._pred is not None and state != self._pred       # predict-then-compare = the learning signal
        # ONE-MODEL arbitration + cost gate: run the generative forward model only where the TABULAR value is
        # INDIFFERENT (no spread across actions last turn -- a dynamics game's novel states, where the tabular loop is
        # starved). Where the tabular value leads (a converged decision), skip the forward model entirely -> it never
        # disturbs that policy and costs nothing there. `_tab_spread` is set each turn by `_choose`.
        use_fwd = frame is not None and self._tab_spread <= 1e-9
        field = self.col.feature_field(frame) if use_fwd else None
        feats = self.field_features(field) if field is not None else None
        if field is not None and self._prev_field is not None and self._prev is not None:
            self.field_error = self._field_err(self._pred_field, field, self._prev_field)   # dense error of last turn's field prediction
            self.col.observe_field(self._prev_field, self._prev[1], field)   # learn the per-location rule online (L5)
            self.field_value.update(self._prev_feats,                         # FM4: TD the GOAL in feature space from the score
                                    score_delta + self.reward.gamma * self.field_value.value(feats))
        if feats is not None and score_delta > 0.0:                          # the rewarded CONFIG itself is valuable (state/config
            self.field_value.update(feats, score_delta)                      # reward) -- so planning greedy on V(next) climbs TO the goal,
            #                                                                  not just to the pre-goal (the terminal-credit fix)
        if self._prev is not None:
            ps, pa = self._prev
            self.col.observe(ps, pa, state)                                  # learn the transition online (graph + SR)
            self.tried.add((ps, pa))                                         # attempted (even if blocked -> no edge)
            err = self.field_error if field is not None else (1.0 if self.surprised else 0.0)
            self.reward.observe_error(ps, err)                              # LEARNING-PROGRESS: dense field error (or binary fallback)
        self.reward.observe(state, score_delta)                             # value: reward where the score rose + novelty
        a = self._choose(state, field=field)
        self._pred = self.col.predict(state, a)                             # enter the predictive state
        if field is not None:
            self._pred_field = self.col.predict_field(field, a)             # the efference copy at field grain (the next predicted field)
        self._prev = (state, a)
        self._prev_field = field
        self._prev_feats = feats
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

    def _choose(self, state, field=None):
        """Plan the value, then let the COLUMN's inverse-model motor (`col.act`) select the action that best achieves
        the highest-value next-state -- selection is seated in the column (the motor), not here.

        The PLANNING VALUE is `reward.py`'s PRIORITIZED SWEEPING over the column's transition model (Mattar & Daw
        prioritized replay -- the brain's efficient DEEP planner): bounded gain×need backups propagate BOTH the
        pragmatic reward AND the epistemic frontier through the map, so distant reward/exploration is valued without a
        per-step closed-form solve. (The SR's role here is the NEED that prioritizes the backups -- reference_brain_planning;
        the closed-form V=M·R is dense/O(states^2) and is reserved for the NEED term, not the whole value.)

        With `field` (FM3/FM4): when the tabular value is INDIFFERENT across actions (a dynamics game's novel states)
        the FORWARD MODEL decides instead -- its per-action (pragmatic field-value + epistemic learning-potential)."""
        T, preds = self._transitions()
        if state in T:                                                       # plan only from a state with observed edges
            self.reward.plan(T, preds, state)                               # prioritized sweeping (an unvisited state -> frontier values)
        tab = [self._tab_value(state, a) for a in self.actions]
        self._tab_spread = max(tab) - min(tab)                              # the arbitration signal (0 = the map is indifferent here)
        bonus = None
        if field is not None and self._tab_spread <= 1e-9:                  # the map leads nowhere here -> the FORWARD MODEL decides
            plan = self._field_plan(field, self.actions, self.plan_depth)
            bonus = {a: prag + self.reward.beta * epi for a, (prag, epi) in plan.items()}
        return self.col.act(state, self.actions, value=lambda s: self.reward.V[s], explore=self.reward.explore,
                            tried=self.tried, rng=self.rng, bonus=bonus)

    def _tab_value(self, state, a):
        """The value of action `a` from `state` that `col.act` will use: the bounded frontier optimism for an untried
        action, else the swept value `reward.V[nxt]` of its outcome. The SPREAD across actions is the arbitration
        signal (flat = the map has no preference here -> the forward model)."""
        if (state, a) not in self.tried:
            return self.reward.explore
        return self.reward.V[self.col.graph.get(state, {}).get(a, state)]

    def _field_plan(self, field, actions=None, depth=1):
        """Per-action `(pragmatic, epistemic)` via the forward model, ONE `field_step` pass each (predict + confidence
        shared). PRAGMATIC (FM4) = the field VALUE of the predicted next field (the learned goal in feature space).
        EPISTEMIC (FM3) = the action's LEARNING POTENTIAL (`1 - confidence`). `actions` restricts the evaluation (the
        caller passes only the UNTRIED actions -- the tabular edge handles the rest). `depth > 1` folds the discounted
        best reachable value into pragmatic -- a shallow rollout (sampled/EZ-V2 when deep). So the agent plans TOWARD
        the score (pragmatic) while still drawn to the unlearned (epistemic), the epistemic winding down as mastered."""
        out = {}
        for a in (self.actions if actions is None else actions):
            nxt, conf = self.col.L5.field_step(field, a)                    # ONE pass: predicted field + confidence
            prag = self.field_value.value(self.field_features(nxt))         # pragmatic: value of the predicted next field
            epi = 1.0 - conf                                                # epistemic: learning potential of the action
            if depth > 1:
                sub = self._field_plan(nxt, None, depth - 1)
                prag += self.reward.gamma * max((p + self.reward.beta * e for p, e in sub.values()), default=0.0)
            out[a] = (prag, epi)
        return out

    def field_features(self, field):
        """The field's GENERALISING features for the value (FM4): per-colour cell COUNTS, binned (`field_bin`) so
        nearby configurations share features -> the value generalises over a field that never recurs. The DIFFERENTIATING
        signal is the changing colours' counts (e.g. cn04's growing tree); the background is a near-constant bias feature.
        Game-agnostic -- no domain tokens; the encoding is load-bearing (a feature that CHANGES with the action carries
        the gradient, the ValueLearner lesson)."""
        c = Counter(v for row in field for v in row)
        return frozenset(("cnt", colour, n // self.field_bin) for colour, n in c.items())
