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

from .basal_ganglia import BasalGanglia
from .column import CorticalColumn, GoalState, IMPASSABLE
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
        self._ep_cache: dict = {}                                          # the last computed eigenpurpose (reused between recomputes)
        self._ep_tick = 0                                                  # THROTTLE: the grid SVD is O(n^3) -> recompute the eigenpurpose only every _ep_every steps
        self._ep_every = 16                                                # the exploration DIRECTION is slow-changing, so a stale-by-a-few-steps eigenpurpose is fine
        self.bg = BasalGanglia(n_columns=1, seed=seed)                     # the GSG gate: arbitrates the column's goal candidates (ACT vs DISAMBIGUATE)
        self._integrate = False                                            # V4: position/integrate mode (set by TbtPolicy) -> the cost-aware achiever is live
        self._goal_pos = None                                              # V4: the remembered goal POSITION (from a completion) -- the `reward` GSG generator's target
        self._goal_raw = None                                              # V4 (S1e non-abelian): the goal in RAW metric coords (the pose achiever measures DISTANCE, not just direction, so it needs raw not the binned node)
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
        self.sensed_surprise = False                                         # C2: last turn's L4-over-L6 feature-at-location mismatch
        self.goal = None                                                     # the basal-ganglia-selected GoalState this turn (GSG in the loop)
        self._prev_pose = None                                               # S1e: the RAW pose belief when `_prev`'s state was observed (so the goal = pre-pose ∘ completing-op)

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
            if self._integrate:                                             # V4: REMEMBER the goal position -- next level the achiever beelines here (transfer)
                self._goal_pos = ps
                # S1e (non-abelian): the RAW metric goal = where the COMPLETING action LANDED = the pre-completion pose
                # composed with that action's learned operator (exact + reset-timing-robust; the env resets on completion,
                # so reading the post-frame position is unreliable). The learned `pose_ops` doing double duty.
                if self._prev_pose is not None and pa in self.col.pose_ops:
                    gp = self._prev_pose @ self.col.pose_ops[pa].M
                    self._goal_raw = (float(gp[0, 2]), float(gp[1, 2]))
        if self._prev_feats is not None:                                     # FM4: the field CONFIG that led to completion is
            self.field_value.update(self._prev_feats, max(score_delta, 1.0))  # valuable -> plan toward it next level (goal in feature space)
        self.reward.observe(_GOAL, max(score_delta, 1.0))                    # the goal is the rewarding terminal
        self.new_episode()

    def step(self, state, score_delta: float = 0.0, frame=None, feature=None, location=None, cloud=None):
        """One turn: COMPARE last turn's prediction to the actual `state`, LEARN the transition (column + reward), PLAN
        value over the column's learned transitions, CHOOSE the value-maximising action, then PREDICT the next state.

        C2 (COLUMN_AUDIT): with `feature` (the egocentric feature sensed at the current L6 LOCATION `state`), the column
        also runs the L4-over-L6 predict-then-compare CYCLE (`sense_at`) -- predict the feature at the location, compare,
        learn -- so the OBJECT emerges as the feature-at-location map. Optional + additive: a featureless world passes
        `feature=None` and the loop is unchanged.

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
            if self._integrate:                                            # V4 COST-FIELD assignment: LEARN repulsion from experience (positions only)
                dx, dy = self.col.L5.move(pa)
                if (dx, dy) != (0, 0) and state == ps:                     # a no-progress BUMP -> the intended cell is a WALL (cost=inf)
                    self.col.learn_cost((ps[0] + dx, ps[1] + dy), IMPASSABLE)
                if score_delta < 0.0:                                      # AVERSION -> a graded hazard cost at the cell entered (the same currency)
                    self.col.learn_cost(state, -float(score_delta))
        self.reward.observe(state, score_delta)                             # value: reward where the score rose + novelty
        if cloud is not None:                                               # C4: the perception->recognition->map pipeline (a whole object)
            loc = location if location is not None else state
            _name, self.sensed_surprise = self.col.sense_object(cloud, loc)  # L2/3 RECOGNISES (pose-invariant) -> map the identity at the location
        elif feature is not None:                                           # C2: the L4-over-L6 cycle -- the object emerges at the location
            loc = location if location is not None else state              # bind at the L6 POSITION (given explicitly when the state is a joint symbol)
            self.sensed_surprise = self.col.sense_at(loc, feature)          # predict feature-at-location -> compare -> LEARN (bind)
        a = self._choose(state, field=field)
        self._pred = self.col.predict(state, a)                             # enter the predictive state
        if field is not None:
            self._pred_field = self.col.predict_field(field, a)             # the efference copy at field grain (the next predicted field)
        self._prev = (state, a)
        self._prev_pose = self.col._pose.copy() if self.col._pose is not None else None   # S1e: the raw pose at this state (for the completion-goal derivation)
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

        B2+B4 (BASAL_GANGLIA_PLAN) -- the DISSOLUTION: there is no exploit/explore SWITCH. The BG disinhibits the max
        of ONE SALIENCE -- `V_exploit` (reward + epistemic) plus the L6 EIGENPURPOSE contribution scaled by the TONIC-DA
        EXPLORE GAIN `g`, plus the L5 FORWARD-MODEL (field) term. `g = exp(-decay * V_reach)` is set by reward
        AVAILABILITY (the SR expected reachable reward from here, M1): no reward reachable -> g~1 (the eigenpurpose
        drives directed exploration); reward reachable -> g -> 0 (V_exploit exploits). This retires BOTH the boolean
        switch AND the global dead-zone flag (now graded + per-state, via SR reachability). The opponent Go/NoGo actor
        (learned by the critic delta) arrives in B3; the field is compute-gated by `_tab_spread` (efficiency, not arbitration)."""
        self._ep_tick += 1                                                  # throttle the O(n^3) grid SVD: recompute the eigenpurpose occasionally, reuse between
        if self._ep_tick % self._ep_every == 1:
            self._ep_cache = self.col.sr.eigenpurpose(self.reward.visits, self.reward.beta)   # L6 owns the grid math
        self.reward.intrinsic = self._ep_cache                             # L6 eigenpurpose -> a standing salience term (propagated by the sweep)
        T, preds = self._transitions()
        if state in T:                                                       # plan only from a state with observed edges
            self.reward.plan(T, preds, state)                               # sweeps the combined value V (reward + epistemic + eigenpurpose) AND V_exploit
        # the EXPLORE GAIN g = exp(-decay * tab_spread): reward AVAILABILITY = whether the clean value has a GRADIENT to
        # exploit here (fast, swept) -- a gradient -> g~0 (V_exploit exploits), flat -> g~1 (eigenpurpose explores).
        ev = [self._explore_value(state, a, self.reward.V_exploit) for a in self.actions]
        self._tab_spread = max(ev) - min(ev)                               # clean-reward action-spread (also the forward-model cost gate in `step`)
        # the EXPLORE GATE g: directed exploration (the eigenpurpose) fires ONLY where there is no reward gradient to
        # follow -- a flat clean value (`_tab_spread<=eps`) OR the reward-less, locally-exhausted DEAD-ZONE; else a
        # reward gradient exists -> EXPLOIT (g=0, V_exploit decides). A sharp VALUE-LANDSCAPE signal, not a domain rule.
        # NB the eigenpurpose gate is inherently SHARP; the GRADED tonic-DA gain is a SEPARATE axis -- it lands on the
        # Go/NoGo opponency (B3/B4), not here (the empirical finding that decoupled B2 from B4).
        dead_zone = not self.reward.R_ext and all((state, a) in self.tried for a in self.actions)
        g = 1.0 if (self._tab_spread <= 1e-9 or dead_zone) else 0.0
        bonus = None
        if field is not None and g > 0.0:                                  # the L5 forward-model dynamics term is an EXPLORE-side salience term (like the eigenpurpose): gated by g
            fplan = self._field_plan(field, self.actions, self.plan_depth)
            bonus = {a: prag + self.reward.beta * epi for a, (prag, epi) in fplan.items()}
        # the per-action SALIENCE the BG disinhibits the max of = the g-blended value of the outcome (+ the field term).
        blended = lambda s: self.reward.V_exploit[s] + g * (self.reward.V[s] - self.reward.V_exploit[s])
        # THE GSG COMPETITION (Phase II unification): propose candidate GOAL-STATES → ONE basal-ganglia arbitration → the
        # winner is EXECUTED by the shared ACHIEVER (a target goal) or the degenerate `col.act` (the act goal). `self.goal`
        # is now LIVE -- it DISPATCHES execution (no longer computed-and-discarded). Candidates: `act` (greedy value,
        # always) + `disambiguate` (L2/3, when hypotheses compete) + `reward` -- the REMEMBERED completing target, a
        # NAVIGABLE goal valued by its propagated exploit value, generated in the EXPLOIT regime (a reward gradient exists:
        # g<=0). This retires the separate V4 exploit branch. [the `dynamics` (transition-lp) generator = a later fold.]
        goals = self.col.propose_goals(max(ev), g_value=self.reward.beta)
        if self._integrate and self._goal_pos is not None and g <= 0.0 and state != self._goal_pos:
            # EXPLOIT regime (g<=0 already arbitrated explore→exploit): the ACT goal competes on its CLEAN exploit value
            # (the untried-frontier optimism in `ev` is an EXPLORE signal, irrelevant here); the REWARD generator adds the
            # remembered completing TARGET valued by the PEAK exploit value (the goal is where value peaks) -> it wins + beelines.
            clean_act = max((self.reward.V_exploit[self.col.predict(state, a)] for a in self.actions), default=0.0)
            goals[0] = (goals[0][0], float(clean_act))
            target = self._goal_raw if (self.col.L5.heading_dependent() and self._goal_raw is not None) else self._goal_pos
            goals.append((GoalState(target=target, kind="reward"), float(max(self.reward.V_exploit.values(), default=0.0))))
        gi = self.bg.gate([gc.kind for gc, _ in goals], [v for _, v in goals]) if len(goals) > 1 else 0
        self.goal = goals[gi][0]
        # DISPATCH on the selected goal: a reward TARGET → the cost-aware ACHIEVER (vector nav -- the beeline, curving
        # around learned walls/hazards, no frontier optimism); the degenerate ACT goal (and, until B5 wires it,
        # disambiguate) → the inverse-model motor `col.act` (the value/eigenpurpose policy).
        if self.goal.kind == "reward" and self.goal.target is not None:
            here = self.col.here_position() if self.col.L5.heading_dependent() else state
            a = self.col.achieve(here, self.goal.target, self.actions)
            if a is not None:
                return a
        return self.col.act(state, self.actions, value=blended,
                            explore=self.reward.explore, tried=self.tried, rng=self.rng, bonus=bonus)

    def _explore_value(self, state, a, V):
        """The value of action `a` for the arbitration: the frontier optimism for an UNTRIED action, else `V[next]` of
        its outcome. Evaluated over V_exploit (no eigenpurpose), its action-spread is the arbitration signal: >0 = the
        value decides (normal reward-seeking / local exploration); flat = the locally-exhausted, reward-less dead-zone
        (switch to the eigenpurpose-laden V). The same flat-gate keeps the dense forward model off during normal nav."""
        if (state, a) not in self.tried:
            return self.reward.explore
        return V[self.col.graph.get(state, {}).get(a, state)]

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
