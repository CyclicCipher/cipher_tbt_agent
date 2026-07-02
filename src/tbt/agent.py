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

from .basal_ganglia import BasalGanglia
from .column import CorticalColumn, GoalState, IMPASSABLE
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
        self.bg = BasalGanglia(n_columns=1, seed=seed)                     # the GSG gate: arbitrates the column's goal candidates (ACT vs DISAMBIGUATE)
        self._integrate = False                                            # V4: position/integrate mode (set by TbtPolicy) -> the cost-aware achiever is live
        self._goal_pos = None                                              # V4: the remembered goal POSITION (from a completion) -- the `reward` GSG generator's target
        self._goal_raw = None                                              # V4 (S1e non-abelian): the goal in RAW metric coords (the pose achiever measures DISTANCE, not just direction, so it needs raw not the binned node)
        self.new_episode()

    def new_episode(self):
        self._prev = None                                                    # (state, action) of the previous turn
        self._pred = None                                                    # the predictive state (predicted current state)
        self.surprised = False
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
        self.reward.observe(_GOAL, max(score_delta, 1.0))                    # the goal is the rewarding terminal
        self.new_episode()

    def step(self, state, score_delta: float = 0.0, frame=None, feature=None, location=None, cloud=None):
        """One turn: COMPARE last turn's prediction to the actual `state`, LEARN the transition (column + reward), PLAN
        value over the column's learned transitions, CHOOSE the value-maximising action, then PREDICT the next state.

        C2 (COLUMN_AUDIT): with `feature` (the egocentric feature sensed at the current L6 LOCATION `state`), the column
        also runs the L4-over-L6 predict-then-compare CYCLE (`sense_at`) -- predict the feature at the location, compare,
        learn -- so the OBJECT emerges as the feature-at-location map. Optional + additive: a featureless world passes
        `feature=None` and the loop is unchanged.

        `frame` is the raw sensory field (ARC's colour grid). It is NOT consumed here yet: the factored perception that
        turns it into (location, content) is P1, and the ONE operator prediction over that is P2 (ARCHITECTURE.md). Until
        then the loop runs on the discrete `state` + the L4-over-L6 cycle; `frame` is accepted only so the sensory
        contract (arc_sdk) stays stable across the P1 build."""
        self.surprised = self._pred is not None and state != self._pred       # predict-then-compare = the learning signal
        if self._prev is not None:
            ps, pa = self._prev
            self.col.observe(ps, pa, state)                                  # learn the transition online (graph + SR)
            self.tried.add((ps, pa))                                         # attempted (even if blocked -> no edge)
            self.reward.observe_error(ps, 1.0 if self.surprised else 0.0)   # LEARNING-PROGRESS: the predict-then-compare error
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
        a = self._choose(state)
        self._pred = self.col.predict(state, a)                             # enter the predictive state
        self._prev = (state, a)
        self._prev_pose = self.col._pose.copy() if self.col._pose is not None else None   # S1e: the raw pose at this state (for the completion-goal derivation)
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

    def _choose(self, state):
        """Plan the ONE value, then let the COLUMN's inverse-model motor (`col.act`) select the value-maximising action
        (selection seated in the column, not here).

        The PLANNING VALUE is `reward.py`'s PRIORITIZED SWEEPING over the column's transition model (Mattar & Daw
        prioritized replay -- the brain's efficient DEEP planner): bounded gain×need backups propagate BOTH the
        pragmatic reward AND the epistemic frontier through the map, so distant reward/exploration is valued without a
        per-step closed-form solve.

        Explore vs exploit EMERGES from this ONE value (ARCHITECTURE.md §8), not a switch: an unvisited state carries the
        bounded frontier optimism (coverage), reward propagates via the sweep (exploitation), the epistemic term winds
        down on mastered/noise. There is no `g`-gate, no `V`/`V_exploit` split, and no eigenpurpose (dropped -- a
        redundant, costly, task-blind duplicate of the epistemic term). When a reward TARGET is remembered, the GSG's
        reward goal competes in the ONE basal-ganglia competition and, if it wins, dispatches the SR-geodesic achiever."""
        T, preds = self._transitions()
        if state in T:                                                       # plan only from a state with observed edges
            self.reward.plan(T, preds, state)                               # one bounded sweep of the ONE value V
        V = self.reward.V
        act_value = max((V[self.col.predict(state, a)] for a in self.actions), default=0.0)  # the ACT goal's value
        # THE GSG COMPETITION: candidate GOAL-STATES → ONE basal-ganglia arbitration → the winner is EXECUTED by the
        # shared ACHIEVER (a target goal) or the degenerate `col.act` (the act goal). Candidates: `act` (greedy value,
        # always) + `disambiguate` (L2/3, when hypotheses compete) + `reward` (the REMEMBERED completing target).
        goals = self.col.propose_goals(act_value, g_value=self.reward.beta)
        if self._integrate and self._goal_pos is not None and state != self._goal_pos:
            # a reward TARGET is remembered -> the REWARD generator adds it, valued by the PEAK value (the goal is where
            # value peaks); on transfer levels it beats acting and the achiever beelines. Cold levels (no goal yet) fall
            # through to `col.act`, whose frontier optimism IS the directed explorer.
            target = self._goal_raw if (self.col.L5.heading_dependent() and self._goal_raw is not None) else self._goal_pos
            goals.append((GoalState(target=target, kind="reward"), float(max(V.values(), default=0.0))))
        gi = self.bg.gate([gc.kind for gc, _ in goals], [v for _, v in goals]) if len(goals) > 1 else 0
        self.goal = goals[gi][0]
        # DISPATCH on the selected goal: a reward TARGET → the cost-aware ACHIEVER (SR-geodesic beeline, curving around
        # learned walls/hazards); the degenerate ACT goal (and, until B5 wires it, disambiguate) → the inverse-model
        # motor `col.act` (the value + frontier-optimism per-action policy, which IS the directed explorer).
        if self.goal.kind == "reward" and self.goal.target is not None:
            here = self.col.here_position() if self.col.L5.heading_dependent() else state
            a = self.col.achieve(here, self.goal.target, self.actions)
            if a is not None:
                return a
        return self.col.act(state, self.actions, value=lambda s: V[s],
                            explore=self.reward.explore, tried=self.tried, rng=self.rng)
