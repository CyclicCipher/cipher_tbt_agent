"""The reward model — learn what is rewarding from the sparse score, then act by active inference.

ARC-AGI-3 gives no goal, only a sparse score (increments on level completion). The agent must LEARN the
reward (the "acquire goals on the fly" capability). This is the piece the column itself does not provide.

Exploration is done the way animals actually do it (see memory: reference_exploration_replay):
  * NOVELTY IS INTRINSIC REWARD, unified with extrinsic value — no separate explore/exploit phase.
    reward(s) = extrinsic_R(s) + beta/(1+visits) (novelty decays with visits). Dopamine encodes
    novelty as reward; one value carries both pragmatic (toward score) and epistemic (toward the unknown).
  * PLANNING = PRIORITIZED REPLAY, not exhaustive value iteration (Mattar & Daw 2018, Nature Neuro).
    Back up states in priority order, priority = GAIN x NEED. GAIN = |value change| (spikes after a
    surprising reward -> propagates backward = reverse replay = credit assignment). NEED = how relevant a
    state is to the agent's future = the SUCCESSOR REPRESENTATION = the grid (here a cheap distance proxy;
    in the column it is place-code similarity). Update only the states that matter, in order.

The world model (transitions) is taken as given — that is the COLUMN's job. State is context-augmented
(cell, ctx) so a context-dependent goal (F_tau(C)) is genuinely PLANNED (navigate switch -> goal), not
stumbled into. Tiny, CPU.
"""

from __future__ import annotations

import random
from collections import defaultdict

MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]                         # 0:up 1:down 2:left 3:right


class GridWorld:
    """Minimal sparse-reward grid: reach the goal cell to score. Optional context: the goal only pays out
    after a switch cell has been visited (a context-dependent goal, to exercise F_tau(C))."""

    def __init__(self, N=7, goal=(6, 6), switch=None, seed=0):
        self.N, self.goal, self.switch = N, goal, switch
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.switched = False
        return self.pos

    def step(self, a):
        dx, dy = MOVES[a]
        x, y = self.pos
        self.pos = (min(max(x + dx, 0), self.N - 1), min(max(y + dy, 0), self.N - 1))
        if self.switch and self.pos == self.switch:
            self.switched = True
        d = 0
        if self.pos == self.goal and (self.switch is None or self.switched):
            d = 1                                                 # scored — caller resets to next level
        return self.pos, d                                       # return GOAL cell (reward attribution)


def augmented_transitions(env):
    """The world model the COLUMN learns, over context-augmented states (cell, ctx): the move operators
    AND that visiting the switch flips a context feature. Returns transitions T and the reverse map."""
    T, preds = {}, defaultdict(list)
    for x in range(env.N):
        for y in range(env.N):
            for ctx in (False, True):
                row = []
                for a in range(4):
                    dx, dy = MOVES[a]
                    nx, ny = min(max(x + dx, 0), env.N - 1), min(max(y + dy, 0), env.N - 1)
                    nctx = ctx or (env.switch is not None and (nx, ny) == env.switch)
                    row.append(((nx, ny), nctx))
                s = ((x, y), ctx)
                T[s] = row
                for nxt in row:
                    preds[nxt].append(s)
    return T, preds


class RewardModel:
    """Unified value (extrinsic reward + intrinsic novelty) maintained by prioritized sweeping
    (priority = gain x need). Set prioritized=False for the exhaustive value-iteration baseline."""

    def __init__(self, N, gamma=0.9, beta=0.3, budget=40, theta=1e-5, prioritized=True, optimistic=True,
                 epistemic="progress", a_fast=0.3, a_slow=0.05, effort=0.0, frontier=1.0, rmax=False):
        self.gamma, self.beta, self.budget, self.theta = gamma, beta, budget, theta
        self.effort = effort                                    # per-step EFFORT cost (a small negative pragmatic value -> efficiency)
        self.frontier = frontier                                # the frontier prior: epistemic optimism for an UNVISITED state (decays to lp)
        self.prioritized, self.sweeps = prioritized, 3 * N
        self.Vmax = 1.0 / (1.0 - gamma)                         # R-MAX optimism = MAX cumulative REWARD (the ablation)
        # The frontier optimism that an UNVISITED state carries. EFE (default): the bounded EPISTEMIC value of an
        # unexplored region = the discounted sum of frontier info-gains (beta*frontier/(1-gamma)) -- it still
        # propagates (so paths toward the unexplored are valued -> coverage), but it is BOUNDED (not max-reward) and
        # DECAYS once visited (reward's lp -> 0 on noise/mastered). `rmax=True` restores the old Vmax ablation.
        self.explore = self.Vmax if rmax else self.beta * self.frontier / (1.0 - gamma)
        self.V = defaultdict((lambda: self.explore) if optimistic else float)   # the ONE value = pragmatic (reward) + epistemic (learning-progress + frontier novelty)
        self.R_ext = {}                                        # extrinsic reward (from the sparse score): +1 rewarding, <0 AVERSIVE (a bad outcome to avoid)
        self.visits = defaultdict(int)                          # visit counts (drive the novelty bonus)
        self.queue = {}                                        # pending value backups: state -> priority
        self.backups = 0                                       # compute counter (efficiency comparison)
        # --- the EPISTEMIC term (what the agent path-integrates TOWARD) -------------------------------------
        # "progress" = LEARNING PROGRESS (the model's prediction-error REDUCTION per state) -- a navigable landscape:
        # the agent is drawn to where it is LEARNING, ignores irreducible noise (the noisy-TV: high error, no
        # reduction) and the already-mastered (no error), and winds into exploitation. "novelty" = count-based (the
        # old bump); "error" = RAW prediction error (the noisy-TV trap -- baseline). See reference_animal_exploration.
        self.epistemic = epistemic
        self.a_fast, self.a_slow = a_fast, a_slow
        self.err_fast: dict = {}                                # fast EWMA of prediction error at s
        self.err_slow: dict = {}                                # slow EWMA (lags) -> slow - fast = recent error REDUCTION

    def observe_error(self, state, error: float) -> None:
        """The model's prediction error at `state` (the agent's surprise when leaving it: 1 mispredicted, 0 nailed).
        Two EWMAs at different rates; their gap (slow - fast) is the LEARNING PROGRESS -- positive while error is
        dropping (learnable), ~0 when error is persistently high (irreducible noise) or persistently low (mastered)."""
        f, s = self.err_fast.get(state, error), self.err_slow.get(state, error)
        self.err_fast[state] = (1.0 - self.a_fast) * f + self.a_fast * error
        self.err_slow[state] = (1.0 - self.a_slow) * s + self.a_slow * error

    def epistemic_value(self, s):
        """The epiplexity-extraction rate at `s` = LEARNING PROGRESS (the prediction-loss drop rate). Visited +
        learnable -> high; a FLAT loss curve -> 0 whether the floor is LOW (mastered) or HIGH (noise), so noise and
        mastery both wind down with NO separate gate; an UNVISITED state -> the frontier prior (optimism that decays to
        the measured lp once visited). The principled, noise-robust epistemic currency (reference_efe_and_epiplexity)."""
        if s not in self.err_slow:                                            # never left s -> unknown epiplexity -> optimism
            return self.frontier
        return max(self.err_slow[s] - self.err_fast[s], 0.0)                  # learning progress = epiplexity-extraction rate

    def reward(self, s):
        """The ONE reward swept into the ONE value: pragmatic (extrinsic score minus a per-step EFFORT cost on non-goal
        states -> efficiency) + epistemic (the learning-progress epiplexity rate, or count novelty / raw error under the
        ablations). Explore/exploit is not a switch -- it EMERGES: an unvisited state carries the bounded frontier
        optimism (coverage), reward propagates via the sweep (exploitation), the epistemic term winds down on
        mastered/noise. There is no separate exploit value and no eigenpurpose (ARCHITECTURE.md §8)."""
        ext = self.R_ext.get(s, 0.0)
        if self.epistemic == "novelty":
            return ext + self.beta / (1.0 + self.visits[s])                    # count-based novelty (the old bump)
        if self.epistemic == "error":
            return ext + self.beta * self.err_fast.get(s, 0.0)                 # RAW error (the noisy-TV trap; baseline)
        # "progress"/EFE = pragmatic (reward minus a per-step EFFORT cost on non-goal states -> efficiency) + epistemic
        # (the epiplexity-extraction rate -- learning progress, no noise gate: a flat loss gives lp -> 0 for noise AND
        # mastered alike). Replaces the count-novelty-gated-by-learnability form with the grounded measure.
        pragmatic = ext - (self.effort if ext <= 0.0 else 0.0)
        return pragmatic + self.beta * self.epistemic_value(s)

    def critic_delta(self, s, s2) -> float:
        """The reward-prediction ERROR δ for the OBSERVED transition s→s2 — the actor-critic TD error the dopamine
        signal represents (B1 of BASAL_GANGLIA_PLAN; the CRITIC that trains the basal-ganglia actor —
        reference_basal_ganglia). δ = r(s) + γ·V(s2) − V(s): the residual of the ONE value's Bellman. δ > 0 = the
        transition did BETTER than the state predicted (a Go signal for the actor); δ < 0 = worse (a NoGo signal). δ → 0
        as the transition is MASTERED — the reward value converges AND lp → 0 — so a fully-predicted reward stops
        training the actor (the dopamine dip). Consumed by the actor in B2/B3."""
        return self.reward(s) + self.gamma * self.V[s2] - self.V[s]

    def _need(self, s, current):
        """NEED = successor-representation relevance of s to the agent's future. The TRUE need is the SR
        under the current (exploring) policy — distant unexplored states have HIGH need because the agent
        will travel to them. A static distance proxy gets this backwards and starves exploration, so we
        leave need flat here (classic gain-prioritized sweeping); the column supplies the real policy-aware
        SR via place-code similarity."""
        return 1.0

    def _push(self, q, s, pri):
        if pri > self.theta:
            q[s] = max(q.get(s, 0.0), pri)

    def observe(self, state, score_delta):
        self.visits[state] += 1
        if score_delta > 0:
            self.R_ext[state] = 1.0                              # infer_goal: reached state was rewarding
        elif score_delta < 0:
            self.R_ext[state] = float(score_delta)              # AVERSION: a bad outcome (score dropped / GAME_OVER) -> a
            #                                                     NEGATIVE preference (`_reward_base` passes it through),
            #                                                     so the EFE value AVOIDS it and the critic δ<0 carries
            #                                                     the cost to the NoGo actor -- the '−' side of pleasure/pain.
        self._push(self.queue, state, 1.0)                     # reward/novelty changed here -> back it up

    def _backup(self, s, T, V, reward_fn):
        self.backups += 1
        nxts = T.get(s, [])                                     # a state absent from T is TERMINAL/unknown (online partial T)
        new_v = (reward_fn(s) if not nxts else
                 max(reward_fn(s) + self.gamma * V[nxt] for nxt in nxts))
        delta = new_v - V[s]
        V[s] = new_v
        return abs(delta)

    def _sweep(self, T, preds, current, V, reward_fn, q):
        """One prioritized sweep (Mattar & Daw) of value dict `V` under `reward_fn`, draining queue `q`: pop the
        highest-priority (gain x need) state, back it up, push its predecessors by the propagated gain. Bounded by
        `budget`; the queue persists pending backups across steps (deep value keeps propagating)."""
        self._push(q, current, 1.0)                            # forward (need) seed from where we are
        for _ in range(self.budget):
            if not q:
                break
            s = max(q, key=q.get)                              # highest priority = gain x need
            del q[s]
            delta = self._backup(s, T, V, reward_fn)
            for p in preds[s]:                                 # propagate to predecessors (reverse replay)
                self._push(q, p, self.gamma * delta * self._need(p, current))

    def plan(self, T, preds, current):
        """Sweep the ONE value V from `current` (Mattar & Daw prioritized replay): pragmatic reward + epistemic
        (learning-progress + frontier novelty), one bounded sweep. Explore/exploit emerges from this one value; there is
        no separate exploit sweep and no eigenpurpose (ARCHITECTURE.md §8)."""
        if self.prioritized:
            self._sweep(T, preds, current, self.V, self.reward, self.queue)
        else:
            for _ in range(self.sweeps):                        # exhaustive value iteration (baseline)
                for s in T:
                    self._backup(s, T, self.V, self.reward)

    def act(self, current, T, preds, rng):
        self.plan(T, preds, current)
        nxts = T[current]
        vals = [self.V[nxt] for nxt in nxts]
        m = max(vals)
        return rng.choice([a for a in range(len(vals)) if vals[a] == m])


class ValueLearner:
    """A TD-learned HORIZON value, for the achiever to bootstrap at the rollout's horizon when a goal is beyond the
    reachable/bounded sweep — so MULTI-STEP goals become plannable (EfficientZero-V2's value, `EZV2_NOTES.md`).
    The piece reward.py was missing: prioritized sweeping propagates value from terminals WITHIN a rollout; this
    estimates the return from a non-terminal horizon state so 'set up the goal' lives in a learned, GENERALISING
    value (no enumeration) — e.g. Tetris L2's multi-piece clear, which the greedy one-piece rollout cannot reach.

    DELIBERATELY GAME-AGNOSTIC: a linear value `V(f) = Σ w[k]` over a FEATURE SET `f` the caller supplies; online
    TD `w[k] += α (target − V(f))`. The caller owns the encoding + the targets (reward + γ·V(next)). The encoding
    is load-bearing — a generic raw-cell encoding fails when a constant substructure dominates; features that
    CHANGE with the action (the gap to a goal) generalise (the lesson from the Tetris L2 validation)."""

    def __init__(self, alpha: float = 0.25):
        self.w: dict = {}
        self.alpha = alpha

    def value(self, feats) -> float:
        w = self.w
        return sum(w.get(k, 0.0) for k in feats)

    def update(self, feats, target: float) -> None:
        delta = self.alpha * (target - self.value(feats))
        for k in feats:
            self.w[k] = self.w.get(k, 0.0) + delta


def run(env, agent="prioritized", steps=400, seed=0, **rmkw):
    rng = random.Random(seed)
    T, preds = augmented_transitions(env)
    rm = None if agent == "random" else RewardModel(env.N, prioritized=(agent == "prioritized"), **rmkw)
    cell = env.reset()
    state = (cell, env.switched)
    completions = 0
    for _ in range(steps):
        a = rng.randrange(4) if rm is None else rm.act(state, T, preds, rng)
        cell, d = env.step(a)
        state = (cell, env.switched)
        if rm is not None:
            rm.observe(state, d)
        if d > 0:
            completions += 1
            cell = env.reset()
            state = (cell, env.switched)
    return completions, (rm.backups if rm else 0)


if __name__ == "__main__":
    print("sparse-reward grid, 400 steps — completions (higher=better) and backups (lower=cheaper)\n")
    for name, kw in [("simple goal", dict(goal=(6, 6))),
                     ("context goal (switch first)", dict(goal=(6, 6), switch=(0, 6)))]:
        pc, pb = run(GridWorld(**kw), agent="prioritized")
        fc, fb = run(GridWorld(**kw), agent="full_vi")
        rc = sum(run(GridWorld(**kw), agent="random", seed=s)[0] for s in range(5)) / 5
        print(f"  {name}")
        print(f"    prioritized   completions {pc:3d}   backups {pb:7d}")
        print(f"    full-VI       completions {fc:3d}   backups {fb:7d}   ({fb / max(pb,1):.0f}x more compute)")
        print(f"    random        completions {rc:5.1f}\n")
