"""The reward model — learn what is rewarding from the sparse score, then act by active inference.

ARC-AGI-3 gives no goal, only a sparse score (increments on level completion). The agent must LEARN the
reward (the "acquire goals on the fly" capability). This is the piece the column itself does not provide.

Exploration is done the way animals actually do it (see memory: reference_exploration_replay):
  * NOVELTY IS INTRINSIC REWARD, unified with extrinsic value — no separate explore/exploit phase.
    reward_total(s) = extrinsic_R(s) + beta/(1+visits) (novelty decays with visits). Dopamine encodes
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
                 epistemic="progress", a_fast=0.3, a_slow=0.05):
        self.gamma, self.beta, self.budget, self.theta = gamma, beta, budget, theta
        self.prioritized, self.sweeps = prioritized, 3 * N
        self.Vmax = 1.0 / (1.0 - gamma)                         # optimism under uncertainty (R-MAX): an
        self.V = defaultdict((lambda: self.Vmax) if optimistic else float)   # unvisited state maximally valued,
                                                                # so the frontier attracts — forward novelty drive.
                                                                # But for PLANNING over a fully-enumerated subgoal
                                                                # -MDP set optimistic=False (0-init), or undecayed
                                                                # optimism on a not-yet-winning self-loop (reach
                                                                # goal before clearing its condition) outvalues
                                                                # real progress and the agent lunges early.
        self.R_ext = {}                                         # extrinsic reward (from the sparse score)
        self.visits = defaultdict(int)                          # visit counts (drive the novelty bonus)
        self.queue = {}                                         # pending backups: state -> priority
        self.backups = 0                                        # compute counter (efficiency comparison)
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

    def reward_total(self, s):
        ext = self.R_ext.get(s, 0.0)
        if self.epistemic == "novelty":
            return ext + self.beta / (1.0 + self.visits[s])                    # count-based novelty (the old bump)
        if self.epistemic == "error":
            return ext + self.beta * self.err_fast.get(s, 0.0)                 # RAW error (the noisy-TV trap; baseline)
        # "progress" = the novelty frontier drive GATED by learnability. The gate is 1 wherever the model is learning or
        # has learned (so on deterministic structure this is IDENTICAL to count-novelty -- no distraction) and falls to
        # 0 on CONFIRMED noise: persistent prediction error NOT explained by ongoing learning progress (the noisy TV).
        # So the agent stops paying novelty to a region it has proven it cannot predict, unlike RAW error which clings.
        lp = max(self.err_slow.get(s, 0.0) - self.err_fast.get(s, 0.0), 0.0)  # measured progress = recent error reduction
        noise = max(self.err_slow.get(s, 0.0) - 4.0 * lp, 0.0)               # persistent error with NO progress = noise
        gate = 1.0 - min(noise, 1.0)
        return ext + self.beta * gate / (1.0 + self.visits[s])

    def _need(self, s, current):
        """NEED = successor-representation relevance of s to the agent's future. The TRUE need is the SR
        under the current (exploring) policy — distant unexplored states have HIGH need because the agent
        will travel to them. A static distance proxy gets this backwards and starves exploration, so we
        leave need flat here (classic gain-prioritized sweeping); the column supplies the real policy-aware
        SR via place-code similarity."""
        return 1.0

    def _push(self, s, pri):
        if pri > self.theta:
            self.queue[s] = max(self.queue.get(s, 0.0), pri)

    def observe(self, state, score_delta):
        self.visits[state] += 1
        if score_delta > 0:
            self.R_ext[state] = 1.0                              # infer_goal: reached state was rewarding
        self._push(state, 1.0)                                  # reward/novelty changed here — back it up

    def _backup(self, s, T):
        self.backups += 1
        nxts = T.get(s, [])                                     # a state absent from T is TERMINAL/unknown (online partial T)
        new_v = (self.reward_total(s) if not nxts else
                 max(self.reward_total(s) + self.gamma * self.V[nxt] for nxt in nxts))
        delta = new_v - self.V[s]
        self.V[s] = new_v
        return abs(delta)

    def plan(self, T, preds, current):
        if self.prioritized:
            self._push(current, 1.0)                            # forward (need) seed from where we are
            for _ in range(self.budget):
                if not self.queue:
                    break
                s = max(self.queue, key=self.queue.get)         # highest priority = gain x need
                del self.queue[s]
                delta = self._backup(s, T)
                for p in preds[s]:                              # propagate to predecessors (reverse replay)
                    self._push(p, self.gamma * delta * self._need(p, current))
        else:
            for _ in range(self.sweeps):                        # exhaustive value iteration (baseline)
                for s in T:
                    self._backup(s, T)

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
