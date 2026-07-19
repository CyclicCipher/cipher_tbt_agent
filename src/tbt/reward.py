"""reward.py — the VALUE CRITIC (ARCHITECTURE.md §3, the loop's "evaluate" arm; ROADMAP Phase 3c).

The cortex is value-free; the CRITIC is the one place a scalar "how good is this state" lives. It is a TEMPORAL-DIFFERENCE
value over the state SDR: `V(s) = Σ w[bit]`, learned by the TD error `δ = r + γ·V(s′) − V(s)`, `w[bit] += lr·δ`. That δ is
the DOPAMINE reward-prediction-error the basal ganglia's OpAL actor already trains on — so the critic REPLACES the faked
immediate `2r−1`: with a critic, δ carries an EXPECTATION (a baseline) and BOOTSTRAPS value backward across steps, which a
single immediate reward cannot.

ONE RULE, THREE PLACES (the unification worth seeing). `δ = r + γV(s′) − V(s)` is the **same delta rule** as
  * Rescorla-Wagner cue competition (`reference_cue_competition_key_discovery`, the operator's KEY): `w += lr·(US − Σ present)`,
  * the `_Readout` SDRClassifier (`reference_htm_canonical_pipeline`),
  * dopamine-RPE in the basal ganglia (`reference_basal_ganglia`).
The critic is not new machinery; it is that rule applied to VALUE over time. Build the critic, and the object forward-model's
cue-competition and the actor's RPE share a substrate.

SCOPE / honest limits. This is a LINEAR value over the state SDR — the ROUTINE / read-off critic (`reference_brain_planning`:
routine value = a cheap dot product; rollout = the sparing fallback). A linear value CANNOT hold a relational V*
(`project_linear_value_cannot_hold_sokoban`), so relational tasks need ROLLOUT, which uses THIS critic as the leaf evaluator —
that is why the critic comes before the rollout (3c before Phase 6+). The SUCCESSOR-REPRESENTATION form (V = M·R, the discounted
resolvent of the 3a operator; grid cells = SR eigenvectors, `reference_grid_sr_eigenbasis`) is the refinement that buys fast
re-tuning when the reward moves + cheap planning — deferred to a task that needs it. Pure stdlib.
"""

from __future__ import annotations


class ValueCritic:
    """A temporal-difference value over the state SDR. `value(state)` reads it; `learn(state, r, next_state, done)` updates it
    by the TD error and RETURNS that δ for the actor (the basal ganglia). `rho()` is a running-average reward — the tonic-DA
    signal the OpAL actor can use for explore/exploit (`reference_basal_ganglia`), available but not yet wired."""

    def __init__(self, lr: float = 0.1, gamma: float = 0.9, rho_lr: float = 0.01) -> None:
        self.w: dict = {}                                    # state SDR bit -> value weight (a sparse linear value)
        self.lr, self.gamma, self.rho_lr = float(lr), float(gamma), float(rho_lr)
        self._rho = 0.0                                      # running-average reward (tonic dopamine)

    def value(self, state) -> float:
        """V(state) = Σ of the weights of the active bits — the sparse linear value. An unseen bit contributes 0 (the honest
        prior: no evidence, no value)."""
        return sum(self.w.get(b, 0.0) for b in state)

    def learn(self, state, r: float, next_state=None, done: bool = True) -> float:
        """One TD update. `δ = r + γ·V(next) − V(state)` (no bootstrap on a terminal step, or when `next_state` is None);
        the active bits of `state` share the move so `V` shifts by ≈`lr·δ`. Returns δ — the dopamine-RPE the actor learns from.
        For an IMMEDIATE-reward
        bandit (`done=True`) this is `δ = r − V(state)`: a real prediction error with a learned baseline, strictly better than
        the faked `2r−1` (which had no expectation)."""
        state = list(state)
        boot = 0.0 if (done or next_state is None) else self.gamma * self.value(next_state)
        delta = float(r) + boot - self.value(state)
        step = self.lr * delta / max(1, len(state))          # normalise by |active bits|, so V moves by ≈lr·δ regardless of
        for b in state:                                      # SDR sparsity — else the effective rate is |state|·lr and diverges
            self.w[b] = self.w.get(b, 0.0) + step
        self._rho += self.rho_lr * (float(r) - self._rho)
        return delta

    def rho(self) -> float:
        """The running-average reward — tonic dopamine, the OpAL explore/exploit + vigor gain (rich → exploit, lean → avoid)."""
        return self._rho


class GoalMemory:
    """The discovered GOAL: WHICH perceptual feature the reward is contingent on. It is the SAME delta rule as the critic above
    — Rescorla-Wagner cue competition (`reference_cue_competition_key_discovery`) applied to reward CONTINGENCY, feeding the
    priority map that sets the goal (`reference_goal_setting_priority_map`: reward → a valued feature → the goal-vector). On
    every step the feature the self REACHED is credited with the step's reward (Δscore): the goal feature (reached AT reward)
    climbs toward 1 while features the self reaches WITHOUT reward (a key, a pad) decay toward 0 — competition washes out the
    spurious co-occurrences that a single-trial memory would enshrine.

    WHY A FEATURE, NOT A PLACE. The credited thing is an OBJECT PROPERTY (colour for a single-cell object; a recognised identity
    once shape matters), so the goal is invariant to WHERE it sits. That is exactly what the absolute-position `ValueCritic`
    cannot give: the critic's value learned at one level's goal cell is worthless at the next level's different cell, but the
    goal FEATURE names the goal wherever it appears — so it transfers across levels. Pure stdlib."""

    def __init__(self, lr: float = 0.5, eps: float = 0.1) -> None:
        self.w: dict = {}                                     # feature -> reward contingency (the delta-rule weight)
        self.lr, self.eps = float(lr), float(eps)

    def credit(self, features, r: float) -> None:
        """One delta-rule trial: the PRESENT features share the prediction error `r − Σ w[present]` (so co-present cues compete),
        each moving toward the reward `r`. Call every step with the feature(s) the self reached and that step's reward (Δscore)."""
        present = list(features)
        if not present:
            return
        err = float(r) - sum(self.w.get(f, 0.0) for f in present)
        for f in present:
            self.w[f] = self.w.get(f, 0.0) + self.lr * err

    def goal(self):
        """The most reward-contingent feature (argmax weight, if it clears `eps`) — the discovered goal, or None until a reward
        has credited one. Feature-based, so it names the goal object wherever on the board it sits (the transfer lever)."""
        if not self.w:
            return None
        f, v = max(self.w.items(), key=lambda kv: kv[1])
        return f if v >= self.eps else None
