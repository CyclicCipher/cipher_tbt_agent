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


class LearningProgress:
    """LEARNABLE NOVELTY — the EPISTEMIC reward, read off the agent's OWN forward model (`reference_learnable_novelty`;
    Finzi 2026 arXiv:2601.03220 Def 8/eq 8, Zhang & Levin 2026). Epiplexity is the structure a COMPUTE-BOUNDED observer can
    actually extract, estimated PREQUENTIALLY: score each observation with the model BEFORE learning it, and watch the loss
    curve. "For random data the loss NEVER decreases; for simple data it drops FAST and stabilises" — both yield ~0; only
    sustained, slow reduction is real learnable structure.

    The bounded observer is THE AGENT'S OWN MODEL (the L6a operator ⊕ the L5 `Transform` ⊕ its occasions, as composed by
    `hippocampus.WorldModel`) — never a second shadow model. `observe(loss)` takes that model's prequential error on the
    transition just seen; `progress()` is the intrinsic reward:

        progress  =  max(0,  slow_EMA(loss) − fast_EMA(loss))          # prediction-error REDUCTION, not raw error

    NOISE → both EMAs sit high together, gap ~0 (the noisy-TV pathology, dissolved: raw error would be MAXIMAL there).
    MASTERED → both sit low together, gap ~0 (a solved mechanic stops paying, so the drive moves on).
    LEARNING → the fast EMA falls below the slow one, gap > 0 — the learnable frontier, and only there.
    A DARK ROOM is the mastered case (loss ~0 immediately), so it pays nothing either.

    This is `reference_animal_exploration`'s "intrinsic reward = prediction-error REDUCTION" and
    `feedback_epistemic_value_is_prediction_error`, with the error taken from the model whose improvement actually matters
    for planning. `total()` accumulates it — the epiplexity extracted so far."""

    def __init__(self, fast: float = 0.15, slow: float = 0.01) -> None:
        self.fast, self.slow = float(fast), float(slow)
        self._fast_l = None                                   # fast EMA of the model's prequential loss (recent competence)
        self._slow_l = None                                   # slow EMA — the SETTLED loss, i.e. the current floor estimate
        self._sum, self._n = 0.0, 0                           # Σloss and N, for the eq-8 area

    def observe(self, loss: float) -> float:
        """One prequential score of the agent's own model (0 = predicted it exactly, 1 = wholly surprised). Returns the
        resulting `progress()` — the intrinsic reward for this step."""
        loss = float(loss)
        self._fast_l = loss if self._fast_l is None else self._fast_l + self.fast * (loss - self._fast_l)
        self._slow_l = loss if self._slow_l is None else self._slow_l + self.slow * (loss - self._slow_l)
        self._sum += loss
        self._n += 1
        return self.progress()

    def progress(self) -> float:
        """The intrinsic REWARD: the rate at which the model is currently learning — how far recent competence has pulled
        ahead of older competence. Rectified, because a model transiently getting WORSE is not somewhere to be drawn to."""
        if self._fast_l is None:
            return 0.0
        return max(0.0, self._slow_l - self._fast_l)

    def total(self) -> float:
        """EPIPLEXITY (eq 8): the AREA under the prequential loss curve above its settled floor, `Σloss − N·floor`. The
        fluctuations of an unlearnable stream CANCEL in that sum (they are signed), which is what makes noise score ~0 — a
        rectified running total would instead accumulate its jitter and quietly re-admit the noisy TV. Measured on synthetic
        loss curves: dark room 0.0 < noise 2.4 < instantly-mastered 19 < learning 141 < slow/complex 201, which is the
        canonical ordering (`reference_learnable_novelty`: random and simple both ~0, only complex-and-learnable is high)."""
        if self._n == 0:
            return 0.0
        return max(0.0, self._sum - self._n * self._slow_l)


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
