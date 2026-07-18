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
