"""The faithful basal ganglia (BASAL_GANGLIA_PLAN; reference_basal_ganglia). Test by MECHANISM, never a score.

B1: the CRITIC's reward-prediction-error δ. `reward.py` IS the actor-critic critic; `critic_delta(s, s2)` = the TD
error δ = r(s) + γ·V(s2) − V(s) the dopamine signal represents. It is the signal the BG ACTOR learns Go/NoGo from
(B2/B3): δ > 0 → a Go (benefit), δ < 0 → a NoGo (cost); δ → 0 as the transition is MASTERED (nothing left to learn)."""

from __future__ import annotations

import os
import sys
from collections import defaultdict

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.reward import RewardModel  # noqa: E402


def _chain():
    """A 4-state chain 0→1→2→3 (one action, deterministic), extrinsic reward only at the goal 3. `preds` is a
    defaultdict(list) — the shape `reward.plan` expects (matching `reward.augmented_transitions`)."""
    T = {0: [1], 1: [2], 2: [3], 3: [3]}
    preds = defaultdict(list, {1: [0], 2: [1], 3: [2, 3]})
    return T, preds


def test_critic_delta_is_the_td_reward_prediction_error():
    """B1: `critic_delta` IS the actor-critic TD reward-prediction-error δ = r(s) + γ·V(s2) − V(s). Once the value has
    CONVERGED and the transitions are MASTERED (lp→0): δ → 0 along the optimal path (a fully-predicted reward drives no
    learning — the dopamine dip), δ equals the explicit formula, and a step AWAY from reward is DISAPPOINTING (δ < 0)."""
    rm = RewardModel(4, gamma=0.9, beta=0.3, epistemic="progress")
    T, preds = _chain()
    for _ in range(120):
        for s in (0, 1, 2, 3):
            rm.observe(s, 1.0 if s == 3 else 0.0)             # extrinsic reward only at the goal
            rm.observe_error(s, 0.0)                          # the world model has MASTERED these transitions → lp = 0
        rm.plan(T, preds, 0)
    # δ IS the explicit TD reward-prediction-error (the residual of the critic's own Bellman)
    for s, s2 in [(0, 1), (1, 2), (2, 3)]:
        expect = rm.reward_exploit(s) + rm.gamma * rm.V_exploit[s2] - rm.V_exploit[s]
        assert abs(rm.critic_delta(s, s2) - expect) < 1e-9
    # mastered + converged → δ ≈ 0 along the optimal path
    opt = [rm.critic_delta(s, s2) for s, s2 in [(0, 1), (1, 2), (2, 3)]]
    assert all(abs(d) < 0.05 for d in opt), opt
    # a step AWAY from reward (to a lower-value state) is worse than the state predicted → δ < 0 (a NoGo signal)
    assert rm.critic_delta(2, 1) < -0.01, rm.critic_delta(2, 1)


def test_critic_delta_winds_down_as_the_reward_is_learned():
    """δ is the LEARNING signal: while the reward is still propagating the value, |δ| along the path is appreciable;
    as the value converges (the reward is mastered) |δ| → 0. The RPE winds down when there is nothing left to learn —
    which is exactly why a mastered edge stops training the actor."""
    rm = RewardModel(4, gamma=0.9, beta=0.0, epistemic="progress")   # beta=0 isolates the reward-prediction term
    T, preds = _chain()

    def one_round():
        for s in (0, 1, 2, 3):
            rm.observe(s, 1.0 if s == 3 else 0.0)
            rm.observe_error(s, 0.0)
        rm.plan(T, preds, 0)

    one_round()                                              # the reward is known but not yet propagated up the chain
    early = abs(rm.critic_delta(2, 3))
    for _ in range(120):
        one_round()
    late = abs(rm.critic_delta(2, 3))
    assert early > 0.1, early                                # early: the reward at 3 has not reached V(2) yet
    assert late < 0.05 and late < early, (early, late)       # mastered: δ → 0


# ── B3: the OpAL Go/NoGo opponent actor (basal_ganglia.OpponentActor) ────────────────────────────────────────
def test_opal_actor_learns_benefits_and_costs_and_represents_aversion():
    """B3: the Go/NoGo actor learns from the critic RPE δ. A BENEFIT action (repeated δ>0) grows Go over NoGo; a COST
    action (repeated δ<0) grows NoGo over Go and earns a NEGATIVE actor value -- the principled AVERSION a single reward
    value cannot represent. The actor prefers the benefit to the cost. Neutral (untrained) actions contribute 0."""
    from tbt.basal_ganglia import OpponentActor
    ac = OpponentActor(alpha_g=0.2, alpha_n=0.2, beta=1.0, init=1.0)
    assert ac.act_value("s", 9) == 0.0                       # untrained (Go=NoGo, ρ=0) -> no bias: behaviour-neutral to wire
    for _ in range(20):
        ac.learn("s", 0, +0.5)                               # action 0 = a BENEFIT (dopamine bursts)
        ac.learn("s", 1, -0.5)                               # action 1 = a COST (dopamine dips)
    assert ac.G[("s", 0)] > ac.N[("s", 0)], "benefit: Go should exceed NoGo"
    assert ac.N[("s", 1)] > ac.G[("s", 1)], "cost: NoGo should exceed Go"
    assert ac.act_value("s", 1) < 0.0, "a cost action has NEGATIVE actor value (aversion)"
    assert ac.act_value("s", 0) > ac.act_value("s", 1), "the actor prefers the benefit to the cost"


def test_opal_tonic_dopamine_sets_the_go_nogo_gain():
    """B3: tonic dopamine `ρ` sets the Go/NoGo gain balance. Rich (ρ>0) amplifies the GO (benefit) side -> a benefit
    action is valued MORE; lean (ρ<0) amplifies the NoGo (cost) side -> a cost action is valued LESS (more aversive).
    This is the OpAL explore/exploit + vigor knob, not a hard switch."""
    from tbt.basal_ganglia import OpponentActor
    ac = OpponentActor(alpha_g=0.2, alpha_n=0.2, beta=1.0, init=1.0)
    for _ in range(20):
        ac.learn("s", 0, +0.5)                               # a benefit-via-Go action
        ac.learn("s", 1, -0.5)                               # a cost-via-NoGo action
    assert ac.act_value("s", 0, rho=+0.8) > ac.act_value("s", 0, rho=-0.8)   # rich DA amplifies the benefit (Go)
    assert ac.act_value("s", 1, rho=-0.8) < ac.act_value("s", 1, rho=+0.8)   # lean DA amplifies the aversion (NoGo)


# ── AVERSION: the '−' side of pleasure/pain, as a NEGATIVE preference in the free-energy value ────────────────
def test_negative_reward_makes_the_efe_value_avoid_aversion():
    """A bad outcome recorded with score_delta<0 becomes a NEGATIVE R_ext -- the aversion the free-energy value lacked.
    The model-based EFE value then AVOIDS it (a fork's reward branch outvalues its aversion branch; the aversive state's
    value is negative), and the critic δ for stepping INTO aversion is NEGATIVE (the cost the NoGo actor learns from).
    This is where 'make pragmatic value negative' actually lands -- the pragmatic term's '−' side."""
    rm = RewardModel(9, gamma=0.9, beta=0.0, epistemic="progress")   # beta=0 isolates the pragmatic (reward) value
    T = {0: [1, 3], 1: [2], 2: [2], 3: [4], 4: [4]}                  # a fork: 0→1→2 (REWARD) vs 0→3→4 (AVERSION)
    preds = defaultdict(list, {1: [0], 2: [1], 3: [0], 4: [3]})
    for _ in range(80):
        rm.observe(2, +1.0)                                          # state 2 = reward
        rm.observe(4, -1.0)                                          # state 4 = AVERSION (a negative preference)
        for s in (0, 1, 2, 3, 4):
            rm.observe(s, 0.0)
            rm.observe_error(s, 0.0)                                 # mastered -> lp = 0 (isolate the pragmatic value)
        rm.plan(T, preds, 0)
    assert rm.V_exploit[2] > 0.5, rm.V_exploit[2]                    # reward: positive value
    assert rm.V_exploit[4] < -0.5, rm.V_exploit[4]                  # AVERSION: negative pragmatic value
    assert rm.V_exploit[1] > rm.V_exploit[3]                        # the value AVOIDS the aversion branch
    # the 'pain' signal: on FIRST experiencing an aversive state (its value not yet learned) the critic δ is negative --
    # the cost the NoGo actor learns from. δ(4→4) = reward(4) + γ·V(4) − V(4) = −1 while V(4) is still ~0.
    fresh = RewardModel(9, gamma=0.9, beta=0.0, epistemic="progress")
    fresh.observe(4, -1.0)                                          # freshly experience the aversion at 4
    fresh.observe_error(4, 0.0)
    assert fresh.critic_delta(4, 4) < 0.0, fresh.critic_delta(4, 4)  # being at an aversive state is worse than expected -> δ<0
