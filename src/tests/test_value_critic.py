"""End-to-end test of the VALUE CRITIC (ARCHITECTURE.md §3; ROADMAP Phase 3c) — `reward.py`, wired into the decision loop.

The critic is a temporal-difference value over the state SDR: `V(s) = Σ w[bit]`, learned by `δ = r + γ·V(s′) − V(s)`. That δ
is the dopamine reward-prediction-error the basal ganglia already trains on — so it REPLACES the faked immediate `2r−1`. What
a real critic buys, and the faked RPE cannot: an EXPECTATION (a baseline, so `δ = r − V` is a true prediction error) and
BOOTSTRAPPING (value propagates backward across steps, so a DELAYED reward can be learned).

The unification worth seeing: `δ = r + γV(s′) − V(s)` is the SAME delta rule as Rescorla-Wagner cue competition (the operator's
KEY), the `_Readout` classifier, and dopamine-RPE (`reference_cue_competition_key_discovery`). One rule, applied to value over
time. Honest limit: a LINEAR value cannot hold a relational V* (`project_linear_value_cannot_hold_sokoban`) — this is the
routine/leaf critic; relational tasks need ROLLOUT (deferred), which uses THIS critic at the leaf.
"""

from __future__ import annotations

import os
import random
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent               # noqa: E402
from tbt.encoders import CategoryEncoder  # noqa: E402
from tbt.reward import ValueCritic        # noqa: E402


def test_the_critic_BOOTSTRAPS_value_backward():
    """The core TD property, on the critic in isolation: a chain s0→s1→s2 with reward only at the terminal step. Value must
    propagate BACK — a discounted gradient V(s0) < V(s1) < V(s2), with each ≈ γ× the next — which no immediate signal produces."""
    critic = ValueCritic(lr=0.3, gamma=0.9)
    s = [frozenset([10 * i, 10 * i + 1]) for i in range(3)]      # three distinct sparse states
    for _ in range(300):
        critic.learn(s[0], 0.0, s[1], done=False)
        critic.learn(s[1], 0.0, s[2], done=False)
        critic.learn(s[2], 1.0, None, done=True)                # terminal reward
    v = [critic.value(x) for x in s]
    assert v[2] > v[1] > v[0] > 0.0, f"value must propagate backward, got {v}"
    assert abs(v[2] - 1.0) < 0.05, f"the terminal state's value ≈ the reward, got {v[2]:.3f}"
    assert abs(v[1] - 0.9 * v[2]) < 0.05 and abs(v[0] - 0.9 * v[1]) < 0.05, f"a γ-discounted gradient, got {v}"


def test_the_immediate_RPE_carries_a_LEARNED_BASELINE():
    """For a bandit the critic gives `δ = r − V`, a real prediction error: V learns the EXPECTED reward, so a fully-predicted
    reward yields δ→0 (nothing left to learn) — where the faked `2r−1` stays ±1 forever, unable to represent 'expected'."""
    critic = ValueCritic(lr=0.2)
    s = frozenset([1, 2, 3])
    deltas = [critic.learn(s, 1.0, None, done=True) for _ in range(200)]
    assert abs(critic.value(s) - 1.0) < 0.05, f"V must converge to the reward it predicts, got {critic.value(s):.3f}"
    assert abs(deltas[-1]) < 0.05, f"a fully-predicted reward must drive δ→0, got {deltas[-1]:.3f}"


# ── the decision loop with the critic: a DELAYED-reward corridor ────────────────────────────────────────────────────
N = 4                                                          # a corridor of N states; reward only at the far end
ADVANCE, WRONG = 0, 1
CTX = CategoryEncoder(range(N), w=8, capacity=N)


def _episode(agent: Agent, eps: float) -> bool:
    s = 0
    while True:
        a = agent.decide(CTX.encode(s), 2, explore=eps)
        if a == ADVANCE:
            if s == N - 1:
                agent.reward(1.0, done=True)                   # reached the goal
                return True
            agent.reward(0.0, next_context=CTX.encode(s + 1), done=False)   # advanced — no reward yet, but BOOTSTRAP
            s += 1
        else:
            agent.reward(0.0, done=True)                       # a wrong move ends the episode with nothing
            return False


def _greedy_reaches_goal(agent: Agent) -> bool:
    s = 0
    for _ in range(N):
        if agent.decide(CTX.encode(s), 2, explore=0.0) != ADVANCE:
            return False
        s += 1
    return s == N


def test_the_agent_solves_a_DELAYED_reward_corridor():
    """The decision loop must now learn a sequence whose reward is only at the END — advancing at every state though each
    advance pays nothing. This is exactly what the faked `2r−1` cannot do: it scores every non-terminal advance as −1 (a
    punishment), so it would learn to AVOID advancing. The critic bootstraps the value backward, so advancing toward the goal
    reads as δ>0. Greedy, the trained agent walks the whole corridor."""
    agent = Agent(feat_n=CTX.n, n_content=2, n_state=2, n_cols=256, seed=0)
    for _ in range(4000):
        _episode(agent, eps=0.25)
    assert _greedy_reaches_goal(agent), "with a real TD critic the agent must learn the delayed-reward corridor"
    vals = [agent.critic.value(agent.thalamus.relay(agent._decision_col.observe(CTX.encode(s), learn=False))) for s in range(N)]
    assert vals[0] < vals[-1], f"state values must rise toward the reward (a value gradient), got {[round(v, 2) for v in vals]}"


if __name__ == "__main__":
    ag = Agent(feat_n=CTX.n, n_content=2, n_state=2, n_cols=256, seed=0)
    for _ in range(4000):
        _episode(ag, eps=0.25)
    vals = [round(ag.critic.value(ag.thalamus.relay(ag._decision_col.observe(CTX.encode(s), learn=False))), 3) for s in range(N)]
    print(f"delayed-reward corridor: greedy reaches goal = {_greedy_reaches_goal(ag)}; learned V(s0..s{N-1}) = {vals}")
