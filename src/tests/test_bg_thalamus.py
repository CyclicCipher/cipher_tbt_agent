"""End-to-end test of the DECISION LOOP (ARCHITECTURE.md §3): perceive → relay (thalamus) → SELECT by value (basal
ganglia, dopamine-RPE) → gate (thalamus) → act → reward → learn.

A contextual decision: K distinct context stimuli, each with ONE rewarding action; the agent must learn the context→action
map from reward alone. Exercises the basal ganglia (OpAL Go/NoGo + RPE selection) and the thalamus (percept relay +
selection gate), wired from `agent.py` — the RULES #3 acceptance for the two new subsystems (the agent selects, which it
could not do before).
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

K, A = 5, 3                                            # 5 contexts, 3 actions; the rewarding action per context:
CTX = CategoryEncoder(range(K), w=8, capacity=K)       # distinct context stimuli (the cortex perceives these)


def _correct(c):
    return c % A


def _run():
    agent = Agent(feat_n=CTX.n, n_content=A, n_state=2, n_cols=256, seed=0)   # (arithmetic dims present but unused here)
    rng = random.Random(0)
    for _ in range(4000):                              # train: random context, ε-explore, reward the correct action
        c = rng.randrange(K)
        a = agent.decide(CTX.encode(c), A, explore=0.2)
        agent.reward(1.0 if a == _correct(c) else 0.0)
    ok = 0                                             # evaluate GREEDILY — read the learned policy
    for c in range(K):
        ok += (agent.decide(CTX.encode(c), A, explore=0.0) == _correct(c))
    return ok / K


def test_bg_learns_contextual_choice():
    acc = _run()
    assert acc >= 0.99, f"decision accuracy {acc:.0%} — the basal ganglia should learn the context→action map"


if __name__ == "__main__":
    print(f"contextual-decision accuracy after training (K={K} contexts, A={A} actions): {_run():.0%}")
