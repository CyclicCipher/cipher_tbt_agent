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

from tbt.agent import Agent                    # noqa: E402
from tbt.encoders import CategoryEncoder, SDR  # noqa: E402

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


# ── Phase 4: the per-bit SDR read-off GENERALISES across overlapping contexts ─────────────────────────────────────
NB = 30
SHAPES = {"A": frozenset({0, 1, 2, 3}), "B": frozenset({4, 5, 6, 7})}   # the shape determines the correct action
CORRECT = {"A": 0, "B": 1}
TRAIN_COLOURS = {1: frozenset({10, 11}), 2: frozenset({12, 13})}        # seen during training
NOVEL_COLOURS = {90: frozenset({20, 21}), 91: frozenset({22, 23})}      # never seen — share ONLY the shape bits


def _ctx(shape_bits, colour_bits) -> SDR:
    return SDR(NB, sorted(shape_bits | colour_bits))


def test_bg_generalises_across_overlapping_contexts():
    """Phase 4's payoff: the OpAL actor now reads value off the ACTIVE BITS, not a whole-context key, so value learned in one
    context TRANSFERS to another that shares bits. The correct action depends on the SHAPE; train on shape × {colour 1,2};
    then a shape at a NOVEL colour (sharing only the shape bits) still selects the shape's action — the shared shape bits
    carry the value. The old exact-frozenset keying could not: a novel colour is a fresh key with no learned value → a coin
    flip. (The value critic keeps δ bounded so the three-factor weights don't run away.)"""
    import random
    agent = Agent(feat_n=NB, n_content=2, n_state=2, n_cols=256, seed=0)
    rng = random.Random(0)
    for _ in range(6000):
        s, c = rng.choice(list(SHAPES)), rng.choice(list(TRAIN_COLOURS))
        a = agent.decide(_ctx(SHAPES[s], TRAIN_COLOURS[c]), 2, explore=0.25)
        agent.reward(1.0 if a == CORRECT[s] else 0.0)
    ok = tot = 0
    for s in SHAPES:
        for c in NOVEL_COLOURS:                             # greedy, on colours never seen
            ok += (agent.decide(_ctx(SHAPES[s], NOVEL_COLOURS[c]), 2, explore=0.0) == CORRECT[s])
            tot += 1
    assert ok == tot, f"the per-bit actor must generalise to unseen contexts via shared bits — got {ok}/{tot}"


if __name__ == "__main__":
    print(f"contextual-decision accuracy after training (K={K} contexts, A={A} actions): {_run():.0%}")
