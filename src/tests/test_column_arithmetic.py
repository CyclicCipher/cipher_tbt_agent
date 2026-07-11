"""End-to-end test of the FIRST COLUMN SLICE (RULES.md #3 acceptance: the agent plays more than the stub did).

The two-column `Agent` (a SENSORY + a TASK column) runs the sensorimotor increment and must generalize the carry to the
HUNDREDS place — a place NEVER trained — through the real column composition. This reproduces
project_place_invariance_needs_factored_state's win (option2.py) as wired architecture: train on 2-digit increments,
test the successor of 3-digit numbers whose hundreds VALUES (3-9) were never produced.
"""

from __future__ import annotations

import os
import random
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent            # noqa: E402
from tbt.encoders import CategoryEncoder  # noqa: E402

P = 3
DIG = CategoryEncoder(range(10), w=8, capacity=10)


def _digits(n):
    return [(n // 10 ** p) % 10 for p in range(P)]


def _feat(d):
    return DIG.encode(d)


def _run():
    agent = Agent(feat_n=DIG.n, n_content=10, n_state=2, n_cols=256, seed=0)
    # training fixations from 2-digit increments 0..97 (units + tens only). state_out = the OBSERVABLE carry (did the
    # higher place change), NOT a hand-coded rule — the model must learn WHEN it happens.
    fixations = []
    for n in range(98):
        d0, d1 = n % 10, (n // 10) % 10
        fixations.append((d0, 1, (d0 + 1) % 10, int(d0 + 1 >= 10)))         # units: state_in = 1 (increment)
        c = int(d0 + 1 >= 10)
        fixations.append((d1, c, (d1 + c) % 10, int(d1 + c >= 10)))         # tens: state_in from units
    rng = random.Random(0)
    for _ in range(12):
        rng.shuffle(fixations)
        for d, cin, nxt, cout in fixations:
            agent.learn_fixation(_feat(d), cin, nxt, cout)

    pp = [0] * P; whole = 0; test = list(range(100, 998))                   # hundreds values 3-9 NEVER trained
    for n in test:
        preds = agent.scan([_feat(d) for d in _digits(n)], state0=1)        # autonomous rollout; state0 = increment
        tgt = _digits(n + 1)
        for p in range(P):
            pp[p] += (preds[p] == tgt[p])
        whole += (preds == tgt)
    t = len(test)
    return [x / t for x in pp], whole / t


def test_place_invariance_through_the_column():
    (units, tens, hundreds), whole = _run()
    assert hundreds >= 0.95, f"hundreds (novel place) {hundreds:.0%} — the factored-state win should carry it"
    assert whole >= 0.95, f"whole {whole:.0%}"
    assert units >= 0.95 and tens >= 0.90, f"units {units:.0%} tens {tens:.0%}"


if __name__ == "__main__":
    (u, t, h), w = _run()
    print(f"two-column sensorimotor increment, test 100-997 (hundreds 3-9 never trained):")
    print(f"  whole {w:.0%}   per-place[units,tens,hundreds] {['%.0f%%' % (x*100) for x in (u, t, h)]}")
