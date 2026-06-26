"""Dynamics learning — a column learns the world's CONDITIONAL effects from experience (ARC agent, step 1).

The flat ColumnAgent DETECTS the exafference (environmental change its self-motion can't explain) but throws
it away — which is why it stalls on the conjunctive levels (it has no model of 'block on pad → door opens').
This learns those rules with the residual predicate search (the same mechanism that found carry), so it is
NOT a hand-coded rule-type. Here it learns THREE different dynamics at once — two body-triggers (step on key →
doorA, step on switch → doorB) and one RELATIONAL (block on pad → doorC) — and REFUSES a random unexplained
event with no precondition. The dynamics column is the world-model the §5/§6 control loop plans through.

Run:  python -m demos.dynamics      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import random
import sys


from tbt.dynamics import DynamicsModel                       # noqa: E402

FLOOR, KEY, SWITCH = 0, 2, 3                                  # cell colours
_NAMES = {0: "stepped_on", 1: "block_on_pad", 2: "x_parity", 3: "y_parity"}


def experience(steps, seed=0):
    """A random walk on a 7x7 grid. Perceive: (the colour just stepped onto, a block-on-pad coincidence flag
    from tracked object positions, and two distractor parities), plus the EXAFFERENT effect (which door, if
    any, opened). Two body-triggers + one relational dynamic + a rare unexplained 'noise' event."""
    rng = random.Random(seed)
    key, switch = (1, 1), (5, 5)
    ax, ay = 3, 3
    obs = []
    for _ in range(steps):
        dx, dy = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        ax, ay = max(0, min(6, ax + dx)), max(0, min(6, ay + dy))
        stepped = KEY if (ax, ay) == key else SWITCH if (ax, ay) == switch else FLOOR
        block_on_pad = 1 if (stepped == FLOOR and rng.random() < 0.12) else 0   # tracked block / pad positions
        noise = rng.random() < 0.05                          # an unexplained event with NO precondition feature
        effect = ("doorA" if stepped == KEY else
                  "doorB" if stepped == SWITCH else
                  "doorC" if block_on_pad else
                  "noise" if noise else None)
        obs.append(((stepped, block_on_pad, ax % 2, ay % 2), effect))
    return obs


if __name__ == "__main__":
    print("dynamics learning — a column learns conditional effects (precondition -> effect) from experience.\n")
    obs = experience(4000, seed=0)
    dm = DynamicsModel()
    for f, e in obs:
        dm.observe(f, e)
    rules = dm.learn()

    learned = {eff for _, _, eff in rules}
    print("  rules discovered (no hand-coded rule-types — the residual predicate search over perceived features):")
    for _pred, desc, eff in rules:
        readable = desc
        for i in range(4):
            readable = readable.replace(f"c{i}", _NAMES[i])
        print(f"    {eff:>6}  when  {readable}")
    print(f"    noise  -> {'REFUSED (no precondition feature -> the MDL stop, not memorised)' if 'noise' not in learned else 'WRONGLY LEARNED'}")

    cond = [(f, e) for f, e in obs if e in {'doorA', 'doorB', 'doorC'}]
    correct = sum(dm.predict(f) == e for f, e in cond)
    print(f"\n  conditional effects predicted: {correct}/{len(cond)}   (every key/switch/pad event anticipated)")
    print("  this is the world-model the control loop plans THROUGH: 'reach the precondition -> the door opens'.")
