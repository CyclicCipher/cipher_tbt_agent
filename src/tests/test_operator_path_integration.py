"""End-to-end test of the TRANSFORM primitive (ARCHITECTURE.md §8, ROADMAP Phase 3a): L6a path integration by the learned
`operator.ModularOperator`, wired into a SPATIAL `Column` and driven from `agent.py`.

The point of the primitive is POSITION-INVARIANT generalization, which a plain sequence memory (the ASSOCIATE primitive)
cannot do — a transition learned at one place is a separate fact from the same transition elsewhere (the place-invariance
failure, ARCHITECTURE §7). So the crown test learns each action's effect in a SMALL region and then dead-reckons correctly
into positions NEVER visited during learning — the operator analogue of the place-invariance win. It also checks the
abelian group property (a sequence dead-reckons without drift) and the identity prior (an unlearned action = staying put).

RULES #3 acceptance: the agent path-integrates its location (dead-reckons), which it could not do before.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent  # noqa: E402

ACTIONS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}      # the world's true (unknown-to-agent) action effects
TRAIN = [(x, y) for x in range(10, 15) for y in range(10, 15)]        # a small 5x5 LEARNING region
NOVEL = [(45, 50), (8, 40), (33, 20), (58, 5), (2, 33)]              # positions OUTSIDE it (the generalization probe)


def _move(p, a):
    dx, dy = ACTIONS[a]
    return (p[0] + dx, p[1] + dy)


def _fresh_agent() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)  # arithmetic dims present but unused here


def _train(agent: Agent) -> None:
    """Learn each action's effect ONLY from moves inside the small training region."""
    for a in ACTIONS:
        for p in TRAIN:
            agent.learn_move(a, p, _move(p, a))


def _novel_single_step(agent: Agent):
    ok = tot = 0
    for p in NOVEL:
        for a in ACTIONS:
            agent.locate(p)                      # anchor at a NOVEL position (never seen in training)
            agent.path_integrate(a)              # dead-reckon one step by the learned operator
            tot += 1
            ok += (agent.where() == _move(p, a))
    return ok, tot


def _sequence(agent: Agent):
    """Cumulative dead-reckoning over a path — no sensory correction — tests the abelian group composition."""
    seq = ["E", "E", "N", "N", "N", "W", "S"]
    p = (40, 40)
    agent.locate(p)
    ok = 0
    for a in seq:
        agent.path_integrate(a)
        p = _move(p, a)
        ok += (agent.where() == p)
    return ok, len(seq)


def test_operator_generalises_to_novel_positions():
    agent = _fresh_agent()
    _train(agent)
    ok, tot = _novel_single_step(agent)
    assert ok == tot, f"path integration wrong at {tot - ok}/{tot} novel (pos, action) — the operator must be position-invariant"


def test_operator_composes_over_a_sequence():
    agent = _fresh_agent()
    _train(agent)
    ok, n = _sequence(agent)
    assert ok == n, f"dead-reckoning drifted ({ok}/{n} steps correct) — the abelian operator must compose over a path"


def test_learned_per_module_shift():
    """White-box: a +1 coordinate step is +1 phase in every module of that axis (module order = x,y,x,y,... over 4 scales)."""
    agent = _fresh_agent()
    _train(agent)
    op = agent._nav_col().operator
    assert op.shift_of("E") == [1, 0, 1, 0, 1, 0, 1, 0], f"E should be +1 on x-modules only, got {op.shift_of('E')}"
    assert op.shift_of("N") == [0, 1, 0, 1, 0, 1, 0, 1], f"N should be +1 on y-modules only, got {op.shift_of('N')}"


def test_unlearned_action_is_identity():
    agent = _fresh_agent()
    agent.locate((20, 20))
    agent.path_integrate("E")                    # never learned → identity prior (predict staying)
    assert agent.where() == (20, 20), "an unlearned action must path-integrate as the identity"


if __name__ == "__main__":
    ag = _fresh_agent()
    _train(ag)
    ok, tot = _novel_single_step(ag)
    sok, sn = _sequence(ag)
    print(f"novel-position single-step accuracy: {ok}/{tot}")
    print(f"sequence dead-reckoning: {sok}/{sn} steps correct")
    print(f"learned shift  E={ag._nav_col().operator.shift_of('E')}  N={ag._nav_col().operator.shift_of('N')}")
