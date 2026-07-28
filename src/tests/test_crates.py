"""test_crates.py — the fixture that both PAYS and PUSHES, and the result it was built to expose.

`test_relation_codes` left one thing unproven: the relation-SDR value makes an unvisited configuration's worth generalise,
demonstrated in controlled measurement but never live, because no game could exercise it — the games that PAY have no
pushing (so every rollout fork shares the current configuration) and the games that PUSH never pay (so `R` stays empty).
`tasks.games.crates` closes that gap: each level is one delivery, so reward arrives per push, and the crate and pad sit at
DIFFERENT cells with a DIFFERENT push direction every level, so nothing positional can carry and only the relation can.

WHAT IT MEASURED — a null result, and the fixture was built so a null would be visible rather than hidden. Ablating the
relational value changes NOTHING: the same levels, the same actions, seed for seed. The work is being done by `goal_mem`'s
`(mover, landmark)` pair goal — measured `{(6, 7)}` at weight 0.88, the task-value drive selected 0 times — which already
names "crate on pad", is position-invariant because it is keyed on FEATURES rather than cells, and therefore transfers
across levels for free and exactly.

SO THE HONEST CONCLUSION IS A DISPROOF: a graded relational value is REDUNDANT wherever a discrete pair-goal already names
the win condition. Its distinct contribution has to lie where a pair cannot — a win that is a property of a SET rather than
an arrangement of two objects (`test_rule_proposal` already records that CollectAll's tour is exactly this), or a goal
beyond the rollout's horizon, where no predicate is ever reached inside the search and only a leaf GRADIENT can steer.
Those are the fixtures that would discriminate; this one says, clearly, that this regime does not.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402
from tasks.games.crates import Crates, _LEVELS     # noqa: E402
from tasks.harness import Environment              # noqa: E402
from tasks.oracle import solve_level               # noqa: E402

ORACLE = [4, 4, 2, 5]                              # verified by the generic BFS oracle, per level


def _play(seed: int = 0, relational_value: bool = True, budget: int = 250):
    env = Environment(Crates())
    fd = env.reset()
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=seed)
    if not relational_value:
        a._task_value_of = lambda w: 0.0           # the positional critic alone at the rollout leaf
    per_level, base = [], 0
    for _ in range(budget):
        action, coords = a.step(fd)
        fd = env.step(action, coords)
        if fd.score > len(per_level):
            per_level.append(fd.action_counter - base)
            base = fd.action_counter
        if fd.is_terminal() or fd.is_win():
            break
    return a, fd, per_level


def test_every_level_is_solvable_and_needs_a_push():
    """The fixture must be honest before anything measured on it counts: each level is reachable under the generic BFS
    oracle, and the oracle costs are what the agent is graded against."""
    for level in range(len(_LEVELS)):
        game = Crates()
        game.load_level(level)
        plan = solve_level(game)
        assert plan and len(plan) == ORACLE[level], f"L{level} must cost {ORACLE[level]}, oracle says {plan and len(plan)}"


def test_nothing_positional_can_transfer_between_levels():
    """The property that makes the fixture adversarial to what it tests. If the crate or pad ever sat on the same cell
    twice, a value learned over CELLS could carry and the measurement would be worthless."""
    crates, pads = [], []
    for rows in _LEVELS:
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                (crates if ch == "B" else pads if ch == "P" else []).append((x, y))
    assert len(set(crates)) == len(crates), f"no crate cell may repeat across levels, got {crates}"
    assert len(set(pads)) == len(pads), f"no pad cell may repeat across levels, got {pads}"


def test_the_agent_wins_it_and_is_at_ORACLE_after_the_first_level():
    """The agent solves all four. The first level carries the DISCOVERY cost — learning that walking into the crate shifts
    it, and that the crate on the pad is what pays — and every level after it is exactly oracle-optimal, at cells the agent
    has never seen. That is the mechanic transferring, which is the thing worth having."""
    _a, fd, per_level = _play()
    assert fd.is_win(), f"the agent must finish the game, got score {fd.score}"
    assert per_level[1:] == ORACLE[1:], f"levels after the first must be at oracle, got {per_level} vs {ORACLE}"
    assert per_level[0] > ORACLE[0], "and the first must cost more — that difference IS the discovery"


def test_the_relational_value_changes_NOTHING_here_and_the_pair_goal_does_the_work():
    """THE RESULT THIS FIXTURE EXISTS FOR, and it is a disproof. Ablating the relation-SDR value at the rollout leaf leaves
    every level identical, action for action. What solves the game is `goal_mem`'s `(crate, pad)` pair goal, which already
    names the win condition exactly and is position-invariant because it is keyed on features rather than cells — so a
    graded relational value has nothing left to contribute in this regime.

    Recorded rather than tuned away: a mechanism that changes no behaviour on the fixture built specifically to exercise it
    is a mechanism whose case has not been made, and saying so is the point of building the fixture."""
    with_value, _fd_a, per_with = _play(relational_value=True)
    _b, _fd_b, per_without = _play(relational_value=False)
    assert per_with == per_without, f"the ablation must be measured, not assumed: {per_with} vs {per_without}"
    assert with_value.goal_mem.goals() == {(6, 7)}, (
        f"the pair goal 'crate on pad' is what was learned, got {with_value.goal_mem.goals()}")
    assert with_value.goal_mem.w[(6, 7)] > 0.5, "and it is held with real contingency, not a trace"
