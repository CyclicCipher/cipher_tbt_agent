"""test_h0_factorisation.py — H0, THE DECISION EXPERIMENT: does ONE frame over the joint state factorise?

The legacy `HETERARCHY_PLAN` makes this the gate on everything above it: *"H0 gates everything — if one column factorises,
H1–H4 simplify to a structured single column. Run it first."* Feed ONE frame the joint `(position, configuration)`
transitions and find out whether position and task-state separate INSIDE it, before allocating a second column for them.
Either answer is a real finding: separation means the heterarchy collapses to one structured column and no task column is
ever built; failure to separate is what justifies the second one. `Agent.task_frame` is fed `_world_key` — the agent's cell
plus every tracked mover's cell — deliberately, so the live loop runs the experiment rather than a mock of it.

WHAT IS MEASURED, AND WHY NOT THE EIGENFRAME. The plan phrases H0 as "does the SR eigenframe factorise", written when the
frame was assumed to be an eigendecomposition. Ours is not, on the record: an O(n^3) decomposition is prohibitive online and
the eigenpurpose drive built on top of it was dropped as redundant AND costly (`successor.py`). The test is therefore run on
the CODE ITSELF, where the agent actually reads it — which is also the sounder instrument, because on a product graph the
eigenvalues are degenerate, an eigenspace has an arbitrary basis, and "is this eigenvector separable" is then a question
about the basis numpy happened to return rather than about the frame.

AND THE DEEPER REASON THE SPECTRAL ROUTE WOULD NOT RESCUE IT. Granting perfect spectral separability for free, the state
count is still multiplicative in EXPERIENCE: you can only decompose states you have visited, so a decomposition is a
post-hoc description of experience already had and cannot supply experience not had. Transfer to a (position,
configuration) pair never visited requires the product structure to be known IN ADVANCE — which is precisely what a
separate frame per factor is. That is the finding, and it is not an artefact of the representation chosen here.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402
from tbt.successor import SuccessorFrame           # noqa: E402
from tasks.games.lockpath import LockPath          # noqa: E402
from tasks.harness import Environment              # noqa: E402

P, C = 8, 4                                        # 8 positions x 4 configurations


def _product_world(passes: int = 60, seed: int = 0) -> SuccessorFrame:
    """THE BEST CASE FOR FACTORISATION, deliberately: position and configuration move INDEPENDENTLY, so the transition graph
    is a true Cartesian product and the two factors are as separable as a world can make them. A real game couples them (the
    key sits at a place), so whatever fails here fails harder there."""
    f, rng = SuccessorFrame(), np.random.default_rng(seed)
    p, c = 0, 0
    for _ in range(passes * P * C):
        a = int(rng.integers(0, 4))
        np_, nc = p, c
        if a == 0:   np_ = min(P - 1, p + 1)
        elif a == 1: np_ = max(0, p - 1)
        elif a == 2: nc = min(C - 1, c + 1)
        else:        nc = max(0, c - 1)
        f.observe((p, c), a, (np_, nc))
        p, c = np_, nc
    return f


def _cos(a: dict, b: dict) -> float:
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _diff(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in set(a) | set(b)}


def test_the_joint_state_count_multiplies_instead_of_adding():
    """The blow-up, stated plainly. A factored representation needs one entry per position PLUS one per configuration; the
    joint frame needs one per PAIR. At this size that is 32 against 12, and the gap is the whole argument for factoring —
    it is combinatorial, so it is not a constant-factor cost that a bigger table absorbs."""
    f = _product_world()
    assert len(f.states()) == P * C, f"the joint frame holds every pair, got {len(f.states())}"
    assert P * C > 2 * (P + C), "multiplicative, not additive — and the ratio grows with either factor"


def test_the_same_position_shares_no_code_with_itself_under_another_configuration():
    """If the frame factorised, the code for a position would have a POSITION part that survives a configuration change. It
    does not: the same physical position under a distant configuration overlaps LESS than a genuinely DIFFERENT position at
    the same configuration. Nothing in the code says "these two are the same place" — collecting a key moves the agent
    further, as the frame sees it, than walking does."""
    f = _product_world()
    same_place = float(np.mean([f.similarity((p, 0), (p, C - 1)) for p in range(P)]))
    other_place = float(np.mean([f.similarity((p, 0), (p + 1, 0)) for p in range(P - 1)]))
    assert same_place < other_place, (
        f"the same place under another configuration ({same_place:.3f}) must not be FARTHER than a different place "
        f"({other_place:.3f}) — that it is, is the failure")


def test_the_same_movement_is_not_the_same_operation_across_configurations():
    """The mechanism behind the failure, and the reason more data cannot fix it. Factoring buys ONE operator shared across
    the other factor — "step east" being the same change of code whatever you are carrying. Here the code-change for the
    same step under different configurations barely aligns, because a row is supported on successors and the successors of
    `(p, c)` all carry `c`. The supports are near-disjoint by construction, so there is no shared operator to find.

    This is `project_place_invariance_needs_factored_state` at one level up: the factoring has to be IN the state, it cannot
    be recovered from a memory indexed by whole states."""
    f = _product_world()
    steps = {}
    for c in range(C):
        for p in range(P - 1):
            steps.setdefault(p, []).append(_diff(f.code((p + 1, c)), f.code((p, c))))
    aligns = [_cos(v[i], v[j]) for v in steps.values() for i in range(len(v)) for j in range(i + 1, len(v))]
    assert float(np.mean(aligns)) < 0.5, (
        f"one shared operator would align at ~1.0; got {float(np.mean(aligns)):.3f}")


def test_a_route_learned_at_one_configuration_does_not_transfer_to_another():
    """THE CONSEQUENCE THAT DECIDES THE BUILD. Learn the corridor exhaustively at one configuration, take a single step at a
    second, and ask for the route: the value is exactly zero. The positions are the same positions and the walk is the same
    walk, but none of it is reachable from the new configuration's row, so the agent must re-walk a world it has already
    mastered every time the task state changes."""
    f, rng = SuccessorFrame(), np.random.default_rng(0)
    for _ in range(6000):
        p, a = int(rng.integers(0, P)), int(rng.integers(0, 2))
        f.observe((p, 0), a, (min(P - 1, p + 1) if a == 0 else max(0, p - 1), 0))
    f.observe((0, 1), 0, (1, 1))                                   # one step ever taken at configuration 1
    rewards = {(P - 1, 0): 1.0, (P - 1, 1): 1.0}                   # the goal pays under EITHER configuration
    assert f.value((0, 0), rewards) > 0.05, "the explored configuration must know its route"
    assert f.value((0, 1), rewards) == 0.0, "and none of it survives a change of configuration"


def test_two_frames_over_the_same_experience_do_transfer():
    """The counterfactual arm, which is what makes the negative result constructive rather than merely a complaint: given the
    SAME experience, a frame indexed by position ALONE has the route and is blind to configuration, so a configuration it has
    never seen costs it nothing.

    Recorded with its honest limit — the split here is made BY HAND, so this measures the CEILING that factoring buys, not a
    mechanism that earns it. Who decides the split is exactly the open problem the second column exists to answer, and it is
    what H1/H2 have to deliver rather than assume."""
    pos, rng = SuccessorFrame(), np.random.default_rng(0)
    for _ in range(6000):
        p, a = int(rng.integers(0, P)), int(rng.integers(0, 2))
        pos.observe(p, a, min(P - 1, p + 1) if a == 0 else max(0, p - 1))
    assert pos.value(0, {P - 1: 1.0}) > 0.05, "indexed by position alone, the route holds under ANY configuration"


def test_the_live_agent_shows_the_same_blow_up_on_lockpath():
    """H0 ON THE LIVE LOOP, not a mock of it: the agent plays LockPath and `task_frame` accumulates its own `_world_key`
    transitions. The same shape appears — states well in excess of cells, most cells re-entered under more than one
    configuration and stored afresh each time, and the same cell under two configurations sharing NO code at all."""
    env = Environment(LockPath())
    fd = env.reset()
    agent = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    tf = agent.task_frame

    def _by_cell():
        out: dict = {}
        for cell, cfg in tf.states():
            out.setdefault(cell, set()).add(cfg)
        return out

    for _ in range(400):                             # stops as soon as the measurement is available — the blow-up shows up
        action, coords = agent.step(fd)              # early, and playing on would only cost suite time to restate it
        fd = env.step(action, coords)
        by_cell = _by_cell()
        if fd.is_terminal() or fd.is_win() or sum(len(c) > 1 for c in by_cell.values()) >= 5:
            break

    states, by_cell = tf.states(), _by_cell()
    repeated = [c for c, cfgs in by_cell.items() if len(cfgs) > 1]

    assert len(states) > len(by_cell), (
        f"the joint frame must hold more states than cells ({len(states)} vs {len(by_cell)})")
    assert repeated, "the agent must have re-entered some cell under a second configuration"
    cell = repeated[0]
    a, b = sorted(by_cell[cell], key=str)[:2]
    assert tf.similarity((cell, a), (cell, b)) < 0.2, (
        "the same cell under two configurations must be shown to share (almost) no code — it is one place, and the frame "
        "has no way to say so")
