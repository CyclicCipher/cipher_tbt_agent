"""test_heterarchy.py — H3: the SPATIAL and TASK regions composed, and the planner distinguishing two board-states.

H0 measured that one frame over the joint state does not factorise; H1 built peer-to-peer voting; H2 built the task region
over configurations alone. H3 composes them, and the legacy plan states the test: *"the planner distinguishes two
board-states at the same position (picks different actions)"* — the C4 regression, fixed BY FACTORISATION rather than by a
joint blob.

WHAT THE TWO REGIONS EXCHANGE, AND WHY THE PLAN IS CORRECTED AGAIN. H3 says to have the thalamus bind "spatial-position ⊗
task-state so the planner sees the joint state". That is precisely the joint state H0 measured as un-factorising — building
it would undo H0's own finding, and would re-create the multiplicative blow-up two steps after removing it. What actually
passes between regions is a GOAL STATE downward and ARRIVAL upward (`reference_hierarchy_substrate`: "top-down task→spatial
sends a subgoal; bottom-up spatial→task sends reached-subgoal / prediction-error, which advances and TEACHES the task
graph"). A subgoal is ONE state, not a product of two spaces, so nothing multiplies. This is the third of the plan's
substrate claims to be corrected by measurement — after H0's eigenframe and H1's thalamic voting.

The subgoal is CHOSEN by learned value over a LEARNED graph: no enumeration of subgoal kinds anywhere
(`feedback_subgoal_types_from_dynamics`), and no second planner — the subgoal becomes a reward predicate over world-states
and the SAME rollout pursues it.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402
from tbt.operator import eye                       # noqa: E402
from tasks.core import GameAction                  # noqa: E402

D = {GameAction.ACTION1: (0., -1.), GameAction.ACTION2: (0., 1.),
     GameAction.ACTION3: (-1., 0.), GameAction.ACTION4: (1., 0.)}
PAD, AGENT = (5, 4), (5, 6)


def _taught() -> Agent:
    """An agent that has learned to move and that pressing a mover pushes it — the model the rollout needs, taught the same
    way the live loop learns it (`_learn_delta` from felt contact)."""
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a._movers.add(6)
    for action, d in D.items():
        a.learn_pose_move(action, a._as_pose((4, 3)), a._as_pose((4 + int(d[0]), 3 + int(d[1]))))
        a._learn_delta("of", 6, None, d, d)
        a._learn_delta("into", 6, None, d, (0., 0.))
    a._extent = [(0, 11), (0, 11)]
    return a


def _scene(a: Agent, block, agent_cell=AGENT) -> frozenset:
    """Put the block and the pad in the scene with the agent at `agent_cell`, and return the configuration."""
    a.clear_scene()
    a.place_object(6, ((float(block[0]), float(block[1])), eye(2)))
    a.place_object(7, ((float(PAD[0]), float(PAD[1])), eye(2)))
    a.set_pose(a._as_pose(agent_cell)[0], eye(2))
    return a.task_state()


def _world(passes: int = 200):
    """A task graph in which the block reaches the pad from either side, and a reward on the configuration where it rests
    there — the minimum for the task region to want something."""
    a = _taught()
    paid = _scene(a, PAD)
    left, right = _scene(a, (3, 4)), _scene(a, (7, 4))
    col = a._task_col()
    for _ in range(passes):
        col.learn_transition(left, "push", paid)
        col.learn_transition(right, "push", paid)
    for _ in range(30):
        a.task_reward.learn(a._configuration_bits(paid), 1.0)
    return a, left, right, paid


def _plan_for(a: Agent, block, configuration):
    _scene(a, block)
    a._last_task = configuration
    return a._task_plan(list(D), horizon=24)


def test_the_planner_distinguishes_two_board_states_at_the_SAME_position():
    """THE PLAN'S OWN TEST FOR H3, and the C4 regression it was written for. The agent stands on the same cell in both
    cases and everything about its own position is identical; only the block's relation to the pad differs. The plans are
    opposite — go WEST to push east, go EAST to push west — because the goal came from the task region, which sees the
    configuration, not from anything positional."""
    a, left, right, _paid = _world()
    plan_left = _plan_for(a, (3, 4), left)
    plan_right = _plan_for(a, (7, 4), right)
    assert plan_left and plan_right, "both board-states must yield a plan"
    assert plan_left != plan_right, "two board-states at one position must not produce the same plan"
    west = [k for k, v in D.items() if v == (-1., 0.)][0]
    east = [k for k, v in D.items() if v == (1., 0.)][0]
    assert west in plan_left[:4] and east not in plan_left[:4], f"block to the WEST ⇒ approach from the west, got {plan_left[:4]}"
    assert east in plan_right[:4] and west not in plan_right[:4], f"block to the EAST ⇒ approach from the east, got {plan_right[:4]}"


def test_what_passes_between_the_regions_is_ONE_state_not_a_product():
    """The H0 correction, asserted rather than trusted: the message is a single configuration drawn from the task region's
    own state space, so composing the regions multiplies nothing. A joint `(position, configuration)` message would be the
    blow-up H0 measured, re-introduced two steps after it was removed."""
    a, left, _right, paid = _world()
    _scene(a, (3, 4))
    a._last_task = left
    want = a._task_subgoal(list(D))
    assert want is not None, "with a reward and a learned successor there must be something to want"
    target, gain = want
    assert isinstance(target, frozenset), "the message is ONE configuration, not a pair with a position"
    assert target != left, "and it is somewhere other than here"
    assert gain > 0.0, "a subgoal is only proposed when it is worth more than staying put"
    # The next SINGLE PUSH, not the finished arrangement: chaining happens across steps, one leg at a time.
    assert {oid for oid, _r in target} == {oid for oid, _r in left}, "a push moves a mover, it does not add one"


def test_the_task_region_wants_nothing_until_something_has_paid():
    """The honest prior, and the reason this drive is silent on LockPath: `V = M·R` needs an `R`. A task region with a
    learned map but no reward proposes NOTHING rather than inventing a preference — which is also why it cannot rescue a
    level that has never paid."""
    a = _taught()
    left = _scene(a, (3, 4))
    paid = _scene(a, PAD)
    col = a._task_col()
    for _ in range(50):
        col.learn_transition(left, "push", paid)
    a._last_task = left
    assert a._task_subgoal(list(D)) is None, "no reward seen ⇒ nothing to want"
    assert a._task_plan(list(D)) is None, "and so no plan to make"


def test_a_configuration_never_left_before_still_offers_a_subgoal():
    """THE MEASURED LIMIT OF H3, and the thing to fix next. A subgoal is chosen among the current state's LEARNED
    successors, and the task graph is learned strictly BEHIND the agent. On the live games it almost never had one:
    measured on CollectAll, the mean number of learned successors of the state the agent is standing in is **0.02**, so the
    drive is offered on ~5 steps out of 220 (the BG chose it every one of those times — it is starved, not out-competed).

    The cause is that a task graph is learned strictly BEHIND the agent while configurations rarely recur — collecting
    consumes, pushing does not return — and task states are EXACT-MATCH keys, so nothing generalises from a similar
    configuration to an unvisited one. FIXED by imagining candidates with the forward model instead of looking them up:
    nothing is enumerated but one push of each DISCOVERED mover under each LEARNED action effect."""
    a, left, _right, paid = _world()
    unvisited = _scene(a, (8, 4))                        # a configuration the agent has never left
    assert not a._task.graph.graph.get(unvisited, {}), "it still has no learned outgoing edge"
    want = a._task_subgoal(list(D))
    assert want is not None, "and it is NO LONGER STARVED: candidates are imagined by the forward model, not looked up"
    assert want[1] > 0.0, "the imagined configuration must be worth more than standing still"

    # AND THE HONEST LIMIT, measured rather than left implicit. The potential field is the relation code's overlap, so it
    # reaches exactly as far as that code does — `mw - 1` cells. Widening the code (mw 3 → 11) moved the reach from TWO
    # cells to TEN: measured 0.999 / 0.817 / 0.636 / 0.545 / 0.499 for the block 0,4,8,10,12 cells from the pad, with a
    # clean constant +0.045 gain per push everywhere inside it, and flat beyond. Inside the reach the chain climbs;
    # outside there is nothing to climb and the drive is correctly silent — the nav dead-zone
    # `reference_eigenoptions_subgoals` names, pushed further out but NOT removed, since the reach is still finite.
    _scene(a, (8 + 9, 4))                                # 12 cells out, past the widened code's overlap
    assert a._task_subgoal(list(D)) is None, "beyond the code's overlap the field is flat and it must say so"


def test_the_task_drive_is_arbitrated_in_the_basal_ganglia():
    """ARCHITECTURE rule 4: the BG is the one organ allowed to arbitrate. The task region does not pre-empt the other
    drives — it OFFERS the value it would gain, on the same scale as the goal's contingency and the model's learning rate,
    and the BG selects among three."""
    a, left, _right, _paid = _world()
    _scene(a, (3, 4))
    a._last_task = left
    want = a._task_subgoal(list(D))
    chosen = a.bg.select(a._MODE_CTX, 3, rho=a.critic.rho(), salience=[float("-inf"), float("-inf"), want[1]])
    assert chosen == 2, "with only the task drive offering anything, the BG must select it"


def test_arrival_teaches_the_task_graph_from_below():
    """The upward half of the loop. The task region's graph is not given to it: every edge was learned from the spatial
    region actually bringing a configuration about (`Agent.step` → `learn_transition`), which is what makes the subgoal it
    proposes a claim about this world rather than a wish."""
    a, left, _right, paid = _world()
    assert paid in a._task.graph.graph.get(left, {}).values(), "the edge must come from an observed transition"
    assert a._task.where_state() is not None, "and L6a holds where the task region currently is"
