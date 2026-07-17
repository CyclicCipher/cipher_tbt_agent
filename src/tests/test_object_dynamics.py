"""End-to-end test of OBJECT DYNAMICS — the operator over OBJECT poses (ARCHITECTURE.md §9; ROADMAP Phase 3b step 2).

THE CLAIM: show the agent ONE demonstration of what an action does to an object, and it predicts that action's effect on
ANY object, at positions and orientations never demonstrated. This is the base for physical law — "any unsupported object
falls" is this mechanism with a condition instead of an action.

WHY ONE DEMO IS ENOUGH, and why that is not a trick: it is §7's place-invariance lesson, one level up. The operator stores
the delta in the frame that holds it INVARIANT, so it applies everywhere BY CONSTRUCTION — the FRAME generalizes, not the
data. You do not learn gravity by watching a thousand falls. Exactly as an action's effect learned in a 5×5 region
dead-reckons correctly at (45,50) (`test_operator_path_integration`), an object's dynamics learned at one pose holds at
every pose.

THE POSES ARE THE MODEL'S OWN OUTPUT, not hand-fed coordinates: every pose here comes from `recognize()` SOLVING where the
object is (R4/R6). So this is the perception stack feeding the dynamics stack end-to-end — which is the whole point of
having built the pose solve first.

INTRINSIC vs EXTRINSIC is the one real design fact here (`operator.MotionOperator`): a BODY's motion is intrinsic ("FORWARD"
means forward from where I face), an OBJECT's is EXTRINSIC (a shoved block goes where it was shoved, whatever way the block
happens to be turned). Getting that backwards is not subtle — `test_a_ROTATED_object_moves_the_SAME_way` is the check.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                                  # noqa: E402
from tbt.operator import eye, from_angle, rotate, to_angle   # noqa: E402

# Two asymmetric objects, with distinct features so identity is never in question here (that is R4/R5/R8's business).
BLOCK = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 3}
BALL = {(0.0, 0.0): 4, (2.0, 0.0): 5, (0.0, 2.0): 6}
PUSH = (3.0, 0.0)        # the world's (unknown-to-the-agent) effect of "PUSH": +3 in x, whatever the object is or faces
FALL = (0.0, -2.0)       # and of "STEP": down. The agent is told neither — it sees only poses.


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _learn_objects(agent: Agent, *objects: dict, passes: int = 6) -> None:
    for _ in range(passes):
        for obj in objects:
            agent.start_object()
            for coord, feature in obj.items():
                agent.locate(coord)
                agent.sense_sweep(feature)
            agent.commit()


def _see(agent: Agent, obj: dict, at, rotation=None):
    """PRESENT the object at a world pose and let the agent SOLVE where it is. Returns its (origin, R) — the model's own
    output, which is what the dynamics learns from."""
    rotation = rotation or eye(2)
    agent.start_object()
    for coord, feature in obj.items():
        p = rotate(rotation, coord)
        agent.locate((p[0] + at[0], p[1] + at[1]))
        agent.sense_sweep(feature)
    pop = agent.recognize()
    assert pop, f"the object must be recognised before its motion can be learned; got {pop}"
    return pop[0].origin, pop[0].rotation


def _demo(agent: Agent, obj: dict, action, effect, at, rotation=None):
    """ONE demonstration: see the object, see it again after the world moved it, learn the transition."""
    before = _see(agent, obj, at, rotation)
    after = _see(agent, obj, (at[0] + effect[0], at[1] + effect[1]), rotation)
    agent.learn_object_move(action, before, after)


def _close(a, b, tol=1e-6) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_one_demonstration_predicts_a_NEVER_DEMONSTRATED_position():
    """The crown, and the §7 lesson on objects: demonstrate PUSH once, at one place, and its effect is known everywhere —
    because the operator is defined over a frame, not fitted to data."""
    agent = _fresh()
    _learn_objects(agent, BLOCK)
    _demo(agent, BLOCK, "PUSH", PUSH, at=(10.0, 10.0))                 # ONE demonstration, at ONE place
    for at in ((40.0, 25.0), (3.0, 51.0), (0.0, 0.0)):                 # places never demonstrated
        pose = _see(agent, BLOCK, at)
        p, _R = agent.predict_object_move(pose, "PUSH")
        assert _close(p, (at[0] + PUSH[0], at[1] + PUSH[1])), f"at {at}: predicted {p}"


def test_a_ROTATED_object_moves_the_SAME_way():
    """An object's motion is EXTRINSIC: a shoved block goes where it was shoved, however the block is turned. Demonstrate
    PUSH on an UNROTATED block; predict for one rotated 90° and 217°. Had the operator stored the delta intrinsically (the
    body convention, right for "FORWARD"), the 90° block would be predicted to fly off at 90° to the shove — so this is the
    test that the two ways the group acts are not interchangeable."""
    agent = _fresh()
    _learn_objects(agent, BLOCK)
    _demo(agent, BLOCK, "PUSH", PUSH, at=(10.0, 10.0))                 # demonstrated only on an UNROTATED block
    for deg in (90.0, 217.0):
        R = from_angle(deg)
        pose = _see(agent, BLOCK, (20.0, 20.0), R)
        p, R2 = agent.predict_object_move(pose, "PUSH")
        assert _close(p, (20.0 + PUSH[0], 20.0 + PUSH[1])), f"a block turned {deg}° must still be shoved east; got {p}"
        assert abs(to_angle(R2) - deg) < 1e-6, f"a shove must not spin it: {to_angle(R2)}° vs {deg}°"


def test_it_generalises_to_an_object_it_was_never_demonstrated_on():
    """"ANY object" is FREE, because nothing is keyed on which one — the operator is the regular free kernel. Demonstrate
    STEP on the BLOCK; the BALL, never once seen to move, is predicted to do the same. That is the shape of "any unsupported
    object falls" — and it is also a HYPOTHESIS the world may refute (feathers), which is the operator's open KEY problem."""
    agent = _fresh()
    _learn_objects(agent, BLOCK, BALL)
    _demo(agent, BLOCK, "STEP", FALL, at=(30.0, 30.0))                 # demonstrated on the BLOCK only
    pose = _see(agent, BALL, (12.0, 44.0))                             # a DIFFERENT object, somewhere else
    p, _R = agent.predict_object_move(pose, "STEP")
    assert _close(p, (12.0 + FALL[0], 44.0 + FALL[1])), f"the BALL must fall like the BLOCK; got {p}"


def test_two_actions_stay_distinct_and_an_unknown_one_predicts_no_change():
    """The operator is keyed by ACTION, so distinct actions stay distinct — and an action never demonstrated predicts no
    change, the honest prior (and a large prediction error the moment it is wrong, which is what would drive learning it)."""
    agent = _fresh()
    _learn_objects(agent, BLOCK)
    _demo(agent, BLOCK, "PUSH", PUSH, at=(10.0, 10.0))
    _demo(agent, BLOCK, "STEP", FALL, at=(10.0, 10.0))
    pose = _see(agent, BLOCK, (25.0, 25.0))
    assert _close(agent.predict_object_move(pose, "PUSH")[0], (28.0, 25.0))
    assert _close(agent.predict_object_move(pose, "STEP")[0], (25.0, 23.0))
    assert _close(agent.predict_object_move(pose, "NEVER_SEEN")[0], (25.0, 25.0)), "an unlearned action must be the identity"


# ── COMMON FATE: what moves together is one thing (ROADMAP 3b; ARCHITECTURE §9) ────────────────────────────────────
# A scene of two things that are ALWAYS SEEN TOGETHER. With no model, nothing can say they are two — until one moves.
SCENE_LEFT = {(0.0, 0.0): 1, (2.0, 0.0): 2}            # the part that will move
SCENE_RIGHT = {(8.0, 0.0): 4, (10.0, 0.0): 5}          # the part that stays


def _sweep_scene(agent: Agent, *parts: dict, shift=(0.0, 0.0)):
    """Sweep a whole SCENE as ONE episode — one onset, no boundary cue, the parts' cells interleaved as the sensor finds
    them. `shift` moves the FIRST part only (the world's doing; the agent is told nothing)."""
    agent.start_object()
    for i, part in enumerate(parts):
        for coord, feature in part.items():
            off = shift if i == 0 else (0.0, 0.0)
            agent.locate((coord[0] + off[0], coord[1] + off[1]))
            agent.sense_sweep(feature)
    return agent.commit()


def _groups(agent: Agent, *parts: dict, shift=(0.0, 0.0)):
    """Sweep the scene, then ask the column how it groups the look by MOTION — before any of it is committed. `look_again`
    declares "the same scene, later", which only the caller can know: the previous EPISODE is not the previous LOOK."""
    agent._nav_col().look_again()
    agent.start_object()
    for i, part in enumerate(parts):
        for coord, feature in part.items():
            off = shift if i == 0 else (0.0, 0.0)
            agent.locate((coord[0] + off[0], coord[1] + off[1]))
            agent.sense_sweep(feature)
    return agent._nav_col()._common_fate_groups()


def test_COMMON_FATE_groups_a_scene_NO_MODEL_could():
    """THE COLD-START CUE. Everywhere else the boundary is a prediction MISMATCH against a model, which is why a wholly novel
    scene can only mint one blob — "the object is a RECOGNITION construct", and with no model there is no object. Motion needs
    no model. Two things always seen together are grouped as one (correctly — nothing yet says otherwise); the moment one of
    them MOVES they are grouped as TWO, by motion alone.

    The scene is swept as ONE episode with ONE onset: the agent is never told there are two things, which cells belong to
    which, or that anything moved."""
    agent = _fresh()
    assert _groups(agent, SCENE_LEFT, SCENE_RIGHT) == [[0, 1, 2, 3]], "nothing has moved yet — one group is correct"
    assert _groups(agent, SCENE_LEFT, SCENE_RIGHT) == [[0, 1, 2, 3]], "still nothing moving"
    assert _groups(agent, SCENE_LEFT, SCENE_RIGHT, shift=(5.0, 0.0)) == [[0, 1], [2, 3]], (
        "the LEFT part moved and the right did not, so the scene is TWO things — grouped by motion, with no model at all")


def test_a_scene_that_moves_TOGETHER_is_ONE_group():
    """The other half, and what stops the cue from shattering everything: parts that move as ONE are ONE. Shift the WHOLE
    scene — every fixation shares a displacement — and it must stay one group, which is exactly right for a rigid object
    seen from a new place."""
    agent = _fresh()
    _groups(agent, SCENE_LEFT, SCENE_RIGHT)
    assert _groups(agent, SCENE_LEFT, SCENE_RIGHT, shift=(0.0, 0.0)) == [[0, 1, 2, 3]], "static scene"
    agent._nav_col().look_again()
    agent.start_object()                                               # the WHOLE scene, moved rigidly
    for part in (SCENE_LEFT, SCENE_RIGHT):
        for coord, feature in part.items():
            agent.locate((coord[0] + 5.0, coord[1]))
            agent.sense_sweep(feature)
    assert agent._nav_col()._common_fate_groups() == [[0, 1, 2, 3]], (
        "a scene that moves as ONE must stay ONE group — common fate groups by SHARED motion, it does not just detect change")


if __name__ == "__main__":
    ag = _fresh()
    _learn_objects(ag, BLOCK, BALL)
    _demo(ag, BLOCK, "PUSH", PUSH, at=(10.0, 10.0))
    print("ONE demonstration of PUSH, on the BLOCK, at (10,10). Now:")
    for at in ((40.0, 25.0), (3.0, 51.0)):
        p, _ = ag.predict_object_move(_see(ag, BLOCK, at), "PUSH")
        print(f"  BLOCK at {at} (never demonstrated) → predicted {tuple(round(c, 2) for c in p)}")
    for deg in (90.0, 217.0):
        p, R = ag.predict_object_move(_see(ag, BLOCK, (20.0, 20.0), from_angle(deg)), "PUSH")
        print(f"  BLOCK turned {deg:5.1f}° at (20,20)      → predicted {tuple(round(c, 2) for c in p)}, "
              f"still facing {to_angle(R):.0f}°")
    p, _ = ag.predict_object_move(_see(ag, BALL, (12.0, 44.0)), "PUSH")
    print(f"  the BALL, never seen to move     → predicted {tuple(round(c, 2) for c in p)}  (the free kernel: ANY object)")
