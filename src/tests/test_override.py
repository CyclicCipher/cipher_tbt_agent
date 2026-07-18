"""End-to-end test of the CONTEXT-GATED OVERRIDE — gravity + support, as a MULTI-COLUMN slice (ARCHITECTURE.md §9;
ROADMAP Phase 3b). This is the first COMPOSITIONAL slice: the override is not a gate bolted onto the single spatial column —
it lives in a second (compositional) column whose features-at-locations are recognised objects, fed up from the sensory
column by the thalamus (`reference_tbt_object_behaviors`: behaviours are a separate, object-independent, STATE-CONDITIONED
frame; the relational/compositional context lives at a HIGHER region).

THE MECHANISM, learned not coded (no `if supported` anywhere):
  * The dynamics operator is the FREE KERNEL — everything falls (the aggressive "everything falls" prior).
  * The effect is keyed on `(action, STATE)`, where STATE is the object's relational geometry (`state_of`). A FREE object is
    the null state (falls); a SUPPORTED object is a non-null state that gets its own keyed effect (stays). The state that
    happens to hold when the effect differs simply gets its own key — that IS the discovery.
  * STATE is GEOMETRY-keyed, not identity-keyed, so a behaviour learned in one state TRANSFERS to any object in that state —
    TBP's object-independent behaviour frame. One demonstration of "supported ⇒ stays" ⇒ every supported object stays.
  * Removing the support changes the state back to null ⇒ the free kernel resumes ⇒ it falls again. Assume, then correct
    ([[feedback_prefer_generalize_then_correct]]).
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent      # noqa: E402
from tbt.operator import eye, rotate  # noqa: E402

BLOCK = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 3}
TABLE = {(0.0, 0.0): 4, (2.0, 0.0): 5, (0.0, 2.0): 6}
OTHER = {(0.0, 0.0): 7, (2.0, 0.0): 8, (0.0, 2.0): 9}     # a DIFFERENT object, for the generalisation claim
STEP = "STEP"                                             # the action (time passes / gravity acts)


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _learn_objects(agent: Agent, passes: int = 6) -> None:
    for _ in range(passes):
        for obj in (BLOCK, TABLE, OTHER):
            agent.start_object()
            for coord, feature in obj.items():
                agent.locate(coord)
                agent.sense_sweep(feature)
            agent.commit()


def _see(agent: Agent, obj: dict, at):
    """Recognise the object presented at world `at` (SENSORY column) → (label, pose). This is the input the thalamus routes
    up to the compositional column."""
    agent.start_object()
    for coord, feature in obj.items():
        agent.locate((coord[0] + at[0], coord[1] + at[1]))
        agent.sense_sweep(feature)
    h = agent.recognize()[0]
    return h.label, (h.origin, h.rotation)


def _place(agent: Agent, obj: dict, at) -> int:
    """Recognise in the sensory column, then ROUTE (object-id, pose) up to the compositional column via the thalamus."""
    label, pose = _see(agent, obj, at)
    agent.place_object(label, pose)
    return label


def _P(x, y):
    return ((float(x), float(y)), eye(2))


def _teach_gravity_and_support(agent: Agent):
    """Two demonstrations: a FREE block falls one unit; a block resting on a table (2 below) stays. Returns the labels."""
    agent.clear_scene()
    b = _place(agent, BLOCK, (5.0, 5.0))                  # a FREE block (null state)...
    agent.learn_behavior(STEP, b, _P(5, 4))              # ...falls one unit  → learns (STEP, ∅) = down

    agent.clear_scene()
    t = _place(agent, TABLE, (0.0, 0.0))
    b = _place(agent, BLOCK, (0.0, 2.0))                  # a block ON the table (supported state)...
    agent.learn_behavior(STEP, b, _P(0, 2))             # ...stays  → learns (STEP, {support}) = no change
    return b, t


def test_the_override_is_MULTI_COLUMN():
    """The slice's architectural claim: recognition happens in the SENSORY column, and the scene/behaviour in a SEPARATE
    COMPOSITIONAL column, joined by the thalamus. Not a single-column gate."""
    agent = _fresh()
    _learn_objects(agent)
    agent.clear_scene()
    t = _place(agent, TABLE, (0.0, 0.0))
    b = _place(agent, BLOCK, (0.0, 2.0))
    assert agent._scene is not None and agent._scene is not agent._nav, "the compositional column is a SECOND column"
    assert set(agent._scene_col()._scene_objects) == {t, b}, "objects recognised in the sensory column are routed into the scene"
    assert t != b, "the two objects were recognised as DISTINCT identities by the sensory column"


def test_a_SUPPORTED_object_stays_but_a_FREE_one_falls():
    """The override itself. Same action STEP, opposite effect, decided by the relational STATE — and both learned, neither
    coded."""
    agent = _fresh()
    _learn_objects(agent)
    b, t = _teach_gravity_and_support(agent)

    agent.clear_scene()                                  # a supported block
    _place(agent, TABLE, (0.0, 0.0))
    b = _place(agent, BLOCK, (0.0, 2.0))
    assert agent.predict_behavior(STEP, b)[0] == (0.0, 2.0), "a SUPPORTED object must STAY"

    agent.clear_scene()                                  # a free block
    b = _place(agent, BLOCK, (9.0, 9.0))
    assert agent.predict_behavior(STEP, b)[0] == (9.0, 8.0), "a FREE object must FALL (the free kernel)"


def test_it_GENERALISES_to_an_object_never_demonstrated():
    """TBP's object-independent behaviour frame: the state is geometry-keyed, so "supported ⇒ stays" — taught only on the
    BLOCK — holds for a DIFFERENT object placed in the same support geometry, with no demonstration of its own."""
    agent = _fresh()
    _learn_objects(agent)
    _teach_gravity_and_support(agent)                    # gravity/support taught with BLOCK only

    agent.clear_scene()
    _place(agent, TABLE, (30.0, 30.0))
    o = _place(agent, OTHER, (30.0, 32.0))               # OTHER, never once demonstrated, supported by the table
    assert agent.predict_behavior(STEP, o)[0] == (30.0, 32.0), "a NEW supported object must also STAY (object-independent)"


def test_REMOVING_the_support_restores_the_fall():
    """The correction half. The relation is not a fact once learned — take the support away and the state returns to null,
    so the free kernel resumes and the object falls. Assume, then correct."""
    agent = _fresh()
    _learn_objects(agent)
    b, t = _teach_gravity_and_support(agent)

    agent.clear_scene()
    _place(agent, TABLE, (0.0, 0.0))
    b = _place(agent, BLOCK, (0.0, 2.0))
    assert agent.predict_behavior(STEP, b)[0] == (0.0, 2.0), "supported ⇒ stays (baseline)"

    agent.clear_scene()                                  # same block, table GONE
    b = _place(agent, BLOCK, (0.0, 2.0))
    assert agent.predict_behavior(STEP, b)[0] == (0.0, 1.0), "support removed ⇒ the free kernel resumes ⇒ it FALLS"


if __name__ == "__main__":
    ag = _fresh()
    _learn_objects(ag)
    bl, tb = _teach_gravity_and_support(ag)
    print("taught: a free block falls, a supported block stays.\n")
    ag.clear_scene(); _place(ag, TABLE, (0.0, 0.0)); b = _place(ag, BLOCK, (0.0, 2.0))
    print(f"  supported block   → {ag.predict_behavior(STEP, b)[0]}  (stays)")
    ag.clear_scene(); b = _place(ag, BLOCK, (9.0, 9.0))
    print(f"  free block        → {ag.predict_behavior(STEP, b)[0]}  (falls)")
    ag.clear_scene(); _place(ag, TABLE, (30.0, 30.0)); o = _place(ag, OTHER, (30.0, 32.0))
    print(f"  NEW obj supported → {ag.predict_behavior(STEP, o)[0]}  (stays — never demonstrated)")
    ag.clear_scene(); b = _place(ag, BLOCK, (0.0, 2.0))
    print(f"  support removed   → {ag.predict_behavior(STEP, b)[0]}  (falls again)")
