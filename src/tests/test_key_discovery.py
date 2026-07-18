"""End-to-end test of the OPERATOR'S KEY DISCOVERY — telling the TRUE condition from a spurious CORRELATE (ARCHITECTURE.md §9;
ROADMAP Phase 3b). The context-gated override learns that "supported ⇒ stays", but if a spurious NEIGHBOUR object is always
co-present, which one is the condition? The system must DISCOVER it is support, not the neighbour — without being told.

THE MECHANISM: Rescorla-Wagner CUE COMPETITION (the canonical model of conditioning; Rescorla-Wagner IS the delta rule the
`_Readout`/basal-ganglia dopamine-RPE already run). Each state feature is a CUE; each learns its correction to the effect by
PREDICTION ERROR. The BLOCKING effect (Kamin) falls out: once support predicts "stays", a co-present neighbour arrives when the
error is already ~0 and gains ~0 weight. And CONTINGENCY: a cue that sometimes appears WITHOUT the effect (a neighbour with no
support, which falls) is driven to zero. Either way the spurious correlate is rejected — emergent, error-driven, not a symbolic
rule and no `if supported`.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent      # noqa: E402
from tbt.operator import eye     # noqa: E402

BLOCK = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 3}
TABLE = {(0.0, 0.0): 4, (2.0, 0.0): 5, (0.0, 2.0): 6}
NEIGHBOR = {(0.0, 0.0): 7, (2.0, 0.0): 8, (0.0, 2.0): 9}   # a spurious object, causally irrelevant to the fall
OTHER = {(0.0, 0.0): 10, (2.0, 0.0): 11, (0.0, 2.0): 12}   # a fresh object, for the generalisation claim
STEP = "STEP"


def _fresh() -> Agent:
    return Agent(feat_n=16, n_content=2, n_state=2, n_cols=64, seed=0)


def _learn_objects(agent: Agent, passes: int = 6) -> None:
    for _ in range(passes):
        for obj in (BLOCK, TABLE, NEIGHBOR, OTHER):
            agent.start_object()
            for coord, feature in obj.items():
                agent.locate(coord)
                agent.sense_sweep(feature)
            agent.commit()


def _place(agent: Agent, obj: dict, at) -> int:
    """Recognise the object in the SENSORY column, then route (id, pose) up to the compositional column via the thalamus."""
    agent.start_object()
    for coord, feature in obj.items():
        agent.locate((coord[0] + at[0], coord[1] + at[1]))
        agent.sense_sweep(feature)
    h = agent.recognize()[0]
    agent.place_object(h.label, (h.origin, h.rotation))
    return h.label


def _P(x, y):
    return ((float(x), float(y)), eye(2))


def _approx(pos, expected, tol=0.06) -> bool:
    return all(abs(a - b) < tol for a, b in zip(pos, expected))


def _cue_weight(agent: Agent, a_id, b_id):
    """The learned Rescorla-Wagner weight of the cue = the relation from object `a_id` to object `b_id`, read from the current
    scene. ~0 means the cue predicts nothing (blocked / rejected); a large value means it carries the effect."""
    col = agent._scene_col()
    cue = col._quantise(col.relate(col._scene_objects[a_id], col._scene_objects[b_id]))
    return col._cue_weights.get((STEP, cue), (0.0,) * agent._dims)


def _teach_free_kernel(agent: Agent, b_id):
    agent.clear_scene()
    b = _place(agent, BLOCK, (5.0, 5.0))                  # a free block...
    agent.learn_behavior(STEP, b, _P(5, 4))             # ...falls one unit (the one-shot kernel)


def test_a_redundant_cue_is_BLOCKED_kamin():
    """Kamin blocking: teach support ALONE first, then support WITH an always-present neighbour. Because support already
    predicts "stays", the neighbour arrives at zero prediction error and gains ~0 weight — the classic result."""
    agent = _fresh()
    _learn_objects(agent)
    _teach_free_kernel(agent, None)

    for _ in range(8):                                   # support established first
        agent.clear_scene()
        t = _place(agent, TABLE, (0.0, 0.0))
        b = _place(agent, BLOCK, (0.0, 2.0))
        agent.learn_behavior(STEP, b, _P(0, 2))
    for _ in range(8):                                   # now a neighbour is always co-present
        agent.clear_scene()
        t = _place(agent, TABLE, (0.0, 0.0))
        n = _place(agent, NEIGHBOR, (3.0, 2.0))
        b = _place(agent, BLOCK, (0.0, 2.0))
        agent.learn_behavior(STEP, b, _P(0, 2))

    assert abs(_cue_weight(agent, b, t)[1]) > 0.5, "the SUPPORT cue must carry the effect (it cancels the fall)"
    assert abs(_cue_weight(agent, b, n)[1]) < 0.1, "the redundant NEIGHBOUR cue must be BLOCKED (~0 weight)"


def test_the_spurious_cue_is_REJECTED_as_a_condition():
    """The behavioural consequence of the discovery: a block with ONLY the neighbour (no support) FALLS — the neighbour never
    became a condition — while a supported block STAYS."""
    agent = _fresh()
    _learn_objects(agent)
    _teach_free_kernel(agent, None)
    for _ in range(8):
        agent.clear_scene(); _place(agent, TABLE, (0.0, 0.0)); b = _place(agent, BLOCK, (0.0, 2.0))
        agent.learn_behavior(STEP, b, _P(0, 2))
    for _ in range(8):
        agent.clear_scene(); _place(agent, TABLE, (0.0, 0.0)); _place(agent, NEIGHBOR, (3.0, 2.0))
        b = _place(agent, BLOCK, (0.0, 2.0)); agent.learn_behavior(STEP, b, _P(0, 2))

    agent.clear_scene(); _place(agent, NEIGHBOR, (3.0, 2.0)); b = _place(agent, BLOCK, (0.0, 2.0))
    assert _approx(agent.predict_behavior(STEP, b)[0], (0.0, 1.0)), "neighbour WITHOUT support must FALL (spurious rejected)"
    agent.clear_scene(); _place(agent, TABLE, (0.0, 0.0)); b = _place(agent, BLOCK, (0.0, 2.0))
    assert _approx(agent.predict_behavior(STEP, b)[0], (0.0, 2.0)), "support (the real condition) must still hold ⇒ STAYS"
    agent.clear_scene(); _place(agent, TABLE, (40.0, 40.0)); o = _place(agent, OTHER, (40.0, 42.0))
    assert _approx(agent.predict_behavior(STEP, o)[0], (40.0, 42.0)), "and it still generalises to a NEW supported object"


def test_CONTRAST_disambiguates_a_co_occurring_correlate():
    """The more general mechanism (no reliance on Kamin ordering): support and the neighbour are ALWAYS learned together, so
    at first they SPLIT the credit — but the neighbour also appears on its own (a neighbour with no support, which falls), and
    that contingency drives its weight to zero while support rises to carry the whole effect."""
    agent = _fresh()
    _learn_objects(agent)
    _teach_free_kernel(agent, None)
    for _ in range(14):
        agent.clear_scene()                              # support + neighbour, together, STAYS
        t = _place(agent, TABLE, (0.0, 0.0))
        n = _place(agent, NEIGHBOR, (3.0, 2.0))
        b = _place(agent, BLOCK, (0.0, 2.0))
        agent.learn_behavior(STEP, b, _P(0, 2))
        agent.clear_scene()                              # neighbour ALONE (no support), FALLS — the contrast
        _place(agent, NEIGHBOR, (3.0, 2.0))
        b2 = _place(agent, BLOCK, (0.0, 2.0))
        agent.learn_behavior(STEP, b2, _P(0, 1))

    agent.clear_scene(); t = _place(agent, TABLE, (0.0, 0.0)); n = _place(agent, NEIGHBOR, (3.0, 2.0)); b = _place(agent, BLOCK, (0.0, 2.0))
    assert abs(_cue_weight(agent, b, t)[1]) > 0.5, "contrast must credit SUPPORT with the effect"
    assert abs(_cue_weight(agent, b, n)[1]) < 0.15, "contrast must drive the co-occurring NEIGHBOUR toward zero"
    agent.clear_scene(); _place(agent, NEIGHBOR, (3.0, 2.0)); b = _place(agent, BLOCK, (0.0, 2.0))
    assert _approx(agent.predict_behavior(STEP, b)[0], (0.0, 1.0)), "so a neighbour-only block still FALLS"


if __name__ == "__main__":
    ag = _fresh()
    _learn_objects(ag)
    _teach_free_kernel(ag, None)
    for _ in range(8):
        ag.clear_scene(); _place(ag, TABLE, (0.0, 0.0)); b = _place(ag, BLOCK, (0.0, 2.0)); ag.learn_behavior(STEP, b, _P(0, 2))
    for _ in range(8):
        ag.clear_scene(); t = _place(ag, TABLE, (0.0, 0.0)); n = _place(ag, NEIGHBOR, (3.0, 2.0)); b = _place(ag, BLOCK, (0.0, 2.0)); ag.learn_behavior(STEP, b, _P(0, 2))
    print("support taught first, then a spurious neighbour always co-present:")
    print(f"  w(support)  = {tuple(round(c, 3) for c in _cue_weight(ag, b, t))}")
    print(f"  w(neighbour)= {tuple(round(c, 3) for c in _cue_weight(ag, b, n))}   <- BLOCKED")
    ag.clear_scene(); _place(ag, NEIGHBOR, (3.0, 2.0)); b = _place(ag, BLOCK, (0.0, 2.0))
    print(f"  neighbour-only block → {tuple(round(c, 2) for c in ag.predict_behavior(STEP, b)[0])}  (falls — neighbour rejected as a condition)")
