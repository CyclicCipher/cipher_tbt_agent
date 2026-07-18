"""test_partial_recognition.py — the STRICT-SUBSET / occlusion falsifier at the COLUMN level (DESIGN §3½;
`reference_recognition_under_occlusion`).

THE INVARIANT: **mint on REFUTATION, never on INCOMPLETENESS.** An object seen only in part — an apple from one angle, the
visible half of a wall — must be recognised as the KNOWN WHOLE, not minted as a new "partial" object. But a view that
CONTRADICTS the whole (a feature where the model forbids one) MUST mint. Missing evidence and contradicting evidence are
opposite things, and this is the test that pins the boundary from both sides on machinery we already have (`recognize`/
`commit` + ART vigilance: a partial view matches the whole because vigilance is `|I∧w|/|I|`, normalised by what was OBSERVED,
so the model having MORE features than were sampled does not lower the match).

FULL is a 4-feature asymmetric object; SUBSET drops one feature-at-location (strict subset, unobserved ≠ absent); CONTRA keeps
the first two but puts a WRONG feature at the third location (a genuine contradiction of the whole).
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                          # noqa: E402
from tbt.operator import from_angle, rotate, to_angle  # noqa: E402

FULL = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 3, (2.0, 2.0): 4}   # 4 distinct features → asymmetric, unambiguous pose
SUBSET = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 3}                # a STRICT subset — the 4th feature-at-location unobserved
CONTRA = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 9}               # 1,2 as FULL, but 9 CONTRADICTS FULL's 3 at (0,2)


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _learn(agent: Agent, *objects: dict, passes: int = 6) -> None:
    for _ in range(passes):
        for obj in objects:
            agent.start_object()
            for coord, feature in obj.items():
                agent.locate(coord)
                agent.sense_sweep(feature)
            agent.commit()


def _present(agent: Agent, obj: dict, omega: float = 0.0, shift=(0.0, 0.0)):
    agent.start_object()
    for coord, feature in obj.items():
        p = rotate(from_angle(omega), coord)
        agent.locate((p[0] + shift[0], p[1] + shift[1]))
        agent.sense_sweep(feature)
    return agent.recognize()


def _study(agent: Agent, obj: dict) -> int:
    agent.start_object()
    for coord, feature in obj.items():
        agent.locate(coord)
        agent.sense_sweep(feature)
    return agent.commit()


def _close(a, b, tol=1e-6) -> bool:
    return abs(a - b) < tol


def test_a_strict_subset_is_recognised_as_the_whole():
    """The apple-from-one-angle case: a partial view is a consistent SUBSET of the whole, so the whole-object hypothesis
    survives — at the canonical pose AND rotated/translated (partial + posed at once). The unobserved 4th location is a GAP,
    not evidence against."""
    agent = _fresh()
    _learn(agent, FULL)
    pop = _present(agent, SUBSET)
    assert len(pop) == 1 and pop[0].label == 0, f"a strict subset must be recognised as the WHOLE object, got {pop}"
    for omega, shift in ((40.0, (11.0, -7.0)), (215.0, (3.0, 20.0))):
        pop = _present(agent, SUBSET, omega, shift)
        assert {h.label for h in pop} == {0}, f"partial + posed ω={omega}: got {[h.label for h in pop]}"
        assert _close(to_angle(pop[0].rotation), omega), f"pose solved from the partial view: {to_angle(pop[0].rotation)}"


def test_committing_a_partial_view_does_not_mint():
    """Incompleteness must not grow the library: committing a partial view REINFORCES the whole (returns its label), and
    leaves the whole model unharmed (the unobserved feature is still there afterwards)."""
    agent = _fresh()
    _learn(agent, FULL)
    before = len(agent._nav_col().pooler.objects)
    assert _study(agent, SUBSET) == 0, "committing a partial view reinforces the whole, it does not mint"
    assert len(agent._nav_col().pooler.objects) == before, "incompleteness must NOT grow the library"
    assert _present(agent, FULL)[0].label == 0, "learning from a partial view must not corrupt the whole model"


def test_a_contradicting_feature_is_refuted_and_mints():
    """The other side of the boundary: a feature the whole FORBIDS at a known location refutes it — recognition does NOT
    return the whole, and committing the contradicting arrangement mints exactly one NEW identity. Refutation is the only
    trigger; incompleteness is not."""
    agent = _fresh()
    _learn(agent, FULL)
    before = len(agent._nav_col().pooler.objects)
    assert 0 not in {h.label for h in _present(agent, CONTRA)}, "a contradicting view must NOT be recognised as the whole"
    assert _study(agent, CONTRA) != 0, "a contradicting arrangement is a NEW object, not the whole"
    assert len(agent._nav_col().pooler.objects) == before + 1, "contradiction mints exactly one new identity"


if __name__ == "__main__":
    ag = _fresh()
    _learn(ag, FULL)
    print(f"FULL learned → {len(ag._nav_col().pooler.objects)} identity")
    print(f"  strict SUBSET (3 of 4)      → {[(h.label, round(to_angle(h.rotation), 1)) for h in _present(ag, SUBSET)]}")
    print(f"  SUBSET rotated 40°, moved   → {[(h.label, round(to_angle(h.rotation), 1)) for h in _present(ag, SUBSET, 40.0, (11.0, -7.0))]}")
    print(f"  CONTRAdicting 3rd feature   → {[h.label for h in _present(ag, CONTRA)]}  (empty/≠0 = refuted)")
