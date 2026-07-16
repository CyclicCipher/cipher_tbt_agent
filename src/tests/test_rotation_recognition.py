"""End-to-end test of POSE-INVARIANT recognition (ARCHITECTURE.md §8; plan `notes/rotation_invariance_plan.md` R4).

THE CROWN PROPERTY: learn an object once, in its own frame; then present it ROTATED to any angle AND TRANSLATED anywhere,
sweep it starting from ANY point, and the agent recovers the object, its rotation, and where it is — from a model that never
saw it that way.

THE MECHANISM — the pose is **SOLVED, not scanned** (Monty; `reference_tbt_pose_invariant_recognition`: "you recognize an
unseen orientation because you SOLVE for the rotation; you don't recall it"). Our features are colour-at-location, carrying
no intrinsic orientation, so the orienting cue is the inter-fixation DISPLACEMENT geometry: an object at (ω, t) puts model
point ℓ at rotate(ℓ,ω)+t, so two fixations give p₁−p₀ = rotate(ℓ₁−ℓ₀, ω) — translation cancels — and ω falls out in closed
form. What is HYPOTHESISED is the correspondence (which model point each fixation touched, seeded by the L4→L6a union); the
pose is DERIVED and then VERIFIED by the model's own prediction. Nothing samples angles.

THE POPULATION IS THE ANSWER. `recognize` returns the tied-best hypotheses, and a tie is information, not a failure: a 4-fold
object returns its whole symmetry ORBIT because it genuinely has no single pose. Reporting one angle there would be a lie.
"""

from __future__ import annotations

import math
import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent  # noqa: E402

# An ASYMMETRIC object: three distinct features in an L — its pose is unambiguous.
OBJ = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 3}
# A SECOND object built to be INDISTINGUISHABLE FROM OBJ FOR THE FIRST TWO FIXATIONS: same features 1 and 2, separated by the
# same displacement (2,0). So the seed CANNOT tell them apart (same solved ω, different origin) and neither can the isometry
# prune — only the THIRD fixation's evidence can. That is the union narrowing (Lewis 2019), and it is what makes the seed a
# hypothesis rather than an answer. Its features sit at different LOCATIONS than OBJ's, so the two are separable at learning.
OTHER = {(1.0, 0.0): 1, (3.0, 0.0): 2, (1.0, -2.0): 6}
# A 4-fold SYMMETRIC object: the SAME feature at four 90°-rotations → pose determined only up to 90°.
SYM = {(2.0, 0.0): 5, (0.0, 2.0): 5, (-2.0, 0.0): 5, (0.0, -2.0): 5}


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _rotate(p, deg: float):
    r = math.radians(deg)
    x, y = p
    return (x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r))


def _learn(agent: Agent, *objects: dict, passes: int = 6) -> None:
    """Learn each object at its canonical pose, in its OWN frame: an EPISODE — onset, buffer the sweep, then commit."""
    for _ in range(passes):
        for obj in objects:
            agent.start_object()
            for coord, feature in obj.items():
                agent.locate(coord)
                agent.sense_sweep(feature)
            agent.commit()


def _present(agent: Agent, obj: dict, omega: float = 0.0, shift=(0.0, 0.0), order=None):
    """Sweep the object as PRESENTED — rotated by ω, translated by `shift`, visited in `order` (default: as stored). The
    agent is told only where IT is and what it senses; nothing tells it the object's pose or which point it started on."""
    agent.start_object()
    coords = list(obj) if order is None else [list(obj)[i] for i in order]
    for coord in coords:
        p = _rotate(coord, omega)
        agent.locate((p[0] + shift[0], p[1] + shift[1]))
        agent.sense_sweep(obj[coord])
    return agent.recognize()


def _close(a, b, tol=1e-6) -> bool:
    return abs(a - b) < tol


def test_recognises_an_object_at_a_novel_orientation_and_solves_the_pose():
    agent = _fresh()
    _learn(agent, OBJ)
    canonical = _present(agent, OBJ)[0].label
    for omega in (15.0, 45.0, 90.0, 180.0, 285.0):
        pop = _present(agent, OBJ, omega)
        assert len(pop) == 1, f"rotated {omega}°: an ASYMMETRIC object must yield ONE hypothesis, got {len(pop)}"
        assert pop[0].label == canonical, f"rotated {omega}°: recognised {pop[0].label}, expected {canonical}"
        assert _close(pop[0].omega, omega), f"rotated {omega}°: solved pose {pop[0].omega}"


def test_solves_an_ARBITRARY_off_grid_angle_exactly():
    """The pose is derived from continuous geometry, so there is no sampling to land on: 37° / 113.5° / 244.25° come out
    EXACT. The retired scan could only ever report an angle it was handed, and the discrete code before it could not
    represent these without N > 2*pi*radius modules."""
    agent = _fresh()
    _learn(agent, OBJ)
    for omega in (37.0, 113.5, 244.25, 359.9):
        pop = _present(agent, OBJ, omega)
        assert len(pop) == 1 and pop[0].label == 0, f"off-grid {omega}°: got {pop}"
        assert _close(pop[0].omega, omega), f"off-grid {omega}°: solved pose {pop[0].omega}"


def test_ENTERED_ANYWHERE_on_an_object_placed_anywhere():
    """THE R4 CROWN. The object is rotated AND translated far from its learned frame, and the sweep starts on a DIFFERENT
    feature each time — so no anchor is shared with the model and the agent cannot assume where on the object it landed.
    Solving the pose recovers the rotation AND the object's origin. The retired scan could not do this at all: it assumed
    the sweep and the model shared an anchor."""
    agent = _fresh()
    _learn(agent, OBJ)
    for omega, shift, order in ((37.0, (20.0, -13.0), (2, 0, 1)),
                                (150.0, (-8.0, 41.0), (1, 2, 0)),
                                (270.0, (33.5, 33.5), (2, 1, 0))):
        pop = _present(agent, OBJ, omega, shift, order)
        assert len(pop) == 1, f"ω={omega} t={shift}: expected ONE hypothesis, got {[(h.omega, h.origin) for h in pop]}"
        h = pop[0]
        assert h.label == 0, f"ω={omega} t={shift}: recognised {h.label}"
        assert _close(h.omega, omega), f"ω={omega} t={shift}: solved rotation {h.omega}"
        assert _close(h.origin[0], shift[0]) and _close(h.origin[1], shift[1]), \
            f"ω={omega} t={shift}: solved origin {h.origin}"


def test_an_AMBIGUOUS_seed_is_narrowed_by_evidence_not_by_the_seed():
    """OBJ and OTHER share features 1 and 2 at the same separation, so after two fixations they are genuinely
    indistinguishable: the union seeds a hypothesis on EACH, both solving the SAME ω, and the isometry prune cannot separate
    them either (the displacement is identical by construction). Only the third fixation's evidence can — under the wrong
    object's pose it lands where that object has nothing, L4 bursts, and the hypothesis is refuted. This is the union
    narrowing to one object (Lewis 2019): the seed proposes, the model's own prediction disposes."""
    agent = _fresh()
    _learn(agent, OBJ, OTHER)
    union = agent._nav_col()._union_for(agent._feat_enc.encode(1))
    assert len(union) == 2, f"feature 1 occurs on BOTH objects — the union must carry both, got {len(union)}"
    seen = {}
    for obj, name in ((OBJ, "OBJ"), (OTHER, "OTHER")):
        for omega in (41.0, 200.0):
            pop = _present(agent, obj, omega, shift=(11.0, 7.0))
            assert len(pop) == 1, f"{name} @{omega}°: evidence must leave ONE hypothesis, got {[h.label for h in pop]}"
            assert _close(pop[0].omega, omega), f"{name} @{omega}°: solved {pop[0].omega}"
            seen.setdefault(name, set()).add(pop[0].label)
    assert all(len(v) == 1 for v in seen.values()), f"each object must recognise consistently, got {seen}"
    assert seen["OBJ"] != seen["OTHER"], f"the two objects must be DIFFERENT identities, got {seen}"


def test_symmetry_yields_the_symmetry_orbit_not_a_forced_pose():
    """A 4-fold symmetric object: identity is recognised, but the pose is undetermined up to 90°. The population must carry
    the whole orbit — an EXACT tie no amount of evidence could break, because the object really is the same at all four."""
    agent = _fresh()
    _learn(agent, SYM)
    pop = _present(agent, SYM, 0.0)
    assert {h.label for h in pop} == {0}, "a symmetric object is still RECOGNISED (symmetry affects pose, not identity)"
    assert sorted(h.omega for h in pop) == [0.0, 90.0, 180.0, 270.0], \
        f"4-fold symmetry → the orbit every 90°, got {sorted(h.omega for h in pop)}"


def test_studying_a_ROTATED_known_object_reinforces_it_rather_than_duplicating():
    """LEARNING goes through the same pose-invariant recognition, so meeting a known object at a novel pose adds no duplicate
    — identity and pose stay FACTORED. This is only possible because `commit` decides per EPISODE: the rotation is not
    apparent from any single fixation. It also has to bind in the object's OWN frame (the replay un-rotates), or the model
    would be corrupted by rotated coordinates — so the test re-checks recognition afterwards."""
    agent = _fresh()
    _learn(agent, OBJ)
    before = len(agent._nav_col().pooler.objects)
    for omega in (90.0, 217.0):
        agent.start_object()
        for coord, feature in OBJ.items():
            p = _rotate(coord, omega)
            agent.locate(p)
            agent.sense_sweep(feature)
        label = agent.commit()
        assert label == 0, f"studying OBJ at {omega}° must recognise it, got {label}"
        assert len(agent._nav_col().pooler.objects) == before, \
            f"studying OBJ at {omega}° minted a DUPLICATE — a known object at a novel pose is the same object"
    pop = _present(agent, OBJ, 45.0)
    assert len(pop) == 1 and pop[0].label == 0 and _close(pop[0].omega, 45.0), \
        f"the model must be unharmed by learning at a rotated pose (bind in the OBJECT's frame), got {pop}"


def test_a_single_fixation_fixes_no_rotation():
    """One point cannot determine a rotation — that is geometry, not a shortcoming. The honest answer is no hypothesis, not
    a guessed angle."""
    agent = _fresh()
    _learn(agent, OBJ)
    agent.start_object()
    agent.locate((5.0, 5.0))
    agent.sense_sweep(1)
    assert agent.recognize() == [], "a single fixation must yield NO pose hypothesis"


if __name__ == "__main__":
    ag = _fresh()
    _learn(ag, OBJ, OTHER)
    for w in (0.0, 37.0, 244.25):
        h = _present(ag, OBJ, w)[0]
        print(f"OBJ at {w:7.2f}°           → object {h.label}, solved ω={h.omega:7.2f}, origin={h.origin}")
    h = _present(ag, OBJ, 150.0, (-8.0, 41.0), (1, 2, 0))[0]
    print(f"OBJ rotated+moved, entered mid → object {h.label}, solved ω={h.omega:7.2f}, origin=({h.origin[0]:.1f}, {h.origin[1]:.1f})")
    h = _present(ag, OTHER, 41.0, (11.0, 7.0))[0]
    print(f"OTHER (shares feature 1)       → object {h.label}, solved ω={h.omega:7.2f}")
    sym = _fresh()
    _learn(sym, SYM)
    print(f"4-fold symmetric               → orbit {sorted(h.omega for h in _present(sym, SYM, 0.0))} (an EXACT tie)")
