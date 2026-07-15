"""End-to-end test of ROTATION-INVARIANT recognition (ARCHITECTURE.md §8; plan `notes/rotation_invariance_plan.md`).

THE CROWN PROPERTY: learn an object at its canonical orientation, present it at a NOVEL orientation, and the agent both
recognises it AND reports the rotation. Mechanism: for each candidate pose ω, REPLAY the buffered sweep with every sensed
location UN-rotated by ω, and score how well the model EXPLAINS it — the count of fixations L4 predicts (does not burst).
Prediction accuracy ranks the hypothesis; no free pose search, no particle filter.

AFTER THE CUT-OVER the location state is CONTINUOUS, so a candidate pose can be ANY angle — the sampling is a free choice,
not a property of the code. The old discrete design could only test multiples of 360/N, and preserving the code under an
off-grid rotation cost N > 2π·radius modules (measured). Hence the off-grid test below, which the old design could not pass
at any sane module count.

TIES are the poses the evidence cannot SEPARATE, and there are two distinct reasons for that — both tested here:
  • SYMMETRY — an exact tie: a 4-fold object genuinely has no single pose, so the orbit is returned, never a forced angle.
  • ANGULAR RESOLUTION — a physical limit: distinguishing ω from ω+δ needs the sensor to resolve the arc it moves, ≈ r·δ.
    So pose precision ≈ (spatial resolution / object radius) — a BIGGER object pins its pose more finely (measured: a
    radius-2 object resolves to ±14°, radius-8 to ±3°). This is geometry, not a substrate limit: it is exactly the law a
    real sensor obeys, and it is why the scan samples above the resolution rather than chasing arbitrary precision.
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
# The SAME shape at 4x the radius — a longer lever arm, so its pose is resolved more finely (the resolution law).
BIG = {(0.0, 0.0): 1, (8.0, 0.0): 2, (0.0, 8.0): 3}
# A 4-fold SYMMETRIC object: the SAME feature at four 90°-rotations → pose determined only up to 90°.
SYM = {(2.0, 0.0): 5, (0.0, 2.0): 5, (-2.0, 0.0): 5, (0.0, -2.0): 5}
GRID15 = [float(a) for a in range(0, 360, 15)]


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _rotate(p, deg: float):
    r = math.radians(deg)
    x, y = p
    return (x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r))


def _learn(agent: Agent, obj: dict, passes: int = 6) -> None:
    """Learn the object at its canonical orientation (ω = 0)."""
    for _ in range(passes):
        agent.start_object()
        for coord, feature in obj.items():
            agent.locate(coord)
            agent.perceive(feature, learn=True)


def _fresh_learned(obj: dict) -> Agent:
    agent = _fresh()
    _learn(agent, obj)
    return agent


def _present(agent: Agent, obj: dict, omega: float, candidates=None):
    """Sweep the object PRESENTED rotated by ω, buffering the sweep; then scan candidate poses."""
    agent.start_object()
    for coord, feature in obj.items():
        agent.locate(_rotate(coord, omega))
        agent.sense_sweep(feature)
    return agent.recognize_rotated(candidates)


def test_recognises_an_object_at_a_novel_orientation_and_reports_the_pose():
    agent = _fresh()
    _learn(agent, OBJ)
    canonical, _, _ = _present(agent, OBJ, 0.0, GRID15)
    for omega in (15.0, 45.0, 90.0, 180.0, 285.0):
        label, pose, ties = _present(agent, OBJ, omega, GRID15)
        assert label == canonical, f"rotated {omega}°: recognised {label}, expected {canonical}"
        assert abs(pose - omega) < 1e-6, f"rotated {omega}°: inferred pose {pose}"
        assert ties == [omega], f"an ASYMMETRIC object must have ONE best pose, got {ties}"


def test_recognises_at_an_ARBITRARY_off_grid_angle():
    """The payoff of the continuous state: a pose is not confined to a module count. 37° / 113.5° / 244.25° lie off ANY
    coarse grid, and the scan simply samples them — the retired discrete design could not represent these without
    N > 2*pi*radius modules. Candidates are spaced 30° apart, i.e. above the radius-2 object's ±14° resolution, so the true
    angle must win OUTRIGHT."""
    agent = _fresh()
    _learn(agent, OBJ)
    for omega in (37.0, 113.5, 244.25):
        cands = [omega - 60.0, omega - 30.0, omega, omega + 30.0, omega + 60.0]
        label, pose, ties = _present(agent, OBJ, omega, cands)
        assert label == 0, f"off-grid {omega}°: recognised {label}"
        assert ties == [omega], f"off-grid {omega}°: expected the true angle to win outright, tied {ties}"
        assert abs(pose - omega) < 1e-6, f"off-grid {omega}°: inferred pose {pose}"


def test_pose_resolution_is_set_by_the_lever_arm_not_the_substrate():
    """Distinguishing ω from ω+δ requires resolving the arc a feature travels, ≈ r·δ — so pose precision ≈ (spatial
    resolution / object radius). The SAME shape at 4x the radius must therefore resolve its pose several times more finely.
    This is the honest limit on the scan: geometry, obeyed by any real sensor — NOT the module-count wall of the retired
    discrete code, which no amount of object size could have widened."""
    fine = [37.0 + k for k in range(-25, 26)]
    _, _, small_ties = _present(_fresh_learned(OBJ), OBJ, 37.0, fine)
    _, _, big_ties = _present(_fresh_learned(BIG), BIG, 37.0, fine)
    small, big = len(small_ties), len(big_ties)
    assert 37.0 in small_ties and 37.0 in big_ties, "the TRUE pose must always be among the indistinguishable set"
    assert big < small, f"a 4x-larger object must resolve pose more finely, got {big}° vs {small}° of tied angles"


def test_symmetry_yields_the_symmetry_orbit_not_a_forced_angle():
    """A 4-fold symmetric object: identity is still recognised, but pose is undetermined up to 90° — the scan must return the
    TIED orbit, not invent a single angle. Unlike the resolution ties above, this tie is EXACT and no sensor improvement
    would break it: the object really is the same at all four angles."""
    agent = _fresh()
    _learn(agent, SYM)
    label, _, ties = _present(agent, SYM, 0.0, GRID15)
    assert label == 0, "a symmetric object is still RECOGNISED (symmetry affects pose, not identity)"
    assert sorted(ties) == [0.0, 90.0, 180.0, 270.0], f"4-fold symmetry → the orbit every 90°, got {ties}"


if __name__ == "__main__":
    ag = _fresh()
    _learn(ag, OBJ)
    for w in (0.0, 15.0, 90.0, 285.0):
        print(f"presented at {w:6.1f}° → {_present(ag, OBJ, w, GRID15)}")
    for w in (37.0, 113.5, 244.25):
        print(f"OFF-GRID   {w:6.2f}° → {_present(ag, OBJ, w, [w - 60.0, w - 30.0, w, w + 30.0, w + 60.0])}")
    fine = [37.0 + k for k in range(-25, 26)]
    print(f"resolution, radius 2 → +/-{(len(_present(_fresh_learned(OBJ), OBJ, 37.0, fine)[2]) - 1) // 2}°")
    print(f"resolution, radius 8 → +/-{(len(_present(_fresh_learned(BIG), BIG, 37.0, fine)[2]) - 1) // 2}°")
    sym = _fresh()
    _learn(sym, SYM)
    print(f"4-fold symmetric → {_present(sym, SYM, 0.0, GRID15)}  (an EXACT tie = the symmetry orbit)")
