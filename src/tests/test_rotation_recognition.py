"""R3 of the rotation plan (`notes/rotation_invariance_plan.md`): the ORIENTATION SCAN — recognise a rotated object and
report its pose.

THE CROWN PROPERTY: learn an object at its canonical orientation, present it at a NOVEL orientation, and the agent both
recognises it AND reports the rotation. Mechanism (Numenta 2021): for each candidate rotation k, REPLAY the buffered sweep
with every sensed location UN-rotated by k (the rotation operator, R2) and score how well the model EXPLAINS it — the count
of fixations L4 predicts (does not burst). Prediction accuracy ranks the hypothesis; no free pose search, no particle filter.

SYMMETRY: a symmetric object produces TIES — equal-scoring k's are its symmetry orbit. That is CORRECT (the pose is genuinely
undetermined up to that symmetry); identity is still recognised, and the tie set is returned rather than an arbitrary angle.

NB the rotation column takes its location by SENSORY FIX (`locate_rotated`), not path integration: the oriented grid makes
ROTATION exact but TRANSLATION path-integration inexact (a unit move shifts module i's phase by cos θ_i — not an integer).
"""

from __future__ import annotations

import math
import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent  # noqa: E402

N = 8
DELTA = 360.0 / N
# An ASYMMETRIC object: three distinct features in an L — its pose is unambiguous.
OBJ = {(0.0, 0.0): 1, (2.0, 0.0): 2, (0.0, 2.0): 3}
# A 4-fold SYMMETRIC object: the SAME feature at four 90°-rotations of each other → pose determined only up to 90°.
SYM = {(2.0, 0.0): 5, (0.0, 2.0): 5, (-2.0, 0.0): 5, (0.0, -2.0): 5}


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _rotate(p, deg: float):
    r = math.radians(deg)
    x, y = p
    return (x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r))


def _learn(agent: Agent, obj: dict, passes: int = 6) -> None:
    """Learn the object at its canonical orientation (ω = 0)."""
    for _ in range(passes):
        agent.start_rotated_object()
        for coord, feature in obj.items():
            agent.locate_rotated(coord)
            agent.perceive_rotated(feature, learn=True)


def _present(agent: Agent, obj: dict, k: int):
    """Sweep the object PRESENTED rotated by ω = k·Δ, buffering the sweep; then scan orientations."""
    agent.start_rotated_object()
    for coord, feature in obj.items():
        agent.locate_rotated(_rotate(coord, k * DELTA))
        agent.sense_sweep_rotated(feature)
    return agent.recognize_rotated()


def test_recognises_an_object_at_a_novel_orientation_and_reports_the_pose():
    agent = _fresh()
    _learn(agent, OBJ)
    canonical, _, _ = _present(agent, OBJ, 0)
    for k in (1, 2, 3, 5, 7):                       # orientations never seen during learning
        label, pose, ties = _present(agent, OBJ, k)
        assert label == canonical, f"rotated by {k}·Δ: recognised {label}, expected {canonical}"
        assert pose == k, f"rotated by {k}·Δ: inferred pose {pose}, expected {k}"
        assert ties == [k], f"an ASYMMETRIC object must have ONE best pose, got ties {ties}"


def test_a_full_turn_reads_as_the_canonical_pose():
    agent = _fresh()
    _learn(agent, OBJ)
    label, pose, _ = _present(agent, OBJ, N)        # 360° == 0°
    assert pose == 0 and label == 0


def test_symmetry_yields_the_symmetry_orbit_not_a_forced_angle():
    """A 4-fold symmetric object: identity is still recognised, but the pose is undetermined up to 90° — the scan must
    return the TIED orbit (4 of the 8 steps), not invent a single angle."""
    agent = _fresh()
    _learn(agent, SYM)
    label, _, ties = _present(agent, SYM, 0)
    assert label == 0, "a symmetric object is still RECOGNISED (symmetry affects pose, not identity)"
    assert sorted(ties) == [0, 2, 4, 6], f"4-fold symmetry → the orbit {{0,2,4,6}} (every 90°), got {ties}"


if __name__ == "__main__":
    ag = _fresh()
    _learn(ag, OBJ)
    for k in range(N):
        label, pose, ties = _present(ag, OBJ, k)
        print(f"presented at {k*DELTA:5.1f}° → object {label}, inferred pose {pose} ({pose*DELTA:5.1f}°), ties {ties}")
    sym = _fresh()
    _learn(sym, SYM)
    print(f"4-fold symmetric object → {_present(sym, SYM, 0)}  (ties = the symmetry orbit)")
