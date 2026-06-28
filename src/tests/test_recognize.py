"""Pose-invariant object recognition (tbt/recognize.py) — the column's TBT/Monty "what + pose" faculty. Recognise a
known object at an UNSEEN continuous angle, under partial views, with the object set learned ONLINE + label-free, and
across columns by voting. Domain-general: only (location, local-descriptor) sensations, no game."""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.recognize import Recognizer, local_disps, rot, vote  # noqa: E402

TETROMINOES = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}


def _lib():
    rec = Recognizer()
    for name, c in TETROMINOES.items():
        rec.add(name, c)
    return rec


def _walk(rec, cloud, order):
    rec.reset()
    locs = [np.asarray(c, float) for c in cloud]
    for i in order:
        rec.observe(locs[i], local_disps(locs, i, rec.radius))
    return rec.best()


def test_recognises_object_at_unseen_continuous_pose():
    """A known object presented at a random continuous angle + translation it never saw: identify it AND recover the
    pose (the operator reproduces the cloud). 100% — pose is SOLVED, not memorised per orientation."""
    rec = _lib()
    rng = np.random.default_rng(0)
    n = ok = pose_ok = 0
    for name, base in TETROMINOES.items():
        model = next(m for m in rec.models if m.name == name)
        for _ in range(20):
            th, t = rng.uniform(0, 2 * np.pi), rng.uniform(-5, 5, 2)
            cloud = [rot(th) @ np.asarray(c, float) + t for c in base]
            res = _walk(rec, cloud, rng.permutation(len(cloud)))
            n += 1
            if res and res[0] == name:
                ok += 1
                pred = {tuple(np.round(p, 2)) for p in model.cells_at(res[1], res[2])}
                true = {tuple(np.round(p, 2)) for p in cloud}
                pose_ok += (pred == true)
    assert ok == n, f"identification {ok}/{n}"
    assert pose_ok == n, f"pose recovery {pose_ok}/{n}"


def test_chiral_pairs_not_confused():
    """S/Z and J/L are mirror images, NOT rotations — the recogniser (rotations only) must keep them distinct."""
    rec = _lib()
    rng = np.random.default_rng(3)
    for a, b in [("S", "Z"), ("J", "L")]:
        for name in (a, b):
            for _ in range(15):
                th, t = rng.uniform(0, 2 * np.pi), rng.uniform(-5, 5, 2)
                cloud = [rot(th) @ np.asarray(c, float) + t for c in TETROMINOES[name]]
                res = _walk(rec, cloud, rng.permutation(4))
                assert res and res[0] == name, f"{name} misread as {res and res[0]}"


def test_partial_observation_accumulates_evidence():
    """A single glance at a local patch is often ambiguous; movement RESOLVES it — the sensorimotor claim. Accuracy
    must rise with the number of fixations and reach 100% well before the whole object is seen."""
    rec = _lib()
    rng = np.random.default_rng(1)
    acc = {}
    for k in (1, 2, 4):
        ok = n = 0
        for name, base in TETROMINOES.items():
            for _ in range(30):
                th, t = rng.uniform(0, 2 * np.pi), rng.uniform(-5, 5, 2)
                cloud = [rot(th) @ np.asarray(c, float) + t for c in base]
                res = _walk(rec, cloud, rng.permutation(len(cloud))[:k])
                n += 1
                ok += (res is not None and res[0] == name)
        acc[k] = ok / n
    assert acc[1] < acc[2], f"evidence did not accumulate: {acc}"
    assert acc[2] == 1.0, f"two fixations should suffice: {acc}"


def test_learns_object_set_online_label_free():
    """Feed shapes via add_if_novel: each distinct one-sided tetromino is novel (→ 7 objects); every ROTATION of a
    known one is recognised, not re-added. The object set is discovered by watching — never injected."""
    rec = Recognizer()
    new = sum(rec.add_if_novel(c)[1] for c in TETROMINOES.values())
    assert new == 7 and len(rec.models) == 7, f"learned {new} new, library {len(rec.models)}"
    readded = 0
    for base in TETROMINOES.values():
        for q in (1, 2, 3):                                 # 90/180/270-degree rotations of each known piece
            cloud = [rot(q * np.pi / 2) @ np.asarray(c, float) for c in base]
            readded += rec.add_if_novel(cloud)[1]
    assert readded == 0, f"{readded} rotations wrongly treated as new objects"
    assert len(rec.models) == 7


def test_voting_resolves_single_glance_ambiguity():
    """Two columns each take ONE glance at DIFFERENT cells, then VOTE (Monty lateral voting). Pooling their pose
    hypotheses resolves ambiguity that a single one-glance column cannot — voting accuracy beats solo, and is high."""
    rng = np.random.default_rng(5)
    n = solo = voted = 0
    for name, base in TETROMINOES.items():
        for _ in range(30):
            th, t = rng.uniform(0, 2 * np.pi), rng.uniform(-5, 5, 2)
            cloud = [rot(th) @ np.asarray(c, float) + t for c in base]
            order = rng.permutation(len(cloud))
            colA, colB = _lib(), _lib()                     # two columns sharing one object library
            colA.reset(); colB.reset()
            colA.observe(cloud[order[0]], local_disps(cloud, order[0], colA.radius))
            colB.observe(cloud[order[1]], local_disps(cloud, order[1], colB.radius))
            n += 1
            ra = colA.best()
            solo += (ra is not None and ra[0] == name)      # one column, one glance
            rv = vote([colA, colB])                          # two columns, one glance each, voted
            voted += (rv is not None and rv[0] == name)
    assert voted > solo, f"voting did not help: solo {solo}/{n}, voted {voted}/{n}"
    assert voted / n >= 0.9, f"voting accuracy too low: {voted}/{n}"


def test_rotation_operator_is_exact():
    """cells_at IS the rotation — continuous, and exact on the grid at 90 degrees (no table)."""
    rec = Recognizer()
    m = rec.add("I", [(0, 0), (1, 0), (2, 0), (3, 0)])
    got = {tuple(np.round(p, 6)) for p in m.cells_at(np.pi / 2, (0.0, 0.0))}
    assert got == {(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)}, got


def test_column_exposes_recognition_faculty():
    """The fold-in: a CorticalColumn carries the recognizer and exposes it (the 'what + pose' channel alongside L6)."""
    from tbt.column import CorticalColumn
    col = CorticalColumn(n_entities=8, seed=0)
    for name, c in TETROMINOES.items():
        col.learn_object(c, name=name)                      # learn the object library through the column
    cloud = [rot(0.7) @ np.asarray(c, float) + (2.0, -1.0) for c in TETROMINOES["T"]]
    res = col.recognize_object(cloud)                       # recognise it at an unseen pose, through the column
    assert res is not None and res[0] == "T"
