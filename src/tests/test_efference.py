"""test_efference.py — L5PT's EFFERENCE COPY + the flow-parsing fix for the moving sensor (step 1 of the L5 build;
notes/l5_efference_broadcast_design.md).

L5PT emits the chosen displacement, which is the motor command AND the efference copy AND the message broadcast to peer columns.
The moving-sensor bug ("one column had the efference, the others did not"): when the body moves by Δ, a static thing shifts by
−Δ in an egocentric frame; a peer that did NOT get the efference reads that as the OBJECT moving. The fix: broadcast the
efference; each peer path-integrates its own L6a by it (`apply_efference`), so `to_world` cancels self-motion — a static object
stays static, only real WORLD-motion survives.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                       # noqa: E402
from tbt.column import Column                     # noqa: E402
from tbt.encoders import GridEncoder              # noqa: E402
from tbt.operator import eye, sub                 # noqa: E402


def _close(a, b, tol=1e-6) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a, b))


def _col() -> Column:
    grid = GridEncoder(scales=(7, 11, 13, 17), dims=2, mw=1, bounds=[(0, 63)] * 2)
    return Column(sensory_n=1, n_cols=64, order=2, seed=0, location=grid)


def _teach_east(col: Column) -> None:
    col.learn_pose_move("E", ((0.0, 0.0), eye(2)), ((1.0, 0.0), eye(2)))


def test_efference_is_the_world_frame_self_motion():
    """L5PT's efference = the world-frame self-motion the operator predicts for the action; None if unlearned."""
    c = _col()
    c.set_pose((10.0, 10.0), eye(2))
    _teach_east(c)
    dp, dR = c.efference("E")
    assert _close(dp, (1.0, 0.0)) and dR == eye(2), f"efference must be the world-frame self-move, got {dp},{dR}"
    assert c.efference("NEVER_TAUGHT") is None, "an unlearned action has no efference (no prediction)"


def test_apply_efference_path_integrates_a_peer():
    """A peer path-integrates its own L6a by a broadcast self-motion."""
    c = _col()
    c.set_pose((5.0, 5.0), eye(2))
    c.apply_efference(((2.0, 0.0), eye(2)))
    assert _close(c.pose()[0], (7.0, 5.0)), f"peer pose must advance by the broadcast motion, got {c.pose()[0]}"


def test_a_static_object_stays_static_only_with_the_efference():
    """THE FALSIFIER. The body moves east; a STATIC object at (20,10) shifts −Δ egocentrically. WITH the broadcast efference the
    peer recovers it at (20,10) (static); WITHOUT, it misreads the self-motion as the object having moved to (19,10)."""
    W = (20.0, 10.0)
    body = _col()
    body.set_pose((10.0, 10.0), eye(2))
    _teach_east(body)
    motion = body.efference("E")                              # emit BEFORE moving (the operator predicts the self-move)
    body.path_integrate("E")                                  # the body moves east → sensor physically at (11,10)
    ego = sub(W, body.pose()[0])                              # what the sensor now reads: the object at (9,0) egocentric

    with_eff = _col(); with_eff.set_pose((10.0, 10.0), eye(2)); with_eff.apply_efference(motion)
    assert _close(with_eff.to_world((ego, eye(2)))[0], W), "with the efference, a static object stays static"

    without = _col(); without.set_pose((10.0, 10.0), eye(2))  # never told the body moved
    bad = without.to_world((ego, eye(2)))[0]
    assert not _close(bad, W) and _close(bad, (19.0, 10.0)), \
        f"without the efference, the static object is misread as moved by the un-cancelled self-motion; got {bad}"


def test_a_moving_object_nets_to_its_true_world_motion():
    """The other side: a peer with the efference recovers a genuinely MOVING object's true world position — self-motion and
    world-motion are separated, not conflated (the object moved +2 while the body moved +1)."""
    body = _col()
    body.set_pose((10.0, 10.0), eye(2))
    _teach_east(body)
    motion = body.efference("E")
    body.path_integrate("E")                                  # sensor 10 → 11
    obj_after = (22.0, 10.0)                                   # the object moved 20 → 22 (world +2)
    ego = sub(obj_after, body.pose()[0])                      # egocentric reading = 22 − 11 = (11,0)
    peer = _col(); peer.set_pose((10.0, 10.0), eye(2)); peer.apply_efference(motion)
    assert _close(peer.to_world((ego, eye(2)))[0], obj_after), \
        "with the efference, the peer recovers the moving object's TRUE world position (+2), not the un-cancelled +1"


def test_agent_broadcasts_the_efference_to_peer_spatial_columns():
    """Wired: the agent (the thalamic relay) emits the self column's efference and routes it to peer spatial columns."""
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a.set_pose((10.0, 10.0), eye(2))
    a.learn_pose_move("E", ((0.0, 0.0), eye(2)), ((1.0, 0.0), eye(2)))
    a.place_object(1, ((5.0, 5.0), eye(2)))                   # builds the compositional (peer) column
    a._scene_col().set_pose((10.0, 10.0), eye(2))            # give the peer a sensor pose to track
    motion = a.broadcast_efference("E")
    assert _close(motion[0], (1.0, 0.0)), "the agent emits the self column's efference"
    assert _close(a._scene_col().pose()[0], (11.0, 10.0)), \
        "a peer spatial column is path-integrated by the broadcast efference"


if __name__ == "__main__":
    b = _col(); b.set_pose((10.0, 10.0), eye(2)); _teach_east(b)
    m = b.efference("E"); b.path_integrate("E")
    e = sub((20.0, 10.0), b.pose()[0])
    on = _col(); on.set_pose((10.0, 10.0), eye(2)); on.apply_efference(m)
    off = _col(); off.set_pose((10.0, 10.0), eye(2))
    print(f"static object (20,10), body moved E:  with efference → {on.to_world((e, eye(2)))[0]}  "
          f"without → {off.to_world((e, eye(2)))[0]}")
