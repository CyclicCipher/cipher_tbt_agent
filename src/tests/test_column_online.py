"""The column's PERCEPTION + forward model: `perceive` unifies recognize -> predict -> correct -> learn over the pose
belief, and `forward` is the one pure prediction over the factored (pose, content) rep. Driven from a coloured cloud
through the peripheral split (cells -> pose; view_signature -> the opaque content id). The tabular/SR/cost/Euclidean
stack these once tested retired with the SDR re-seat (ARCHITECTURE.md §10 P4a)."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.column import CorticalColumn  # noqa: E402
from tbt.retina import view_signature  # noqa: E402


def _perceive(col, action, cloud):
    """Drive `column.perceive` from a coloured cloud, doing the PERIPHERAL split: cells (colour-blind -> the pose) +
    the opaque content descriptor (`retina.view_signature` -> an L4 feature id). The column stays content-opaque."""
    cells = [(x, y) for (x, y, _c) in cloud]
    return col.perceive(action, cells, view_signature(cloud))


def test_perceive_unifies_recognize_predict_correct_learn():
    """`column.perceive` unifies recognize -> PREDICT -> CORRECT -> LEARN -> content, one path for abelian + non-abelian.
    A shape TRANSLATING by a consistent per-action delta is recognized each step; the pose is path-integrated +
    snap-corrected, the operator is LEARNED (a ~(2,0) translation), and content (`view_signature`) is invariant to the
    mover's position (same shape -> one content id)."""
    col = CorticalColumn(n_entities=16, seed=0)
    shape = [(0, 0), (1, 0), (2, 0), (2, 1)]                       # an asymmetric L-tromino (unique pose)

    def cloud_at(ox, oy, colour=7):
        return [(x + ox, y + oy, colour) for (x, y) in shape]

    c0, (x0, y0, _t0) = _perceive(col, None, cloud_at(0, 0))        # cold: observe at the origin
    contents, xs = [c0], [x0]
    for k in range(1, 6):                                          # the mover translates +2x each step under action 0
        c, (x, y, _t) = _perceive(col, 0, cloud_at(2 * k, 0))
        contents.append(c)
        xs.append(x)
    assert len(set(contents)) == 1, contents                      # content invariant to position (same shape)
    assert xs[-1] > xs[0] + 6 and xs == sorted(xs), xs            # the location tracked the translation (monotone +x)
    here = col.here_position()                                     # the gain field LEARNED action 0 -> forward PREDICTS a ~+2x move
    pred, _ = col.forward(0, ("x",))
    assert pred is not None and abs(pred[0] - (here[0] + 2.0)) < 0.6 and abs(pred[1] - here[1]) < 0.6, (here, pred)


def test_forward_predicts_self_motion_over_the_factored_rep():
    """The ONE forward prediction over the factored (pose, content) rep -- apply the operator to the LOCATION, read the
    CONTENT. Self-motion: once the operator is learned, `forward` PREDICTS the next pose and the content is INVARIANT
    (reafference); it is a PURE query (does not mutate the belief). An unlearned action predicts NO movement."""
    col = CorticalColumn(n_entities=16, seed=0)
    shape = [(0, 0), (1, 0), (2, 0), (2, 1)]                       # asymmetric (unique pose)

    def sense(a, ox):
        return col.perceive(a, [(x + ox, y) for (x, y) in shape], ("stuff",))

    sense(None, 0); sense(0, 2); sense(0, 4)                       # learn action 0 = +2x, belief now at (4,0)
    here_before = col.here_position()
    pred_pose, pred_content = col.forward(0, ("stuff",))          # PURE forward prediction from (4,0)
    assert col.here_position() == here_before                    # pure: did NOT mutate the belief
    assert pred_content == ("stuff",)                             # self-motion: content invariant (reafference)
    _feat, obs_pose = sense(0, 6)                                 # the ACTUAL next observation (mover at 6,0)
    assert abs(pred_pose[0] - obs_pose[0]) < 0.6 and abs(pred_pose[1] - obs_pose[1]) < 0.6   # forward matched the observation
    pred_stay, _ = col.forward(9, ("stuff",))                    # action 9 has no learned operator -> predicts staying put
    here_now = col.here_position()
    assert abs(pred_stay[0] - here_now[0]) < 0.6 and abs(pred_stay[1] - here_now[1]) < 0.6


def test_perceive_prediction_error_is_the_forward_model_residual():
    """`perceive._pred_error` is the FORWARD MODEL's residual -- the PREDICTED pose (operator) vs the OBSERVED pose, as a
    fraction of the move (§4c, the learning signal). The FIRST action-0 move has no learned operator (identity predicts
    staying put) -> the prediction MISSES (error high); once the operator converges, the prediction NAILS it (error ~0)."""
    col = CorticalColumn(n_entities=16, seed=0)
    shape = [(0, 0), (1, 0), (2, 0), (2, 1)]

    def sense(a, ox):
        return col.perceive(a, [(x + ox, y) for (x, y) in shape], ("stuff",))

    sense(None, 0)                                                 # cold observe at the origin
    sense(0, 2)                                                    # FIRST action-0 move: operator unlearned -> identity predict misses
    first_error = col._pred_error
    for k in range(2, 8):                                          # keep moving +2x -> the operator converges
        sense(0, 2 * k)
    late_error = col._pred_error
    assert first_error > 0.5, first_error                         # identity predicted staying -> a big residual
    assert late_error < 0.2, late_error                           # operator learned -> the prediction nails the pose
