"""Layer 5 — the per-action OPERATOR layer: the column's dynamics + motor-output seat.

L5 is the cortex's main OUTPUT layer and its displacement-cell layer (reference_layer5_role; TBP arxiv 2412.18354). L5a
originates the MOTOR command AND its **efference copy** — the predicted BODY-FRAME pose displacement the action produces.
That efference copy is the SOLE home of the self's per-action displacement (`self.eff`, `efference()`): **L6 reads it and
applies the head-direction gain field to path-integrate the location — L6 keeps NO copy** (HIPPOCAMPUS.md). The controllable
body is then NOT selected anywhere — it FALLS OUT as the sensory change consistent with this efference (reafference, von
Holst; the column's common-fate attention). The move-vs-turn KIND is emergent (a large translation vs a large heading
increment), never assigned. Two live uses of the one displacement: the MOTOR command and the EFFERENCE COPY (self); the
OTHER-object DRIVER + the config-state/tabular operator were retired with the symbolic path (only the SDR/bump/recognition
loop remains).

The continuous-pose geometry (`rot`/`pose_operator`/`apply_pose`) is the displacement-cell group action L2/3 recognition
reads through — its ONE home is here, not a side library. The ANALYTIC pose SOLVE (`pose_between`, an atan2 angle read-off)
was RETIRED with the SDR-native migration (SDR_MIGRATION.md M4): L2/3 now infers orientation by VIRTUAL ROTATION over
discrete bins (`l23_object`), reading through these rotation OPERATORS, not solving an angle."""

from __future__ import annotations

import numpy as np                                              # the pose operators act on small point clouds
import torch.nn as nn

from .operator import Operator                                # L6_NONABELIAN Stage 0: the composable (matrix) form of an action


# ---- pose = a GROUP ELEMENT acting on displacements (the displacement-cell geometry) -------------------------
# A "pose" re-expresses one set of displacement vectors as another under a group element. The SPATIAL instance
# (this visual column) is SO(2) + a translation, so pose = (theta, t). For an ABSTRACT column the group is
# LEARNED from the action-orbit structure (see the memory reference_tbt_frames_and_hippocampus): these functions
# are the SO(2) plug-in, deliberately shaped as "apply a group element to displacements" so the abstract case
# slots in without a rewrite. They are the ONE home of this geometry -- recognition (L2/3) imports them here, it
# does not keep its own copy.

def rot(theta: float) -> np.ndarray:
    """A group element in matrix form: the 2-D rotation R(theta) in SO(2) -- now the 2x2 block of the unified SE(2)
    `pose_operator` (L6_NONABELIAN: ONE source of the rotation, in the operator machinery; L2/3's direct `rot` uses flow
    through it transparently)."""
    return pose_operator(theta).M[:2, :2]


def local_disps(locs, i, radius):
    """Node `i`'s local patch = the displacement vectors to cells within `radius` (group-EQUIVARIANT -- they
    transform with the object). This is the sensed 'feature pose': the local neighbourhood geometry a pose acts on."""
    p = locs[i]
    r2 = radius * radius
    return np.array([locs[j] - p for j in range(len(locs))
                     if j != i and (locs[j][0] - p[0]) ** 2 + (locs[j][1] - p[1]) ** 2 <= r2])


def pose_operator(theta, t=(0.0, 0.0)) -> Operator:
    """A pose (SE(2) group element) as an `Operator` (L6_NONABELIAN unification): the 3x3 HOMOGENEOUS rotation-θ +
    translation-t. This makes L2/3's pose ONE instance of the ONE operator machinery (L5/L6), not a separate SO(2)
    plug-in: poses COMPOSE via `.then` (non-abelian SE(2)), interpolate/extrapolate via `.power` (the CONTINUOUS group),
    and can be LEARNED via `Operator.fit` (Procrustes IS the pose solve) -- so the hand-coded `rot`/`apply_pose` become the
    special case, and an abstract column's group slots in by LEARNING the operator instead of exponentiating SO(2)."""
    c, s = np.cos(theta), np.sin(theta)
    return Operator([[c, -s, float(t[0])], [s, c, float(t[1])], [0.0, 0.0, 1.0]])


def apply_pose(cloud, theta, t):
    """Apply a pose (group element + translation) to a point cloud = the SE(2) `pose_operator(theta, t)` acting on each
    (homogeneous) point -- `R(theta)·loc + t`. ONE operator machinery (L6_NONABELIAN); correct by construction."""
    P = pose_operator(theta, t)
    return [P.apply((float(loc[0]), float(loc[1]), 1.0))[:2] for loc in cloud]


class L5_Displacement(nn.Module):

    def __init__(self):
        super().__init__()
        self.eff: dict = {}                                          # action -> (dx, dy, dtheta): the EFFERENCE COPY -- the action's BODY-FRAME pose delta

    # ---- the EFFERENCE COPY: the per-action BODY-FRAME pose delta (L5's core, TBP paper) ----------------------
    # TBT: L5a originates the motor command AND its EFFERENCE COPY -- the predicted pose displacement the action
    # produces (arxiv 2412.18354). It is the ONE per-action operator in the BODY frame: a translation `(dx, dy)` + a
    # turn `dtheta`. L6 (the allocentric map) reads it and applies the head-direction GAIN FIELD (`R(head)·(dx,dy)`,
    # the heading turned by `dtheta`) to path-integrate the location -- so FORWARD is learned ONCE and rotated to every
    # heading (HIPPOCAMPUS.md). This is the SOLE home of the self's displacement; L6 keeps NO copy.
    def learn_efference(self, action, disp, dtheta: float = 0.0, rate: float = 0.5) -> None:
        """Learn the action's BODY-FRAME efference from one observed transition (`disp=(dx,dy)` de-rotated to the body
        frame by L6, `dtheta` the heading change) -- an EWMA (first sighting sets it; later ones average out a noisy
        recognition). The move-vs-turn KIND is emergent (a large `disp` vs a large `dtheta`), never assigned."""
        dx, dy, dth = float(disp[0]), float(disp[1]), float(dtheta)
        old = self.eff.get(action)
        self.eff[action] = (dx, dy, dth) if old is None else (
            (1 - rate) * old[0] + rate * dx, (1 - rate) * old[1] + rate * dy, (1 - rate) * old[2] + rate * dth)

    def efference(self, action):
        """The action's efference copy `((dx, dy), dtheta)` in the BODY frame -- what L6 path-integrates by (the gain
        field rotates the translation by the head-direction). `((0,0), 0)` for an action whose effect is not yet learned
        (dead-reckon says 'stay', the sighting then corrects)."""
        dx, dy, dth = self.eff.get(action, (0.0, 0.0, 0.0))
        return (dx, dy), dth

    def learned(self, action) -> bool:
        """Has `action`'s efference been learned as a real MOVE or TURN? (a nonzero displacement or heading change)."""
        e = self.eff.get(action)
        return e is not None and ((e[0] * e[0] + e[1] * e[1]) ** 0.5 > 0.5 or abs(e[2]) > 1e-2)

    # ---- motor output + thalamus driver (the other two uses of the one operator) -------------------------
    def motor(self, a):
        """The MOTOR command: the enacted action. L5 is the cortex's output layer -- the chosen action is its output
        (the name->GameAction mapping is the motor ORGAN, in arc_sdk). Identity over discrete actions, by design."""
        return a

    # ---- pose operators (the displacement-cell geometry, the layer's API) --------------------------------
    # The continuous-pose half of L5: a pose is a GROUP ELEMENT acting on displacements (spatial instance: SO(2)
    # + translation; abstract columns: a learned group -- reference_tbt_frames_and_hippocampus). Recognition
    # (L2/3) reads its (object, pose) hypotheses through these; the module-level functions are the shared home.
    local_disps = staticmethod(local_disps)                  # a patch's neighbour-displacement vectors (the feature pose)
    apply_pose = staticmethod(apply_pose)                   # APPLY a (group element, translation) to a point cloud
