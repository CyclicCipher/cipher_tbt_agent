"""The cortical column — a CONTAINER for the four layers and a thin COORDINATOR (rule 3). Math + state live in the
layers / modules; the column only routes.

  L6  LOCATION / value — successor features (`sf`) over the allocentric grid-cell SDR. The allocentric POSE belief
      (position + head-direction) lives in the HIPPOCAMPUS module (`self.hip`), path-integrated by the head-direction
      GAIN FIELD (HIPPOCAMPUS.md); the column FEEDS it recognition + efference (entorhinal-IN) and READS the belief back
      (thalamus-OUT). It is NOT re-implemented here (that was the leaked conjunctive per-heading operator, now retired).
  L5  DISPLACEMENT / motor / driver.
  L4  FEATURE-at-location — the label-free content codebook.
  L23 OBJECT / identity — graph-memory + evidence recognition (pose inferred, symmetry quotiented) + voting.

Perception (`perceive`): L2/3 recognises the mover's pose → the hippocampus LEARNS its gain field from the transition and
CORRECTS the allocentric belief → the forward-model residual is `_pred_error`. Navigation delegates to the hippocampus's
inverse gain field (reorient-then-advance). No conjunctive pose code, no per-heading operator, no rollout, no tabular graph.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from .l4_feature_location import L4_FeatureLocation
from .l5_displacement import L5_Displacement
from .l6_sr import SuccessorFeatures
from .l23_object import L23_Object
from .hippocampus import Hippocampus
from .retina import connected_figure, view_signature


class CorticalColumn(nn.Module):
    def __init__(self, n_entities, feat_dim=256, d_mem=512, seed=0, board=64, throttle=20, **_):
        super().__init__()
        self.L5 = L5_Displacement()
        self.L4 = L4_FeatureLocation(n_entities, feat_dim=feat_dim, seed=seed)
        self.L23 = L23_Object(feat_dim=feat_dim, d_mem=d_mem)
        self.board = int(board)
        self.hip = Hippocampus(self.L5, board=board)         # the allocentric map + head-direction gain field (reads L5's efference copy)
        self.sf = SuccessorFeatures(d=self.hip.grid.n)       # value over the allocentric position SDR (the grid-cell code)
        self._pred_error = 0.0
        self._prev_cells: dict = {}                          # the previous frame's feature-at-locations {(x,y): colour} (for common-fate bootstrap)
        self._self_cells = None                              # the last self body's world cells (the top-down footprint the model predicts forward)

    # ----- perception: recognise → feed the hippocampus (entorhinal-IN) → read the belief back -------------------
    def perceive(self, action, coloured_cells):
        """The unified TBT perception step (§5) — the column GROUPS the frame's feature-at-locations into the SELF by
        COMMON FATE, then recognises its pose and corrects the location, returning `(feature, (x, y, theta))`. No sensor
        segmentation (TBP: object membership emerges HERE from displacement/prediction, not a connected-component blob).
        L2/3 recognition reads the pose; the **hippocampus** learns its gain field from the transition (the efference copy
        `action`) and corrects the allocentric belief (§1c entorhinal-IN); `_pred_error` is the residual (§4c).
        `coloured_cells` = the frame's non-background cells `[(x, y, colour), ...]`, ungrouped."""
        self_cells = self._attend_self(action, coloured_cells)     # COMMON FATE: the cells whose motion the efference explains
        cells = [(float(x), float(y)) for (x, y, _c) in self_cells]
        content = view_signature(self_cells)                        # the peripheral's opaque content id (colour signature)
        field = self._sense_field(cells)                           # L2/3 recognition → the sensory EVIDENCE FIELD (a population)
        self._pred_error = self._residual(action, field)          # the forward-model residual (§4c) BEFORE the update
        self.hip.observe(action, field)                            # PATH-INTEGRATE the prior + SUPERIMPOSE the likelihood (no gain/conf)
        feat = self.L4.encode(content)                             # the opaque content → an L4 feature id (SDR)
        p = self.hip.here()
        x, y = p if p is not None else (0.0, 0.0)
        return feat, (float(x), float(y), float(self.hip.head))

    def _attend_self(self, action, coloured_cells, tol: float = 1.6):
        """Group the frame's feature-at-locations into the SELF by TOP-DOWN FIGURE-GROUND (reference_tbt_segmentation_and_
        grouping): the recognised object MODEL predicts its own FOOTPRINT — the self is the coloured cells that fall on that
        predicted footprint. The footprint is the LAST self body carried forward by the efference (rigid motion: rotate about
        its centroid by the turn, translate by the world move `R(head)·ego`), so a TURN grabs the WHOLE rigid body (the
        anchor cell that stayed included), NOT a fragment — the failure mode of a bottom-up 'changed cells' or radius blob.
        A differently-placed object (a marker) is off the footprint → excluded, no connected-components. Bootstrap / re-
        acquire (no body yet, a BLOCKED action, or the prediction missed): the connected body the cells MOVED into reveals
        the self (common fate + cohesion). Both cues are computed and the MORE COMPLETE body wins, so a turn's partial
        top-down match is superseded by the full connected body, while a blocked (unmoved) body is still held by top-down."""
        now = {(int(round(x)), int(round(y))): c for (x, y, c) in coloured_cells}
        prev = self._prev_cells
        self._prev_cells = now
        fp = self._predicted_footprint(action)
        if fp is not None:                                       # TOP-DOWN (priority once a model exists): the predicted footprint IS the figure
            sel = [(x, y, now[(x, y)]) for (x, y) in now
                   if float(np.min(np.abs(fp - np.array([x, y])).sum(axis=1))) <= tol]
            if len(sel) >= max(1, len(fp) - 1):                 # grabbed essentially the whole predicted body → keep the model-consistent grouping
                self._self_cells = [(float(x), float(y)) for (x, y, _c) in sel]
                return sel
        moved = {(x, y) for (x, y) in now if (x, y) not in prev} if prev else set()   # BOOTSTRAP / RE-ACQUIRE: common fate
        if moved:
            figure = connected_figure(now, moved)                # COHESION: complete the moved seed into the whole moving body (the static marker is a separate figure)
        else:                                                    # no motion: seed only a SINGLE unambiguous body (boundedness); else defer to motion
            figure = connected_figure(now, {next(iter(now))})
            if figure != set(now):                               # several disconnected bodies and nothing moved → cannot tell self from ground → hold
                return []
        if not figure:
            return []
        sel = [(x, y, now[(x, y)]) for (x, y) in figure]
        self._self_cells = [(float(x), float(y)) for (x, y, _c) in sel]
        return sel

    def _predicted_footprint(self, action):
        """The self body's predicted world cells next frame — the LAST self-cells under the efference's rigid motion (rotate
        about the body centroid by the turn `dth`, translate by the world move `R(head)·ego`). None before a body is tracked
        or the belief is unseeded. This is the top-down prediction that segments figure from ground."""
        cells = getattr(self, "_self_cells", None)
        if not cells or not self.hip._set:
            return None
        pts = np.asarray(cells, float)
        ego, dth = self.hip._eff(action) if action is not None else (np.zeros(2), 0.0)
        from .hippocampus import _rot
        world = _rot(self.hip.head) @ np.asarray(ego, float)
        c = pts.mean(axis=0)
        return (pts - c) @ _rot(dth).T + c + world               # rigid body motion of the footprint

    def track_reset(self):
        """A level boundary: drop the pose belief AND the previous-frame store (the board resets)."""
        self.hip.reset()
        self._prev_cells = {}
        self._self_cells = None

    def forward(self, action, content):
        """The ONE forward prediction (§5) — a PURE query via the hippocampus (`predict`); the content is INVARIANT under
        self-motion (reafference). Returns `((x,y,theta), content)`, or `(None, content)` before the belief is set."""
        pred = self.hip.predict(action)
        if pred is None:
            return None, content
        (x, y), th = pred
        return (float(x), float(y), float(th)), content

    def here_position(self):
        """The allocentric position of the belief (from the hippocampus), or None."""
        return self.hip.here()

    def predicted_position(self, action):
        """The forward model's predicted next LOCATION (§5) — where L5's efference path-integrates the belief to. None
        before the belief is set. (Perception uses it to find the REAFFERENT change — the one consistent with the efference.)"""
        p = self.hip.predict(action)
        return p[0] if p is not None else None

    def controllable(self) -> bool:
        """Does some learned action MOVE the location? (the tracked mover responds to actions) — the hippocampus knows."""
        return self.hip.controllable()


    # ----- the SDR location value + the navigator (delegated to the hippocampus's inverse gain field, §8) --------
    def location_sdr(self, pos):
        """The location SDR φ for a position — the grid-cell code the SF values (the hippocampus's grid)."""
        return self.hip.location_sdr(pos)

    def learn_location_value(self, pos, pos_next, reward: float = 0.0):
        """One SF TD update for pos->pos_next carrying `reward` (>0 appetitive; <0 aversive) — generalises via SDR overlap."""
        self.sf.observe(self.location_sdr(pos), self.location_sdr(pos_next), reward)

    def location_value(self, pos):
        """V(pos) = the SF value over the location SDR — expected discounted future reward, generalising over position."""
        return self.sf.value(self.location_sdr(pos))

    def navigate_vector(self, here, goal, actions, lam: float = 1.0):
        """The ONE navigator (§8) — delegated to the hippocampus's **inverse gain field**: rotate the goal vector by
        −(head-direction) and reorient-then-advance (HIPPOCAMPUS.md). `here` is the belief the hippocampus already holds.
        (SF-value modulation for cost-avoidance is a follow-up — the value is learned via `learn_location_value` meanwhile.)"""
        return self.hip.navigate(goal, actions)

    # ----- L5 output + L2/3 recognition routing -----------------------------------------------------------------
    def motor(self, action):
        return self.L5.motor(action)

    def learn_object(self, cloud, name=None):
        return self.L23.learn(cloud, name=name)

    def recognize_object(self, cloud):
        return self.L23.recognize(cloud)

    def identify_object(self, cloud):
        return self.L23.identify(cloud)

    def content_code(self, label):
        return self.L4.E[label]

    def _sense_field(self, cloud):
        """PERCEIVE the mover as a sensory EVIDENCE FIELD (a population, not a point) — the near-top L2/3 (object, pose)
        hypotheses, each reported as `((x, y) centroid, theta, weight)`. L2/3 recognises the shape and SOLVES its pose;
        the hippocampus superimposes these bumps on its prior, so the SHAPE of this field (sharp / flat / bimodal) IS the
        sighting's reliability — no scalar confidence (§2 L2/3). The CENTROID of the pose-placed cloud (not the raw
        hypothesis translation) is the position: it is pose-invariant, so a symmetric object's tied poses share ONE
        centroid — a sharp position bump — while its headings stay ambiguous (a flat head bump). `[]` when nothing is
        recognised (→ the belief holds and path-integrates)."""
        if not cloud:
            return []
        if self.recognize_object(list(cloud)) is None:
            return []
        hyps = self.L23.hyps
        if not hyps:
            return []
        oriented = len(cloud) >= 2                             # a view of < 2 cells carries NO orientation → it cannot constrain heading
        top = max(h.ev for h in hyps)
        field = []
        for h in hyps:
            if top - h.ev >= 1.0:                              # keep the near-top (competing) hypotheses = the evidence population
                continue
            c = np.mean(np.asarray(h.obj.cells_at(h.theta, h.t), float), axis=0)   # the pose-invariant centroid = the position
            theta = float(h.theta) if oriented else None       # None = heading UNCONSTRAINED → a flat head population (path-integrate the heading)
            field.append(((float(c[0]), float(c[1])), theta, float(np.exp(h.ev - top))))
        return field

    def _residual(self, action, field):
        """The forward-model residual (§4c): the dominant sighting vs the efference-PREDICTED pose, as a fraction of the
        move. High when the operator is unlearned (identity predicts staying → the sighting misses); ~0 once it converges."""
        here = self.hip.here()
        if not field or here is None or action is None:
            return 0.0
        (sx, sy), _th, _w = max(field, key=lambda m: m[2])
        pred = self.hip.predict(action)
        if pred is None:
            return 0.0
        (px, py), _ph = pred
        r = ((sx - px) ** 2 + (sy - py) ** 2) ** 0.5
        m = ((px - here[0]) ** 2 + (py - here[1]) ** 2) ** 0.5
        return min(1.0, r / (m + 1.0))
