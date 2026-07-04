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

import torch.nn as nn

from .l4_feature_location import L4_FeatureLocation
from .l5_displacement import L5_Displacement
from .l6_sr import SuccessorFeatures
from .l23_object import L23_Object
from .hippocampus import Hippocampus


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

    # ----- perception: recognise → feed the hippocampus (entorhinal-IN) → read the belief back -------------------
    def perceive(self, action, cells, content):
        """The unified TBT perception step (§5) — recognise → PREDICT → COMPARE → (LEARN+CORRECT) → content, returning
        `(feature, (x, y, theta))`. L2/3 recognition reads the mover's pose; the **hippocampus** learns its gain field from
        the transition (the efference copy `action`) and corrects the allocentric belief (HIPPOCAMPUS.md §1c entorhinal-IN);
        `_pred_error` is the predicted-vs-observed residual (§4c). The column stays a thin coordinator — the pose math is
        the hippocampus's."""
        cells = [(float(p[0]), float(p[1])) for p in cells]
        pb = self.hip.here()                                         # the belief BEFORE this observation
        self.sense_heading(cells)                                   # OBSERVE (L2/3 recognition → sensed pos + heading)
        self._pred_error = 0.0
        if getattr(self, "_heading_reliable", False):
            sx, sy = self._sensed_pos
            pred = self.hip.predict(action) if (action is not None and pb is not None) else None   # PREDICT (efference)
            if pred is not None:                                   # COMPARE (residual as a fraction of the move)
                (px, py), _th = pred
                r = ((sx - px) ** 2 + (sy - py) ** 2) ** 0.5
                m = ((px - pb[0]) ** 2 + (py - pb[1]) ** 2) ** 0.5
                self._pred_error = min(1.0, r / (m + 1.0))
            self.hip.observe(action, (sx, sy), self._heading)      # LEARN (gain field) + CORRECT (snap the belief)
        feat = self.L4.encode(content)                             # the opaque content → an L4 feature id (SDR)
        p = self.hip.here()
        x, y = p if p is not None else (0.0, 0.0)
        return feat, (float(x), float(y), float(self.hip.head))

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

    def controllable(self) -> bool:
        """Does some learned action MOVE the location? (the tracked mover responds to actions) — the hippocampus knows."""
        return self.hip.controllable()

    def track_reset(self):
        """A level boundary: drop the pose belief (the board resets; do not path-integrate across it)."""
        self.hip.reset()

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

    def driver(self, symbol, action):
        return self.L5.driver(symbol, action)

    def learn_object(self, cloud, name=None):
        return self.L23.learn(cloud, name=name)

    def recognize_object(self, cloud):
        return self.L23.recognize(cloud)

    def identify_object(self, cloud):
        return self.L23.identify(cloud)

    def content_code(self, label):
        return self.L4.E[label]

    def sense_heading(self, cloud):
        """PERCEIVE the pose from the mover's SHAPE (the unified recognition path): L2/3 recognises the object and SOLVES
        its pose theta + position (symmetry quotiented in `L23.best`). A PARTIAL view is UNRELIABLE: hold + flag it."""
        if not cloud:
            self._heading_reliable = False
            return getattr(self, "_heading", 0.0)
        self._obj_size = max(getattr(self, "_obj_size", 0), len(cloud))
        res = self.recognize_object(list(cloud)) if len(cloud) >= self._obj_size else None
        if res is None:
            self._heading_reliable = False
            return getattr(self, "_heading", 0.0)
        _name, theta, t, _ev = res
        self._heading = float(theta)
        self._sensed_pos = (float(t[0]), float(t[1]))
        self._heading_reliable = True
        return self._heading

    # ----- content dynamics (L5 recolor) as a permutation operator (P3; independent of the pose machinery) ------
    def content_operator(self, shape, action):
        """The CONTENT operator for `(shape, action)`: L5's `recolor` map as a permutation `Operator`."""
        from .operator import permutation_operator
        mapping = self.L5.recolor.get((shape, action))
        return permutation_operator(mapping) if mapping else None
