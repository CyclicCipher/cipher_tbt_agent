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
from .l4 import L4Layer
from .l5_displacement import L5_Displacement
from .l6_sr import SuccessorFeatures
from .l23_object import L23_Object, ObjectGraph, _canonical
from .hippocampus import Hippocampus, _rot
from .encoders import SDR, CategoryEncoder, GridEncoder
from .retina import connected_figure, view_sdr


class CorticalColumn(nn.Module):
    def __init__(self, n_entities, feat_dim=256, d_mem=512, seed=0, board=64, throttle=20, **_):
        super().__init__()
        self.L5 = L5_Displacement()
        self.L4 = L4_FeatureLocation(n_entities, feat_dim=feat_dim, seed=seed)
        self.L4map = L4Layer(cells_per_column=8, activation_threshold=6, min_threshold=4, init_perm=0.55)   # the feature-at-location content map (MICROCIRCUIT Stage 4)
        self._feat_enc = CategoryEncoder(range(16), w=8, capacity=16)                                       # colour → feature minicolumns
        self._objloc_enc = GridEncoder(scales=(5, 7, 11), dims=2, mw=1, bounds=[(-24, 24), (-24, 24)])      # the OBJECT-FRAME location code (crisp: exact under translation)
        self.L23 = L23_Object()
        self.board = int(board)
        self.hip = Hippocampus(self.L5, board=board)         # the allocentric map + head-direction gain field (reads L5's efference copy)
        self.sf = SuccessorFeatures(d=self.hip.grid.n)       # value over the allocentric position SDR (the grid-cell code)
        self._pred_error = 0.0
        self._content_burst = 0.0                            # the fraction of self-cells whose colour was UNPREDICTED at its location (content surprise, §3.4)
        self._prev_cells: dict = {}                          # the previous frame's feature-at-locations {(x,y): colour} (for common-fate bootstrap)
        self._self_cells = None                              # the last self body's world cells (the top-down footprint the model predicts forward)
        self._tracked = None                                 # the currently CONVERGED object identity (persistent recognition session, §10-A)
        self._cand = None                                    # a provisional candidate object — committed only when it CONVERGES across a frame

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
        feat = view_sdr(self_cells)                                # the peripheral's OVERLAP-BEARING content SDR (M3; similar views share bits)
        field = self._sense_field(cells)                           # L2/3 recognition → the sensory EVIDENCE FIELD (a population)
        self._pred_error = self._residual(action, field)          # the forward-model residual (§4c) BEFORE the update
        self.hip.observe(action, field)                            # PATH-INTEGRATE the prior + SUPERIMPOSE the likelihood (no gain/conf)
        self._bind_content(self_cells, self._object_pose())        # bind colour-at-object-frame-location in L4 (the content map; additive, MICROCIRCUIT Stage 4)
        p = self.hip.here()
        x, y = p if p is not None else (0.0, 0.0)
        return feat, (float(x), float(y), float(self.hip.head))

    # ----- L4 feature-at-location content map (MICROCIRCUIT Stage 4): content prediction / imagination + burst --------
    def _object_pose(self):
        """The recognised object's pose `(theta, t)` from L2/3 (M4), or None — the anchoring that maps a WORLD cell into
        the object's own frame (`R(-theta)·(cell - t)`), so the content map is keyed by feature-at-OBJECT-frame-location."""
        rec = self.L23.best()
        return (float(rec[1]), rec[2]) if rec is not None and rec[2] is not None else None

    def _obj_frame(self, wx, wy, pose):
        """A world cell → its integer OBJECT-FRAME location under `pose=(theta, t)` (the L4 map's key)."""
        theta, t = pose
        ofl = _rot(-theta) @ (np.array([float(wx), float(wy)]) - np.asarray(t, float))
        return (int(round(float(ofl[0]))), int(round(float(ofl[1]))))

    def _bind_content(self, self_cells, pose):
        """Bind each self-cell's COLOUR at its OBJECT-FRAME location in L4 (the feature-at-location map), and record the
        BURST fraction (colours sensed where a DIFFERENT colour was learned = content surprise, §3.4). No-op until a pose is
        recognised (needs the object frame). Additive: touches only `L4map` + `_content_burst`, never the localization loop."""
        if pose is None or not self_cells:
            return
        burst = 0
        for (wx, wy, colour) in self_cells:
            loc = self._objloc_enc.encode(self._obj_frame(wx, wy, pose)).active
            self.L4map.observe(self._feat_enc.encode(int(colour)).active, loc)
            burst += int(self.L4map.burst())
        self._content_burst = burst / len(self_cells)

    def predict_content_at_world(self, wx, wy, pose=None):
        """IMAGINATION (§3.3, object permanence): the COLOUR L4 predicts at world cell `(wx, wy)` — map it into the object
        frame (the current recognised pose unless given) and query the L4 content map WITHOUT sensing it. None if unmapped
        (the map does not cover that object-frame location). A pure query; does not mutate the map."""
        pose = pose or self._object_pose()
        if pose is None:
            return None
        cols = self.L4map.predict_feature(self._objloc_enc.encode(self._obj_frame(wx, wy, pose)).active)
        return self._feat_enc.decode(SDR(self._feat_enc.n, cols)) if cols else None

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
        """A level boundary: drop the pose belief AND the previous-frame store (the board resets). The object LIBRARY
        persists (the same objects recur next level), but the tracked identity + candidate re-acquire by settling."""
        self.hip.reset()
        self._prev_cells = {}
        self._self_cells = None
        self._tracked = None
        self._cand = None

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
        """PERCEIVE the mover as a sensory EVIDENCE FIELD (a population, not a point), via the PERSISTENT recognition
        session (§2 L2/3, §10-A): recognition is a SETTLING process, and the recognizer's pruning IS convergence — a
        hypothesis that keeps matching survives, one that stops matching is pruned; surviving hyps ⟺ CONVERGED, none ⟺
        a burst (using only the sensor tolerance, no novelty threshold). Three cases, no scalar/heuristic gate:
          1. TRACK — verify the currently-tracked identity ALONE (no library scan). If it still converges, keep it.
          2. SETTLE — else recognize against the whole library; if a hypothesis converges, that IS the object (track it).
          3. NON-CONVERGENCE (novelty) — else a provisional CANDIDATE must CONVERGE across a frame before it is committed:
             a rigid object moves predictably so its candidate re-explains the next frame (→ commit, a real object); an
             unstable multi-object blob never does (→ never committed, so the library stays bounded). Meanwhile the belief
             still gets a POSITION sighting (the cloud CENTROID; recognition only adds ORIENTATION).
        The field's centroid (pose-invariant) is the position; a symmetric object's tied poses share ONE centroid (sharp
        position bump) while its headings stay ambiguous (flat head bump)."""
        if not cloud:
            return []
        L = self.L23
        if self._tracked is not None:                          # 1. TRACK the converged identity (verify it alone)
            L._sense_shape(list(cloud), only=[self._tracked])
            if L.hyps:
                return self._field_from_hyps(cloud)
        L._sense_shape(list(cloud))                            # 2. SETTLE over the whole library
        if L.hyps:
            self._tracked = max(L.hyps, key=lambda h: h.ev).obj   # a hypothesis converged → that is the object
            self._cand = None
            return self._field_from_hyps(cloud)
        if self._cand is not None:                             # 3. NON-CONVERGENCE: the candidate must re-explain THIS frame to be committed
            L._sense_shape(list(cloud), only=[self._cand])
            if L.hyps:                                         # it converged across the frame → a real, trackable object → COMMIT
                self._tracked = L._learn_canonical(list(cloud))
                self._cand = None
                L._sense_shape(list(cloud), only=[self._tracked])
                return self._field_from_hyps(cloud)
        self._cand = ObjectGraph("cand", _canonical(list(cloud)), L.radius)   # (re)seed the provisional candidate from this cloud
        c = np.mean(np.asarray([(float(x), float(y)) for (x, y) in cloud], float), axis=0)
        return [((float(c[0]), float(c[1])), None, 1.0)]        # POSITION-only sighting (no orientation → flat head)

    def _field_from_hyps(self, cloud):
        """Build the sensory evidence field from the surviving (converged) hypotheses — the near-top population, each a
        `((x, y) centroid, theta, weight)`; theta is None for a <2-cell view (no orientation → a flat head population)."""
        hyps = self.L23.hyps
        if not hyps:
            return []
        oriented = len(cloud) >= 2
        top = max(h.ev for h in hyps)
        field = []
        for h in hyps:
            if top - h.ev >= 1.0:                              # keep the near-top (competing) hypotheses = the evidence population
                continue
            c = np.mean(np.asarray(h.obj.cells_at(h.theta, h.t), float), axis=0)   # the pose-invariant centroid = the position
            theta = float(h.theta) if oriented else None
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
