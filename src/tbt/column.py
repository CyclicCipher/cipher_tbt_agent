"""The cortical column — a CONTAINER for the four layers and a thin COORDINATOR of information flow. Math + state
live in the layers; the column only routes (ARCHITECTURE.md rule 3).

  L6  LOCATION / value — successor features (`sf`) over the grid-cell SDR (`loc_enc`), path-integrated by the
      operator (the pose belief `_pose`). The ONE location/value substrate (the localist OnlineSR retired, §10 P4a).
  L5  DISPLACEMENT / operator / motor / driver — the per-action operator (`pose_ops`, learned by `perceive`).
  L4  FEATURE-at-location — the label-free content codebook (`encode`).
  L23 OBJECT / identity — graph-memory + evidence recognition (pose inferred, symmetry quotiented) + voting.

Navigation is the goal-oriented VECTOR field (`navigate_vector`: the grid-cell goal vector modulated by the SF
value, §8). Perception is `perceive` (recognize → predict → correct → learn, §5). No tabular graph, no cost field,
no Euclidean achiever, no state-node — those retired with the SDR re-seat.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from .l4_feature_location import L4_FeatureLocation
from .l5_displacement import L5_Displacement
from .l6_sr import SuccessorFeatures                # successor features over the SDR (the ONE L6 value/predictive map)
from .encoders import GridEncoder                    # the location SDR encoder (grid cells; §2 L6)
from .l23_object import L23_Object                    # the object/identity layer: graph-memory + recognition + voting
from .operator import Operator                        # the per-action operator (composable; learned online)


class CorticalColumn(nn.Module):
    def __init__(self, n_entities, feat_dim=256, d_mem=512, seed=0, board=64, **_):
        super().__init__()
        self.L5 = L5_Displacement()                                       # displacement / operator / motor / driver
        self.L4 = L4_FeatureLocation(n_entities, feat_dim=feat_dim, seed=seed)  # feature-at-location + content codebook
        self.L23 = L23_Object(feat_dim=feat_dim, d_mem=d_mem)             # object/identity: graph-memory + recognition
        self.loc_enc = GridEncoder(scales=(7, 11, 13, 17), dims=2, mw=3, bounds=[(0, board - 1)] * 2)  # the LOCATION SDR (§2 L6)
        self.sf = SuccessorFeatures(d=self.loc_enc.n)         # L6 value/predictive map over the SDR (successor features; generalises)
        self._pose = None                                    # the POSE belief (SE(2) matrix) path-integrated by the operator
        self.pose_ops: dict = {}                             # per-action body-frame SE(2) operator (learned by `perceive`)
        self._pred_error = 0.0                               # the forward model's last prediction error (§4c)

    # ----- the SDR location + successor-features value + the goal-oriented VECTOR navigator (§2 L6, §8) --------
    def location_sdr(self, pos):
        """The location SDR φ for a metric position (the pose translation) — the grid-cell code (§2 L6): the currency
        the SF values. Semantic overlap: nearby positions share bits (generalisation)."""
        return self.loc_enc.encode((float(pos[0]), float(pos[1]))).dense()

    def learn_location_value(self, pos, pos_next, reward: float = 0.0):
        """One SF TD update for the observed move pos->pos_next carrying `reward` (>0 appetitive; <0 aversive cost — the
        ONE signed value, §3). Value generalises across nearby positions via SDR overlap (unlike the localist SR)."""
        self.sf.observe(self.location_sdr(pos), self.location_sdr(pos_next), reward)

    def location_value(self, pos):
        """V(pos) = the SF value over the location SDR — expected discounted future reward incl. cost, GENERALISING to
        positions not individually visited. The modulation the vector navigator warps by."""
        return self.sf.value(self.location_sdr(pos))

    def navigate_vector(self, here, goal, actions, lam: float = 1.0):
        """The ONE navigator (§8) — the goal-oriented VECTOR FIELD. ATTRACTION = the grid-cell goal vector (the unit
        displacement `goal − here`, the direction grid cells encode — Bush, Barry & Burgess 2015), scored by how well
        each action's operator displacement advances along it. MODULATION = the SF value at the destination (reward
        pulls, learned cost/barrier repels — de Cothi & Barry). NOT greedy value-ascent (the periodic grid traps it —
        `project_sf_value_not_greedy_navigable`). Returns the best action, or None at the goal."""
        vx, vy = goal[0] - here[0], goal[1] - here[1]
        norm = (vx * vx + vy * vy) ** 0.5
        if norm < 1e-9:
            return None                                                    # already at the goal
        ux, uy = vx / norm, vy / norm
        best_a, best = None, float("-inf")
        for a in actions:
            m = np.asarray(self.operator(a).M, dtype=float)                # the ONE perceive-learned operator; its translation = the displacement
            dx, dy = float(m[0, 2]), float(m[1, 2])                         # clean now the stabiliser pins the symmetric mover's pose (abelian nav)
            dest = (here[0] + dx, here[1] + dy)
            score = (dx * ux + dy * uy) + lam * self.location_value(dest)   # grid-cell vector attraction + SF value modulation
            if score > best:
                best, best_a = score, a
        return best_a

    # ----- L5 output (the motor + the thalamus driver) ----------------------------------------------
    def motor(self, action):
        """The MOTOR output — the enacted action (L5 is the cortex's output layer; name->GameAction is the motor organ)."""
        return self.L5.motor(action)

    def driver(self, symbol, action):
        """The feed-forward DRIVER message to other columns (via the higher-order thalamus) — L5's trans-thalamic output."""
        return self.L5.driver(symbol, action)

    # ----- L2/3 recognition routing (the object library lives in the layer, not here) ---------------
    def learn_object(self, cloud, name=None):
        """Add an object to L2/3's graph-memory (named → store; else learn online, label-free)."""
        return self.L23.learn(cloud, name=name)

    def recognize_object(self, cloud):
        """Identify a sensed cloud's (name, theta, t, evidence) at a pose never seen, learning it online if novel —
        pose-invariant recognition (the symmetry is quotiented in L23.best)."""
        return self.L23.recognize(cloud)

    def identify_object(self, cloud):
        """Recognise a sensed shape against L2/3's library WITHOUT adding a new one — the name, or None."""
        return self.L23.identify(cloud)

    def content_code(self, label):
        """This column's content (What / L4) code for an entity label — the thalamus interface."""
        return self.L4.E[label]

    # ----- the location belief + pose path integration (§2 L6: efference predict, recognition correct) ----------
    def track_reset(self):
        """A level boundary: drop the pose belief (the board resets; do not path-integrate across it)."""
        self._pose = None

    def operator(self, action) -> Operator:
        """The ONE per-action OPERATOR: the learned SE(2) POSE operator (`pose_ops[action]`) when one was learned — the
        non-abelian general case — else the abelian TRANSLATION (`L5.operator`, the commuting special case). Its
        PRESENCE selects the form; no game-type gate."""
        op = self.pose_ops.get(action)
        return op if op is not None else self.L5.operator(action)

    def _operator_gens(self):
        """The learned per-action operators as a generator list (for `discover_relations` / `factor_dynamics`)."""
        keys = sorted(set(self.pose_ops) | set(self.L5.move_delta))
        return [self.operator(a) for a in keys]

    def path_integrate(self, action):
        """Path-integrate the location belief by the action's operator — `location ← operator(action)·location` (§2 L6):
        the EFFERENCE-driven PREDICT half of predict-then-correct. ONE mechanism for abelian and non-abelian. Returns
        (x, y, theta)."""
        return self.track_pose(self.operator(action))

    def perceive(self, action, cells, content):
        """The unified TBT perception step (§5) — recognize → PREDICT → CORRECT → LEARN → read content, returning
        `(feature, pose)`. ONE path for abelian and non-abelian:
          PREDICT   — dead-reckon the pose belief by the action's operator (`path_integrate`, the efference copy);
          OBSERVE   — recognize the mover's pose from its `cells` (L2/3 `sense_heading`, symmetry quotiented);
          COMPARE   — the forward model's ERROR = PREDICTED vs OBSERVED pose, as a fraction of the move (`_pred_error`);
          CORRECT   — snap the belief to the observed pose (`sense_pose`);
          LEARN     — the per-action operator = the observed transformation (`learn_pose_op`);
          CONTENT   — encode the OPAQUE content descriptor to an L4 feature id (an SDR).
        `cells` = the mover's cells (colour-blind → the pose). `content` = the OPAQUE peripheral descriptor; the column
        is content-opaque (no colour here)."""
        cells = [(float(p[0]), float(p[1])) for p in cells]
        pose_before = self._pose.copy() if self._pose is not None else None
        before_pos = (float(pose_before[0, 2]), float(pose_before[1, 2])) if pose_before is not None else None
        if action is not None and self._pose is not None:            # PREDICT (efference)
            self.path_integrate(action)
        predicted_pos = (float(self._pose[0, 2]), float(self._pose[1, 2])) if self._pose is not None else None
        self.sense_heading(cells)                                    # OBSERVE: recognize the pose (L2/3)
        self._pred_error = 0.0                                        # the forward model's prediction error (§4c)
        if getattr(self, "_heading_reliable", False):                # a reliable sighting -> COMPARE + CORRECT + LEARN
            sx, sy = self._sensed_pos                                # the RECOGNIZED position (L2/3), not a hand-computed centroid (rule 5)
            if predicted_pos is not None and before_pos is not None and action is not None:   # COMPARE: predicted vs observed
                r = ((sx - predicted_pos[0]) ** 2 + (sy - predicted_pos[1]) ** 2) ** 0.5       # residual
                m = ((predicted_pos[0] - before_pos[0]) ** 2 + (predicted_pos[1] - before_pos[1]) ** 2) ** 0.5   # move size
                self._pred_error = min(1.0, r / (m + 1.0))          # residual as a FRACTION of the move (0 = nailed, ~1 = missed)
            self.sense_pose(sx, sy, self._heading)                  # CORRECT
            if pose_before is not None and action is not None:
                self.learn_pose_op(action, pose_before, self._pose)  # LEARN
        elif self._pose is None:                                     # no reliable view yet -> cold start at identity
            self.track_pose_reset()
        feat = self.L4.encode(content)                              # the opaque content -> an L4 feature id (SDR); no colour in the column
        x, y = float(self._pose[0, 2]), float(self._pose[1, 2])
        th = float(np.arctan2(self._pose[1, 0], self._pose[0, 0]) % (2 * np.pi))
        return feat, (x, y, th)

    def forward(self, action, content):
        """The ONE forward prediction (§5) — predict the next observation given the current pose + action + content:
        apply the operator to the LOCATION and read the CONTENT there. A PURE query (does NOT mutate the belief).
        CONTENT — self-motion: the mover's own view is INVARIANT under its own motion (reafference), so it is unchanged.
        Returns `(predicted (x, y, theta), predicted_content)`, or `(None, content)` before the belief is set."""
        if self._pose is None:
            return None, content
        p = self._pose @ np.asarray(self.operator(action).M, dtype=float)   # pure: operator applied to a COPY, no assign
        return (float(p[0, 2]), float(p[1, 2]), float(np.arctan2(p[1, 0], p[0, 0]) % (2 * np.pi))), content

    def here_position(self):
        """The RAW metric position of the location belief (the pose translation) — the navigator's coordinate frame.
        `None` before the belief is set."""
        return (float(self._pose[0, 2]), float(self._pose[1, 2])) if self._pose is not None else None

    def controllable(self) -> bool:
        """Does the learned per-action OPERATOR move things? A non-trivial translation OR rotation for some action means
        the tracked mover responds to actions (its location is informative). A non-controllable in-place animation's
        per-action deltas average to ~identity -> False."""
        for op in self.pose_ops.values():
            m = np.asarray(op.M, dtype=float)
            if abs(m[0, 2]) + abs(m[1, 2]) > 0.5 or abs(m[0, 0] - 1.0) + abs(m[1, 0]) > 0.05:
                return True                                          # a non-trivial translation OR rotation
        return False

    def track_pose_reset(self):
        """Reset the POSE belief (an SE(2) matrix) to the identity (origin, heading 0)."""
        self._pose = np.eye(3)

    def track_pose(self, op: Operator):
        """Path-integrate the POSE by RIGHT-composing the body-frame action operator: `P ← P·G`. Because composition is
        NON-COMMUTATIVE, the pose distinguishes HEADINGS (non-abelian); a pure translation op is the abelian special
        case. Returns (x, y, theta)."""
        if self._pose is None:
            self._pose = np.eye(3)
        self._pose = self._pose @ np.asarray(op.M, dtype=float)
        return (float(self._pose[0, 2]), float(self._pose[1, 2]),
                float(np.arctan2(self._pose[1, 0], self._pose[0, 0]) % (2 * np.pi)))

    def learn_pose_op(self, action, pose_before, pose_after, rate: float = 0.4):
        """LEARN the per-action body-frame SE(2) operator ONLINE. The body-frame increment `G = pose_before⁻¹·pose_after`
        is CONSTANT per action, so EWMA it and RE-PROJECT to SE(2) (orthogonalise the 2x2 rotation block)."""
        before = np.asarray(pose_before, dtype=float)
        after = np.asarray(pose_after, dtype=float)
        g = np.linalg.inv(before) @ after
        cur = self.pose_ops.get(action)
        m = g if cur is None else (1.0 - rate) * cur.M + rate * g
        u, _, vt = np.linalg.svd(m[:2, :2])                  # re-project the rotation block to O(2) (Procrustes)
        m = m.copy()
        m[:2, :2] = u @ vt
        m[2] = [0.0, 0.0, 1.0]
        self.pose_ops[action] = Operator(m)
        return self.pose_ops[action]

    def sense_heading(self, cloud):
        """PERCEIVE heading from the mover's SHAPE (the unified recognition path): L2/3 recognises the object and SOLVES
        its pose theta (symmetry quotiented in `L23.best`). Learns the shape online on first sight. A PARTIAL view is
        UNRELIABLE: hold the heading and flag it (`_heading_reliable=False`). Returns the heading belief."""
        if not cloud:
            self._heading_reliable = False
            return getattr(self, "_heading", 0.0)
        self._obj_size = max(getattr(self, "_obj_size", 0), len(cloud))    # the full shape = the most cells ever seen
        res = self.recognize_object(list(cloud)) if len(cloud) >= self._obj_size else None
        if res is None:                                                    # a partial / degenerate cloud -> hold
            self._heading_reliable = False
            return getattr(self, "_heading", 0.0)
        _name, theta, t, _ev = res
        self._heading = float(theta)
        self._sensed_pos = (float(t[0]), float(t[1]))                      # the recognizer's anchor (symmetry-stable via L23.best)
        self._heading_reliable = True
        return self._heading

    def sense_pose(self, x, y, theta):
        """CORRECT the pose belief to an OBSERVED pose: snap `_pose` to the recognized (x, y, heading). Complements
        `path_integrate` (dead-reckon from efference) as the correct half of predict-then-correct."""
        c, s = np.cos(theta), np.sin(theta)
        self._pose = np.array([[c, -s, float(x)], [s, c, float(y)], [0.0, 0.0, 1.0]])
        return (float(x), float(y), float(theta) % (2 * np.pi))

    # ----- relation / factor discovery over the learned operators (P3) ------------------------------
    def discover_relations(self, tol: float = 1e-6, max_elements: int = 64):
        """DISCOVER the RELATIONS among the LEARNED per-action operators (the QUOTIENT of the free monoid on them) by
        loop closure under predictive sufficiency (`operator.discover_group`) — the finite Cayley graph a geodesic
        planner searches. Returns (elements, relations)."""
        from .operator import discover_group
        gens = self._operator_gens()
        if not gens:
            return [], []
        return discover_group(gens, tol=tol, max_elements=max_elements)

    def factor_dynamics(self, tol: float = 1e-6, max_elements: int = 256):
        """Factor the LEARNED dynamics into a DIRECT PRODUCT of cyclic factors (`operator.factor_group`) — counters /
        toggles / patrols as cycles. Returns the [(generator_index, period)] factors, or None if not a finite product."""
        from .operator import factor_group
        gens = self._operator_gens()
        return factor_group(gens, tol=tol, max_elements=max_elements) if gens else None

    def content_operator(self, shape, action):
        """The CONTENT operator for `(shape, action)`: L5's `recolor` transition map as a permutation `Operator`, so
        content dynamics are a factorable part of the model (a TOGGLE is a 2-cycle). Returns `(Operator, alphabet)`, or
        None if no content transition is learned."""
        from .operator import permutation_operator
        mapping = self.L5.recolor.get((shape, action))
        return permutation_operator(mapping) if mapping else None
