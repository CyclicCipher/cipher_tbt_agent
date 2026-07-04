"""The cortical column — a CONTAINER for the four layers and a thin COORDINATOR (rule 3). Math + state live in the
layers; the column only routes.

  L6  LOCATION / value — successor features (`sf`) over the grid-cell SDR (`loc_enc`); the pose belief `_pose_code` (a
      CONJUNCTIVE position ⊗ heading SDR) path-integrated by a LEARNED group-representation operator (§2 L6, Gao 2021).
  L5  DISPLACEMENT / motor / driver.
  L4  FEATURE-at-location — the label-free content codebook.
  L23 OBJECT / identity — graph-memory + evidence recognition (pose inferred, symmetry quotiented) + voting.

The operator is a learned matrix `M(action)` acting on the pose code (`action_ops`, a `ModularOperator`) — dimension-
general, non-abelian by construction (matrix product), NO center of rotation, NO SE(2) matrix on (x,y,θ). Path
integration is `φ ← M(action)·φ`; recognition corrects by snapping the code to the observed pose. Navigation is the
goal-oriented VECTOR field over the code (heading-aware, so a TURN then advances). No tabular graph, no cost field.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from .l4_feature_location import L4_FeatureLocation
from .l5_displacement import L5_Displacement
from .l6_sr import SuccessorFeatures
from .encoders import GridEncoder, ScalarEncoder, ConjunctiveEncoder, SDR
from .l23_object import L23_Object
from .operator import Operator, ModularOperator


class CorticalColumn(nn.Module):
    def __init__(self, n_entities, feat_dim=256, d_mem=512, seed=0, board=64, throttle=20, **_):
        super().__init__()
        self.L5 = L5_Displacement()
        self.L4 = L4_FeatureLocation(n_entities, feat_dim=feat_dim, seed=seed)
        self.L23 = L23_Object(feat_dim=feat_dim, d_mem=d_mem)
        # ONE bump-SDR position code (mw=3): its OVERLAP serves the SF value AND its rigid bump-shift serves the operator
        # (the continuous-attractor code -- Burak & Fiete; a delta would need two encoders, a bump unifies them).
        self.board = int(board)
        self.loc_enc = GridEncoder(scales=(7, 11, 13, 17), dims=2, mw=3, bounds=[(0, board - 1)] * 2)
        self.head_enc = ScalarEncoder(0.0, 2 * np.pi, n=4, w=1, periodic=True)                        # heading (a small ring)
        self.pose_enc = ConjunctiveEncoder([("pos", self.loc_enc), ("head", self.head_enc)])          # the conjunctive pose code (SE(2) rep space)
        self.sf = SuccessorFeatures(d=self.loc_enc.n)        # value over position (the same bump code)
        self._pose_code = None                               # the POSE belief, path-integrated by the block-diagonal operator
        self.action_ops: dict = {}                           # action -> ModularOperator (block-diagonal per pose-module; coverage-robust)
        self._I = np.eye(self.pose_enc.n)
        self._pred_error = 0.0

    # ----- the pose code + the learned operator (path integration = φ ← M(action)·φ, §2 L6) ---------------------
    def _apply_op(self, action, phi):
        """Apply the learned operator to the pose code BLOCK-WISE (identity if unlearned) -- the ModularOperator's
        per-module Procrustes (coverage-robust, cheap; no full-matrix assembly)."""
        oo = self.action_ops.get(action)
        return oo.apply(phi) if oo is not None else np.asarray(phi, dtype=float)

    def _decode_pose(self):
        """(position, heading) decoded from the belief code, or None."""
        if self._pose_code is None:
            return None
        active = np.where(np.asarray(self._pose_code) > 0.5)[0]
        if len(active) == 0:
            return None
        d = self.pose_enc.decode(SDR(self.pose_enc.n, active))
        return d["pos"], d["head"]

    def _decode_pos_near(self, phi, center, radius: int = 5):
        """Decode the POSITION from a pose code `phi`, resolved to the alias NEAREST `center`. A grid SDR is decodable only
        within `lcm(scales)`; a predicted move that runs off the board (x = −2 at the edge) otherwise decodes to a spurious
        CRT ALIAS (e.g. x=32), which a navigator reads as huge progress and chases forever. A single action moves only a
        few cells, so decoding within a small window around `center` picks the true destination. Returns None if empty."""
        active = np.where(np.asarray(phi) > 0.5)[0]
        if len(active) == 0:
            return None
        rest_n = self.head_enc.n
        pos_active = {int(c) // rest_n for c in active}                # project the conjunctive code onto the position field
        cx, cy, b = int(round(center[0])), int(round(center[1])), self.board - 1
        bounds = [(max(0, cx - radius), min(b, cx + radius)), (max(0, cy - radius), min(b, cy + radius))]
        try:
            return self.loc_enc.decode(SDR(self.loc_enc.n, pos_active), bounds=bounds)
        except Exception:
            return None

    def _dest_after(self, action, here):
        """The position after `action` from the CURRENT pose code (heading-aware) — decode `M(action)·φ` near `here` (a move
        is small, so the alias nearest `here` is the true destination); `here` if unknown."""
        if self._pose_code is None or action not in self.action_ops:
            return here
        dest = self._decode_pos_near(self._apply_op(action, self._pose_code), here)
        return dest if dest is not None else here

    def path_integrate(self, action):
        """PREDICT: path-integrate the belief by the action's operator — `φ ← M(action)·φ` (the efference copy). Returns
        (position, heading), or None before the belief is set."""
        if self._pose_code is None:
            return None
        self._pose_code = self._apply_op(action, self._pose_code)
        return self._decode_pose()

    def perceive(self, action, cells, content):
        """The unified TBT perception step (§5) — recognize → PREDICT → CORRECT → LEARN → content, returning
        `(feature, (x, y, theta))`:
          PREDICT — dead-reckon the pose code by the action's operator (`path_integrate`);
          OBSERVE — recognize the mover's pose from `cells` (L2/3 `sense_heading`, symmetry quotiented);
          COMPARE — the forward model's error (predicted vs observed position, as a fraction of the move; `_pred_error`);
          CORRECT — snap the code to the observed pose (`pose_enc.encode`);
          LEARN   — `M(action)` from the observed transition (`ModularOperator.observe`)."""
        cells = [(float(p[0]), float(p[1])) for p in cells]
        code_before = None if self._pose_code is None else np.asarray(self._pose_code).copy()
        dp = self._decode_pose(); pb = dp[0] if dp else None
        if action is not None and self._pose_code is not None:       # PREDICT (efference)
            self.path_integrate(action)
        dp = self._decode_pose(); pp = dp[0] if dp else None
        self.sense_heading(cells)                                    # OBSERVE
        self._pred_error = 0.0
        if getattr(self, "_heading_reliable", False):
            sx, sy = self._sensed_pos
            observed = self.pose_enc.encode({"pos": (sx, sy), "head": self._heading}).dense().astype(float)
            if pp is not None and pb is not None and action is not None:   # COMPARE
                r = ((sx - pp[0]) ** 2 + (sy - pp[1]) ** 2) ** 0.5
                m = ((pp[0] - pb[0]) ** 2 + (pp[1] - pb[1]) ** 2) ** 0.5
                self._pred_error = min(1.0, r / (m + 1.0))
            self._pose_code = observed                              # CORRECT
            if code_before is not None and action is not None:      # LEARN M(action): code_before -> observed
                self.action_ops.setdefault(action, ModularOperator(self.pose_enc.module_grids(), self.pose_enc.n)).observe(code_before, observed)
        feat = self.L4.encode(content)                             # the opaque content -> an L4 feature id (SDR)
        dp = self._decode_pose()
        (x, y), th = dp if dp else ((0.0, 0.0), 0.0)
        return feat, (float(x), float(y), float(th))

    def forward(self, action, content):
        """The ONE forward prediction (§5) — a PURE query: apply the operator to the pose code, decode the next pose. The
        content is INVARIANT under self-motion (reafference). Returns `((x,y,theta), content)`, or `(None, content)`."""
        if self._pose_code is None:
            return None, content
        active = np.where((self._apply_op(action, self._pose_code)) > 0.5)[0]
        if len(active) == 0:
            return None, content
        d = self.pose_enc.decode(SDR(self.pose_enc.n, active))
        (x, y), th = d["pos"], d["head"]
        return (float(x), float(y), float(th)), content

    def here_position(self):
        """The position of the belief (decoded from the pose code), or None."""
        dp = self._decode_pose()
        return dp[0] if dp is not None else None

    def controllable(self) -> bool:
        """Does some learned operator MOVE the location? -> the tracked mover responds to actions (its location informs).
        Read from the operators' learned displacement (belief-independent), not the current pose (which may sit at a
        boundary where every predicted move clamps)."""
        return any(self.can_move_at(h) for h in self.headings_seen())

    def track_reset(self):
        """A level boundary: drop the pose belief (the board resets; do not path-integrate across it)."""
        self._pose_code = None

    def headings_seen(self):
        """The heading slices in which SOME action has produced an observed transition (the reachable-heading set so far)."""
        seen = set()
        for oo in self.action_ops.values():
            seen |= oo.rest_seen()
        return seen

    def can_move_at(self, h: int) -> bool:
        """From heading `h`, does SOME learned action displace the location? (else the body would be stuck facing that way)."""
        return any(oo.moves_at(h) for oo in self.action_ops.values())

    # ----- the SDR location value + the goal-oriented VECTOR navigator (§8) --------------------------------------
    def location_sdr(self, pos):
        """The location SDR φ for a position — the grid-cell code the SF values."""
        return self.loc_enc.encode((float(pos[0]), float(pos[1]))).dense()

    def learn_location_value(self, pos, pos_next, reward: float = 0.0):
        """One SF TD update for pos->pos_next carrying `reward` (>0 appetitive; <0 aversive) — generalises via SDR overlap."""
        self.sf.observe(self.location_sdr(pos), self.location_sdr(pos_next), reward)

    def location_value(self, pos):
        """V(pos) = the SF value over the location SDR — expected discounted future reward, generalising over position."""
        return self.sf.value(self.location_sdr(pos))

    def navigate_vector(self, here, goal, actions, lam: float = 1.0):
        """The ONE navigator (§8) — the goal-oriented VECTOR FIELD, HEADING-AWARE via the operator with a DEPTH-2 rollout
        (a sparing deliberation, `reference_brain_planning`). A single step is MYOPIC on a non-abelian body: a TURN leaves
        the position unchanged, so by immediate displacement it ties an unhelpful perpendicular FORWARD and the agent never
        turns to face the goal. Scoring each action by the BEST position reachable ONE MORE step later (orient-then-advance)
        breaks the tie — a turn that ENABLES a goalward FORWARD wins, and the turn DIRECTION is disambiguated by which
        follow-up advances toward the goal. Abelian (translations) is the special case (the 2nd step just doubles the
        aligned move — same argmax). ATTRACTION = the displacement's alignment with the goal vector; MODULATION = the SF
        value. Returns the best action, or None at the goal."""
        vx, vy = goal[0] - here[0], goal[1] - here[1]
        norm = (vx * vx + vy * vy) ** 0.5
        if norm < 1e-9 or self._pose_code is None:
            return None
        ux, uy = vx / norm, vy / norm

        def progress(p):
            return (p[0] - here[0]) * ux + (p[1] - here[1]) * uy + lam * self.location_value(p)

        best_a, best = None, float("-inf")
        for a1 in actions:
            phi1 = self._apply_op(a1, self._pose_code)
            reach = [phi1] + [self._apply_op(a2, phi1) for a2 in actions]     # orient with a1, then the best advance
            s = max((progress(p) for p in (self._decode_pos_near(ph, here, radius=6) for ph in reach) if p is not None),
                    default=float("-inf"))
            if s > best:
                best, best_a = s, a1
        return best_a

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

    # ----- relation / factor discovery over the learned operators (P3) ------------------------------------------
    def operator(self, action) -> Operator:
        """The learned operator for `action` (identity if unlearned) — for relation/factor discovery."""
        oo = self.action_ops.get(action)
        return oo.operator() if oo is not None else Operator(self._I)

    def _operator_gens(self):
        """The learned per-action operators as a generator list (for `discover_relations` / `factor_dynamics`)."""
        return [self.action_ops[a].operator() for a in sorted(self.action_ops)]

    def discover_relations(self, tol: float = 1e-6, max_elements: int = 64):
        """DISCOVER the RELATIONS among the learned operators by loop closure (`operator.discover_group`). NB the operators
        are only defined on the code SUBSPACE (Procrustes fills the null space arbitrarily), so relation-discovery needs a
        code-subspace comparison — the current whole-matrix closure under-closes for the SDR operator (a known gap)."""
        from .operator import discover_group
        gens = self._operator_gens()
        if not gens:
            return [], []
        return discover_group(gens, tol=tol, max_elements=max_elements)

    def factor_dynamics(self, tol: float = 1e-6, max_elements: int = 256):
        """Factor the learned dynamics into a DIRECT PRODUCT of cyclic factors (`operator.factor_group`)."""
        from .operator import factor_group
        gens = self._operator_gens()
        return factor_group(gens, tol=tol, max_elements=max_elements) if gens else None

    def content_operator(self, shape, action):
        """The CONTENT operator for `(shape, action)`: L5's `recolor` map as a permutation `Operator`."""
        from .operator import permutation_operator
        mapping = self.L5.recolor.get((shape, action))
        return permutation_operator(mapping) if mapping else None
