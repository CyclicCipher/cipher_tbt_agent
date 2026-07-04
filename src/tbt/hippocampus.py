"""The hippocampal formation — the allocentric map + the head-direction GAIN FIELD (HIPPOCAMPUS.md).

Assembled from the SAME mechanisms the cortical column already has ("evolution in reverse"): allocentric position = the
column's grid cells (`encoders.GridEncoder`); the head-direction ring = a periodic `encoders.ScalarEncoder`; value =
`l6_sr.SuccessorFeatures` (added when wired); loop closure = L2/3 recognition (the sensed pose). The ONE new mechanism is
the **gain field**:

  * IN  (path integration): world displacement = **R(head-direction) · (one egocentric displacement)** — so a body-frame
    action (FORWARD) is learned ONCE and rotated to every heading; no per-heading operator, no coverage sweep.
  * OUT (navigation): rotate the goal vector by **−(head-direction)** into the body frame; advance if the goal is ahead,
    else TURN toward it (reorient-then-advance) — the same transform inverted.

Dimension-general: `R` is SO(2) here, SO(3) later — a rotation of the displacement, NO centre of rotation
(`reference_operator_as_group_representation`). The action KINDS (move vs turn) EMERGE from the learned effect (a move has
a large egocentric displacement; a turn a large heading increment) — not hand-coded (the bitter lesson). Separate from the
column's object-centric L6: this is the GLOBAL, world-anchored frame. Pure numpy + the SDR encoders (torch-free)."""

from __future__ import annotations

import numpy as np

from .encoders import GridEncoder, ScalarEncoder


def _rot(theta: float) -> np.ndarray:
    """The SO(2) rotation by `theta` (the gain-field's head-direction rotation; SO(3) is the same shape in 3-D)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _wrap(a: float) -> float:
    """Wrap an angle to (−π, π] — the head-direction ring is periodic."""
    return (float(a) + np.pi) % (2 * np.pi) - np.pi


class Hippocampus:
    """The allocentric belief `(position, head-direction)` path-integrated by the gain field and reset by sensing.

    `observe(action, sensed_pos, sensed_head)` LEARNS each action's effect (one egocentric displacement + one heading
    increment, a running mean) and CORRECTS the belief; `path_integrate(action)` dead-reckons by the gain field;
    `navigate(goal, actions)` returns the action that reorients-then-advances toward `goal`."""

    def __init__(self, board: int = 64, scales=(7, 11, 13, 17), n_head: int = 8, lr: float = 0.5, gain: float = 0.3):
        self.board = int(board)
        self.grid = GridEncoder(scales=scales, dims=2, mw=3, bounds=[(0, board - 1)] * 2)   # allocentric position (grid cells)
        self.hd = ScalarEncoder(0.0, 2 * np.pi, n=int(n_head), w=1, periodic=True)          # head-direction ring
        self.lr = float(lr)
        self.gain = float(gain)                              # the CORRECTION (Kalman) gain once the action is learned — how much the SIGHTING corrects the path-integrated belief
        self.pos = None                                     # allocentric position belief (x, y), or None
        self.head = 0.0                                     # head-direction belief (radians)
        self.ego: dict = {}                                 # action -> learned egocentric displacement (ONE vector)
        self.dtheta: dict = {}                              # action -> learned heading increment (the turn)

    def reset(self) -> None:
        """A level boundary: drop the belief (the world resets; do not path-integrate across it)."""
        self.pos, self.head = None, 0.0

    # ----- the SDR codes (reused by value / routing) ------------------------------------------------------------
    def location_sdr(self, pos):
        """The allocentric position as a grid-cell SDR (the code `SuccessorFeatures` values / the thalamus broadcasts)."""
        return self.grid.encode((float(pos[0]), float(pos[1]))).dense()

    def head_sdr(self, theta):
        """The head-direction as a ring SDR (the head-direction bump)."""
        return self.hd.encode(float(theta)).dense()

    # ----- IN: path integration by the gain field ---------------------------------------------------------------
    def path_integrate(self, action):
        """Dead-reckon the belief by `action` (the efference copy): rotate the ONE egocentric displacement by the current
        head-direction and add to position (the gain field), then advance the head-direction ring by the turn. Returns
        `(position, head)`, or None before the belief is set."""
        if self.pos is None:
            return None
        ego = self.ego.get(action)
        if ego is not None:
            self.pos = tuple(np.asarray(self.pos, float) + _rot(self.head) @ ego)   # world Δ = R(head)·ego
        self.head = _wrap(self.head + self.dtheta.get(action, 0.0))                  # angular path integration
        return self.pos, self.head

    def predict(self, action):
        """PURE forward query (§5): the belief `(position, head)` AFTER `action`, WITHOUT mutating it. None if unset."""
        if self.pos is None:
            return None
        ego = self.ego.get(action)
        pos = tuple(np.asarray(self.pos, float) + _rot(self.head) @ ego) if ego is not None else self.pos
        return pos, _wrap(self.head + self.dtheta.get(action, 0.0))

    # ----- LEARN (the gain field, inverted) + CORRECT (path-integrate, reliability-weighted) --------------------
    def observe(self, action, sensed_pos, sensed_head) -> None:
        """LEARN the action's effect from the sensed transition, then CORRECT the belief — RELIABILITY-WEIGHTED, not a hard
        snap (HIPPOCAMPUS.md H4/H5). The belief is PATH-INTEGRATED by the action (the gain field) and only NUDGED toward the
        sighting by a Kalman `gain`: place cells path-integrate (direction-invariant), landmarks correct DRIFT, not every
        frame (Etienne/Jeffery; the reliability weighting is Kalman — PLOS CB 2021). So a turn-in-place keeps the position
        stable and an orientation-dependent / wobbling recognised anchor AVERAGES OUT instead of jerking the position. The
        gain is reliability-adaptive: an UNLEARNED action's prediction is worthless → adopt the sighting fully (gain 1); a
        LEARNED action's prediction is trusted → correct gently (`self.gain`). The egocentric displacement is the world move
        DE-ROTATED by the head-direction (`R(−head)·Δworld`) — one sighting at any heading pins FORWARD for all headings."""
        if sensed_pos is None or sensed_head is None:
            return
        sx, sy, sh = float(sensed_pos[0]), float(sensed_pos[1]), _wrap(float(sensed_head))
        if self.pos is None:                                # cold start: adopt the first sighting
            self.pos, self.head = (sx, sy), sh
            return
        (px, py), ph = self.predict(action) if action is not None else (self.pos, self.head)   # PREDICT (established operator)
        reliable = action is not None and action in self.ego   # was the operator ESTABLISHED *before* this observation?
        if action is not None:                              # LEARN the action's effect from the observed transition
            dworld = np.asarray((sx - self.pos[0], sy - self.pos[1]), float)
            ego = _rot(-self.head) @ dworld                 # de-rotate to the body frame (the gain field, inverted)
            dth = _wrap(sh - self.head)
            self.ego[action] = ego if action not in self.ego else (1 - self.lr) * self.ego[action] + self.lr * ego
            self.dtheta[action] = dth if action not in self.dtheta else (1 - self.lr) * self.dtheta[action] + self.lr * dth
        g = self.gain if reliable else 1.0                  # reliability-adaptive: adopt an unlearned action's sighting fully
        self.pos = ((1 - g) * px + g * sx, (1 - g) * py + g * sy)               # CORRECT (Kalman blend toward the sighting)
        self.head = _wrap(ph + g * _wrap(sh - ph))

    def here(self):
        """The allocentric position belief, or None."""
        return self.pos

    def controllable(self) -> bool:
        """Does some learned action DISPLACE the position? (the tracked body responds to actions)."""
        return any(float(np.linalg.norm(e)) > 0.5 for e in self.ego.values())

    def moves(self, actions):
        """The actions whose learned effect is a MOVE (a large egocentric displacement) — the kind emerges from experience."""
        return [a for a in actions if float(np.linalg.norm(self.ego.get(a, (0.0, 0.0)))) > 0.5]

    def turns(self, actions):
        """The actions whose learned effect is a TURN (a heading increment, negligible displacement)."""
        return [a for a in actions
                if abs(self.dtheta.get(a, 0.0)) > 1e-2 and float(np.linalg.norm(self.ego.get(a, (0.0, 0.0)))) <= 0.5]

    # ----- OUT: reorient-then-advance via the inverse gain field ------------------------------------------------
    def navigate(self, goal, actions):
        """The action that best moves toward `goal` — the inverse gain field. If the body can TURN, reorient-then-advance:
        face the goal (the world heading `atan2(Δ)`), then advance; otherwise (a translation-only body) pick the world-move
        whose displacement best aligns with the goal vector. Returns the action, or None at the goal."""
        if self.pos is None:
            return None
        gv = np.asarray(goal, float) - np.asarray(self.pos, float)
        dist = float(np.hypot(*gv))
        if dist < 1e-9:
            return None
        u = gv / dist
        moves, turns = self.moves(actions), self.turns(actions)

        def world_disp(a):                                  # the action's predicted WORLD displacement from the current pose
            ego = self.ego.get(a)
            return _rot(self.head) @ ego if ego is not None else np.zeros(2)

        if not turns:                                       # ABELIAN body (translations): the move best aligned with the goal
            return max(moves, key=lambda a: float(world_disp(a) @ u), default=None)
        ideal = float(np.arctan2(gv[1], gv[0]))             # the world heading that points at the goal
        # ADVANCE when the goal is within the forward cone; else TURN toward it. The cone is > the discrete-heading
        # half-step (π/4 for 4 headings) so a goal exactly between two headings still advances (which then zig-zags in)
        # rather than oscillating between the two equidistant turns.
        if moves and abs(_wrap(ideal - self.head)) < np.pi / 3:    # facing the goal (within the forward cone) -> ADVANCE
            return max(moves, key=lambda a: float(world_disp(a) @ u))
        return min(turns, key=lambda a: abs(_wrap(self.head + self.dtheta[a] - ideal)))   # else TURN toward the goal
