"""The hippocampal formation — the allocentric map + the head-direction GAIN FIELD (HIPPOCAMPUS.md).

Assembled from the SAME mechanisms the cortical column already has ("evolution in reverse"): allocentric position = the
column's grid cells (`encoders.GridEncoder`); the head-direction ring = a periodic `encoders.ScalarEncoder`; value =
`l6_sr.SuccessorFeatures` (added when wired); loop closure = L2/3 recognition (the sensed pose). The per-action
**efference copy** (the body-frame displacement + turn) is NOT owned here — it is **L5's** (`l5.efference`, the TBT motor/
efference layer); L6 READS it and applies the head-direction gain field. The ONE mechanism this module adds is the **gain
field** itself:

  * IN  (path integration): world displacement = **R(head-direction) · (one egocentric displacement)** — so a body-frame
    action (FORWARD) is learned ONCE and rotated to every heading; no per-heading operator, no coverage sweep.
  * OUT (navigation): rotate the goal vector by **−(head-direction)** into the body frame; advance if the goal is ahead,
    else TURN toward it (reorient-then-advance) — the same transform inverted.

**The belief is a POPULATION, not a point.** Position is a place-cell activity MAP over the board and head-direction a
ring-attractor bump; the *location* of each bump is the estimate, its *width/peakedness* IS the uncertainty (probabilistic
population codes — Ma, Beck, Latham & Pouget 2006). There is no scalar "confidence": reliability-weighted cue combination
FALLS OUT of superimposing two populations — the prior (path-integrated bump) and the sensory likelihood (the L2/3
evidence field, one bump per near-top hypothesis). A crisp unimodal sighting is a sharp likelihood → it dominates the
posterior; a diffuse or bimodal sighting is a flat/split likelihood → it barely moves the prior (path integration carries
through). A SYMMETRIC object (whose stabiliser orbit places it at ONE centroid but many headings) is thus a sharp POSITION
bump and a flat HEAD bump — position corrects, heading is carried — with no special case. An UNLEARNED action's motion is
unknown → the prior diffuses to uniform → the sighting is adopted (the old "adopt-fully" gain, now emergent).

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


# ----- population-code helpers: a place-cell activity MAP (2-D) + a head-direction RING (1-D) -------------------
def _norm(a: np.ndarray) -> np.ndarray:
    s = a.sum()
    return a / s if s > 0 else np.full_like(a, 1.0 / a.size)


def _roll_pad(P: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """Shift `P` by INTEGER (ox, oy) with zero fill (position is NOT periodic — the board has edges)."""
    n = P.shape[0]
    out = np.zeros_like(P)
    xs0, xs1 = max(0, -ox), n - max(0, ox)
    ys0, ys1 = max(0, -oy), n - max(0, oy)
    if xs1 > xs0 and ys1 > ys0:
        out[xs0 + ox:xs1 + ox, ys0 + oy:ys1 + oy] = P[xs0:xs1, ys0:ys1]
    return out


def _shift2d(P: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Bilinear (sub-cell) shift of the activity map by a continuous world displacement — path integration moves the bump."""
    ix, iy = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = dx - ix, dy - iy
    out = np.zeros_like(P)
    for ox, oy, w in ((ix, iy, (1 - fx) * (1 - fy)), (ix + 1, iy, fx * (1 - fy)),
                      (ix, iy + 1, (1 - fx) * fy), (ix + 1, iy + 1, fx * fy)):
        if w:
            out += w * _roll_pad(P, ox, oy)
    return out


def _blur2d(P: np.ndarray) -> np.ndarray:
    """One separable 3-tap diffusion pass — path-integration NOISE (the bump broadens = uncertainty grows)."""
    Q = 0.5 * P
    Q[1:] += 0.25 * P[:-1]
    Q[:-1] += 0.25 * P[1:]
    R = 0.5 * Q
    R[:, 1:] += 0.25 * Q[:, :-1]
    R[:, :-1] += 0.25 * Q[:, 1:]
    return R


def _com_peak(P: np.ndarray, r: int = 2):
    """Read out the bump's estimate = the peak plus a local centre-of-mass (sub-cell precision, robust to a minor second
    mode — a point BETWEEN two peaks is where the object is NOT, so we localise around the winning peak)."""
    ix, iy = np.unravel_index(int(np.argmax(P)), P.shape)
    n = P.shape[0]
    x0, x1, y0, y1 = max(0, ix - r), min(n, ix + r + 1), max(0, iy - r), min(n, iy + r + 1)
    W = P[x0:x1, y0:y1]
    s = W.sum()
    if s <= 0:
        return float(ix), float(iy)
    xs, ys = np.arange(x0, x1), np.arange(y0, y1)
    return float((W.sum(1) * xs).sum() / s), float((W.sum(0) * ys).sum() / s)


def _ring_shift(Q: np.ndarray, dth: float) -> np.ndarray:
    """Rotate the head-direction ring bump by `dth` (periodic bilinear — the ring attractor path-integrates the turn)."""
    n = len(Q)
    sh = dth / (2 * np.pi) * n
    k = int(np.floor(sh))
    f = sh - k
    return (1 - f) * np.roll(Q, k) + f * np.roll(Q, k + 1)


def _ring_blur(Q: np.ndarray) -> np.ndarray:
    return 0.5 * Q + 0.25 * np.roll(Q, 1) + 0.25 * np.roll(Q, -1)


class Hippocampus:
    """The allocentric belief as a POPULATION `(place-cell map, head ring)`, path-integrated by the gain field and updated
    by sensing via superposition (the sensory evidence field × the prior). `observe(action, sighting)` LEARNS each action's
    effect and combines the populations; `path_integrate(action)` dead-reckons; `navigate(goal, actions)` reorients-then-
    advances. `pos`/`head` are READ-OUTS of the two bumps (the peak); there is no scalar-confidence register anywhere."""

    def __init__(self, l5, board: int = 64, scales=(7, 11, 13, 17), n_head: int = 8, gain: float = 0.3,
                 n_ring: int = 36, sig_pos: float = 1.2, sig_head: float = np.pi / 8):
        self.l5 = l5                                        # the EFFERENCE-COPY source (L5); L6 keeps NO copy of the operator
        self.board = int(board)
        self.grid = GridEncoder(scales=scales, dims=2, mw=3, bounds=[(0, board - 1)] * 2)   # allocentric position (grid cells)
        self.hd = ScalarEncoder(0.0, 2 * np.pi, n=int(n_head), w=1, periodic=True)          # head-direction ring (SDR readout)
        self.n_ring = int(n_ring)                            # the head-direction BELIEF ring resolution (finer than the SDR)
        self.sig_pos = float(sig_pos)                        # place-field / sensory-bump width (cells) — the tuning width, NOT a per-frame conf
        self.sig_head = float(sig_head)                      # head-direction tuning width (radians)
        self._ring_ang = -np.pi + (np.arange(self.n_ring) + 0.5) * 2 * np.pi / self.n_ring
        self._P = np.full((self.board, self.board), 1.0 / self.board ** 2)   # position place-cell map [ix, iy]
        self._Q = np.full(self.n_ring, 1.0 / self.n_ring)    # head-direction ring bump
        self._set = False                                    # has a sighting seeded the belief yet? (cold start until then)

    def reset(self) -> None:
        """A level boundary: drop the belief (the world resets; do not path-integrate across it)."""
        self._P = np.full((self.board, self.board), 1.0 / self.board ** 2)
        self._Q = np.full(self.n_ring, 1.0 / self.n_ring)
        self._set = False

    def _eff(self, action):
        """L5's efference for `action` as `(body_displacement_vector, dtheta)` — the operator L6 path-integrates by."""
        (dx, dy), dth = self.l5.efference(action)
        return np.array([dx, dy], float), float(dth)

    # ----- read-outs of the two population bumps ----------------------------------------------------------------
    @property
    def pos(self):
        """The allocentric position estimate = the place-cell map's peak, or None before the belief is seeded."""
        return _com_peak(self._P) if self._set else None

    @property
    def head(self) -> float:
        """The head-direction estimate = the ring bump's circular mean, or 0 before the belief is seeded."""
        if not self._set:
            return 0.0
        return float(np.arctan2((self._Q * np.sin(self._ring_ang)).sum(), (self._Q * np.cos(self._ring_ang)).sum()))

    def here(self):
        """The allocentric position belief, or None."""
        return self.pos

    # ----- the sensory likelihood: a bump per near-top L2/3 hypothesis (the evidence FIELD) ---------------------
    def _pos_like(self, sighting) -> np.ndarray:
        """The position likelihood = superposition of a place-field bump at each sighting mode's centroid, evidence-weighted.
        Symmetric poses share ONE centroid → a single sharp peak; a bimodal recognition → two peaks → a flat/split
        likelihood that pulls weakly. `sighting` = `[((x, y), theta, weight), ...]`."""
        xs = np.arange(self.board)
        L = np.zeros((self.board, self.board))
        for (cx, cy), _th, w in sighting:
            gx = np.exp(-((xs - cx) ** 2) / (2 * self.sig_pos ** 2))
            gy = np.exp(-((xs - cy) ** 2) / (2 * self.sig_pos ** 2))
            L += w * np.outer(gx, gy)
        return _norm(L + 1e-9)

    def _head_like(self, sighting) -> np.ndarray:
        """The head-direction likelihood = superposition of a ring bump at each mode's heading, evidence-weighted. A mode
        with `theta is None` carries NO heading information (an orientation-less view, e.g. a single cell) → it contributes
        nothing, so an all-None sighting leaves a FLAT likelihood and the heading path-integrates (no false correction)."""
        L = np.zeros(self.n_ring)
        for _c, th, w in sighting:
            if th is None:
                continue
            d = (self._ring_ang - th + np.pi) % (2 * np.pi) - np.pi
            L += w * np.exp(-(d ** 2) / (2 * self.sig_head ** 2))
        return _norm(L + 1e-9)

    # ----- IN: path integration by the gain field ---------------------------------------------------------------
    def _predict_shift(self, action, was_learned: bool) -> None:
        """PREDICT: move the population by the action (L5's efference copy, in effect BEFORE this step) and add motion
        noise. A LEARNED action → shift the bump by `R(head)·ego` + a small diffusion; an UNLEARNED one → the motion is
        unknown, so the prior collapses to UNIFORM (the sighting is then adopted — the old adopt-fully gain, emergent)."""
        if action is not None and was_learned:
            ego, dth = self._eff(action)
            world = _rot(self.head) @ ego
            self._P = _norm(_blur2d(_shift2d(self._P, world[0], world[1])))
            self._Q = _norm(_ring_blur(_ring_shift(self._Q, dth)))
        elif action is not None:                             # unknown motion → maximal uncertainty (a flat prior)
            self._P = np.full_like(self._P, 1.0 / self._P.size)
            self._Q = np.full_like(self._Q, 1.0 / self._Q.size)
        else:                                                # no action (a static observation) → mild diffusion only
            self._P = _norm(_blur2d(self._P))
            self._Q = _norm(_ring_blur(self._Q))

    def path_integrate(self, action):
        """Dead-reckon the belief by `action` (L5's efference copy) — MUTATES the population. Returns `(pos, head)`, or
        None before the belief is seeded."""
        if not self._set:
            return None
        self._predict_shift(action, self.l5.learned(action) if action is not None else False)
        return self.pos, self.head

    def predict(self, action):
        """PURE forward query (§5): the belief estimate `(pos, head)` AFTER `action`, WITHOUT mutating the population.
        None if the belief is unseeded. (A point read-out of the predicted bump — for the residual / navigation / attention.)"""
        if not self._set:
            return None
        ego, dth = self._eff(action)
        p = self.pos
        return tuple(np.asarray(p, float) + _rot(self.head) @ ego), _wrap(self.head + dth)

    # ----- LEARN (the gain field, inverted) + UPDATE (superimpose the sensory likelihood) -----------------------
    def observe(self, action, sighting) -> None:
        """The perceptual update: LEARN the action's effect from the sensed transition, PATH-INTEGRATE the prior, then
        SUPERIMPOSE the sensory likelihood (Bayesian product of populations — reliability weighting is emergent, HIPPOCAMPUS.md
        H4/H5). `sighting` = the L2/3 evidence field `[((x, y), theta, weight), ...]` (empty / None = no sighting → predict
        only; the belief broadens and path integration carries it). The dominant mode learns L5's efference (de-rotated to the
        body frame — one sighting at any heading pins FORWARD for all headings). No `gain`, no `conf`: a sharp likelihood
        dominates, a flat/split one barely moves the prior — the bump widths do the weighting."""
        sighting = list(sighting) if sighting else []
        prev_pos, prev_head = self.pos, self.head           # the estimate BEFORE the update (for efference learning)
        was_learned = action is not None and self.l5.learned(action)
        if not self._set:                                   # cold start: the first sighting SEEDS the population
            if sighting:
                self._P, self._Q, self._set = self._pos_like(sighting), self._head_like(sighting), True
            return
        self._predict_shift(action, was_learned)            # PREDICT (path integration by the efference in effect before now)
        dom = max(sighting, key=lambda m: m[2]) if sighting else None
        if action is not None and dom is not None and dom[1] is not None and prev_pos is not None:   # LEARN from the dominant ORIENTED sighting
            (sx, sy), sth, _w = dom                          # (an orientation-less view cannot supply the turn dth → don't learn the operator from it)
            body = _rot(-prev_head) @ np.asarray((sx - prev_pos[0], sy - prev_pos[1]), float)   # the gain field, inverted
            self.l5.learn_efference(action, (body[0], body[1]), _wrap(sth - prev_head))
        if sighting:                                        # UPDATE: superimpose the sensory likelihood (Bayes: prior × likelihood)
            self._P = _norm(self._P * self._pos_like(sighting))
            self._Q = _norm(self._Q * self._head_like(sighting))

    def controllable(self) -> bool:
        """Does some learned action DISPLACE the position? (the tracked body responds to actions) — read from L5's efference."""
        return any((dx * dx + dy * dy) ** 0.5 > 0.5 for (dx, dy, _dth) in self.l5.eff.values())

    def moves(self, actions):
        """The actions whose learned effect is a MOVE (a large body displacement) — the kind emerges from L5's efference."""
        return [a for a in actions if float(np.linalg.norm(self._eff(a)[0])) > 0.5]

    def turns(self, actions):
        """The actions whose learned effect is a TURN (a heading increment, negligible displacement)."""
        return [a for a in actions if abs(self._eff(a)[1]) > 1e-2 and float(np.linalg.norm(self._eff(a)[0])) <= 0.5]

    # ----- the SDR codes (reused by value / routing) ------------------------------------------------------------
    def location_sdr(self, pos):
        """The allocentric position as a grid-cell SDR (the code `SuccessorFeatures` values / the thalamus broadcasts)."""
        return self.grid.encode((float(pos[0]), float(pos[1]))).dense()

    def head_sdr(self, theta):
        """The head-direction as a ring SDR (the head-direction bump)."""
        return self.hd.encode(float(theta)).dense()

    # ----- OUT: reorient-then-advance via the inverse gain field ------------------------------------------------
    def navigate(self, goal, actions):
        """The action that best moves toward `goal` — the inverse gain field. If the body can TURN, reorient-then-advance:
        face the goal (the world heading `atan2(Δ)`), then advance; otherwise (a translation-only body) pick the world-move
        whose displacement best aligns with the goal vector. Returns the action, or None at the goal."""
        here = self.pos
        if here is None:
            return None
        gv = np.asarray(goal, float) - np.asarray(here, float)
        dist = float(np.hypot(*gv))
        if dist < 1e-9:
            return None
        u = gv / dist
        head = self.head
        moves, turns = self.moves(actions), self.turns(actions)

        def world_disp(a):                                  # the action's predicted WORLD displacement from the current pose
            return _rot(head) @ self._eff(a)[0]

        if not turns:                                       # ABELIAN body (translations): the move best aligned with the goal
            return max(moves, key=lambda a: float(world_disp(a) @ u), default=None)
        ideal = float(np.arctan2(gv[1], gv[0]))             # the world heading that points at the goal
        # ADVANCE when the goal is within the forward cone; else TURN toward it. The cone is > the discrete-heading
        # half-step (π/4 for 4 headings) so a goal exactly between two headings still advances (which then zig-zags in)
        # rather than oscillating between the two equidistant turns.
        if moves and abs(_wrap(ideal - head)) < np.pi / 3:  # facing the goal (within the forward cone) -> ADVANCE
            return max(moves, key=lambda a: float(world_disp(a) @ u))
        return min(turns, key=lambda a: abs(_wrap(head + self._eff(a)[1] - ideal)))   # else TURN toward the goal
