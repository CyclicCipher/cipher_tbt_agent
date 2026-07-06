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

**The belief is a POPULATION, and it is SDR-NATIVE (SDR_MIGRATION.md M1⊕M2).** Position is NOT a dense localist board-map;
it is the **grid-cell SDR** — a bump over each `GridEncoder` MODULE ring (a `(scale, axis)` continuous-attractor ring), so
the belief lives in the SAME grid code `SuccessorFeatures` values and the thalamus broadcasts. Head-direction is a
ring-attractor bump. The *location* of each bump is the estimate, its *width/peakedness* IS the uncertainty (probabilistic
population codes — Ma, Beck, Latham & Pouget 2006). There is no scalar "confidence": reliability-weighted cue combination
FALLS OUT of superimposing two populations — the prior (path-integrated bump) and the sensory likelihood (the L2/3
evidence field). A crisp unimodal sighting is a sharp likelihood → it dominates; a diffuse or bimodal one barely moves the
prior (path integration carries through). A SYMMETRIC object (one centroid, many headings) → a sharp POSITION bump and a
flat HEAD bump — position corrects, heading is carried — with no special case. An UNLEARNED action's motion is unknown →
the prior diffuses to uniform → the sighting is adopted.

**Path integration = the OPERATOR on the grid SDR** (`φ ← M(v_world)·φ`): each module ring is rigidly SHIFTED by the world
velocity (Burak & Fiete 2009's continuous-attractor bump move), a translation being a per-module PHASE shift and a turn a
ring-shift of the HD SDR. The gain field reads the head as a **population VECTOR** (the ring's circular-mean = the standard
HD decode) to turn the body-frame efference into ONE world velocity `v_world = R(head)·ego`; it is NOT a marginalisation
over the head population (that would smear an abelian world-frame move across every rotation — SDR_MIGRATION.md M1). For a
world-frame (abelian) body the head is flat → `R(0)=identity` (no smear); for a sharp SE(2) head it is the true rotation.

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


# ----- population-code helpers: a bump over a periodic RING (a grid module, or the head-direction ring) --------
def _norm(a: np.ndarray) -> np.ndarray:
    s = a.sum()
    return a / s if s > 0 else np.full_like(a, 1.0 / a.size)


def _ring_shift(ring: np.ndarray, d_cells: float) -> np.ndarray:
    """Rigidly SHIFT a periodic ring bump by a CONTINUOUS `d_cells` (bilinear, wraps around the ring) — the
    continuous-attractor / grid phase-shift operator (Burak & Fiete). The abelian translation shifts every location
    module by the same world velocity; a turn shifts the head ring; sub-cell `d` interpolates (fractional path integration)."""
    n = len(ring)
    if n == 1:
        return ring.copy()
    k = int(np.floor(d_cells))
    f = d_cells - k
    return (1 - f) * np.roll(ring, k) + f * np.roll(ring, k + 1)


def _ring_blur(ring: np.ndarray) -> np.ndarray:
    """One 3-tap CIRCULAR diffusion pass — path-integration NOISE broadens the bump (uncertainty grows)."""
    n = len(ring)
    if n <= 2:
        return ring.copy()
    return 0.5 * ring + 0.25 * np.roll(ring, 1) + 0.25 * np.roll(ring, -1)


class Hippocampus:
    """The allocentric belief as a POPULATION `(grid-cell SDR position, head ring)`, path-integrated by the gain-field
    operator and updated by sensing via superposition (the sensory evidence field × the prior). `observe(action, sighting)`
    LEARNS each action's effect and combines the populations; `path_integrate(action)` dead-reckons; `navigate(goal,
    actions)` reorients-then-advances. `pos`/`head` are READ-OUTS of the bumps; there is no scalar-confidence register."""

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
        # the position belief = a bump over each grid MODULE ring `(scale, axis)`, in the SAME order `grid.encode` builds
        self._modinfo = [(int(s), axis) for s in self.grid.scales for axis in range(self.grid.dims)]
        self._pos = self._uniform_pos()                      # the grid-cell SDR population (per-module ring bumps)
        self._Q = np.full(self.n_ring, 1.0 / self.n_ring)    # head-direction ring bump
        self._set = False                                    # has a sighting seeded the belief yet? (cold start until then)
        self._readout = None                                 # cached decoded (x, y) — invalidated on any mutation
        self._anchor = None                                  # the CONTINUITY anchor (the path-integrated prediction) — a grid-cell bump is a continuous attractor; the read-out cannot teleport, so it decodes within a WINDOW of this (grid→place disambiguation by trajectory), breaking the CRT aliasing of independent module argmax
        self._win = 3                                        # the continuity-window half-width (cells): < the smallest scale, so no two positions inside it share every module phase → no aliasing within the window

    def _uniform_pos(self):
        return [np.full(s, 1.0 / s) for (s, _axis) in self._modinfo]

    def reset(self) -> None:
        """A level boundary: drop the belief (the world resets; do not path-integrate across it)."""
        self._pos = self._uniform_pos()
        self._Q = np.full(self.n_ring, 1.0 / self.n_ring)
        self._set = False
        self._readout = None
        self._anchor = None

    def _eff(self, action):
        """L5's efference for `action` as `(body_displacement_vector, dtheta)` — the operator L6 path-integrates by."""
        (dx, dy), dth = self.l5.efference(action)
        return np.array([dx, dy], float), float(dth)

    # ----- read-outs of the two population bumps ----------------------------------------------------------------
    def _decode(self):
        """Decode the position from the grid-cell SDR (the CRT read-out): for each axis, the coordinate whose per-module
        phases best explain the module bumps (argmax of the summed module mass), refined to sub-cell by a local
        centre-of-mass. Co-prime scales make the joint unique within the board; the read-out is the encoder inverse
        (SDR_MIGRATION.md — a point read-out, at the periphery only). The search is CONFINED to a WINDOW around the
        path-integrated prediction (`_anchor`): individual grid cells are ambiguous, disambiguated by the trajectory (the
        continuous attractor moves smoothly — it cannot teleport). The window is narrower than the smallest scale, so no
        two positions inside it share every module phase → the raw per-module argmax can no longer alias to a spurious
        phase-alignment far away (the blocked-move drift near a wall that otherwise poisons efference learning via a
        garbage `prev_pos`). Within the window it is the ordinary CRT read-out (no bias), so a clear sighting relocalises
        freely and there is no lag/overshoot. Anchor unset (cold start / after an unknown-motion reset) → full-board decode
        of the (single, clean) likelihood bump."""
        xs = np.arange(self.board)
        out = []
        for axis in range(self.grid.dims):
            score = np.zeros(self.board)
            for m, (s, ax) in enumerate(self._modinfo):
                if ax == axis:
                    score += self._pos[m][xs % s]            # each module votes its bump mass for every candidate x
            if self._anchor is not None:                     # continuity: search only NEAR the prediction (no teleport)
                a = int(round(self._anchor[axis]))
                w0, w1 = max(0, a - self._win), min(self.board - 1, a + self._win)
            else:
                w0, w1 = 0, self.board - 1
            i = w0 + int(np.argmax(score[w0:w1 + 1]))
            lo, hi = max(w0, i - 1), min(w1, i + 1)
            seg = score[lo:hi + 1]
            c = float((np.arange(lo, hi + 1) * seg).sum() / seg.sum()) if seg.sum() > 0 else float(i)
            out.append(c)
        return tuple(out)

    @property
    def pos(self):
        """The allocentric position estimate = the grid-cell SDR decoded, or None before the belief is seeded."""
        if not self._set:
            return None
        if self._readout is None:
            self._readout = self._decode()
        return self._readout

    @property
    def head(self) -> float:
        """The head-direction estimate = the ring bump's circular mean (the HD population-vector decode), or 0 before
        the belief is seeded. On a flat ring this is a defined 0 (an abelian body's identity rotation), not garbage."""
        if not self._set:
            return 0.0
        return float(np.arctan2((self._Q * np.sin(self._ring_ang)).sum(), (self._Q * np.cos(self._ring_ang)).sum()))

    def here(self):
        """The allocentric position belief, or None."""
        return self.pos

    # ----- the sensory likelihood: a bump per near-top L2/3 hypothesis (the evidence FIELD) ---------------------
    def _pos_like(self, sighting):
        """The position likelihood over the grid modules = per module, a superposition of a wrapped place-field bump at
        each sighting mode's phase (`centroid mod scale`), evidence-weighted. Symmetric poses share ONE centroid → a
        single sharp peak; a bimodal recognition → two peaks → a flat/split likelihood that pulls weakly. `sighting` =
        `[((x, y), theta, weight), ...]`. Returns a per-module list of rings (parallel to `self._pos`)."""
        rings = []
        for (s, axis) in self._modinfo:
            cells = np.arange(s)
            ring = np.zeros(s)
            for (cx, cy), _th, w in sighting:
                c = (cx if axis == 0 else cy) % s
                d = (cells - c + s / 2.0) % s - s / 2.0      # wrapped signed distance on the ring (the grid is periodic)
                ring += w * np.exp(-(d ** 2) / (2 * self.sig_pos ** 2))
            rings.append(_norm(ring + 1e-9))
        return rings

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

    # ----- IN: path integration by the gain-field operator on the grid SDR --------------------------------------
    def _predict_shift(self, action, was_learned: bool) -> None:
        """PREDICT: move the population by the action (L5's efference copy, in effect BEFORE this step) and add motion
        noise. A LEARNED action → SHIFT each grid module by the world velocity `v_world = R(head)·ego` (the operator on
        φ; the gain field reads the head as a population vector) + a small diffusion; an UNLEARNED one → the motion is
        unknown, so the prior collapses to UNIFORM (the sighting is then adopted — the old adopt-fully gain, emergent)."""
        if action is not None and was_learned:
            ego, dth = self._eff(action)
            world = _rot(self.head) @ ego                    # gain field: body ego → ONE world velocity (population-vector head)
            if self._anchor is not None:                     # advance the continuity prediction by the same world velocity
                self._anchor = (min(max(self._anchor[0] + world[0], 0.0), self.board - 1),
                                min(max(self._anchor[1] + world[1], 0.0), self.board - 1))
            for m, (s, axis) in enumerate(self._modinfo):
                d = world[0] if axis == 0 else world[1]      # each module shifts by the world velocity on its axis
                self._pos[m] = _norm(_ring_blur(_ring_shift(self._pos[m], d)))
            self._Q = _norm(_ring_blur(_ring_shift(self._Q, dth / (2 * np.pi) * self.n_ring)))
        elif action is not None:                             # unknown motion → maximal uncertainty (a flat prior)
            self._pos = self._uniform_pos()
            self._Q = np.full(self.n_ring, 1.0 / self.n_ring)
            self._anchor = None                              # motion unknown → drop the continuity anchor (the sighting re-seeds)
        else:                                                # no action (a static observation) → mild diffusion only
            self._pos = [_norm(_ring_blur(r)) for r in self._pos]
            self._Q = _norm(_ring_blur(self._Q))
        self._readout = None

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
                dom0 = max(sighting, key=lambda m: m[2])
                self._pos, self._Q, self._set, self._readout = self._pos_like(sighting), self._head_like(sighting), True, None
                self._anchor = (float(dom0[0][0]), float(dom0[0][1]))   # seed the continuity anchor at the sighting centroid
            return
        self._predict_shift(action, was_learned)            # PREDICT (path integration by the efference in effect before now)
        dom = max(sighting, key=lambda m: m[2]) if sighting else None
        if action is not None and dom is not None and dom[1] is not None and prev_pos is not None:   # LEARN from the dominant ORIENTED sighting
            (sx, sy), sth, _w = dom                          # (an orientation-less view cannot supply the turn dth → don't learn the operator from it)
            body = _rot(-prev_head) @ np.asarray((sx - prev_pos[0], sy - prev_pos[1]), float)   # the gain field, inverted
            self.l5.learn_efference(action, (body[0], body[1]), _wrap(sth - prev_head))
        if sighting:                                        # UPDATE: superimpose the sensory likelihood (Bayes: prior × likelihood)
            like = self._pos_like(sighting)
            self._pos = [_norm(self._pos[m] * like[m]) for m in range(len(self._pos))]
            self._Q = _norm(self._Q * self._head_like(sighting))
            self._readout = None
            self._anchor = self._decode()                   # re-sync the continuity anchor to the corrected read-out

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
