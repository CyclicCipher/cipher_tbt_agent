"""agent.py — the live entry point of the TBT agent, and the ROOT of the reachability graph (RULES.md #2).

FIRST VERTICAL SLICE (2026-07-10): the agent composes TWO cortical columns — a SENSORY column and a TASK column
(ARCHITECTURE §5.1: always multi-column) — and runs a sensorimotor SCAN in which the task column carries a factored STATE
between fixations that conditions the sensory column's next-content prediction (project_place_invariance_needs_factored_state).
This is `option2.py`'s validated place-invariance mechanism, now built as the real column composition and wired from here
(so `column` → `htm`/`encoders` are all reachable — the reachability test flips to fully-wired). It is deliberately thin
(RULES.md #4/#5, feedback_thin_shell_agent): the columns do the work; the agent only routes fixations + carries state + reads
out. Task-format code (encoding a number into per-place feature SDRs, the train/test split) lives in the TEST, not here.

NEXT: thicken toward the full loop — the deep layers (L5 motor / L6 location) + the hippocampal rollout + BG select +
thalamus gate come as tasks exercise them (STATUS.md 'Next'). The generic game interface `step(observation)` is still a stub.
"""

from __future__ import annotations

import numpy as np

from .basal_ganglia import BasalGanglia
from .column import Column
from .encoders import SDR, CategoryEncoder, GridEncoder
from .thalamus import Thalamus


class _Readout:
    """The agent's learned READOUT of a column's L4 cells into a discrete content/state bucket — the canonical HTM
    SDRClassifier (softmax over active cells, delta rule `Δw = λ(y − z)x`; reference_htm_canonical_pipeline). This is the
    value-readout the analytic encoder-inverse cannot do faithfully on blurred predictive cells, so it lives at the motor
    periphery (the agent), not inside the value-free column."""

    def __init__(self, n_cells: int, n_buckets: int, lr: float = 0.1) -> None:
        self.W = np.zeros((n_buckets, n_cells)); self.lr = float(lr)

    def infer(self, cells) -> np.ndarray:
        if not cells:
            return np.ones(self.W.shape[0]) / self.W.shape[0]
        a = self.W[:, cells].sum(1); a -= a.max(); e = np.exp(a); return e / e.sum()

    def pred(self, cells) -> int:
        return int(np.argmax(self.infer(cells)))

    def learn(self, cells, bucket: int) -> None:
        if not cells:
            return
        z = self.infer(cells); y = np.zeros(self.W.shape[0]); y[bucket] = 1.0
        self.W[:, cells] += (self.lr * (y - z))[:, None]


class Agent:
    """A SENSORY column + a TASK column + their readouts. The sensory column predicts the next CONTENT at each fixation;
    the task column predicts the STATE that propagates to the next fixation (the carry, in arithmetic). Both are conditioned
    on the current state (fed via the proximal path — Column.observe §15). Generic over content/state cardinality; it does
    not know 'digit' or 'arithmetic' (that is the task/test)."""

    def __init__(self, feat_n: int, n_content: int, n_state: int, n_cols: int = 256, seed: int = 0) -> None:
        self.state_enc = CategoryEncoder(range(n_state), w=8, capacity=n_state)   # the factored-state code (generic)
        sensory_n = feat_n + self.state_enc.n                                     # L4 proximal = feature ⊕ state (§15)
        self.sensory = Column(sensory_n=sensory_n, n_cols=n_cols, order=1, seed=seed)      # predicts next CONTENT
        self.task = Column(sensory_n=sensory_n, n_cols=n_cols, order=1, seed=seed + 1)     # carries/predicts the STATE
        self.read_content = _Readout(n_cols, n_content)      # order=1 → M=1 → n_cells == n_cols
        self.read_state = _Readout(n_cols, n_state)
        # decision loop (ARCHITECTURE §3): the BASAL GANGLIA selects an action by value; the THALAMUS relays the percept
        # in + gates the winner out to the motor. The decision column is created lazily (its input size = the context's).
        # Exercised by a reward-driven task; the arithmetic scan does not use it.
        self.thalamus = Thalamus()
        self.bg = BasalGanglia(seed=seed)
        self._decision_col = None
        self._pending = None
        self._nav = None
        self._n_cols = int(n_cols)
        self._seed = int(seed)

    # ----- one fixation: sense the feature in the current state, read content + next state -----------------------------
    def _sense(self, feature: SDR, state: int, learn: bool):
        st = self.state_enc.encode(state)
        s_cells = self.sensory.observe(feature, st, learn=learn)
        t_cells = self.task.observe(feature, st, learn=learn)
        return s_cells, t_cells

    def learn_fixation(self, feature: SDR, state_in: int, next_content: int, state_out: int) -> None:
        """Teacher-forced training step: sense (feature, state_in); the sensory column learns → next_content, the task
        column learns → state_out (state_out is the OBSERVABLE outcome — did the next place change — not a hand-coded rule)."""
        s_cells, t_cells = self._sense(feature, state_in, learn=True)
        self.read_content.learn(s_cells, next_content)
        self.read_state.learn(t_cells, state_out)

    def scan(self, features, state0: int, learn: bool = False):
        """Drive the sensorimotor SCAN over a sequence of feature SDRs, the task column carrying its OWN predicted state
        forward (autonomous rollout — the factored-state loop). Returns the per-fixation predicted content buckets."""
        state, out = int(state0), []
        for feature in features:
            s_cells, t_cells = self._sense(feature, state, learn=learn)
            out.append(self.read_content.pred(s_cells))
            state = self.read_state.pred(t_cells)
        return out

    # ----- the decision loop: perceive → relay → SELECT (BG) → gate → act; then reward() trains it ---------------------
    def decide(self, context: SDR, n_actions: int, explore: float = 0.0) -> int:
        """Perceive a CONTEXT (a decision column), RELAY the percept (thalamus), SELECT an action by value (basal ganglia),
        GATE the winner to the motor (thalamus). Call `reward(r)` afterwards to train the choice by RPE. The decision column
        is created lazily on the first call (its input size = the context SDR's) and read frozen (a deterministic percept)."""
        if self._decision_col is None:
            self._decision_col = Column(sensory_n=context.n, n_cols=self._n_cols, order=1, seed=self._seed + 2)
        cells = self._decision_col.observe(context, learn=False)                      # perceive (frozen → stable percept)
        ctx = self.thalamus.relay(cells)                                              # cortex → BG relay
        action = self.thalamus.gate(self.bg.select(ctx, n_actions, explore=explore))  # select by value, gate to the motor
        self._pending = (ctx, action)
        return action

    def reward(self, r: float) -> None:
        """Train the last `decide` by reward-prediction error. For an immediate reward r∈{0,1}, the centered RPE is 2r−1
        (rewarded → Go, unrewarded → NoGo). A proper TD critic (`reward.py`) comes with a multi-step-value task."""
        if self._pending is None:
            return
        ctx, action = self._pending
        self.bg.learn(ctx, action, rpe=2.0 * float(r) - 1.0)
        self._pending = None

    # ----- the SPATIAL slice: L6a path integration via the TRANSFORM operator (ARCHITECTURE §8) ----------------------
    def _nav_col(self) -> Column:
        """The lazily-built SPATIAL column: a location frame → L6a's `ModularOperator` (the TRANSFORM primitive). Kept lazy
        like the decision column; the first thing to drive L6a path integration. The column's L4/L2·3/L5 are present but
        undriven here (this slice exercises L6a only), exactly as the arithmetic slice leaves the deep layers undriven."""
        if self._nav is None:
            # mw=1: a SHARP place code — crisp addressing for feature-at-location binding (the bump's graded overlap, mw>1,
            # buys path-integration noise-robustness, deferred to real frames). bounds = a 64×64 ARC frame.
            grid = GridEncoder(scales=(7, 11, 13, 17), dims=2, mw=1, bounds=[(0, 63), (0, 63)])
            head = GridEncoder(scales=(4,), dims=1, mw=1, bounds=[(0, 3)])   # a 4-cell head-direction ring (for SE(2))
            # order=2: L4's OUTPUT (active cells) must encode feature-AT-location (a location-specific cell per feature
            # column) so L2/3 pools features-at-locations, not bare features — order=1 would collapse the location.
            self._nav = Column(sensory_n=1, n_cols=self._n_cols, order=2, seed=self._seed + 3, location=grid, heading=head)
            self._feat_enc = CategoryEncoder(range(16), w=8, capacity=16)   # the feature (ARC 16-colour palette) transducer
        return self._nav

    def learn_move(self, action, before, after) -> None:
        """Learn what `action` does to the body's location (the L6a operator; ARCHITECTURE §8), from an observed move
        `before → after` (coordinates). Position-invariant: learned at some places, it holds everywhere."""
        self._nav_col().learn_move(action, before, after)

    def locate(self, coord) -> SDR:
        """Anchor the body's location to a sensed coordinate (reset the path integrator)."""
        return self._nav_col().locate(coord)

    def path_integrate(self, action) -> SDR:
        """Dead-reckon the body forward by `action`, no sensory input — the learned operator applied to the location code."""
        return self._nav_col().path_integrate(action)

    def where(self):
        """The body's current dead-reckoned coordinate (decode L6a's location state)."""
        return self._nav_col().where()

    # ----- SE(2) non-abelian path integration (ARCHITECTURE §8): heading-dependent motion -----------------------------
    def set_pose(self, coord, heading) -> None:
        """Anchor the full SE(2) pose (location + heading) from a sensory fix."""
        self._nav_col().set_pose(coord, heading)

    def learn_pose_move(self, action, before_pose, after_pose) -> None:
        """Learn an action's SE(2) effect: the heading-CONDITIONED location shift + the heading shift, from an observed pose
        move `*_pose = ((x, y), heading)`. FORWARD's location shift depends on heading (non-abelian); TURN shifts heading."""
        self._nav_col().learn_pose_move(action, before_pose, after_pose)

    def path_integrate_pose(self, action):
        """Dead-reckon the body's SE(2) pose forward by `action`, no sensory input — non-commutative (FORWARD;TURN ≠
        TURN;FORWARD) because FORWARD's effect is keyed on the current heading."""
        return self._nav_col().path_integrate_pose(action)

    def pose(self):
        """The body's current dead-reckoned SE(2) pose ((x, y), heading)."""
        return self._nav_col().pose()

    def sense_at(self, feature, learn: bool = True) -> None:
        """Bind the FEATURE sensed at the body's current location (the L4↔L6a loop; ARCHITECTURE §8). Order-invariant: what
        is learned here is later predicted from the LOCATION, in any traversal order."""
        self._nav_col().sense_at(self._feat_enc.encode(feature), learn=learn)

    def predict_feature(self):
        """Predict the feature at the body's current (dead-reckoned) location, decoded to a feature value; None if unbound.
        Composes the two primitives: the operator supplies WHERE (path integration), L4 supplies WHAT (feature-at-location)."""
        cols = self._nav_col().predict_feature()
        return self._feat_enc.decode(SDR(self._feat_enc.n, cols)) if cols else None

    def reset_object(self) -> None:
        """L2/3 object boundary: start recognising/learning a fresh object (the sensor has moved onto a new one)."""
        self._nav_col().reset_object()

    def perceive_object(self, learn: bool = True) -> int:
        """Pool the CURRENT L4 feature-at-location code (from the last `sense_at`) into the stable L2/3 object IDENTITY, and
        return its integer label (−1 if unrecognised). Across a traversal the label is STABLE — the object, not the fixation."""
        self._nav_col().pool(learn=learn)
        return self._nav_col().object_id()

    # ----- OBJECT-CENTRIC recognition (ARCHITECTURE §8): re-anchor the frame per object; emergent boundary ------------
    def start_object(self) -> None:
        """Object ONSET — the coupled event: re-anchor the L6a frame to its origin (so sensing is OBJECT-RELATIVE =
        translation-invariant) AND start a fresh L2/3 identity. Call at a learning-time boundary; `perceive` fires it
        emergently on recognition failure at inference."""
        self._nav_col().start_object()

    def perceive(self, feature, learn: bool = True) -> int:
        """Sense a feature at the body's current object-relative location and pool it into the object IDENTITY; on a
        recognition FAILURE this fires the coupled onset (re-anchor + fresh identity) — the emergent object boundary.
        Returns the object's integer label (−1 if none)."""
        self._nav_col().perceive(self._feat_enc.encode(feature), learn=learn)
        return self._nav_col().object_id()

    def step(self, observation):
        """The generic game interface (observation → action) — still a STUB. The wired slices are `scan` (forward model)
        and `decide`/`reward` (selection); the full game loop (perceive → plan → act → win) composes them (STATUS.md
        'Next'). Raises rather than a silent no-op (RULES.md #3)."""
        raise NotImplementedError("Agent.step: full game loop not built — wired slices are Agent.scan + Agent.decide.")
