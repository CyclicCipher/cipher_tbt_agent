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

    def __init__(self, feat_n: int, n_content: int, n_state: int, n_cols: int = 256, seed: int = 0,
                 dims: int = 2) -> None:
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
        self._feat_enc = CategoryEncoder(range(16), w=8, capacity=16)   # the feature (ARC 16-colour palette) transducer —
        #                                                                 shared by every spatial column (the peripheral)
        self._n_cols = int(n_cols)
        self._seed = int(seed)
        self._dims = int(dims)    # the SPACE the body moves in — 2 for an ARC frame, 3 for a 3-D environment. A property of
        #                           the ENVIRONMENT, so it is given here; the column's mechanism is identical either way.

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
        """The lazily-built SPATIAL column: a location frame → L6a's `MotionOperator` (the TRANSFORM primitive). Kept lazy
        like the decision column; the first thing to drive L6a path integration. The column's L4/L2·3/L5 are present but
        undriven here (this slice exercises L6a only), exactly as the arithmetic slice leaves the deep layers undriven."""
        if self._nav is None:
            # mw=1: a SHARP place code — crisp addressing for feature-at-location binding (the bump's graded overlap, mw>1,
            # buys path-integration noise-robustness, deferred to real frames). bounds = a 64-cell ARC frame per axis.
            # The grid is the READ-OUT (what L4 binds to); the location STATE is continuous inside the column, so orientation
            # needs no ring and rotation needs no orientation modules — ONE column dead-reckons AND rotates, exactly, in ANY
            # dimension (`dims`): the orientation is a rotation MATRIX, so SO(3) is the same code path as SO(2).
            grid = GridEncoder(scales=(7, 11, 13, 17), dims=self._dims, mw=1, bounds=[(0, 63)] * self._dims)
            # order=2: L4's OUTPUT (active cells) must encode feature-AT-location (a location-specific cell per feature
            # column) so L2/3 pools features-at-locations, not bare features — order=1 would collapse the location.
            self._nav = Column(sensory_n=1, n_cols=self._n_cols, order=2, seed=self._seed + 3, location=grid)
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

    # ----- SE(n) path integration (ARCHITECTURE §8): orientation-dependent motion, CONTINUOUS orientation -------------
    def set_pose(self, coord, rotation) -> None:
        """Anchor the full pose (location + ORIENTATION as an n×n rotation matrix — `operator.from_angle(deg)` builds one in
        2-D) from a sensory fix."""
        self._nav_col().set_pose(coord, rotation)

    def learn_pose_move(self, action, before_pose, after_pose) -> None:
        """Learn an action's effect from an observed pose move `*_pose = (position, R)`. The operator stores the BODY-frame
        displacement + body-frame rotation, so ONE observation generalises to every position AND every orientation —
        FORWARD's world effect then depends on which way the body faces (non-abelian), with no keying and no ring."""
        self._nav_col().learn_pose_move(action, before_pose, after_pose)

    def pose(self):
        """The body's current dead-reckoned pose `(position, R)`."""
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

    # ----- OBJECT-CENTRIC recognition (ARCHITECTURE §8): re-anchor the frame per object; emergent boundary ------------
    def start_object(self) -> None:
        """Object ONSET — the coupled event: re-anchor the L6a frame to its origin (so sensing is OBJECT-RELATIVE =
        translation-invariant), start a fresh L2/3 identity, and clear the sweep buffer. Call at a learning-time boundary;
        `perceive` fires it emergently on recognition failure at inference."""
        self._nav_col().start_object()

    def perceive(self, feature) -> int:
        """INFER, online: sense a feature wherever the body actually is, and SOLVE which object it is and where on it we are
        — a hypothesis population narrowed per fixation. Returns the object's integer label (−1 = nothing recognised, or
        genuinely ambiguous: too few fixations to fix a pose AND the feature is shared). Recognises an object entered
        ANYWHERE, at ANY pose; the object boundary is still emergent. LEARNING is `sense_sweep` × n → `commit`."""
        col = self._nav_col()
        return col.label_of(col.perceive(self._feat_enc.encode(feature)))

    def commit(self) -> int:
        """LEARN the buffered sweep (`start_object` → `sense_sweep` × n → `commit`) — the end-of-episode step: does a known
        object, at ANY pose, explain this sweep? If so reinforce it; else mint a new identity and bind the sweep to it.
        Returns the committed object's label (−1 if deferred because L4 has not learned the features yet — look again).
        Committing per EPISODE rather than per fixation is what lets L2/3 revise, and is why objects that share a
        feature-at-location no longer merge."""
        col = self._nav_col()
        return col.label_of(col.commit())

    # ----- POSE-INVARIANT recognition (plan R4, on the SAME column — the two-frame seam is gone) ---------------------
    def sense_sweep(self, feature) -> None:
        """Sense + BUFFER a fixation of an object whose POSE is unknown (the online `perceive` cannot recognise it until the
        pose is undone, so the sweep is recorded and `recognize` solves the pose from it)."""
        self._nav_col().sense_sweep(self._feat_enc.encode(feature))

    # ----- OBJECT DYNAMICS (ARCHITECTURE §9): what an action does to a THING ------------------------------------------
    def learn_object_move(self, action, before_pose, after_pose) -> None:
        """Learn what `action` does to an OBJECT, from an observed `(pose → pose)` move — the poses being what `recognize`
        SOLVED. Generalises to every position and orientation BY CONSTRUCTION (the frame does it, not the data), and to every
        OBJECT because nothing is keyed on which one — the base for physical law (ARCHITECTURE §9)."""
        self._nav_col().learn_object_move(action, before_pose, after_pose)

    def predict_object_move(self, pose, action):
        """Predict where `action` puts an object now at `pose` — the forward model over objects."""
        return self._nav_col().predict_object_move(pose, action)

    def recognize(self) -> list:
        """Recognise the buffered sweep → the surviving `Hypothesis` POPULATION (object + pose + evidence), best-first. The
        pose is SOLVED from the inter-fixation displacement geometry, so it is exact at ANY rotation and the object may be
        ENTERED ANYWHERE. Several tied hypotheses = the evidence genuinely does not separate them (a symmetry orbit, or an
        ambiguous object); an empty list = nothing recognised."""
        return self._nav_col().recognize()

    def step(self, observation):
        """The generic game interface (observation → action) — still a STUB. The wired slices are `scan` (forward model)
        and `decide`/`reward` (selection); the full game loop (perceive → plan → act → win) composes them (STATUS.md
        'Next'). Raises rather than a silent no-op (RULES.md #3)."""
        raise NotImplementedError("Agent.step: full game loop not built — wired slices are Agent.scan + Agent.decide.")
