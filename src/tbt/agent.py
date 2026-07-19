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

import random

import numpy as np

from .basal_ganglia import BasalGanglia
from .column import Column
from .encoders import SDR, CategoryEncoder, GridEncoder
from .operator import eye
from .hippocampus import Hippocampus, WorldMap, WorldModel
from .perceive import SelfTracker, segment
from .reward import GoalMemory, ValueCritic
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
        self.critic = ValueCritic()      # the TD value critic (ROADMAP 3c) — its δ replaces the faked 2r−1 RPE
        self.hippocampus = Hippocampus(n_inputs=512, dims=dims, seed=seed)   # the composed hippocampus: map⊕replay⊕CA3⊕DG⊕CA1 (one handle)
        self.self_tracker = SelfTracker()   # discover the controllable ROOT (the 'self') from motion — not colour (bitter lesson)
        self.goal_mem = GoalMemory()        # discover WHICH feature the reward is contingent on (the goal), by the delta rule
        self._rng = random.Random(seed)     # exploration randomness (bootstrap + tie-break), seeded
        self._visited: set = set()          # cells the self has occupied (novelty target = the UNvisited frontier)
        self._blocked: set = set()          # cells a move was blocked into = LEARNED obstacles (walls); the rollout avoids them
        self._last = None                   # (objects, action, self-cell, score) of the previous frame — for online learning
        self._decision_col = None
        self._pending = None
        self._nav = None
        self._scene = None      # the COMPOSITIONAL column (lazy) — the SCENE one level up, fed by the sensory column
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
        """Perceive a CONTEXT in a cortical column, project it to the BG through the column's **L5IT** (`Column.striatum` — the
        intratelencephalic layer that projects to the striatum, `reference_cortical_layers_research`), SELECT an action by value
        (basal ganglia, with the critic's tonic dopamine `ρ` setting explore/exploit), GATE the winner to the motor (thalamus).
        Call `reward(r)` afterwards to train the choice by RPE. The perception column is created lazily (its input size = the
        context SDR's); the BG's cortical input is now a column's L5IT projection, not an ad-hoc raw-L4 relay."""
        if self._decision_col is None:
            self._decision_col = Column(sensory_n=context.n, n_cols=self._n_cols, order=1, seed=self._seed + 2)
        ctx = self.thalamus.relay(self._decision_col.striatum(context))               # cortex L5IT → BG relay
        rho = self.critic.rho()                                                       # tonic DA from the critic (rich→exploit)
        action = self.thalamus.gate(self.bg.select(ctx, n_actions, rho=rho, explore=explore))   # select by value, gate to motor
        self._pending = (ctx, action)
        return action

    def reward(self, r: float, next_context: SDR = None, done: bool = True) -> None:
        """Train the last `decide` by the VALUE CRITIC's dopamine-RPE (ROADMAP 3c) — the δ replaces the faked `2r−1`. Pass
        `next_context` (the SDR of the state the action led to) + `done=False` for a MULTI-STEP transition, so the critic
        BOOTSTRAPS value backward (`δ = r + γ·V(next) − V(now)`); omit them for an IMMEDIATE-reward step (`δ = r − V(now)`,
        a real prediction error with a learned baseline). The critic learns V; the basal ganglia's actor learns from δ."""
        if self._pending is None:
            return
        ctx, action = self._pending
        next_ctx = None
        if next_context is not None and not done:                       # relay the next state exactly as `decide` would
            next_ctx = self.thalamus.relay(self._decision_col.striatum(next_context))
        delta = self.critic.learn(ctx, r, next_ctx, done)               # TD update + the dopamine-RPE
        self.bg.learn(ctx, action, delta)                              # the actor learns from δ, not from raw reward
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

    def broadcast_efference(self, action):
        """L5PT's EFFERENCE COPY broadcast — the moving-sensor fix (`notes/l5_efference_broadcast_design.md`). The SELF column
        (the body's spatial column, `_nav`) emits its efference — the world-frame self-motion for `action` — and the agent,
        the thalamic relay (ARCHITECTURE: 'agent = plumbing'), routes it to every PEER spatial column, which path-integrates
        its OWN L6a by it (`apply_efference`, flow parsing). So the WHOLE agent knows the body moved and no column mistakes
        SELF-motion for the world moving ('one column had the efference, the others did not', `reference_hippocampus`;
        `reference_layer5_role`: L5PT's displacement IS the motor command + efference copy + inter-column message). Returns the
        broadcast motion, or None if the action is unlearned."""
        motion = self._nav_col().efference(action)
        if motion is None:
            return None
        for col in self._peer_spatial_columns():
            col.apply_efference(motion)
        return motion

    def _peer_spatial_columns(self):
        """The spatial columns OTHER than the self/body column (`_nav`) that track the sensor and must be told the body moved —
        the compositional column when present; the set grows as the agent gains more spatial columns (multi-column voting, a
        moving sensor)."""
        return [c for c in (self._scene,) if c is not None]

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

    # ----- THE COMPOSITIONAL COLUMN (ARCHITECTURE §9): the SCENE + state-conditioned behaviour = the override ---------
    def _scene_col(self) -> Column:
        """The lazily-built COMPOSITIONAL column — the SCENE, one level up. Its features-at-locations are recognised objects
        (object-id at pose), routed from the sensory column by the thalamus. It represents relations + STATE-CONDITIONED
        behaviour (the context-gated override: everything falls, but a supported object stays). The SAME `Column` class as the
        sensory one — "one column, used thrice" (sensory ⊕ task ⊕ compositional), honouring §5.1: the override is genuinely
        multi-column, not a gate bolted onto the single spatial column (`reference_tbt_object_behaviors`)."""
        if self._scene is None:
            grid = GridEncoder(scales=(7, 11, 13, 17), dims=self._dims, mw=1, bounds=[(0, 63)] * self._dims)
            self._scene = Column(sensory_n=1, n_cols=self._n_cols, order=2, seed=self._seed + 5, location=grid)
        return self._scene

    def place_object(self, object_id, pose) -> None:
        """Route a recognised `(object-id, pose)` from the sensory column UP to the compositional column via the thalamus —
        the first real cross-column binding (content ⊗ location)."""
        content, location = self.thalamus.project(object_id, pose)
        self._scene_col().place_object(content, location)

    def clear_scene(self) -> None:
        """Start a fresh scene configuration in the compositional column."""
        self._scene_col().clear_scene()

    def object_state(self, object_id) -> frozenset:
        """The relational STATE of a scene object (the geometry of its relations) — what the behaviour is conditioned on."""
        return self._scene_col().state_of(object_id)

    def learn_behavior(self, action, object_id, after_pose) -> None:
        """Learn what `action` does to a scene object, CONDITIONED on its relational state — the override, learned not coded
        (a supported object staying is just the effect keyed on the support state)."""
        self._scene_col().learn_behavior(action, object_id, after_pose)

    def predict_behavior(self, action, object_id):
        """Predict a scene object's next pose under `action`, gated by its relational state — supported stays, free falls."""
        return self._scene_col().predict_behavior(action, object_id)

    # ----- THE HIPPOCAMPAL WORLD-MAP (DESIGN §2): the forkable allocentric STATE the rollout simulates in -------------
    def world_state(self) -> WorldMap:
        """Assemble the current allocentric WORLD-STATE (`hippocampus/map.py`) from the live columns: the agent's
        path-integrated pose (L6a self-location) + the scene's objects at world poses + the frame extent. The forkable
        state a ROLLOUT simulates in — DERIVED on demand from the columns, not a parallel persistent store (DESIGN §5):
        the columns hold the slow learned model, the returned map is the fast forkable state. The map borrows the nav
        column's learned body operator BY REFERENCE, so a fork path-integrates the agent under the one shared model.
        ARC's board is world-anchored, so the nav pose and the scene objects already share one world frame (DESIGN §4)."""
        objects = self._scene_col().scene_snapshot() if self._scene is not None else {}
        bounds = [(0, 63)] * self._dims
        return WorldMap(self.pose(), objects, bounds, body=self._nav_col().operator, blocked=self._blocked)

    def world_model(self) -> WorldModel:
        """The learned FORWARD MODEL over the world-state (`hippocampus/replay.py`) — composes the agent's path-integration
        (nav operator, via the map) with the compositional column's STATE-CONDITIONED object dynamics. Used to LEARN the
        model from observed transitions (`WorldModel.learn`) and to ROLL it forward in planning."""
        return WorldModel(self._scene_col())

    def plan(self, reward, actions, horizon: int = 12, value=None) -> list:
        """Plan by hippocampal ROLLOUT (`hippocampus/replay.py`): fork the current world-state and search the learned model
        forward for the shortest action sequence reaching the goal (`reward(world) > 0`), the value critic scoring leaves
        when the goal is beyond the horizon. Returns the action sequence (empty = already satisfied). The GOAL is a reward
        predicate over world-states — selected elsewhere, never hand-coded into the planner (`feedback_bitter_lesson`). The
        leaf value defaults to the trained VALUE CRITIC over the featurised world (`value_of`) — 0 for an untrained critic
        (so goal-reward still drives), meaningful once `learn_value` has taught it which world-states pay off."""
        if value is None:
            value = self.value_of
        return self.hippocampus.plan(self.world_state(), self.world_model(), reward, actions, horizon, value)

    def value_of(self, world) -> float:
        """The learned value of a world-STATE — the `ValueCritic` scoring the featurised world (`Hippocampus.featurize`, the
        overlap-bearing world→SDR). This is the rollout's leaf heuristic; it replaces `replay.py`'s stand-in with the real
        critic. Returns 0 until the critic is trained (the honest prior)."""
        return self.critic.value(self.hippocampus.featurize(world))

    def learn_value(self, before, reward: float, after=None, done: bool = True) -> float:
        """Train the value critic on a world-state TRANSITION by TD (`δ = r + γ·V(after) − V(before)`), over the featurised
        worlds — so the rollout's leaf heuristic learns which world-states pay off from reward. Returns δ. Pass `after` +
        `done=False` for a multi-step transition (value bootstraps backward); omit for an immediate/terminal reward."""
        nxt = self.hippocampus.featurize(after) if (after is not None and not done) else None
        return self.critic.learn(self.hippocampus.featurize(before), reward, nxt, done)

    # ----- HIPPOCAMPAL EPISODIC MEMORY (DESIGN §2, slice 3): one-shot store + partial-cue completion via CA3 -----------
    def scene_tokens(self) -> frozenset:
        """The current scene as an EPISODE — a set of `(object_id, quantised-cell)` tokens, content-at-place. This is the
        pattern CA3 stores and completes: a glimpse of some objects recalls the whole configuration (the maze-wall case)."""
        return frozenset((oid, tuple(round(c) for c in pose[0])) for oid, pose in self.world_state().objects.items())

    def remember_scene(self) -> None:
        """Store the current scene as an EPISODE (one-shot) — routed through the hippocampus (`Hippocampus.remember`)."""
        self.hippocampus.remember(self.scene_tokens())

    def recall_scene(self, glimpse) -> set:
        """Complete the whole remembered scene from a PARTIAL glimpse (some of its `(object, cell)` tokens) — CA3 pattern
        completion via the hippocampus. An ambiguous glimpse recalls the union of the scenes it fits; a novel one recalls
        nothing (DESIGN §3½)."""
        return self.hippocampus.recall(glimpse)

    def chart_key(self, signature) -> frozenset:
        """The DG-separated CHART KEY of an environment `signature` (a set of active input bits) — distinct environments get
        well-separated keys, the same one returns the same key (DESIGN §2, slice 4). Routed through the hippocampus."""
        return self.hippocampus.chart_key(signature)

    def visit_environment(self, observed):
        """Enter an environment described by `observed` content tokens (its landmarks): RECALL its chart if a stored one
        explains the observation, else MINT a new one (DESIGN §2, slice 5). Returns `(chart_id, CA1Result)`. A PARTIAL view
        still recalls the chart (absence ≠ novelty); a CONTRADICTED view (a landmark the chart lacks) mismatches and remaps."""
        return self.hippocampus.visit(observed)

    # ----- CROSS-COLUMN VOTING via the thalamus content⊗location register (ARCHITECTURE §3; Phase 5) -----------------
    def vote_consensus(self, votes, location, min_support: int = 1):
        """Reach a CONSENSUS across columns' votes through the thalamus's content⊗location register. Each vote is a
        `(content_bits, location_bits)` pair a column casts — content BOUND to location (the structure-preserving CMP vote:
        an object bound to a pose, a digit bound to a place). Bind + bundle them, then read the content at `location` that
        at least `min_support` columns agree on (`min_support=k` filters a single column's distractor; `=1` is exact recall,
        i.e. place-value). Returns the consensus content bits. The thalamus is the shared register; the columns do the voting."""
        register = self.thalamus.bundle(*(self.thalamus.bind(c, l) for c, l in votes))
        return self.thalamus.read(register, location, min_support)

    # ----- L5 DISPLACEMENT / RELATIONS (ARCHITECTURE §9): location + location → the relation -------------------------
    def relate(self, pose_a, pose_b):
        """The relative pose of object B in object A's frame — the displacement cell's output, position/orientation-invariant.
        Poses are what `recognize` SOLVES."""
        return self._nav_col().relate(pose_a, pose_b)

    def observe_relation(self, id_a, pose_a, id_b, pose_b):
        """Update the RELATION between two recognised objects — a displacement that PERSISTS as the pair moves ("resting on",
        "part of"). Returns the current relative pose."""
        return self._nav_col().observe_relation(id_a, pose_a, id_b, pose_b)

    def relation_of(self, id_a, id_b, min_count: int = 2):
        """The STABLE relative pose of the pair, or None if they have no fixed relation."""
        return self._nav_col().relation_of(id_a, id_b, min_count)

    def recognize(self) -> list:
        """Recognise the buffered sweep → the surviving `Hypothesis` POPULATION (object + pose + evidence), best-first. The
        pose is SOLVED from the inter-fixation displacement geometry, so it is exact at ANY rotation and the object may be
        ENTERED ANYWHERE. Several tied hypotheses = the evidence genuinely does not separate them (a symmetry orbit, or an
        ambiguous object); an empty list = nothing recognised."""
        return self._nav_col().recognize()

    # ----- PERCEPTION: the peripheral RETINA (perceive.py) — a game frame → OBJECTS, and the DISCOVERED self ------------
    def transduce(self, grid) -> list:
        """The peripheral RETINA: segment a game frame (a colour grid) into OBJECTS (colour ⊕ 4-connected cells) — the agent's
        front-end for a raw frame. Core-Knowledge OBJECTNESS, NO semantics (the mechanic is inferred from colour + score); the
        deeper common-fate/recognition grouping refines it (`perceive.segment`)."""
        return segment(grid)

    def observe_self(self, before, after, action) -> None:
        """Update the discovered controllable ROOT (the 'self') from one transition — which object moved when acted. The self
        is LEARNED from motion, never colour-coded (`feedback_bitter_lesson`; `reference_l5_operator_kinds`)."""
        self.self_tracker.observe(before, after, action)

    def self_color(self):
        """The colour of the discovered controllable root (the 'self'), or None until it has been seen to move under action."""
        return self.self_tracker.root()

    def new_level(self) -> None:
        """A LEVEL boundary — a fresh environment: drop the per-level exploration state (visited/blocked/last). The learned
        MODEL (operator, dynamics, critic) AND the discovered GOAL (`goal_mem`) persist across levels — that persistence IS the
        cross-level transfer; only the map memory resets."""
        self._visited, self._blocked, self._last = set(), set(), None

    def step(self, fd):
        """The thin-agent GAME LOOP (one interaction): PERCEIVE the frame → CREDIT the reached feature by the reward (discover
        the goal) + LEARN the self's dynamics from the last transition → PLAN (pragmatic to a known goal, else epistemic novelty)
        → ACT. Returns `(action, coords)`. Composes the built regions: the retina (`transduce`), the discovered self
        (`self_tracker`), the goal-feature learner (`goal_mem`), the L6a operator (`learn_pose_move`), and the hippocampal
        rollout (`plan`). No game semantics are read (`feedback_bitter_lesson`): walls are LEARNED by bumping, the goal by reward."""
        objs = self.transduce(fd.grid)
        cur = self._self_pos(objs)                                    # the self's cell this frame (via the discovered root)
        if self._last is not None:                                   # LEARN from the previous transition
            prev_objs, action, prev_cur, prev_score = self._last
            reward = float(fd.score - prev_score)                    # the sparse score's delta = this step's reward
            moved = prev_cur is not None and getattr(action, "is_movement", False)
            if moved:                                                # CREDIT the feature the self REACHED (cue competition:
                reached = self._feature_at(prev_objs, self._target_cell(prev_cur, action))   # r=0 washes out non-goals)
                if reached is not None:
                    self.goal_mem.credit({reached}, reward)
            if reward > 0:                                           # a LEVEL boundary (the board just advanced): fresh map,
                self.new_level()                                     # but the learned model + the discovered goal PERSIST
            else:
                self.observe_self(prev_objs, objs, action)           # keep discovering / confirming the controllable root
                if moved and cur is not None:
                    if cur != prev_cur:                              # the self MOVED → the operator learns this action's Δ
                        self.learn_pose_move(action, self._as_pose(prev_cur), self._as_pose(cur))
                    elif self._nav_col().operator.known(action):     # tried but did NOT move → the target cell is an obstacle
                        cell = self._target_cell(prev_cur, action)
                        if cell is not None and cell != prev_cur:
                            self._blocked.add(cell)
        if cur is not None:
            self._visited.add(cur)
        action = self._act(objs, cur, fd.available_actions)
        self._last = (objs, action, cur, fd.score)
        return action, None

    def _as_pose(self, cell):
        """A grid cell → a continuous pose `(position, R=identity)` in the nav frame."""
        return (tuple(float(c) for c in cell), eye(self._dims))

    def _target_cell(self, cell, action):
        """The cell `action` moves the self INTO from `cell`, via the LEARNED operator (not the action's declared delta — the
        effect is discovered, `reference_l5_operator_kinds`). None until the operator has learned this action."""
        op = self._nav_col().operator
        if not op.known(action):
            return None
        tgt = op.apply(self._as_pose(cell), action)
        return tuple(round(c) for c in tgt[0])

    def _feature_at(self, objs, cell):
        """The feature (colour) of the object occupying `cell`, or None if the cell is empty — the pre-move content of the cell
        the self reached, which is what the goal credit attaches to (the self's own colour never appears here: it is read from
        the frame BEFORE the self arrives)."""
        if cell is None:
            return None
        for o in objs:
            if cell in o.cells:
                return o.color
        return None

    def _goal_cell(self, objs):
        """The cell of the visible object whose feature is the discovered goal — the pragmatic planner's target. None if no goal
        has been discovered yet, or it is not (unambiguously) on the board this frame."""
        g = self.goal_mem.goal()
        if g is None:
            return None
        goals = [o for o in objs if o.color == g]
        return goals[0].anchor if len(goals) == 1 else None

    def _self_pos(self, objs):
        """The self's cell this frame — the anchor of the object whose colour is the discovered controllable root; None until
        the root is discovered, or if it is ambiguous (more than one object of that colour)."""
        c = self.self_color()
        if c is None:
            return None
        selves = [o for o in objs if o.color == c]
        return selves[0].anchor if len(selves) == 1 else None

    def _act(self, objs, cur, available):
        """The one EFE planner (`reference_efe_and_epiplexity`, `feedback_epistemic_value_is_prediction_error`): PRAGMATIC when a
        goal is known — rollout the shortest path to the visible goal-featured object — and EPISTEMIC otherwise — rollout to the
        nearest UNVISITED reachable cell (novelty). Bootstraps the operator first by trying UNLEARNED actions; random when the
        self is unknown or nothing is reachable. Walls are avoided via the LEARNED `_blocked` set. `reference_goal_setting_priority_map`."""
        movement = [a for a in available if getattr(a, "is_movement", False)]
        if not movement:
            return self._rng.choice(available)
        if cur is None:                                              # self not discovered yet → move to generate the evidence
            return self._rng.choice(movement)
        op = self._nav_col().operator
        unlearned = [a for a in movement if not op.known(a)]
        if unlearned:                                               # bootstrap: learn each action's displacement
            return self._rng.choice(unlearned)
        self.set_pose(self._as_pose(cur)[0], eye(self._dims))       # anchor the nav pose to the perceived self
        goal_cell = self._goal_cell(objs)
        if goal_cell is not None and goal_cell != cur:              # PRAGMATIC: a known goal is on the board → plan straight to it
            at_goal = lambda w: 1.0 if (w.agent is not None and tuple(round(c) for c in w.agent[0]) == goal_cell) else 0.0
            plan = self.plan(at_goal, movement, horizon=64)
            if plan:
                return plan[0]                                     # (empty ⇒ the goal is unreachable given learned walls: explore)
        visited = set(self._visited)                               # EPISTEMIC: explore toward the unvisited frontier
        novel = lambda w: 1.0 if (w.agent is not None and tuple(round(c) for c in w.agent[0]) not in visited) else 0.0
        plan = self.plan(novel, movement, horizon=32)
        return plan[0] if plan else self._rng.choice(movement)
