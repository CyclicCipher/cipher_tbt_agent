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

import math
import random

import numpy as np

from .basal_ganglia import BasalGanglia
from .column import Column
from .encoders import SDR, CategoryEncoder, GridEncoder
from .operator import eye
from .hippocampus import Hippocampus, WorldMap, WorldModel
from .behavior import Transform
from .modality import touch, vision
from .operator import add, norm, solve_rotation, sub
from .perceive import SelfTracker, segment
from .reward import GoalMemory, LearningProgress, ValueCritic
from .thalamus import Thalamus
from .touch import contact_toward


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
                 dims: int = 2, modalities=None) -> None:
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
        self.progress = LearningProgress()  # LEARNABLE NOVELTY: the epistemic reward = how fast the agent's OWN forward model
        #                                     is learning (prediction-error REDUCTION, prequential — `reward.py`). No shadow model.
        self._MODE_CTX = frozenset({("drive",)})   # the striatal context the BG arbitrates the two DRIVES in (goal vs seek)
        self._last_mode = None                     # the drive the BG last selected — trained by the payoff RPE next step
        # SENSORY MODALITIES (modality.py): a sense = (transduce, feature, location, pose_source); the column + connections are
        # modality-INVARIANT, so this list is all it takes. VISION (segment) + TOUCH (the skin) ship by default; touch's skin is
        # the AGENCY signal (self-caused contact), so ContactDynamics learns object push behaviour only from motions the self
        # actually CAUSED (`notes/touch_and_body_design.md`). The recognising columns (Gap-3 vision, recognise-by-touch) defer.
        self._modalities = {m.name: m for m in (modalities if modalities is not None else [vision(), touch()])}
        self._dynamics = Transform(dims)        # the L5 TRANSFORM: one cortical layer + its population read-out (behavior.py)
        self._param_enc = GridEncoder(scales=(7, 11, 13, 17), dims=dims, mw=3)   # the press's free displacement (its SITUATION)
        self._param_const = self._param(tuple(0.0 for _ in range(dims)))         # the constant situation of a press-free frame
        self._static: dict = {}                 # {cell: feature} — what PERCEPTION sees at each non-self, non-mover cell
        self._rng = random.Random(seed)     # exploration randomness (bootstrap + tie-break), seeded
        self._movers: set = set()           # non-self object features observed to MOVE = pushable objects (routed to the scene)
        self._tried: set = set()            # (cell, action) already attempted in bootstrap — so an always-blocked action here
        #                                     is tried once, then the agent moves on (learns it wherever it first succeeds)
        self._last = None                   # (objects, action, self-cell, score, positions) of the previous frame — online learning
        self._last_plan: list = []          # the action sequence the last decision committed to — the agent's current INTENT
        #                                     (introspectable for the two-pane imagination widget: imagined future vs reality)
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
        bounds = getattr(self, "_extent", None) or [(0, 63)] * self._dims
        return WorldMap(self.pose(), objects, bounds, body=self._nav_col().operator, static=self._static)

    def world_model(self) -> WorldModel:
        """The learned FORWARD MODEL over the world-state (`hippocampus/replay.py`), driven by the ONE L5 transform: the rollout
        path-integrates the agent freely and, where its move lands on something (contact, geometric in imagination), the transform
        supplies the CORRECTION to that free motion and the felt thing's own delta. Nothing is enumerated — "blocks", "gets pushed"
        and "is walked through" are just different learned corrections, and solidity is one of them (`behavior.py`)."""
        return WorldModel(self._dynamics_delta)

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
        """The peripheral RETINA — the VISION modality's transducer (`modality.vision`): segment a game frame (a colour grid)
        into OBJECTS (colour ⊕ 4-connected cells). Core-Knowledge OBJECTNESS, NO semantics (the mechanic is inferred from colour
        + score); the deeper common-fate/recognition grouping refines it (`perceive.segment`)."""
        return self._modalities["vision"].transduce(grid, None)

    def observe_self(self, before, after, action) -> None:
        """Update the discovered controllable ROOT (the 'self') from one transition — which object moved when acted. The self
        is LEARNED from motion, never colour-coded (`feedback_bitter_lesson`; `reference_l5_operator_kinds`)."""
        self.self_tracker.observe(before, after, action)

    def self_color(self):
        """The colour of the discovered controllable root (the 'self'), or None until it has been seen to move under action."""
        return self.self_tracker.root()

    def new_level(self) -> None:
        """A LEVEL boundary — a fresh environment: drop the per-level exploration state (visited/blocked/last + the scene's
        object placements). The learned MODEL (operator, object dynamics, critic), the discovered GOAL (`goal_mem`), and which
        features are MOVERS all persist across levels — that persistence IS the cross-level transfer; only the map memory resets."""
        self._static, self._tried, self._last = {}, set(), None
        if self._scene is not None:
            self.clear_scene()

    def step(self, fd):
        """The thin-agent GAME LOOP (one interaction): PERCEIVE the frame → discover MOVERS + LEARN the self's motion AND the
        movers' push dynamics from the last transition → CREDIT the reward to discover the goal → ROUTE movers to the scene →
        PLAN (pragmatic to the discovered goal, else epistemic novelty) → ACT. Returns `(action, coords)`. Composes the built
        regions: the retina (`transduce`), the discovered self + movers (from motion), the goal learner (`goal_mem`), the L6a
        operator (`learn_pose_move`), the scene column's object dynamics, and the hippocampal rollout (`plan`). No game semantics
        are read (`feedback_bitter_lesson`): walls learned by bumping, movers by motion, the goal by reward."""
        objs = self.transduce(fd.grid)
        self._extent = [(0, len(fd.grid[0]) - 1), (0, len(fd.grid) - 1)] if fd.grid else None
        pos = self._positions(objs)                                   # {feature: cell} for the unambiguous single-instance objects
        if self._last is not None:
            prev_objs, action, prev_cur, prev_score, prev_pos, prev_skin = self._last
            reward = float(fd.score - prev_score)                    # the sparse score's delta = this step's reward
            if reward == 0:                                          # a within-level transition: confirm the self FIRST, so `sc`
                self.observe_self(prev_objs, objs, action)           #   below reflects this move (else the first push is missed)
        sc = self.self_color()
        cur = self._self_pos(objs)                                    # the self's cell this frame (via the discovered root)
        if self._last is not None:                                    # LEARN from the previous transition
            moved = prev_cur is not None and getattr(action, "is_movement", False)
            payoff = float(reward)
            if reward == 0:                                          # (frames across a level boundary don't correspond)
                payoff += self.progress.observe(self._model_loss(prev_cur, action, prev_pos, cur, pos))   # SCORE the model
                self._track_movers(prev_pos, pos, sc)               # BEFORE it learns — the PREQUENTIAL loss (epiplexity)
                self._learn_dynamics(action, prev_objs, prev_skin, prev_pos, pos, prev_cur, cur, sc)  # learn the change from FELT contact
            if self._last_mode is not None:                          # TONIC DOPAMINE = the average PAYOFF rate, extrinsic ⊕
                self.bg.learn(self._MODE_CTX, self._last_mode, payoff - self.critic.rho())   # epistemic; the BG's drive choice
            self.critic.tonic(payoff)                                # is trained by that payoff's RPE against the rate
            if moved:
                self._credit_goal(prev_objs, prev_pos, action, prev_cur, reward, sc)   # discover the goal by the delta rule
            if reward > 0:                                           # a LEVEL boundary (the board just advanced): fresh map,
                self.new_level()                                     # but the learned model + goal + movers PERSIST
            elif moved and cur is not None and cur != prev_cur:      # the self MOVED → the operator learns this action's Δ
                self.learn_pose_move(action, self._as_pose(prev_cur), self._as_pose(cur))
        self._route_movers(pos)                                      # place visible movers into the scene column (world_state/rollout)
        self._static = self._static_cells(objs, sc)                  # what is SEEN at every other cell — the model decides the rest
        skin = self._skin(fd.grid, objs, cur)                       # what the body FEELS now — the agency signal for next step
        action = self._act(objs, cur, pos, fd.available_actions)
        self._last = (objs, action, cur, fd.score, pos, skin)
        return action, None

    def _unlearned_cells(self, pos) -> set:
        """The cells holding something the MODEL cannot yet predict — LEARNABLE NOVELTY read as a LEVEL, prospectively
        (`reference_learnable_novelty`; the same `Transform.confident` primitive the occasion gate uses, so no new machinery).

        This is what replaced the hand-maintained `_visited` set, and the difference is the whole point. Visitation paid for any
        cell the body had not occupied — geometric bookkeeping OUTSIDE the model, which keeps paying on a world the agent
        understands perfectly and is blind to whether anything is left to learn. Confidence is a read of the MODEL'S OWN state:
        it GENERALISES (a feature learned anywhere is confident everywhere, so a never-visited cell whose features are all
        modelled pays NOTHING) and it goes to zero at mastery, so the drive dies exactly when there is nothing more to extract.

        It also explains why this finds a hidden goal: a goal is a perceptible FEATURE the model has never interacted with, so
        it is precisely what is not-yet-confident, and the epistemic pull points at it until it has been touched."""
        unit = tuple(1.0 if i == 0 else 0.0 for i in range(self._dims))
        seen = dict(self._static)                                   # statics (walls, pads, landmarks) …
        for c in self._movers:                                      # … and any visible mover, whose dynamics also want learning
            if c in pos:
                seen[pos[c]] = c
        return {cell for cell, feat in seen.items()
                if not self._confident(self._cue("press", "into", feat, None), unit)}

    def _model_loss(self, prev_cur, action, prev_pos, cur, pos) -> float:
        """The agent's OWN forward model's PREQUENTIAL loss on the transition just observed: it predicted where the body and
        each tracked mover would end up, reality arrived, and this is the fraction it got WRONG — scored BEFORE the model
        learns from it, which is what makes the loss curve prequential (`reward.LearningProgress`).

        The model is the one that already exists and already matters for planning — the L6a operator path-integrating the body
        ⊕ the L5 `Transform`'s contact deltas ⊕ its occasions, composed by `hippocampus.WorldModel`. Nothing here is a second
        model of the frames: a shadow predictor would have to re-learn what this one already knows, and its errors would not be
        the errors that make plans fail. Features (colour) are intact — movers are keyed by feature and the statics the model
        presses into are the perceived features themselves."""
        if prev_cur is None or cur is None or not getattr(action, "is_movement", False):
            return 0.0
        world = WorldMap(self._as_pose(prev_cur),
                         {c: self._as_pose(prev_pos[c]) for c in self._movers if c in prev_pos},
                         bounds=self._extent, body=self._nav_col().operator, static=self._static)
        nxt = self.world_model().step(world, action)
        wrong = 0.0 if nxt.agent is not None and tuple(round(x) for x in nxt.agent[0]) == tuple(cur) else 1.0
        n = 1
        for c, pose in nxt.objects.items():                          # each tracked mover the model also had to predict
            if c in pos:
                n += 1
                wrong += 0.0 if tuple(round(x) for x in pose[0]) == tuple(pos[c]) else 1.0
        return wrong / n

    def _param(self, displacement):
        """The interaction's basal SITUATION as a context SDR -- the press's free displacement. Grid-coded, so nearby presses
        share bits and the layer's conjunction over them is metric rather than a lookup table."""
        return {("p", b) for b in self._param_enc.encode(displacement).active}

    @staticmethod
    def _cue(frame, tag, felt, beyond):
        """The single proximal cue naming this contact in one reference frame. Beyond is absent for the context-free BASE
        (`beyond=None`) and present for an OCCASION-specific exception, and the two are DIFFERENT cues -- so they drive
        different columns and are combined by SELECT (override), never by SUM. That is the whole fix for the dilution the
        summing form had: an obstructed push no longer shares its error with the free push, because it teaches a different
        cue entirely."""
        return (frame, tag, felt) if beyond is None else (frame, tag, felt, "on", beyond)

    def _press_frame(self, eff):
        """The rotation carrying the canonical press axis onto THIS press direction -- SOLVED from the displacement, not
        constructed (`operator.solve_rotation`, the same closed form recognition uses)."""
        e1 = tuple(1.0 if i == 0 else 0.0 for i in range(self._dims))
        return solve_rotation([eff], [e1]) or eye(self._dims)

    def _canon(self, eff):
        """The press expressed in its own frame: all of it along the canonical axis. Every direction has the SAME canonical
        press, which is exactly what makes one observation cover all of them."""
        return tuple(norm(eff) if i == 0 else 0.0 for i in range(self._dims))

    def _press(self, tag, felt, beyond, eff):
        """The PRESS-ALIGNED prediction for a cue: tuned in the frame the press itself defines, so east/west/north/south are
        literally the same local operation and ONE observation covers all of them (`reference_lid_locally_in_distribution`;
        `project_representation_shortcut_lesson`). Direction generality comes from the FRAME, not a prior -- the deleted
        `ObjectBehavior` asserted it by initialising a change matrix to the identity, a claim about physics welded in."""
        R, cp = self._press_frame(eff), self._param(self._canon(eff))
        return self._dynamics.predict({self._cue("press", tag, felt, beyond)}, cp, frame=R)

    def _learn_press(self, tag, felt, beyond, eff, target) -> None:
        R, cp = self._press_frame(eff), self._param(self._canon(eff))
        self._dynamics.learn({self._cue("press", tag, felt, beyond)}, cp, target, frame=R)

    def _world(self, tag, felt):
        """The WORLD-ALIGNED prediction for the OBJECT: tuned in the world frame and independent of the press, so it can hold
        behaviour anchored to the world rather than to whoever pushed -- a balloon rising however it is shoved. Its situation
        is constant for that reason."""
        return self._dynamics.predict({self._cue("world", tag, felt, None)}, self._param_const)

    def _learn_world(self, tag, felt, target) -> None:
        self._dynamics.learn({self._cue("world", tag, felt, None)}, self._param_const, target)

    def _confident(self, cue, eff) -> bool:
        """Has a CONNECTED segment been learned for this cue -- epiplexity read as a LEVEL (is reusable structure committed
        here yet)? Keyed on the canonical press, so an occasion learned in one direction is confident in all."""
        return self._dynamics.confident({cue}, self._param(self._canon(eff)))

    def _dynamics_delta(self, tag, felt, beyond, eff):
        """L5's predicted change for one contact -- a press-following BASE plus OCCASION overrides minted on refutation
        (`reference_recognition_under_occlusion`; occasion setting). The default is press-following, bold and direction-general
        from one observation (our generalize-then-correct thesis, `feedback_prefer_generalize_then_correct`). Two confident
        occasions can override it, each a separate cue so the base is never diluted:

          * a WORLD occasion on the OBJECT -- minted when press-following is refuted with nothing behind the object, i.e. the
            thing moves in a world-fixed direction however it is pushed (the balloon). It overrides, because the press frame is
            simply the wrong frame for that object.
          * a BACKDROP occasion -- minted when press-following is refuted with something behind the object (a wall ⇒ blocked).
            It overrides for that backdrop only, so an UNSEEN backdrop (the pad the winning push lands on, unobservable because
            that push ends the level) falls through to the base and inherits the object's general behaviour."""
        if self._confident(self._cue("world", tag, felt, None), eff):
            return self._world(tag, felt)
        if beyond is not None and self._confident(self._cue("press", tag, felt, beyond), eff):
            return self._press(tag, felt, beyond, eff)
        return self._press(tag, felt, None, eff)

    def _learn_delta(self, tag, felt, beyond, eff, observed, impeded=False) -> None:
        """Teach the model from one felt interaction. The routing turns on ONE physical fact -- was the body IMPEDED (it did
        not advance by its full efference)? -- which cleanly separates the two ways press-following can be wrong:

          * IMPEDED -- something STOPPED the contact (a wall, the edge, a mover behind). That is a BACKDROP occasion, keyed on
            what is behind, ALWAYS -- even on the very first contact, so an obstructed push never teaches the base "it stays".
            This is the fix for the bootstrap poisoning: blocking is never mistaken for the object's free behaviour.
          * FREE but REFUTED -- the body advanced yet the object did NOT move the press-following way. Nothing stopped it, so
            the press frame is simply the wrong frame for it: it is WORLD-anchored (a balloon), and a world occasion is minted.

        A confident occasion OWNS its contact and SHIELDS the base (so every later obstructed push refines the wall-occasion
        and never dilutes the free-push base). A free, unrefuted contact teaches the base -- press-following, bold, one-shot."""
        if self._confident(self._cue("world", tag, felt, None), eff):
            self._learn_world(tag, felt, observed)
            return
        if beyond is not None and self._confident(self._cue("press", tag, felt, beyond), eff):
            self._learn_press(tag, felt, beyond, eff, observed)
            return
        if impeded and beyond is not None:                         # STOPPED by something → a backdrop occasion, always
            self._learn_press(tag, felt, beyond, eff, observed)
            return
        base = self._press(tag, felt, None, eff)
        if (not impeded and self._confident(self._cue("press", tag, felt, None), eff)
                and norm(sub(observed, base)) > 1e-9):             # free yet refuted → the object is WORLD-anchored
            self._learn_world(tag, felt, observed)
            return
        self._learn_press(tag, felt, None, eff, observed)          # a free contact → the base learns it

    def _beyond(self, objs, cell):
        """What backs a contact: the feature at `cell`, or the "edge" beyond the board, or None (in-bounds and empty). The
        boundary is a perceived backdrop like any other, so an object pressed against the edge mints an EDGE occasion rather
        than being mistaken either for an open-space push (which would wrongly teach the base "it stays") or for a world-
        anchored object (which would wrongly mint a world override). Matches `WorldMap.occupant`, so learning and the rollout
        see the same backdrop."""
        ext = getattr(self, "_extent", None)
        if ext is not None and any(not (lo <= c <= hi) for c, (lo, hi) in zip(cell, ext)):
            return "edge"
        return self._feature_at(objs, cell)

    def _feature_at(self, objs, cell):
        """The feature occupying `cell` in a perceived frame, or None if it is empty — the plain perceptual question the
        contact's cue set is built from."""
        for o in objs:
            if cell in o.cells:
                return o.color
        return None

    def _static_cells(self, objs, sc) -> dict:
        """`{cell: feature}` for every cell holding a non-self, non-mover object — RAW PERCEPTION, not a conclusion. This is what
        replaced the hand-maintained obstacle set: we no longer record "the agent bumped here so this cell is impassable"; we
        record what is SEEN, and L5 predicts what pressing into that feature does. The payoff is generalisation — bump one wall
        cell and every other cell of that feature is predicted impassable without ever being touched."""
        out = {}
        for o in objs:
            if o.color != sc and o.color not in self._movers:
                for cell in o.cells:
                    out[cell] = o.color
        return out

    def _as_pose(self, cell):
        """A grid cell → a continuous pose `(position, R=identity)` in the nav frame."""
        return (tuple(float(c) for c in cell), eye(self._dims))

    def _positions(self, objs) -> dict:
        """`{feature: anchor}` for each feature realised by EXACTLY ONE object this frame — the single-cell tracking key that
        lets a mover be followed across frames (recognition disambiguates same-feature objects later; deferred)."""
        by_color: dict = {}
        for o in objs:
            by_color.setdefault(o.color, []).append(o)
        return {c: os[0].anchor for c, os in by_color.items() if len(os) == 1}

    def _track_movers(self, prev_pos, pos, sc) -> None:
        """A NON-self object whose cell changed between frames is a controllable MOVER — discovered from motion (like the self),
        never from colour semantics. Movers are the objects a rollout must simulate as pushable (routed to the scene column).
        Gated on the self being KNOWN, or the self (which moves every step) would be mislabelled a mover before it is identified."""
        if sc is None:
            return
        for c, cell in pos.items():
            if c != sc and c in prev_pos and prev_pos[c] != cell:
                self._movers.add(c)

    def _pushed(self, prev_pos, pos) -> bool:
        """Did any known mover change cell in this transition? (So a self non-move that PUSHED a box is not mislabelled a wall bump.)"""
        return any(c in prev_pos and c in pos and prev_pos[c] != pos[c] for c in self._movers)

    def _skin(self, grid, objs, cur) -> dict:
        """The SKIN sense over the body surface (the TOUCH modality's transducer) — what is pressed against each face of the
        self's body right now. The AGENCY signal: an object the body FELT (here) that then moves was moved by the SELF, not by a
        collision between other objects. `{}` until the self is localised."""
        body = self._body_cells(objs, cur)
        return self._modalities["touch"].transduce(grid, body) if body else {}

    def _body_cells(self, objs, cur) -> set:
        """The self's BODY — the cells of the discovered controllable object (single-cell = `{its cell}`); empty until localised."""
        c = self.self_color()
        if c is None or cur is None:
            return set()
        for o in objs:
            if o.color == c and o.anchor == cur:
                return set(o.cells)
        return {cur}

    def _learn_dynamics(self, action, prev_objs, prev_skin, prev_pos, pos, prev_cur, cur, sc) -> None:
        """Teach the L5 TRANSFORM from one felt interaction (`behavior.py`, `notes/touch_and_body_design.md` §7). The self FELT
        something at its leading face (AGENCY — the skin, in the efference direction), so this transition is SELF-CAUSED and may
        be attributed; an object that moved without being felt (a box shoved by another box) teaches nothing here, which is why
        touch and not geometry grounds it. Two facts, both plain deltas, neither of them a category:

        * ("into", felt) — the CORRECTION to the body's own free motion when pressing into `felt`. Zero when it gave way, −eff
          when it did not. Solidity is that correction, learned; there is no "resist".
        * ("of", felt)   — the FELT thing's own displacement under this press. Zero when it stayed. There is no "yield" either.

        The body gets a correction and the felt thing an absolute delta because only the body has a free-motion baseline to
        correct — the efference copy. That asymmetry is the anatomy, not a special case."""
        if prev_cur is None or not getattr(action, "is_movement", False) or not prev_skin:
            return
        # the direction the body PRESSED: its ACTUAL displacement if it advanced (no operator needed, which is what lets the
        # FIRST move in a direction, even if it is the push, be learned), else the operator's PREDICTED free move (a press that
        # went nowhere, whose direction the body never actually took).
        if cur is not None and cur != prev_cur:
            eff = sub(tuple(float(c) for c in cur), tuple(float(c) for c in prev_cur))
        else:
            op = self._nav_col().operator
            if not op.known(action):
                return
            eff = sub(op.apply(self._as_pose(prev_cur), action)[0], tuple(float(c) for c in prev_cur))
        felt = contact_toward(prev_skin, eff)                                   # the thing the body pressed into (agency)
        if felt is None or felt == sc:
            return
        z = tuple(0.0 for _ in range(self._dims))
        body_disp = sub(tuple(float(c) for c in cur), tuple(float(c) for c in prev_cur)) if cur is not None else z
        obj_disp = (sub(tuple(float(c) for c in pos[felt]), tuple(float(c) for c in prev_pos[felt]))
                    if felt in pos and felt in prev_pos else z)
        contact = tuple(int(round(c + d)) for c, d in zip(prev_cur, eff))       # the cell the body pressed into
        beyond = self._beyond(prev_objs, tuple(int(round(c + d)) for c, d in zip(contact, eff)))
        impeded = norm(sub(body_disp, eff)) > 1e-9                               # the body did NOT advance by its full efference
        self._learn_delta("into", felt, beyond, eff, sub(body_disp, eff), impeded)
        self._learn_delta("of", felt, beyond, eff, obj_disp, impeded)

    def _route_movers(self, pos) -> None:
        """Place every currently-visible known mover into the SCENE column, so `world_state()` (hence the rollout) includes it.
        STATIC features (pads, goals, walls — never seen to move) are NOT routed: they are landmarks for the goal predicate, not
        simulated bodies."""
        if not self._movers:
            return
        self.clear_scene()
        for c in self._movers:
            if c in pos:
                self.place_object(c, self._as_pose(pos[c]))

    def _target_cell(self, cell, action):
        """The cell `action` moves the self INTO from `cell`, via the LEARNED operator (not the action's declared delta — the
        effect is discovered, `reference_l5_operator_kinds`). None until the operator has learned this action."""
        op = self._nav_col().operator
        if not op.known(action):
            return None
        tgt = op.apply(self._as_pose(cell), action)
        return tuple(round(c) for c in tgt[0])

    def _static_feature_at(self, objs, cell, sc):
        """The STATIC feature at a cell — the colour of a non-self, non-mover object there (a landmark: pad, goal tile) — or None
        if the cell is empty or occupied by the self / a mover. What a discovered GOAL attaches to (a place-to-reach, not a body)."""
        if cell is None:
            return None
        for o in objs:
            if cell in o.cells:
                return o.color if (o.color != sc and o.color not in self._movers) else None
        return None

    def _credit_goal(self, prev_objs, prev_pos, action, prev_cur, reward, sc) -> None:
        """Discover the goal by the delta rule (`goal_mem`), REACHER-typed so the two credit paths never confound. The SELF
        reaching a static landmark credits the bare feature (a NAV goal — LockPath). A MOVER the self PUSHES onto a static
        landmark credits the PAIR `(mover, landmark)` (a RELATIONAL goal — 'put the box on the pad'). The self reaching a mover is
        the PUSH itself, never a goal (the dynamics own it), so it is never credited — which keeps 'reach the box' from
        masquerading as the goal. Both landings are INFERRED from the pre-move frame + the action (the winning frame is never
        returned — the board advances on completion), so the credit is boundary-safe."""
        self_target = self._target_cell(prev_cur, action)
        reached = self._static_feature_at(prev_objs, self_target, sc)      # the static landmark the SELF stepped onto
        if reached is not None:
            self.goal_mem.credit({reached}, reward)
        for c in self._movers:                                            # a mover the self PUSHED (stepped into its cell)
            if c in prev_pos and self_target == prev_pos[c]:
                landing = self._target_cell(prev_pos[c], action)          # the pushed mover advances one cell along the push
                landmark = self._static_feature_at(prev_objs, landing, sc)
                if landmark is not None:
                    self.goal_mem.credit({(c, landmark)}, reward)

    def _self_pos(self, objs):
        """The self's cell this frame — the anchor of the object whose colour is the discovered controllable root; None until
        the root is discovered, or if it is ambiguous (more than one object of that colour)."""
        c = self.self_color()
        if c is None:
            return None
        selves = [o for o in objs if o.color == c]
        return selves[0].anchor if len(selves) == 1 else None

    def _act(self, objs, cur, pos, available):
        """The one EFE planner (`reference_efe_and_epiplexity`, `feedback_epistemic_value_is_prediction_error`): PRAGMATIC toward
        the discovered goal (a NAV goal — self to a landmark; or a RELATIONAL goal — the rollout PUSHES a mover onto a landmark),
        else EPISTEMIC toward the nearest thing the MODEL HAS NOT YET LEARNED (`_unlearned_cells`). Bootstraps the operator by
        trying UNLEARNED actions first; random when the self is unknown or nothing is reachable."""
        movement = [a for a in available if getattr(a, "is_movement", False)]
        if not movement:
            return self._rng.choice(available)
        if cur is None:                                              # self not discovered yet → move to generate the evidence
            return self._rng.choice(movement)
        op = self._nav_col().operator
        unlearned = [a for a in movement if not op.known(a) and (cur, a) not in self._tried]
        if unlearned:                                               # bootstrap: try an unlearned action HERE (once per cell, so an
            a = self._rng.choice(unlearned)                        #   always-blocked one doesn't loop — it's learned wherever it works)
            self._tried.add((cur, a))
            return a
        self.set_pose(self._as_pose(cur)[0], eye(self._dims))       # anchor the nav pose to the perceived self
        # ARBITRATE the two drives IN THE BASAL GANGLIA — the one organ allowed to arbitrate (ARCHITECTURE rule 4; this was a
        # hard if/else ladder, i.e. arbitration outside the BG). Priority = salience ⊕ the learned Go/NoGo value, with TONIC
        # DOPAMINE as the gain: what each drive OFFERS right now is its salience (the goal's learned reward contingency; the
        # rate the model is currently learning), the BG learns which drive pays in THIS game, and ρ shifts the balance —
        # rich rate ⇒ commit and exploit, collapsed rate ⇒ the patch is spent, go and seek.
        # The cortex proposes what each drive OFFERS (the goal's learned reward contingency; the rate the model is learning);
        # the BG selects on that salience ⊕ its own learned Go/NoGo value, with tonic DA as the gain.
        # KNOWN LIMIT (measured, `STATUS.md`): the goal's contingency is a constant that cannot FALL, so a pragmatic drive that
        # has stopped paying still wins — which is why LockPath stalls on L1. Affordance-only salience fixes that (2 levels) but
        # costs Push its oracle (6 → 10), because the BG's drive value is dominated by whichever drive was active when the
        # first reward landed — always exploration. The fix both need is a per-drive payoff RATE (marginal value theorem).
        g = self.goal_mem.goal()
        targets = self._unlearned_cells(pos)
        salience = [self.goal_mem.w.get(g, 0.0) if g is not None else float("-inf"),
                    self.progress.progress() if targets else float("-inf")]
        if max(salience) > float("-inf"):
            mode = self.bg.select(self._MODE_CTX, 2, rho=self.critic.rho(), salience=salience)
            self._last_mode = mode
            plan = self._goal_plan(objs, cur, pos, movement) if mode == 0 else self._seek_plan(targets, movement)
            if not plan:                                           # the selected drive had nothing reachable → try the other
                plan = self._seek_plan(targets, movement) if mode == 0 else self._goal_plan(objs, cur, pos, movement)
            if plan:
                self._last_plan = plan
                return plan[0]
        self._last_plan = []
        return self._rng.choice(movement)                          # nothing to pursue and nothing left to learn

    def _seek_plan(self, targets, movement):
        """The EPISTEMIC drive's plan: reach something the model cannot yet predict (None if there is nothing, or nothing
        reachable). The pragmatic drive's counterpart is `_goal_plan`; the BG chooses between them."""
        if not targets:
            return None
        reach = lambda w: 1.0 if (w.agent is not None and tuple(round(c) for c in w.agent[0]) in targets) else 0.0
        return self.plan(reach, movement, horizon=32)

    def imagine(self, actions=None):
        """The agent's IMAGINED future: unroll a plan (default: the last committed one) through the LEARNED forward model from the
        current world-state, returning the sequence of imagined world CELLS `[{'agent': cell, 'objects': {id: cell}}, ...]`. This
        is the substrate for the two-pane imagination widget (`project_hippocampus_imagination_and_widget`): laid beside the real
        frames, the imagined trajectory either tracks reality or visibly DIVERGES — and a divergence localises the model's error."""
        actions = self._last_plan if actions is None else actions
        world, model = self.world_state(), self.world_model()
        traj = [self._world_cells(world)]
        for a in actions:
            world = model.step(world, a)
            traj.append(self._world_cells(world))
        return traj

    @staticmethod
    def _world_cells(world) -> dict:
        """A world-state's occupied CELLS (integer positions) — the agent and each object — for rendering / comparison."""
        agent = tuple(round(c) for c in world.agent[0]) if world.agent is not None else None
        return {"agent": agent, "objects": {oid: tuple(round(c) for c in p[0]) for oid, p in world.objects.items()}}

    def _goal_plan(self, objs, cur, pos, movement):
        """Plan toward the discovered goal (empty ⇒ none / already satisfied / unreachable, so the caller explores). A bare-feature
        goal = navigate the SELF to that feature's cell (the rollout over agent moves). A `(mover, landmark)` goal = rollout the
        MOVER onto the landmark's cell — the PUSH, where the rollout beats a one-step value (`project_linear_value_cannot_hold_sokoban`)."""
        g = self.goal_mem.goal()
        if g is None:
            return None
        if isinstance(g, tuple):                                    # RELATIONAL: put mover g[0] onto a g[1] landmark cell
            mover_c, landmark_c = g
            if mover_c not in pos or landmark_c not in pos:
                return None
            target = pos[landmark_c]
            on_target = lambda w: 1.0 if (mover_c in w.objects
                                          and tuple(round(c) for c in w.objects[mover_c][0]) == target) else 0.0
            return self.plan(on_target, movement, horizon=48)
        goals = [o for o in objs if o.color == g]                   # NAV: navigate the self to the landmark
        if len(goals) != 1 or goals[0].anchor == cur:
            return None
        goal_cell = goals[0].anchor
        act = self._nav_inverse(cur, goal_cell, movement)          # CHEAP default: the L6a→L5 inverse read-out, BG-selected
        if act is not None:
            return [act]
        at_goal = lambda w: 1.0 if (w.agent is not None and tuple(round(c) for c in w.agent[0]) == goal_cell) else 0.0
        return self.plan(at_goal, movement, horizon=64)            # SPARING fallback: deliberate when the read-out stalls

    def _nav_inverse(self, cur, target, movement):
        """The NAV INVERSE MODEL, routed through its real anatomy (`notes/l5_unified_transform_design.md` §4). The agent only
        ROUTES (`feedback_thin_shell_agent`): it hands the nav column the goal VECTOR (target − here — the hippocampal
        goal-vector), the column's **L5IT** projection proposes `(per-action drive, context)` by reading its own L5 transform
        backwards, the learned FORWARD MODEL vetoes channels it predicts go nowhere, and the **basal ganglia** selects on
        salience ⊕ its own value.
        O(actions), no search. Returns None when nothing reduces the goal vector (a concave obstacle), so the caller
        DELIBERATES instead (`reference_brain_planning`: cheap read-off by default, rollout sparingly)."""
        drive, context = self._nav_col().striatum_proposal(
            movement, sub(tuple(float(c) for c in target), tuple(float(c) for c in cur)))
        if not drive:
            return None
        world, model = self.world_state(), self.world_model()      # the veto: one imagined step per action, O(actions), no search
        salience = []
        for a in movement:
            stuck = norm(sub(model.step(world, a).agent[0], world.agent[0])) <= 1e-9
            salience.append(float("-inf") if (a not in drive or stuck) else drive[a])
        if max(salience) <= 0.0:
            return None                                            # no action closes the gap → hand over to the rollout
        return movement[self.bg.select(context, len(movement), salience=salience)]
