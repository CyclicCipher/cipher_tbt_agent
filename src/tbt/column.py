"""The cortical column — a CONTAINER for the four layers and a COORDINATOR of information flow between them and
out to the thalamus / other columns. It holds no layer functionality of its own (the coordinator principle,
REFACTOR_PLAN): math + state live in the layers; the column only routes.

  L6  the LOCATION frame. The online TD successor representation (`l6_sr.OnlineSR`, eigendecomposition-free —
      Stachenfeld 2017: grid cells ARE the SR eigenvectors, topology-general) is the navigational / relational
      frame; the innate hex grid (`l6_grid`) is a shelved metric prior. Path integration = discrete graph
      tracking (`loc_*`).
  L5  the DISPLACEMENT / operator / motor / thalamus-driver layer. The per-action operator (observed edges +
      the position-invariant generalizing delta) AND the continuous-pose group operators recognition reads.
  L4  FEATURE-at-location. The label-free content codebook (`encode`), the rotation-invariant feature
      descriptor, feature ⊗ location bind/readout, and `predict_feature`.
  L23 the OBJECT / identity layer. The graph-memory of objects + evidence-based recognition (pose inferred) +
      lateral CMP voting; plus the within-object content store S.

The object MODEL is DISTRIBUTED across the layers (L6 locations + L4 features + L5 displacements + L23 identity)
— there is no separate object library (the dissolved `recognize.py`). The column's methods are thin routing:
`observe` feeds L6 + L5; `predict`/`motor`/`driver` delegate to L5; `learn_object`/`recognize_object` route to
L23; `content_code`/`place_code` expose the thalamus interface; `refresh` builds L6 place codes from the online
SR; `loc_*` coordinate L6's belief with L5's operator (path integration).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .l4_feature_location import L4_FeatureLocation
from .l5_displacement import L5_Displacement
from .l6_grid import L6_GridLocation
from .l6_sr import OnlineSR                            # the online TD successor representation (eigendecomposition-free L6)
from .l23_object import L23_Object                     # the object/identity layer: graph-memory + recognition + voting


@dataclass(frozen=True)
class GoalState:
    """A CMP-style goal-state MESSAGE -- active inference: L5 emits the desired OUTCOME, not a command
    ('predictions not commands'; reference_gsg_goal_generation). `target` = the location/pose to bring about (here,
    to go SENSE); `kind` = the uncertainty it resolves ('disambiguate' now; 'explore'/'reward' later); `source` =
    the originating column (None = self). Shaped so a column can SELF-generate it OR RECEIVE it from a connected
    column -- the heterarchy scale-up is just WHERE the message comes from, not a different mechanism."""
    target: tuple
    kind: str = "disambiguate"
    source: object = None


class CorticalColumn(nn.Module):
    def __init__(self, n_entities, feat_dim=256, d_mem=512, torus_size=12, scales=(11, 13, 17),
                 place_k=1, seed=0):
        super().__init__()
        self.L6 = L6_GridLocation(torus_size=torus_size, scales=scales, place_k=place_k)   # innate metric PRIOR (off; SR is the substrate)
        self.L5 = L5_Displacement()                                       # displacement / operator / motor / driver
        self.L4 = L4_FeatureLocation(n_entities, feat_dim=feat_dim, seed=seed)  # feature-at-location + content codebook
        self.L23 = L23_Object(feat_dim=feat_dim, d_mem=d_mem)             # object/identity: graph-memory + recognition + content store
        self.d_mem = d_mem
        self.sr = OnlineSR(gamma=0.95, alpha=0.3)             # L6: the location frame, learned ONLINE by TD (no batch eigh)
        self.loc, self.rel = {}, {}                           # symbol→frame index; relation→operator key (the EDGES live in L5)
        self.place = None                                     # (n, d_mem) per-node SR-frame place codes (set by refresh)
        self._cur = None                                      # the current node -- discrete path integration over the graph
        self._map = torch.zeros(feat_dim, d_mem)             # M5/L7-A: the online allocentric MAP (Σ feature ⊗ place), read by feature_at

    # ----- routing: learn the transition structure online (L6 + L5) ---------------------------------
    @property
    def graph(self):
        """A read view of the per-action operator's edges -- they live in L5 (the operator layer), not a bare dict."""
        return self.L5.edges

    def observe(self, s, a, s2):
        """Record one observed transition: it feeds L6 (the ONLINE SR location frame, TD every step -- no batch eigh) AND
        L5 (the per-action operator -- the edges; a blocked move is a state-dependent exception with no edge). The
        geometry -- line, ring, 2-D grid, tree -- falls out of the SR; no metric-vs-non-metric switch."""
        self.sr.observe(s, s2)                                       # L6: online location learning (incl. blocked moves)
        self.L5.observe(s, a, s2)                                    # L5: the per-action operator (edges)

    def refresh(self):
        """ONLINE, eigendecomposition-free consolidation: take the online SR (L6, TD-learned by `observe`) rows as place
        codes (padded into d_mem), then learn the L5 operators + pool L4/L23 content from them with cheap outer products
        (no `eigh`). Builds the place codes the thalamus interface (`place_code`) reads. (Up to d_mem states are padded
        in; beyond d_mem a random projection is needed -- deferred.)"""
        syms = sorted(self.sr.idx, key=lambda s: self.sr.idx[s])
        if not syms:
            return
        self.loc = dict(self.sr.idx)
        codes = torch.zeros(len(syms), self.d_mem)
        for s in syms:                                               # SR row -> place code (padded into d_mem)
            row = torch.as_tensor(self.sr.code(s), dtype=torch.float32)
            codes[self.loc[s], :row.shape[0]] = row[:self.d_mem]
        self.place = torch.nn.functional.normalize(codes, dim=1)
        relations: dict = {}
        for s, e in self.graph.items():
            for a, s2 in e.items():
                if s in self.loc and s2 in self.loc:
                    relations.setdefault(a, []).append((self.loc[s], self.loc[s2]))
        self.rel = {}
        for a, edges in relations.items():                          # operators from the current codes (outer products; no eigh)
            self.L5.learn(("rel", a), self.place, edges)
            self.rel[a] = ("rel", a)
        for s, i in self.loc.items():                              # content bound at the online place codes (L4 -> L23 store)
            self.L23.pool(self.L4.bind(s, self.place[i]))

    # ----- routing: the SR value / reachability read (L6, the deep-planning substrate) --------------
    def value(self, s, reward_map):
        """The SR VALUE V(s) = M[s]·R over the (sparse) rewarding states -- the expected discounted future reward as a
        few cached lookups, the deep multi-step propagation precomputed into the SR row (no rollout; reference_brain_
        planning). Routes to L6's online SR; an unknown state -> 0."""
        return self.sr.value(s, reward_map)

    def reachable(self, s, reward_map) -> bool:
        """Whether any REWARD state is reachable from `s` via the learned transitions -- read NATIVELY from the SR
        (M[s,g] > 0 iff g is reachable = nonzero discounted future-occupancy), no graph BFS. The per-level dead-zone
        signal: a fresh level whose states have no SR path to a reward is a fresh dead-zone, not a global 'reward-ever'
        flag (the bug the oracle/human/agent trace exposed)."""
        return self.value(s, reward_map) > 1e-9

    def locate(self, state):
        """C1 (COLUMN_AUDIT) — the column's WHERE: L6 READ as the location substrate. Returns `state`'s L6
        SR-eigenframe place code (in the d_mem binding space) -- the location L4 binds a feature to (C2) and L5
        path-integrates (C3). Topology-encoding (nearby-in-graph states get similar locations); `None` for a state the
        L6 frame has not seen. Closes the doc's 'L6 is updated but not READ' loose thread."""
        if state not in self.sr.idx:
            return None
        return self._place_code(state)

    # ----- routing: the feature-at-location MAP (M5 / L7-A: L4 feature ⊗ L6 location) ----------------
    def _place_code(self, loc):
        """The L6 place code for `loc` -- its online SR row, DG-SPARSIFIED (top-k active units) and padded into d_mem
        (the binding space). The raw SR row at gamma~0.95 is too DIFFUSE for binding (all states ~0.98 similar -> the
        feature-at-location map degenerates to a global bag); sparse pattern separation makes distant locations
        near-ORTHOGONAL (their top-k units are disjoint) while nearby ones still OVERLAP (topology kept) -- the
        dentate-gyrus orthogonalisation (reference_brain_reference_frames_orthogonalization)."""
        code = torch.as_tensor(self.sr.code(loc), dtype=torch.float32)    # the L6 place code (normalized SR row)
        n = code.shape[0]
        k = max(2, n // 4)                                               # DG sparsity: keep the top-k magnitudes
        if k < n:
            keep = torch.zeros_like(code)
            idx = torch.topk(code.abs(), k).indices
            keep[idx] = code[idx]
            nrm = float(keep.norm())
            code = keep / nrm if nrm > 0 else keep
        p = torch.zeros(self.d_mem)
        m = min(n, self.d_mem)
        p[:m] = code[:m]
        return p

    def bind_at(self, loc, feature_id: int) -> None:
        """M5/L7-A: bind a SENSED feature at a LOCATION into the online allocentric MAP (L4 feature ⊗ L6 place code),
        accumulated across the sensorimotor sequence -- 'features at locations' (Monty). The map the agent reads to
        REMEMBER the layout (an object seen then left is still mapped) -- the substrate the §3 mechanic library needs."""
        if loc in self.sr.idx:
            self._map = self._map + self.L4.bind(feature_id, self._place_code(loc))

    def feature_at(self, loc):
        """Read the MAP: the feature predicted at `loc` (L4 readout over the accumulated map) -- the predict half of
        predict-then-compare, seated in L4. `None` for a location unknown to the L6 frame OR one where nothing is
        CONFIDENTLY bound yet (a near-zero readout -> no prediction, so a fresh location is not a false surprise)."""
        if loc not in self.sr.idx:
            return None
        scores = self.L4.readout(self._map, self._place_code(loc))
        return int(scores.argmax()) if float(scores.max()) > 0.5 else None

    def sense_at(self, location, sensed_feature: int) -> bool:
        """C2 (COLUMN_AUDIT) -- the TBT cycle step, L4 over L6: PREDICT the feature at the L6 `location` (from the map),
        COMPARE to the `sensed_feature`, then LEARN by binding the sensed feature there. Returns True if SURPRISED (a
        confident prediction MISSED) -- the single predict-then-compare learning signal (HTM burst). The object EMERGES
        as the accumulated feature-at-location map; a persistent mismatch is a boundary (a different object)."""
        predicted = self.feature_at(location)
        surprised = predicted is not None and predicted != sensed_feature
        self.bind_at(location, sensed_feature)                            # learn: bind the sensed feature at the location
        return surprised

    # ----- routing: the L5 operator (predict / motor / driver) --------------------------------------
    def predict(self, symbol, action):
        """Where `action` leads from `symbol` -- the L5 operator (the efference copy). L5 owns the per-action operator:
        the observed EDGES are the state-dependent exceptions (each (s,a) its own next state; a wall/door is a blocked
        self-edge), and the position-invariant DISPLACEMENT GENERALIZES the operator to UNVISITED (s,a). The online SR
        (L6) carries value/topology; recognition carries continuous pose. Edge first, else displacement, else stay."""
        return self.L5.predict(symbol, action)

    def motor(self, action):
        """The MOTOR output -- the enacted action (L5 is the cortex's output layer; the name->GameAction mapping is the
        motor organ in arc_sdk). The chosen displacement IS the motor command, the efference copy, and the driver."""
        return self.L5.motor(action)

    def driver(self, symbol, action):
        """The feed-forward DRIVER message to other columns (via the higher-order thalamus): the displacements `action`
        causes among the features in `symbol` -- L5's trans-thalamic output (Sherman & Guillery)."""
        return self.L5.driver(symbol, action)

    # ----- the generative forward model (FM1): L5's operator at LOCATION grain over L4's field ----------
    # The column reads L4 (feature-at-location) and indexes by L6 (the frame); L5 owns the per-location operator.
    # This is the TEM objective -- predict the next sensory observation (L4 content) at each position (L6) given the
    # action -- seated as a COLUMN capability, never a raw-pixel buffer (reference_brain_generative_model). The
    # whole-object disp/recolor stay the coarser form of the SAME operator.
    def feature_field(self, frame):
        """L4's feature-at-location field for a frame: each location's L4 feature id (cell grain: the colour is the
        descriptor `(v,)` -> `L4.encode`, growing the content vocabulary online). The field the forward model predicts
        IN -- content x bound to location g, the TEM canvas at cell grain."""
        return [[self.L4.encode((v,)) for v in row] for row in frame]

    def observe_field(self, field, action, next_field):
        """Learn one feature-field transition (route to L5's per-location operator)."""
        self.L5.observe_field(field, action, next_field)

    def predict_field(self, field, action):
        """Predict the next feature-field via L5's per-location operator -- the field-grain efference copy."""
        return self.L5.predict_field(field, action)

    def act(self, state, actions, value, explore, tried, rng, bonus=None):
        """The MOTOR as an INVERSE MODEL (the action-selection seat -- in the COLUMN, not the agent script). Choose the
        action whose predicted effect (L5's forward operator, via the learned graph) is most VALUABLE -- i.e. INVERT
        the operator against `value` to find the action that best achieves the highest-value next-state (the implicit
        goal-state of active inference: act to bring about the preferred prediction). An UNTRIED (state, a) takes the
        frontier `explore` optimism (its outcome is uncertain -> resolving T(s,a) is epistemically valued). This
        GENERALISES: a continuous effector inverts the SAME operator against the SAME value -- only the organ (discrete
        action here) differs. `value(s)` is the planned EFE value of state `s` (supplied by the agent's reward model).
        `bonus` is the forward model's per-action value (pragmatic + epistemic). The ONE-MODEL arbitration is the
        CALLER's: it supplies `bonus` ONLY when the tabular value is INDIFFERENT (no spread across actions) -- so a
        converged tabular decision is never disturbed, and on a dynamics game (flat tabular value) the forward model
        fills the vacuum and decides. Here `bonus` is simply added (it is None / absent when the tabular value leads)."""
        vals = []
        for a in actions:
            nxt = self.graph.get(state, {}).get(a, state)
            if (state, a) not in tried:
                v = explore                                     # untried -> bounded, decaying frontier optimism
            else:
                v = value(nxt)                                  # tried -> the value of its outcome
            vals.append(v + (bonus.get(a, 0.0) if bonus is not None else 0.0))   # + the forward-model value (when tabular is indifferent)
        best = max(vals)
        return rng.choice([a for a in actions if vals[a] == best])

    # ----- routing: the "what + pose" recognition faculty -> L2/3 (the object/identity layer) --------
    # Complementary to L6 (the "where"): L6 recognises navigable locations in a fixed frame; L2/3 SOLVES an object's
    # pose (via L5's pose operators) so a known object is recognised at an orientation never seen. The column merely
    # routes the call to the layer that owns the graph-memory -- the object library is NOT a column faculty.
    def learn_object(self, cloud, name=None):
        """Add an object to L2/3's graph-memory. `name` given → store under it; else learn ONLINE + label-free
        (recognise-or-add). Returns the ObjectGraph (named) or (name, is_new) (online)."""
        return self.L23.learn(cloud, name=name)

    def recognize_object(self, cloud):
        """Identify a sensed point cloud's (name, theta, t, evidence) at a pose never seen, learning it online if
        novel — pose-invariant recognition (object permanence under rotation)."""
        return self.L23.recognize(cloud)

    def identify_object(self, cloud):
        """Recognise a sensed shape against L2/3's library WITHOUT adding a new one — the name, or None."""
        return self.L23.identify(cloud)

    # ----- the GOAL STATE GENERATOR (per-column GSG; the column proposes, L5 emits) -------------------
    def propose_goal(self):
        """The column's Goal State Generator. TBT-faithful: its CORE is UNCERTAINTY-RESOLUTION -- when L2/3's top
        (object, pose) hypotheses still compete, propose a hypothesis-TEST goal (the graph-mismatch sample point
        that most discriminates them) as a message-shaped `GoalState`; None when there is nothing to resolve. The
        value / transition-`lp` goal candidates + the basal-ganglia arbitration among them are the next build
        steps; the heterarchy adds RECEIVED goal-messages to the same competition (reference_gsg_goal_generation)."""
        target = self.L23.disambiguation_goal()
        return GoalState(target=target, kind="disambiguate") if target is not None else None

    def examine(self, sense_at, first, max_samples: int = 12, confident: float = 2.0):
        """ACTIVE recognition -- the GSG DIRECTING the motor, in Monty's order: PASSIVE-narrow then hypothesis-TEST.
        Begin a session with `first` (loc, disps). Each step: if the GSG fires (the field has NARROWED -- the
        graph-mismatch on the top-2 is now worthwhile), COVERTLY pick the disambiguating point and OVERTLY sample it;
        ELSE (not yet narrowed) PASSIVELY sample the leading hypothesis's nearest UNSENSED predicted cell to gather
        evidence. The motor sample is `sense_at(target) -> (loc, disps)` for a cell there, or `None` for empty
        (present -> `sense` confirms a predictor; absent -> `sense_absent` falsifies them). Stop when one hypothesis
        leads by `confident` (or budget). Returns `(best, n_overt_samples)`. The GSG spends each ACTIVE sample where
        it resolves the most uncertainty (think, then act); the live loop arbitrates this against value/explore goals
        via the basal ganglia."""
        import numpy as np

        def _key(p):
            return (round(float(p[0]), 1), round(float(p[1]), 1))

        self.L23.start()
        self.L23.sense(*first)
        sensed = {_key(first[0])}
        n = 1
        while n < max_samples:
            hyps = self.L23.hyps
            if len(hyps) < 2:
                break
            ev = sorted((h.ev for h in hyps), reverse=True)
            if ev[0] - ev[1] >= confident:                  # a hypothesis leads clearly -> resolved
                break
            goal = self.propose_goal()                      # the GSG fires only once narrowed
            if goal is not None:
                target = goal.target                        # ACTIVE: the graph-mismatch discriminating point
            else:                                           # PASSIVE: narrow via the leading hypothesis's next cell
                top = max(hyps, key=lambda h: h.ev)
                cands = [p for p in top.obj.cells_at(top.theta, top.t) if _key(p) not in sensed]
                if not cands:
                    break
                ref = self.L23.prev if self.L23.prev is not None else np.zeros(2)
                target = tuple(round(float(x), 3) for x in min(cands, key=lambda p: np.linalg.norm(np.asarray(p, float) - ref)))
            obs = sense_at(target)                          # OVERT: the motor samples at the target
            n += 1
            if obs is None:
                self.L23.sense_absent(target)               # empty -> falsify the predictors
            else:
                self.L23.sense(*obs)                        # a cell there -> confirm
                sensed.add(_key(obs[0]))
        return self.L23.best(), n

    def propose_goals(self, act_value, g_value=0.0, effort=0.0):
        """The candidate goals the BASAL GANGLIA arbitrates (Cisek's affordance competition): ALWAYS the ACT goal
        (pursue the value/explore policy via the inverse-model motor; value = `act_value` = the best action's EFE
        value), PLUS the DISAMBIGUATION goal when L2/3's hypotheses still compete (value = `g_value` = the epistemic
        value of resolving the identity, supplied on the same EFE scale by the caller). `effort` is the per-distance
        cost the EFFICIENCY tie-breaker charges the disambiguation goal for the trip to its target (from the current
        sensor locus) -- so a FAR test is less attractive than acting, but a uniquely-worth-it test still wins
        (g_value >> effort*dist). Returns [(GoalState, value), ...]; the BG selects one. The heterarchy adds
        RECEIVED goal-messages to this same list -- the competition is the only mechanism, wherever a candidate came from."""
        goals = [(GoalState(target=None, kind="act"), float(act_value))]
        dg = self.propose_goal()
        if dg is not None:
            v = float(g_value)
            if effort > 0.0 and self.L23.prev is not None:          # the goal-distance tie-breaker (efficiency)
                dx, dy = dg.target[0] - float(self.L23.prev[0]), dg.target[1] - float(self.L23.prev[1])
                v -= effort * (dx * dx + dy * dy) ** 0.5
            goals.append((dg, v))
        return goals

    # ----- the inter-column interface (the thalamus binds content ⊗ location) -----------------------
    def content_code(self, label):
        """This column's content (What / L4) code for an entity label."""
        return self.L4.E[label]

    def place_code(self, node):
        """This column's location (Where / SR-frame) place code for a node — built by `refresh` from the online SR."""
        return self.place[self.loc[node]]

    # ----- routing: path integration as DISCRETE graph tracking (L6 belief ⊕ L5 operator) ----------
    # NOT a matrix operator over codes: the brain path-integrates by a continuous-attractor bump shifted by velocity,
    # with discrete-attractor SNAPPING on a clear sighting (reference_brain_reference_frames_orthogonalization). Here
    # that is exact + online -- PREDICT the next node by the learned edge (L5; the efference copy, so it survives
    # PARTIAL observability), CORRECT by snapping to a sensed node. The column coordinates L6's location belief with
    # L5's operator; the online SR carries value/topology, recognition carries continuous pose.
    def loc_reset(self, node):
        """Begin dead reckoning at a known node."""
        self._cur = node
        return node

    def loc_move(self, action):
        """PREDICT: path-integrate by the L5 operator (the efference copy) -- no observation needed. An unobserved or
        blocked move stays put."""
        self._cur = self.L5.predict(self._cur, action)
        return self._cur

    def loc_sense(self, node):
        """CORRECT: snap the belief to an observed node (the discrete-attractor sighting). Call only when a node is
        actually sensed; otherwise keep dead-reckoning via loc_move."""
        self._cur = node
        return self._cur

    def loc_where(self):
        """The current node."""
        return self._cur
