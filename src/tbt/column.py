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

import torch
import torch.nn as nn

from .l4_feature_location import L4_FeatureLocation
from .l5_displacement import L5_Displacement
from .l6_grid import L6_GridLocation
from .l6_sr import OnlineSR                            # the online TD successor representation (eigendecomposition-free L6)
from .l23_object import L23_Object                     # the object/identity layer: graph-memory + recognition + voting


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
