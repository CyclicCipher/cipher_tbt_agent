"""The cortical column — ONE mechanism: learn a structural map, then predict from it.

Every domain (metric, relational, abstract, spatial) is handled the same way through the four layers —
no parallel programs on different operating principles:

  L6  structure / location code — the SR-EIGENVECTOR FRAME of the transition graph (Stachenfeld 2017:
      grid cells ARE the SR eigenvectors). Hexagonal is just the eigenbasis of open 2-D, so the frame comes
      out grid-like on metric graphs and gives the correct frame on a tree — ONE code for any topology. The
      hard-coded multi-scale hex grid (l6_grid.py) is kept as an INNATE METRIC PRIOR (vector-navigation to
      unvisited goals, CRT error-correction) to switch on when a task needs it — it is not the substrate.
  L5  per-relation operators (displacement cells) — learned, one per relation; composed for inference.
  L4  content (the entity codebook) — binds entity ⊗ location, reads it back out.
  L23 the ONE shared object memory S — pooled across all domains, each in its own orthogonal slot.

Both ways in use the SAME SR frame (one code source, no parallel systems):
  learn_domain        : structure GIVEN (relations) → SR frame → orthogonal slot → bind content (L4→L23) + operators (L5).
  observe/consolidate : structure DISCOVERED from raw transitions → the same SR frame + operators (online).
  recall  : read the entity at a location (L4 readout of L23).
  infer   : compose relation operators (L5) then read out — relations never stored (a+b, grandparent, …).
  revise  : delta-rule overwrite when the world changes (L23) — the microwave.
  anchor  : Bayes-filter location correction from a sensed feature (drift / loop closure).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .l4_feature_location import L4_FeatureLocation
from .l5_displacement import L5_Displacement
from .l6_grid import L6_GridLocation
from .l6_sr import OnlineSR                            # the online TD successor representation (eigendecomposition-free L6)
from .l23_object import L23_Object
from .recognize import Recognizer                      # the "what + pose" evidence faculty (object identity + orientation)
from .residual import _find_predicate                  # the predicate search the column's dynamics faculty reuses


def _orthonormal(rows, cols, gen):
    return torch.linalg.qr(torch.randn(rows, cols, generator=gen))[0]      # (rows, cols), orthonormal cols


def _sr_frame(W):
    """The successor-representation eigenvector frame of an undirected adjacency W (n×n) — near-orthonormal
    node codes that hold the graph's structure for ANY topology (Stachenfeld 2017: grid cells ARE the SR
    eigenvectors; hexagonal is just the eigenbasis of open 2-D, so these come out grid-like on a metric graph
    and give the correct frame on a tree). Returns Z (n, n-1): all non-trivial modes, rows near-orthonormal."""
    deg = W.sum(1).clamp_min(1e-6).sqrt()
    M = W / (deg[:, None] * deg[None, :])                                  # symmetric-normalized transition graph
    _, evecs = torch.linalg.eigh(M)                                       # ascending eigenvalues
    return evecs[:, :-1]                                                  # drop the trivial constant mode


def _sparsify_topk(M, k):
    """Keep the signed top-k entries per row, zero the rest, renormalize — sparse high-capacity codes. A random
    projection followed by this winner-take-all is locality-sensitive hashing (Dasgupta-Stevens-Navlakha 2017,
    the fly mushroom body): similarity-preserving AND near-orthogonal in C(d, k) >> d numbers."""
    idx = M.abs().topk(min(k, M.shape[1]), dim=1).indices
    out = torch.zeros_like(M)
    out.scatter_(1, idx, M.gather(1, idx))
    return torch.nn.functional.normalize(out, dim=1)


class CorticalColumn(nn.Module):
    def __init__(self, n_entities, feat_dim=256, d_mem=512, torus_size=12, scales=(11, 13, 17),
                 place_k=1, seed=0):
        super().__init__()
        self.gen = torch.Generator().manual_seed(seed)
        self.L6 = L6_GridLocation(torus_size=torus_size, scales=scales, place_k=place_k)   # innate metric PRIOR (off; SR frame is the substrate)
        self.L5 = L5_Displacement()                                       # per-relation operators
        self.L4 = L4_FeatureLocation(n_entities, feat_dim=feat_dim, seed=seed)  # content codebook
        self.L23 = L23_Object(feat_dim=feat_dim, d_mem=d_mem)             # the one shared memory
        self.recognizer = Recognizer()                                   # "what + pose" (object id + orientation), evidence-based
        self.d_mem = d_mem
        self._shared_U = _orthonormal(d_mem, d_mem, self.gen)            # shared slot for the no-remap control (sliced to frame size)
        self.dom = {}                                                    # name -> {place, labels, U}
        # online structure learning (discover a structure from raw transitions) -------------------------------
        self.sr = OnlineSR(gamma=0.95, alpha=0.3)             # the location frame, learned ONLINE by TD (no batch eigh)
        self.graph, self.loc, self.rel = {}, {}, {}           # observed edges; symbol→frame index; relation operators
        self.place = None                                     # (n, d_mem) per-node SR-frame codes (set by consolidate)
        self._inv_loc = {}                                    # frame index → symbol (set by consolidate)
        self._sparse_place = False                            # True once n > d_mem → sparse place codes + cleanup
        self._place_k_loc = 16                                # active units per sparse place code (~3% of d_mem)
        self._cur = None                                      # the current node -- discrete path integration over the graph
        self._dyn_obs, self.dyn_rules = [], []                # the conditional-dynamics faculty (forward model of world responses)

    # ----- learn a model of a domain (structure GIVEN) via the SR frame ----------------------------
    def learn_domain(self, name, entity_labels, relations, remap=True):
        """entity_labels[i] = global entity index at node i. relations = {name: [(src,dst), ...]}. Placed via
        the SAME SR-eigenvector frame as the online path (one code source) — built here from the GIVEN
        relations rather than discovered from observation — then given an orthogonal slot for no-interference."""
        n = len(entity_labels)
        W = torch.zeros(n, n)
        for edges in relations.values():                                 # undirected adjacency over the given edges
            for s, t in edges:
                W[s, t] += 1.0
                W[t, s] += 1.0
        Z = _sr_frame(W)
        U = (_orthonormal(self.d_mem, Z.shape[1], self.gen) if remap else self._shared_U[:, :Z.shape[1]])
        place = torch.nn.functional.normalize(Z @ U.t(), dim=1)          # SR frame → orthogonal slot
        self.dom[name] = {"place": place, "labels": list(entity_labels), "U": U}
        for r, edges in relations.items():                               # L5: one operator per relation
            self.L5.learn((name, r), place, edges)
        for node, lab in enumerate(entity_labels):                       # L4 bind → L23 pool
            self.L23.pool(self.L4.bind(lab, place[node]))

    # ----- predict ----------------------------------------------------------------------------------
    def recall(self, name, node):
        return self.L4.readout(self.L23.S, self.dom[name]["place"][node]).argmax().item()

    def infer(self, name, relation_chain, node, topk=1):
        """Answer a relation never stored by COMPOSING operators (L5), then reading the landing location."""
        v = self.dom[name]["place"][node]
        for r in reversed(relation_chain):
            v = self.L5.apply((name, r), v)
        idx = self.L4.readout(self.L23.S, v).topk(topk).indices.tolist()
        return idx[0] if topk == 1 else idx

    # ----- update when the world changes (the microwave) --------------------------------------------
    def revise(self, name, node, new_label):
        self.L23.revise(self.dom[name]["place"][node], self.L4.E[new_label])
        self.dom[name]["labels"][node] = new_label

    # ----- correct a drifting location estimate from what is sensed (gap D / loop closure) ----------
    @torch.no_grad()
    def anchor(self, name, node_estimate, observed_label, window=3, prior_decay=0.5):
        """Bayes filter: prior = grid neighbourhood of the believed node; likelihood = which nearby node's
        stored content matches the observation; posterior peak = corrected node."""
        place, n = self.dom[name]["place"], len(self.dom[name]["labels"])
        cand = list(range(max(0, node_estimate - window), min(n, node_estimate + window + 1)))
        like = torch.stack([self.L4.readout(self.L23.S, place[c]).softmax(-1)[observed_label] for c in cand])
        prior = torch.tensor([torch.exp(torch.tensor(-prior_decay * abs(c - node_estimate))) for c in cand])
        return cand[int((like * prior).argmax())]

    # ----- discover a structure from raw transitions (online; the counterpart to learn_domain) ----------
    def observe(self, s, a, s2):
        """Record one observed transition. It feeds the ONLINE SR (the location frame, TD-learned every step — no batch
        eigh; `refresh()` reads it) AND the per-action edge graph (for the L5 operators). The geometry — line, ring,
        2-D grid, tree — falls out of the SR; no metric-vs-non-metric switch."""
        self.sr.observe(s, s2)                                       # online location learning (incl. self-loops / blocked moves)
        if s2 != s:                                                  # the per-action operator has no edge for a blocked move
            self.graph.setdefault(s, {})[a] = s2

    def consolidate(self):
        """Discover the structure as the SR-eigenvector frame of the OBSERVED transition graph — the online
        counterpart of learn_domain, same code source, structure discovered not given. One universal
        mechanism: line, ring, 2-D grid and tree are all handled with no metric-vs-non-metric switch;
        per-relation operators (L5) then read off any relation."""
        syms = sorted(set(self.graph) | {s2 for e in self.graph.values() for s2 in e.values()})
        idx = {s: i for i, s in enumerate(syms)}
        n = len(syms)
        W = torch.zeros(n, n)
        relations = {}
        for s, e in self.graph.items():
            for a, s2 in e.items():
                W[idx[s], idx[s2]] += 1.0
                W[idx[s2], idx[s]] += 1.0                            # symmetric (undirected) adjacency
                relations.setdefault(a, []).append((idx[s], idx[s2]))
        Z = _sr_frame(W)
        nz = Z.shape[1]
        if nz <= self.d_mem:                                       # exact near-orthonormal (dense) place codes, up to d_mem
            P = _orthonormal(self.d_mem, nz, self.gen)
            self.place = torch.nn.functional.normalize(Z @ P.t(), dim=1)
            self._sparse_place = False
        else:                                                      # past d_mem: SPARSE place codes (LSH) — capacity C(d_mem,k) >> d_mem
            Wp = torch.randn(self.d_mem, nz, generator=self.gen) / (nz ** 0.5)
            self.place = _sparsify_topk(Z @ Wp.t(), self._place_k_loc)
            self._sparse_place = True
        self.loc = idx
        self._inv_loc = {i: s for s, i in idx.items()}
        for s, i in idx.items():
            self.L23.pool(self.L4.bind(s, self.place[i]))
        for a, edges in relations.items():
            self.L5.learn(("rel", a), self.place, edges)
            self.rel[a] = ("rel", a)

    def refresh(self):
        """ONLINE, eigendecomposition-free consolidation — the per-step-affordable replacement for `consolidate()`.
        The location frame is the ONLINE SR (TD-learned by `observe`, no `eigh`); take its rows as place codes (padded
        into d_mem so L4/L5/L23 are unchanged), then learn the L5 operators + pool L4/L23 from them with the SAME cheap
        outer products `consolidate` uses. The batch `eigh` `consolidate` is kept only as the offline reference. (Up to
        d_mem states are padded in; beyond d_mem a random projection is needed — deferred.)"""
        syms = sorted(self.sr.idx, key=lambda s: self.sr.idx[s])
        if not syms:
            return
        self.loc = dict(self.sr.idx)
        self._inv_loc = {i: s for s, i in self.loc.items()}
        codes = torch.zeros(len(syms), self.d_mem)
        for s in syms:                                               # SR row -> place code (padded into d_mem)
            row = torch.as_tensor(self.sr.code(s), dtype=torch.float32)
            codes[self.loc[s], :row.shape[0]] = row[:self.d_mem]
        self.place = torch.nn.functional.normalize(codes, dim=1)
        self._sparse_place = False
        relations: dict = {}
        for s, e in self.graph.items():
            for a, s2 in e.items():
                if s in self.loc and s2 in self.loc:
                    relations.setdefault(a, []).append((self.loc[s], self.loc[s2]))
        self.rel = {}
        for a, edges in relations.items():                          # operators from the current codes (outer products; no eigh)
            self.L5.learn(("rel", a), self.place, edges)
            self.rel[a] = ("rel", a)
        for s, i in self.loc.items():                              # content bound at the online place codes
            self.L23.pool(self.L4.bind(s, self.place[i]))

    def _cleanup(self, v):
        """Snap a noisy location vector to the nearest place code (attractor cleanup) — keeps sparse-code
        operator composition from accumulating error. Used only on the sparse path (n > d_mem)."""
        return self.place[(self.place @ v).argmax()]

    def predict(self, symbol, action):
        """Where `action` leads from `symbol`: the learned next state, read straight from the transition GRAPH. The
        graph is the state-dependent operator (each (s, a) has its own next state -- it subsumes the L5 matrix operator
        AND residual's conditional structure, exactly and online); the online SR carries value/topology, recognition
        carries continuous pose. An unobserved / blocked (s, a) stays put. (See the brain-reference-frames research:
        the brain path-integrates by a continuous-attractor bump / discrete snapping, not a matrix op over codes.)"""
        return self.graph.get(symbol, {}).get(action, symbol)

    def add(self, start, count, succ_action=0):
        """Arithmetic = navigation: apply the successor operator `count` times from `start`, read out."""
        v = self.place[self.loc[start]]
        for _ in range(count):
            v = self.L5.apply(self.rel[succ_action], v)
            if self._sparse_place:
                v = self._cleanup(v)
        return self.L4.readout(self.L23.S, v).argmax().item()

    # ----- the "what + pose" recognition faculty (object identity + orientation; Monty evidence-based) --------
    # Complementary to L6 (the "where"): L6 recognises navigable locations in a fixed frame; this SOLVES an object's
    # pose so a known object is recognised at an orientation never seen (the 2019 grid-cell model's stated gap).
    def learn_object(self, cloud, name=None):
        """Add an object to this column's recogniser. `name` given → store under it; else learn ONLINE + label-free
        (recognise-or-add). Returns the ObjectModel (named) or (name, is_new) (online)."""
        return self.recognizer.add(name, cloud) if name is not None else self.recognizer.add_if_novel(cloud)

    def recognize_object(self, cloud):
        """Identify a sensed point cloud's (name, theta, t, evidence) at a pose never seen, learning it online if
        novel — pose-invariant recognition (object permanence under rotation)."""
        return self.recognizer.recognize(cloud)

    def identify_object(self, cloud):
        """Recognise a sensed shape against the learned library WITHOUT adding a new one — the name, or None."""
        return self.recognizer.identify(cloud)

    # ----- codes exposed for cross-column composition (the thalamus binds content ⊗ location) --------
    def content_code(self, label):
        """This column's content (What / L4) code for an entity label."""
        return self.L4.E[label]

    def place_code(self, node):
        """This column's location (Where / SR-frame) code for a node — discovered by observe/consolidate."""
        return self.place[self.loc[node]]

    # ----- path integration as DISCRETE graph tracking (predict by the learned edge, snap to a sighting) ----------
    # NOT a matrix operator over codes: the brain path-integrates by a continuous-attractor bump shifted by velocity,
    # with discrete-attractor SNAPPING on a clear sighting (reference_brain_reference_frames_orthogonalization). Here
    # that is exact and online -- PREDICT the next node by the learned edge (efference copy; needs no observation, so it
    # survives PARTIAL observability), CORRECT by snapping to a sensed node. The online SR carries value/topology;
    # recognition carries continuous pose.
    def loc_reset(self, node):
        """Begin dead reckoning at a known node."""
        self._cur = node
        return node

    def loc_move(self, action):
        """PREDICT: path-integrate by the learned edge (the efference copy) -- no observation needed. An unobserved or
        blocked move stays put."""
        self._cur = self.graph.get(self._cur, {}).get(action, self._cur)
        return self._cur

    def loc_sense(self, node):
        """CORRECT: snap the belief to an observed node (the discrete-attractor sighting). Call only when a node is
        actually sensed; otherwise keep dead-reckoning via loc_move."""
        self._cur = node
        return self._cur

    def loc_where(self):
        """The current node."""
        return self._cur

    # ----- the conditional-DYNAMICS faculty: the column's forward model of the WORLD's responses ------------
    # The location code above predicts where the BODY goes; this predicts what the WORLD does that self-motion
    # cannot explain (the exafferent residual) — a door opens BECAUSE of a precondition. The SAME predicate
    # search that found carry (residual.py): under what PRECONDITION (a feature of the sensed state) does each
    # EFFECT occur. Folded INTO the column (it used to be a separate DynamicsModel) so the column that LEARNS the
    # dynamics is the column that PREDICTS them — no external module the planner reads. The effect is a discrete
    # world-change rather than a location delta, but the mechanism is identical, so there is no hand-coded rule.
    def observe_effect(self, features, effect):
        """One step of experience: the sensed-state `features` and the EXAFFERENT `effect` (hashable, or None)."""
        self._dyn_obs.append((tuple(features), effect))

    def learn_dynamics(self):
        """For each distinct effect, find the SIMPLEST precondition (a predicate over the features) that selects
        exactly the states where it occurred — the residual predicate search. An effect with no compressing
        precondition is REFUSED (the MDL stop). Returns the rules [(predicate, description, effect)]."""
        self.dyn_rules = []
        obs = list(set(self._dyn_obs))                        # the search needs only the DISTINCT (features→effect) rows
        for eff in sorted({e for _, e in obs if e is not None}, key=repr):
            need = [f for f, e in obs if e == eff]
            correct = [f for f, e in obs if e != eff]         # states where this effect did NOT occur
            pred, desc = _find_predicate(need, correct)
            if pred is not None:
                self.dyn_rules.append((pred, desc, eff))
        return self.dyn_rules

    def predict_effect(self, features):
        """The effect the current sensed state triggers (the first matching rule), or None — what the control
        loop rolls forward to plan ('reach the precondition and the door opens')."""
        f = tuple(features)
        for pred, _desc, eff in self.dyn_rules:
            if pred(f):
                return eff
        return None
