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
from .l23_object import L23_Object


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
        self.d_mem = d_mem
        self._shared_U = _orthonormal(d_mem, d_mem, self.gen)            # shared slot for the no-remap control (sliced to frame size)
        self.dom = {}                                                    # name -> {place, labels, U}
        # online structure learning (discover a structure from raw transitions; driven by agent.py) -------
        self.graph, self.loc, self.rel = {}, {}, {}           # observed edges; symbol→frame index; relation operators
        self.place = None                                     # (n, d_mem) per-node SR-frame codes (set by consolidate)
        self._inv_loc = {}                                    # frame index → symbol (set by consolidate)
        self._sparse_place = False                            # True once n > d_mem → sparse place codes + cleanup
        self._place_k_loc = 16                                # active units per sparse place code (~3% of d_mem)
        self._h = None                                        # L6 as a DYNAMIC path-integrated belief (the recurrence)

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
        """Record one observed transition as a 1-shot edge. The geometry — line, ring, or 2-D grid — and
        where each symbol sits is discovered later by consolidate() from the whole graph."""
        if s2 != s:                                                   # ignore blocked moves (no edge there)
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

    def _cleanup(self, v):
        """Snap a noisy location vector to the nearest place code (attractor cleanup) — keeps sparse-code
        operator composition from accumulating error. Used only on the sparse path (n > d_mem)."""
        return self.place[(self.place @ v).argmax()]

    def predict(self, symbol, action):
        """Apply the learned action operator at a symbol's place, read out the landing symbol."""
        v = self.L5.apply(self.rel[action], self.place[self.loc[symbol]])
        if self._sparse_place:
            v = self._cleanup(v)
        return self.L4.readout(self.L23.S, v).argmax().item()

    def add(self, start, count, succ_action=0):
        """Arithmetic = navigation: apply the successor operator `count` times from `start`, read out."""
        v = self.place[self.loc[start]]
        for _ in range(count):
            v = self.L5.apply(self.rel[succ_action], v)
            if self._sparse_place:
                v = self._cleanup(v)
        return self.L4.readout(self.L23.S, v).argmax().item()

    # ----- codes exposed for cross-column composition (the thalamus binds content ⊗ location) --------
    def content_code(self, label):
        """This column's content (What / L4) code for an entity label."""
        return self.L4.E[label]

    def place_code(self, node):
        """This column's location (Where / SR-frame) code for a node — discovered by observe/consolidate."""
        return self.place[self.loc[node]]

    # ----- L6 driven as a DYNAMIC, path-integrated belief — the recurrence (architecture doc §14 stage 11) -----
    # The static place_code above is a lookup; here L6 becomes a persistent state h, the same selective gated
    # recurrence as the language SSM (precursor/language_recurrent.py) but on the LOCATION code: PREDICT by the
    # L5 displacement operator (grid-cell path integration — needs no observation, so it works when position is
    # NOT visible), CORRECT by the SSM decay gate toward a sensed node. This is what lets a column track where it
    # is and integrate an observation sequence under PARTIAL observability, where the static lookup cannot.
    def loc_reset(self, node):
        """Begin the recurrent belief at a known node (the origin of dead reckoning)."""
        self._h = self.place_code(node).clone()
        return node

    def loc_move(self, action):
        """PREDICT: path-integrate the belief by the action's displacement operator (the efference copy). No
        observation needed — this is the update that survives partial observability. Snapped to the nearest
        place code (position is discrete) to keep the belief crisp over a long trajectory."""
        self._h = self._cleanup(self.L5.apply(self.rel[action], self._h))
        return self.loc_where()

    def loc_sense(self, node, keep=0.0):
        """CORRECT: selectively blend the belief toward an observed node. `keep` ∈ [0,1] is the SSM decay gate —
        keep=0 snaps to a reliable sighting, keep→1 trusts the path integration (no / uncertain sighting)."""
        self._h = keep * self._h + (1.0 - keep) * self.place_code(node)
        return self.loc_where()

    def loc_where(self):
        """READ OUT the most likely current node — attractor match of the belief against the place codebook."""
        return self._inv_loc[int((self.place @ self._h).argmax())]
