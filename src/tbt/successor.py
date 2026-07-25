"""successor.py — a LEARNED frame: the online successor representation over an arbitrary transition graph.

`GridEncoder` is a frame you are GIVEN. It works for physical space because physical space is metric — you can hand it
coordinates and path-integrate motion on them. A task space has no coordinates to hand it: "keys collected, doors open" is
not a point in R^n, and inventing an embedding for it would be hand-specifying the very structure the frame exists to learn.

This is the frame you LEARN. Under the predictive-map account (Stachenfeld 2017, [[reference_grid_sr_eigenbasis]]) place
cells ARE the successor representation and grid cells are its eigenvectors, so hexagonal coding is not a primitive — it is
what the SR of generic open 2-D space happens to look like. Run the same machinery on a different transition structure and
you get that structure's code instead. That is why one mechanism serves both: TEM's path integration over an ARBITRARY
graph is this, and it is what lets a region above the sensorimotor one have an L6a at all.

WHAT THIS DELIBERATELY DOES NOT DO — all three from the record, and each was a real cost or a measured failure:
  * NO EIGENDECOMPOSITION. Grid cells being SR eigenvectors is a statement about what the code LOOKS like, not a
    prescription to compute it: an O(n^3) decomposition is prohibitive online, and the eigenpurpose drive built on top of
    it was reassessed as redundant AND costly and dropped. The SR ITSELF carries what is needed — value as `V = M.R`, and
    topological distance for reaching — so the eigenbasis is a detour ([[reference_eigenoptions_subgoals]]).
  * NO MATRIX OPERATOR OVER SR ROWS. That hybrid was built once and failed measurably, path-integrating [2,2,2,4,4,4]
    where [1,2,3,4,5,6] was required. The brain does not do matrix operators over correlated codes; it does
    continuous-attractor bump-shift over overlapping codes and sparse-orthogonal recall
    ([[reference_brain_reference_frames_orthogonalization]]).
  * NO ORTHOGONALISATION HERE. SR codes are meant to be CORRELATED — that overlap IS the topology, and it is what makes
    nearby states generalise. Sparse pattern separation is the hippocampus's job (DG), at a different stage.

It also does not replace the grid. Physical space keeps its metric prior, whose genuinely metric powers (vector navigation
to never-visited goals, error-correcting capacity) an SR does not reproduce; this serves the spaces that have no metric to
assume. Pure stdlib, sparse dicts, states are any hashable.
"""

from __future__ import annotations

from collections import defaultdict


class SuccessorFrame:
    """An online SR over a discrete transition graph. `M[s][s2]` is the discounted expected future occupancy of `s2` given
    you are at `s` — so a row of `M` IS the location code for `s`, expressed as "what tends to follow from here".

    Learned by the TD rule that every other learner here uses (`reference_grid_sr_eigenbasis`; the same delta rule as the
    value critic and Rescorla-Wagner):

        M[s]  <-  M[s] + alpha * ( 1_s + gamma * M[s'] - M[s] )

    The graph itself is kept exactly (`graph[s][a] = s'`), because a discrete transition table is state-dependent by
    construction and subsumes what a conditional operator was doing. Both are online and incremental: no batch, no
    decomposition, nothing that has to see the whole space before it is useful."""

    def __init__(self, alpha: float = 0.3, gamma: float = 0.9) -> None:
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.M: dict = defaultdict(lambda: defaultdict(float))   # state -> successor occupancy vector (CORRELATED, sparse)
        self.graph: dict = defaultdict(dict)                     # state -> action -> next state (the exact structure)

    def observe(self, state, action, nxt) -> None:
        """One transition. Records it in the graph and folds it into the SR by the TD rule."""
        self.graph[state][action] = nxt
        # BEING in a state means OCCUPYING it, so the destination's self-occupancy is pulled toward 1 whether or not it is
        # ever acted FROM. Without this a state that is only ever arrived at — a terminal, or a goal — keeps an empty row,
        # so no reward placed there can propagate back and `V = M.R` reads zero everywhere. Measured, before this line.
        self.M[nxt][nxt] += self.alpha * (1.0 - self.M[nxt].get(nxt, 0.0))
        row, nxt_row = self.M[state], self.M[nxt]
        keys = set(row) | set(nxt_row) | {state}
        for k in keys:                                           # target = 1_s + gamma * M[s'], the discounted occupancy
            target = (1.0 if k == state else 0.0) + self.gamma * nxt_row.get(k, 0.0)
            row[k] += self.alpha * (target - row.get(k, 0.0))

    def code(self, state) -> dict:
        """The learned LOCATION CODE for a state: its SR row. Correlated with the codes of nearby states by construction —
        that overlap is the topology, and it is what a hand-given coordinate cannot supply for a non-metric space."""
        return dict(self.M.get(state, {}))

    def similarity(self, a, b) -> float:
        """Cosine overlap of two states' codes — how near they are IN THE TRANSITION STRUCTURE, which is the only sense of
        "near" a task space has. Traversability, not Euclidean distance: two states one action apart are close even if
        nothing about their contents looks alike."""
        ra, rb = self.M.get(a, {}), self.M.get(b, {})
        if not ra or not rb:
            return 0.0
        dot = sum(v * rb.get(k, 0.0) for k, v in ra.items())
        na = sum(v * v for v in ra.values()) ** 0.5
        nb = sum(v * v for v in rb.values()) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def value(self, state, rewards) -> float:
        """`V = M . R` — the SR's whole point as a planner: once occupancy is known, value for ANY reward function is a dot
        product, so a moved goal re-values the space without re-learning it (`reference_brain_planning`)."""
        row = self.M.get(state, {})
        return sum(w * rewards.get(k, 0.0) for k, w in row.items())

    def states(self) -> set:
        return set(self.M)
