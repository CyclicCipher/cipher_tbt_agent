"""The OPERATOR — a composable action (L6_NONABELIAN Stage 0).

The GENERAL form of the thing L5 applies to the L6 location code. Where `l5_displacement.move_delta` is an additive
DISPLACEMENT — a vector you ADD, which HARD-WIRES commutativity (`a + b = b + a`) — an `Operator` is a linear MAP you
COMPOSE by matrix product, and matrix products do NOT commute in general (`A∘B ≠ B∘A`). The abelian TRANSLATION is the
special case (a homogeneous translation matrix, or the grid's block-diagonal phase rotation); a ROTATION is a
non-commuting witness. This OPENS THE DOOR to non-abelian structure (rotations / orderings / constrained dynamics) that a
commuting-phase code cannot hold — WITHOUT changing abelian behaviour (Stage 0 is the "negative first step").

A faithful GROUP REPRESENTATION means COMPOSITION FIDELITY: `op(a)∘op(b) == op(a∘b)`, i.e. `M(a)·M(b) = M(a∘b)`.
Translations satisfy this (and commute); the interface does NOT assume they commute. Stage 1 makes the per-action operator
LEARNED — a matrix constrained to a proper representation (Gao/TEM) — so a non-commuting action becomes representable;
the additive `move` is then the abelian special case, not the substrate. Pure numpy (small matrices; matches `l6_sr`)."""

from __future__ import annotations

import numpy as np


class Operator:
    """A linear map on a state code, COMPOSED by matrix product. `apply(z) = M·z`; `a.then(b)` = do `a` THEN `b`."""

    def __init__(self, M):
        self.M = np.asarray(M, dtype=float)

    @property
    def dim(self) -> int:
        return self.M.shape[0]

    def apply(self, z):
        """Act on a state code `z` (a `dim`-vector, or `dim×n` batch): `M·z`."""
        return self.M @ np.asarray(z, dtype=float)

    def then(self, other: "Operator") -> "Operator":
        """Compose: apply SELF first, then OTHER. The composed matrix is `other.M @ self.M` — NON-commutative in general."""
        return Operator(other.M @ self.M)

    def inverse(self) -> "Operator":
        return Operator(np.linalg.inv(self.M))

    def commutes_with(self, other: "Operator", tol: float = 1e-9) -> bool:
        """Does composition order matter? `M·N == N·M`? Abelian ⇒ True for all pairs; non-abelian ⇒ False for some."""
        return np.allclose(self.M @ other.M, other.M @ self.M, atol=tol)

    # ----- the CONTINUOUS (Lie-group) form: a learned DISCRETE step -> any group element along its 1-parameter subgroup ---
    def power(self, t: float) -> "Operator":
        """`M^t = exp(t·log M)` -- the operator at CONTINUOUS parameter `t` along its 1-parameter subgroup (`t=1`→self,
        `t=0`→identity, `t=0.5`→the "half" step). For an orthogonal M (a rotation) this is a valid rotation by `t`× the
        angle: the LIE-GROUP form that turns a LEARNED DISCRETE-STEP operator into the CONTINUOUS group -- learn the step,
        read off ANY pose (what L2/3's HAND-CODED `rot(θ)` does, but LEARNED); also = fractional path integration. Via
        eigendecomposition (principal branch -> well-defined for |rotation angle| < π; a ½-turn step is the branch limit)."""
        vals, vecs = np.linalg.eig(self.M)
        return Operator((vecs @ np.diag(vals ** t) @ np.linalg.inv(vecs)).real)

    def generator(self):
        """The Lie-algebra GENERATOR `G` with `M = exp(G)` (i.e. `log M`); `power(t) = exp(t·G)`. SKEW-SYMMETRIC for an
        orthogonal M (the `so(n)` algebra) -- the INFINITESIMAL form of the learned continuous group (its tangent at I)."""
        vals, vecs = np.linalg.eig(self.M)
        return (vecs @ np.diag(np.log(vals)) @ np.linalg.inv(vecs)).real

    def __eq__(self, other) -> bool:
        return isinstance(other, Operator) and self.M.shape == other.M.shape and np.allclose(self.M, other.M)

    def __repr__(self) -> str:
        return f"Operator(dim={self.dim})"

    # ----- abelian instances (Stage 0): translation is the special case that reproduces `pos += delta` ---------
    @staticmethod
    def identity(dim: int) -> "Operator":
        return Operator(np.eye(dim))

    @staticmethod
    def translation(delta) -> "Operator":
        """The abelian TRANSLATION operator in HOMOGENEOUS coords: on `[x…, 1]` it ADDS `delta`. A d-vector → a
        (d+1)×(d+1) matrix. `translation(a).then(translation(b)) == translation(a+b)` (composes additively, commutes) —
        so `move_delta` IS this operator, viewed additively."""
        delta = np.asarray(delta, dtype=float)
        d = delta.shape[0]
        M = np.eye(d + 1)
        M[:d, d] = delta
        return Operator(M)

    @staticmethod
    def rotation(theta: float) -> "Operator":
        """A 2-D ROTATION about the origin, in homogeneous coords (3×3) so it composes with `translation`. The
        NON-COMMUTING witness: `rotation ∘ translation ≠ translation ∘ rotation` — the interface is not baked-in abelian."""
        c, s = float(np.cos(theta)), float(np.sin(theta))
        return Operator([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # ----- LEARN an operator from transitions (Stage 1) — the constraint is what buys composition fidelity ----
    @staticmethod
    def fit(before, after, orthogonal: bool = True) -> "Operator":
        """LEARN the operator mapping each `before` state code to its `after` (rows = samples, cols = dim) — L6_NONABELIAN
        Stage 1. `orthogonal=True` imposes the GAO REPRESENTATION CONSTRAINT (orthogonal Procrustes: `M = U Vᵀ` from
        `afterᵀ·before = U S Vᵀ`) → the learned operator is a proper rotation (spectral radius 1), so its COMPOSITION /
        POWERS stay faithful and EXTRAPOLATE. `False` = unconstrained least-squares (`M = after⁺·before`): fits ONE step
        but its powers DRIFT (spectral radius ≠ 1). The abelian composition-fidelity gate (`test_operator.py`) shows the
        constraint is REQUIRED — validated empirically before trusting non-abelian. (Batch; an ONLINE constrained update is
        a later Stage-1 refinement.)"""
        before = np.atleast_2d(np.asarray(before, dtype=float))
        after = np.atleast_2d(np.asarray(after, dtype=float))
        if orthogonal:
            U, _, Vt = np.linalg.svd(after.T @ before)
            return Operator(U @ Vt)
        return Operator(after.T @ np.linalg.pinv(before.T))

    def spectral_radius(self) -> float:
        """max |eigenvalue| — 1.0 for a faithful (unitary) representation; ≠ 1 means COMPOSITIONS/POWERS drift."""
        return float(np.max(np.abs(np.linalg.eigvals(self.M))))


class OnlineOperator:
    """LEARN an operator ONLINE from a STREAM of (before, after) transitions — the AGENT form of `Operator.fit`
    (L6_NONABELIAN Stage 1). Maintains a running cross-covariance `C ≈ Σ after ⊗ before` (a cheap rank-1 update per
    transition); the operator is read as its orthogonal PROCRUSTES `M = U Vᵀ` (SVD only on read — throttle it).

    KEY PROPERTY: this DECOUPLES accumulation (unconstrained → EXPRESSIVE) from the representation CONSTRAINT (a PROJECTION
    at read), so the constraint⊥expressivity tension does NOT bite — the constraint never fights the fit, and the operator
    stays a proper rotation (spectral radius 1) throughout. The real online challenge is COVERAGE, not the constraint: the
    operator is well-estimated only over the region the stream samples, so it needs BROAD/relatively-uniform exploration
    (a running SUM, `decay=1.0`, for a stationary operator; a gentle `decay<1` for a drifting one). A LOCAL random walk's
    peaked occupancy under-covers → poor extrapolation. (Empirically validated in `test_operator.py`.)"""

    def __init__(self, dim: int, decay: float = 1.0):
        self.dim = dim
        self.decay = decay                                            # 1.0 = a pure running SUM (stationary op); <1 = EWMA (drift)
        self.C = np.zeros((dim, dim))
        self._M = np.eye(dim)
        self._dirty = False

    def observe(self, before, after) -> None:
        """Accumulate one transition into the cross-covariance (cheap; the SVD is deferred to `operator()`)."""
        outer = np.outer(np.asarray(after, dtype=float), np.asarray(before, dtype=float))
        self.C = self.C + outer if self.decay >= 1.0 else self.decay * self.C + (1.0 - self.decay) * outer
        self._dirty = True

    def operator(self) -> Operator:
        """Read the current operator = orthogonal Procrustes of the accumulated cross-covariance (SVD; cached until the
        next `observe`). The agent throttles this (read every N steps), like the eigenpurpose SVD."""
        if self._dirty:
            U, _, Vt = np.linalg.svd(self.C)
            self._M = U @ Vt
            self._dirty = False
        return Operator(self._M)


def discover_group(generators, tol: float = 1e-6, max_elements: int = 256):
    """L6_NONABELIAN Stage 2 — DISCOVER a group's RELATIONS by LOOP CLOSURE (the QUOTIENT of the free monoid on the learned
    `generators`). BFS the tree of operator-words; a new word CLOSES (denotes an element already found) when its operator is
    EQUAL to a known one — the PREDICTIVE-SUFFICIENCY criterion made exact: equal operators act identically on EVERY code, so
    they have the same future (bisimulation / causal-state equivalence, `MATH_PHASE.md`). This collapses the infinite free
    TREE (the free-monoid words) into the FINITE Cayley graph — the spanning/free part READS OFF, the closures are the
    SEARCHED relations (the master boundary: free/abelian = read-off, quotient = search). Returns `(elements, relations)`:
      elements  = list of (word, Operator); word = a tuple of generator indices, the SHORTEST (BFS) representative of the element;
      relations = list of (word, equal_word); each discovered CLOSURE (a non-tree edge). `equal_word=()` ⇒ a loop to identity (r=e).
    ABELIAN generators produce COMMUTATIVITY closures ((i,j)==(j,i)); non-abelian ones do not (that gap IS the boundary).
    `max_elements` caps an infinite/large group (e.g. free translations) — the early relations (incl. commutativity) still
    surface before the cap. The generators come from S1 (learned `pose_ops` / `action_ops`); this is the mechanism a
    geodesic planner (S3) searches, and the group-theoretic form of the number domain's factored loop closure."""
    gens = [np.asarray(g.M, dtype=float) for g in generators]
    dim = gens[0].shape[0]
    words = [()]                                           # BFS-ordered canonical words (the SHORTEST rerep of each element)
    mats = [np.eye(dim)]                                   # the element operators (parallel to `words`)
    relations = []

    def _find(m):
        for i, em in enumerate(mats):
            if np.allclose(em, m, atol=tol):              # predictive sufficiency: identical action ⇒ same element
                return words[i]
        return None

    head = 0
    while head < len(words):
        w, wm = words[head], mats[head]
        head += 1
        for gi, gm in enumerate(gens):
            nm = gm @ wm                                   # apply `w`, then generator `gi`
            hit = _find(nm)
            if hit is not None:
                relations.append((w + (gi,), hit))        # a CLOSURE = a relation (this word ≡ the found element)
            elif len(words) < max_elements:
                words.append(w + (gi,))
                mats.append(nm)
    return [(w, Operator(m)) for w, m in zip(words, mats)], relations


def discover_periods(op, tol: float = 1e-6, max_period: int = 128):
    """L6_NONABELIAN Stage 2 (the FACTORED case) — DISCOVER the cyclic factor PERIODS latent in a learned operator from its
    SPECTRUM: the ORDERS of its root-of-unity eigenvalues (= the operator's PERIODIC invariant subspaces / irreducible cyclic
    reps). The factorisation thus comes from the DYNAMICS (the learned operator), NOT from the input labelling — the
    PRINCIPLED answer to the similarity-kernel SMUGGLE: you cannot loop-close on a projection you did not earn from the
    operator itself. A BROADBAND / non-recurring spectrum (eigenvalues off the unit circle) yields NO period — the NEGATIVE
    CONTROL that a raw, non-recurring count stays a LINE. And the spectrum distinguishes a single n-CYCLE (a PRIMITIVE nth
    root present) from a genuine PRODUCT of smaller cycles (only the small roots) — so the notational factoring of one big
    cycle (a raw Z/1000 count → 10×100) is NOT hallucinated; only genuine product structure in the dynamics is found.
    Returns the sorted distinct periods (>1). Feeds FACTORED loop closure (close a loop every `period` steps in that factor's
    eigenspace, guarded by predictive sufficiency) — the group-theoretic form of place value, honest about what is GIVEN."""
    vals = np.linalg.eigvals(np.asarray(op.M, dtype=float))
    periods = set()
    for lam in vals:
        if abs(abs(lam) - 1.0) > tol:                     # off the unit circle → decaying/nilpotent, not a cycle
            continue
        frac = float(np.angle(lam)) / (2.0 * np.pi)       # λ = exp(2πi·frac); its ORDER = smallest q with q·frac an integer
        if abs(frac - round(frac)) < tol:                 # frac ≈ 0 → λ = 1, the trivial fixed direction (period 1)
            continue
        for q in range(2, max_period + 1):
            if abs(frac * q - round(frac * q)) < tol:
                periods.add(q)
                break
    return sorted(periods)


def factor_group(generators, tol: float = 1e-6, max_elements: int = 256):
    """L6_NONABELIAN Stage 2 (FACTORED closure) — decompose the group the learned `generators` generate into a DIRECT PRODUCT
    of cyclic FACTORS (the product-of-cycles map behind place value / a game's independent counter+toggle). Each generator's
    cyclic subgroup has order = its period; the factoring is PREDICTIVELY SUFFICIENT (loses no information → reproduces the
    dynamics) iff the generators COMMUTE and the product of their orders equals the group order |G| — i.e. every element is a
    UNIQUE product of factor-powers. Non-commuting generators (no direct product) or overlapping factors (product ≠ |G|)
    FAIL — the wrong-merge the guard rejects (a factoring that mispredicts is not accepted). Basis-INDEPENDENT: the factors
    live in the COMMUTING operators' joint eigenstructure, not the code's axes, so the factorisation cannot be smuggled from a
    pre-separated code. Returns the list of (generator_index, order) factors, or None if no valid direct-product factoring."""
    dim = generators[0].dim
    ident = np.eye(dim)
    orders = []
    for g in generators:
        m, k = np.asarray(g.M, dtype=float), 1
        while not np.allclose(m, ident, atol=tol):
            m = g.M @ m
            k += 1
            if k > max_elements:
                return None                                 # non-torsion generator (no finite cyclic factor)
        orders.append(k)
    for i in range(len(generators)):                        # a DIRECT product needs COMMUTING factors
        for j in range(i + 1, len(generators)):
            if not generators[i].commutes_with(generators[j], tol):
                return None
    elements, _ = discover_group(generators, tol=tol, max_elements=max_elements)
    prod = 1
    for o in orders:
        prod *= o
    if prod != len(elements):                               # factors overlap / don't span → the product loses/duplicates info
        return None                                         # (predictive sufficiency fails — reject the wrong factoring)
    return [(i, orders[i]) for i in range(len(generators))]


def is_predictively_sufficient(transitions, project) -> bool:
    """L6_NONABELIAN Stage 2 — the PREDICTIVE-SUFFICIENCY (bisimulation / lumpability) test that GUARDS a factored closure.
    A projection (a candidate factor) is SUFFICIENT iff states sharing a projected value ALWAYS transition to states sharing
    a projected value — i.e. the projection is a CONGRUENCE for the dynamics, so closing/merging within it loses no predictive
    information. `transitions` = `(state, next_state)` pairs under ONE action; `project` = `state -> factor value` (hashable).
    An INDEPENDENT factor passes; a COUPLED factor FAILS — e.g. base-b carry: the tens digit's next value depends on whether
    the UNITS wrapped, so projecting to the tens alone is NOT a congruence. That failure IS the carry, detected — and the
    wrong-merge the guard rejects. The causal-state / bisimulation criterion (Crutchfield) made an explicit checker."""
    seen = {}
    for s, nxt in transitions:
        k, pn = project(s), project(nxt)
        if k in seen and seen[k] != pn:
            return False
        seen[k] = pn
    return True


def homog(pos):
    """Lift a position to homogeneous coords `[x…, 1]` — the state that `translation`/`rotation` operators act on."""
    return np.concatenate([np.asarray(pos, dtype=float), [1.0]])


def dehomog(z):
    """Project a homogeneous state `[x…, w]` back to a position `[x…]/w`."""
    z = np.asarray(z, dtype=float)
    return z[:-1] / z[-1]
