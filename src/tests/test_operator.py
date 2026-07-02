"""L6_NONABELIAN Stage 0 — the OPERATOR interface: composition instead of vector-addition, WITHOUT changing abelian
behaviour, and PROVING the interface can hold a NON-COMMUTING operator (commutativity is no longer hard-wired).

Gates: (1) translation is a faithful ABELIAN group representation (composition fidelity + commutes); (2) the interface
HOLDS non-commuting operators (a rotation vs a translation); (3) `l6_grid.path_integrate` IS the operator acting (no
behaviour change); (4) L5's per-action operator reproduces the additive `move` on a NavGame path + a counting succession
(no regression)."""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.operator import Operator, OnlineOperator, dehomog, homog  # noqa: E402


def test_operator_apply_compose_identity_basics():
    """apply = the map; then = self THEN other; identity = no-op."""
    T = Operator.translation([3.0, -1.0])
    assert np.allclose(dehomog(T.apply(homog([2.0, 5.0]))), [5.0, 4.0])          # translation ADDS the displacement
    assert np.allclose(Operator.identity(3).apply(homog([2.0, 5.0])), homog([2.0, 5.0]))
    A, B = Operator.translation([1.0, 0.0]), Operator.translation([0.0, 2.0])
    assert np.allclose(dehomog(A.then(B).apply(homog([0.0, 0.0]))), [1.0, 2.0])  # do A then B


def test_translation_is_a_faithful_ABELIAN_representation():
    """Composition fidelity M(a)·M(b)=M(a∘b): translations compose ADDITIVELY, and every pair COMMUTES (abelian)."""
    a, b = [3.0, -1.0], [2.0, 4.0]
    assert Operator.translation(a).then(Operator.translation(b)) == Operator.translation([5.0, 3.0])
    assert Operator.translation(a).commutes_with(Operator.translation(b))


def test_interface_HOLDS_non_commuting_operators():
    """THE Stage-0 headline: composition is matrix product, so the interface does NOT bake in commutativity -- a rotation
    and a translation do NOT commute (rotate-then-move ≠ move-then-rotate). This is the door non-abelian structure needs."""
    T, R = Operator.translation([1.0, 0.0]), Operator.rotation(np.pi / 2)
    assert not T.commutes_with(R)
    assert T.then(R) != R.then(T)
    p = homog([0.0, 0.0])                                                        # concretely, from the origin, order matters:
    assert not np.allclose(T.then(R).apply(p), R.then(T).apply(p))               #   +x then rotate ≠ rotate then +x


def test_grid_path_integrate_IS_the_operator_acting():
    """l6_grid.path_integrate re-expressed as an Operator (block-diagonal phase rotation): identical output, and the grid
    operators compose FAITHFULLY and COMMUTE = an abelian (unitary) representation of translation."""
    import torch

    from tbt.l6_grid import L6_GridLocation
    g = L6_GridLocation()
    torch.manual_seed(0)
    z = torch.randn(g.dim)
    for disp in ([1.0, 0.0], [2.0, -3.0], [0.7, 0.4]):
        want = g.path_integrate(z, torch.tensor(disp, dtype=torch.float32)).numpy()
        assert np.allclose(g.operator(disp).apply(z.numpy()), want, atol=1e-4)   # the operator IS path_integrate
    a, b = [1.0, 2.0], [0.5, -1.0]
    assert np.allclose(g.operator(a).then(g.operator(b)).M, g.operator([1.5, 1.0]).M, atol=1e-4)   # composition fidelity
    assert g.operator(a).commutes_with(g.operator(b), tol=1e-6)                  # abelian


def test_L5_operator_reproduces_additive_move_NO_regression():
    """The GATE: L5's per-action OPERATOR reproduces the additive `move` -- on a NavGame-like path AND a counting
    succession the abelian behaviour is UNCHANGED when routed through the operator interface (Stage 0 = no regression)."""
    from tbt.l5_displacement import L5_Displacement
    L5 = L5_Displacement()
    moves = {0: (1.0, 0.0), 1: (-1.0, 0.0), 2: (0.0, -1.0), 3: (0.0, 1.0)}       # NavGame directions
    for a, d in moves.items():
        L5.observe_move(a, d)
    for a in moves:                                                             # one step: operator == additive move
        assert np.allclose(dehomog(L5.operator(a).apply(homog([4.0, 7.0]))), np.array([4.0, 7.0]) + L5.move(a))
    seq, pos_add, z = [0, 0, 3, 3, 0, 2, 1, 3], np.array([0.0, 0.0]), homog([0.0, 0.0])
    for a in seq:                                                               # a whole PATH: operators == additive accumulation
        pos_add = pos_add + np.asarray(L5.move(a))
        z = L5.operator(a).apply(z)
    assert np.allclose(dehomog(z), pos_add)
    succ = L5_Displacement()                                                    # COUNTING: a +1 operator composed n times = n
    succ.observe_move(0, (1.0, 0.0))
    z = homog([0.0, 0.0])
    for n in range(1, 25):
        z = succ.operator(0).apply(z)
        assert np.allclose(dehomog(z), [float(n), 0.0])


def test_LEARNED_operator_composition_fidelity_abelian_gate():
    """L6_NONABELIAN Stage 1 GATE (learnability, ABELIAN FIRST): operators LEARNED from noisy transitions on the grid code
    satisfy COMPOSITION FIDELITY -- but ONLY with the GAO orthogonality constraint. The constrained operator is a proper
    rotation (spectral radius 1) that COMMUTES (abelian), COMPOSES to the right multi-step target, and EXTRAPOLATES far
    beyond the trained one-steps; the UNCONSTRAINED fit predicts one step but its powers DRIFT (spectral radius != 1).
    Passing here means later NON-ABELIAN failures are diagnosable as non-abelianness, not the operator-learning machinery."""
    import torch

    from tbt.l6_grid import L6_GridLocation
    g = L6_GridLocation()
    rng = np.random.default_rng(0)
    SC, noise = 3.0, 0.03                                                       # |grid code| = sqrt(#modules) = 3

    def code(pos):
        return g.code_at(torch.tensor(pos, dtype=torch.float32)).numpy()

    def extrap_err(op, n):                                                      # apply the learned op n times from the origin
        z = code([0.0, 0.0])
        for _ in range(n):
            z = op.apply(z)
        return float(np.linalg.norm(z - code([float(n), 0.0])) / SC)

    # learn the +1 (succession) operator from noisy pairs, constrained vs not
    Z = np.stack([code([n, 0.0]) for n in range(201)])
    Zn = Z + noise * rng.standard_normal(Z.shape)
    plus_o = Operator.fit(Zn[:-1], Zn[1:], orthogonal=True)
    plus_u = Operator.fit(Zn[:-1], Zn[1:], orthogonal=False)
    assert abs(plus_o.spectral_radius() - 1.0) < 1e-6                           # constrained => a faithful rotation
    assert extrap_err(plus_o, 100) < 0.05                                       # composition stays faithful 100 steps out
    assert extrap_err(plus_o, 200) < 0.5 * extrap_err(plus_u, 200)             # the CONSTRAINT is required (unconstrained drifts ~5x)

    # 2-D: learn EAST + NORTH from a random walk; abelian => they must COMMUTE and COMPOSE
    moves = {"E": (1.0, 0.0), "N": (0.0, 1.0)}
    before = {k: [] for k in moves}
    after = {k: [] for k in moves}
    pos = np.array([0.0, 0.0])
    for _ in range(400):
        k = rng.choice(list(moves))
        before[k].append(code(pos) + noise * rng.standard_normal(g.dim))
        pos = pos + np.array(moves[k])
        after[k].append(code(pos) + noise * rng.standard_normal(g.dim))
    E, N = Operator.fit(before["E"], after["E"]), Operator.fit(before["N"], after["N"])
    assert np.linalg.norm(E.M @ N.M - N.M @ E.M) / SC < 0.05                    # learned E, N COMMUTE (abelian)
    assert np.linalg.norm(E.then(N).apply(code([0.0, 0.0])) - code([1.0, 1.0])) / SC < 0.05   # compose E∘N -> (1,1)


def test_ONLINE_operator_learns_from_a_stream__COVERAGE_not_constraint_is_the_bottleneck():
    """L6_NONABELIAN Stage 1 ONLINE gate (needed for an agent): `OnlineOperator` learns from a STREAM of transitions. On a
    BROAD-coverage stream it CONVERGES to a faithful operator (spectral radius 1, extrapolates far); the CONSTRAINT is never
    the bottleneck (orthogonality is a projection at read, so constraint⊥expressivity does NOT bite). The real requirement is
    COVERAGE: a LOCAL random walk's peaked occupancy under-samples the state space → a much worse operator."""
    import torch

    from tbt.l6_grid import L6_GridLocation
    g = L6_GridLocation()
    rng = np.random.default_rng(0)
    SC, noise = 3.0, 0.03

    def code(pos):
        return g.code_at(torch.tensor([pos, 0.0], dtype=torch.float32)).numpy()

    def extrap_err(op, n):
        z = code(0.0)
        for _ in range(n):
            z = op.apply(z)
        return float(np.linalg.norm(z - code(float(n))) / SC)

    # BROAD coverage (a forward sweep): converges to a faithful operator
    sweep = OnlineOperator(g.dim)
    for n in range(4000):
        sweep.observe(code(n) + noise * rng.standard_normal(g.dim), code(n + 1) + noise * rng.standard_normal(g.dim))
    assert abs(sweep.operator().spectral_radius() - 1.0) < 1e-6                 # constraint held THROUGHOUT (not the bottleneck)
    good = extrap_err(sweep.operator(), 100)
    assert good < 0.15                                                          # converged: composition fidelity / extrapolation

    # CONFINED walk (bouncing in a narrow band): the state space is genuinely under-covered → a much worse operator, no
    # matter the seed. This isolates COVERAGE as the bottleneck (the constraint is still perfectly held).
    confined = OnlineOperator(g.dim)
    pos = 0
    for _ in range(4000):
        npos = pos + int(rng.choice([1, -1]))
        if abs(npos) > 4:                                                       # reflect at ±4 -> occupancy pinned to 9 positions
            npos = pos - (npos - pos)
        if npos == pos + 1:                                                     # a +1 transition (the operator we read)
            confined.observe(code(pos) + noise * rng.standard_normal(g.dim), code(npos) + noise * rng.standard_normal(g.dim))
        pos = npos
    assert abs(confined.operator().spectral_radius() - 1.0) < 1e-6             # constraint STILL held (it is never the issue)
    assert extrap_err(confined.operator(), 100) > max(0.3, 3.0 * good)         # COVERAGE is the bottleneck, not the constraint


def test_L2_3_pose_machinery_is_ONE_operator_machinery():
    """L6_NONABELIAN Stage 1d (the cross-layer FOLD): L2/3's SO(2)+translation pose machinery is ONE instance of the
    operator machinery. (i) pose INFERENCE (`pose_between`) = `Operator.fit` (Procrustes IS the pose solve); (ii) pose
    APPLICATION (`apply_pose`) = the SE(2) `pose_operator` acting; (iii) poses COMPOSE non-abelianly (SE(2)); (iv) the
    CONTINUOUS rotation family = `power` (learn the step, read any pose) -- so the hand-coded `rot(θ)` becomes replaceable
    by the learned Operator, general for an abstract column whose group isn't SO(2)."""
    from tbt.l5_displacement import apply_pose, pose_between, pose_operator, rot
    rng = np.random.default_rng(0)
    model = rng.standard_normal((6, 2))
    theta = 0.7
    sensed = (rot(theta) @ model.T).T                                          # the patch rotated by theta

    # (i) INFERENCE: pose_between's angle and Operator.fit's rotation agree
    th = pose_between(list(model), list(sensed))[0]
    assert abs(((th - theta + np.pi) % (2 * np.pi)) - np.pi) < 1e-6            # pose_between recovers theta
    learned = Operator.fit(model, sensed, orthogonal=True)                     # Procrustes = the pose solve, group-general
    assert np.allclose(learned.M, rot(theta), atol=1e-6)                       # ... the SAME rotation

    # (ii) APPLICATION: apply_pose == the SE(2) pose_operator acting on homogeneous points
    cloud = rng.standard_normal((5, 2))
    t = (1.5, -2.0)
    P = pose_operator(theta, t)
    for got, loc in zip(apply_pose(list(cloud), theta, t), cloud):
        assert np.allclose(got, P.apply((loc[0], loc[1], 1.0))[:2])
        assert np.allclose(got, rot(theta) @ loc + np.array(t))               # ... and both match R·loc + t

    # (iii) SE(2) is NON-ABELIAN: rotate-then-translate ≠ translate-then-rotate
    assert not pose_operator(theta).commutes_with(pose_operator(0.0, (1.0, 0.0)))
    # (iv) CONTINUOUS: from the learned step, power reads off any fraction of the rotation
    assert np.allclose(learned.power(0.5).M, rot(theta / 2), atol=1e-6)


def test_operator_power_gives_the_CONTINUOUS_group_from_a_learned_step():
    """L6_NONABELIAN Stage 1d (the CONTINUOUS / Lie form): from a LEARNED discrete-step rotation operator, `power(t)`
    generates the CONTINUOUS group (rotation by ANY angle) -- so the pose group can be LEARNED (fit the step) and any pose
    read off (power a fractional amount), replacing L2/3's HAND-CODED `rot(θ)`. Also = fractional path integration."""
    R = Operator.rotation(np.pi / 3)                                            # a 60-degree step
    assert R.power(1.0) == R                                                    # t=1 -> self
    assert np.allclose(R.power(0.0).M, np.eye(3), atol=1e-9)                    # t=0 -> identity
    assert np.allclose(R.power(0.5).M, Operator.rotation(np.pi / 6).M, atol=1e-8)      # half step = 30 deg
    assert np.allclose(R.power(2.0).M, Operator.rotation(2 * np.pi / 3).M, atol=1e-8)  # double = 120 deg
    assert np.allclose(R.power(0.3).then(R.power(0.4)).M, R.power(0.7).M, atol=1e-8)   # 1-parameter subgroup: R^a∘R^b = R^(a+b)
    G = R.generator()
    assert np.allclose(G + G.T, 0.0, atol=1e-9)                                 # the generator is SKEW-SYMMETRIC (so(n))

    # LEARN the continuous group: fit the discrete step from rotation transitions, then power(t) reconstructs ANY angle
    rng = np.random.default_rng(0)
    theta_step = np.pi / 5                                                      # a 36-degree "turn" action
    Rt = Operator.rotation(theta_step)
    pts = np.array([[float(x), float(y), 1.0] for x, y in rng.standard_normal((30, 2))])
    learned = Operator.fit(pts, (Rt.M @ pts.T).T, orthogonal=True)
    assert np.allclose(learned.M, Rt.M, atol=1e-6)                             # recovered the step
    for t in (0.5, 1.0, 2.5, 4.0):                                             # the CONTINUOUS family from the learned step
        assert np.allclose(learned.power(t).M, Operator.rotation(t * theta_step).M, atol=1e-5)


def _s3():
    """S₃ (the smallest NON-ABELIAN group) as permutations of (0,1,2): elements, index, composition p∘q, one-hot, and the
    regular-representation operator for a generator g (the permutation matrix mapping one-hot(h) -> one-hot(g∘h))."""
    from itertools import permutations
    elts = list(permutations((0, 1, 2)))
    idx = {e: i for i, e in enumerate(elts)}

    def compose(p, q):
        return tuple(p[q[i]] for i in range(3))

    def onehot(e):
        v = np.zeros(len(elts))
        v[idx[e]] = 1.0
        return v

    def perm(g):
        P = np.zeros((len(elts), len(elts)))
        for h in elts:
            P[idx[compose(g, h)], idx[h]] = 1.0
        return P

    return elts, compose, onehot, perm


def test_NON_ABELIAN_gate_learned_operators_represent_S3():
    """L6_NONABELIAN Stage 1 -- THE non-abelian gate (the refactor's REASON TO EXIST). Learn per-generator operators for S₃
    from Cayley-graph transitions (one-hot(h) -> one-hot(g∘h)); the LEARNED operators must (i) be FAITHFUL (recover the
    permutation matrices), (ii) NOT commute (the order-dependence a commuting-phase grid CANNOT hold), (iii) satisfy the
    group RELATIONS (a²=e, (ab)³=e), (iv) COMPOSE faithfully over the group. And a COMMUTING (abelian) model is order-blind,
    so it has IRREDUCIBLE error on the order-dependent composite -- exactly why non-abelian structure needs matrices."""
    elts, compose, onehot, perm = _s3()
    a, b = (1, 0, 2), (0, 2, 1)                                                 # two transpositions generate S₃; a∘b ≠ b∘a

    def learn(g, orthogonal=True):
        before = np.stack([onehot(h) for h in elts])
        after = np.stack([onehot(compose(g, h)) for h in elts])
        return Operator.fit(before, after, orthogonal=orthogonal)

    Ma, Mb = learn(a), learn(b)
    assert np.allclose(Ma.M, perm(a), atol=1e-6) and np.allclose(Mb.M, perm(b), atol=1e-6)   # (i) FAITHFUL
    assert not Ma.commutes_with(Mb)                                                          # (ii) NON-commuting...
    assert np.linalg.norm(Ma.M @ Mb.M - Mb.M @ Ma.M) > 1.0                                   #      ...by a clear margin
    assert np.allclose(Ma.then(Ma).M, np.eye(len(elts)), atol=1e-6)                          # (iii) a² = e (transposition)
    ab = Ma.then(Mb)                                                                         #       (ab) is a 3-cycle...
    assert np.allclose(ab.then(ab).then(ab).M, np.eye(len(elts)), atol=1e-6)                 #       ...(ab)³ = e
    e = onehot((0, 1, 2))                                                                    # (iv) COMPOSITION fidelity:
    assert np.allclose(Ma.then(Mb).apply(e), onehot(compose(b, a)))                          #      a-op then b-op lands on b∘a
    assert np.allclose(Mb.then(Ma).apply(e), onehot(compose(a, b)))                          #      reversed -> a∘b (a DIFFERENT state)

    # THE ABELIAN CONTRAST: the two orders land on different states, but a COMMUTING model predicts them EQUAL -> its error
    # is >= half their distance, while the learned MATRIX operators nail both. This gap is the whole reason for the refactor.
    true_ab, true_ba = onehot(compose(a, b)), onehot(compose(b, a))
    assert np.linalg.norm(true_ab - true_ba) > 1.0                                           # a∘b and b∘a are different states
    matrix_err = max(np.linalg.norm(Mb.then(Ma).apply(e) - true_ab), np.linalg.norm(Ma.then(Mb).apply(e) - true_ba))
    commuting_lower_bound = np.linalg.norm(true_ab - true_ba) / 2.0                          # any commuting model: ab == ba
    assert matrix_err < 1e-6 < commuting_lower_bound                                         # matrices ~0; abelian >= 0.7

    # ONLINE + non-abelian: streaming a random walk on the 6-node Cayley graph (coverage is trivial) recovers the operators
    online = {a: OnlineOperator(len(elts)), b: OnlineOperator(len(elts))}
    rng = np.random.default_rng(0)
    h = (0, 1, 2)
    for _ in range(400):
        g = (a, b)[int(rng.integers(2))]
        online[g].observe(onehot(h), onehot(compose(g, h)))
        h = compose(g, h)
    assert np.allclose(online[a].operator().M, perm(a), atol=1e-6)                           # online learns the non-abelian op too
    assert not online[a].operator().commutes_with(online[b].operator())


def test_S2_factored_periods_from_the_spectrum_the_smuggle_guard():
    """L6_NONABELIAN Stage 2 (the FACTORED case) -- the PRINCIPLED smuggle guard. Cyclic factors are DISCOVERED from the
    operator's SPECTRUM (root-of-unity eigenvalues = periodic invariant subspaces), NOT from the input labelling -- so the
    factorisation cannot be smuggled in. (i) A single n-CYCLE has a PRIMITIVE nth root -> period n. (ii) NEGATIVE CONTROL: a
    non-recurring shift (nilpotent) AND a random map have NO root-of-unity structure -> NO period (a raw count stays a line).
    (iii) The spectrum DISTINGUISHES one big cycle (Z/6: a primitive 6th root) from a genuine PRODUCT of small cycles
    (Z/2⊕Z/3: only 2nd+3rd roots, NO 6) -- same group order, different dynamics -> the notational factoring of a raw count is
    NOT hallucinated; only genuine product structure in the dynamics is found."""
    from tbt.operator import Operator, discover_periods

    def cycle(n):                                                              # the regular-rep n-cycle shift (a permutation)
        M = np.zeros((n, n))
        for i in range(n):
            M[(i + 1) % n, i] = 1.0
        return Operator(M)

    def block_diag(*mats):
        n = sum(m.shape[0] for m in mats)
        B = np.zeros((n, n))
        i = 0
        for m in mats:
            k = m.shape[0]
            B[i:i + k, i:i + k] = m
            i += k
        return B

    assert 4 in discover_periods(cycle(4))                                     # (i) Z/4 -> period 4
    assert 6 in discover_periods(cycle(6))                                     #     Z/6 single cycle -> a PRIMITIVE 6th root

    # (ii) NEGATIVE CONTROL -- no periodic structure -> nothing discovered (the smuggle guard)
    nilpotent = np.zeros((6, 6))
    for i in range(5):
        nilpotent[i + 1, i] = 1.0                                             # open line, no wrap -> all eigenvalues 0
    assert discover_periods(Operator(nilpotent)) == []
    rng = np.random.default_rng(0)
    assert discover_periods(Operator(rng.standard_normal((6, 6)))) == []       # structureless map -> no roots of unity

    # (iii) PRODUCT of small cycles (Z/2 ⊕ Z/3) is DISTINCT from one 6-cycle: periods {2,3}, NO 6
    prod = discover_periods(Operator(block_diag(cycle(2).M, cycle(3).M)))
    assert 2 in prod and 3 in prod and 6 not in prod                          # genuine product, not a smuggled 6-factoring

    # (iv) from a LEARNED operator (not hand-built): stream a walk on the Z/6 cycle -> the period is discovered
    from tbt.operator import OnlineOperator
    n = 6
    onl = OnlineOperator(n)
    h = 0
    for _ in range(60):
        b = np.zeros(n); b[h] = 1.0
        a = np.zeros(n); a[(h + 1) % n] = 1.0
        onl.observe(b, a); h = (h + 1) % n
    assert 6 in discover_periods(onl.operator())                              # the factor period read from the LEARNED operator's spectrum


def test_S2c_recolor_content_becomes_a_factorable_cycle_operator():
    """L6_NONABELIAN Stage 2 step (c) -- the FM `g × x` unification, the content half: L5's `recolor` transition map (the
    *what changed* operator) bridges into a permutation OPERATOR (`permutation_operator` / `col.content_operator`), so
    content dynamics are FACTORABLE the same way as structure. A learned TOGGLE is recognised as a 2-cycle and a 3-counter as
    a 3-cycle, purely from the content operator's spectrum -- so `x` joins `g` as a first-class factor of the one model."""
    from tbt.column import CorticalColumn
    from tbt.operator import discover_periods, permutation_operator

    # the bridge in isolation: a toggle map -> a 2-cycle; a 3-counter -> a 3-cycle
    op2, _a2 = permutation_operator({0: 1, 1: 0})
    assert discover_periods(op2) == [2]
    op3, _a3 = permutation_operator({0: 1, 1: 2, 2: 0})
    assert discover_periods(op3) == [3]

    # LEARNED through the column: config-state transitions whose CONTENT toggles -> L5.recolor -> content_operator = 2-cycle
    col = CorticalColumn(n_entities=8, seed=0)
    a = 0
    col.L5.observe(((1, (0, 0), 0),), a, ((1, (0, 0), 1),))                     # content 0 -> 1 (in place)
    col.L5.observe(((1, (0, 0), 1),), a, ((1, (0, 0), 0),))                     # content 1 -> 0
    op, alphabet = col.content_operator((1,), a)
    assert set(alphabet) == {(0,), (1,)} and discover_periods(op) == [2]        # a learned toggle = a content 2-cycle
    assert col.content_operator((1,), 99) is None                              # no content transition learned -> None


def test_S2c_predict_gx_binds_structure_and_content():
    """L6_NONABELIAN Stage 2 step (c) -- the `g × x` FORWARD prediction binds the two halves the fragmented FM kept apart:
    the STRUCTURE predictor (`predict` -> g') with the CONTENT map (`feature_at` -> x' at g'). This is TEM's objective
    ('predict the next observation | position, action') as ONE model -- where I'll be + what's there -- vs the location-blind
    `field_rule` CA."""
    from tbt.column import CorticalColumn
    col = CorticalColumn(n_entities=8, seed=0)
    for _ in range(5):                                                          # train the structure g (A<->B) so the SR place codes are non-trivial
        col.observe("A", 0, "B")
        col.observe("B", 1, "A")
    col.sense_at("B", 3)                                                        # bind content x = feature 3 at node B (= g)
    g2, x2 = col.predict_gx("A", 0)
    assert g2 == "B" and x2 == 3                                                # predicts: after action 0 from A -> at B, feature 3 (where + what)


class _CounterToggle:
    """A two-factor MICROWORLD: a COUNTER (Z/n, advanced by TICK) and an independent TOGGLE (Z/2, flipped by FLIP). The two
    actions act on ORTHOGONAL factors, so the dynamics are a genuine PRODUCT of cycles (Z/n × Z/2). Encoded as one-hot(counter)
    ⊕ one-hot(toggle) so the operators are LEARNABLE from the transition stream."""

    def __init__(self, n=6):
        self.n, self.c, self.t = n, 0, 0

    def code(self):
        v = np.zeros(self.n + 2)
        v[self.c] = 1.0
        v[self.n + self.t] = 1.0
        return v

    def step(self, action):
        if action == "TICK":
            self.c = (self.c + 1) % self.n
        else:
            self.t ^= 1
        return self.code()


def test_S2_factored_closure_product_of_cycles_from_a_microworld():
    """L6_NONABELIAN Stage 2 (FACTORED closure) -- on a two-factor MICROWORLD (a counter Z/n + a toggle Z/2), LEARN the two
    operators online, then `factor_group` decomposes the dynamics into a DIRECT PRODUCT of cycles guarded by predictive
    sufficiency (commute + orders multiply to |G| = unique factorisation). It (i) recovers [(TICK, n), (FLIP, 2)] from the
    LEARNED operators; (ii) is BASIS-INDEPENDENT -- a random orthogonal change of code gives the SAME factoring (the factors
    live in the operators' joint eigenstructure, not the code's axes -> not smuggled from a pre-separated code); (iii) the
    GUARD rejects the wrong factoring -- non-commuting generators (S₃) and overlapping factors (TICK & TICK²) return None."""
    from tbt.operator import Operator, OnlineOperator, discover_periods, factor_group

    n = 6
    w = _CounterToggle(n)
    tick, flip = OnlineOperator(n + 2), OnlineOperator(n + 2)
    rng = np.random.default_rng(0)
    for _ in range(300):                                                       # drive a mix; route each transition to its operator
        a = "TICK" if rng.random() < 0.5 else "FLIP"
        before = w.code()
        after = w.step(a)
        (tick if a == "TICK" else flip).observe(before, after)
    TICK, FLIP = tick.operator(), flip.operator()

    assert n in discover_periods(TICK) and discover_periods(FLIP) == [2]       # the periods, from the LEARNED operators
    assert factor_group([TICK, FLIP]) == [(0, n), (1, 2)]                      # (i) the product-of-cycles decomposition

    # (ii) BASIS-INDEPENDENCE -- a random orthogonal recoding scrambles the one-hot blocks, factoring is unchanged
    Q, _ = np.linalg.qr(rng.standard_normal((n + 2, n + 2)))
    TICKr, FLIPr = Operator(Q @ TICK.M @ Q.T), Operator(Q @ FLIP.M @ Q.T)
    assert factor_group([TICKr, FLIPr]) == [(0, n), (1, 2)]

    # (iii) the GUARD rejects non-factorable dynamics
    _elts, _compose, _onehot, perm = _s3()
    assert factor_group([Operator(perm((1, 0, 2))), Operator(perm((0, 2, 1)))]) is None   # S₃ non-abelian -> no direct product
    assert factor_group([TICK, TICK.then(TICK)]) is None                      # overlapping factors (⟨TICK⟩ = ⟨TICK²⟩ region) -> product ≠ |G|


def test_S2_nested_carry_is_the_coupling_detected_by_predictive_sufficiency():
    """L6_NONABELIAN Stage 2 (the NESTED/coupled case = place value / CARRY). The honest boundary + the principled result:
    (i) HONEST BOUNDARY -- a raw single '+1' count is ONE big cycle Z/(b^k); from that generator alone there is NO free
    digit-factoring (`factor_group([+1]) = [(0, b^k)]`). The notational factoring must come from OBSERVING the digits.
    (ii) Given the digits, CARRY is the COUPLING, detected by predictive sufficiency: the UNITS digit is an autonomous cycle
    (its next value depends only on itself -> sufficient), but the TENS digit is NOT sufficient alone (its next value depends
    on whether the units WRAPPED -> the carry). (iii) CONTRAST: an INDEPENDENT counter+toggle passes predictive sufficiency
    on BOTH factors (a true direct product) -- so the checker discriminates coupled (odometer) from independent."""
    from tbt.operator import Operator, discover_periods, factor_group, is_predictively_sufficient

    b, k = 4, 2
    N = b ** k

    # (i) HONEST BOUNDARY: the raw +1 over the full one-hot code is ONE cycle of order N (no free factoring from one generator)
    plus1 = np.zeros((N, N))
    for i in range(N):
        plus1[(i + 1) % N, i] = 1.0
    assert N in discover_periods(Operator(plus1))                              # a primitive Nth root -> one big cycle
    assert factor_group([Operator(plus1)]) == [(0, N)]                         # one generator -> one Z/N cycle, NOT b*b

    # the ODOMETER's +1 as digit-transitions (units then carry)
    def digits(v):
        return tuple((v // (b ** i)) % b for i in range(k))
    odo = [(digits(v), digits((v + 1) % N)) for v in range(N)]
    # (ii) units autonomous (sufficient); tens COUPLED to the units wrap (NOT sufficient) = carry detected
    assert is_predictively_sufficient(odo, lambda d: d[0])
    assert not is_predictively_sufficient(odo, lambda d: d[1])

    # (iii) CONTRAST -- an INDEPENDENT counter(Z/n) + toggle(Z/2) under one combined step: BOTH factors sufficient (product)
    n = 5
    indep = [((c, t), ((c + 1) % n, 1 - t)) for c in range(n) for t in range(2)]
    assert is_predictively_sufficient(indep, lambda s: s[0])                   # counter autonomous
    assert is_predictively_sufficient(indep, lambda s: s[1])                   # toggle autonomous -> a true direct product


def test_S2_discover_relations_by_loop_closure_cyclic_nonabelian_abelian():
    """L6_NONABELIAN Stage 2 -- DISCOVER relations by LOOP CLOSURE (the quotient), the Stage-2 gate on KNOWN presentations.
    Predictive sufficiency = operator EQUALITY (identical action ⇒ same element ⇒ same future). A 90° rotation generator:
    closure finds the CYCLIC group Z/4 with the relation g⁴=e (a length-4 loop to identity). S₃ (two transpositions):
    closure finds the 6-element NON-ABELIAN group -- word (a,b) is a DISTINCT element from (b,a) (no commutativity closure).
    Two COMMUTING rotations (Z/6): the SAME order 6 but (a,b)==(b,a) DOES close -- same order, opposite RELATIONS = the
    master boundary (free/abelian READ-OFF vs quotient SEARCH) made discoverable from the learned operators alone."""
    from tbt.operator import Operator, discover_group

    def op_of(word, gens):
        m = np.eye(gens[0].dim)
        for gi in word:
            m = gens[gi].M @ m
        return m

    # Z/4: one 90° rotation -> 4 elements; g⁴=e discovered as a length-4 loop back to identity
    elts4, rel4 = discover_group([Operator.rotation(np.pi / 2)])
    assert len(elts4) == 4
    assert any(w == (0, 0, 0, 0) and eqw == () for w, eqw in rel4)             # g⁴ = e (loop closure to identity)

    # S₃ (non-abelian, order 6): 6 elements discovered; (a,b) is a DISTINCT element from (b,a)
    _elts, _compose, _onehot, perm = _s3()
    a, b = (1, 0, 2), (0, 2, 1)
    Ma, Mb = Operator(perm(a)), Operator(perm(b))
    eltsS3, _relS3 = discover_group([Ma, Mb])
    assert len(eltsS3) == 6 and not Ma.commutes_with(Mb)
    assert not np.allclose(op_of((0, 1), [Ma, Mb]), op_of((1, 0), [Ma, Mb]))   # (a,b) ≠ (b,a): no commutativity closure (SEARCH)

    # Z/6 (two COMMUTING rotations, 120° & 180°): same order 6, but commutativity CLOSES
    Ra, Rb = Operator.rotation(2 * np.pi / 3), Operator.rotation(np.pi)
    elts6, _rel6 = discover_group([Ra, Rb])
    assert len(elts6) == 6 and Ra.commutes_with(Rb)
    assert np.allclose(op_of((0, 1), [Ra, Rb]), op_of((1, 0), [Ra, Rb]))       # (a,b) == (b,a): commutativity (READ-OFF)
