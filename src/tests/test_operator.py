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

from tbt.operator import Operator, dehomog, homog  # noqa: E402


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
