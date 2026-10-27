"""A small 2D plane-stress finite-element solver (docs/SURROGATES.md): the simulator that the surrogates learn.

Bilinear quadrilaterals on a rectangular grid (E = 1, nu = 0.3, thickness 1), sparse direct solve. Holes are modelled
by each element's solid fraction (sampled on a 16 x 16 grid inside the element): E_element = E_min + (1 - E_min) *
solid fraction. So the outputs change smoothly with a hole's size and position instead of jumping as elements switch
on and off.

Three problems, all dimensionless (E = 1, load 1, thickness 1):
    beam(L)                       tip deflection of a clamped cantilever of length L and height 1, tip shear load
    plate_hole(r, yc)             compliance of a unit square plate in uniaxial tension with a hole of radius r centred
                                  at (0.5, yc), relative to the plate without the hole
    bracket(xc, yc, r)            tip deflection of a clamped 4 x 1 cantilever with a hole of radius r at (xc, yc),
                                  relative to the solid cantilever
"""
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

NU, E_MIN = 0.3, 1e-6


@lru_cache(maxsize=None)
def q4_stiffness(a, b, nu=NU):
    """[8, 8] stiffness of an a x b bilinear element (E = 1, plane stress), 2 x 2 Gauss quadrature.
    Local node order: (0, 0), (a, 0), (a, b), (0, b); dofs (ux, uy) per node."""
    D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu ** 2)
    K = np.zeros((8, 8))
    g = 1 / np.sqrt(3)
    for xi in (-g, g):
        for eta in (-g, g):
            dN_dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4
            dN_deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4
            dx, dy = dN_dxi * 2 / a, dN_deta * 2 / b
            B = np.zeros((3, 8))
            B[0, 0::2], B[1, 1::2] = dx, dy
            B[2, 0::2], B[2, 1::2] = dy, dx
            K += B.T @ D @ B * (a * b / 4)
    return K


def node(i, j, nely):
    return i * (nely + 1) + j


def solve(nelx, nely, lx, ly, solid, fixed, force):
    """solid: [nelx, nely] solid fractions; fixed: dof indices; force: [n_dof]. Returns the displacements."""
    a, b = lx / nelx, ly / nely
    KE = q4_stiffness(round(a, 12), round(b, 12))
    i, j = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    n1, n2 = node(i, j, nely), node(i + 1, j, nely)
    n3, n4 = node(i + 1, j + 1, nely), node(i, j + 1, nely)
    edof = np.stack([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1], -1).reshape(-1, 8)
    Ee = (E_MIN + (1 - E_MIN) * solid).reshape(-1)
    rows = np.repeat(edof, 8, 1).reshape(-1)
    cols = np.tile(edof, 8).reshape(-1)
    vals = (KE.reshape(1, 64) * Ee[:, None]).reshape(-1)
    n_dof = 2 * (nelx + 1) * (nely + 1)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsc()
    free = np.setdiff1d(np.arange(n_dof), fixed)
    u = np.zeros(n_dof)
    u[free] = spla.spsolve(K[free][:, free], force[free])
    return u


def solid_fraction(nelx, nely, lx, ly, holes, sub=16):
    """[nelx, nely]: the solid fraction of each element, with circular holes [(xc, yc, r)]."""
    a, b = lx / nelx, ly / nely
    s = (np.arange(sub) + 0.5) / sub
    x = (np.arange(nelx)[:, None] + s[None, :]).reshape(-1) * a          # sample points along x
    y = (np.arange(nely)[:, None] + s[None, :]).reshape(-1) * b
    X, Y = np.meshgrid(x, y, indexing="ij")
    inside = np.zeros_like(X, bool)
    for xc, yc, r in holes:
        inside |= (X - xc) ** 2 + (Y - yc) ** 2 < r ** 2
    return 1 - inside.reshape(nelx, sub, nely, sub).mean((1, 3))


def edge_load(nelx, nely, total, direction):
    """A uniform load on the right edge (consistent nodal forces), total force `total` along x (0) or y (1)."""
    f = np.zeros(2 * (nelx + 1) * (nely + 1))
    w = np.full(nely + 1, 1.0 / nely)
    w[[0, -1]] /= 2
    for j in range(nely + 1):
        f[2 * node(nelx, j, nely) + direction] += total * w[j]
    return f


def clamped_left(nely):
    return np.array([d for j in range(nely + 1) for d in (2 * node(0, j, nely), 2 * node(0, j, nely) + 1)])


def tip_deflection(nelx, nely, lx, ly, holes):
    u = solve(nelx, nely, lx, ly, solid_fraction(nelx, nely, lx, ly, holes), clamped_left(nely),
              edge_load(nelx, nely, -1.0, 1))
    return -np.mean([u[2 * node(nelx, j, nely) + 1] for j in range(nely + 1)])


def beam(L, per_unit=10):
    """Tip deflection of a clamped L x 1 cantilever under a unit tip shear (10 elements per unit length)."""
    return tip_deflection(max(1, round(per_unit * L)), per_unit, L, 1.0, [])


PLATE_N = 40


def _plate_compliance(holes):
    n = PLATE_N
    fixed = np.array([2 * node(0, j, n) for j in range(n + 1)] + [2 * node(0, 0, n) + 1])   # ux = 0 left; one uy
    f = edge_load(n, n, 1.0, 0)
    u = solve(n, n, 1.0, 1.0, solid_fraction(n, n, 1.0, 1.0, holes), fixed, f)
    return f @ u


@lru_cache(maxsize=1)
def _plate_reference():
    return _plate_compliance([])


def plate_hole(r, yc):
    """Compliance of a unit square plate in uniaxial tension with a hole at (0.5, yc), relative to no hole."""
    return _plate_compliance([(0.5, yc, r)]) / _plate_reference()


BRACKET = dict(nelx=80, nely=20, lx=4.0, ly=1.0)


@lru_cache(maxsize=1)
def _bracket_reference():
    return tip_deflection(holes=[], **BRACKET)


def bracket(xc, yc, r):
    """Tip deflection of a clamped 4 x 1 cantilever with a hole at (xc, yc), relative to the solid cantilever."""
    return tip_deflection(holes=[(xc, yc, r)], **BRACKET) / _bracket_reference()
