"""GridSSM — the grid-cell modification to a Mamba-family recurrent block, tested in isolation.

A standard SSM transition is a scalar decay `exp(A·dt) ∈ (0,1)` — the state forgets. The grid-cell
modification: make the state transition a **rotation** `R = exp(G − Gᵀ)` (norm-preserving, no decay), so
the recurrent state **path-integrates** the input movements instead of forgetting them. Feeding a stream
of movements rotates the state; the state after cumulative movement `m` is the grid code for location `m`.
Multi-scale = a block-diagonal rotation (several frequencies) — the slots Mamba already has for its
per-channel complex SSM.

This file validates the mechanism *in isolation* (per the build plan: test the block before a full model):
the recurrent state should path-integrate and support **zero-shot navigation** — decode any cumulative
movement, including movement sequences/sums never trained on — matching the hand-applied-operator result.
Tiny + CPU; no full training (the 4GB constraint is irrelevant here).
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

# Reuse the validated anti-collapse (SIGReg — the standard; VICReg gave no advantage when tested).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "RecurrentWorldModel"))
from baselines.sigreg import SIGReg  # noqa: E402


class GridSSM(nn.Module):
    """discover=False (impose): the transition is forced to be a rotation `R = exp(G − Gᵀ)`.
    discover=True: the transition is a *free* matrix — the objective (closure + anti-collapse) must
    *discover* that a rotation is the solution, with nothing baked in."""

    def __init__(self, dim: int = 64, modulus: int = 17, seed: int = 0,
                 discover: bool = False, primitive: bool = False):
        super().__init__()
        torch.manual_seed(seed)
        self.dim = dim
        self.m = modulus
        self.discover = discover
        self.primitive = primitive
        if discover:
            self.Rfree = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))  # unconstrained
        elif primitive:
            # reserve a 2-D slice for a FIXED full-period rotation (exactly 2π/m) so the orbit can
            # never alias (period = m for any m, incl. composite); the rest stays learned/multi-frequency
            self.G = nn.Parameter(0.01 * torch.randn(dim - 2, dim - 2))
            a = 2 * math.pi / modulus
            self.register_buffer("Rprim", torch.tensor([[math.cos(a), -math.sin(a)],
                                                        [math.sin(a), math.cos(a)]]))
        else:
            self.G = nn.Parameter(0.01 * torch.randn(dim, dim))  # rotation generator (imposed)
        self.h0 = nn.Parameter(0.1 * torch.randn(dim))           # location-0 state

    def R(self) -> torch.Tensor:
        if self.discover:
            return self.Rfree
        if self.primitive:
            return torch.block_diag(self.Rprim, torch.matrix_exp(self.G - self.G.t()))
        return torch.matrix_exp(self.G - self.G.t())

    def location_codes(self) -> torch.Tensor:
        """The recurrent state after 0,1,...,m-1 unit movements: hₐ = Rᵃ·h0 (the grid code per location)."""
        R = self.R()
        h = self.h0
        codes = [h]
        for _ in range(self.m - 1):
            h = R @ h
            codes.append(h)
        return torch.stack(codes)                                # (m, dim)

    def integrate(self, moves: torch.Tensor) -> torch.Tensor:
        """Path integration: process a (B, T) stream of integer movements, return the final state (B, dim).
        Each step applies R^move — the recurrent transition driven by the input movement."""
        R = self.R()
        Rp = [torch.eye(self.dim, device=R.device)]
        for _ in range(self.m):
            Rp.append(R @ Rp[-1])                                # R^0 .. R^m
        Rp = torch.stack(Rp)
        h = self.h0.expand(moves.shape[0], -1)
        for t in range(moves.shape[1]):
            h = torch.bmm(Rp[moves[:, t] % self.m], h.unsqueeze(-1)).squeeze(-1)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode a state to a location by nearest location code (dot product)."""
        return (h @ self.location_codes().t()).argmax(-1)


def train_grid_ssm(m=17, dim=64, steps=800, lr=3e-3, seed=0, w_sig=1.0, discover=False, primitive=False):
    """Objective (the sensorimotor/equivariance loss, shaping the *recurrent state*): orbit closes
    (Rᵐ·h0 = h0 → the ring) + SIGReg anti-collapse. Path integration is automatic (the recurrence IS R),
    so zero-shot navigation follows. (Robust for m≳11 — realistic reference-frame sizes; very small rings
    m≤7 train unreliably, an anti-collapse finickiness at tiny moduli, not a path-integration failure.)
    discover=True uses a free transition (must discover the rotation); False imposes `R=exp(G−Gᵀ)`.
    primitive=True reserves a fixed full-period (2π/m) rotation slice so composite m can't alias."""
    model = GridSSM(dim, m, seed, discover=discover, primitive=primitive)
    sig = SIGReg(n_slices=256)
    g = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        codes = model.location_codes()
        R = model.R()
        l_close = (R @ codes[-1] - model.h0).pow(2).sum()        # Rᵐ·h0 = h0 (orbit closes)
        loss = l_close + w_sig * sig(codes, generator=g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def navigation_acc(model, n=2000, max_steps=6, seed=1):
    """Zero-shot navigation: random movement sequences (sums/compositions not trained on), decode the
    path-integrated final state, check it equals the true cumulative location."""
    g = torch.Generator().manual_seed(seed)
    moves = torch.randint(0, model.m, (n, max_steps), generator=g)
    final = model.integrate(moves)
    target = moves.sum(1) % model.m
    return (model.decode(final) == target).float().mean().item()


class MultiScaleGridSSM:
    """Multiple ring-modules at coprime periods (the exponential-range grid code). Carries the full
    gap-A multi-scale geometry: CRT range AND the **error-correcting** decode (redundant modules outvote
    noise — Sreenivasan & Fiete's analog error-correcting code). Code a range smaller than lcm for
    redundancy; decode by nearest *valid* codeword (uses the redundancy) vs naive per-module CRT (fragile)."""

    def __init__(self, periods, dim=64, steps=800, seed=0):
        self.periods = list(periods)
        self.modules = [train_grid_ssm(m=p, dim=dim, steps=steps, seed=seed) for p in periods]
        self.lcm = 1
        for p in periods:
            self.lcm = self.lcm * p // math.gcd(self.lcm, p)

    @torch.no_grad()
    def codeword(self, positions):
        """Full multi-scale code (concat of module codes) for given positions (LongTensor)."""
        return torch.cat([mod.location_codes()[positions % p]
                          for mod, p in zip(self.modules, self.periods)], dim=-1)

    @torch.no_grad()
    def decode_naive(self, full_code):
        """Per-module nearest residue → CRT. Fragile: one wrong module ruins the answer."""
        d = self.modules[0].dim
        res = [mod.decode(full_code[:, k * d:(k + 1) * d]) for k, mod in enumerate(self.modules)]
        M = self.lcm
        out = torch.zeros(full_code.shape[0], dtype=torch.long)
        for r, p in zip(res, self.periods):
            Mi = M // p
            out = out + r * Mi * pow(Mi, -1, p)
        return out % M

    @torch.no_grad()
    def decode_error_correcting(self, full_code, rng):
        """Nearest *valid* codeword over the coded range [0, rng) — uses cross-module redundancy."""
        valid = self.codeword(torch.arange(rng))                  # (rng, K*dim)
        return torch.cdist(full_code, valid).argmin(-1)


class GridSSM2D(nn.Module):
    """Two-generator grid block (a 2-torus). State path-integrates 2-D movements: each step applies
    Rx^dx · Ry^dy. Same rotational-transition mechanism, one generator per axis (multi-input SSM)."""

    def __init__(self, dim: int = 64, mx: int = 5, my: int = 5, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.dim, self.mx, self.my = dim, mx, my
        # Each generator acts on its own channel-subspace (block-diagonal) — independent grid modules
        # per axis. Guarantees a faithful product action (distinct codes) and exact commutativity.
        self.half = dim // 2
        self.Gx = nn.Parameter(0.01 * torch.randn(self.half, self.half))
        self.Gy = nn.Parameter(0.01 * torch.randn(self.half, self.half))
        self.h0 = nn.Parameter(0.1 * torch.randn(dim))

    def Rx(self):
        rx = torch.matrix_exp(self.Gx - self.Gx.t())
        return torch.block_diag(rx, torch.eye(self.half))           # rotate first half only

    def Ry(self):
        ry = torch.matrix_exp(self.Gy - self.Gy.t())
        return torch.block_diag(torch.eye(self.half), ry)           # rotate second half only

    def _pows(self, R, k):
        P = [torch.eye(self.dim)]
        for _ in range(k):
            P.append(R @ P[-1])
        return torch.stack(P)

    def location_codes(self):
        Rxp, Ryp = self._pows(self.Rx(), self.mx - 1), self._pows(self.Ry(), self.my - 1)
        return torch.stack([Ryp[y] @ (Rxp[x] @ self.h0)
                            for x in range(self.mx) for y in range(self.my)])   # index x*my+y

    def integrate(self, mxv, myv):
        Rxp, Ryp = self._pows(self.Rx(), self.mx), self._pows(self.Ry(), self.my)
        h = self.h0.expand(mxv.shape[0], -1)
        for t in range(mxv.shape[1]):
            h = torch.bmm(Rxp[mxv[:, t] % self.mx], h.unsqueeze(-1)).squeeze(-1)
            h = torch.bmm(Ryp[myv[:, t] % self.my], h.unsqueeze(-1)).squeeze(-1)
        return h

    def decode(self, h):
        return (h @ self.location_codes().t()).argmax(-1)


def train_grid_ssm_2d(mx=5, my=5, dim=64, steps=1500, lr=3e-3, seed=0, w_sig=5.0):
    """Two generators: closure on each axis + Rx,Ry commute + SIGReg. w_sig=5 (Finding 6: anti-collapse
    must scale with the number of generators)."""
    model = GridSSM2D(dim, mx, my, seed)
    sig = SIGReg(n_slices=256)
    g = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        codes = model.location_codes()
        Rx, Ry = model.Rx(), model.Ry()
        hx, hy = model.h0, model.h0
        for _ in range(mx): hx = Rx @ hx
        for _ in range(my): hy = Ry @ hy
        l_close = (hx - model.h0).pow(2).sum() + (hy - model.h0).pow(2).sum()
        l_comm = (Rx @ Ry - Ry @ Rx).pow(2).sum()
        loss = l_close + l_comm + w_sig * sig(codes, generator=g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def navigation_acc_2d(model, n=2000, max_steps=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    mxv = torch.randint(0, model.mx, (n, max_steps), generator=g)
    myv = torch.randint(0, model.my, (n, max_steps), generator=g)
    tx, ty = mxv.sum(1) % model.mx, myv.sum(1) % model.my
    pred = model.decode(model.integrate(mxv, myv))
    return ((pred // model.my == tx) & (pred % model.my == ty)).float().mean().item()


class GridSSMHex(nn.Module):
    """Hexagonal / conformal-isometry block: three generators sharing one plane at 120°, with
    R0·R1·R2 = I (built in: R2 = (R0·R1)⁻¹ is the third, negative-diagonal direction). The isotropic
    2-D lattice. Codes = R1^r·R0^q·h0; isotropy (equal step size in all 3 directions) is enforced by a
    conformal term. (Shared-plane generators — the collision-prone case for a generative recurrence.)"""

    def __init__(self, dim=64, m=7, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.dim, self.m = dim, m
        self.G0 = nn.Parameter(0.01 * torch.randn(dim, dim))
        self.G1 = nn.Parameter(0.01 * torch.randn(dim, dim))
        self.h0 = nn.Parameter(0.1 * torch.randn(dim))

    def R0(self): return torch.matrix_exp(self.G0 - self.G0.t())
    def R1(self): return torch.matrix_exp(self.G1 - self.G1.t())
    def R2(self): return torch.linalg.inv(self.R0() @ self.R1())     # R0·R1·R2 = I (hex closure)

    def _pows(self, R, k):
        P = [torch.eye(self.dim)]
        for _ in range(k):
            P.append(R @ P[-1])
        return torch.stack(P)

    def location_codes(self):
        R0p, R1p = self._pows(self.R0(), self.m - 1), self._pows(self.R1(), self.m - 1)
        return torch.stack([R1p[r] @ (R0p[q] @ self.h0) for q in range(self.m) for r in range(self.m)])

    def integrate(self, qv, rv):
        R0p, R1p = self._pows(self.R0(), self.m), self._pows(self.R1(), self.m)
        h = self.h0.expand(qv.shape[0], -1)
        for t in range(qv.shape[1]):
            h = torch.bmm(R0p[qv[:, t] % self.m], h.unsqueeze(-1)).squeeze(-1)
            h = torch.bmm(R1p[rv[:, t] % self.m], h.unsqueeze(-1)).squeeze(-1)
        return h

    def decode(self, h):
        return (h @ self.location_codes().t()).argmax(-1)


def train_grid_ssm_hex(m=7, dim=64, steps=1500, lr=3e-3, seed=0, w_sig=5.0, w_iso=2.0):
    model = GridSSMHex(dim, m, seed)
    sig = SIGReg(n_slices=256)
    g = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        codes = model.location_codes()
        R0, R1, R2 = model.R0(), model.R1(), model.R2()
        h0a, h1a = model.h0, model.h0
        for _ in range(m): h0a = R0 @ h0a
        for _ in range(m): h1a = R1 @ h1a
        l_close = (h0a - model.h0).pow(2).sum() + (h1a - model.h0).pow(2).sum()
        mags = torch.stack([((R - torch.eye(dim)) @ model.h0).norm() for R in (R0, R1, R2)])
        l_iso = mags.var()                                          # conformal isometry (equal steps)
        loss = l_close + w_iso * l_iso + w_sig * sig(codes, generator=g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def navigation_acc_hex(model, n=2000, max_steps=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    qv = torch.randint(0, model.m, (n, max_steps), generator=g)
    rv = torch.randint(0, model.m, (n, max_steps), generator=g)
    tq, tr = qv.sum(1) % model.m, rv.sum(1) % model.m
    pred = model.decode(model.integrate(qv, rv))
    return ((pred // model.m == tq) & (pred % model.m == tr)).float().mean().item()


if __name__ == "__main__":
    model = train_grid_ssm(m=17, steps=800)
    print(f"GridSSM (m=17): zero-shot path-integration navigation = "
          f"{navigation_acc(model):.3f}  (chance {1/17:.3f})")
    m2 = train_grid_ssm_2d(5, 5, steps=1200)
    print(f"GridSSM2D (5x5): zero-shot 2-D navigation = {navigation_acc_2d(m2):.3f}  (chance {1/25:.3f})")
