"""Tier 2 in thought space: OPERATOR codes for continuous thoughts (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §18).

In the mock (global keys) a thought's effect is a softmax mixture of the edges it points at, with weights that do not
depend on the node, so on any exact matrix code E a thought acts as a GATED MIXTURE OF OPERATORS:
    c′ = Σ_k g_k(z)·(M_k·c + b_k),   g(z) = softmax(G·unit(z) + a),   c = pᵀE (the code of a distribution over nodes)
`OperatorModel` learns E (whitened; or takes it fixed), the gate (G, a) and K operators (M_k, b_k) from (state,
thought, next state) transitions -- no action labels. Its gate's rows give PROTOTYPE thoughts, one per component.

  path_integration   walks of KEY thoughts (each selects one edge; built from the hidden keys, for evaluation only):
                     decode the node after each predicted step
  prototype_report   how cleanly each prototype moves the real walker along one edge
  op_look_own / op_look_sr / sim_look_sr   the planners of §18.2 (registered in ALGOS); they read `task.opm` (the
                     model) and `task.metric` (a node code used as the distance, here the SR eigenmap)
"""
from __future__ import annotations

import math

import torch

from mock_task import Sim, _unit
from searchers import ALGOS
from matrix_code import whiten


class OperatorModel:
    def __init__(self, N, d, m, K=8):
        self.N, self.d, self.m, self.K = N, d, m, K

    def fit(self, X, Z, X2, E_fixed=None, steps=3000, lr=0.01, restarts=2, seed=0):
        N, d, m, K = self.N, self.d, self.m, self.K
        p, p2, zu = X[:, :N], X2[:, :N], _unit(Z)
        best = None
        for r in range(restarts):
            g = torch.Generator().manual_seed(1000 * seed + r)
            Eraw = (E_fixed.clone() if E_fixed is not None else torch.randn(N, m, generator=g)).requires_grad_(E_fixed is None)
            G = (0.1 * torch.randn(K, d, generator=g)).requires_grad_(True)
            a = torch.zeros(K, requires_grad=True)
            A = (0.01 * torch.randn(K, m, m, generator=g)).requires_grad_(True)
            b = torch.zeros(K, m, requires_grad=True)
            opt = torch.optim.Adam([G, a, A, b] + ([Eraw] if E_fixed is None else []), lr=lr)
            for _ in range(steps):
                E = Eraw if E_fixed is not None else whiten(Eraw)
                loss = self._loss(E, G, a, torch.eye(m) + A, b, p, zu, p2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                E = Eraw.detach() if E_fixed is not None else whiten(Eraw.detach())
                final = float(self._loss(E, G, a, torch.eye(m) + A, b, p, zu, p2))
            if best is None or final < best[0]:
                best = (final, E, G.detach(), a.detach(), (torch.eye(m) + A).detach(), b.detach())
        self.train_unexplained, self.E, self.G, self.a, self.M, self.b = best
        self.protos, self.proto_gate = self._prototypes()
        return self

    @staticmethod
    def _loss(E, G, a, M, b, p, zu, p2):
        c, c2 = p @ E, p2 @ E
        w = torch.softmax(zu @ G.T + a, -1)                                           # (n, K)
        pred = (w[..., None] * (torch.einsum("kij,nj->nki", M, c) + b)).sum(1)
        return ((pred - c2) ** 2).sum() / ((c2 - c) ** 2).sum().clamp_min(1e-9)

    # -- use ----------------------------------------------------------------------------------------------------------------
    def code(self, X):
        return X[:, :self.N] @ self.E

    def gate(self, z):
        return torch.softmax(_unit(z) @ self.G.T + self.a, -1)

    def predict_all(self, c):
        """Each component's prediction: c (n, m) -> (n, K, m)."""
        return torch.einsum("kij,nj->nki", self.M, c) + self.b

    def predict(self, c, z):
        return (self.gate(z)[..., None] * self.predict_all(c)).sum(1)

    def unexplained(self, X, Z, X2):
        with torch.no_grad():
            return float(self._loss(self.E, self.G, self.a, self.M, self.b, X[:, :self.N], _unit(Z), X2[:, :self.N]))

    def _prototypes(self, steps=300):
        """The thought that most selects each component (gradient ascent on log g_k over unit thoughts)."""
        protos, gates = [], []
        for k in range(self.K):
            z = (self.G[k] - self.G.mean(0)).clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=0.05)
            for _ in range(steps):
                loss = -torch.log_softmax(_unit(z) @ self.G.T + self.a, -1)[k]
                opt.zero_grad()
                loss.backward()
                opt.step()
            protos.append(_unit(z.detach()))
            gates.append(float(self.gate(z.detach()[None])[0, k]))
        return torch.stack(protos), gates


class AdditiveModel:
    """GCML's additive form for comparison: c′ = c + V·unit(z), V by ridge on a given code."""

    def __init__(self, E, lam=1e-2):
        self.E, self.lam = E, lam

    def fit(self, X, Z, X2):
        N = self.E.shape[0]
        dc = (X2[:, :N] - X[:, :N]) @ self.E
        zu = _unit(Z)
        A = zu.T @ zu
        self.V = torch.linalg.solve(A + self.lam * A.trace() / A.shape[0] * torch.eye(A.shape[0]), zu.T @ dc)  # (d, m)
        return self

    def predict(self, c, z):
        return c + _unit(z) @ self.V


# ── measurements ────────────────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def path_integration(task, E, predict, gen, n=300, K=8):
    """Walks of key thoughts from random nodes; decode the nearest node's code after each predicted step."""
    N = task.cfg.N
    key_thoughts = _unit(task.keys[0] @ task.P)                                      # global keys: the same at every node
    nodes = torch.randint(N, (n,), generator=gen)
    acts = torch.randint(key_thoughts.shape[0], (n, K), generator=gen)
    X = task.onehot(nodes)
    c, clean = X[:, :N] @ E, torch.ones(n, dtype=torch.bool)
    acc, acc_clean = [], []
    for s in range(K):
        z = key_thoughts[acts[:, s]]
        X2 = task.step(X, z)
        prev, now = X[:, :N].argmax(1), X2[:, :N].argmax(1)
        clean &= now != prev
        c = predict(c, z)
        hit = torch.cdist(c, E).argmin(1) == now
        acc.append(float(hit.float().mean()))
        acc_clean.append(float(hit[clean].float().mean()) if bool(clean.any()) else float("nan"))
        X = X2
    return acc, acc_clean


@torch.no_grad()
def prototype_report(task, om):
    """Per prototype, from every node: the share of mass that moved, and the share on the single main destination."""
    N = task.cfg.N
    X = task.onehot(torch.arange(N))
    out = []
    for z in om.protos:
        p2 = task.step(X, z.expand(N, -1))[:, :N]
        stay = p2[torch.arange(N), torch.arange(N)]
        moved = 1 - stay
        top = p2.clone()
        top[torch.arange(N), torch.arange(N)] = 0
        out.append(dict(moved=float(moved.mean()), main=float(top.max(1).values.mean())))
    return out


# ── planners ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _choose(dist, eps, gen):
    k = dist.argmin(1)
    explore = torch.rand(k.shape[0], generator=gen) < eps
    k[explore] = torch.randint(dist.shape[1], (int(explore.sum()),), generator=gen)
    return k


def _op_look(sim: Sim, gen, metric, eps=0.1, sigma=0.1, chains=32):
    t, T = sim.task, sim.task.cfg.T
    om, N = t.opm, t.cfg.N
    while sim.budget.left >= T:
        C = max(1, min(chains, sim.budget.left // T))
        X = t.p0.expand(C, -1)
        for _ in range(T):
            with torch.no_grad():
                P = om.predict_all(om.code(X))                                          # (C, K, m)
                if metric == "own":
                    dist = (P - om.E[t.target]).norm(dim=-1)
                else:
                    near = torch.cdist(P.reshape(-1, om.m), om.E).argmin(1).reshape(C, om.K)
                    dist = (t.metric[near] - t.metric[t.target]).norm(dim=-1)
            k = _choose(dist, eps, gen)
            z = _unit(om.protos[k] + sigma * torch.randn(C, t.cfg.d, generator=gen) / math.sqrt(t.cfg.d))
            X = sim.step(X, z)


def sim_look_sr(sim: Sim, gen, eps=0.1, chains=32):
    """Lookahead with the TRUE dynamics: simulate every prototype (K thought-steps per step), pick by SR distance."""
    t, T = sim.task, sim.task.cfg.T
    om, N = t.opm, t.cfg.N
    while sim.budget.left >= T * om.K:
        C = max(1, min(chains, sim.budget.left // (T * om.K)))
        X = t.p0.expand(C, -1)
        for _ in range(T):
            kids = sim.step(X.repeat_interleave(om.K, 0), om.protos.repeat(C, 1))      # (C·K, ·)
            dist = ((kids[:, :N] @ t.metric) - t.metric[t.target]).norm(dim=-1).reshape(C, om.K)
            k = _choose(dist, eps, gen)
            X = kids.reshape(C, om.K, -1)[torch.arange(C), k]


ALGOS["op_look_own"] = lambda sim, gen: _op_look(sim, gen, "own")
ALGOS["op_look_sr"] = lambda sim, gen: _op_look(sim, gen, "sr")
ALGOS["sim_look_sr"] = sim_look_sr
