"""GCML's inverse model, in thought space (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §12; GCML: Lin et al., Nature MI 2026,
notes in src/tbt/notes/gcml_neural_sampling_cognitive_maps.md).

An inverse model maps a wanted change of the state CODE, Δc, to the thought that produces it: z = W·Δc. GCML learns W
by a local Hebbian rule from observed transitions, ΔW = η·z·(c' − c)ᵀ (its eq 14); we also fit it by ridge regression,
and as a state-conditioned network W(c, Δc) (GCML generalised past state-invariant action effects).

  transitions_*        experience to learn from: random thoughts at random nodes; demonstrations (the right thought,
                       noised); or the transitions a search simulated while solving OTHER problems in the same world
  InverseModel         kind "hebb" | "ridge" | "mlp", over a named state code
  goal_delta           Δc = c(target) − c(state)                      (a known goal: GCML's setting)
  value_delta          Δc = ∇_c V(c), V a kernel regression of the value table over the node codes (no goal needed:
                       the analogue of a value head's gradient in state space -- no backprop through the dynamics)
  jacobian_inverse     z ∝ J⁺·Δc with J = ∂c(step(x, z))/∂z at a reference thought: the EXACT local inverse (reference)
  invariance           G0: 1 − the fraction of a thought's effect (in a code) that depends on the state
  direction_quality    G1/G2: from every node of a problem, is one step along the proposal PROGRESS (more than half the
                       mass moved one step closer to the target)? and its cosine with the right thought
"""
from __future__ import annotations

import torch

from mock_task import Budget, Sim, Solved, BudgetExhausted, _unit


# ── experience ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def transitions_random(task, n, gen):
    nodes = torch.randint(task.cfg.N, (n,), generator=gen)
    X = task.onehot(nodes)
    z = _unit(torch.randn(n, task.cfg.d, generator=gen))
    with torch.no_grad():
        return X, z, task.step(X, z)


def transitions_demo(task, n, gen, noise=0.3):
    N = task.cfg.N
    i = torch.randint(N, (4 * n,), generator=gen)
    t = torch.randint(N, (4 * n,), generator=gen)
    d = task.dist_all[i, t]
    ok = torch.isfinite(d) & (d >= 1)
    i, t = i[ok][:n], t[ok][:n]
    z = _unit(task.right_thought(i, t) + noise * _unit(torch.randn(len(i), task.cfg.d, generator=gen)))
    X = task.onehot(i)
    with torch.no_grad():
        return X, z, task.step(X, z)


class RecordingSim(Sim):
    """A Sim that keeps every transition it simulates (stepped and gradient base points alike)."""

    def __init__(self, task, budget):
        super().__init__(task, budget)
        self.log = []

    def step(self, p, z, check=True):
        self.budget.spend(p.shape[0])
        with torch.no_grad():
            q = self.task.step(p, z)
        self.log.append((p.detach(), _unit(z).detach(), q))
        if check and bool(self.task.success(q).any()):
            raise Solved
        return q

    def value_grad(self, p, z):
        g, q = super().value_grad(p, z)
        self.log.append((p.detach(), _unit(z).detach(), q))
        return g, q


def transitions_search(make_task, problem_seeds, algo, budget, gen):
    X, Z, X2 = [], [], []
    for ps in problem_seeds:
        sim = RecordingSim(make_task(ps), Budget(budget))
        try:
            algo(sim, gen)
        except (Solved, BudgetExhausted):
            pass
        for a, b, c in sim.log:
            X.append(a), Z.append(b), X2.append(c)
    return torch.cat(X), torch.cat(Z), torch.cat(X2)


# ── the inverse model ───────────────────────────────────────────────────────────────────────────────────────────────────
class InverseModel:
    def __init__(self, kind, code, lam=1e-2, steps=1500, seed=0):
        self.kind, self.code, self.lam, self.steps, self.seed = kind, code, lam, steps, seed

    def fit(self, task, X, z, X2):
        c, c2 = task.code(X, self.code), task.code(X2, self.code)
        dc = c2 - c
        scale = task.codes[self.code].norm(dim=1).mean()
        keep = dc.norm(dim=1) > 1e-3 * scale                         # a transition that changed nothing teaches nothing
        self.n_used, self.n_seen = int(keep.sum()), int(len(keep))
        c, dc, z = c[keep], dc[keep], z[keep]
        m, d = dc.shape[1], z.shape[1]
        if self.n_used < 2:
            self.W, self.net = torch.zeros(d, m), None
            return self
        if self.kind == "hebb":
            self.W = z.T @ dc / self.n_used
        elif self.kind == "ridge":
            A = dc.T @ dc
            self.W = z.T @ dc @ torch.linalg.inv(A + self.lam * (A.trace() / m) * torch.eye(m))
        elif self.kind == "mlp":
            torch.manual_seed(self.seed)
            self.net = torch.nn.Sequential(torch.nn.Linear(2 * m, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                                           torch.nn.ReLU(), torch.nn.Linear(256, d))
            self.cs, self.ds = c.std() + 1e-8, dc.std() + 1e-8
            opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
            inp = torch.cat([c / self.cs, dc / self.ds], 1)
            for _ in range(self.steps):
                idx = torch.randint(len(inp), (min(256, len(inp)),))
                out = self.net(inp[idx])
                loss = 1 - torch.nn.functional.cosine_similarity(out, z[idx], dim=1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def propose(self, task, X, dc):
        if self.kind == "mlp":
            if self.n_used < 2:
                return _unit(torch.randn(X.shape[0], task.cfg.d))
            return _unit(self.net(torch.cat([task.code(X, self.code) / self.cs, dc / self.ds], 1)))
        return _unit(dc @ self.W.T)


# ── where to go ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def goal_delta(task, X, code, target=None):
    E = task.codes[code]
    return E[task.target if target is None else target][None] - task.code(X, code)


def value_delta(task, X, code):
    """∇_c V(c), V(c) = Σ_j softmax_j(−‖c − E_j‖² / 2σ²)·ṽ_j: the value as a smooth function of the state code."""
    E = task.codes[code]
    if not hasattr(task, "_sigma2"):
        task._sigma2 = {}
    if code not in task._sigma2:
        D = torch.cdist(E, E)
        task._sigma2[code] = float((0.5 * D[D > 0].median()) ** 2)
    c = task.code(X, code).detach().requires_grad_(True)
    w = (-(torch.cdist(c, E) ** 2) / (2 * task._sigma2[code])).softmax(-1)
    (g,) = torch.autograd.grad((w @ task.v).sum(), c)
    return g


def jacobian_inverse(task, X, dc, code, z0, eps=1e-3):
    """z ∝ J⁺·Δc, J = ∂c(step(x, z))/∂z at z0, by central differences in the effective control u (exact up to eps)."""
    E, P = task.codes[code], task.P
    out = []
    for r in range(X.shape[0]):
        p = X[r:r + 1, :task.cfg.N]
        zh = _unit(z0[r:r + 1])
        u0 = zh @ P.T
        cols = []
        for a in range(task.cfg.n):
            du = torch.zeros_like(u0)
            du[0, a] = eps
            cols.append((task.move(p, u0 + du) - task.move(p, u0 - du)) / (2 * eps))
        Ju = torch.cat(cols, 0).T                                                 # (N, n)
        Jz = E.T @ Ju @ P @ (torch.eye(task.cfg.d) - zh.T @ zh)                    # (m, d)
        out.append(torch.linalg.pinv(Jz) @ dc[r])
    return _unit(torch.stack(out))


# ── measurements ────────────────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def invariance(task, code, n_thoughts=32, gen=None):
    """G0: for thoughts z in the effective subspace (so that they move), the effect Δc_i(z) at every node i; the share of
    its variance that depends on the node; returns 1 − that share, averaged over thoughts (1 = state-invariant)."""
    N = task.cfg.N
    a = torch.randn(n_thoughts, task.cfg.n, generator=gen)
    Z = _unit(a @ task.P)
    X = task.onehot(torch.arange(N))
    c = task.code(X, code)
    vals = []
    for z in Z:
        dc = task.code(task.step(X, z.expand(N, -1)), code) - c                 # (N, m)
        tot = (dc ** 2).sum(1).mean()
        if tot <= 1e-12:
            continue
        vals.append(float(1 - ((dc - dc.mean(0)) ** 2).sum(1).mean() / tot))
    return sum(vals) / max(1, len(vals))


@torch.no_grad()
def progress(task, X, z, target):
    """Did one step along z move more than half the mass one step closer to the target?"""
    p2 = task.step(X, z)[:, :task.cfg.N]
    d = task.dist_all[:, target]
    here = X[:, :task.cfg.N].argmax(-1)
    closer = (d[None, :] == (d[here] - 1)[:, None]).float()
    return ((p2 * closer).sum(1) > 0.5).float()


def nodes_toward(task, target):
    d = task.dist_all[:, target]
    return (torch.isfinite(d) & (d >= 1)).nonzero().flatten()
