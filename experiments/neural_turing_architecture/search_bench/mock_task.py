"""The mock task for comparing search algorithms over continuous thoughts (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §11).

A THOUGHT-SPACE GRAPH WALK, built to have the properties of the real problem and nothing else:

  state      a probability distribution p over the N nodes of a random directed graph (k out-edges per node) -- a
             continuous state that can hold several branches at once, as a Coconut thought can.
  thought    z ∈ R^d, normalised to unit length (real thoughts are normalised). It acts only through u = P z ∈ R^n,
             P a hidden n × d matrix with orthonormal rows: an n-dimensional subspace of the thought matters, and no
             search is told which (the low effective dimension the planning doc's §10 is about).
  dynamics   at node i, out-edge e has a unit key c_{i,e} ∈ R^n; the walker takes edge e with probability
             softmax_e(beta · c_{i,e}·u) against a STAY option of logit beta · gate, so a thought has to point at an
             edge's key (cosine above `gate`) to move at all. Exact and differentiable:
                 p' = Σ_i p_i [ Σ_e w_{i,e} onehot(dest(i,e)) + w_{i,stay} onehot(i) ].
             A random thought in high d has ‖u‖ ≈ √(n/d) and mostly does nothing -- the needle regime.
  verifier   success iff argmax p == target and p[target] > 0.5. The stop value is this verdict, exact, at every
             state, as in the real problem.
  difficulty the target is at graph distance L from the start, the horizon is T ≥ L: a solution is a chain of at least
             L thoughts, each choosing the right edge. Reward is sparse and terminal.
  value      a LEARNED value is mocked as V(p) = p · ṽ, ṽ_j = gamma^dist(j, target) + noise (fixed per instance;
             `value_noise` is its quality). Differentiable in z through the soft dynamics.
  value hack `value_hack = λ > 0` makes the mocked value EXPLOITABLE, the real risk of following a learned value's
             gradient: the state also carries a nuisance vector h ∈ R^8 that thoughts push along 8 directions
             orthogonal to P (h' = h + Q z), the verifier ignores h, and the learned value has a spurious term
             λ·tanh(a·h). Directions that fool the value cost the thought its effective norm (‖P z‖² + ‖Q z‖² ≤ 1).
             The state tensor is X = [p, h] (N + 8 columns).
  prior      a policy prior. `prior_cos = 0`: isotropic (an untrained policy). `prior_cos = c > 0`: at each node a mean
             direction with cosine c to the right thought, plus a fixed per-node error (a partly trained policy);
             samples are normalise(prior_conc · mean + unit noise).

`Budget` counts THOUGHT-STEPS: one candidate advanced one step costs 1; a gradient through a step costs 3 (forward +
backward). Every algorithm gets the same budget.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import torch


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, total: int):
        self.total, self.used = int(total), 0

    @property
    def left(self) -> int:
        return self.total - self.used

    def spend(self, n: int):
        if self.used + n > self.total:
            raise BudgetExhausted
        self.used += n


class Solved(Exception):
    """Raised the moment any simulated state from the true state satisfies the verifier: the dynamics are exact and
    deterministic, so a simulated solution can be replayed."""


@dataclass
class TaskConfig:
    N: int = 64          # nodes
    k: int = 4           # out-edges per node
    n: int = 8           # effective dimension of the thought
    d: int = 64          # thought dimension
    beta: float = 8.0    # sharpness of edge selection
    gate: float = 0.5    # cosine a thought needs (against the key) to beat STAY
    L: int = 4           # graph distance start -> target
    T: int = 8           # horizon (thoughts)
    gamma: float = 0.7   # value discount per graph step
    value_noise: float = 0.1
    value_hack: float = 0.0
    hack_dim: int = 8
    prior_cos: float = 0.0
    prior_conc: float = 1.0


def _unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


class GraphWalk:
    def __init__(self, cfg: TaskConfig, seed: int):
        self.cfg = c = cfg
        g = torch.Generator().manual_seed(seed)
        for _attempt in range(200):
            dest = torch.stack([torch.randperm(c.N - 1, generator=g)[:c.k] for _ in range(c.N)])
            dest = dest + (dest >= torch.arange(c.N)[:, None]).long()                     # no self-loops
            start = int(torch.randint(c.N, (1,), generator=g))
            dist_from = self._bfs(dest, start)
            at_L = [j for j, dj in enumerate(dist_from) if dj == c.L]
            if at_L:
                target = at_L[int(torch.randint(len(at_L), (1,), generator=g))]
                break
        else:
            raise RuntimeError("no target at distance L")
        self.dest, self.start, self.target = dest, start, target
        self.keys = _unit(torch.randn(c.N, c.k, c.n, generator=g))                         # (N, k, n)
        H = c.hack_dim if c.d >= c.n + c.hack_dim else 0
        basis = torch.linalg.qr(torch.randn(c.d, c.n + H, generator=g))[0].T.contiguous()
        self.P, self.Q = basis[:c.n], basis[c.n:]                                          # (n, d), (H, d), mutually ⟂
        self.H = H
        self.hack_dir = _unit(torch.randn(max(H, 1), generator=g))
        # distances TO the target (reverse BFS) -> the true value, and the optimal edge per node
        rev = [[] for _ in range(c.N)]
        for i in range(c.N):
            for e in range(c.k):
                rev[int(dest[i, e])].append(i)
        dist_to = [math.inf] * c.N
        dist_to[target] = 0
        q = deque([target])
        while q:
            j = q.popleft()
            for i in rev[j]:
                if dist_to[i] == math.inf:
                    dist_to[i] = dist_to[j] + 1
                    q.append(i)
        self.dist_to = torch.tensor(dist_to)
        v = torch.where(torch.isfinite(self.dist_to), c.gamma ** self.dist_to.clamp(max=1e6), torch.zeros(c.N))
        self.v_true = v
        self.v = v + c.value_noise * torch.randn(c.N, generator=g)                          # the mocked learned value
        best_e = torch.zeros(c.N, dtype=torch.long)
        for i in range(c.N):
            dd = self.dist_to[dest[i]]
            best_e[i] = int(torch.argmin(dd))
        self.best_edge = best_e
        z_star = self.keys[torch.arange(c.N), best_e] @ self.P                             # (N, d): the right thought
        err = torch.randn(c.N, c.d, generator=g)
        err = _unit(err - (err * z_star).sum(-1, keepdim=True) * z_star)                   # ⟂ to z*
        self.prior_mean = _unit(c.prior_cos * z_star + math.sqrt(max(0.0, 1 - c.prior_cos ** 2)) * err)
        self.p0 = torch.zeros(c.N + H)                                                    # X0 = [p0, h0 = 0]
        self.p0[start] = 1.0
        self.dest_flat = dest.reshape(1, -1)

    @staticmethod
    def _bfs(dest, s):
        dist = [math.inf] * dest.shape[0]
        dist[s] = 0
        q = deque([s])
        while q:
            i = q.popleft()
            for j in dest[i].tolist():
                if dist[j] == math.inf:
                    dist[j] = dist[i] + 1
                    q.append(j)
        return dist

    # -- the exact dynamics ---------------------------------------------------------------------------------------------
    def step(self, X, z):
        """X = [p, h] (C, N + H), z (C, d) -> X'. Differentiable in z (and X)."""
        c = self.cfg
        p, h = X[:, :c.N], X[:, c.N:]
        z = _unit(z)
        u = z @ self.P.T                                                                  # (C, n)
        p2 = self.move(p, u)
        if self.H:
            return torch.cat([p2, h + z @ self.Q.T], -1)
        return p2

    def move(self, p, u):
        """The node distribution after the effective control u = P·unit(z): p (C, N), u (C, n) -> p' (C, N)."""
        c = self.cfg
        s = torch.einsum("ikn,cn->cik", self.keys, u)                                     # (C, N, k)
        logits = torch.cat([c.beta * s, torch.full_like(s[..., :1], c.beta * c.gate)], -1)
        w = logits.softmax(-1)                                                            # (C, N, k+1)
        mass = p.unsqueeze(-1) * w
        out = torch.zeros_like(p).scatter_add(1, self.dest_flat.expand(p.shape[0], -1), mass[..., :c.k].reshape(p.shape[0], -1))
        return out + mass[..., c.k]

    def success(self, X):
        p = X[:, :self.cfg.N]
        return (p.argmax(-1) == self.target) & (p[:, self.target] > 0.5)

    def value(self, X):
        v = X[:, :self.cfg.N] @ self.v
        if self.H and self.cfg.value_hack > 0:
            v = v + self.cfg.value_hack * torch.tanh(X[:, self.cfg.N:] @ self.hack_dir)
        return v

    # -- the policy prior -------------------------------------------------------------------------------------------------
    def prior_sample(self, p, m: int, gen: torch.Generator, widen: float = 1.0):
        """m samples per row of p (C, N) -> (C*m, d), row-major. widen > 1 flattens the prior (EZ-V2's AS2)."""
        c = self.cfg
        C = p.shape[0]
        noise = _unit(torch.randn(C * m, c.d, generator=gen))
        if c.prior_cos <= 0 or c.prior_conc <= 0:
            return noise
        mean = self.prior_mean[p[:, :c.N].argmax(-1)].repeat_interleave(m, 0)
        return _unit((c.prior_conc / widen) * mean + noise)

    def prior_mean_of(self, p):
        if self.cfg.prior_cos <= 0:
            return None
        return self.prior_mean[p[:, :self.cfg.N].argmax(-1)]


class Sim:
    """The task plus a budget: every simulated step is paid for, and every simulated state is checked against the
    verifier (raising `Solved`)."""

    def __init__(self, task: GraphWalk, budget: Budget):
        self.task, self.budget = task, budget

    def step(self, p, z, check=True):
        self.budget.spend(p.shape[0])
        with torch.no_grad():
            q = self.task.step(p, z)
        if check and bool(self.task.success(q).any()):
            raise Solved
        return q

    def value_grad(self, p, z):
        """∇_z V(step(p, z)) for each row; costs 3 per row. Also checks the stepped states."""
        self.budget.spend(3 * p.shape[0])
        z = z.detach().clone().requires_grad_(True)
        q = self.task.step(p.detach(), z)
        if bool(self.task.success(q.detach()).any()):
            raise Solved
        (g,) = torch.autograd.grad(self.task.value(q).sum(), z)
        return g, q.detach()
