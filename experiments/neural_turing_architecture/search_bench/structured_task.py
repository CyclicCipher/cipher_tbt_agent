"""The structure dial for testing GCML's inverse model (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §12.2).

`StructuredWalk` is the mock of `mock_task.GraphWalk` with two changes:
  WORLD vs PROBLEM   the world (graph, edge keys, the hidden projection P) is fixed by `world_seed`; a problem (start,
                     target, the noisy value table) by `problem_seed`. An inverse model is learned once per world, from
                     experience on some problems, and used on others -- as a real network's dynamics are fixed while
                     problems vary.
  STRUCTURE          graph "random" (k random out-edges per node) or "grid" (√N × √N, N/E/S/W, an off-grid move stays);
                     keys "node" (every node's edges have their own keys: a thought means different things at different
                     nodes) or "global" (k keys shared by all nodes: the same thought does the same operation everywhere),
                     optionally perturbed per node by `kappa`.
  STATE CODES        the representations an inverse model maps differences of: "raw" (the node distribution itself),
                     "sr" (an eigenmap of the symmetrised successor representation of the random walk -- the old TBT
                     column's frame, in which s* − s points along paths), "coord" (grid only: the node's (x, y)).
The variants of §12.2: V0 = random/node/raw, V1 = random/global (raw or sr), V2 = grid/global (coord or sr).
"""
from __future__ import annotations

import math
from collections import deque

import torch

from mock_task import GraphWalk, TaskConfig, _unit

VARIANTS = {
    "V0":      dict(graph="random", keys="node",   code="raw"),
    "V1raw":   dict(graph="random", keys="global", code="raw"),
    "V1sr":    dict(graph="random", keys="global", code="sr"),
    "V2coord": dict(graph="grid",   keys="global", code="coord"),
    "V2sr":    dict(graph="grid",   keys="global", code="sr"),
}


def all_pairs(dest):
    N = dest.shape[0]
    D = torch.full((N, N), math.inf)
    for s in range(N):
        D[s, s] = 0
        q = deque([s])
        while q:
            i = q.popleft()
            for j in dest[i].tolist():
                if D[s, j] == math.inf:
                    D[s, j] = D[s, i] + 1
                    q.append(j)
    return D


class StructuredWalk(GraphWalk):
    def __init__(self, cfg: TaskConfig, world_seed: int, problem_seed: int, graph="random", keys="node", kappa=0.0,
                 sr_dim=16, sr_gamma=0.9):
        self.cfg = c = cfg
        self.graph_kind, self.keys_kind, self.kappa = graph, keys, kappa
        g = torch.Generator().manual_seed(world_seed)
        # -- the world --------------------------------------------------------------------------------------------------
        if graph == "grid":
            S = int(round(math.sqrt(c.N)))
            if S * S != c.N or c.k != 4:
                raise ValueError("grid needs N a square and k = 4")
            xy = torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij"), -1).reshape(-1, 2)
            nxt = xy[:, None, :] + torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]])[None]
            inside = ((nxt >= 0) & (nxt < S)).all(-1)
            dest = torch.where(inside, nxt[..., 0] * S + nxt[..., 1], torch.arange(c.N)[:, None].expand(-1, 4))
            self.xy = xy.float()
        else:
            dest = torch.stack([torch.randperm(c.N - 1, generator=g)[:c.k] for _ in range(c.N)])
            dest = dest + (dest >= torch.arange(c.N)[:, None]).long()
            self.xy = None
        if keys == "global":
            base = _unit(torch.randn(c.k, c.n, generator=g))
            self.keys = _unit(base[None].expand(c.N, -1, -1) + kappa * torch.randn(c.N, c.k, c.n, generator=g))
        else:
            self.keys = _unit(torch.randn(c.N, c.k, c.n, generator=g))
        H = c.hack_dim if c.d >= c.n + c.hack_dim else 0
        basis = torch.linalg.qr(torch.randn(c.d, c.n + H, generator=g))[0].T.contiguous()
        self.P, self.Q, self.H = basis[:c.n], basis[c.n:], H
        self.hack_dir = _unit(torch.randn(max(H, 1), generator=g))
        self.dest, self.dest_flat = dest, dest.reshape(1, -1)
        self.dist_all = all_pairs(dest)
        # state codes
        self.codes = {"raw": torch.eye(c.N)}
        T_rw = torch.zeros(c.N, c.N).index_put_((torch.arange(c.N).repeat_interleave(c.k), dest.reshape(-1)),
                                                 torch.full((c.N * c.k,), 1.0 / c.k), accumulate=True)
        M = torch.linalg.inv(torch.eye(c.N) - sr_gamma * T_rw)
        ev, U = torch.linalg.eigh((M + M.T) / 2)
        self.codes["sr"] = (U[:, -sr_dim:] * ev[-sr_dim:].clamp_min(0).sqrt()).contiguous()
        if self.xy is not None:
            self.codes["coord"] = self.xy / (S - 1)
        # -- the problem ------------------------------------------------------------------------------------------------
        gp = torch.Generator().manual_seed(problem_seed)
        for _ in range(1000):
            start = int(torch.randint(c.N, (1,), generator=gp))
            at_L = (self.dist_all[start] == c.L).nonzero().flatten()
            if at_L.numel():
                target = int(at_L[int(torch.randint(at_L.numel(), (1,), generator=gp))])
                break
        else:
            raise RuntimeError("no target at distance L")
        self.start, self.target = start, target
        self.set_target(target, gp)
        self.p0 = torch.zeros(c.N + H)
        self.p0[start] = 1.0

    def set_target(self, target, gp):
        c = self.cfg
        self.target = target
        self.dist_to = self.dist_all[:, target]
        self.v_true = torch.where(torch.isfinite(self.dist_to), c.gamma ** self.dist_to.clamp(max=1e6), torch.zeros(c.N))
        self.v = self.v_true + c.value_noise * torch.randn(c.N, generator=gp)
        self.best_edge = self.dist_to[self.dest].argmin(-1)                               # (N,)
        z_star = self.keys[torch.arange(c.N), self.best_edge] @ self.P
        err = torch.randn(c.N, c.d, generator=gp)
        err = _unit(err - (err * z_star).sum(-1, keepdim=True) * z_star)
        self.prior_mean = _unit(c.prior_cos * z_star + math.sqrt(max(0.0, 1 - c.prior_cos ** 2)) * err)

    # -- codes and the right thought -------------------------------------------------------------------------------------
    def code(self, X, name):
        return X[:, :self.cfg.N] @ self.codes[name]

    def right_thought(self, nodes, target):
        """The thought that takes each node one step closer to `target` (unit, in the effective subspace)."""
        tgt = torch.as_tensor(target)
        e = self.dist_all[self.dest[nodes], tgt[:, None] if tgt.dim() else tgt].argmin(-1)
        return self.keys[nodes, e] @ self.P

    def onehot(self, nodes):
        X = torch.zeros(len(nodes), self.cfg.N + self.H)
        X[torch.arange(len(nodes)), nodes] = 1.0
        return X
