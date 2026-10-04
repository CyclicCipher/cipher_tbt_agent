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
Added for the geometry test (§14–§15):
  "grid2"            the PRODUCT of two 4 × 4 grids (N = 256, k = 8: four moves in each part) -- a compositional world;
                     `goal_mode` "one" (the target differs from the start in one part) | "both" | "any"; code "coord"
                     = (x_a, y_a, x_b, y_b)
  "perm"             the permutation group S5 (N = 120, k = 4: rotate left, rotate right, swap 0-1, swap 1-2) -- a
                     Cayley graph of a NON-commuting group, the negative control: no code makes its actions translations
  code "learned"     not built here: injected per world (`gcml.learn_code_als`) from search experience
Added for the tier test (§17):
  "heading"          an S × S grid × 4 headings (N = 4·S², k = 6): forward, back, step left, step right -- moves in the
                     frame of the heading (off the grid = stay) -- turn left, turn right. Movement with heading: the
                     actions do not commute. `goal_mode` "pos" (same heading, position differs) | "any"; code "allo"
                     = (x, y, heading unit vector) -- the grid-cell plus head-direction code; `factor` = the heading
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
    adj = dest.tolist()
    rows = []
    for s in range(N):
        dist = [math.inf] * N
        dist[s] = 0
        q = deque([s])
        while q:
            i = q.popleft()
            for j in adj[i]:
                if dist[j] == math.inf:
                    dist[j] = dist[i] + 1
                    q.append(j)
        rows.append(dist)
    return torch.tensor(rows)


_WORLDS = {}                                                  # world structure, cached: problems in a world share it


class StructuredWalk(GraphWalk):
    def __init__(self, cfg: TaskConfig, world_seed: int, problem_seed: int, graph="random", keys="node", kappa=0.0,
                 sr_dim=16, sr_gamma=0.9, goal_mode="any", extra_codes=None):
        self.cfg = c = cfg
        self.graph_kind, self.keys_kind, self.kappa = graph, keys, kappa
        key = (c.N, c.k, c.n, c.d, c.hack_dim, world_seed, graph, keys, kappa, sr_dim, sr_gamma)
        if key not in _WORLDS:
            _WORLDS[key] = self._build_world(c, world_seed, graph, keys, kappa, sr_dim, sr_gamma)
        for attr, val in _WORLDS[key].items():
            setattr(self, attr, val)
        self.codes = dict(self.codes)
        for name, E in (extra_codes or {}).items():
            self.codes[name] = E
        # -- the problem ------------------------------------------------------------------------------------------------
        gp = torch.Generator().manual_seed(problem_seed)
        for _ in range(1000):
            start = int(torch.randint(c.N, (1,), generator=gp))
            at_L = (self.dist_all[start] == c.L).nonzero().flatten()
            if graph == "heading" and goal_mode == "pos":
                at_L = at_L[self.factor[at_L] == self.factor[start]]
            if graph == "grid2" and goal_mode != "any":
                a_s, b_s = start // 16, start % 16
                a_t, b_t = at_L // 16, at_L % 16
                diff = (a_t != a_s).long() + (b_t != b_s).long()
                at_L = at_L[diff == (1 if goal_mode == "one" else 2)]
            if at_L.numel():
                target = int(at_L[int(torch.randint(at_L.numel(), (1,), generator=gp))])
                break
        else:
            raise RuntimeError("no target at distance L")
        self.start, self.target = start, target
        self.set_target(target, gp)
        self.p0 = torch.zeros(c.N + self.H)
        self.p0[start] = 1.0

    def _build_world(self, c, world_seed, graph, keys, kappa, sr_dim, sr_gamma):
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
        elif graph == "grid2":
            S = 4
            if c.N != S ** 4 or c.k != 8:
                raise ValueError("grid2 needs N = 256 and k = 8")
            xy = torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij"), -1).reshape(-1, 2)  # 16 cells
            mv = torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]])
            part = xy[:, None, :] + mv[None]                                           # (16, 4, 2)
            ok = ((part >= 0) & (part < S)).all(-1)
            step1 = torch.where(ok, part[..., 0] * S + part[..., 1], torch.arange(S * S)[:, None].expand(-1, 4))
            a = torch.arange(c.N) // (S * S)
            b = torch.arange(c.N) % (S * S)
            dest = torch.cat([step1[a] * (S * S) + b[:, None], a[:, None] * (S * S) + step1[b]], 1)
            self.xy = torch.cat([xy[a], xy[b]], 1).float()                                # (N, 4)
        elif graph == "heading":
            S = int(round(math.sqrt(c.N // 4)))
            if 4 * S * S != c.N or c.k != 6:
                raise ValueError("heading needs N = 4·S² and k = 6")
            node = torch.arange(c.N)
            cell, h = node // 4, node % 4                                   # node = (x·S + y)·4 + heading
            pos = torch.stack([cell // S, cell % S], -1)
            dirs = torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]])          # heading 0 N, 1 E, 2 S, 3 W; right = +1
            cols = []
            for turn in (0, 2, 3, 1):                                        # forward, back, step left, step right
                nxt = pos + dirs[(h + turn) % 4]
                inside = ((nxt >= 0) & (nxt < S)).all(-1)
                cols.append(torch.where(inside, (nxt[:, 0] * S + nxt[:, 1]) * 4 + h, node))
            cols += [cell * 4 + (h + 3) % 4, cell * 4 + (h + 1) % 4]         # turn left, turn right
            dest = torch.stack(cols, 1)
            self.xy, self.factor = pos.float(), h
            heading_vec = dirs[h].float()
        elif graph == "perm":
            import itertools
            perms = list(itertools.permutations(range(5)))
            if c.N != len(perms) or c.k != 4:
                raise ValueError("perm needs N = 120 and k = 4")
            index = {q: i for i, q in enumerate(perms)}
            gens = [lambda q: q[1:] + q[:1], lambda q: q[-1:] + q[:-1],
                    lambda q: (q[1], q[0]) + q[2:], lambda q: (q[0], q[2], q[1]) + q[3:]]
            dest = torch.tensor([[index[gf(q)] for gf in gens] for q in perms])
            self.xy = None
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
        if graph == "heading":
            self.codes["allo"] = torch.cat([self.xy / (S - 1), heading_vec], 1)
        elif self.xy is not None:
            self.codes["coord"] = self.xy / (S - 1)
        if graph != "heading":
            self.factor = None
        return dict(xy=self.xy, keys=self.keys, P=self.P, Q=self.Q, H=self.H, hack_dir=self.hack_dir, dest=self.dest,
                    dest_flat=self.dest_flat, dist_all=self.dist_all, codes=self.codes, factor=self.factor)

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
