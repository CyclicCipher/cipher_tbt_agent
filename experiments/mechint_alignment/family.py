"""The world model for the world family (E8), with per-world FAST parameters (hierarchical Bayes, option A).

Shared across worlds: the trunk (worldmodel.WorldModel: relative-position attention, locality, persistence prior).
Per world w, fast parameters phi_w ~ N(mu, 1/Lambda) with a LEARNED prior mean mu:
- the action embeddings [8, d]          (where a world's action permutation can live)
- the terrain-code embeddings [9, d]    (where its object-role permutation can live)
- the output heads (weights and biases) (where its numeric rules can live)
Nothing in the model names a rule: whatever a world's rules are, they are inferred as phi_w - mu from that world's
own data. forward_fast takes the fast parameters of a batch of worlds and, per row, the index of its world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldmodel import WorldModel


class FamilyModel(WorldModel):
    def __init__(self, spec, d=128, layers=4, heads=4, mlp=4, persist=5.0, radius=1):
        super().__init__(spec, d, layers, heads, mlp, pos="local", persist=persist, radius=radius)
        # the prior means of the fast parameters, initialised from the shared modules they replace
        mu = {"act": self.act.weight, "terr": self.cell[0].weight}
        for f, m in enumerate(self.cell_heads):
            mu[f"cw{f}"], mu[f"cb{f}"] = m.weight, m.bias
        for f, m in enumerate(self.glob_heads):
            mu[f"gw{f}"], mu[f"gb{f}"] = m.weight, m.bias
        self.mu = nn.ParameterDict({k: nn.Parameter(v.detach().clone()) for k, v in mu.items()})
        # the prior SPREAD of the fast parameters (log std, per element), learned by empirical Bayes when used
        self.log_sigma = nn.ParameterDict({k: nn.Parameter(torch.full_like(v, -2.3)) for k, v in mu.items()})

    def fast_init(self, n_worlds):
        """phi_w = mu for n_worlds fresh worlds (leaf tensors, requires_grad)."""
        return {k: v.detach().unsqueeze(0).expand(n_worlds, *v.shape).clone().requires_grad_() for k, v in self.mu.items()}

    def forward_fast(self, cells, glob, act, fast, wid):
        b, P = cells.shape[0], self.P
        grid = self.border.expand(b, P, P, -1).clone()
        grid[:, 1:-1, 1:-1] = cells.view(b, P - 2, P - 2, -1).long()
        grid = grid.view(b, P * P, -1)
        x = fast["terr"][wid[:, None], grid[..., 0]]                         # this world's code embeddings
        x = x + sum(e(grid[..., i]) for i, e in enumerate(self.cell) if i > 0)
        g = sum(e(glob[..., i].long()) for i, e in enumerate(self.glob)) + self.gtype
        x = torch.cat([g[:, None], x], 1) + fast["act"][wid, act.long()][:, None]   # this world's action embedding
        bias = self.bias()
        for block in self.blocks:
            x = block(x, bias)
        h = self.ln(x.float())
        hc, hg = h[:, self.interior], h[:, 0]
        cl = [torch.einsum("bnd,bvd->bnv", hc, fast[f"cw{f}"][wid]) + fast[f"cb{f}"][wid][:, None]
              for f in range(len(self.cell_heads))]
        gl = [torch.einsum("bd,bvd->bv", hg, fast[f"gw{f}"][wid]) + fast[f"gb{f}"][wid] for f in range(len(self.glob_heads))]
        nc = len(cl)
        cl = [x_ + self.persist[f] * F.one_hot(cells[..., f].long(), x_.shape[-1]) for f, x_ in enumerate(cl)]
        gl = [x_ + self.persist[nc + f] * F.one_hot(glob[..., f].long(), x_.shape[-1]) for f, x_ in enumerate(gl)]
        return cl, gl

    def shared_parameters(self):
        """The trunk's parameters, without the modules the fast parameters replace (they would get no gradient)."""
        skip = {id(p) for p in [self.act.weight, self.cell[0].weight]}
        skip |= {id(p) for m in list(self.cell_heads) + list(self.glob_heads) for p in m.parameters()}
        skip |= {id(p) for p in self.mu.parameters()} | {id(p) for p in self.log_sigma.parameters()}
        return [p for p in self.parameters() if id(p) not in skip]
