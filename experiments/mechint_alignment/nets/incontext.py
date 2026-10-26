"""The world model for the world family, option B (E8): a world's rules inferred IN CONTEXT, in activations.

For each past step j of the current world, a transition token
    u_j = MLP([pool(obs_j), E_act(a_j), pool(obs_{j+1})])
where pool is a LEARNED attention pooling (K queries) over the action-free trunk features of an observation -- nothing
tells it where to look. A small causal transformer reads u_0 .. u_{t-1} and gives a belief state c_t; the prediction
for step t is the shared world model on (obs_t, a_t) with c_t added to every token. All weights are shared by all
worlds: whatever distinguishes this world must come from its own past steps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.worldmodel import Block, WorldModel


class ContextModel(WorldModel):
    def __init__(self, spec, d=128, layers=4, heads=4, mlp=4, persist=5.0, radius=1, ctx_layers=2, queries=4,
                 max_len=129):
        super().__init__(spec, d, layers, heads, mlp, pos="local", persist=persist, radius=radius)
        self.q = nn.Parameter(torch.randn(queries, d) * 0.02)
        self.kv = nn.Linear(d, 2 * d)
        self.u = nn.Sequential(nn.Linear(2 * queries * d + d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.cpos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.cblocks = nn.ModuleList(Block(d, heads, mlp) for _ in range(ctx_layers))
        self.cln = nn.LayerNorm(d)
        self.null = nn.Parameter(torch.zeros(d))
        self.ctx_proj = nn.Linear(d, d)
        nn.init.zeros_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)
        mask = torch.full((max_len, max_len), float("-inf")).triu(1)
        self.register_buffer("causal", mask.expand(heads, max_len, max_len).contiguous(), persistent=False)

    def pool(self, h):
        """[B, T, d] token features -> [B, queries * d]."""
        k, v = self.kv(h).chunk(2, -1)
        att = torch.einsum("qd,btd->bqt", self.q, k) / k.shape[-1] ** 0.5
        return torch.einsum("bqt,btd->bqd", att.softmax(-1), v).flatten(1)

    def contexts(self, obs_c, obs_g, act, nxt_c, nxt_g):
        """For W worlds of L steps (tensors [W, L, ...]): the belief state before each step, c [W, L, d]."""
        W, L = act.shape
        flat = lambda x: x.reshape(W * L, *x.shape[2:])
        p0 = self.pool(self.features(flat(obs_c), flat(obs_g), None))
        p1 = self.pool(self.features(flat(nxt_c), flat(nxt_g), None))
        u = self.u(torch.cat([p0, self.act(flat(act).long()), p1], -1)).view(W, L, -1)
        x = u + self.cpos[:L]
        bias = self.causal[:, :L, :L]
        for block in self.cblocks:
            x = block(x, bias)
        x = self.cln(x)                                   # x[:, j] summarises steps 0..j
        return torch.cat([self.null.expand(W, 1, -1), x[:, :-1]], 1)   # before step t: steps 0..t-1

    def predict(self, cells, glob, act, ctx):
        """The world model for (obs_t, a_t) with the belief state added to every token."""
        b = cells.shape[0]
        x_ctx = self.ctx_proj(ctx)
        h = self.features(cells, glob, act, extra=x_ctx)
        hc, hg = h[:, self.interior], h[:, 0]
        with torch.autocast(h.device.type, enabled=False):
            cl = [m(hc) for m in self.cell_heads]
            gl = [m(hg) for m in self.glob_heads]
            nc = len(cl)
            cl = [x + self.persist[f] * F.one_hot(cells[..., f].long(), x.shape[-1]) for f, x in enumerate(cl)]
            gl = [x + self.persist[nc + f] * F.one_hot(glob[..., f].long(), x.shape[-1]) for f, x in enumerate(gl)]
        return cl, gl
