"""The world model (E2): (observation, action) -> the next observation, one categorical prediction per field of every
cell token and of the global token. Built from a world's WorldSpec (gridworld.py), so it serves any grid world.

TRUNK and ADAPTERS. The adapters are world-specific: the field embeddings, the action embedding, the output heads,
the persistence prior (and an absolute position table). The trunk is world-agnostic: the transformer blocks, the final
norm, the global-token type and the relative-position biases. load_trunk moves a trained trunk to a model of another
world (another spec): the start of transfer between worlds.

The n x n map is padded with its border walls to (n + 2) x (n + 2) tokens, so position can be either
- "abs": a learned absolute embedding per token (the default transformer), or
- "rel": no absolute position at all, only a learned attention bias per head and per relative offset (dr, dc).
  Translation-equivariant: a rule seen at one place applies everywhere. The explicit border tokens let the model
  see the edge of the world without knowing where it is.
- "local": "rel" plus a LOCALITY PRIOR: a cell attends only to cells within `radius` (Chebyshev) and to the global
  token; the global token attends to everything. Physics is local; depth widens the reach (radius x layers).
The action is embedded and added to every token.

persist > 0 adds a learnable PERSISTENCE PRIOR: each field's logits get +beta_f on the field's current value, so the
model starts out predicting "nothing changes" and its gradient comes from the refutations, not the ~700 confirmed
entries per transition. A general prior (inertia: things stay as they are unless acted on), not a rule of Village.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TRUNK = ("blocks.", "ln.", "gtype", "rel", "gbias")     # parameter-name prefixes of the world-agnostic trunk


class Block(nn.Module):
    def __init__(self, d, heads, mlp):
        super().__init__()
        self.heads = heads
        self.ln1, self.qkv, self.proj = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ln2, self.fc, self.out = nn.LayerNorm(d), nn.Linear(d, mlp * d), nn.Linear(mlp * d, d)

    def forward(self, x, bias=None):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        mask = None if bias is None else bias.to(q.dtype)[None]
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class WorldModel(nn.Module):
    def __init__(self, spec, d=192, layers=4, heads=4, mlp=4, pos="abs", persist=0.0, radius=1):
        super().__init__()
        self.spec = spec
        n = spec.n
        self.P = P = n + 2                                           # padded side
        self.tok = 1 + P * P                                         # global token + padded cells
        n_fields = len(spec.cell_vocab) + len(spec.global_vocab)
        self.persist = nn.Parameter(torch.full((n_fields,), persist)) if persist else None
        self.d, self.layers, self.heads, self.posmode = d, layers, heads, pos
        self.cell = nn.ModuleList(nn.Embedding(v, d) for v in spec.cell_vocab)
        self.glob = nn.ModuleList(nn.Embedding(v, d) for v in spec.global_vocab)
        self.act = nn.Embedding(spec.n_actions, d)
        self.gtype = nn.Parameter(torch.randn(d) * 0.02)
        if pos == "abs":
            self.pos = nn.Parameter(torch.randn(self.tok, d) * 0.02)
        else:
            self.rel = nn.Parameter(torch.zeros(heads, 2 * P - 1, 2 * P - 1))
            self.gbias = nn.Parameter(torch.zeros(heads, 3))          # global->cell, cell->global, global->global
            r = torch.arange(P * P) // P
            c = torch.arange(P * P) % P
            self.register_buffer("dr", (r[:, None] - r[None, :] + P - 1), persistent=False)
            self.register_buffer("dc", (c[:, None] - c[None, :] + P - 1), persistent=False)
            far = torch.maximum((r[:, None] - r[None, :]).abs(), (c[:, None] - c[None, :]).abs()) > radius
            self.register_buffer("far", far if pos == "local" else torch.zeros_like(far), persistent=False)
        self.blocks = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.cell_heads = nn.ModuleList(nn.Linear(d, v) for v in spec.cell_vocab)
        self.glob_heads = nn.ModuleList(nn.Linear(d, v) for v in spec.global_vocab)
        for e in list(self.cell) + list(self.glob) + [self.act]:
            nn.init.normal_(e.weight, std=0.02)
        border = torch.zeros(len(spec.cell_vocab), dtype=torch.long)
        border[0] = spec.wall
        self.register_buffer("border", border, persistent=False)
        interior = torch.tensor([1 + (r + 1) * P + (c + 1) for r in range(n) for c in range(n)])
        self.register_buffer("interior", interior, persistent=False)

    @torch.no_grad()
    def load_trunk(self, src):
        """Copy the world-agnostic trunk of `src` (a model of any world with the same width, depth and heads). The
        relative-position biases are copied for the offsets both grids have (centred), so grids may differ in size."""
        mine = dict(self.named_parameters())
        for name, w in src.named_parameters():
            if not name.startswith(TRUNK) or name not in mine:
                continue
            dst = mine[name]
            if name == "rel" and dst.shape != w.shape:
                k = min(dst.shape[-1], w.shape[-1])
                a, b = (dst.shape[-1] - k) // 2, (w.shape[-1] - k) // 2
                dst[:, a:a + k, a:a + k] = w[:, b:b + k, b:b + k]
            else:
                dst.copy_(w)

    def bias(self):
        if self.posmode == "abs":
            return None
        h = self.heads
        cells = self.rel[:, self.dr, self.dc]                         # [h, P*P, P*P]
        out = torch.empty(h, self.tok, self.tok, device=cells.device, dtype=cells.dtype)
        out[:, 1:, 1:] = cells.masked_fill(self.far, float("-inf"))
        out[:, 0, 1:] = self.gbias[:, 0, None]
        out[:, 1:, 0] = self.gbias[:, 1, None]
        out[:, 0, 0] = self.gbias[:, 2]
        return out

    def features(self, cells, glob, act, extra=None):
        """Final token states [B, tok, d] (after the last LayerNorm). cells [B, n*n, F], glob [B, G], act [B] integers."""
        b, P = cells.shape[0], self.P
        grid = self.border.expand(b, P, P, -1).clone()
        grid[:, 1:-1, 1:-1] = cells.view(b, P - 2, P - 2, -1).long()
        x = sum(e(grid[..., i]) for i, e in enumerate(self.cell)).view(b, P * P, self.d)
        g = sum(e(glob[..., i].long()) for i, e in enumerate(self.glob)) + self.gtype
        x = torch.cat([g[:, None], x], 1)
        if act is not None:                                          # None: action-free features (AgentNet's trunk)
            x = x + self.act(act.long())[:, None]
        if extra is not None:                                        # a per-row vector added to every token (E8 B)
            x = x + extra[:, None]
        if self.posmode == "abs":
            x = x + self.pos
        bias = self.bias()
        for block in self.blocks:
            x = block(x, bias)
        return self.ln(x.float())

    def forward(self, cells, glob, act):
        """-> (cell logits: one tensor [B, n*n, V_f] per cell field; global logits: one [B, V_g] per global field;
        the features read by the heads)."""
        h = self.features(cells, glob, act)
        hc, hg = h[:, self.interior], h[:, 0]
        with torch.autocast(h.device.type, enabled=False):
            cl = [m(hc) for m in self.cell_heads]
            gl = [m(hg) for m in self.glob_heads]
            if self.persist is not None:
                nc = len(cl)
                cl = [x + self.persist[f] * F.one_hot(cells[..., f].long(), x.shape[-1]) for f, x in enumerate(cl)]
                gl = [x + self.persist[nc + f] * F.one_hot(glob[..., f].long(), x.shape[-1]) for f, x in enumerate(gl)]
            return cl, gl, (hc, hg)


def flops_per_forward(net):
    weights = sum(p.numel() for n, p in net.named_parameters() if n.startswith("blocks") and p.dim() == 2)
    return 2 * weights * net.tok + 4 * net.layers * net.tok * net.tok * net.d
