"""The robot's world model (E2): (observation, action) -> the next observation, one categorical prediction per field
of every cell token and of the global token.

The 9 x 9 map is padded with its border walls to 11 x 11 tokens, so position can be either
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

from village import CELL_VOCAB, GLOBAL_VOCAB, N, N_ACTIONS, WALL

P = N + 2                                  # padded side
TOK = 1 + P * P                            # global token + padded cells
INTERIOR = torch.tensor([1 + (r + 1) * P + (c + 1) for r in range(N) for c in range(N)])


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
    def __init__(self, d=192, layers=4, heads=4, mlp=4, pos="abs", persist=0.0, radius=1):
        super().__init__()
        self.persist = nn.Parameter(torch.full((len(CELL_VOCAB) + len(GLOBAL_VOCAB),), persist)) if persist else None
        self.d, self.layers, self.heads, self.posmode = d, layers, heads, pos
        self.cell = nn.ModuleList(nn.Embedding(n, d) for n in CELL_VOCAB)
        self.glob = nn.ModuleList(nn.Embedding(n, d) for n in GLOBAL_VOCAB)
        self.act = nn.Embedding(N_ACTIONS, d)
        self.gtype = nn.Parameter(torch.randn(d) * 0.02)
        if pos == "abs":
            self.pos = nn.Parameter(torch.randn(TOK, d) * 0.02)
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
        self.cell_heads = nn.ModuleList(nn.Linear(d, n) for n in CELL_VOCAB)
        self.glob_heads = nn.ModuleList(nn.Linear(d, n) for n in GLOBAL_VOCAB)
        for e in list(self.cell) + list(self.glob) + [self.act]:
            nn.init.normal_(e.weight, std=0.02)
        border = torch.zeros(len(CELL_VOCAB), dtype=torch.long)
        border[0] = WALL
        self.register_buffer("border", border, persistent=False)
        self.register_buffer("interior", INTERIOR, persistent=False)

    def bias(self):
        if self.posmode == "abs":
            return None
        h = self.heads
        cells = self.rel[:, self.dr, self.dc]                         # [h, P*P, P*P]
        out = torch.empty(h, TOK, TOK, device=cells.device, dtype=cells.dtype)
        out[:, 1:, 1:] = cells.masked_fill(self.far, float("-inf"))
        out[:, 0, 1:] = self.gbias[:, 0, None]
        out[:, 1:, 0] = self.gbias[:, 1, None]
        out[:, 0, 0] = self.gbias[:, 2]
        return out

    def features(self, cells, glob, act, extra=None):
        """Final token states [B, TOK, d] (after the last LayerNorm). cells [B,81,9], glob [B,4], act [B] integers."""
        b = cells.shape[0]
        grid = self.border.expand(b, P, P, -1).clone()
        grid[:, 1:-1, 1:-1] = cells.view(b, N, N, -1).long()
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
        """-> (cell logits: 9 tensors [B, 81, V_f]; global logits: 4 tensors [B, V_g]; the features read by the heads)."""
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
    return 2 * weights * TOK + 4 * net.layers * TOK * TOK * net.d
