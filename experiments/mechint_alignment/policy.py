"""The robot's network: a small transformer over the full map (one token per cell + one global token).

Pre-norm blocks written out explicitly so the units G3/G4 study are plain tensors: the MLP neurons are
`F.gelu(block.fc(block.ln2(x)))`, the attention heads are the `heads` slices of `block.qkv`. The policy and value
heads read the global token's final state. The trunk runs in bf16 on CUDA; the heads run in fp32, so the
log-probabilities PPO compares across passes are not limited by 16-bit rounding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from village import CELL_VOCAB, GLOBAL_VOCAB, N, N_ACTIONS

TOKENS = N * N + 1


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


class Block(nn.Module):
    def __init__(self, d, heads, mlp):
        super().__init__()
        self.heads = heads
        self.ln1, self.qkv, self.proj = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ln2, self.fc, self.out = nn.LayerNorm(d), nn.Linear(d, mlp * d), nn.Linear(mlp * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class Policy(nn.Module):
    def __init__(self, d=192, layers=4, heads=4, mlp=4):
        super().__init__()
        self.d, self.layers = d, layers
        self.cell = nn.ModuleList(nn.Embedding(n, d) for n in CELL_VOCAB)
        self.glob = nn.ModuleList(nn.Embedding(n, d) for n in GLOBAL_VOCAB)
        self.pos = nn.Parameter(torch.randn(TOKENS, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.pi, self.v = nn.Linear(d, N_ACTIONS), nn.Linear(d, 1)
        for e in list(self.cell) + list(self.glob):
            nn.init.normal_(e.weight, std=0.02)
        nn.init.normal_(self.pi.weight, std=0.01)     # the initial policy is ~uniform: log2(8) = 3 bits of entropy
        nn.init.zeros_(self.pi.bias)

    def forward(self, cells, glob):
        """cells [B, N*N, 9], glob [B, 4] (integer codes) -> action logits [B, 8] (fp32), value [B] (fp32)."""
        x = sum(e(cells[..., i].long()) for i, e in enumerate(self.cell))
        g = sum(e(glob[..., i].long()) for i, e in enumerate(self.glob))
        x = torch.cat([g[:, None], x], 1) + self.pos
        for block in self.blocks:
            x = block(x)
        with torch.autocast(x.device.type, enabled=False):
            h = self.ln(x[:, 0].float())
            return self.pi(h), self.v(h).squeeze(-1)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def flops_per_forward(net):
    """Forward FLOPs per observation: 2 x (non-embedding weights) x tokens + attention scores and mixing."""
    weights = sum(p.numel() for n, p in net.named_parameters() if n.startswith("blocks") and p.dim() == 2)
    return 2 * weights * TOKENS + 4 * net.layers * TOKENS * TOKENS * net.d
