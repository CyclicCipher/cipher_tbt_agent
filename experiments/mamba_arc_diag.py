"""Baseline diagnostic: can a vanilla Mamba-3 ONLINE-learn the dynamics of a real ARC-AGI-3 game from scratch (no
pretraining), as it must during a private game? We stream the captured real frames and, at each step, the model
predicts the NEXT frame from the current frame + action and takes ONE gradient step (online SGD) -- exactly the
private-game setting (learn from very little, by changing weights).

The decisive metric is NOT overall next-frame accuracy (the trivial COPY predictor -- next == current -- already
scores high because most cells never change), but accuracy on the cells that actually CHANGE: the dynamics. A model
that only learns "copy the frame" scores ~0 there. This is the grounding test for the binding hypothesis: statistical
next-frame prediction without grounding learns the trivial structure, not the action->effect dynamics.

Usage: python experiments/mamba_arc_diag.py <game> [d_model] [n_layers]   (game = cn04 | ls20)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mamba"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba3 import Mamba3

N_COLORS = 16
ACTIONS = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}


class MambaARC(nn.Module):
    """Per-cell colour embedding + a 2-D learned position + an action token, a stack of Mamba-3 layers, a per-cell
    16-way head predicting the next frame. The frame is flattened to a length-H*W sequence (Mamba is O(n))."""

    def __init__(self, H, W, d_model=96, n_layers=2):
        super().__init__()
        self.H, self.W = H, W
        self.cell = nn.Embedding(N_COLORS, d_model)
        self.pos = nn.Parameter(torch.randn(1, H * W, d_model) * 0.02)
        self.act = nn.Embedding(len(ACTIONS), d_model)
        self.layers = nn.ModuleList([Mamba3(d_model=d_model, headdim=d_model // 2, d_state=64, chunk_size=64)
                                     for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_COLORS)

    def forward(self, grid, action):                                  # grid: (B,H,W) long; action: (B,) long
        B = grid.shape[0]
        x = self.cell(grid.view(B, -1)) + self.pos + self.act(action)[:, None, :]
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))                                    # residual Mamba block
        return self.head(self.out_norm(x)).view(B, self.H, self.W, N_COLORS)


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "cn04"
    d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-3
    scratch = os.path.join(os.path.dirname(__file__), "..",
                           "..", "..", "AppData", "Local", "Temp", "claude")  # fallback only
    path = None
    for cand in [os.path.join(os.path.dirname(__file__), f"frames_{game}.json"),
                 os.path.join(os.environ.get("SCRATCH", ""), f"frames_{game}.json")]:
        if cand and os.path.exists(cand):
            path = cand; break
    if path is None:                                                  # the captured frames live in the scratchpad
        import glob
        hits = glob.glob(os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "claude", "**",
                                      f"frames_{game}.json"), recursive=True)
        path = hits[0] if hits else None
    assert path, f"frames_{game}.json not found"
    frames = json.load(open(path))
    grids = [torch.tensor(f["grid"], dtype=torch.long) for f in frames]
    acts = [ACT_IDX.get(f["action"], 0) for f in frames]
    H, W = grids[0].shape
    dev = "cuda"
    torch.manual_seed(0)
    model = MambaARC(H, W, d_model, n_layers).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"{game}: {len(grids)} frames {H}x{W}, Mamba-3 d_model={d_model} x{n_layers} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")

    logs = []                                                         # (overall_acc, changed_acc, copy_acc, n_changed)
    for t in range(len(grids) - 1):
        g0 = grids[t].unsqueeze(0).cuda()
        g1 = grids[t + 1].unsqueeze(0).cuda()
        a = torch.tensor([acts[t]], device=dev)
        logits = model(g0, a)                                        # (1,H,W,16)
        loss = F.cross_entropy(logits.reshape(-1, N_COLORS), g1.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = logits.argmax(-1)
            changed = (g1 != g0)                                     # the dynamics: cells that actually change
            nch = int(changed.sum())
            overall = (pred == g1).float().mean().item()
            ch_acc = ((pred == g1) & changed).float().sum().item() / max(nch, 1)
            copy = (g0 == g1).float().mean().item()                  # the trivial copy-the-frame baseline
            logs.append((overall, ch_acc, copy, nch))

    def avg(rows, i):
        return sum(r[i] for r in rows) / max(len(rows), 1)
    half = len(logs) // 2
    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated()//1048576} MB")
    print(f"  mean cells changed/step: {avg(logs,3):.0f} / {H*W}  (copy-baseline overall acc: {avg(logs,2):.3f})")
    print(f"  EARLY (first half): overall acc {avg(logs[:half],0):.3f}   CHANGED-cell acc {avg(logs[:half],1):.3f}")
    print(f"  LATE  (last  half): overall acc {avg(logs[half:],0):.3f}   CHANGED-cell acc {avg(logs[half:],1):.3f}")
    print(f"  --> the test: does LATE changed-cell acc rise above ~0 (learning the dynamics) or stay flat (copy only)?")


if __name__ == "__main__":
    main()
