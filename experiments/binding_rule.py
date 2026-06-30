"""Binding experiment step 2: RULE binding (bind-and-APPLY), the ARC-relevant test (the wall, the carry rule). Each
sequence binds each symbol to a random RULE, demonstrates it ONCE, then queries the symbol on a DIFFERENT input; the
model must infer the rule from the single demo and APPLY it to the new input -- not recall the demo answer.

Rules = cyclic shifts r_k(x) = (x+k) mod V (one demo (x -> (x+k)%V) determines k unambiguously; small lookup, not
multi-digit arithmetic). Load S is kept SMALL (within the recall-capacity ceiling found in step 1) so this isolates
bind-and-APPLY from capacity. The query input is forced != the demo input, so a copy-the-demo-answer baseline scores 0
and chance is 1/V. Question: can vanilla Mamba-3 learn one-shot rule-binding + compositional application?

Everything is VECTORISED (no per-step Python loop) and a SINGLE fixed shape (one Triton compile, then cached) -- so it
runs in seconds, not minutes. Usage: python experiments/binding_rule.py [S] [steps]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mamba"))
import torch, torch.nn as nn, torch.nn.functional as F
from mamba_ssm.modules.mamba3 import Mamba3

V = 10                  # value vocab (0..V-1)
N_SYM = 16              # symbol pool
PAD = V + N_SYM
VOCAB = V + N_SYM + 1
CHUNK = 16


def gen(B, S, dev):
    """Vectorised. Returns (tokens, targets, mask) for a batch of rule-binding sequences. Layout per sequence:
    demo block [sym, x, (x+k)%V] x S, then query block [sym, x', (x'+k)%V] x S (re-ordered); the model predicts the
    query answer (x'+k)%V at the position after each query (sym, x'). q_x' is forced != demo_x so copying fails."""
    sym = torch.argsort(torch.rand(B, N_SYM, device=dev), dim=1)[:, :S] + V    # (B,S) symbol token ids
    k = torch.randint(0, V, (B, S), device=dev)                               # the rule (shift) per symbol
    dx = torch.randint(0, V, (B, S), device=dev)                              # demo input
    dy = (dx + k) % V                                                         # demo output
    qx = (dx + torch.randint(1, V, (B, S), device=dev)) % V                   # query input (guaranteed != dx)
    qy = (qx + k) % V                                                         # query target (the APPLICATION)
    order = torch.argsort(torch.rand(B, S, device=dev), dim=1)               # query order
    g = lambda t: torch.gather(t, 1, order)
    demo = torch.stack([sym, dx, dy], dim=2).reshape(B, 3 * S)               # [sym,x,y]...
    query = torch.stack([g(sym), g(qx), g(qy)], dim=2).reshape(B, 3 * S)     # [sym,x',y']... reordered
    L = 6 * S
    Lpad = ((L + CHUNK - 1) // CHUNK) * CHUNK
    toks = torch.full((B, Lpad), PAD, device=dev, dtype=torch.long)
    toks[:, :3 * S] = demo
    toks[:, 3 * S:6 * S] = query
    tgt = torch.zeros(B, Lpad, device=dev, dtype=torch.long)
    tgt[:, :L - 1] = toks[:, 1:L]
    mask = torch.zeros(B, Lpad, device=dev, dtype=torch.bool)
    qx_pos = 3 * S + torch.arange(S, device=dev) * 3 + 1                      # position of each query input x'
    mask[:, qx_pos] = True                                                    # predict (x'+k)%V at the next token
    return toks, tgt, mask


class MambaLM(nn.Module):
    def __init__(self, d=128, n_layers=2, d_state=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.layers = nn.ModuleList([Mamba3(d_model=d, headdim=64, d_state=d_state, chunk_size=CHUNK) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.out = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, toks):
        x = self.emb(toks)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        return self.head(self.out(x))


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    B, dev = 128, "cuda"
    torch.manual_seed(0)
    model = MambaLM().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(steps):
        toks, tgt, mask = gen(B, S, dev)
        loss = F.cross_entropy(model(toks)[mask], tgt[mask])
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            t_first = time.time() - t0                                        # includes the one-time Triton compile
    train_t = time.time() - t0
    model.eval(); correct = total = 0
    with torch.no_grad():
        for _ in range(40):
            toks, tgt, mask = gen(B, S, dev)
            pred = model(toks).argmax(-1)
            correct += int((pred[mask] == tgt[mask]).sum()); total += int(mask.sum())
    acc = correct / total
    print(f"RULE-binding (bind a shift from ONE demo, apply to a NEW input) S={S} load")
    print(f"  application acc = {acc:.3f}   (chance 1/V = {1/V:.2f}; copy-the-demo = 0.00)")
    print(f"  timing: first step (incl. compile) {t_first:.1f}s | {steps} steps total {train_t:.1f}s "
          f"({1000*(train_t - t_first)/(steps-1):.1f} ms/step after compile) | peakmem {torch.cuda.max_memory_allocated()//1048576}MB")


if __name__ == "__main__":
    main()
