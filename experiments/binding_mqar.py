"""Binding experiment, step 1: the elementary role-filler binding probe = associative recall (MQAR). Each sequence
presents m (key, value) pairs then queries the keys in a new order; the model must recall each bound value. The
binding is RE-RANDOMISED every sequence, so it cannot be memorised -- the model must BIND key->value in-context and
retrieve. This is the cleanest test of role(key) (X) filler(value) binding.

The stress axis is the binding LOAD (m, the number of pairs): a fixed-size SSM state (Mamba) must cram all bindings
into one vector, so recall should degrade as m grows -- the binding bottleneck. We measure vanilla Mamba-3's recall
accuracy vs load. (Next: add an outer-product / VSA binding state and see if it fixes the high-load failure.)

Usage: python experiments/binding_mqar.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mamba"))
import torch, torch.nn as nn, torch.nn.functional as F
from mamba_ssm.modules.mamba3 import Mamba3

K, V = 64, 64                                                # key vocab, value vocab
PAD = K + V                                                 # pad id
VOCAB = K + V + 1
CHUNK = 16


def gen_batch(B, m, dev):
    """B sequences of m (key,value) pairs then m queries. Returns tokens, next-token targets, and a loss MASK that is
    True only at the query-key positions (where the model must predict the bound value)."""
    L = 4 * m
    Lpad = ((L + CHUNK - 1) // CHUNK) * CHUNK
    toks = torch.full((B, Lpad), PAD, dtype=torch.long)
    tgt = torch.zeros(B, Lpad, dtype=torch.long)
    mask = torch.zeros(B, Lpad, dtype=torch.bool)
    for b in range(B):
        keys = torch.randperm(K)[:m]
        vals = torch.randint(0, V, (m,)) + K
        seq = []
        for i in range(m):
            seq += [int(keys[i]), int(vals[i])]              # the (key, value) pairs
        order = torch.randperm(m)
        for i in order.tolist():
            seq += [int(keys[i]), int(vals[i])]              # the queries: key then its value (predicted)
        seq = torch.tensor(seq)
        toks[b, :L] = seq
        tgt[b, :L - 1] = seq[1:]
        for j in range(m):                                   # answer is the token after each query key
            mask[b, 2 * m + 2 * j] = True
    return toks.to(dev), tgt.to(dev), mask.to(dev)


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


def run(m, steps=2500, B=64, lr=1e-3, d=128, n_layers=2, d_state=64, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    model = MambaLM(d, n_layers, d_state).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        toks, tgt, mask = gen_batch(B, m, dev)
        logits = model(toks)
        loss = F.cross_entropy(logits[mask], tgt[mask])
        opt.zero_grad(); loss.backward(); opt.step()
    # eval on fresh (novel-binding) batches
    model.eval(); correct = total = 0
    with torch.no_grad():
        for _ in range(20):
            toks, tgt, mask = gen_batch(B, m, dev)
            pred = model(toks).argmax(-1)
            correct += int((pred[mask] == tgt[mask]).sum()); total += int(mask.sum())
    return correct / total, torch.cuda.max_memory_allocated() // 1048576


if __name__ == "__main__":
    if len(sys.argv) > 1:                                    # single config: m [steps] [d_state]
        m = int(sys.argv[1])
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
        ds = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        acc, mem = run(m, steps=steps, d_state=ds)
        print(f"m={m} steps={steps} d_state={ds} recall_acc={acc:.3f} peakmem={mem}MB")
    else:
        print(f"MQAR (binding load m), vanilla Mamba-3 (d=128 x2, {VOCAB}-vocab, re-randomised bindings)")
        for m in [4, 8, 16, 32, 64]:
            torch.cuda.reset_peak_memory_stats()
            acc, mem = run(m)
            print(f"  load m={m:3d}  seqlen={((4*m+CHUNK-1)//CHUNK)*CHUNK:4d}  recall_acc={acc:.3f}  peakmem={mem}MB")
