"""MIXING versus APPLICATION: is attention the wrong primitive for composition?

THE CLAIM UNDER TEST. Attention MIXES — it sums value vectors weighted by similarity into a residual stream. It never
APPLIES one as a function to another. Composition needs a retrieved thing to PARAMETERISE a transformation, which is a
different operation, and its absence would explain why the same failure appeared on Mamba-3 (`binding_rule.py`): an SSM
also mixes, via state, so swapping one mixer for another changes nothing.

`compositional_rep.py` sharpened this: the transformer's task CODE is compositional (held-out R² +0.52 against a shuffled
control of −0.63) while its BEHAVIOUR on those same held-out compositions is 0.038. It can represent `a∘b` and cannot
execute it. So the missing primitive is application.

THE DESIGN IS A ONE-VARIABLE TEST, which is the whole point of building it rather than just adding architectures. Both
arms share the demonstration encoder, the input embedding, and the decoder. They differ ONLY in what happens to the task
code `z`:

    MIX    h = MLP([e(x), z])           the code is concatenated and mixed — attention's move
    APPLY  h = M(z) · e(x),  M(z) = Σ zₖ Bₖ    the code parameterises an OPERATOR applied to the input

APPLY is cross-pollinated from `src/tbt/operator.py`, where a learned operator acts on a state and one demonstration
generalises to every position — `reference_operator_as_group_representation` takes the group-representation claim
literally: motion is a learned matrix acting on a code, so composing motions is composing matrices.

WHAT IS *NOT* BUILT IN. Nothing tells APPLY that a task is a composition, that compositions factor, or that `M(z_{a∘b})`
should equal `M(z_b)M(z_a)`. It sees the same demonstrations and the same loss. The operator form is a general
architectural bias (bilinear / hypernetwork / FiLM-style conditioning), not task-specific structure — the
`reference_lid_locally_in_distribution` triage: scalable inductive bias, legitimate; task-specific rigging, not. If APPLY
generalises to held-out compositions, that is the bias earning it, not the answer being handed over.

The transformer is kept as a third arm so the result is anchored to everything measured so far.

Usage:  python experiments/transformers/apply_vs_mix.py
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from diversity import apply_pair, build
from h1_lid import L, V, Model, out_mask


def make(B, K, tasks, dev, g, fixed=None):
    """K−1 demonstrations plus one query, as explicit tensors. Same data as the token-stream version, reshaped so an
    architecture that is not a sequence model can consume it."""
    idx = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    y = torch.empty_like(x)
    for t in idx.unique():
        m = idx == t
        y[m] = apply_pair(x[m], tasks[int(t)])
    return x[:, :-1], y[:, :-1], x[:, -1], y[:, -1]           # demos_in, demos_out, query_in, query_out


class Shared(nn.Module):
    """Everything both arms have in common, so the comparison isolates one thing."""

    def __init__(self, d, code):
        super().__init__()
        self.tok = nn.Embedding(V, d // L * 2)
        self.demo = nn.Sequential(nn.Linear(2 * L * (d // L * 2), 2 * d), nn.GELU(), nn.Linear(2 * d, code))
        self.embed = nn.Sequential(nn.Linear(L * (d // L * 2), 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.out = nn.Linear(d, L * V)

    def code(self, dx, dy):
        """Pool over demonstrations. Mean-pooling is permutation-invariant, which is CORRECT here — demonstration order
        carries no information — and both arms get it, so it cannot explain a difference between them."""
        B, K = dx.shape[0], dx.shape[1]
        e = torch.cat([self.tok(dx), self.tok(dy)], dim=-1).reshape(B, K, -1)
        return self.demo(e).mean(1)

    def query(self, qx):
        return self.embed(self.tok(qx).reshape(qx.shape[0], -1))

    def decode(self, h):
        return self.out(h).reshape(-1, L, V)


class Mix(Shared):
    """The task code is CONCATENATED with the input and mixed — what attention does, stripped to its essence."""

    def __init__(self, d=96, code=48):
        super().__init__(d, code)
        self.head = nn.Sequential(nn.Linear(d + code, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, dx, dy, qx):
        return self.decode(self.head(torch.cat([self.query(qx), self.code(dx, dy)], dim=-1)))


class Apply(Shared):
    """The task code parameterises an OPERATOR applied to the input: `M(z) = Σ zₖ Bₖ`, then `h = M(z)·e(x)`. A product of
    operators is itself an operator, so a composition never trained on has somewhere to live BY CONSTRUCTION — which is
    the hypothesis, not a guarantee: nothing forces the model to use the basis that way."""

    def __init__(self, d=96, code=48, rank=10):
        super().__init__(d, code)
        # A RANK projection, so the comparison is parameter-matched. A full `code`-sized basis is code*d*d = 442k
        # parameters against MIX's 92k head, and an arm that wins on 2.5x the parameters has not shown anything about
        # the primitive. `rank` is chosen to equalise the two heads, not tuned for accuracy.
        self.to_r = nn.Linear(code, rank)
        self.basis = nn.Parameter(torch.randn(rank, d, d) / math.sqrt(d * rank))
        self.scale = nn.LayerNorm(d)

    def forward(self, dx, dy, qx):
        z = self.to_r(self.code(dx, dy))
        M = torch.einsum("bk,kij->bij", z, self.basis)
        return self.decode(self.scale(torch.einsum("bij,bj->bi", M, self.query(qx))))


def run_arm(name, model, pool, test, args, dev):
    model = model.to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for _ in range(args.steps):
        dx, dy, qx, qy = make(args.batch, args.k, pool, dev, g)
        loss = F.cross_entropy(model(dx, dy, qx).reshape(-1, V), qy.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    @torch.no_grad()
    def acc(tasks, n=384):
        hits = []
        for t in range(len(tasks)):
            dx, dy, qx, qy = make(n, args.k, tasks, dev, None, fixed=t)
            hits.append((model(dx, dy, qx).argmax(-1) == qy).all(-1).float().mean().item())
        return sum(hits) / len(hits), sum(h >= 0.8 for h in hits)
    tr, tr_s = acc(pool)
    te, te_s = acc(test)
    print(f"{name:<14}{n_par/1000:>8.0f}k{tr:>12.3f}{tr_s:>7}/{len(pool):<4}{te:>14.3f}{te_s:>8}/{len(test):<4}"
          f"{time.time()-t0:>8.0f}")
    return te


def run_transformer(pool, test, args, dev):
    """The architecture everything so far was measured on, as an anchor."""
    model = Model(max_len=args.k * 2 * L + 2, pos="rope").to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    trainer = torch.compile(model) if dev == "cuda" else model
    t0 = time.time()
    for _ in range(args.steps):
        dx, dy, qx, qy = make(args.batch, args.k, pool, dev, g)
        tok = torch.stack([torch.cat([dx, qx[:, None]], 1), torch.cat([dy, qy[:, None]], 1)], 2).reshape(args.batch, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        m = out_mask(args.k, dev)[1:]
        loss = F.cross_entropy(logits[:, m].reshape(-1, V).float(), tok[:, 1:][:, m].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    @torch.no_grad()
    def acc(tasks, n=384):
        hits = []
        for t in range(len(tasks)):
            dx, dy, qx, qy = make(n, args.k, tasks, dev, None, fixed=t)
            tok = torch.stack([torch.cat([dx, qx[:, None]], 1), torch.cat([dy, qy[:, None]], 1)], 2).reshape(n, -1)
            pred = model(tok)[:, :-1].argmax(-1)
            msk = out_mask(args.k, dev)[1:]
            ok = (pred[:, msk] == tok[:, 1:][:, msk]).reshape(n, args.k, L).all(-1)
            hits.append(ok[:, -1].float().mean().item())
        return sum(hits) / len(hits), sum(h >= 0.8 for h in hits)
    tr, tr_s = acc(pool)
    te, te_s = acc(test)
    print(f"{'transformer':<14}{n_par/1000:>8.0f}k{tr:>12.3f}{tr_s:>7}/{len(pool):<4}{te:>14.3f}{te_s:>8}/{len(test):<4}"
          f"{time.time()-t0:>8.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default="mix,apply,transformer")
    p.add_argument("--rank", type=int, default=10, help="operator basis size; set to match MIX's params")
    p.add_argument("--d", type=int, default=96, help="APPLY width; lowering it buys basis RANK at fixed params")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    pool, test = build(args.seed)
    print(f"device {dev} | {len(pool)} training compositions, {len(test)} held out | steps={args.steps} seed={args.seed}")
    print(f"\n{'arm':<14}{'params':>9}{'train acc':>12}{'solved':>12}{'HELD-OUT acc':>14}{'solved':>13}{'secs':>8}")
    arms = args.arms.split(",")
    if "mix" in arms:
        torch.manual_seed(args.seed); run_arm("mix", Mix(), pool, test, args, dev)
    if "apply" in arms:
        torch.manual_seed(args.seed); run_arm("apply", Apply(d=args.d, rank=args.rank), pool, test, args, dev)
    if "transformer" in arms:
        torch.manual_seed(args.seed); run_transformer(pool, test, args, dev)
    print("\nMIX and APPLY share encoder, embedding and decoder; they differ ONLY in whether the task code is mixed with")
    print("the input or APPLIED to it as an operator. A gap between them is that one primitive and nothing else.")


if __name__ == "__main__":
    main()
