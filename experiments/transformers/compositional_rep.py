"""Is the FAILURE representational, or is it in the read-out? A TRE-style probe of the task code.

THE SPLIT THIS DECIDES. The model solves its training compositions and none of the held-out ones. Two very different
stories fit that:
  (A) REPRESENTATIONAL — it never built the primitives as separable reusable objects, and stored 42 atoms. Then there are
      no "local operations" for a harness to recombine, and the missing thing is architectural.
  (B) READ-OUT — the task code IS compositional, and what fails is turning it into behaviour on a combination never
      practised. A much more tractable problem, and one a harness could plausibly address.
The behavioural numbers cannot tell these apart. The representation can.

THE MEASURE. Andreas 2019 (`Measuring Compositionality in Representation Learning`, arXiv:1902.07181) gives the one formal,
computable definition in ML: TREE RECONSTRUCTION ERROR. Given representations of composed inputs with known derivations,
INFER primitive embeddings and a composition operator, and measure the residual — zero means perfectly compositional.
Crucially the primitives are FITTED, not read off, which is what makes it usable here: this model never sees a primitive
alone, only 2-compositions, so there is no `rep(a)` to look up.

Composition here is ORDERED (`a∘b ≠ b∘a`), so a plain additive code is the wrong hypothesis; the fitted form is additive
WITH ROLES, `rep(a,b) ≈ α_a + β_b`, which is order-sensitive and properly constrained: 24 vectors to explain 42 training
representations, so a good fit is a real compression rather than a re-description. A per-primitive MATRIX operator
(`rep ≈ M_b·η_a`) is the more natural "apply b to a" form, but at d=96 that is ~110k parameters for 4k target numbers — it
would fit anything and measure nothing, so it is deliberately not used.

THE TEST IS GENERALISATION, NOT FIT. The roles are fitted on TRAINING compositions only and used to predict the
representations of HELD-OUT ones. Two controls, because a 24-vector basis explaining 42 vectors is not by itself
surprising:
  * MEAN — predict every representation by the training mean. This is what R²=0 means.
  * SHUFFLED — refit after permuting which (a,b) label each representation carries. Any apparent structure that survives
    label destruction was an artefact of the fit's flexibility, not of the model.

Usage:  python experiments/transformers/compositional_rep.py
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from diversity import NAMES, batch, build
from h1_lid import L, V, Model, out_mask


def hidden(model, tok):
    """The residual stream just before the read-out — `Model.forward` without the final linear."""
    h = model.emb(tok)
    if model.pos is not None:
        h = h + model.pos[:, :tok.shape[1]]
    for b in model.blocks:
        h = b(h)
    return model.norm(h)


@torch.no_grad()
def task_codes(model, tasks, dev, K, n=256):
    """One vector per task: the hidden state at the LAST demonstration's final input token — the point where the model has
    seen every example and must commit to an answer, so whatever identifies the task has to be there. AVERAGED over many
    random inputs, which marginalises the input content out and leaves the task code."""
    at = (K - 1) * 2 * L + L - 1
    out = []
    for t in range(len(tasks)):
        tok = batch(n, K, tasks, dev, None, fixed=t)
        out.append(hidden(model, tok)[:, at].float().mean(0))
    return torch.stack(out)


def fit_roles(codes, pairs, n_prim):
    """Least-squares fit of `rep(a,b) ≈ α_a + β_b`. Returns the (2·n_prim, d) role matrix."""
    X = torch.zeros(len(pairs), 2 * n_prim, device=codes.device)
    for i, (a, b) in enumerate(pairs):
        X[i, a] = 1.0
        X[i, n_prim + b] = 1.0
    # PSEUDO-INVERSE, not lstsq. The design is rank-deficient BY CONSTRUCTION: every row carries exactly one alpha and
    # one beta indicator, so adding a constant to every alpha and subtracting it from every beta changes nothing — a gauge
    # freedom, rank <= 2*n_prim - 1. CUDA's lstsq assumes full rank and silently returned NaN for every score.
    # pinv gives the minimum-norm solution, which is the right choice for a degenerate design rather than a patch over it.
    return torch.linalg.pinv(X) @ codes, X


def predict(W, pairs, n_prim, dev):
    X = torch.zeros(len(pairs), 2 * n_prim, device=dev)
    for i, (a, b) in enumerate(pairs):
        X[i, a] = 1.0
        X[i, n_prim + b] = 1.0
    return X @ W


def r2(true, pred, baseline):
    """1 − residual/baseline-residual. 0 = no better than predicting the training mean; 1 = exact."""
    ss = ((true - pred) ** 2).sum().item()
    bs = ((true - baseline) ** 2).sum().item()
    return 1.0 - ss / bs if bs > 1e-12 else float("nan")   # nan iff the codes have collapsed, which is itself a finding


@torch.no_grad()
def accuracy(model, tasks, dev, K, n=256):
    hits = []
    for t in range(len(tasks)):
        tok = batch(n, K, tasks, dev, None, fixed=t)
        pred = model(tok)[:, :-1].argmax(-1)
        msk = out_mask(K, dev)[1:]
        ok = (pred[:, msk] == tok[:, 1:][:, msk]).reshape(n, K, L).all(-1)
        hits.append(ok[:, -1].float().mean().item())
    return sum(hits) / len(hits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    pool, test = build(args.seed)
    print(f"device {dev} | training on {len(pool)} compositions, {len(test)} held out | steps={args.steps}")

    model = Model(max_len=args.k * 2 * L + 2, pos="rope").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98), fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    trainer = torch.compile(model) if dev == "cuda" else model
    g = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for _ in range(args.steps):
        tok = batch(args.batch, args.k, pool, dev, g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        m = out_mask(args.k, dev)[1:]
        loss = F.cross_entropy(logits[:, m].reshape(-1, V).float(), tok[:, 1:][:, m].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"trained in {time.time() - t0:.0f}s | BEHAVIOUR: train acc {accuracy(model, pool, dev, args.k):.3f}, "
          f"held-out acc {accuracy(model, test, dev, args.k):.3f}")

    tr_codes, te_codes = task_codes(model, pool, dev, args.k), task_codes(model, test, dev, args.k)
    n_prim = len(NAMES)
    W, _ = fit_roles(tr_codes, pool, n_prim)
    mean = tr_codes.mean(0, keepdim=True)

    tr_r2 = r2(tr_codes, predict(W, pool, n_prim, dev), mean)
    te_r2 = r2(te_codes, predict(W, test, n_prim, dev), mean)

    # SHUFFLED control: destroy the label-to-representation correspondence and refit. Anything that survives is the fit's
    # flexibility, not the model's structure.
    perm = torch.randperm(len(pool), generator=torch.Generator().manual_seed(args.seed + 1))
    Ws, _ = fit_roles(tr_codes[perm], pool, n_prim)
    sh_r2 = r2(te_codes, predict(Ws, test, n_prim, dev), mean)

    print(f"\nTRE-style probe: does  rep(a,b) ~ alpha_a + beta_b  hold?   ({2 * n_prim} role vectors for "
          f"{len(pool)} training codes)")
    print(f"   R2 on TRAINING compositions (fit)        : {tr_r2:+.3f}")
    print(f"   R2 on HELD-OUT compositions (the test)   : {te_r2:+.3f}")
    print(f"   R2 held-out, SHUFFLED labels (control)   : {sh_r2:+.3f}")
    print("\nHigh held-out R2 with a near-zero shuffled control => the task CODE is compositional and the failure is in")
    print("the READ-OUT. Held-out R2 near the shuffled control => the representation itself is not compositional, and")
    print("there are no reusable parts for any harness to recombine.")


if __name__ == "__main__":
    main()
