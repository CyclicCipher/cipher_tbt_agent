"""Can a transformer, trained with AdamW + LR scheduling, learn a linear regression?

"Learn a linear regression" has two readings, and they are not the same experiment, so both are run here.

  MEMORISE  — one weight vector `w` for the whole dataset. The model has to fit a single linear map. Any model with
              capacity does this, so it is not an interesting claim about transformers; it IS a good control, because it
              tells you whether the optimiser + schedule are working at all before you read anything into the hard case.
  IN-CONTEXT — a FRESH `w` per sequence, never shown to the model. The prompt is x1 y1 x2 y2 ... xk, and the model must
              predict yk. `w` cannot be memorised because it is different every sequence, so the only way to succeed is to
              perform the regression IN THE FORWARD PASS from the examples in the prompt. This is the real question, and
              the one where the answer is not trivially yes (Garg et al. 2022, arXiv:2208.01066).

THE MEASUREMENT IS THE LOSS AGAINST THE NUMBER OF IN-CONTEXT EXAMPLES, not a single number. A model that has learned the
task shows loss FALLING with k as evidence accumulates. A model that has learned only the marginal distribution shows a
FLAT curve at the variance of y — it predicts the prior mean and ignores the prompt. A single final loss cannot tell those
apart, which is why a lot of "it learned it" claims are unfalsifiable.

THREE BASELINES, so the number means something (`experiments/NOTES.md`: ground every claim with a baseline):
  * ZERO     — always predict 0, the prior mean. Normalised MSE 1.0 by construction. Anything at this level learned nothing.
  * RIDGE    — the Bayes-optimal predictor under this exact prior (w ~ N(0,I), x ~ N(0,I)): the posterior mean, which for
               noiseless data is the minimum-norm least-squares solution. NOTHING can beat this on average, so it is the
               floor, not a competitor. With d dimensions it reaches 0 once k >= d and the system is determined.
  * FIXED-w  — the MEMORISE control above.

Usage:  python experiments/transformer_linreg.py            # both settings + the schedule ablation
        python experiments/transformer_linreg.py --steps 6000 --d 16
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn


# ── the task ────────────────────────────────────────────────────────────────────────────────────────────────────────
def sample_batch(B: int, K: int, d: int, dev, fixed_w: torch.Tensor | None = None, noise: float = 0.0):
    """B sequences of K (x, y) pairs. `y = w·x` with w ~ N(0, I_d) drawn afresh PER SEQUENCE — unless `fixed_w` is given,
    in which case every sequence shares it and the map can simply be memorised (the control)."""
    x = torch.randn(B, K, d, device=dev)
    w = fixed_w.expand(B, d) if fixed_w is not None else torch.randn(B, d, device=dev)
    y = torch.einsum("bkd,bd->bk", x, w)
    if noise:
        y = y + noise * torch.randn_like(y)
    return x, y, w


def as_tokens(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Interleave into x1 y1 x2 y2 ... as one stream of (d+1)-vectors: an x sits in the first d slots, a y alone in the
    last. One embedding then serves both, and the slot a value occupies is what says which kind of token it is."""
    B, K, d = x.shape
    tok = torch.zeros(B, 2 * K, d + 1, device=x.device)
    tok[:, 0::2, :d] = x
    tok[:, 1::2, d] = y
    return tok


# ── the model ───────────────────────────────────────────────────────────────────────────────────────────────────────
class Reader(nn.Module):
    """A small causal decoder over CONTINUOUS tokens: read `y_k` off the position holding `x_k`, so the prediction is made
    from the k-1 complete examples before it plus the query — exactly the information a regression would have."""

    def __init__(self, d: int, d_model: int = 64, n_layer: int = 4, n_head: int = 4, max_len: int = 128):
        super().__init__()
        self.embed = nn.Linear(d + 1, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, batch_first=True,
                                           norm_first=True, dropout=0.0, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        L = tok.shape[1]
        h = self.embed(tok) + self.pos[:, :L]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tok.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(h).squeeze(-1)[:, 0::2]          # one prediction per x-position


# ── the baselines ───────────────────────────────────────────────────────────────────────────────────────────────────
def ridge_curve(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-8) -> torch.Tensor:
    """The BAYES-OPTIMAL predictor for this prior, evaluated at every k: fit on the first k examples, predict the (k+1)-th.
    Solved in the DUAL (`w = Xᵀ(XXᵀ + λI)⁻¹y`) because k < d for most of the curve and the primal is singular there; the
    dual gives the minimum-norm solution, which is exactly the posterior mean under w ~ N(0, I)."""
    B, K, d = x.shape
    out = torch.zeros(B, K, device=x.device)
    for k in range(1, K):
        xs, ys = x[:, :k], y[:, :k]
        gram = xs @ xs.transpose(1, 2) + lam * torch.eye(k, device=x.device)
        w = xs.transpose(1, 2) @ torch.linalg.solve(gram, ys.unsqueeze(-1))
        out[:, k] = (x[:, k].unsqueeze(1) @ w).squeeze(-1).squeeze(-1)
    return out                                            # k=0 has no evidence, so the optimal guess is the prior mean 0


# ── training ────────────────────────────────────────────────────────────────────────────────────────────────────────
def train(setting: str, schedule: str, args, dev):
    torch.manual_seed(args.seed)
    fixed_w = torch.randn(1, args.d, device=dev) if setting == "memorise" else None
    model = Reader(args.d, args.d_model, args.n_layer, args.n_head, max_len=2 * args.k + 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))

    warm = max(1, args.steps // 20)
    def lr_at(s):                                          # linear warmup then cosine decay — the standard recipe; the
        if schedule == "constant":                         # ablation is whether it matters, so `constant` is a real arm
            return 1.0
        if s < warm:
            return s / warm
        p = (s - warm) / max(1, args.steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    t0 = time.time()
    for step in range(args.steps):
        x, y, _ = sample_batch(args.batch, args.k, args.d, dev, fixed_w, args.noise)
        pred = model(as_tokens(x, y))
        loss = ((pred - y) ** 2).mean()                    # every position is supervised: cheap, and it is the same task
        opt.zero_grad(set_to_none=True)                    #   at every k, which is what makes the curve readable
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, fixed_w, time.time() - t0


@torch.no_grad()
def evaluate(model, fixed_w, args, dev, n: int = 2048):
    """Normalised MSE per k, for the transformer and for the baselines. Var(y) = d, so dividing by d puts the
    ZERO-predictor at exactly 1.0 and makes the numbers comparable across dimensions."""
    x, y, _ = sample_batch(n, args.k, args.d, dev, fixed_w, args.noise)
    pred = model(as_tokens(x, y))
    tf = ((pred - y) ** 2).mean(0) / args.d
    rg = ((ridge_curve(x, y) - y) ** 2).mean(0) / args.d
    zero = (y ** 2).mean(0) / args.d
    return tf.cpu(), rg.cpu(), zero.cpu()


def report(name, tf, rg, zero, secs, args):
    print(f"\n=== {name}  ({secs:.0f}s) " + "=" * max(0, 52 - len(name)))
    ks = sorted({0, 1, 2, args.d // 2, args.d - 1, args.d, args.d + 2, args.k - 1})
    ks = [k for k in ks if 0 <= k < args.k]
    print("   k (examples in context) : " + "".join(f"{k:>8d}" for k in ks))
    print("   transformer  (norm MSE) : " + "".join(f"{tf[k]:>8.3f}" for k in ks))
    print("   ridge = Bayes-optimal   : " + "".join(f"{rg[k]:>8.3f}" for k in ks))
    print("   predict-zero (no learn) : " + "".join(f"{zero[k]:>8.3f}" for k in ks))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=8, help="input dimension")
    p.add_argument("--k", type=int, default=20, help="in-context examples per sequence")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default="all", choices=["all", "incontext"],
                   help="incontext = just the two schedule arms, for a seed sweep")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev} | d={args.d} k={args.k} steps={args.steps} batch={args.batch} "
          f"model={args.n_layer}L/{args.d_model}d")

    settings = ("memorise", "incontext") if args.arms == "all" else ("incontext",)
    for setting in settings:
        model, fw, secs = train(setting, "cosine", args, dev)
        report(f"{setting} + AdamW + cosine schedule [seed {args.seed}]", *evaluate(model, fw, args, dev), secs, args)

    model, fw, secs = train("incontext", "constant", args, dev)      # does the SCHEDULE earn its place?
    report(f"incontext + AdamW + constant LR (schedule ablated) [seed {args.seed}]",
           *evaluate(model, fw, args, dev), secs, args)


if __name__ == "__main__":
    main()
