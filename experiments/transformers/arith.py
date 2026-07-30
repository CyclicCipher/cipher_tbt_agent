"""THE DISCRETE-LOG PROBE: what do the weights of a MULTIPLICATION operator look like?

THE PREDICTION, derived rather than guessed, and stated before running. Nanda et al. 2023 (arXiv:2301.05217) reverse-
engineered modular ADDITION in a one-layer transformer: the embedding is sparse in the FOURIER basis, the MLP forms
products of trig terms, and the unembedding scores `cos(w(a+b-c))`, maximal at `c = a+b`. Chughtai, Chan & Nanda
(arXiv:2302.03025) generalised it — a network learning a GROUP operation implements it through the group's IRREDUCIBLE
REPRESENTATIONS, and the irreps of a cyclic group are exactly the Fourier modes. (Zhong et al., arXiv:2306.17844, add the
caution that the circuit is not unique: "clock" and "pizza" algorithms both occur.)

Now apply that to multiplication. The units mod `p` form a cyclic group of order `p-1` under multiplication, and the
isomorphism carrying it to addition is the DISCRETE LOGARITHM. So if the irrep account is right, a transformer trained on
modular multiplication must be sparse in the Fourier basis of the DISCRETE-LOG INDEX, not of the value index:

    embed(a) ~ sum_k c_k [ cos(w_k * log_g a), sin(w_k * log_g a) ]

i.e. "the weights of a multiplication operator" are `log -> add -> exp`. That is one training run and one FFT.

THE CROSSED DESIGN, which is what makes it a measurement rather than a fishing trip. Run the SAME pipeline on modular
ADDITION over the same domain. Prediction:

                        value-basis      log-basis
    trained on  +         SPARSE          dense
    trained on  x         dense           SPARSE

A pipeline that reported "log-basis is sparse" for both operations would be detecting something about the log reindexing,
not about the model. Both bases are FFTs of length `p-1` over the SAME p-1 embedding rows — inputs are restricted to the
units `1..p-1` for both operations precisely so the two spectra are directly comparable.

CONTROLS, because a Fourier peak can come from the structure of the DATA regardless of what the model computes, and this
line has already manufactured three false positives from its own measuring apparatus:
  * `--steps 0`   — an untrained model through the identical pipeline. Must be flat in both bases.
  * `--shuffle 1` — trained on RANDOM labels (pure memorisation). Must be flat in both bases.
  * CAUSAL ABLATION — correlational basis analysis is not evidence. Keeping ONLY the named frequencies must PRESERVE
    accuracy and keeping a random subspace of the SAME RANK must destroy it; removing the named frequencies must destroy
    accuracy and removing a matched random subspace must not. Rank is matched exactly: keeping `k` non-DC frequencies
    spans `2k+1` real dimensions of the row space, so the random control projects onto a random `2k+1`-dim subspace.

Usage:  python experiments/transformers/arith.py --op mul
        python experiments/transformers/arith.py --op add
        python experiments/transformers/arith.py --op mul --shuffle 1
        python experiments/transformers/arith.py --op mul --steps 0
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from h1_lid import Model


def factorise(n):
    fs, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def primitive_root(p):
    """A generator of the multiplicative group mod p — found by checking order, never hardcoded, since the whole probe
    is indexed by it."""
    fs = factorise(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fs):
            return g
    raise RuntimeError(f"no primitive root mod {p}")


def log_table(p, g):
    """`order[i]` = the unit whose discrete log is `i`; `dlog[x]` = the discrete log of `x`. Row `i` of the reindexed
    embedding is therefore the embedding of `g**i`."""
    order = [pow(g, i, p) for i in range(p - 1)]
    dlog = {x: i for i, x in enumerate(order)}
    return order, dlog


def build(p, op, dev, train_frac, seed, shuffle):
    """Every (a, b) with a, b in the UNITS 1..p-1 — the same domain for both operations, so the two spectra are FFTs of
    equal length over the same embedding rows."""
    a = torch.arange(1, p, device=dev).repeat_interleave(p - 1)
    b = torch.arange(1, p, device=dev).repeat(p - 1)
    c = {"mul": (a * b) % p, "add": (a + b) % p, "addm": (a + b) % (p - 1)}[op]
    if shuffle:                                     # the memorisation control: labels carry no group structure at all
        g = torch.Generator(device=dev).manual_seed(seed + 999)
        c = c[torch.randperm(len(c), generator=g, device=dev)]
    tok = torch.stack([a, b, torch.full_like(a, p)], dim=1)          # [a, b, EQ]; EQ is token id p
    g = torch.Generator(device=dev).manual_seed(seed)
    perm = torch.randperm(len(tok), generator=g, device=dev)
    cut = int(train_frac * len(tok))
    return (tok[perm[:cut]], c[perm[:cut]]), (tok[perm[cut:]], c[perm[cut:]])


@torch.no_grad()
def accuracy(model, data):
    tok, c = data
    return (model(tok)[:, -1].argmax(-1) == c).float().mean().item()


def rows(model, p, order):
    """The value-token embedding rows, in VALUE order (1..p-1) and in DISCRETE-LOG order (g^0, g^1, ...)."""
    E = model.emb.weight.detach()
    val = E[1:p]
    lg = E[torch.tensor(order, device=E.device)]
    return val, lg


def spectrum(E):
    """Normalised power per frequency, DC removed. The row mean is a constant offset shared by every element, which is
    not structure in the index, so it is subtracted rather than allowed to dominate the spectrum."""
    F_ = torch.fft.rfft((E - E.mean(0, keepdim=True)).float(), dim=0)
    power = (F_.abs() ** 2).sum(1)
    power[0] = 0.0
    return power / power.sum().clamp(min=1e-12)


def sparsity(power, k=5):
    """Two numbers rather than one. `top-k` is the share of power in the k strongest frequencies; the PARTICIPATION
    RATIO `1/sum(p^2)` is the EFFECTIVE NUMBER of frequencies in use and needs no choice of k — a uniform spectrum over
    n frequencies gives n, a single spike gives 1."""
    top = torch.topk(power, k)
    return top.values.sum().item(), 1.0 / (power ** 2).sum().clamp(min=1e-12).item(), top.indices.tolist()


def keep_freqs(E, freqs, keep=True):
    """Project the row space onto (or away from) the given frequencies. DC is always retained on both sides so the two
    conditions differ only in the frequencies under test."""
    F_ = torch.fft.rfft(E.float(), dim=0)
    mask = torch.zeros(F_.shape[0], device=E.device)
    mask[torch.tensor(freqs, device=E.device)] = 1.0
    if not keep:
        mask = 1.0 - mask
    mask[0] = 1.0
    return torch.fft.irfft(F_ * mask[:, None], n=E.shape[0], dim=0).to(E.dtype)


def real_rank(freqs, n):
    """The real dimension a set of rfft bins spans: a conjugate pair is 2 dimensions, but DC and (for even n) Nyquist
    are 1 each. Getting this wrong is how a rank-matched control silently stops being rank-matched."""
    return sum(1 if (f == 0 or (n % 2 == 0 and f == n // 2)) else 2 for f in freqs)


def random_basis(n, rank, dev, gen):
    """An orthonormal basis for a random `rank`-dim subspace ORTHOGONAL TO THE CONSTANT direction. Orthogonality to the
    constant is what lets DC be retained on BOTH sides of the comparison, so the two conditions differ only in which
    non-DC directions they use — otherwise 'keep random' is handicapped by also losing the row mean, and the gap it
    shows is partly that handicap rather than the circuit."""
    ones = torch.ones(n, 1, device=dev) / math.sqrt(n)
    A = torch.randn(n, rank, generator=gen, device=dev, dtype=torch.float32)
    return torch.linalg.qr(A - ones @ (ones.T @ A))[0]


@torch.no_grad()
def ablate(model, idx, freqs, data, seed):
    """Accuracy under four surgeries on the embedding rows ordered by `idx`, with the random controls matched to the
    NON-DC rank of the named frequencies and the row mean retained in every condition. Run in BOTH bases: the claim is
    not merely that the named frequencies matter, but that they matter in ONE basis and not the other."""
    E0 = model.emb.weight.detach().clone()
    lg = E0[idx].float()
    n = lg.shape[0]
    mu = lg.mean(0, keepdim=True)
    rank = real_rank(freqs, n)                             # non-DC real dimensions the named frequencies span
    Q = random_basis(n, rank, E0.device, torch.Generator(device=E0.device).manual_seed(seed))
    proj = Q @ (Q.T @ (lg - mu))
    out, made = {}, (("keep named", keep_freqs(lg, freqs, True)),
                     ("keep random", mu + proj),
                     ("drop named", keep_freqs(lg, freqs, False)),
                     ("drop random", lg - proj))
    for name, new in made:
        model.emb.weight[idx] = new.to(E0.dtype)
        out[name] = accuracy(model, data)
        model.emb.weight.copy_(E0)
    out["rank"] = rank
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    # `addm` is the EXACTLY-MATCHED crossed control. Addition mod p over the units is a group of order p, sampled at
    # only p-1 points, so a length-(p-1) FFT of its embedding leaks across bins and would understate its value-basis
    # sparsity. Addition mod (p-1) over a=1..p-1 is a clean group operation on Z_{p-1} (96 maps to 0, and a cyclic
    # relabelling of rows leaves the power spectrum unchanged) — the SAME order as the multiplicative group, so both
    # operations live on isomorphic groups and differ only in whether the natural coordinate is the value or its log.
    ap.add_argument("--op", default="mul", choices=["mul", "add", "addm"])
    ap.add_argument("--steps", type=int, default=20000, help="0 = report the UNTRAINED model's spectra (the control)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0, help="strong decay is what drives the generalising solution")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--shuffle", type=int, default=0, help="1 = random labels (the memorisation control)")
    ap.add_argument("--k", type=int, default=5, help="frequencies named as the circuit, for sparsity and ablation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--trace", type=int, default=0, help="print train/test/sparsity every N steps (0 = off)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    p = args.p

    g = primitive_root(p)
    order, dlog = log_table(p, g)
    assert sorted(order) == list(range(1, p)), "the log table must enumerate every unit exactly once"
    assert all(order[(dlog[x] + dlog[y]) % (p - 1)] == (x * y) % p for x in (2, 3, 5) for y in (7, 11, 13)), \
        "discrete log must carry multiplication to addition — the premise of the whole probe"

    train, test = build(p, args.op, dev, args.train_frac, args.seed, args.shuffle)
    print(f"device {dev} | p={p} primitive root g={g} | op={args.op} shuffle={bool(args.shuffle)} | "
          f"{len(train[0])} train / {len(test[0])} test pairs | d_model={args.d_model} 1 layer | steps={args.steps}")

    model = Model(d_model=args.d_model, n_layer=1, n_head=4, max_len=8, pos="learned", n_vocab=p + 1).to(dev)
    if args.steps > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98),
                                fused=(dev == "cuda"))
        warm = max(1, args.steps // 50)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 1.0)
        # Full batch with a fixed shape is the ideal case for `torch.compile`; the same compile+bf16 pairing measured
        # 3.1x elsewhere in this line. Master weights stay fp32 under autocast, and the embedding read by the analysis
        # below is never touched in reduced precision.
        trainer = torch.compile(model) if args.compile and dev == "cuda" else model
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.bf16 and dev == "cuda"))
        t0 = time.time()
        for s in range(args.steps):
            with amp:
                logits = trainer(train[0])[:, -1]                          # FULL BATCH, as the grokking setup requires
            loss = F.cross_entropy(logits.float(), train[1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if args.trace and (s + 1) % args.trace == 0:
                print(f"   step {s + 1:>6}  train {accuracy(model, train):.3f}  test {accuracy(model, test):.3f}  "
                      f"eff.freqs(log) {sparsity(spectrum(rows(model, p, order)[1]), args.k)[1]:.1f}")
        print(f"trained in {time.time() - t0:.0f}s | final train loss {loss.item():.4f}")
    tr, te = accuracy(model, train), accuracy(model, test)
    print(f"train acc {tr:.3f} | TEST acc {te:.3f}")

    val, lg = rows(model, p, order)
    bases = (("value index", torch.arange(1, p, device=dev), val),
             ("DISCRETE LOG", torch.tensor(order, device=dev), lg))
    print(f"\n{'basis':<14}{'top-'+str(args.k)+' power':>14}{'eff. freqs':>13}   strongest frequencies")
    named = {}
    for name, _idx, E in bases:
        top, pr, idxs = sparsity(spectrum(E), args.k)
        named[name] = idxs
        print(f"{name:<14}{top:>14.3f}{pr:>13.1f}   {idxs}")
    print(f"(a uniform spectrum over {(p - 1) // 2} frequencies would give eff. freqs ~{(p - 1) // 2}, a single spike 1)")

    if args.steps > 0 and te > 0.5:
        print(f"\nCAUSAL ABLATION in BOTH bases - the top {args.k} frequencies of each, random controls matched to the")
        print("same non-DC rank, row mean retained in every condition.")
        print(f"   {'basis':<14}{'keep named':>12}{'keep random':>13}{'drop named':>12}{'drop random':>13}")
        for name, idx, _E in bases:
            r = ablate(model, idx, named[name], test, args.seed)
            print(f"   {name:<14}{r['keep named']:>12.3f}{r['keep random']:>13.3f}"
                  f"{r['drop named']:>12.3f}{r['drop random']:>13.3f}   (rank {r['rank']})")
        print("   CLAIM: in the operation's OWN basis, keep named >> keep random and drop named << drop random; in the")
        print("   other basis, keep named ~ keep random. A test that fired in both bases would be measuring nothing.")
    elif args.steps > 0:
        print("\nAblation SKIPPED: test accuracy is at or below chance, so there is no circuit to ablate.")


if __name__ == "__main__":
    main()
