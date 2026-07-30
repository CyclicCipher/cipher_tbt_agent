"""THE ARITHMETIC ENGINE: one model, both operations, and the question of HOW IT SWITCHES.

WHAT `arith.py` SETTLED. A single operation is a change of basis in which it becomes coordinate-wise. Modular
multiplication is sparse in the Fourier basis of the DISCRETE-LOG index (6.6 effective frequencies of 48) and flat in the
value index; addition is the exact mirror. Causally: 11 of 96 row-dimensions in the operation's own basis carry ~0.89
accuracy where a rank-matched random subspace carries 0.015.

THE PRIOR, stated before running, because it decides what is worth measuring. **There is no single basis in which
addition and multiplication are simultaneously diagonal** — the additive and multiplicative structures of a field are
genuinely different, and the map between their diagonalising bases is the discrete logarithm, which is not linear. So a
model trained on both CANNOT share one coordinate system. It must carry BOTH, and the interesting question is not
whether it duplicates the representation but HOW IT SELECTS between them.

Note what this predicts about "recruiting old circuitry": here duplication is forced by mathematics, not chosen by the
optimiser. That is worth knowing as a boundary on the recruit-versus-duplicate framing — some pairs of operations have no
shared basis to recruit, and the cost is then unavoidable.

FIVE MEASUREMENTS. The first two are correlational, the last three causal, and it is the causal ones that answer the
question.

  1. PER-OPERATION accuracy. Both must generalise or nothing below is interpretable.
  2. SPECTRA in both bases. Prediction: BOTH sparse now, where a single-op model was sparse in one and flat in the other.
     That is the signature of carrying two bases at once.
  3. PER-BASIS, PER-OPERATION ABLATION — the duplication test. Removing the discrete-log frequencies must destroy
     multiplication and SPARE addition; removing the value-index frequencies must do the reverse. Rank-matched random
     controls throughout, row mean retained, exactly as in `arith.py`.
  4. MLP NEURON SELECTIVITY — the switch itself, if it lives in the MLP. Each neuron gets a selectivity index
     `(m_add - m_mul)/(m_add + m_mul)` over its activation magnitude at the answer position. Take the most add-selective
     and most mul-selective deciles (matched sizes) and ablate each: if the populations are disjoint and each ablation
     kills only its own operation, the switch is neuron selection. A matched-size RANDOM population is the control,
     because ablating any 10% of neurons hurts.
  5. OPERATION-TOKEN SWAP — the cleanest test of whether the token is a pure selector. Feed `+` where the answer is a
     product, and ask not merely whether accuracy drops but whether the model now outputs the OTHER operation's answer.
     A pure selector flips the output; a token that merely perturbs a shared computation degrades it into neither.

DOMAIN. Inputs are the units 1..p-1 for both operations, so both spectra are FFTs over the SAME p-1 embedding rows and
are directly comparable — the same choice `arith.py` made. Addition is taken mod (p-1) by default so that both operations
live on cyclic groups of the SAME ORDER, which removes the confound of two different frequency sets being trivially
distinguishable by their group order. `--add_mod_p` switches to addition mod p if the standard task is wanted, at the
cost of spectral leakage in the value basis.

Usage:  python experiments/transformers/engine.py
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from arith import (accuracy, keep_freqs, log_table, primitive_root, random_basis, real_rank, spectrum, sparsity)
from h1_lid import Model


def build(p, dev, train_frac, seed, add_mod_p):
    """Every (a, op, b) over the units, both operations. Token ids: 0..p-1 values, p = PLUS, p+1 = TIMES, p+2 = EQ."""
    a = torch.arange(1, p, device=dev).repeat_interleave(p - 1)
    b = torch.arange(1, p, device=dev).repeat(p - 1)
    add = (a + b) % (p if add_mod_p else p - 1)
    mul = (a * b) % p
    eq = torch.full_like(a, p + 2)
    tok = torch.cat([torch.stack([a, torch.full_like(a, p), b, eq], dim=1),
                     torch.stack([a, torch.full_like(a, p + 1), b, eq], dim=1)])
    c = torch.cat([add, mul])
    is_mul = torch.cat([torch.zeros_like(add, dtype=torch.bool), torch.ones_like(mul, dtype=torch.bool)])
    g = torch.Generator(device=dev).manual_seed(seed)
    perm = torch.randperm(len(tok), generator=g, device=dev)
    cut = int(train_frac * len(tok))
    tr, te = perm[:cut], perm[cut:]
    def split(ix):
        return (tok[ix], c[ix]), (tok[ix][~is_mul[ix]], c[ix][~is_mul[ix]]), (tok[ix][is_mul[ix]], c[ix][is_mul[ix]])
    return split(tr), split(te)


class Neurons:
    """A hook on the MLP non-linearity: captures the post-activation 'neurons' and can mask them for ablation. Reading
    and ablating through one object keeps the measurement and the surgery on exactly the same units."""

    def __init__(self, model):
        self.act, self.mask = None, None
        model.blocks[0].mlp[1].register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out):
        self.act = out.detach()
        return None if self.mask is None else out * self.mask


@torch.no_grad()
def selectivity(model, nrn, add_data, mul_data):
    """Per-neuron `(m_add - m_mul)/(m_add + m_mul)` on activation MAGNITUDE at the answer position. +1 = purely
    additive, -1 = purely multiplicative, 0 = shared."""
    mags = []
    for data in (add_data, mul_data):
        model(data[0])
        mags.append(nrn.act[:, -1].abs().mean(0).float())
    add, mul = mags
    return (add - mul) / (add + mul).clamp(min=1e-9)


@torch.no_grad()
def ablate_neurons(model, nrn, which, add_data, mul_data):
    d = nrn.act.shape[-1]
    m = torch.ones(d, device=which.device)
    m[which] = 0.0
    nrn.mask = m
    out = (accuracy(model, add_data), accuracy(model, mul_data))
    nrn.mask = None
    return out


@torch.no_grad()
def ablate_basis(model, idx, freqs, add_data, mul_data, seed):
    """Keep-only and drop, each against a rank-matched random subspace, measuring BOTH operations.

    KEEP is what separates a real loss of sparsity from an artifact of superposition. The value-index SPECTRUM of this
    model necessarily contains addition's sparse structure PLUS multiplication's structure viewed in the wrong basis,
    which is diffuse — so the participation ratio must rise even if addition's own circuit is untouched. Asking whether
    the top few frequencies still SUFFICE answers the question the spectrum cannot."""
    E0 = model.emb.weight.detach().clone()
    rows = E0[idx].float()
    n = rows.shape[0]
    mu = rows.mean(0, keepdim=True)
    rank = real_rank(freqs, n)
    Q = random_basis(n, rank, E0.device, torch.Generator(device=E0.device).manual_seed(seed))
    proj = Q @ (Q.T @ (rows - mu))
    out = {"rank": rank}
    for name, new in (("keep named", keep_freqs(rows, freqs, True)),
                      ("keep random", mu + proj),
                      ("drop named", keep_freqs(rows, freqs, False)),
                      ("drop random", rows - proj)):
        model.emb.weight[idx] = new.to(E0.dtype)
        out[name] = (accuracy(model, add_data), accuracy(model, mul_data))
        model.emb.weight.copy_(E0)
    return out


@torch.no_grad()
def swap_ops(model, p, test_add, test_mul, add_mod_p):
    """Replace the operation token and ask what the model computes. Reports accuracy against the token's OWN operation
    (what it was just told to do) and against the operation the operands were drawn for."""
    out = {}
    for name, (tok, _c), other_op in (("+ -> x", test_add, p + 1), ("x -> +", test_mul, p)):
        t = tok.clone()
        t[:, 1] = other_op
        a, b = t[:, 0], t[:, 2]
        told = (a * b) % p if other_op == p + 1 else (a + b) % (p if add_mod_p else p - 1)
        orig = (a + b) % (p if add_mod_p else p - 1) if other_op == p + 1 else (a * b) % p
        pred = model(t)[:, -1].argmax(-1)
        out[name] = ((pred == told).float().mean().item(), (pred == orig).float().mean().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.1, help="decile size for the selective neuron populations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--add_mod_p", type=int, default=0)
    ap.add_argument("--trace", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    p = args.p

    g = primitive_root(p)
    order, _dlog = log_table(p, g)
    (train, tr_add, tr_mul), (test, te_add, te_mul) = build(p, dev, args.train_frac, args.seed, args.add_mod_p)
    print(f"device {dev} | p={p} g={g} | + is mod {p if args.add_mod_p else p - 1}, x is mod {p} | "
          f"{len(train[0])} train / {len(test[0])} test | d_model={args.d_model} 1 layer | steps={args.steps}")

    model = Model(d_model=args.d_model, n_layer=1, n_head=4, max_len=8, pos="learned", n_vocab=p + 3).to(dev)
    nrn = Neurons(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = max(1, args.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 1.0)
    trainer = torch.compile(model) if args.compile and dev == "cuda" else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.bf16 and dev == "cuda"))
    t0 = time.time()
    for s in range(args.steps):
        with amp:
            logits = trainer(train[0])[:, -1]
        loss = F.cross_entropy(logits.float(), train[1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if args.trace and (s + 1) % args.trace == 0:
            E = model.emb.weight.detach()
            print(f"   step {s + 1:>6}  test + {accuracy(model, te_add):.3f}  x {accuracy(model, te_mul):.3f}  "
                  f"eff.freqs val {sparsity(spectrum(E[1:p]), args.k)[1]:.1f} "
                  f"log {sparsity(spectrum(E[torch.tensor(order, device=dev)]), args.k)[1]:.1f}")
    print(f"trained in {time.time() - t0:.0f}s | final train loss {loss.item():.4f}")
    print(f"\n(1) ACCURACY   train + {accuracy(model, tr_add):.3f}  x {accuracy(model, tr_mul):.3f}   "
          f"TEST + {accuracy(model, te_add):.3f}  x {accuracy(model, te_mul):.3f}")
    if min(accuracy(model, te_add), accuracy(model, te_mul)) < 0.5:
        print("!! one operation did not generalise -- everything below is UNINTERPRETABLE, not a result.")

    E = model.emb.weight.detach()
    bases = (("value index", torch.arange(1, p, device=dev), E[1:p]),
             ("DISCRETE LOG", torch.tensor(order, device=dev), E[torch.tensor(order, device=dev)]))
    print(f"\n(2) SPECTRA    {'top-'+str(args.k):>10}{'eff. freqs':>13}   strongest")
    named = {}
    for name, _idx, rows_ in bases:
        top, pr, idxs = sparsity(spectrum(rows_), args.k)
        named[name] = idxs
        print(f"    {name:<14}{top:>8.3f}{pr:>13.1f}   {idxs}")
    print(f"    (uniform over {(p - 1) // 2} frequencies -> {(p - 1) // 2}; a single spike -> 1. A single-op model was")
    print("     sparse in ONE basis and flat in the other; two sparse spectra means two bases are being carried.)")

    print(f"\n(3) BASIS ABLATION -- top {args.k} frequencies of each basis, rank-matched random controls")
    print(f"    {'basis':<14}{'':>4}{'keep named':>12}{'keep random':>13}{'drop named':>12}{'drop random':>13}")
    for name, idx, _rows in bases:
        r = ablate_basis(model, idx, named[name], te_add, te_mul, args.seed)
        for j, op in enumerate(("+", "x")):
            lbl = f"{name}" if j == 0 else ""
            print(f"    {lbl:<14}{op:>4}{r['keep named'][j]:>12.3f}{r['keep random'][j]:>13.3f}"
                  f"{r['drop named'][j]:>12.3f}{r['drop random'][j]:>13.3f}"
                  f"{('   (rank ' + str(r['rank']) + ')') if j == 0 else ''}")
    print("    DUPLICATION: dropping the LOG frequencies kills x and spares +; the VALUE frequencies do the reverse.")
    print("    SPARSITY: if KEEPING only the top few still carries the operation, that circuit is as sparse as it was")
    print("    ALONE, and the risen participation ratio is superposition of the other basis rather than degradation.")

    sel = selectivity(model, nrn, te_add, te_mul)
    d_mlp = sel.shape[0]
    n_sel = max(1, int(args.frac * d_mlp))
    add_pop, mul_pop = torch.topk(sel, n_sel).indices, torch.topk(-sel, n_sel).indices
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rand_pop = torch.randperm(d_mlp, generator=gen, device=dev)[:n_sel]
    print(f"\n(4) MLP NEURONS  {d_mlp} total; selectivity (add - mul)/(add + mul) at the answer position")
    print(f"    |s| > 0.5: {int((sel.abs() > 0.5).sum())}   > 0.9: {int((sel.abs() > 0.9).sum())}   "
          f"< 0.1 (shared): {int((sel.abs() < 0.1).sum())}   median |s| {sel.abs().median():.3f}")
    print(f"    {'ablate':<22}{'+ acc':>9}{'x acc':>9}")
    for label, pop in ((f"top {n_sel} ADD-selective", add_pop), (f"top {n_sel} MUL-selective", mul_pop),
                       (f"{n_sel} random (control)", rand_pop)):
        a, m = ablate_neurons(model, nrn, pop, te_add, te_mul)
        print(f"    {label:<22}{a:>9.3f}{m:>9.3f}")
    print("    CLAIM: if the switch is neuron selection, each population's ablation kills ONLY its own operation and")
    print("    the random control kills neither. If both ablations hurt both, the computation is shared and the switch")
    print("    is elsewhere -- in which case (3) and (5) say where.")

    print("\n(5) OPERATION-TOKEN SWAP -- feed the other op token and see what it computes")
    print(f"    {'swap':<10}{'matches the TOKEN':>20}{'matches the OPERANDS-original':>32}")
    for name, (told, orig) in swap_ops(model, p, te_add, te_mul, args.add_mod_p).items():
        print(f"    {name:<10}{told:>20.3f}{orig:>32.3f}")
    print("    CLAIM: a PURE SELECTOR follows the token -- accuracy against the token's own operation stays high and")
    print("    against the original collapses. A token that merely perturbs a shared computation gives neither.")


if __name__ == "__main__":
    main()
