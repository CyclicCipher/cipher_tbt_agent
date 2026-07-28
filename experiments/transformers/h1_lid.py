"""H1 — is "locally in-distribution" even the right explanatory variable for speed of skill acquisition?

THE CLAIM UNDER TEST (Zhang, alexzhang13.github.io/blog/2026/harness/): a task can be globally out-of-distribution and
still be acquired quickly, provided every LOCAL operation in it is in-distribution. If that is right, then how fast a
system picks up a novel task should be predicted by the familiarity of its PARTS, not by the novelty of the whole.

HOW THIS IS MADE FALSIFIABLE. The obvious version of this experiment — train on some tasks, test on others, observe that
the far-away ones are harder — measures distribution shift and would "confirm" LID no matter what. So GLOBAL NOVELTY IS
HELD CONSTANT: every test task is a composition that was NEVER trained on, so all of them are equally novel as wholes.
The only thing that varies across test tasks is how familiar their two constituent operations are, and that is arranged by
training the primitives at frequencies spanning ~150x. If acquisition speed does NOT track local familiarity, LID is not
the explanatory variable here and the rest of the ladder in BEE.md is moot. That outcome is the reason this runs first.

WHAT "LOCALLY IN-DISTRIBUTION" IS MEASURED BY. Not token edit distance — "in-distribution" is a fact about the MODEL, not
about the strings. It is measured as the model's own error on the local operation (`feedback_epistemic_value_is_prediction_error`),
read off TRAINED compositions containing that primitive, so the familiarity measurement never touches the test tasks.

THE DEPENDENT MEASURE IS TRIALS-TO-CRITERION — the number of in-context demonstrations needed before the model solves the
query — because that is the bee measure (BEE.md) and the quantity ARC scores. Not final accuracy.

Task domain: sequences of L digits; a primitive is a bijection on them (reverse, rotate, swap pairs/halves, +1, -1, x2);
a task is an ordered composition of two primitives, presented as in-context (input, output) demonstrations.

Usage:  python experiments/transformers/h1_lid.py
        python experiments/transformers/h1_lid.py --steps 8000 --seed 1
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

L, V = 4, 6            # sequence length, digit vocabulary. Kept SMALL on purpose: the point of H1 is whether
#                        acquisition speed tracks local familiarity, and that needs a model that has actually
#                        learned the training distribution inside a runnable budget. A domain the model cannot
#                        fit censors every task at the same value and measures nothing (see the sanity gate).


# ── the primitives: each a bijection on a batch of digit sequences ───────────────────────────────────────────────────
PRIMS = {
    "reverse":     lambda s: s.flip(-1),
    "rot_left":    lambda s: torch.roll(s, -1, dims=-1),
    "rot_right":   lambda s: torch.roll(s, 1, dims=-1),
    "swap_pairs":  lambda s: s.reshape(*s.shape[:-1], L // 2, 2).flip(-1).reshape(*s.shape),
    "swap_halves": lambda s: torch.roll(s, L // 2, dims=-1),
    "inc":         lambda s: (s + 1) % V,
    "dec":         lambda s: (s - 1) % V,
    "double":      lambda s: (s * 2) % V,
}
NAMES = list(PRIMS)


def apply_pair(s, pair):
    return PRIMS[NAMES[pair[1]]](PRIMS[NAMES[pair[0]]](s))


def signature(pair, probe):
    """What a composition DOES, as a hashable fingerprint. Two pairs with the same signature are the same function however
    differently they are written — `(rot_left, rot_right)` is the identity, and `(inc, inc)` is not any single primitive.
    Used to keep a 'held-out' task from being secretly identical to a trained one, which would make it trivially easy and
    quietly corrupt the whole measurement."""
    return tuple(apply_pair(probe, pair).flatten().tolist())


def build_tasks(seed: int):
    """All non-degenerate 2-compositions, split into TRAIN and HELD-OUT, with per-primitive training frequencies spanning
    ~150x so that local familiarity is GRADED rather than binary."""
    g = torch.Generator().manual_seed(seed)
    probe = torch.randint(0, V, (16, L), generator=g)
    ident = signature((NAMES.index("inc"), NAMES.index("dec")), probe)   # any pair equal to this is a no-op task

    by_sig, pairs = {}, []
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            sig = signature((i, j), probe)
            if sig == ident or sig in by_sig:        # drop no-ops and functional duplicates: keep one writing of each map
                continue
            by_sig[sig] = (i, j)
            pairs.append((i, j))

    order = torch.randperm(len(pairs), generator=g).tolist()
    n_test = max(8, len(pairs) // 4)
    test = [pairs[k] for k in order[:n_test]]
    train = [pairs[k] for k in order[n_test:]]

    weights = torch.tensor([1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.007])   # ~150x spread over the primitives
    weights = weights[torch.randperm(len(NAMES), generator=g)]
    w_pair = torch.tensor([weights[i] * weights[j] for i, j in train])
    return train, test, w_pair / w_pair.sum(), weights


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def make_batch(B, K, task_list, probs, dev, g=None, fixed=None):
    """B sequences of K (input, output) demonstrations of ONE task each. The task is re-drawn per sequence, so the model
    cannot memorise any particular map — it must infer the rule from the demonstrations it has already seen."""
    if fixed is not None:
        idx = torch.full((B,), fixed, dtype=torch.long)
    else:
        idx = torch.multinomial(probs, B, replacement=True, generator=g)
    x = torch.randint(0, V, (B, K, L), generator=g)
    y = torch.empty_like(x)
    for t in idx.unique():
        m = idx == t
        y[m] = apply_pair(x[m], task_list[int(t)])
    tok = torch.stack([x, y], dim=2).reshape(B, K * 2 * L)     # in,out,in,out,… ; roles are fixed by position
    return tok.to(dev), idx


class Model(nn.Module):
    """A small causal decoder over digits. Predicts every OUTPUT block from everything before it, so the accuracy at the
    j-th block IS the accuracy after j-1 demonstrations — the trials curve falls out of one forward pass."""

    def __init__(self, d_model=96, n_layer=3, n_head=4, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(V, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, batch_first=True,
                                           norm_first=True, dropout=0.0, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, V)

    def forward(self, tok):
        n = tok.shape[1]
        h = self.emb(tok) + self.pos[:, :n]
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=tok.device)
        return self.head(self.blocks(h, mask=mask, is_causal=True))


def out_mask(K, dev):
    """True at positions holding an OUTPUT digit — the only places a prediction is scored."""
    m = torch.zeros(K * 2 * L, dtype=torch.bool, device=dev)
    for k in range(K):
        m[k * 2 * L + L: (k + 1) * 2 * L] = True
    return m


def loss_of(model, tok, K):
    logits = model(tok)[:, :-1]                                # predict position t+1 from t
    tgt, m = tok[:, 1:], out_mask(K, tok.device)[1:]
    ce = F.cross_entropy(logits[:, m].reshape(-1, V), tgt[:, m].reshape(-1), reduction="none")
    return ce.reshape(tok.shape[0], K, L)                      # per-sequence, per-demonstration, per-digit


@torch.no_grad()
def per_demo_exact(model, tok, K):
    """Exact-match accuracy of the whole output block, per demonstration index. Exact match, not per-digit: the skill is
    acquired when the answer is RIGHT, and partial credit would hide that."""
    pred = model(tok)[:, :-1].argmax(-1)
    tgt, m = tok[:, 1:], out_mask(K, tok.device)[1:]
    ok = (pred[:, m] == tgt[:, m]).reshape(tok.shape[0], K, L)
    return ok.all(-1).float().mean(0)


# ── the two measurements ────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def familiarity(model, train, probs, dev, K, n=256):
    """LOCAL familiarity per primitive: the model's mean error on TRAINED compositions containing it. Measured only on
    training tasks, so nothing about the held-out tasks leaks into the predictor."""
    err = {i: [] for i in range(len(NAMES))}
    for t, pair in enumerate(train):
        tok, _ = make_batch(n, K, train, probs, dev, fixed=t)
        e = loss_of(model, tok, K).mean().item()
        err[pair[0]].append(e)
        err[pair[1]].append(e)
    return {i: sum(v) / len(v) for i, v in err.items() if v}   # lower = more locally in-distribution


@torch.no_grad()
def trials_to_criterion(model, test, dev, K, crit=0.8, n=512):
    """The bee measure: how many demonstrations before the model gets the query right at least `crit` of the time. `K`
    (censored) means it never got there inside the context we gave it."""
    out = []
    for t, pair in enumerate(test):
        tok, _ = make_batch(n, K, test, torch.ones(len(test)), dev, fixed=t)
        acc = per_demo_exact(model, tok, K)
        hit = (acc >= crit).nonzero()
        out.append((pair, int(hit[0]) if len(hit) else K, float(acc[-1])))
    return out


def spearman(a, b):
    """Rank correlation, written out so the experiment needs no scipy. Rank-based because the prediction is about ORDER —
    less familiar parts ⇒ more trials — not about a linear relationship."""
    def rank(v):
        """AVERAGE ranks for ties. Assigning distinct ranks to tied values is not a rounding detail: with every task
        censored at the same trial count, it ranked eight identical numbers 0..7 in whatever order they were passed and
        reported rho = +1.000 on data that contained no signal whatsoever. A false positive manufactured by the metric."""
        order = sorted(range(len(v)), key=lambda i: v[i])
        r, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=6, help="demonstrations per sequence")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crit", type=float, default=0.8)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train, test, probs, weights = build_tasks(args.seed)
    print(f"device {dev} | {len(train)} trained compositions, {len(test)} held out | K={args.k} steps={args.steps}")
    print("primitive training weight: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, weights.tolist())))

    model = Model(max_len=args.k * 2 * L + 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))

    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        tok, _ = make_batch(args.batch, args.k, train, probs, dev, g)
        loss = loss_of(model, tok, args.k).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"trained in {time.time() - t0:.0f}s | final train loss {loss.item():.4f}")

    # SANITY GATE. Held-out numbers mean nothing until the model can do the tasks it WAS trained on: if it solves none of
    # those either, every held-out task is censored at the same value and the correlation is measuring nothing. The first
    # run of this experiment reported a perfect correlation in exactly that state, so the gate is not optional.
    seen = trials_to_criterion(model, train, dev, args.k, args.crit, n=256)
    solved = [r for r in seen if r[1] < args.k]
    print(f"\nSANITY: the model solves {len(solved)}/{len(seen)} TRAINED compositions to criterion "
          f"(mean final acc {sum(r[2] for r in seen) / len(seen):.2f})")
    if not solved:
        print("   => it has not learned the training distribution, so the held-out measurement below is UNINTERPRETABLE.")
        print("      Fix the training run before reading anything into it (more steps / larger batch / easier domain).")

    fam = familiarity(model, train, probs, dev, args.k)
    print("\nLOCAL familiarity (mean error on TRAINED compositions containing it; lower = more in-distribution)")
    for i in sorted(fam, key=lambda i: fam[i]):
        print(f"   {NAMES[i]:<12} weight {weights[i]:.3f}   local error {fam[i]:.4f}")

    rows = trials_to_criterion(model, test, dev, args.k, args.crit)
    print(f"\nHELD-OUT compositions — global novelty is IDENTICAL for all of them (none was ever trained)")
    print(f"   {'composition':<26}{'worst local err':>16}{'trials':>9}{'final acc':>11}")
    xs, ys = [], []
    for pair, trials, final in sorted(rows, key=lambda r: max(fam[r[0][0]], fam[r[0][1]])):
        worst = max(fam[pair[0]], fam[pair[1]])
        name = f"{NAMES[pair[0]]}>{NAMES[pair[1]]}"
        print(f"   {name:<26}{worst:>16.4f}{trials:>9d}{final:>11.2f}")
        xs.append(worst)
        ys.append(trials)

    accs = [r[2] for r in sorted(rows, key=lambda r: max(fam[r[0][0]], fam[r[0][1]]))]
    censored = all(y >= args.k for y in ys)
    rho = spearman(xs, ys)
    print(f"\nPRIMARY  Spearman(worst local error, trials-to-criterion) = {rho:+.3f}   over {len(xs)} held-out tasks")
    if censored:
        print("   !! EVERY held-out task is CENSORED at the context limit, so the primary measure carries no information.")
        print("      A rank correlation over identical values says nothing about LID -- it says the model is too weak.")
        print(f"   SECONDARY  Spearman(worst local error, FINAL accuracy) = {spearman(xs, accs):+.3f}"
              "   (graded, so it can still separate the tasks)")
        print("      Reported as secondary BECAUSE the primary was censored -- not chosen after seeing which one looked better.")
    else:
        print("LID predicts a POSITIVE correlation: less familiar parts => more trials, global novelty constant.")
        print("A correlation near zero says local familiarity is NOT what governs acquisition speed here.")


if __name__ == "__main__":
    main()
