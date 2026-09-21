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

from aurora import aurora

L, V = 6, 5            # sequence length, digit vocabulary. Kept SMALL on purpose: the point of H1 is whether
#                        acquisition speed tracks local familiarity, and that needs a model that has actually
#                        learned the training distribution inside a runnable budget. A domain the model cannot
#                        fit censors every task at the same value and measures nothing (see the sanity gate).


# ── the primitives: each a bijection on a batch of digit sequences ───────────────────────────────────────────────────
# SEVEN, at L=6/V=5, CHOSEN BY COUNTING rather than by taste: the shrink to 6 primitives at L=4 collapsed 36 writings
# to just 14 distinct functions, because rotations form a small cyclic group and most compositions coincided — which
# left 6 training tasks against 8 held-out ones, i.e. the split inverted. This config yields 25 distinct compositions
# (17 train / 8 test). Distinct-function count, not primitive count, is what the design actually needs.
# `double` was DELETED, and it was a real flaw rather than a trim: x*2 mod V is a bijection only when gcd(2,V)=1, so at
# V=6 it sent {0,1,2,3,4,5} to {0,2,4,0,2,4} and was not invertible at all, while the docstring claimed every primitive
# was a bijection. `negate` replaces it — x -> -x mod V, which genuinely is one.
PRIMS = {
    "reverse":     lambda s: s.flip(-1),
    "rot_left":    lambda s: torch.roll(s, -1, dims=-1),
    "rot_right":   lambda s: torch.roll(s, 1, dims=-1),
    "swap_pairs":  lambda s: s.reshape(*s.shape[:-1], L // 2, 2).flip(-1).reshape(*s.shape),
    "swap_halves": lambda s: torch.roll(s, L // 2, dims=-1),
    "inc":         lambda s: (s + 1) % V,
    "negate":      lambda s: (V - s) % V,
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
    ident = tuple(probe.flatten().tolist())      # the do-nothing signature, taken directly rather than via some pair of
    #                                              primitives that happen to be inverses — which broke the moment one of
    #                                              those primitives was removed, and would break again on any edit here

    by_sig, pairs = {}, []
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            sig = signature((i, j), probe)
            if sig == ident or sig in by_sig:        # drop no-ops and functional duplicates: keep one writing of each map
                continue
            by_sig[sig] = (i, j)
            pairs.append((i, j))

    # Split, with the guard that EVERY primitive must appear in the training half. Without it a primitive can end up only
    # in held-out tasks, and then it has no measured familiarity at all — which is not a missing number but a crash, and
    # before that guard existed it was a KeyError in the middle of the results table.
    n_test = 8
    for _ in range(200):
        order = torch.randperm(len(pairs), generator=g).tolist()
        test = [pairs[k] for k in order[:n_test]]
        train = [pairs[k] for k in order[n_test:]]
        if {i for pr in train for i in pr} == set(range(len(NAMES))):
            break
    else:
        raise RuntimeError("could not split with every primitive represented in training")

    # A geometric spread SIZED TO THE PRIMITIVE COUNT, so the graded exposure the whole experiment depends on survives a
    # change to that count. A hardcoded 8-long list silently used only its first 6 entries when the set shrank, quietly
    # halving the spread that makes local familiarity graded rather than binary.
    n = len(NAMES)
    weights = torch.tensor([150.0 ** (-i / (n - 1)) for i in range(n)])        # ~150x from most to least frequent
    weights = weights[torch.randperm(n, generator=g)]
    w_pair = torch.tensor([weights[i] * weights[j] for i, j in train])
    return train, test, w_pair / w_pair.sum(), weights


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def make_batch(B, K, task_list, probs, dev, g=None, fixed=None):
    """B sequences of K (input, output) demonstrations of ONE task each. The task is re-drawn per sequence, so the model
    cannot memorise any particular map — it must infer the rule from the demonstrations it has already seen."""
    if fixed is not None:
        idx = torch.full((B,), fixed, dtype=torch.long, device=dev)
    else:
        idx = torch.multinomial(probs.to(dev), B, replacement=True, generator=g)
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    y = torch.empty_like(x)
    for t in idx.unique():
        m = idx == t
        y[m] = apply_pair(x[m], task_list[int(t)])
    tok = torch.stack([x, y], dim=2).reshape(B, K * 2 * L)     # in,out,in,out,… ; roles are fixed by position
    return tok, idx                                            # already on the device: the old version built every batch
    #                                                            on the CPU and copied it across on every single step


def _freqs(hd, dev, base=10000.0):
    """Per-channel angular frequencies. RoPE runs over d/2 PAIRS, PoPE over all d channels — that difference is part of
    the method, not a detail. Decaying exponents, so wavelengths span ~2π to ~2π·base."""
    return base ** (-torch.arange(hd, device=dev, dtype=torch.float32) / hd)


class Attn(nn.Module):
    """Causal self-attention with a swappable positional scheme, because position has to reach the QK product for RoPE and
    PoPE and `nn.TransformerEncoderLayer` gives no way in.

      learned — no position here; a learned absolute embedding is added to the input instead (the usual baseline).
      rope    — rotate each (2c, 2c+1) pair of q and k by position·θ_c, so the score depends on s−t.
      pope    — POLAR COORDINATE POSITIONAL EMBEDDINGS (arXiv:2509.10534, ICML 2026).

    PoPE's argument is that RoPE ENTANGLES what and where: its score is
    `Σ_c μ_q μ_k cos((s−t)θ_c + φ_k − φ_q)`, where the content-dependent phases φ interact, so the model cannot match on
    content and position independently. PoPE makes the MAGNITUDE carry content and the PHASE carry position ALONE —
    magnitudes are `softplus(q)`, `softplus(k)` (non-negative), phases are `t·θ_c` and `s·θ_c` with no content term — giving
    `Σ_c μ_q μ_k cos((s−t)θ_c + δ_c)` with no interaction term. Written in Cartesian form it is an ordinary dot product in
    2·hd dims, which is how it is computed here. `δ_c` is a learnable per-channel bias, initialised U(−2π, 0) and clipped
    there (the paper's in-distribution setting; zero-init is for length extrapolation, which we are not testing).
    Reported effect: 11% → 95% on their indexing diagnostic, and small consistent perplexity gains at 124M–774M."""

    def __init__(self, d_model, n_head, pos, n_zero=0):
        super().__init__()
        self.h, self.hd, self.pos, self.n_zero = n_head, d_model // n_head, pos, n_zero
        self.causal = True                                # a written simulator over a grid sets this False (E28)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        if pos == "pope":
            self.delta = nn.Parameter(torch.empty(self.hd).uniform_(-2 * math.pi, 0.0))
        if pos in ("rope", "pope"):                       # cached, not rebuilt every forward: it never changes
            theta = _freqs(self.hd // 2 if pos == "rope" else self.hd, "cpu")
            if pos == "pope" and n_zero:                  # ziplearn DESIGN §15: a few channels with NO position, so a
                theta[:n_zero] = 0.0                      # pure content match has a home that does not depend on distance
            self.register_buffer("theta", theta, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (z.view(B, T, self.h, self.hd).transpose(1, 2) for z in (q, k, v))
        t = torch.arange(T, device=x.device, dtype=torch.float32)[:, None]
        if self.pos == "rope":
            ang = t * self.theta                                              # (T, hd/2)
            cos, sin = ang.cos()[None, None], ang.sin()[None, None]
            def rot(z):
                a, b = z[..., 0::2], z[..., 1::2]
                return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)
            q, k = rot(q), rot(k)
        elif self.pos == "pope":
            ang = t * self.theta                                      # phase is POSITION ONLY — the whole point
            d = self.delta.clamp(-2 * math.pi, 0.0)
            if self.n_zero:
                d = d * (self.theta != 0)                             # position-free channels: no offset either
            mq, mk = F.softplus(q), F.softplus(k)                     # magnitude is CONTENT only
            q = torch.cat([mq * ang.cos(), mq * ang.sin()], dim=-1)
            k = torch.cat([mk * (ang + d).cos(), mk * (ang + d).sin()], dim=-1)
        # FUSED attention. The hand-rolled version materialised a T*T score matrix and allocated a fresh causal mask on
        # every block of every step (3 blocks x 3200 steps = ~9600 allocations of each), which is pure overhead for a model
        # this small. `scale` is passed explicitly because PoPE doubles the query width to 2*hd and SDPA would otherwise
        # rescale by the wrong dimension, silently changing the attention temperature between schemes.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=1.0 / math.sqrt(self.hd))
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class AttnRes(nn.Module):
    """Attention residuals (Kimi Team, arXiv:2603.15031; notes in experiments/ziplearn/refs/attention_residuals.md).
    A layer's input is a softmax-weighted mix of SOURCES -- the token embedding and earlier outputs -- instead of their
    plain sum: one learned pseudo-query vector per mixer (initialised to ZERO, so training starts from the equal-weight
    average, as the paper requires), RMSNorm on the keys, the raw sources as values. `last` keeps the mean weight per
    source from the latest forward when `record` is set, which is how the learned routes are read."""

    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.RMSNorm(d_model)
        self.record, self.last = False, None

    def forward(self, sources):
        v = torch.stack(sources)                                      # (n_sources, B, T, d)
        a = torch.einsum("d,nbtd->nbt", self.w.to(v.dtype), self.norm(v)).softmax(0)
        if self.record:
            self.last = a.detach().float().mean(dim=(1, 2))
        return torch.einsum("nbt,nbtd->btd", a, v)


class Block(nn.Module):
    def __init__(self, d_model, n_head, pos, res="std", n_zero=0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attn(d_model, n_head, pos, n_zero)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        if res != "std":                                              # one mixer before each sub-layer, as in the paper
            self.res_attn, self.res_mlp = AttnRes(d_model), AttnRes(d_model)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    """A small causal decoder over digits. Predicts every OUTPUT block from everything before it, so the accuracy at the
    j-th block IS the accuracy after j-1 demonstrations — the trials curve falls out of one forward pass."""

    def __init__(self, d_model=96, n_layer=3, n_head=4, max_len=256, pos="learned", n_vocab=V, res="std", n_zero=0):
        super().__init__()
        # `n_vocab` defaults to the digit alphabet; `halt.py` widens it by one for a HALT token the model emits itself.
        # `res`: "std" = the usual residual sum; "attnres" = attention residuals with one transformer block per AttnRes
        # block (the paper's Block AttnRes, S = 2 sub-layers); "attnres_full" = every sub-layer output its own source.
        self.emb = nn.Embedding(n_vocab, d_model)
        self.pos_kind, self.res = pos, res
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02) if pos == "learned" else None
        self.blocks = nn.ModuleList([Block(d_model, n_head, pos, res, n_zero) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_vocab)
        if res != "std":
            self.res_final = AttnRes(d_model)                         # the output aggregates all block representations

    def forward(self, tok):
        h = self.emb(tok)
        if self.pos is not None:
            h = h + self.pos[:, :tok.shape[1]]                        # rope/pope inject position inside attention instead
        if self.res == "std":
            for b in self.blocks:
                h = b(h)
        elif self.res == "attnres":
            blocks, partial = [], h                                   # b0 = the embedding, appended at the first boundary
            for b in self.blocks:
                x = b.res_attn(blocks + [partial])
                blocks.append(partial)                                # block boundary: the finished sum becomes a source
                partial = b.attn(b.n1(x))                             # a block's sum holds OUTPUTS only, not its input
                x = b.res_mlp(blocks + [partial])
                partial = partial + b.mlp(b.n2(x))
            h = self.res_final(blocks + [partial])
        else:                                                         # attnres_full
            src = [h]
            for b in self.blocks:
                x = b.res_attn(src)
                src.append(b.attn(b.n1(x)))
                x = b.res_mlp(src)
                src.append(b.mlp(b.n2(x)))
            h = self.res_final(src)
        return self.head(self.norm(h))

    @torch.no_grad()
    def routes(self, tok):
        """Mean attention-residual weight per source for each mixer, on this batch: the learned wiring diagram."""
        mixers = [(n, m) for n, m in self.named_modules() if isinstance(m, AttnRes)]
        for _n, m in mixers:
            m.record = True
        self(tok)
        for _n, m in mixers:
            m.record = False
        return {n: [round(float(a), 3) for a in m.last] for n, m in mixers}


class LoopedModel(nn.Module):
    """A looped transformer after Chen, Vegesna, Dahal & Wilson (arXiv:2609.19107; their MIT-licensed code at
    github.com/qlabs-eng/scaling-exponents, models/transformer.py): prelude block(s) -> a CORE applied K times ->
    coda block(s). Between core passes the stream is RMS-normalised and the prelude's output (the "anchor") is
    re-injected, x <- norm(x) + alpha * anchor, and once more before the coda -- the boundary operator, which stops the
    residual stream growing with depth and keeps every pass conditioned on the input. `tied`: one core applied K
    times (weights shared); untied: K distinct cores (a deeper plain stack with the operator). `active_k` may be
    raised during training (model growth); untied growth copies core j % previous into the new cores."""

    def __init__(self, d_model=96, n_head=4, max_len=256, pos="rope", n_vocab=V, loops=4, tied=True, n_prelude=1, n_coda=1,
                 alpha=1.0, res="std", n_zero=0):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02) if pos == "learned" else None
        self.prelude = nn.ModuleList([Block(d_model, n_head, pos, "std", n_zero) for _ in range(n_prelude)])
        self.cores = nn.ModuleList([Block(d_model, n_head, pos, "std", n_zero) for _ in range(1 if tied else loops)])
        self.coda = nn.ModuleList([Block(d_model, n_head, pos, "std", n_zero) for _ in range(n_coda)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_vocab)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.loops, self.tied, self.active_k = loops, tied, loops

    def grow(self, new_k):
        """Model growth: run more passes from now on; untied cores are copy-initialised from the trained ones."""
        if not self.tied:
            prev = self.active_k
            for j in range(prev, new_k):
                src, dst = self.cores[j % prev], self.cores[j]
                for ps, pd in zip(src.parameters(), dst.parameters()):
                    pd.data.copy_(ps.data)
        self.active_k = new_k

    def forward(self, tok):
        h = self.emb(tok)
        if self.pos is not None:
            h = h + self.pos[:, :tok.shape[1]]
        for b in self.prelude:
            h = b(h)
        anchor = h
        for k in range(self.active_k):
            core = self.cores[0] if self.tied else self.cores[k]
            h = core(h)
            h = F.rms_norm(h, (h.shape[-1],)) + self.alpha * anchor      # the boundary operator, every pass
        for b in self.coda:
            h = b(h)
        return self.head(self.norm(h))


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
    p.add_argument("--k", type=int, default=8, help="demonstrations per sequence")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--opt", default="adamw", choices=["adamw", "aurora"])
    p.add_argument("--aurora_lr", type=float, default=0.05, help="Aurora eta (the repo default)")
    p.add_argument("--crit", type=float, default=0.8)
    p.add_argument("--pos", default="learned", choices=["learned", "rope", "pope"],
                   help="positional scheme: learned-absolute, RoPE, or PoPE (arXiv:2509.10534)")
    p.add_argument("--n_zero", type=int, default=0, help="PoPE only: channels per head with no position (content-only)")
    p.add_argument("--res", default="std", choices=["std", "attnres", "attnres_full", "loop"],
                   help="residual scheme: plain sum, attention residuals (arXiv:2603.15031) per block / per sub-layer, or "
                        "a looped transformer with a boundary operator (arXiv:2609.19107; --loops, --untied, --grow_at)")
    p.add_argument("--loops", type=int, default=4, help="loop: core passes (final count if growing)")
    p.add_argument("--untied", type=int, default=0, help="loop: 1 = distinct core weights per pass")
    p.add_argument("--grow_at", type=float, default=0.0, help="loop: fraction of training at which passes double to --loops (0 = no growth)")
    p.add_argument("--json", default="", help="also write the headline numbers (and the learned routes) to this file")
    p.add_argument("--save", default="", help="save the trained model's state_dict here (for the E8b Jacobian analysis)")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train, test, probs, weights = build_tasks(args.seed)
    print(f"device {dev} | pos={args.pos} n_zero={args.n_zero} res={args.res}"
          + (f" loops={args.loops} untied={args.untied} grow_at={args.grow_at}" if args.res == "loop" else "")
          + f" opt={args.opt} | {len(train)} trained, {len(test)} held out | K={args.k} steps={args.steps}")
    print("primitive training weight: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, weights.tolist())))

    if args.res == "loop":
        model = LoopedModel(max_len=args.k * 2 * L + 2, pos=args.pos, loops=args.loops, tied=not args.untied, n_zero=args.n_zero).to(dev)
        if args.grow_at > 0:
            model.active_k = max(1, args.loops // 2)
    else:
        model = Model(max_len=args.k * 2 * L + 2, pos=args.pos, res=args.res, n_zero=args.n_zero).to(dev)
    # AURORA (github.com/tilde-research/aurora-release, vendored in `aurora.py`) is a MUON variant: it orthogonalises the
    # momentum before applying it, and for TALL matrices additionally balances the update across ROWS. Following Muon
    # practice it takes only the 2D HIDDEN weights; embeddings, the output head and every 1D parameter (LayerNorm, biases)
    # stay on AdamW, which is also what `aurora()` requires — it raises on anything that is not 2D.
    # In this model the tall matrices are `qkv` (288x96) and the first MLP layer (384x96), 6 in all, so Aurora's
    # distinctive path is genuinely exercised rather than falling back to plain Muon everywhere.
    hidden = [p for n, p in model.named_parameters()
              if p.ndim == 2 and not n.startswith(("emb", "head"))] if args.opt == "aurora" else []
    hidden_ids = {id(p) for p in hidden}
    rest = [p for p in model.parameters() if id(p) not in hidden_ids]
    opt = torch.optim.AdamW(rest, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), fused=(dev == "cuda"))
    momenta = [torch.zeros_like(p) for p in hidden]
    warm = args.steps // 20
    def lr_at(s):                                  # one schedule, shared: Aurora's eta gets the same warmup+cosine shape
        # (s+1) so the first step is not exactly zero: Aurora rejects a non-positive eta, and a zero-LR first
        # step is meaningless anyway. Applied to BOTH arms so the schedules stay identical.
        return (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    g = torch.Generator(device=dev).manual_seed(args.seed)
    # COMPUTE, measured rather than guessed. Profiling said forward+backward was 94% of the step (55 ms) and data
    # generation only 3.5 ms, so the model — not the pipeline — was the cost, and 55 ms for a 340k-parameter model is ~5x
    # off what the card should do: many tiny kernels in fp32. Measured on this box, batch 256:
    #     baseline fp32 54.6 ms | bf16 32.5 | torch.compile fp32 41.5 | compile + bf16 17.6  => 3.1x
    # With the fused attention and GPU-side batching above, ~3.7x overall, which is spent on STEPS rather than saved.
    trainer = torch.compile(model) if args.compile and dev == "cuda" else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.bf16 and dev == "cuda"))
    t0 = time.time()
    for step in range(args.steps):
        if args.res == "loop" and args.grow_at > 0 and step == int(args.grow_at * args.steps):
            model.grow(args.loops)                                             # model growth: double the passes
            print(f"  grew to {args.loops} core passes at step {step}", flush=True)
        tok, _ = make_batch(args.batch, args.k, train, probs, dev, g)
        with amp:
            logits = trainer(tok)[:, :-1]
        m = out_mask(args.k, dev)[1:]
        loss = F.cross_entropy(logits[:, m].reshape(-1, V).float(), tok[:, 1:][:, m].reshape(-1))
        opt.zero_grad(set_to_none=True)
        for p in hidden:
            p.grad = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for w, mom in zip(hidden, momenta):       # Aurora owns its own weight write (decoupled decay, then W -= eta*U),
            if w.grad is not None:                #   so it is applied here rather than through the optimiser; note it
                aurora(w, w.grad, mom, eta=args.aurora_lr * lr_at(step))   # MUTATES the grad in place under Nesterov
        opt.step()
        sched.step()
    print(f"trained in {time.time() - t0:.0f}s | final train loss {loss.item():.4f}")
    if args.save:
        torch.save(dict(state=model.state_dict(), pos=args.pos, res=args.res, n_zero=args.n_zero, k=args.k, seed=args.seed,
                        d_model=96, n_layer=3, n_head=4), args.save)

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

    if args.json:
        routes = None
        if args.res in ("attnres", "attnres_full"):
            tok, _ = make_batch(64, args.k, train, probs, dev, g)
            routes = model.routes(tok)
            print("\nLEARNED ROUTES (mean attention-residual weight per source; sources = [embedding, block 1, block 2, ...])")
            for name, a in routes.items():
                print(f"   {name:<22} " + " ".join(f"{x:.2f}" for x in a))
        import json
        json.dump(dict(pos=args.pos, n_zero=args.n_zero, res=args.res, steps=args.steps, seed=args.seed,
                       trained_solved=len(solved), n_trained=len(seen),
                       trained_mean_acc=sum(r[2] for r in seen) / len(seen),
                       held_solved=sum(1 for y in ys if y < args.k), n_held=len(ys), held_mean_acc=sum(accs) / len(accs),
                       held_trials=ys, routes=routes, train_loss=loss.item()), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
