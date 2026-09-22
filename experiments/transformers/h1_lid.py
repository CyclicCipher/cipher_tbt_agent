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


def rope_theta(hd, two_d=False, n_zero=0, dev="cpu"):
    """The ONE definition of a RoPE head's per-pair frequencies (ziplearn's instructions and the Attn below both read it).
    1-D (`two_d=False`): `_freqs(P)` over the P = hd // 2 pairs, pair 0 the fastest -- today's RoPE, unchanged. 2-D
    (`two_d=True`, the `rope2d` codec of BrainBuilder, DESIGN §21.4 edit 4): the row axis owns pairs 0 .. P//2-1 and the
    column axis pairs P//2 .. P-1, and EACH axis gets its own full ladder `_freqs(P//2)` (axial RoPE), so both axes hold
    the high frequencies that tell a distance of one from zero -- under one shared ladder the column half would keep only
    the slow frequencies (theta <= base^-1/2) and no column offset could be resolved. `n_zero` (2-D only) sets the LAST
    n_zero pairs -- the lowest-frequency pairs, where `Layout.channels` puts content -- to theta = 0: position-free
    channels, so a content match between two positions is exact (DESIGN §15's position-free channels, here for RoPE)."""
    P = hd // 2
    if not two_d:
        return _freqs(P, dev)
    half = P // 2
    theta = torch.cat([_freqs(half, dev), _freqs(P - half, dev)])
    if n_zero:
        theta[P - int(n_zero):] = 0.0
    return theta


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
        if pos == "rope":                                 # the 2-D ladder `coords` selects (rope_theta); n_zero = exact content pairs
            self.register_buffer("theta2d", rope_theta(self.hd, True, n_zero), persistent=False)

    def forward(self, x, cache=None, start=0, coords=None):
        """`cache` (a dict) and `start` make this an incremental forward: the new positions start..start+T-1 attend to the
        cached keys/values of all earlier positions and to each other causally; the cache is extended. Positions enter
        RoPE/PoPE as absolute indices, so a cached prefix and a fresh full pass agree exactly (E31).

        `coords` (a LongTensor T x 2, BrainBuilder's `rope2d` codec, ziplearn DESIGN §21.4 edit 4): with `pos == "rope"`
        the rotary pairs 0 .. P/2-1 rotate by coords[:, 0]·θ_c (the row axis) and the pairs P/2 .. P-1 by coords[:, 1]·θ_c
        (the column axis), each axis with its own frequency ladder `theta2d` (`rope_theta(hd, True, n_zero)`); a sequence
        is the row-0 case with col = position. `coords=None` is the one-dimensional RoPE above, exactly. Under
        `pos == "learned"` (E28's one-hot coordinates) `coords` is ignored — position is in the residual, not here; under
        PoPE it is not defined and raises."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (z.view(B, T, self.h, self.hd).transpose(1, 2) for z in (q, k, v))
        t = torch.arange(start, start + T, device=x.device, dtype=torch.float32)[:, None]
        if coords is not None and self.pos == "pope":
            raise ValueError("Attn: coords (rope2d) is defined for pos='rope' only")
        if self.pos == "rope":
            if coords is None:
                ang = t * self.theta                                          # (T, hd/2)
            else:                                                             # rope2d: row phases on the first half of the pairs, column on the second
                P = self.theta2d.shape[0]
                c = coords.to(device=x.device, dtype=torch.float32)
                ang = torch.cat([c[:, 0:1] * self.theta2d[:P // 2], c[:, 1:2] * self.theta2d[P // 2:]], dim=-1)
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
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=1.0 / math.sqrt(self.hd))
        else:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            S = k.shape[2]
            qpos = torch.arange(start, start + T, device=x.device)[:, None]
            kpos = torch.arange(S, device=x.device)[None, :]
            mask = (kpos <= qpos) if self.causal else None                    # a non-causal head (E28) sees the whole bank
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / math.sqrt(self.hd))
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
        self.record, self.last, self.log = False, None, []

    def forward(self, sources):
        v = torch.stack(sources)                                      # (n_sources, B, T, d)
        a = torch.einsum("d,nbtd->nbt", self.w.to(v.dtype), self.norm(v)).softmax(0)
        if self.record:
            self.last = a.detach().float().mean(dim=(1, 2))
            self.log.append(self.last)                                # one entry per call: a tied mixer logs every pass
        return torch.einsum("nbt,nbtd->btd", a, v)


def _norm(d_model, norm):
    """`norm="layer"`: LayerNorm (the trained arm). `norm="none"`: Identity — a WRITTEN block over one-hot codes (ziplearn
    DESIGN §21.4 edit 1; E28 bypassed `Block` because LayerNorm rescales a one-hot, E18 survived it by margin)."""
    if norm == "layer":
        return nn.LayerNorm(d_model)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"norm must be 'layer' or 'none', got {norm!r}")


def _per_layer(n_head, n_layer, what="n_head"):
    """`n_head` as one int for every layer, or a list with one entry per layer (edit 3: a layer holding one whole-window
    match is a one-head layer with hd = d, as `WrittenSim.lookup = Attn(d, 1)` is)."""
    if isinstance(n_head, int):
        return [n_head] * n_layer
    heads = list(n_head)
    if len(heads) != n_layer:
        raise ValueError(f"{what}: per-layer list has {len(heads)} entries for {n_layer} layers")
    return heads


class Block(nn.Module):
    def __init__(self, d_model, n_head, pos, res="std", n_zero=0, norm="layer"):
        super().__init__()
        self.n1, self.n2 = _norm(d_model, norm), _norm(d_model, norm)
        self.attn = Attn(d_model, n_head, pos, n_zero)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        if res != "std":                                              # one mixer before each sub-layer, as in the paper
            self.res_attn, self.res_mlp = AttnRes(d_model), AttnRes(d_model)

    def forward(self, x, cache=None, start=0, coords=None):
        x = x + self.attn(self.n1(x), cache, start, coords)
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    """A small causal decoder over digits. Predicts every OUTPUT block from everything before it, so the accuracy at the
    j-th block IS the accuracy after j-1 demonstrations — the trials curve falls out of one forward pass."""

    def __init__(self, d_model=96, n_layer=3, n_head=4, max_len=256, pos="learned", n_vocab=V, res="std", n_zero=0, norm="layer"):
        super().__init__()
        # `n_vocab` defaults to the digit alphabet; `halt.py` widens it by one for a HALT token the model emits itself.
        # `res`: "std" = the usual residual sum; "attnres" = attention residuals with one transformer block per AttnRes
        # block (the paper's Block AttnRes, S = 2 sub-layers); "attnres_full" = every sub-layer output its own source.
        # `n_head`: an int, or a list with one head count per layer; `norm`: "layer" or "none" (a written brain).
        self.emb = nn.Embedding(n_vocab, d_model)
        self.pos_kind, self.res = pos, res
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02) if pos == "learned" else None
        self.blocks = nn.ModuleList([Block(d_model, nh, pos, res, n_zero, norm) for nh in _per_layer(n_head, n_layer)])
        self.norm = _norm(d_model, norm)
        self.head = nn.Linear(d_model, n_vocab)
        if res != "std":
            self.res_final = AttnRes(d_model)                         # the output aggregates all block representations

    def forward(self, tok):
        return self.head(self.norm(self.forward_embedded(self.embed(tok))))

    def embed(self, tok, start=0):
        h = self.emb(tok)
        if self.pos is not None:
            h = h + self.pos[:, start:start + tok.shape[1]]           # rope/pope inject position inside attention instead
        return h

    def new_caches(self):
        return [{} for _ in self.blocks]

    def forward_embedded(self, h, caches=None, start=0, coords=None):
        """The blocks on already-embedded inputs; returns the final residual (before the output norm and head). With
        `caches` (one dict per block, from `new_caches`) the call is incremental from position `start` (E31). `coords`
        is passed through to every head (rope2d, see `Attn.forward`)."""
        cs = caches if caches is not None else [None] * len(self.blocks)
        if self.res == "std":
            for b, c in zip(self.blocks, cs):
                h = b(h, c, start, coords)
        elif self.res == "attnres":
            blocks, partial = [], h                                   # b0 = the embedding, appended at the first boundary
            for b, c in zip(self.blocks, cs):
                x = b.res_attn(blocks + [partial])
                blocks.append(partial)                                # block boundary: the finished sum becomes a source
                partial = b.attn(b.n1(x), c, start, coords)           # a block's sum holds OUTPUTS only, not its input
                x = b.res_mlp(blocks + [partial])
                partial = partial + b.mlp(b.n2(x))
            h = self.res_final(blocks + [partial])
        else:                                                         # attnres_full
            src = [h]
            for b, c in zip(self.blocks, cs):
                x = b.res_attn(src)
                src.append(b.attn(b.n1(x), c, start, coords))
                x = b.res_mlp(src)
                src.append(b.mlp(b.n2(x)))
            h = self.res_final(src)
        return h

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


class BoundaryOp(nn.Module):
    """The WRITTEN boundary operator between core passes (ziplearn DESIGN §21.3 item 9, §21.4 edit 5) — a module of
    tensors, no learned weight: what `WrittenSim.rollout` (e28.py lines 178–189) did in Python between two passes.

      keep           a d-vector mask, 1 = keep the dim, 0 = clear it (E28: the gathered-neighbour subspaces are cleared);
      quantise       a list of (read matrix d x g, write slice of width g): argmax over the g pre-activations `x @ read`,
                     a one-hot into the slice (E28: the colour subspace re-quantised to one-hot after the lookup);
      commit         a list of (from slice, to slice) copies, applied after quantise (`pred -> colour`: the simulator
                     advances one step);
      anchor         the slice the anchor vector is copied into, on cell tokens only (E28: the action, the per-pass anchor);
      halt_flag      a dim index the runner reads, or None;
      register_dims  the dims compared between passes for convergence (exact: one-hot after quantise), possibly empty;
      cls_flags      the token-class flag dims the operator acts on (E28: cells; a brain: cells and registers); None = every
                     token. Entry, border and zero tokens carry no such flag and pass through untouched, as E28's memory
                     tokens did (`x[0, n:, ...]`);
      cell_flag      the flag dim of the tokens the anchor goes into; None = every operated token;
      register_flag  the flag dim of the register tokens, whose `register_dims` the convergence test compares; None = every
                     operated token;
      anchor_token   the sequence index of the register token whose anchor slice is the anchor when the runner passes
                     none (the blueprint's `anchor.from = ACTION`: the register's quantised action, read by the operator
                     itself -- a tensor op, no Python around the model); None = the runner's `anchor_vec` only;
      tie_tol        the tolerance of the quantise: a group is a TIE when its maximum is not 1 to `tie_tol` (E28 line 185:
                     1e-3 at M = 30, where the softmax leaks 1e-13), and the argmax breaks ties toward the LOWEST index among
                     the values within `tie_tol` of the maximum (torch.argmax's rule on exact ties, made robust to a leak):
                     the compiler passes 1 - p*, the leak the derived sharpness allows (DESIGN §21.4 step 5), so that the
                     tie count and the tie-break are those of the exact arithmetic. A group that is all zero (no evidence
                     reached the token: a register with nothing to select, a cell without a prediction) is neither
                     quantised nor a tie: it stays zero.

    `forward(x, anchor_vec=None, prev_regs=None) -> (x, halted)`: the halt flag is read first, as the pass raised it (so a
    blueprint may clear it with the scratch and the MLP recomputes it every pass), then quantise -> commit -> keep ->
    anchor -> the convergence test. `halted` is a bool tensor (B,), true where the halt flag is set or the register dims
    equal `prev_regs` (the previous pass's `x[..., register_dims]` on the register tokens). `ties` counts, as E28 line 185 did, the operated
    tokens whose quantised group was not already one-hot to `tie_tol` — the tie count B0 must reproduce (1,434 / 1,475).
    `inject(x, anchor_vec)` is the anchor step alone, for the first pass."""

    def __init__(self, keep, quantise=(), commit=(), anchor=None, halt_flag=None, register_dims=(), cls_flags=None, cell_flag=None,
                 register_flag=None, anchor_token=None, tie_tol=1e-3):
        super().__init__()
        keep = torch.as_tensor(keep, dtype=torch.float32)
        self.d = int(keep.numel())
        self.register_buffer("keep", keep)
        self.n_quant = len(quantise)
        self.quant_write = []
        for i, (read, write) in enumerate(quantise):
            read = torch.as_tensor(read, dtype=torch.float32)
            g = write.stop - write.start
            if read.shape != (self.d, g):
                raise ValueError(f"BoundaryOp.quantise[{i}]: read matrix {tuple(read.shape)} must be ({self.d}, {g}) for the write slice")
            self.register_buffer(f"quant_read_{i}", read)
            self.quant_write.append((write.start, write.stop))
        self.commit = [(f.start, f.stop, t.start, t.stop) for f, t in commit]
        for f0, f1, t0, t1 in self.commit:
            if f1 - f0 != t1 - t0:
                raise ValueError(f"BoundaryOp.commit: slices {f0}:{f1} and {t0}:{t1} differ in width")
        self.anchor = (anchor.start, anchor.stop) if anchor is not None else None
        self.halt_flag = halt_flag
        self.register_buffer("register_dims", torch.as_tensor(list(register_dims), dtype=torch.long))
        self.register_buffer("cls_flags", torch.as_tensor(list(cls_flags), dtype=torch.long) if cls_flags is not None else torch.zeros(0, dtype=torch.long))
        self.cell_flag = cell_flag
        self.register_flag = register_flag
        self.anchor_token = anchor_token
        self.tie_tol = float(tie_tol)
        self.ties = 0

    def _mask(self, x):
        """Which tokens the operator acts on, read BEFORE anything is changed."""
        if self.cls_flags.numel() == 0:
            return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        return (x[..., self.cls_flags] > 0.5).any(-1)

    def inject(self, x, anchor_vec, mask=None):
        """The anchor alone: `anchor_vec` (a tensor of the anchor slice's width, or (B, width)) into the anchor slice of
        the cell tokens (E28 line 179–180). With `anchor_vec=None` and an `anchor_token`, the anchor is that token's own
        anchor slice (the ACTION register, after its quantise)."""
        if self.anchor is None:
            return x
        a0, a1 = self.anchor
        if anchor_vec is None:
            if self.anchor_token is None:
                return x
            anchor_vec = x[:, self.anchor_token, a0:a1]                        # (B, width): the register's selection
        cells = self._mask(x) if mask is None else mask
        if self.cell_flag is not None:
            cells = cells & (x[..., self.cell_flag] > 0.5)
        a = torch.as_tensor(anchor_vec, dtype=x.dtype, device=x.device)
        if a.dim() == 2:                                                       # per batch element: broadcast over tokens
            a = a[:, None, :].expand(x.shape[0], x.shape[1], a1 - a0)
        else:
            a = a.expand(x.shape[0], x.shape[1], a1 - a0)
        sub = x[..., a0:a1]
        x = x.clone()
        x[..., a0:a1] = torch.where(cells[..., None], a, sub)
        return x

    def quantise(self, pre):
        """The tolerant argmax of one group (…, g): among the values within `tie_tol` of the maximum the LOWEST index
        wins (torch.argmax on an exact tie), the one-hot of that index; an all-zero group gives an all-zero row."""
        top = pre.max(-1, keepdim=True).values
        near = (top - pre) <= self.tie_tol                                     # the tied set (the exact tie, leak-robust)
        idx = near.float().argmax(-1)                                          # the lowest tied index
        one = F.one_hot(idx, pre.shape[-1]).to(pre.dtype)
        return one * (pre.abs().sum(-1, keepdim=True) > 0).to(pre.dtype)

    def forward(self, x, anchor_vec=None, prev_regs=None):
        mask = self._mask(x)                                                   # (B, T)
        x = x.clone()
        m3 = mask[..., None]
        halted = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if self.halt_flag is not None:                                         # the flag as the pass raised it (before a clear)
            halted = halted | ((x[..., self.halt_flag] > 0.5) & mask).any(-1)
        for i in range(self.n_quant):                                          # quantise: argmax over the group -> one-hot
            w0, w1 = self.quant_write[i]
            pre = x @ getattr(self, f"quant_read_{i}")                         # (B, T, g)
            live = pre.abs().sum(-1) > 0                                       # an all-zero group is no evidence, not a tie
            self.ties += int((((pre.max(-1).values - 1).abs() > self.tie_tol) & mask & live).sum())
            x[..., w0:w1] = torch.where(m3, self.quantise(pre), x[..., w0:w1])
        for f0, f1, t0, t1 in self.commit:                                     # commit: copy after quantise
            x[..., t0:t1] = torch.where(m3, x[..., f0:f1], x[..., t0:t1])
        x = torch.where(m3, x * self.keep, x)                                  # keep / clear
        x = self.inject(x, anchor_vec, mask)                                   # the next pass's anchor
        if prev_regs is not None and self.register_dims.numel():
            same = x[..., self.register_dims] == prev_regs                     # (B, T, n_reg)
            if self.register_flag is not None:
                same = same | ~(x[..., self.register_flag] > 0.5)[..., None]   # only the register tokens are compared
            halted = halted | same.flatten(1).all(-1)
        return x, halted


class LoopedModel(nn.Module):
    """A looped transformer after Chen, Vegesna, Dahal & Wilson (arXiv:2609.19107; their MIT-licensed code at
    github.com/qlabs-eng/scaling-exponents, models/transformer.py): prelude block(s) -> a CORE applied K times ->
    coda block(s). Between core passes the stream is RMS-normalised and the prelude's output (the "anchor") is
    re-injected, x <- norm(x) + alpha * anchor, and once more before the coda -- the boundary operator, which stops the
    residual stream growing with depth and keeps every pass conditioned on the input. `tied`: one core applied K
    times (weights shared); untied: K distinct cores (a deeper plain stack with the operator). `active_k` may be
    raised during training (model growth); untied growth copies core j % previous into the new cores.

    BrainBuilder's flags (ziplearn DESIGN §21.4, defaults unchanged): `core_layers` = Blocks per core pass (E28's core is
    gather then lookup: two); `boundary` = a `BoundaryOp` in place of `rms_norm(x) + alpha * anchor`; `norm="none"` for
    written one-hot codes; `n_head` an int or one entry per layer over prelude + core + coda (the same list for every
    untied core copy). The core layers are `self.blocks` (state_dict names `blocks.{L}.…`, the addresses the compiler
    writes; core copy j, layer l = `blocks[j * core_layers + l]`); `cores` is the old name for the same list."""

    def __init__(self, d_model=96, n_head=4, max_len=256, pos="rope", n_vocab=V, loops=4, tied=True, n_prelude=1, n_coda=1,
                 alpha=1.0, res="std", n_zero=0, mix="tied", window=0, core_layers=1, boundary=None, norm="layer"):
        super().__init__()
        heads = _per_layer(n_head, n_prelude + core_layers + n_coda)
        zeros = _per_layer(n_zero, n_prelude + core_layers + n_coda, "n_zero")   # int, or one per layer (a brain's exact content pairs)
        h_pre, h_core, h_coda = heads[:n_prelude], heads[n_prelude:n_prelude + core_layers], heads[n_prelude + core_layers:]
        z_pre, z_core, z_coda = zeros[:n_prelude], zeros[n_prelude:n_prelude + core_layers], zeros[n_prelude + core_layers:]
        self.n_cores, self.core_layers = (1 if tied else loops), core_layers
        if res == "attnres" and (core_layers != 1 or boundary is not None):
            raise ValueError("LoopedModel: attention residuals across passes (E30) are defined for a one-Block core without a BoundaryOp")
        self.emb = nn.Embedding(n_vocab, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02) if pos == "learned" else None
        self.prelude = nn.ModuleList([Block(d_model, nh, pos, "std", nz, norm) for nh, nz in zip(h_pre, z_pre)])
        self.blocks = nn.ModuleList([Block(d_model, nh, pos, "std", nz, norm) for _ in range(self.n_cores) for nh, nz in zip(h_core, z_core)])
        self.coda = nn.ModuleList([Block(d_model, nh, pos, "std", nz, norm) for nh, nz in zip(h_coda, z_coda)])
        self.norm = _norm(d_model, norm)
        self.head = nn.Linear(d_model, n_vocab)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.boundary = boundary
        self.last_passes = 0
        self.loops, self.tied, self.active_k = loops, tied, loops
        # E30 (DESIGN §18): attention residuals ACROSS PASSES. The sources at pass k are the anchor (the prelude's output)
        # and the sums of passes 1..k-1 (the paper's Block form: a pass's sum holds its outputs only); the mixers read
        # them with a learned query, softmax over sources, RMSNorm on the keys, raw sources as values. `mix`: "tied" = one
        # mixer per sub-layer shared by every pass (routing by content); "per_pass" = one query per pass. `window` = m > 0
        # keeps only the anchor and the last m pass sums, so the pass is a function of a fixed-size state (what a fixed
        # point and a convergence test need); m = 1 is the fixed boundary operator's source set. The fixed operator
        # (res="std") is the special case: norm(last pass) + alpha * anchor.
        self.res, self.mix, self.window = res, mix, window
        if res == "attnres":
            n_mix = 1 if mix == "tied" else loops
            self.mix_attn = nn.ModuleList([AttnRes(d_model) for _ in range(n_mix)])
            self.mix_mlp = nn.ModuleList([AttnRes(d_model) for _ in range(n_mix)])
            self.res_final = AttnRes(d_model)

    def grow(self, new_k):
        """Model growth: run more passes from now on; untied cores (and per-pass mixers) are copy-initialised from the
        trained ones."""
        prev = self.active_k
        if not self.tied:
            for j in range(prev, new_k):
                for src, dst in zip(self.core_blocks(j % prev), self.core_blocks(j)):
                    for ps, pd in zip(src.parameters(), dst.parameters()):
                        pd.data.copy_(ps.data)
        if self.res == "attnres" and self.mix == "per_pass":
            for mixers in (self.mix_attn, self.mix_mlp):
                for j in range(prev, new_k):
                    mixers[j].w.data.copy_(mixers[j % prev].w.data)
        self.active_k = new_k

    @property
    def cores(self):
        """The core Blocks under their pre-BrainBuilder name (`self.blocks` holds them)."""
        return self.blocks

    def core_blocks(self, k):
        """The `core_layers` Blocks applied at pass k (tied: always the same; untied: core k % n_cores — beyond the
        trained passes the cores cycle)."""
        j = 0 if self.tied else k % self.n_cores
        return list(self.blocks[j * self.core_layers:(j + 1) * self.core_layers])

    def _sources(self, anchor, sums):
        return [anchor] + (sums[-self.window:] if self.window > 0 else sums)

    def _mixer(self, mixers, k):
        return mixers[0] if self.mix == "tied" else mixers[k % len(mixers)]

    def embed(self, tok, start=0):
        h = self.emb(tok)
        if self.pos is not None:
            h = h + self.pos[:, start:start + tok.shape[1]]
        return h

    def new_caches(self):
        """One k/v cache per Block APPLICATION — prelude, then `active_k` x `core_layers` core applications, then coda —
        so a tied core's passes do not share one cache (E31's incremental forward on the looped model)."""
        return [{} for _ in range(len(self.prelude) + self.active_k * self.core_layers + len(self.coda))]

    def forward_embedded(self, h, caches=None, start=0, coords=None, anchors=None, converge=None):
        """Prelude -> `active_k` core passes with the boundary operator between them -> coda, on already-embedded inputs;
        returns the final residual (before the output norm and head), the mirror of `Model.forward_embedded` so that a
        written brain's sequence axis (`Brain.think`) can append positions through the caches. `anchors` (a list, one
        per pass) feeds the `BoundaryOp`: `anchors[0]` is injected before the first pass, the operator after pass k
        receives `anchors[k + 1]` — E28's order (the action before its pass; re-quantise and clear after). Passes stop
        early when the operator reports every batch element halted; `last_passes` records the count. `converge` = whether
        register convergence may halt the loop: by default only when the runner gives no anchors (a plan of n actions is
        n passes whatever the registers do; the executive's own loop stops when its registers settle)."""
        n_pre = len(self.prelude)
        cache_at = (lambda i: caches[i]) if caches is not None else (lambda i: None)
        for i, b in enumerate(self.prelude):
            h = b(h, cache_at(i), start, coords)
        anchor = h
        self.last_passes = 0
        if converge is None:
            converge = anchors is None
        if self.res == "std":
            if self.boundary is not None and anchors is not None and len(anchors) > 0:
                h = self.boundary.inject(h, anchors[0])
            prev_regs = None
            for k in range(self.active_k):
                for l, b in enumerate(self.core_blocks(k)):
                    h = b(h, cache_at(n_pre + k * self.core_layers + l), start, coords)
                self.last_passes = k + 1
                if self.boundary is None:
                    h = F.rms_norm(h, (h.shape[-1],)) + self.alpha * anchor  # the boundary operator, every pass
                else:
                    nxt = anchors[k + 1] if anchors is not None and k + 1 < len(anchors) else None
                    h, halted = self.boundary(h, anchor_vec=nxt, prev_regs=prev_regs if converge else None)
                    if converge and self.boundary.register_dims.numel():
                        prev_regs = h[..., self.boundary.register_dims]
                    if bool(halted.all()):
                        break
        else:                                                         # attention residuals across passes (E30)
            sums, partial = [], anchor
            for k in range(self.active_k):
                core = self.core_blocks(k)[0]
                c = cache_at(n_pre + k)
                x = self._mixer(self.mix_attn, k)(self._sources(anchor, sums))
                partial = core.attn(core.n1(x), c, start, coords)
                x = self._mixer(self.mix_mlp, k)(self._sources(anchor, sums) + [partial])
                partial = partial + core.mlp(core.n2(x))
                sums.append(partial)                                  # the pass's sum: its outputs only
                self.last_passes = k + 1
            h = self.res_final(self._sources(anchor, sums))
        for i, b in enumerate(self.coda):
            h = b(h, cache_at(n_pre + self.active_k * self.core_layers + i), start, coords)
        return h

    def forward(self, tok):
        return self.head(self.norm(self.forward_embedded(self.embed(tok))))

    @torch.no_grad()
    def routes(self, tok):
        """The learned wiring across passes: for each mixer, the mean weight per source at every pass it was called on
        (sources = [anchor, pass 1, pass 2, ...] within the window; the MLP mixer also sees the pass's own attention)."""
        if self.res != "attnres":
            return None
        mixers = [(n, m) for n, m in self.named_modules() if isinstance(m, AttnRes)]
        for _n, m in mixers:
            m.record, m.log = True, []
        self(tok)
        for _n, m in mixers:
            m.record = False
        return {n: [[round(float(a), 3) for a in call] for call in m.log] for n, m in mixers}


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
    p.add_argument("--loop_res", default="std", choices=["std", "attnres"],
                   help="loop: std = the fixed boundary operator; attnres = attention residuals across passes (E30)")
    p.add_argument("--mix", default="tied", choices=["tied", "per_pass"], help="loop+attnres: mixers shared by every pass, or one query per pass")
    p.add_argument("--window", type=int, default=0, help="loop+attnres: keep the anchor + the last m pass sums as sources (0 = all)")
    p.add_argument("--extrap", type=int, default=2, help="loop: also evaluate at this multiple of the trained passes (0 = skip)")
    p.add_argument("--json", default="", help="also write the headline numbers (and the learned routes) to this file")
    p.add_argument("--save", default="", help="save the trained model's state_dict here (for the E8b Jacobian analysis)")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train, test, probs, weights = build_tasks(args.seed)
    print(f"device {dev} | pos={args.pos} n_zero={args.n_zero} res={args.res}"
          + (f" loops={args.loops} untied={args.untied} grow_at={args.grow_at} loop_res={args.loop_res} mix={args.mix} window={args.window}" if args.res == "loop" else "")
          + f" opt={args.opt} | {len(train)} trained, {len(test)} held out | K={args.k} steps={args.steps}")
    print("primitive training weight: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, weights.tolist())))

    if args.res == "loop":
        model = LoopedModel(max_len=args.k * 2 * L + 2, pos=args.pos, loops=args.loops, tied=not args.untied, n_zero=args.n_zero,
                            res=args.loop_res, mix=args.mix, window=args.window).to(dev)
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

    extrap = None
    if args.res == "loop" and args.extrap and args.extrap > 1:
        # E30: the same model run at more passes than it was trained with -- does the recurrence extrapolate?
        k0 = model.active_k
        model.active_k = k0 * args.extrap
        ex_seen = trials_to_criterion(model, train, dev, args.k, args.crit, n=256)
        ex_rows = trials_to_criterion(model, test, dev, args.k, args.crit)
        model.active_k = k0
        extrap = dict(passes=k0 * args.extrap, trained_mean_acc=sum(r[2] for r in ex_seen) / len(ex_seen),
                      trained_solved=sum(1 for r in ex_seen if r[1] < args.k),
                      held_mean_acc=sum(r[2] for r in ex_rows) / len(ex_rows), held_solved=sum(1 for r in ex_rows if r[1] < args.k))
        print(f"\nEXTRAPOLATION to {extrap['passes']} passes (trained with {k0}): trained acc {extrap['trained_mean_acc']:.2f} "
              f"({extrap['trained_solved']}/{len(ex_seen)} solved), held-out acc {extrap['held_mean_acc']:.2f} ({extrap['held_solved']}/{len(ex_rows)})")
    if args.json:
        routes = None
        if args.res in ("attnres", "attnres_full") or (args.res == "loop" and args.loop_res == "attnres"):
            tok, _ = make_batch(64, args.k, train, probs, dev, g)
            routes = model.routes(tok)
            if args.res == "loop":
                print("\nLEARNED ROUTES ACROSS PASSES (mean weight per source at each call; sources = [anchor, pass 1, pass 2, ...])")
                for name, calls in routes.items():
                    for k, a in enumerate(calls):
                        print(f"   {name:<22} call {k}: " + " ".join(f"{x:.2f}" for x in a))
            else:
                print("\nLEARNED ROUTES (mean attention-residual weight per source; sources = [embedding, block 1, block 2, ...])")
                for name, a in routes.items():
                    print(f"   {name:<22} " + " ".join(f"{x:.2f}" for x in a))
        import json
        json.dump(dict(pos=args.pos, n_zero=args.n_zero, res=args.res, steps=args.steps, seed=args.seed,
                       loops=args.loops, untied=args.untied, grow_at=args.grow_at, loop_res=args.loop_res, mix=args.mix, window=args.window,
                       trained_solved=len(solved), n_trained=len(seen),
                       trained_mean_acc=sum(r[2] for r in seen) / len(seen),
                       held_solved=sum(1 for y in ys if y < args.k), n_held=len(ys), held_mean_acc=sum(accs) / len(accs),
                       held_trials=ys, routes=routes, extrap=extrap, train_loss=loss.item()), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
