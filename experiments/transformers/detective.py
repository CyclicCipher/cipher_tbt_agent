"""THE DETECTIVE: the model chooses its own experiments, and nothing tells it which ones are good.

WHY. `canonicaliser.py` showed a contrastive encoder recovers the value map (0.689) and NEVER the correspondence (0.000)
from RANDOMLY SAMPLED demonstrations. The information-theoretic reading (Gwern, "Death Note: L, Anonymity & Eluding
Entropy") says why that is the wrong thing to ask of it. Identifying a transformation costs log2(7200) = 12.8 bits, and a
random length-6 input over 5 symbols carries only ~3.7 distinct values, so positions sharing a value are indistinguishable
and recovery needs an INTERSECTION across many observations — precisely the operation the encoder cannot do.

Design the query instead. With `x = [0,1,2,3,4,0]` the duplicated output value pins `c` in `v(t) = ±t + c`, leaving 2
sign candidates, and the permutation is pinned except for the swap of the two positions holding `v(0)` — 4 candidates, 2
bits of 12.8. A second probe with the duplicate moved resolves it exactly. **Two designed probes suffice where eight
random ones are needed**, and more importantly the inference becomes a LOOKUP ON A NEARLY-UNIQUE KEY rather than a
constraint intersection. That is the shape attention is good at.

THE POINT OF THIS FILE IS NOT TO HAND IT THAT PROBE. It is to see whether the model DISCOVERS it, given only agency and
an outcome.

THE BITTER-LESSON SHAPE, which is the whole design decision:
  * The hypothesis space is NOT enumerated. It is the model's own predictive distribution over responses — predicting
    `y` for an unseen `x` is exactly "having identified the transformation", so no symbolic candidate set is written down.
  * Information-seeking is NOT an objective. There is no expected-information-gain term anywhere. Instead the query
    BUDGET is tight and only the final answer is rewarded, so informative probes emerge because nothing else wins.

THE GAME. Against a hidden transformation T, for a budget of B queries:

    [BOS]  x1 -> y1   x2 -> y2   ...   xB -> yB   [ANSWER: idx (6 tokens) + vmp (5 tokens)]

The model EMITS every `x`; the environment supplies every `y`. The answer is the canonical form, which `canon2prog.py`
established is the thing worth identifying (canonical -> program is then learnable at 0.733).

THREE LOSSES, and the split is the interesting part:
  * ANSWER   — supervised always. We sampled T, so we know it.
  * RESPONSE — predict `y_i` BEFORE seeing it. Free, dense, and it IS the hypothesis representation.
  * QUERY    — no ground truth exists. So: sample K traces, keep the one with the lowest answer loss, and self-imitate
    its queries. That is XM's Forward objective (`min_i J`) applied to ACTIONS rather than generations, and unlike the
    factorisation case the bootstrap precondition holds, because random queries already succeed sometimes.

WHAT IS MEASURED, against a known floor and a known ceiling:
  1. ANSWER ACCURACY at budget B — learned queries vs RANDOM queries vs the HAND-DESIGNED optimum derived above.
  2. DISTINCT-VALUE COUNT of emitted queries — random gives ~3.7, the optimum is 5. Does it climb? This is "did it
     discover experimental design".
  3. ADAPTIVITY — force x1 to be IDENTICAL across tasks, then measure how much x2 varies. A good FIXED probe set is not
     L; conditioning the next probe on the last response is. This is the sharpest of the three, because it separates
     "learned a good habit" from "learned to investigate".

THE ONLY PRIOR is the interaction format — a query slot, a budget, an answer slot. That is an action space, not domain
knowledge.

Usage:  python experiments/transformers/detective.py --budget 2
"""
from __future__ import annotations

import argparse
import collections
import math
import time

import torch
import torch.nn.functional as F

from bigroup import cayley, compile_progs
from h1_lid import L, V, Model

MAXM = 5
DIG0 = 0                                   # response/query digits 0..V-1
IDXA = V                                   # answer: permutation entries, L classes
VMPA = V + L                               # answer: value-map entries, V classes
BOS = V + L + V
N_VOCAB = BOS + 1
ANS = L + V                                # answer length in tokens


def seq_len(B):
    return 1 + B * 2 * L + ANS


def q_slice(b):
    """Token positions holding the b-th QUERY (what the model emits)."""
    return slice(1 + b * 2 * L, 1 + b * 2 * L + L)


def r_slice(b):
    """Token positions holding the b-th RESPONSE (what the environment supplies)."""
    return slice(1 + b * 2 * L + L, 1 + (b + 1) * 2 * L)


def a_slice(B):
    return slice(1 + B * 2 * L, 1 + B * 2 * L + ANS)


def respond(tabs, sel, x):
    """The environment: y = T(x), computed with the canonical tables."""
    idx_t, vmp_t = tabs
    return vmp_t[sel].gather(1, x.gather(1, idx_t[sel]))


def answer_tokens(tabs, sel):
    idx_t, vmp_t = tabs
    return torch.cat([idx_t[sel] + IDXA, vmp_t[sel] + VMPA], dim=1)


@torch.no_grad()
def rollout(model, tabs, sel, B, dev, g, mode="model", temp=1.0, forced_first=None):
    """Play the game. `mode` decides where queries come from: the MODEL's own samples, uniformly RANDOM, or the
    HAND-DESIGNED probes. Returns the full token sequence with the true answer appended."""
    n = sel.shape[0]
    tok = torch.full((n, 1), BOS, dtype=torch.long, device=dev)
    for b in range(B):
        if mode == "random":
            x = torch.randint(0, V, (n, L), generator=g, device=dev)
        elif mode == "designed":
            # The optimum derived by hand: ALL V values present (the maximum possible at L=6, V=5), with the duplicated
            # value MOVED between probes so the second resolves the ambiguity the first leaves. Rolling the base instead
            # would shift the duplicate's position rather than change which value is duplicated, and drops to 4 distinct.
            x = torch.cat([torch.arange(V, device=dev),
                           torch.tensor([b % V], device=dev)]).unsqueeze(0).expand(n, L).clone()
        else:
            x = torch.zeros(n, L, dtype=torch.long, device=dev)
            for j in range(L):
                logit = model(tok)[:, -1, DIG0:DIG0 + V].float()
                nxt = (torch.multinomial((logit / temp).softmax(-1), 1, generator=g) if temp > 0
                       else logit.argmax(-1, keepdim=True))
                x[:, j] = nxt.squeeze(1)
                tok = torch.cat([tok, nxt], dim=1)
        if forced_first is not None and b == 0:
            x = forced_first.unsqueeze(0).expand(n, L).clone()
        if mode != "model" or forced_first is not None:
            tok = tok[:, :1 + b * 2 * L]                      # rebuild cleanly when queries were not sampled in place
            tok = torch.cat([tok, x], dim=1)
        tok = torch.cat([tok, respond(tabs, sel, x)], dim=1)
    return torch.cat([tok, answer_tokens(tabs, sel)], dim=1)


def losses(model, tok, B, dev):
    """Per-sequence answer loss, plus the response and query terms."""
    logits = model(tok)[:, :-1]
    tgt = tok[:, 1:]
    def ce(sl, lo, hi):
        s = slice(sl.start - 1, sl.stop - 1)
        return F.cross_entropy(logits[:, s, lo:hi].reshape(-1, hi - lo).float(),
                               (tgt[:, s] - lo).reshape(-1), reduction="none").reshape(tok.shape[0], -1).mean(1)
    ans = ce(a_slice(B), 0, N_VOCAB) if False else None
    a = a_slice(B)
    s = slice(a.start - 1, a.stop - 1)
    ans = F.cross_entropy(logits[:, s].reshape(-1, N_VOCAB).float(), tgt[:, s].reshape(-1),
                          reduction="none").reshape(tok.shape[0], -1).mean(1)
    resp = sum(ce(r_slice(b), DIG0, DIG0 + V) for b in range(B)) / B
    qry = sum(ce(q_slice(b), DIG0, DIG0 + V) for b in range(B)) / B
    return ans, resp, qry


@torch.no_grad()
def evaluate(model, tabs, ws, B, dev, seed, mode, n=256):
    g = torch.Generator(device=dev).manual_seed(seed)
    sel = torch.randint(0, len(ws), (n,), generator=g, device=dev)
    tok = rollout(model, tabs, sel, B, dev, g, mode=mode, temp=0.0)
    a = a_slice(B)
    pred = model(tok)[:, a.start - 1:a.stop - 1].argmax(-1)
    exact = (pred == tok[:, a]).all(-1).float().mean().item()
    qs = torch.stack([tok[:, q_slice(b)] for b in range(B)], dim=1)          # (n, B, L)
    per = [torch.tensor([len(set(row)) for row in qs[:, b].tolist()], dtype=torch.float).mean().item()
           for b in range(B)]
    ex = [tuple(qs[i, b].tolist()) for i in range(3) for b in range(B)]      # a few actual queries, to read the strategy
    return exact, sum(per) / len(per), per, ex


@torch.no_grad()
def adaptivity(model, tabs, ws, B, dev, seed, n=256):
    """Force x1 IDENTICAL across tasks, then measure how much x2 varies. A fixed probe policy gives ~0; an investigator
    that conditions on what it just saw gives a high number. Reported as the fraction of sequences whose x2 differs from
    the modal x2."""
    if B < 2:
        return float("nan")
    g = torch.Generator(device=dev).manual_seed(seed)
    sel = torch.randint(0, len(ws), (n,), generator=g, device=dev)
    first = torch.tensor([0, 1, 2, 3, 4, 0], device=dev)
    tok = rollout(model, tabs, sel, B, dev, g, mode="model", temp=0.0, forced_first=first)
    x2 = tok[:, q_slice(1)]
    rows = [tuple(r) for r in x2.tolist()]
    modal = collections.Counter(rows).most_common(1)[0][1]
    return 1.0 - modal / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--k", type=int, default=4, help="traces sampled per task; the best one is self-imitated")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--w_query", type=float, default=1.0, help="weight on self-imitating the best trace's queries")
    # Where QUERIES come from during TRAINING. Evaluation is then ON-POLICY, which is the only valid comparison: a model
    # trained on its own queries collapses to 0.000 when fed someone else's, which measures distribution shift rather
    # than probe quality. The first run of this file reported exactly that and it is not a baseline.
    ap.add_argument("--train_mode", default="model", choices=["model", "random", "designed"],
                    help="query source during training; evaluation is on-policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    B = args.budget

    _p, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    g0 = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(progs), generator=g0).tolist()
    progs = [progs[i] for i in order]
    held, train = progs[:args.held], progs[args.held:]
    tabs_tr, tabs_he = compile_progs(train, dev), compile_progs(held, dev)
    print(f"device {dev} | {len(progs)} transformations ({math.log2(len(progs)):.1f} bits) | train {len(train)} | "
          f"held-out {len(held)} | budget {B} queries | seq {seq_len(B)} | K={args.k}")
    print("The model EMITS every query. Nothing supervises which query to ask; only the ANSWER is supervised.\n")

    model = Model(max_len=seq_len(B) + 2, pos="rope", n_vocab=N_VOCAB).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98),
                            fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        sel = torch.randint(0, len(train), (args.batch,), generator=g, device=dev)
        # K traces per task, differing ONLY in the sampled queries.
        if args.train_mode == "model":
            toks = [rollout(model, tabs_tr, sel, B, dev, g, mode="model", temp=args.temp) for _ in range(args.k)]
            with torch.no_grad():
                scores = torch.stack([losses(model, t, B, dev)[0] for t in toks])    # (K, batch) answer loss
            best = scores.argmin(0)
            tok = torch.stack(toks)[best, torch.arange(args.batch, device=dev)]       # the winning trace per task
        else:
            tok = rollout(model, tabs_tr, sel, B, dev, g, mode=args.train_mode)
        ans, resp, qry = losses(model, tok, B, dev)
        # Self-imitating the queries only makes sense when the model CHOSE them.
        loss = ans.mean() + resp.mean() + (args.w_query * qry.mean() if args.train_mode == "model" else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"trained in {time.time() - t0:.0f}s | answer {ans.mean():.3f} response {resp.mean():.3f} query {qry.mean():.3f}")

    print(f"\n{'queries':<14}{'train exact':>13}{'HELD exact':>12}{'distinct/6':>12}{'per-probe':>16}")
    for mode in ("random", "designed", "model"):
        e_tr, _d, _p, _e = evaluate(model, tabs_tr, train, B, dev, args.seed, mode)
        e_he, d_he, per, ex = evaluate(model, tabs_he, held, B, dev, args.seed + 1, mode)
        tag = "   <- ON-POLICY" if mode == args.train_mode else ""
        print(f"{mode:<14}{e_tr:>13.3f}{e_he:>12.3f}{d_he:>12.2f}"
              f"{' ' + ' '.join(f'{v:.2f}' for v in per):>16}{tag}")
        if mode == args.train_mode:
            print(f"    example queries (task, probe order): {ex[:4]}")
    print("ONLY the ON-POLICY row is a statement about information. The others feed the model queries it never trained")
    print("on, so they measure distribution shift -- the first run of this file reported 0.000 there and it is not a")
    print("baseline. distinct/6: random ~3.72, the hand-derived optimum 5.00.")
    ad = adaptivity(model, tabs_he, held, B, dev, args.seed)
    print(f"\nadaptivity (x1 forced identical, fraction of x2 differing from the modal x2): {ad:.3f}")
    print("A fixed probe policy gives ~0.0; an investigator that conditions on the last response gives a high value.")


if __name__ == "__main__":
    main()
