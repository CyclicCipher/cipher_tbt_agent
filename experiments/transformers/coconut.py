"""E31 — continuous thoughts (Coconut, Hao et al. arXiv:2412.06769) in the gradient arm, under plain and attention
residuals, with and without a boundary operator on the fed-back state. For research insight only (DESIGN §18): the
recurrence along the SEQUENCE -- a position's final state becomes the next position's input -- and how it interacts with
attention residuals, plus Coconut's reported hazards (a curriculum is needed; instability as thoughts per step grow).

The task is E0's composition task (h1_lid: two primitives composed, K demonstrations per sequence, 17 trained / 8 held-out
compositions). Its chain-of-thought form writes the INTERMEDIATE result between input and output:
    stage 0 (cot):      [ x (L digits) | mid = f1(x) (L) | y = f2(mid) (L) ] x K       loss on mid and y
    stage 1 (thought):  [ x (L)        | c continuous thoughts              | y (L) ] x K   loss on y
A continuous thought is a position whose input embedding is the previous position's final residual, fed through
`feedback`: "raw" (the residual as it is, Coconut's choice) or "operator" (RMSNorm + a learned thought vector: the
boundary operator on the sequence axis). Positions are appended one at a time through the model's key/value caches, so
every earlier state is final (causal attention) and computed once. Coconut's curriculum: stage 0 for the first
`--stage_frac` of training, stage 1 after; `--curriculum 0` trains stage 1 from the start (the paper's negative control).

Measured: trained / held-out compositions solved to criterion and mean final accuracy in thought mode (the comparison
with E26/E30); the same in cot mode (the language-reasoning ceiling); training-loss spikes after the stage switch; a
linear probe from each thought's state to the intermediate digits (does the thought hold the intermediate?) and how many
candidate digits it holds at once (the breadth-first claim, crudely).

    python experiments/transformers/coconut.py --c 1 --res attnres --feedback operator --json out.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_lid as H                                                   # noqa: E402
from h1_lid import L, V, NAMES, PRIMS, build_tasks, Model            # noqa: E402


def make_batch_cot(B, K, task_list, probs, dev, g=None, fixed=None):
    """x, mid, y each (B, K, L): the input, the first primitive's result, the composition's result."""
    if fixed is not None:
        idx = torch.full((B,), fixed, dtype=torch.long, device=dev)
    else:
        idx = torch.multinomial(probs.to(dev), B, replacement=True, generator=g)
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    mid, y = torch.empty_like(x), torch.empty_like(x)
    for t in idx.unique():
        m = idx == t
        pair = task_list[int(t)]
        mid[m] = PRIMS[NAMES[pair[0]]](x[m])
        y[m] = PRIMS[NAMES[pair[1]]](mid[m])
    return x, mid, y, idx


class Thinker(torch.nn.Module):
    """The model plus the sequence-axis feedback (E31)."""

    def __init__(self, model, feedback):
        super().__init__()
        self.model, self.feedback = model, feedback
        self.thought = torch.nn.Parameter(torch.randn(model.emb.embedding_dim) * 0.02)   # the thought marker (operator)

    def fed(self, h_last):
        if self.feedback == "raw":
            return h_last
        return F.rms_norm(h_last, (h_last.shape[-1],)) + self.thought

    def run_cot(self, x, mid, y):
        """Stage 0: one plain forward over [x | mid | y] x K. Returns logits (B, T, V) and the token sequence."""
        tok = torch.cat([x, mid, y], dim=2).reshape(x.shape[0], -1)
        h = self.model.forward_embedded(self.model.embed(tok))
        return self.model.head(self.model.norm(h)), tok

    def run_thought(self, x, y, c):
        """Stage 1: [x | c thoughts | y] x K, the thoughts appended one at a time through the caches. Returns the logits at
        every position, the token layout (None at thought positions), and the thought states (B, K, c, d)."""
        m, B, K = self.model, x.shape[0], x.shape[1]
        caches = m.new_caches()
        logits, layout, states = [], [], []
        pos = 0
        for k in range(K):
            h = m.forward_embedded(m.embed(x[:, k], start=pos), caches, pos)                 # the input block
            logits.append(m.head(m.norm(h))); layout.extend([("x", k, j) for j in range(L)]); pos += L
            last = h[:, -1]
            th = []
            for j in range(c):                                                                # the thoughts
                inp = self.fed(last)[:, None, :]
                h = m.forward_embedded(inp, caches, pos)
                logits.append(m.head(m.norm(h))); layout.append(("t", k, j)); pos += 1
                last = h[:, -1]
                th.append(last)
            states.append(torch.stack(th, 1))
            h = m.forward_embedded(m.embed(y[:, k], start=pos), caches, pos)                 # the output block
            logits.append(m.head(m.norm(h))); layout.extend([("y", k, j) for j in range(L)]); pos += L
        return torch.cat(logits, 1), layout, torch.stack(states, 1)


def loss_cot(logits, tok, K):
    """Next-token loss on the mid and y blocks (positions predicting a mid or y digit)."""
    B, T, _ = logits.shape
    tgt = tok[:, 1:]
    pred = logits[:, :-1]
    pos = torch.arange(1, T, device=tok.device)
    block = (pos % (3 * L)) // L                                       # 0 = x, 1 = mid, 2 = y (of the target position)
    m = block >= 1
    return F.cross_entropy(pred[:, m].reshape(-1, V).float(), tgt[:, m].reshape(-1))


def y_positions(layout):
    """For each (k, j): the index of the position that predicts y[k][j] (the one before it)."""
    idx = {}
    for p, tag in enumerate(layout):
        if tag is not None and tag[0] == "y":
            idx[(tag[1], tag[2])] = p - 1
    return idx


def loss_thought(logits, layout, y):
    B, K, _ = y.shape
    ypos = y_positions(layout)
    ps = torch.tensor([ypos[(k, j)] for k in range(K) for j in range(L)], device=y.device)
    pred = logits[:, ps].reshape(B, K, L, V)
    return F.cross_entropy(pred.reshape(-1, V).float(), y.reshape(-1))


@torch.no_grad()
def per_demo_exact_thought(thinker, x, y, c):
    logits, layout, _ = thinker.run_thought(x, y, c)
    B, K, _ = y.shape
    ypos = y_positions(layout)
    ps = torch.tensor([ypos[(k, j)] for k in range(K) for j in range(L)], device=y.device)
    pred = logits[:, ps].reshape(B, K, L, V).argmax(-1)
    return (pred == y).all(-1).float().mean(0)                       # (K,)


@torch.no_grad()
def per_demo_exact_cot(thinker, x, mid, y):
    logits, tok = thinker.run_cot(x, mid, y)
    B, K, _ = y.shape
    T = tok.shape[1]
    pred = logits[:, :-1].argmax(-1)
    tgt = tok[:, 1:]
    pos = torch.arange(1, T, device=y.device)
    m = (pos % (3 * L)) // L == 2
    ok = (pred[:, m] == tgt[:, m]).reshape(B, K, L).all(-1)
    return ok.float().mean(0)


def trials(thinker, tasks, dev, K, c, mode, crit=0.8, n=256):
    out = []
    for t, pair in enumerate(tasks):
        x, mid, y, _ = make_batch_cot(n, K, tasks, torch.ones(len(tasks)), dev, fixed=t)
        acc = per_demo_exact_thought(thinker, x, y, c) if mode == "thought" else per_demo_exact_cot(thinker, x, mid, y)
        hit = (acc >= crit).nonzero()
        out.append((pair, int(hit[0]) if len(hit) else K, float(acc[-1])))
    return out


@torch.no_grad()
def probe(thinker, train, test, probs, dev, K, c, n=2048):
    """A least-squares linear map from each thought's state to the intermediate digits (one-hot), fitted on trained
    compositions and tested on held-out ones; per-thought digit accuracy, and the mean number of extra candidate digits
    (probe scores above 0.2 at a position, beyond the top one) as a crude count of superposed alternatives."""
    def collect(tasks, n):
        g = torch.Generator(device=dev).manual_seed(1)
        x, mid, y, _ = make_batch_cot(n, K, tasks, torch.ones(len(tasks)), dev, g)
        _l, _lay, st = thinker.run_thought(x, y, c)                 # (n, K, c, d)
        return st.float(), mid
    st_tr, mid_tr = collect(train, n)
    st_te, mid_te = collect(test, n)
    res = []
    for j in range(c):
        A = st_tr[:, :, j].reshape(-1, st_tr.shape[-1])
        A = torch.cat([A, torch.ones(A.shape[0], 1, device=dev)], 1)
        Y = F.one_hot(mid_tr.reshape(-1, L), V).float().reshape(A.shape[0], -1)
        W = torch.linalg.lstsq(A, Y).solution
        def score(st, mid):
            A2 = st[:, :, j].reshape(-1, st.shape[-1])
            A2 = torch.cat([A2, torch.ones(A2.shape[0], 1, device=dev)], 1)
            P = (A2 @ W).reshape(-1, L, V)
            acc = (P.argmax(-1) == mid.reshape(-1, L)).float().mean().item()
            extra = ((P > 0.2).sum(-1).float() - 1).clamp(min=0).mean().item()
            return acc, extra
        res.append(dict(thought=j, train=score(st_tr, mid_tr), held=score(st_te, mid_te)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c", type=int, default=1, help="continuous thoughts per demonstration")
    ap.add_argument("--res", default="std", choices=["std", "attnres"])
    ap.add_argument("--feedback", default="raw", choices=["raw", "operator"])
    ap.add_argument("--curriculum", type=int, default=1, help="1: stage 0 (chain of thought) first; 0: thoughts from the start")
    ap.add_argument("--stage_frac", type=float, default=0.3)
    ap.add_argument("--crit", type=float, default=0.8)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    torch.manual_seed(args.seed)
    train, test, probs, weights = build_tasks(args.seed)
    K = args.k
    model = Model(max_len=K * 3 * L + 2, pos="rope", res=args.res).to(dev)
    thinker = Thinker(model, args.feedback).to(dev)
    opt = torch.optim.AdamW(thinker.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext()
    g = torch.Generator(device=dev).manual_seed(args.seed)
    switch = int(args.stage_frac * args.steps) if args.curriculum else 0
    print(f"E31 coconut: c={args.c} res={args.res} feedback={args.feedback} curriculum={args.curriculum} (switch at step {switch}) "
          f"K={K} steps={args.steps} batch={args.batch} | {len(train)} trained, {len(test)} held out", flush=True)
    t0 = time.time()
    log = []
    for step in range(args.steps):
        x, mid, y, _ = make_batch_cot(args.batch, K, train, probs, dev, g)
        with amp:
            if step < switch:
                logits, tok = thinker.run_cot(x, mid, y)
                loss = loss_cot(logits, tok, K)
            else:
                logits, layout, _ = thinker.run_thought(x, y, args.c)
                loss = loss_thought(logits, layout, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(thinker.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            log.append((step, loss.item()))
            if step % 400 == 0:
                print(f"  step {step} loss {loss.item():.4f} ({'cot' if step < switch else 'thought'})", flush=True)
    print(f"trained in {time.time() - t0:.0f}s | final loss {loss.item():.4f}", flush=True)
    # spikes after the switch: a logged loss more than 1.5x the running minimum since the switch
    after = [(s, l) for s, l in log if s >= switch]
    spikes, run_min = 0, float("inf")
    for _s, l in after:
        if l > 1.5 * run_min and run_min < float("inf"):
            spikes += 1
        run_min = min(run_min, l)
    seen = trials(thinker, train, dev, K, args.c, "thought", args.crit)
    held = trials(thinker, test, dev, K, args.c, "thought", args.crit)
    seen_cot = trials(thinker, train, dev, K, args.c, "cot", args.crit) if args.curriculum else None
    held_cot = trials(thinker, test, dev, K, args.c, "cot", args.crit) if args.curriculum else None
    def summ(rows):
        return dict(solved=sum(1 for r in rows if r[1] < K), n=len(rows), mean_acc=sum(r[2] for r in rows) / len(rows))
    res = dict(thought=dict(trained=summ(seen), held=summ(held)),
               cot=(dict(trained=summ(seen_cot), held=summ(held_cot)) if seen_cot else None))
    pr = probe(thinker, train, test, probs, dev, K, args.c)
    print(f"THOUGHT mode: trained {res['thought']['trained']['solved']}/{len(seen)} (acc {res['thought']['trained']['mean_acc']:.2f}) | "
          f"held-out {res['thought']['held']['solved']}/{len(held)} (acc {res['thought']['held']['mean_acc']:.2f})", flush=True)
    if res["cot"]:
        print(f"COT mode:     trained {res['cot']['trained']['solved']}/{len(seen)} (acc {res['cot']['trained']['mean_acc']:.2f}) | "
              f"held-out {res['cot']['held']['solved']}/{len(held)} (acc {res['cot']['held']['mean_acc']:.2f})", flush=True)
    print(f"loss spikes after the switch: {spikes}; probe of the thought for the intermediate digits: " +
          "; ".join(f"thought {p['thought']}: train acc {p['train'][0]:.2f} (+{p['train'][1]:.2f} candidates), held acc {p['held'][0]:.2f} (+{p['held'][1]:.2f})" for p in pr), flush=True)
    print("HELD-OUT (thought mode): " + ", ".join(f"{NAMES[a]}>{NAMES[b]} {acc:.2f}" for (a, b), _t, acc in held), flush=True)
    if args.json:
        json.dump(dict(vars(args), results=res, spikes=spikes, probe=pr, loss_log=log,
                       held_rows=[(NAMES[a] + ">" + NAMES[b], t, acc) for (a, b), t, acc in held]), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
