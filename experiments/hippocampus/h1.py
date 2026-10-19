"""Experiment H1: store + read on synthetic facts (EXPERIMENT_H1.md).

Arms:  A  weights only, no memory. The test world's paired facts are trained in both directions, then its one-way facts
          forward for e epochs (e = exposures per fact).
       B  memory with the sparse union index. A reader is trained on 20 other worlds (same entities, different random
          facts, so the weights cannot hold the facts and must read them); the test world's facts are then WRITTEN
          ONCE, forward only, through the surprise gate.
       C  memory with the dense, order-free index (the control for pattern separation).

Usage:  python h1.py --arm B --seed 0          -> runs/h1/B_s0.json
        python h1.py --arm A --e 8 --seed 0    -> runs/h1/A_e8_s0.json
Refuses to run if the result file exists (never re-run an unchanged experiment).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "intelligent_constraints"))
from data import make_facts                       # noqa: E402  the synthetic facts (reused)
from model import Net                             # noqa: E402  the tiny transformer (reused)
from store import Store, DEV                      # noqa: E402

StoreClass = Store

N_ENT, N_REL = 1000, 6
BASE = N_ENT + 2 * N_REL                          # entities + relations + converse tokens
MEM, SEP, PAD = BASE, BASE + 1, BASE + 2
VOCAB = BASE + 3
R, THETA, TAU = 12, 20.0, 2.0                     # episodes read, overlap threshold (bits), write gate (bits)
CTX = 1 + 4 * R + 2
TRAIN_WORLDS, TEST_WORLD, LEAK_WORLD = range(1, 21), 0, 1
MODEL = dict(d=128, n_layer=4, n_head=4, m=512)


def rel(r):
    return N_ENT + r


def conv(r):
    return N_ENT + N_REL + r


def episodes(F):
    return torch.tensor([(a, rel(r), b) for a, r, b in F.facts])


def questions(F):
    """One training example per fact and direction: forward [a, r] -> b, reverse [b, r°] -> a."""
    P = [(a, rel(r)) for a, r, b in F.facts] + [(b, conv(r)) for a, r, b in F.facts]
    Y = [b for a, r, b in F.facts] + [a for a, r, b in F.facts]
    return torch.tensor(P), torch.tensor(Y)


def eval_sets(F):
    """The test world's evaluation sets, identical for every arm: (prompt, correct set, correct episodes' tokens)."""
    fwd, rev = {}, {}
    for a, r, b in F.facts:
        fwd.setdefault((a, r), set()).add(b)
        rev.setdefault((r, b), set()).add(a)
    oneway = sorted({(a, r) for a, r, b in F.facts if (r, b) not in F.paired})
    sets = {"fwd_oneway": [((a, rel(r)), sorted(fwd[(a, r)]), [(a, rel(r), b) for b in fwd[(a, r)]])
                           for a, r in oneway]}
    for name in ("rev_oneway_clean", "rev_paired"):
        sets[name] = []
        for (b, ct), correct, _ in F.evals[name]:
            r = ct - N_ENT - N_REL
            sets[name].append(((b, ct), correct, [(a, rel(r), b) for a in correct]))
    return sets


def contexts(store, prompts, gen, oracle=None):
    """[MEM] ep_R [SEP] ... ep_1 [SEP] q0 q1 (best episode last). oracle: per question, the correct episodes' tokens;
    if none of them was retrieved, one is put in place of the worst-ranked episode."""
    prompts = prompts.to(DEV)
    ids = store.retrieve(prompts, R, THETA, gen)
    if store.read_order in ("shuffled", "mixed"):            # H1c: rank position carries no information
        perm = torch.argsort(torch.rand(ids.shape, device=DEV, generator=gen), dim=1)
        shuffled = torch.gather(ids, 1, perm)
        if store.read_order == "mixed":                      # H1d: a random half of the questions shuffled
            half = torch.rand(ids.shape[0], device=DEV, generator=gen) < 0.5
            ids = torch.where(half[:, None], shuffled, ids)
        else:
            ids = shuffled
    if oracle is not None:
        for i, eps in enumerate(oracle):
            rows = [j for e in eps for j in store.where.get(e, [])]
            if rows and not any(int(j) in rows for j in ids[i].tolist()):
                ids[i, R - 1] = rows[0]
    if len(store) == 0:
        ep = torch.full((*ids.shape, 3), PAD, device=DEV)
    else:
        ep = torch.where(ids[..., None] >= 0, store.rows[ids.clamp(min=0)], torch.full_like(ids[..., None], PAD))
    sep = torch.where(ids >= 0, torch.full_like(ids, SEP), torch.full_like(ids, PAD))[..., None]
    mem = torch.cat([ep, sep], 2).flip(1).reshape(len(prompts), 4 * R)
    return torch.cat([torch.full((len(prompts), 1), MEM, device=DEV), mem, prompts], 1), ids


def score(logits, sets):
    prob = logits.float()[:, :N_ENT].softmax(-1).cpu()
    top = prob.argmax(-1)
    hits = sum(int(top[i]) in c for i, (_, c, _) in enumerate(sets)) / len(sets)
    mass = sum(prob[i, c].sum().item() for i, (_, c, _) in enumerate(sets)) / len(sets)
    return round(hits, 4), round(mass, 4)


def classify_reverse(store, s, ids, logits):
    """H1c: what kind of answer did the reader give to each reverse question (b, r°)?"""
    pred = logits.float()[:, :N_ENT].argmax(-1).tolist()
    rows = store.rows.tolist() if len(store) else []
    c = dict(correct=0, rel_ok_entity_wrong=0, entity_ok_rel_wrong=0, other_in_context=0, not_in_context=0)
    for i, ((b, ct), correct, _) in enumerate(s):
        r_tok, p = ct - N_REL, pred[i]
        read = [rows[j] for j in ids[i].tolist() if j >= 0]
        if p in correct:
            c["correct"] += 1
        elif any(t[0] == p and t[1] == r_tok for t in read):
            c["rel_ok_entity_wrong"] += 1
        elif any(t[0] == p and t[2] == b for t in read):
            c["entity_ok_rel_wrong"] += 1
        elif any(p in t for t in read):
            c["other_in_context"] += 1
        else:
            c["not_in_context"] += 1
    return {k: round(v / len(s), 4) for k, v in c.items()}


@torch.no_grad()
def eval_mem(net, store, sets, gen, oracle=False, classify=False):
    net.eval()
    out = {}
    for name, s in sets.items():
        P = torch.tensor([p for p, _, _ in s])
        ctx, ids = contexts(store, P, gen, oracle=[f for _, _, f in s] if oracle else None)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            logits = net(ctx)
        hits, mass = score(logits, s)
        good = [{j for e in f for j in store.where.get(e, [])} for _, _, f in s]
        recall = sum(any(int(j) in good[i] for j in ids[i].tolist()) for i in range(len(s))) / len(s)
        out[name] = dict(hits=hits, mass=mass, recall=round(recall, 4))
        if classify and name.startswith("rev"):
            out[name]["classes"] = classify_reverse(store, s, ids, logits)
    net.train()
    return out


@torch.no_grad()
def eval_plain(net, sets):
    net.eval()
    out = {}
    for name, s in sets.items():
        with torch.autocast(DEV, dtype=torch.bfloat16):
            logits = net(torch.tensor([p for p, _, _ in s], device=DEV))
        hits, mass = score(logits, s)
        out[name] = dict(hits=hits, mass=mass)
    net.train()
    return out


def opt_for(net, lr, fused=False):
    return torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01, fused=fused)


def step(net, opt, X, Y, lr, t, total=None):
    """total=None: constant lr after warm-up (H1); otherwise cosine decay to 10% at step `total` (H1b)."""
    f = min(1.0, t / 200)
    if total:
        f *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, total) / total))
    for grp in opt.param_groups:
        grp["lr"] = lr * f
    with torch.autocast(DEV, dtype=torch.bfloat16):
        loss = Fn.cross_entropy(net(X).float(), Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def surprise_bits(net, store, facts, gen):
    """-log2 p(b | memory read for [a, r]) for each fact (a, r, b)."""
    P = torch.tensor([(a, rel(r)) for a, r, b in facts])
    Y = torch.tensor([b for a, r, b in facts], device=DEV)
    ctx, _ = contexts(store, P, gen)
    with torch.autocast(DEV, dtype=torch.bfloat16):
        lp = net(ctx).float().log_softmax(-1)
    return (-lp[torch.arange(len(facts)), Y] / math.log(2)).cpu()


def gated_write(net, store, facts, gen, chunk=256, familiarity=False, stats=None):
    """Present facts in order, chunk by chunk; write a fact only if its surprise under the current store exceeds TAU.
    familiarity=True (H1b) first skips facts the store recognises as already stored. Returns the facts written;
    `stats` (a dict) collects how many the surprise gate alone would have written."""
    written = []
    for i in range(0, len(facts), chunk):
        part = facts[i:i + chunk]
        s = surprise_bits(net, store, part, gen)
        fam = store.familiar(torch.tensor([(a, rel(r), b) for a, r, b in part])) if familiarity else [False] * len(part)
        if stats is not None:
            stats["surprise_only"] = stats.get("surprise_only", 0) + int((s > TAU).sum())
            stats["familiar"] = stats.get("familiar", 0) + sum(fam)
        keep = [j for j in range(len(part)) if s[j] > TAU and not fam[j]]
        if keep:
            store.write(torch.tensor([(part[j][0], rel(part[j][1]), part[j][2]) for j in keep]), s[keep].tolist())
        written += [part[j] for j in keep]
    return written


def store_factory(expand=False, read_order="ranked"):
    """A Store constructor whose stores all read the same way: with or without the learned expansion, ranked or
    shuffled."""
    table = None
    if expand:
        from expand import expansion_table
        table = expansion_table()

    def make(*a, **k):
        st_ = StoreClass(*a, **k)
        st_.read_order = read_order
        if table is not None:
            st_.expansion = table
        return st_
    return make


def training_set(Store, index, gen):
    """The reader's training contexts: worlds 1-20, each written whole (ungated) into its own store, every fact asked in
    both directions."""
    X, Y = [], []
    for w in TRAIN_WORLDS:
        F = make_facts(seed=w)
        st = Store(VOCAB, index)
        st.write(episodes(F))
        P, A = questions(F)
        X.append(contexts(st, P, gen)[0])
        Y.append(A.to(DEV))
    return torch.cat(X), torch.cat(Y)


SLOTS = [0] + [1, 2, 3, 4] * R + [5, 6]          # H1b: [MEM], (a, r, b, [SEP]) per episode, question tokens


def run_memory(index, seed, steps=3500, lr=1e-3, batch=256, eval_every=500, fast=False, cosine=False, slots=False,
               familiarity=False, save=None, expand=False, read_order="ranked", init_from=None):
    """H1 defaults; H1b: fast (torch.compile + fused AdamW), cosine (decay to 10%), slots (slot embedding),
    familiarity (recognition before the surprise gate)."""
    t0 = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Store = store_factory(expand, read_order)                  # every store in this run reads the same way (H1c)
    X, Y = training_set(Store, index, gen)
    F0 = make_facts(seed=TEST_WORLD)
    sets = eval_sets(F0)
    full = Store(VOCAB, index)                                 # for the training curve only: every fact written
    full.write(episodes(F0))
    net = Net(VOCAB, ctx=CTX, seg_ids=SLOTS if slots else None, **MODEL).to(DEV)
    if init_from:                                              # H1d: warm start from a trained reader
        net.load_state_dict(torch.load(init_from, map_location=DEV))
    opt = opt_for(net, lr, fused=fast)
    train_net = torch.compile(net) if fast else net            # evaluation uses the uncompiled module (same weights)
    log = []
    for t in range(1, steps + 1):
        i = torch.randint(0, len(X), (batch,), device=DEV, generator=gen)
        loss = step(train_net, opt, X[i], Y[i], lr, t, total=steps if cosine else None)
        if t % eval_every == 0 or t == steps:
            log.append(dict(step=t, loss=round(loss, 4), **eval_mem(net, full, sets, gen)))
    # the test world: a fresh store, facts presented once in random order through the surprise gate
    order = torch.randperm(len(F0.facts), generator=torch.Generator().manual_seed(seed)).tolist()
    facts = [F0.facts[j] for j in order]
    store = Store(VOCAB, index)
    st1, st2 = {}, {}
    w1 = gated_write(net, store, facts, gen, familiarity=familiarity, stats=st1)
    final = eval_mem(net, store, sets, gen, classify=True)    # the store after one presentation
    final_oracle = eval_mem(net, store, sets, gen, oracle=True)
    w2 = gated_write(net, store, facts, gen, familiarity=familiarity, stats=st2)   # presented again
    kinds = F0.kinds
    res = dict(arm={"sparse": "B", "dense": "C"}[index], index=index, seed=seed, steps=steps, lr=lr, batch=batch,
               R=R, theta=THETA, tau=TAU, model=MODEL, train_examples=len(X), log=log,
               options=dict(fast=fast, cosine=cosine, slots=slots, familiarity=familiarity, expand=expand,
                            read_order=read_order, init_from=str(init_from) if init_from else None),
               final=final, final_oracle=final_oracle, leak_control=None,
               gate=dict(facts=len(facts), written_first=len(w1), written_again=len(w2),
                         again_by_kind={k: sum(kinds[f[1]] == k for f in w2) for k in sorted(set(kinds))},
                         first_stats=st1, again_stats=st2,
                         rows=len(store)),
               store=dict(row_bytes=store.row_bytes(),
                          key_cache_bytes=(store.keys.shape[1] // 8 if index == "sparse" else 2 * store.keys.shape[1])))
    wrong = Store(VOCAB, index)                                # leak control: another world's facts in the store
    wrong.write(episodes(make_facts(seed=LEAK_WORLD)))
    res["leak_control"] = eval_mem(net, wrong, sets, gen)
    if save:                                                   # the reader's weights (git-ignored), for H2/H3
        torch.save(net.state_dict(), save)
    res["secs"] = round(time.time() - t0, 1)
    return res


def run_weights(e, seed, steps1=3000, lr=1e-3, batch=256):
    t0 = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    F0 = make_facts(seed=TEST_WORLD)
    sets = eval_sets(F0)
    pf = [((a, rel(r)), b) for a, r, b in F0.facts if (r, b) in F0.paired]
    pr = [((b, conv(r)), a) for a, r, b in F0.facts if (r, b) in F0.paired]
    ow = [((a, rel(r)), b) for a, r, b in F0.facts if (r, b) not in F0.paired]
    PX = torch.tensor([p for p, _ in pf + pr], device=DEV)
    PY = torch.tensor([y for _, y in pf + pr], device=DEV)
    OX = torch.tensor([p for p, _ in ow], device=DEV)
    OY = torch.tensor([y for _, y in ow], device=DEV)
    net = Net(BASE, ctx=2, **MODEL).to(DEV)
    opt = opt_for(net, lr)
    log = []
    for t in range(1, steps1 + 1):                             # phase 1: the paired facts, both directions
        i = torch.randint(0, len(PX), (batch,), device=DEV, generator=gen)
        loss = step(net, opt, PX[i], PY[i], lr, t)
        if t % 500 == 0:
            log.append(dict(phase=1, step=t, loss=round(loss, 4), **eval_plain(net, sets)))
    t = steps1
    half = batch // 2
    for ep in range(1, e + 1):                                 # phase 2: each one-way fact once per epoch, + rehearsal
        perm = torch.randperm(len(OX), device=DEV, generator=gen)
        for k in range(0, len(OX), half):
            j = perm[k:k + half]
            i = torch.randint(0, len(PX), (half,), device=DEV, generator=gen)
            t += 1
            loss = step(net, opt, torch.cat([OX[j], PX[i]]), torch.cat([OY[j], PY[i]]), lr, t)
        if ep <= 8 or ep % 8 == 0:
            log.append(dict(phase=2, epoch=ep, step=t, loss=round(loss, 4), **eval_plain(net, sets)))
    return dict(arm="A", e=e, seed=seed, steps1=steps1, lr=lr, batch=batch, model=MODEL, log=log,
                final=log[-1], secs=round(time.time() - t0, 1))


H1B = {"B2": dict(fast=True, cosine=True, familiarity=True),               # H1b: stronger training + recognition
       "B3": dict(fast=True, cosine=True, familiarity=True, slots=True)}   # ... + the slot embedding
H1B_STEPS = 4500

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "B2", "B3"], required=True)
    ap.add_argument("--e", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="smoke tests only (no file written)")
    a = ap.parse_args()
    name = f"A_e{a.e}_s{a.seed}" if a.arm == "A" else f"{a.arm}_s{a.seed}"
    out = HERE / "runs" / ("h1b" if a.arm in H1B else "h1") / f"{name}.json"
    if a.steps == 0 and out.exists():
        print(f"{out} exists: not re-run")
        raise SystemExit
    if a.arm == "A":
        r = run_weights(a.e, a.seed, **({"steps1": a.steps} if a.steps else {}))
    elif a.arm in H1B:
        r = run_memory("sparse", a.seed, steps=a.steps or H1B_STEPS, **H1B[a.arm])
        r["arm"] = a.arm
    else:
        r = run_memory("sparse" if a.arm == "B" else "dense", a.seed, **({"steps": a.steps} if a.steps else {}))
    print(json.dumps({k: r[k] for k in r if k not in ("log",)}, indent=1))
    if a.steps == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(r, open(out, "w"), indent=1)
