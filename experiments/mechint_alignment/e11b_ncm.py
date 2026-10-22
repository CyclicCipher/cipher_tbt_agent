"""E11b: the fair NCM rerun. E11's diagnostic found three problems; each gets its fix (EXPERIMENT_E11.md):
1. codebook collapse      -> the quantizer's dead-code restarts (restart_threshold), per-level code use logged, and
                             tokenizer training until held-out reconstruction is >= 99.5% exact (chained segments);
2. the rare symbol lost   -> rare cells up-weighted x20 in the reconstruction loss (the agent's cell for the "state"
                             target, changed cells for the "delta" target);
3. no "copy, then edit"   -> a DELTA target: each cell is SAME or its new symbol, so copying the start state is the
                             default, as the one-step model's persistence prior makes it for that model.
Reuses E11's data, helpers and trained one-step model (runs/e11/stepmodel.pt, unchanged, not re-run).

    python e11b_ncm.py tokenizer --target delta --minutes 9      (repeat: resumes, stops when good or out of budget)
    python e11b_ncm.py generator --target delta --horizon 8 --minutes 9
    python e11b_ncm.py eval --target delta --horizon 8
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import e11_ncm as E
import ncm_model as nm
from canvas_jumps import N_CELLS, N_SYM, jumps

OUT = os.environ.get("E11B_OUT", "runs/e11b")
T_OFF = E.IN_VOCAB                    # delta targets use their own token ids, after the prefix's states and actions
RARE_W, RESTART = 20.0, 1.0
TOKENIZER_BUDGET, GENERATOR_BUDGET = 18, 18          # minutes, over chained segments
GOOD = dict(exact=0.995, rare=0.99)                  # the tokenizer is done when held-out reconstruction reaches these
dev = E.dev


def in_vocab(target):
    return E.IN_VOCAB + (1 + N_SYM if target == "delta" else 0)


def encode_target(start, end, target):
    if target == "state":
        return end
    lib = torch if torch.is_tensor(end) else np
    return lib.where(end == start, 0, end + 1) + T_OFF


def decode_target(start, tgt, target):
    if target == "state":
        return tgt
    d = tgt - T_OFF
    return torch.where(d <= 0, start, d - 1)


def rare(tgt, target):
    return ((tgt >= 4) & (tgt <= 7)) if target == "state" else (tgt != T_OFF)


def make_tokenizer(target):
    return nm.make_tokenizer(in_vocab(target), N_CELLS, restart_threshold=RESTART)


def targets(split, target):
    """(start, end) pairs whose end the tokenizer must encode: all states, or the deltas of 8- and 32-step jumps."""
    S, A = E.data(split)
    if target == "state":
        st = S.reshape(-1, N_CELLS)
        return st, st
    pairs = [jumps(S, A, h) for h in (8, 32)]
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[2] for p in pairs])


def utilization(vq, codes):
    out, o = [], 0
    for lv in vq.vq.levels:
        out.append(int(codes[:, o:o + lv].unique().numel()))
        o += lv
    return out


def log(name, rec):
    with open(f"{OUT}/{name}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def load(path, *objs):
    ck = torch.load(path, map_location=dev, weights_only=False)
    for o, k in zip(objs, ("model", "opt")):
        o.load_state_dict(ck[k])
    return ck


@torch.no_grad()
def tokenizer_check(vq, st, en, target):
    vq.eval()
    tgt = encode_target(st, en, target)
    codes = vq.encode(tgt)
    out = vq.decode(codes).argmax(-1)
    r = rare(tgt, target)
    return dict(exact=float((out == tgt).all(1).float().mean()), cell=float((out == tgt).float().mean()),
                rare=float((out == tgt)[r].float().mean()), use=utilization(vq, codes))


def train_tokenizer(target, minutes):
    name, path = f"tokenizer_{target}", f"{OUT}/tokenizer_{target}.pt"
    vq = make_tokenizer(target).to(dev)
    opt = torch.optim.AdamW(vq.parameters(), lr=1e-3, weight_decay=0.0)
    step, t_prev, done = 0, 0.0, False
    if os.path.exists(path):
        ck = load(path, vq, opt)
        step, t_prev, done = ck["step"], ck["t_train"], ck["done"]
    if done or t_prev >= 60 * TOKENIZER_BUDGET:
        print(f"{name}: already finished ({'good' if done else 'budget spent'})")
        return
    st, en = targets("train", target)
    tst, ten = targets("test", target)
    pick = np.random.default_rng(4).choice(len(ten), 2048, replace=False)
    tst, ten = (torch.from_numpy(x[pick].astype(np.int64)).to(dev) for x in (tst, ten))
    rng, t0 = np.random.default_rng(step), time.time()
    while True:
        idx = rng.integers(0, len(en), 128)
        tgt = encode_target(*(torch.from_numpy(x[idx].astype(np.int64)).to(dev) for x in (st, en)), target)
        vq.train()
        loss, rec = nm.tokenizer_loss(vq, tgt, 1.0 + (RARE_W - 1.0) * rare(tgt, target).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        t_seg = time.time() - t0
        if step % 300 == 0 or t_seg > 60 * minutes:
            chk = tokenizer_check(vq, tst, ten, target)
            done = chk["exact"] >= GOOD["exact"] and chk["rare"] >= GOOD["rare"]
            log(name, dict(step=step, min=round((t_prev + t_seg) / 60, 2), loss=round(float(loss), 4), **chk))
            if done or t_seg > 60 * minutes or t_prev + t_seg >= 60 * TOKENIZER_BUDGET:
                break
    torch.save(dict(model=vq.state_dict(), opt=opt.state_dict(), step=step, t_train=t_prev + t_seg, done=done), path)


def load_tokenizer(target):
    vq = make_tokenizer(target).to(dev)
    vq.load_state_dict(torch.load(f"{OUT}/tokenizer_{target}.pt", map_location=dev, weights_only=False)["model"])
    for p in vq.parameters():
        p.requires_grad_(False)
    return vq.eval()


def train_generator(target, horizon, minutes):
    name, path = f"generator_{target}_h{horizon}", f"{OUT}/generator_{target}_h{horizon}.pt"
    vq = load_tokenizer(target)
    S, A = E.data("train")
    start, acts, end = jumps(S, A, horizon)
    pre = torch.from_numpy(E.prefix_tokens(start, acts))
    tgt = torch.from_numpy(encode_target(start, end, target).astype(np.int64))
    net = nm.make_generator(vq, pre.shape[1], **E.GEN_SIZE).to(dev)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
    step, t_prev = 0, 0.0
    if os.path.exists(path):
        ck = load(path, net, opt)
        step, t_prev = ck["step"], ck["t_train"]
    if t_prev >= 60 * GENERATOR_BUDGET:
        print(f"{name}: budget spent")
        return
    w = nm.level_weights(vq.vq.levels, device=dev)
    rng, t0 = np.random.default_rng(1000 + step), time.time()
    while True:
        idx = rng.integers(0, len(tgt), E.GEN_BATCH)
        net.train()
        with E.amp():
            loss, acc = nm.generator_loss(net, vq, pre[idx].to(dev), tgt[idx].to(dev), w=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        t_seg = time.time() - t0
        if step % 300 == 0 or t_seg > 60 * minutes:
            log(name, dict(step=step, min=round((t_prev + t_seg) / 60, 2), loss=round(float(loss), 4),
                           code_acc=round(float(acc), 4)))
            if t_seg > 60 * minutes or t_prev + t_seg >= 60 * GENERATOR_BUDGET:
                break
    torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, t_train=t_prev + t_seg), path)


@torch.no_grad()
def evaluate(target, horizon, n=2048):
    S, A = E.data("test")
    start, acts, end = jumps(S, A, horizon)
    pick = np.random.default_rng(3).choice(len(end), n, replace=False)           # E11's 2,048 futures
    start, acts, end = (torch.from_numpy(x[pick].astype(np.int64)).to(dev) for x in (start, acts, end))
    pre = torch.cat([start, acts + N_SYM], 1)
    tgt = encode_target(start, end, target)
    vq = load_tokenizer(target)
    net = nm.make_generator(vq, pre.shape[1], **E.GEN_SIZE).to(dev)
    net.load_state_dict(torch.load(f"{OUT}/generator_{target}_h{horizon}.pt", map_location=dev,
                                   weights_only=False)["model"])
    net.eval()
    step = E.make_stepmodel().to(dev)
    step.load_state_dict(torch.load("runs/e11/stepmodel.pt", map_location=dev, weights_only=False)["model"])
    step.eval()
    chunks = lambda f, b: torch.cat([f(i, i + b) for i in range(0, n, b)])
    with E.amp():
        true = chunks(lambda i, j: vq.encode(tgt[i:j]), 256)
        dec = lambda codes: chunks(lambda i, j: decode_target(start[i:j], vq.decode(codes[i:j]).argmax(-1), target), 256)
        tf = chunks(lambda i, j: net(torch.cat([vq.encoder.wte(pre[i:j]),
                                                net.build_l0_query_input(j - i, torch.float32),
                                                vq.vq.idx_to_ncm_input(true[i:j])], 1)).argmax(-1), 128)
        gen = chunks(lambda i, j: net.generate(None, pre[i:j], temperature=0.0, top_k=0, top_p=1.0), 64)
        preds = {"copy": start, "tokenizer_ceiling": dec(true), "ncm_teacher_forced": dec(tf), "ncm": dec(gen),
                 "step": chunks(lambda i, j: E.step_rollout(step, start[i:j], acts[i:j]), 256)}
    per_level = lambda codes: [round(float(x), 3) for x in torch.stack(
        [(codes[:, o:o + lv] == true[:, o:o + lv]).float().mean()
         for o, lv in zip(np.cumsum([0] + vq.vq.levels[:-1]), vq.vq.levels)])]
    rec = dict(target=target, horizon=horizon, n=n, levels_use=utilization(vq, true),
               code_acc_teacher_forced=per_level(tf), code_acc_free=per_level(gen))
    for name, p in preds.items():
        rec[name] = E.scores(p, start, end)

    def ncm_fn(p):
        with E.amp():
            return decode_target(start[:len(p)], vq.decode(net.generate(None, p, temperature=0.0, top_k=0,
                                                                          top_p=1.0)).argmax(-1), target)
    for b in (16, 144):
        rec[f"sec_ncm_b{b}"] = E.timed(lambda: ncm_fn(pre[:b]))
        rec[f"sec_step_b{b}"] = E.timed(lambda: E.step_rollout(step, start[:b], acts[:b]))
    log(f"eval_{target}_h{horizon}", rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("tokenizer", "generator", "eval"))
    p.add_argument("--target", choices=("state", "delta"), default="delta")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--minutes", type=float, default=9)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    {"tokenizer": lambda: train_tokenizer(a.target, a.minutes),
     "generator": lambda: train_generator(a.target, a.horizon, a.minutes),
     "eval": lambda: evaluate(a.target, a.horizon)}[a.stage]()


if __name__ == "__main__":
    main()
