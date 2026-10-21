"""E11: Next-Concept Modeling (ncm_model.py, the authors' architecture) against our one-step world model
(worldmodel.py), for imagining the Canvas H steps ahead under random action sequences (what the planner needs).

    python e11_ncm.py tokenizer --minutes 3
    python e11_ncm.py generator --horizon 8 --minutes 7
    python e11_ncm.py generator --horizon 32 --minutes 7
    python e11_ncm.py stepmodel --minutes 7
    python e11_ncm.py eval --horizon 8
    python e11_ncm.py eval --horizon 32
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import ncm_model as nm
from canvas_jumps import N_CELLS, N_SYM, SCREEN, SPEC, jumps, trajectories
from worldmodel import WorldModel

OUT = "runs/e11"
TRAIN_EPISODES, TEST_SEED0, TEST_EPISODES = 2000, 10**6, 100
IN_VOCAB = N_SYM + SPEC.n_actions                    # state symbols, then action tokens
GEN_SIZE = dict(n_embd=192, n_layer=4, n_head=6)     # sized for the 4 GB GPU (the repo's attention mask is float32,
GEN_BATCH, STEP_BATCH = 24, 64                       # so under bf16 attention falls back to the memory-hungry kernel)
dev = torch.device("cuda")


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def data(split):
    path = f"{OUT}/traj_{split}.npz"
    if not os.path.exists(path):
        S, A = trajectories(0, TRAIN_EPISODES) if split == "train" else trajectories(TEST_SEED0, TEST_EPISODES)
        np.savez(path, S=S, A=A)
    z = np.load(path)
    return z["S"], z["A"]


def prefix_tokens(start, acts):
    return np.concatenate([start, acts.astype(np.int64) + N_SYM], 1)


def batches(rng, n, size):
    while True:
        yield rng.integers(0, n, size)


def log(name, rec):
    with open(f"{OUT}/{name}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def train_tokenizer(minutes):
    S, _ = data("train")
    St, _ = data("test")
    states = torch.from_numpy(S.reshape(-1, N_CELLS).astype(np.int64))
    test = torch.from_numpy(St.reshape(-1, N_CELLS)[::7][:2048].astype(np.int64)).to(dev)
    vq = nm.make_tokenizer(IN_VOCAB, N_CELLS).to(dev)
    opt = torch.optim.AdamW(vq.parameters(), lr=1e-3, weight_decay=0.0)
    rng, t0, step = np.random.default_rng(0), time.time(), 0
    for idx in batches(rng, len(states), 128):
        vq.train()
        loss, rec = nm.tokenizer_loss(vq, states[idx].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0 or time.time() - t0 > 60 * minutes:
            vq.eval()
            with torch.no_grad():
                out = vq.decode(vq.encode(test)).argmax(-1)
            log("tokenizer", dict(step=step, sec=round(time.time() - t0), loss=round(float(loss), 4),
                                  rec_ce=round(float(rec), 4), cell_acc=round(float((out == test).float().mean()), 5),
                                  exact=round(float((out == test).all(1).float().mean()), 4)))
            if time.time() - t0 > 60 * minutes:
                break
    torch.save(dict(model=vq.state_dict()), f"{OUT}/tokenizer.pt")


def load_tokenizer():
    vq = nm.make_tokenizer(IN_VOCAB, N_CELLS).to(dev)
    vq.load_state_dict(torch.load(f"{OUT}/tokenizer.pt", map_location=dev, weights_only=False)["model"])
    return vq.eval()


def train_generator(horizon, minutes):
    S, A = data("train")
    start, acts, end = jumps(S, A, horizon)
    pre = torch.from_numpy(prefix_tokens(start, acts))
    tgt = torch.from_numpy(end.astype(np.int64))
    vq = load_tokenizer()
    for p in vq.parameters():
        p.requires_grad_(False)
    net = nm.make_generator(vq, pre.shape[1], **GEN_SIZE).to(dev)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
    w = nm.level_weights(vq.vq.levels, device=dev)
    rng, t0, step = np.random.default_rng(1), time.time(), 0
    for idx in batches(rng, len(tgt), GEN_BATCH):
        net.train()
        with amp():
            loss, acc = nm.generator_loss(net, vq, pre[idx].to(dev), tgt[idx].to(dev), w=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0 or time.time() - t0 > 60 * minutes:
            log(f"generator_h{horizon}", dict(step=step, sec=round(time.time() - t0), loss=round(float(loss), 4),
                                              code_acc=round(float(acc), 4)))
            if time.time() - t0 > 60 * minutes:
                break
    torch.save(dict(model=net.state_dict()), f"{OUT}/generator_h{horizon}.pt")


def make_stepmodel():
    return WorldModel(SPEC, d=192, layers=6, heads=4, pos="local", persist=5.0, radius=1)


def train_stepmodel(minutes):
    S, A = data("train")
    cur = torch.from_numpy(S[:, :-1].reshape(-1, N_CELLS, 1).astype(np.int64))
    nxt = torch.from_numpy(S[:, 1:].reshape(-1, N_CELLS).astype(np.int64))
    act = torch.from_numpy(A.reshape(-1).astype(np.int64))
    model = make_stepmodel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    glob = torch.zeros(STEP_BATCH, 1, dtype=torch.long, device=dev)
    rng, t0, step = np.random.default_rng(2), time.time(), 0
    for idx in batches(rng, len(act), STEP_BATCH):
        model.train()
        with amp():
            cl, _, _ = model(cur[idx].to(dev), glob, act[idx].to(dev))
        loss = F.cross_entropy(cl[0].reshape(-1, N_SYM).float(), nxt[idx].to(dev).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0 or time.time() - t0 > 60 * minutes:
            log("stepmodel", dict(step=step, sec=round(time.time() - t0), loss=round(float(loss), 5)))
            if time.time() - t0 > 60 * minutes:
                break
    torch.save(dict(model=model.state_dict()), f"{OUT}/stepmodel.pt")


@torch.no_grad()
def step_rollout(model, start, acts):
    """H one-step passes, each fed the model's own previous prediction."""
    cells = start[..., None]
    glob = torch.zeros(len(start), 1, dtype=torch.long, device=start.device)
    for h in range(acts.shape[1]):
        with amp():
            cl, _, _ = model(cells, glob, acts[:, h])
        cells = cl[0].argmax(-1)[..., None]
    return cells[..., 0]


def scores(pred, start, end):
    keep = end != SCREEN
    changed = (start != end) & keep
    agent = end >= 4
    agent = agent & (end != SCREEN) & (end <= 7)
    return dict(
        cell_acc=float(((pred == end) | ~keep).float().mean()),
        exact=float(((pred == end) | ~keep).all(1).float().mean()),
        changed_acc=float((pred == end)[changed].float().mean()) if changed.any() else None,
        agent_acc=float(((pred == end) | ~agent).all(1).float().mean()),
    )


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def evaluate(horizon, n=2048):
    S, A = data("test")
    start, acts, end = jumps(S, A, horizon)
    pick = np.random.default_rng(3).choice(len(end), n, replace=False)
    start, acts, end = (torch.from_numpy(x[pick].astype(np.int64)).to(dev) for x in (start, acts, end))
    pre = torch.cat([start, acts + N_SYM], 1)
    vq = load_tokenizer()
    net = nm.make_generator(vq, pre.shape[1], **GEN_SIZE).to(dev)
    net.load_state_dict(torch.load(f"{OUT}/generator_h{horizon}.pt", map_location=dev, weights_only=False)["model"])
    net.eval()
    model = make_stepmodel().to(dev)
    model.load_state_dict(torch.load(f"{OUT}/stepmodel.pt", map_location=dev, weights_only=False)["model"])
    model.eval()

    def ncm(p):
        with amp():
            return nm.generate(net, vq, p)

    preds = {"copy": start, "ncm": torch.cat([ncm(pre[i:i + 64]) for i in range(0, n, 64)]),
             "step": torch.cat([step_rollout(model, start[i:i + 256], acts[i:i + 256]) for i in range(0, n, 256)])}
    rec = dict(horizon=horizon, n=n, ncm_params=nm.n_params(net), tokenizer_params=nm.n_params(vq),
               step_params=nm.n_params(model), ncm_passes=len(vq.vq.levels), step_passes=horizon,
               recon_exact=float((vq.decode(vq.encode(end)).argmax(-1) == end).all(1).float().mean()))
    for name, p in preds.items():
        rec[name] = scores(p, start, end)
    for b in (16, 144):                     # 144 = one planning decision (3 rounds x 48 sequences)
        rec[f"sec_ncm_b{b}"] = timed(lambda: ncm(pre[:b]))
        rec[f"sec_step_b{b}"] = timed(lambda: step_rollout(model, start[:b], acts[:b]))
    log(f"eval_h{horizon}", rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("tokenizer", "generator", "stepmodel", "eval"))
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--minutes", type=float, default=7)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    {"tokenizer": lambda: train_tokenizer(a.minutes), "generator": lambda: train_generator(a.horizon, a.minutes),
     "stepmodel": lambda: train_stepmodel(a.minutes), "eval": lambda: evaluate(a.horizon)}[a.stage]()


if __name__ == "__main__":
    main()
