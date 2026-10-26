"""E13: the recurring-worlds stream (EXPERIMENT_E13.md). What is episodic memory worth, and how close does a simple one
get to the ideal?

Streams of 48 episodes (512 steps each, random play, fresh layouts): 12 distinct physics, each recurring 4 times in
random order. Four predictors, scored by encounter (1st, 2nd, 3rd, 4th time a physics is met):
- ideal, no memory:  E12's exact observer of robot motion, fresh every episode;
- ideal, memory:     the same observer, starting each episode from the stream's exact recurrence prior, a mixture of
                     a uniform "new physics" and the final posteriors of the earlier episodes;
- learner, no memory: E8c's L-Ac (adaptation with the calibrated prior), fresh every episode (as in E8);
- learner, library:  L-Ac plus a hippocampus-lite: every finished episode's adapted fast weights go into a library;
                     each chunk is predicted by the Bayesian mixture of the library entries and the fresh learner,
                     weighted by a Chinese-restaurant prior and by each candidate's evidence on this episode so far.

    python -m lab.e13.e13_stream data ; python -m lab.e13.e13_stream ideal ; python -m lab.e13.e13_stream learner ; python -m lab.e13.e13_stream report
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from worlds import rollouts
from agents.ideal_observer import SHAPE, MotionObserver, robot_state
from lab.e2.e2_world import load, losses, robot_mask
from lab.e8.e8_hier import Runner, world_index
from lab.e8.e8_score import ChunkScorer
from nets.family import FamilyModel
from worlds.village import SPEC
from worlds.village_events import EVENTS

OUT, LENGTH, CHUNK = "runs/e13", 512, 8
STREAMS, DISTINCT, REPEATS = 8, 12, 4
EPISODES = DISTINCT * REPEATS
ALPHA = 1.0                                              # the learner's Chinese-restaurant concentration
MOVE, TURN = EVENTS.index("move"), EVENTS.index("turn")


def schedule(s):
    """Stream s: the physics id of each episode (each of 12 ids 4 times, random order) and the physics seeds."""
    order = np.random.default_rng([s, 13]).permutation(np.repeat(np.arange(DISTINCT), REPEATS))
    return order, [3_000_000_000 + 1000 * s + int(d) for d in order]


def encounters(order):
    seen, out = {}, []
    for d in order:
        seen[d] = seen.get(d, 0) + 1
        out.append(seen[d])
    return np.array(out)


# ---- data -------------------------------------------------------------------------------------------------------------
def gen_stream(s):
    order, pseeds = schedule(s)
    out = rollouts.run_lanes(("village", 4_000_000_000 + 1000 * s, 1, EPISODES * LENGTH, s, (1.0,), "train", LENGTH, pseeds))
    out = {k: v[0] for k, v in out.items()}
    k = np.repeat(np.arange(EPISODES), LENGTH)
    out.update(stream=np.full(len(k), s, np.int16), episode_k=k.astype(np.int16), physics_id=order[k].astype(np.int16),
               encounter=encounters(order)[k].astype(np.int16))
    return out


def make_data():
    with Pool(STREAMS) as pool:
        parts = pool.map(gen_stream, range(STREAMS))
    z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(f"{OUT}/stream.npz", **z)
    print(f"{len(z['act']):,} transitions, {STREAMS} streams x {EPISODES} episodes", flush=True)


def stream_worlds(z, s):
    """Stream s's episodes as [EPISODES, LENGTH] row indices, in stream order."""
    rows = np.nonzero(z["stream"] == s)[0]
    w = world_index({k: z[k][rows] for k in ("episode", "tstep")}, LENGTH)
    return rows[w]


# ---- ideal observers --------------------------------------------------------------------------------------------------
def ideal_stream(s):
    """Both ideal observers over stream s. The memory observer's prior is the stream's exact recurrence prior: a new
    physics with probability (copies of unseen physics left) / (episodes left), else a physics already seen, in
    proportion to its copies left, represented by the average final posterior of its earlier episodes. It is told
    which earlier episodes shared a physics: an upper bound on what memory can do."""
    with np.load(f"{OUT}/stream.npz") as f:
        z = {k: f[k] for k in ("obs", "gob", "act", "nxt", "gnx", "events", "episode", "tstep", "stream", "encounter",
                               "physics_id")}
    worlds = stream_worlds(z, s)
    n_chunks = LENGTH // CHUNK
    res = {m: {k: dict(hit_move=np.zeros(n_chunks), tot_move=np.zeros(n_chunks), hit_turn=np.zeros(n_chunks),
                       tot_turn=np.zeros(n_chunks), bits=np.zeros(n_chunks), n=0) for k in range(1, REPEATS + 1)}
           for m in ("nomem", "mem")}
    posts, uniform = [], np.full(SHAPE, 1.0 / np.prod(SHAPE))          # posts: (physics id, final posterior)
    for e, rows in enumerate(worlds):
        k = int(z["encounter"][rows[0]])
        seen = {}
        for pid, post in posts:
            seen.setdefault(pid, []).append(post)
        remaining = EPISODES - e
        prior = uniform * (REPEATS * (DISTINCT - len(seen)) / remaining)
        for pid, ps in seen.items():
            prior = prior + (sum(ps) / len(ps)) * ((REPEATS - len(ps)) / remaining)
        obs = {"nomem": MotionObserver(), "mem": MotionObserver(prior)}
        for m in res:
            res[m][k]["n"] += 1
        for t, r in enumerate(rows):
            seen = robot_state(z["nxt"][r], z["gnx"][r])
            ev, c = int(z["events"][r]), t // CHUNK
            for m, o in obs.items():
                pred, prob = o.step(z["obs"][r], z["gob"][r], int(z["act"][r]), z["nxt"][r], z["gnx"][r])
                rr = res[m][k]
                rr["bits"][c] += -math.log2(max(prob, 1e-12))
                if (ev >> MOVE) & 1:
                    rr["tot_move"][c] += 1
                    rr["hit_move"][c] += pred[0] == seen[0]
                if (ev >> TURN) & 1:
                    rr["tot_turn"][c] += 1
                    rr["hit_turn"][c] += pred[1] == seen[1]
        posts.append((int(z["physics_id"][rows[0]]), obs["mem"].posterior()))
    return {m: {k: {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in d.items()} for k, d in r.items()}
            for m, r in res.items()}


def run_ideal():
    t0 = time.time()
    with Pool(STREAMS) as pool:
        parts = pool.map(ideal_stream, range(STREAMS))
    json.dump(parts, open(f"{OUT}/ideal.json", "w"))
    print(f"ideal observers: {time.time() - t0:.0f}s", flush=True)


# ---- learners ---------------------------------------------------------------------------------------------------------
def run_learner():
    dev = torch.device("cuda")
    z = load(f"{OUT}/stream.npz")
    m = FamilyModel(SPEC).to(dev)
    ck = torch.load("runs/e8/L_Ac/model.pt", map_location=dev, weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    run = Runner(m, z, dev, em_sigma=True)
    run.Fbar = ck["Fbar"]
    worlds = np.stack([stream_worlds(z, s) for s in range(STREAMS)])       # [S, EPISODES, LENGTH]
    S, C, n_chunks = STREAMS, CHUNK, LENGTH // CHUNK
    sc = {(w, k): ChunkScorer(z, n_chunks) for w in ("fresh", "library") for k in range(1, REPEATS + 1)}
    recall = {k: np.zeros((n_chunks, 2)) for k in range(1, REPEATS + 1)}  # mass on same-physics entries, count
    library = []                                                           # per position: (fast dict [S], physics [S])
    keys = list(m.mu)
    t0 = time.time()
    for e in range(EPISODES):
        enc = z["encounter"][worlds[:, e, 0]]
        pid = z["physics_id"][worlds[:, e, 0]]
        fast = m.fast_init(S)
        n_lib = len(library)
        log_prior = torch.tensor([math.log(ALPHA / (n_lib + ALPHA))] + [math.log(1 / (n_lib + ALPHA))] * n_lib,
                                 device=dev)
        evidence = torch.zeros(S, 1 + n_lib, device=dev)
        lib_fast = {k: torch.stack([lf[k] for lf, _ in library], 1).flatten(0, 1) for k in keys} if n_lib else None
        lib_pid = np.array([p for _, p in library]).T if n_lib else None              # [S, n_lib]
        for t in range(n_chunks):
            rows = worlds[:, e, t * C:(t + 1) * C].reshape(-1)
            o, g, a, nx, gx = run.tensors(rows)
            mask = robot_mask(o, run.focus)
            wid = torch.arange(S, device=dev).repeat_interleave(C)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cl, gl = m.forward_fast(o, g, a, fast, wid)
            loss, sur = losses(cl, gl, nx, gx, per_row=True, mask=mask)
            ll = [-sur.detach().view(S, C).sum(1)]                                   # fresh: the chunk's log-lik
            probs = [[x.float().softmax(-1)] for x in cl + gl]
            if n_lib:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    rep = lambda x: x.view(S, C, *x.shape[1:]).unsqueeze(1).expand(S, n_lib, C, *x.shape[1:]).flatten(0, 2)
                    lwid = torch.arange(S * n_lib, device=dev).repeat_interleave(C)
                    lcl, lgl = m.forward_fast(rep(o), rep(g), rep(a), lib_fast, lwid)
                    _, lsur = losses(lcl, lgl, rep(nx), rep(gx), per_row=True, mask=rep(mask))
                ll.append(-lsur.float().view(S, n_lib, C).sum(2))
                for f, x in enumerate(lcl + lgl):
                    probs[f].append(x.float().softmax(-1).view(S, n_lib, C, *x.shape[1:]))
            ll = torch.cat([ll[0][:, None], ll[1]] if n_lib else [ll[0][:, None]], 1)       # [S, 1 + n_lib]
            logw = torch.log_softmax(log_prior + evidence, 1)                                # before this chunk
            mix_sur = -torch.logsumexp(logw + ll, 1)                                       # [S] nats, the chunk
            w = logw.exp()
            mix = []
            for f, plist in enumerate(probs):
                fresh_p = plist[0].view(S, C, *plist[0].shape[1:])
                pm = w[:, 0].view(S, *[1] * (fresh_p.dim() - 1)) * fresh_p
                if n_lib:
                    pm = pm + (w[:, 1:].view(S, n_lib, *[1] * (fresh_p.dim() - 1)) * plist[1]).sum(1)
                mix.append(pm.reshape(S * C, *fresh_p.shape[2:]).clamp_min(1e-12).log())
            nc = len(cl)
            for k in range(1, REPEATS + 1):
                sel = np.nonzero(enc == k)[0]
                if len(sel) == 0:
                    continue
                rsel = (sel[:, None] * C + np.arange(C)).reshape(-1)
                idx = torch.from_numpy(rsel).to(dev)
                sc["fresh", k].add(t, [x[idx] for x in cl], [x[idx] for x in gl], rows[rsel], sur.detach()[idx])
                sc["library", k].add(t, [x[idx] for x in mix[:nc]], [x[idx] for x in mix[nc:]], rows[rsel],
                                     mix_sur[torch.from_numpy(sel).to(dev)])
                if n_lib:
                    same = torch.from_numpy((lib_pid[sel] == pid[sel][:, None]).astype(np.float32)).to(dev)
                    recall[k][t, 0] += float((w[torch.from_numpy(sel).to(dev), 1:] * same).sum())
                    recall[k][t, 1] += len(sel)
            evidence = evidence + ll
            grads = torch.autograd.grad(loss, [fast[k] for k in keys])
            fast = run.kalman_step(fast, dict(zip(keys, grads)), S, C, (t + 1) * C)
        library.append(({k: v.detach() for k, v in fast.items()}, pid))
        if (e + 1) % 8 == 0:
            print(f"episode {e + 1}/{EPISODES}  {time.time() - t0:.0f}s", flush=True)
    out = {f"{w}_{k}": sc[w, k].result() for (w, k) in sc}
    out["recall"] = {k: (recall[k][:, 0] / np.maximum(recall[k][:, 1], 1)).tolist() for k in recall}
    json.dump(out, open(f"{OUT}/learner.json", "w"))
    print(f"learners: {time.time() - t0:.0f}s", flush=True)


# ---- report -----------------------------------------------------------------------------------------------------------
def first(acc, level):
    hits = [i for i, x in enumerate(acc) if x >= level]
    return None if not hits else (hits[0] + 1) * CHUNK


def report():
    ideal = json.load(open(f"{OUT}/ideal.json"))
    learner = json.load(open(f"{OUT}/learner.json"))
    rec = {}
    for k in range(1, REPEATS + 1):
        row = {}
        for m in ("nomem", "mem"):
            agg = {x: np.sum([np.array(p[m][str(k)][x]) for p in ideal], 0) for x in ("hit_move", "tot_move", "hit_turn", "tot_turn", "bits")}
            n = sum(p[m][str(k)]["n"] for p in ideal)
            move = (agg["hit_move"] / np.maximum(agg["tot_move"], 1)).tolist()
            turn = (agg["hit_turn"] / np.maximum(agg["tot_turn"], 1)).tolist()
            row[f"ideal_{m}"] = dict(move=move, turn=turn, move95=first(move, 0.95), turn95=first(turn, 0.95),
                                     motion_bits_episode=float(agg["bits"].sum() / max(n, 1)))
        for w in ("fresh", "library"):
            r = learner[f"{w}_{k}"]
            move, turn = r["acc"]["move"], r["acc"]["turn"]
            row[f"learner_{w}"] = dict(move=move, turn=turn, move95=first(move, 0.95), turn95=first(turn, 0.95),
                                       bits_mean=float(np.mean(r["bits"])), bits_early=float(np.mean(r["bits"][:2])))
        row["recall"] = learner["recall"][str(k)]
        rec[k] = row
        print(f"encounter {k}:")
        for name in ("ideal_nomem", "ideal_mem", "learner_fresh", "learner_library"):
            d = row[name]
            extra = (f" bits/step mean {d['bits_mean']:.2f} early {d['bits_early']:.2f}" if "bits_mean" in d
                     else f" motion bits/episode {d['motion_bits_episode']:.1f}")
            print(f"   {name:16s} move95 {d['move95']}  turn95 {d['turn95']}  move first chunks "
                  f"{[round(x, 2) for x in d['move'][:6]]}{extra}")
        if k > 1:
            print(f"   library mass on same-physics entries by chunk: {[round(x, 2) for x in row['recall'][:6]]}")
    json.dump(rec, open(f"{OUT}/report.json", "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("data", "ideal", "learner", "report"))
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"data": make_data, "ideal": run_ideal, "learner": run_learner, "report": report}[a.stage]()


if __name__ == "__main__":
    main()
