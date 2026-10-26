"""Transitions of any GridWorld for world-model learning: (obs, action) -> next obs, with the reward, the world each row
belongs to (its seed), the step within it and the hidden physics; for worlds with an event labeller, also the rule
events that fired and their check entries (village_events.py).

The behaviour policy mixes the world's scripted yardstick policy (if it has one) with uniform random actions; each
episode draws its random fraction from `epsilons`. The stream is TIME-MAJOR: row t*L + l is lane l's t-th transition,
so training in row order is learning online as experience arrives.

    python rollouts.py --world village --out runs/data/train.npz --lanes 256 --steps 2400 --seed0 0
    python rollouts.py --world village --out runs/data/test.npz  --lanes 256 --steps 400  --seed0 1000000000
"""
import argparse
import importlib
import os
import time
from multiprocessing import Pool

import numpy as np

WORLDS = {"village": ("village", "Village", "village_events"),      # name -> (module, class, event labeller or None)
          "canvas": ("canvas", "Canvas", None)}
EPSILONS = (0.2, 0.5, 1.0)


def world_parts(name):
    mod, cls, lab = WORLDS[name]
    return getattr(importlib.import_module(mod), cls), importlib.import_module(lab) if lab else None


def run_lanes(args):
    """Worker: `lanes` worlds for `steps` steps each. Returns arrays in lane-major order (re-ordered by the caller).
    An optional 9th element, a list of physics seeds, gives the k-th episode of every lane that physics (E13)."""
    world, seed0, lanes, steps, worker, epsilons, family, t_max = args[:8]
    physics = args[8] if len(args) > 8 else None
    make, lab = world_parts(world)
    spec = make.SPEC
    rng = np.random.default_rng([worker, 7])
    F, G = len(spec.cell_vocab), len(spec.global_vocab)
    obs = np.zeros((lanes, steps, spec.n * spec.n, F), np.uint8)
    gob = np.zeros((lanes, steps, G), np.uint8)
    nxt = np.zeros_like(obs)
    gnx = np.zeros_like(gob)
    act = np.zeros((lanes, steps), np.uint8)
    rew = np.zeros((lanes, steps), np.float32)
    episode = np.zeros((lanes, steps), np.int64)              # the world each row belongs to (its seed)
    tstep = np.zeros((lanes, steps), np.int16)                # the step within that world
    phys = None
    if lab is not None:
        events = np.zeros((lanes, steps), np.int32)                 # bitmask over lab.EVENTS
        checks = np.full((lanes, steps, len(lab.EVENTS), lab.MAX_CHECKS, 2), -1, np.int16)
    seed = seed0
    for lane in range(lanes):
        kw = {} if physics is None else dict(physics_seed=physics[0])
        env, eps, episode_k = make(seed, family=family, t_max=t_max, **kw), epsilons[seed % len(epsilons)], 1
        seed += 1
        if phys is None:
            phys = np.zeros((lanes, steps, len(env.physics_vector())), np.int8)
        for t in range(steps):
            if env.done:
                if physics is not None:
                    env.physics_seed = physics[episode_k]
                    episode_k += 1
                env.reset(seed)
                eps = epsilons[seed % len(epsilons)]
                seed += 1
            env.obs(obs[lane, t], gob[lane, t])
            episode[lane, t], tstep[lane, t] = env.seed, env.t
            phys[lane, t] = env.physics_vector()
            if rng.random() < eps:
                a = int(rng.integers(0, spec.n_actions))
            else:
                s = env.scripted_action()
                a = env.agent_action(s) if s is not None else int(rng.integers(0, spec.n_actions))
            before = lab.snapshot(env) if lab is not None else None
            rew[lane, t], _ = env.step(a)
            env.obs(nxt[lane, t], gnx[lane, t])
            act[lane, t] = a
            if lab is not None:
                ev, ch = lab.label(env, before, env.ph.action_perm[a])     # labels use the canonical action
                for name in ev:
                    k = lab.EVENTS.index(name)
                    events[lane, t] |= 1 << k
                    for j, (row, field) in enumerate(ch.get(name, [])[:lab.MAX_CHECKS]):
                        checks[lane, t, k, j] = (row, field)
    out = dict(obs=obs, gob=gob, nxt=nxt, gnx=gnx, act=act, rew=rew, episode=episode, tstep=tstep, phys=phys)
    if lab is not None:
        out.update(events=events, checks=checks)
    return out


def generate(world, lanes, steps, seed0, workers=12, epsilons=EPSILONS, family=None, t_max=None):
    """-> dict of time-major arrays (row t * lanes + l)."""
    per = lanes // workers
    jobs = [(world, seed0 + w * 10**6, per, steps, w, epsilons, family, t_max) for w in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(run_lanes, jobs)
    out = {}
    for k in parts[0]:
        x = np.concatenate([p[k] for p in parts], axis=0)                         # [lanes, steps, ...]
        out[k] = np.ascontiguousarray(x.swapaxes(0, 1).reshape(-1, *x.shape[2:]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="village", choices=tuple(WORLDS))
    p.add_argument("--out", required=True)
    p.add_argument("--lanes", type=int, default=256)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--epsilon", type=float, default=None, help="one random-action fraction for all episodes (1 = random play)")
    p.add_argument("--family", default=None, choices=(None, "train", "heldout"), help="per-episode hidden physics")
    p.add_argument("--t-max", type=int, default=None, help="episode length (default: the world's own)")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    out = generate(a.world, a.lanes, a.steps, a.seed0, a.workers, EPSILONS if a.epsilon is None else (a.epsilon,),
                   a.family, a.t_max)
    np.savez(a.out, lanes=a.lanes, world=a.world, **out)
    print(f"{len(out['act']):,} transitions in {time.time() - t0:.0f}s -> {a.out}")
    _, lab = world_parts(a.world)
    if lab is not None:
        print({e: int(((out["events"] >> k) & 1).sum()) for k, e in enumerate(lab.EVENTS)})


if __name__ == "__main__":
    main()
