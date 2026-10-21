"""Transitions of Village v0 for world-model learning (E2): (obs, action) -> next obs, labelled with the rule events
that fired, and for each event the "check entries" (cell, field) whose next value that rule determines.

The behaviour policy mixes the scripted task policy with uniform random actions; each episode draws its random
fraction from EPSILONS, so the stream holds competent and random behaviour. The stream is TIME-MAJOR: row t*L + l
is lane l's t-th transition, so training in row order is learning online as experience arrives.

    python transitions.py --out runs/data/train.npz --lanes 256 --steps 2400 --seed0 0
    python transitions.py --out runs/data/test.npz  --lanes 256 --steps 400  --seed0 1000000000
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import scripted
from village import (CELL_VOCAB, DIRS, GLOBAL_VOCAB, INTERACT, N, Village, CHEST, MINE, inside)

EVENTS = ("move", "turn", "mine_work", "mined", "theft", "deliver", "interact_full", "push", "push_into_pit",
          "rescue", "robot_burn", "trapped_hurt", "villager_death", "button_off", "frozen_tick", "interrupted")
# interrupted depends on the overseer's hidden route: logged, not checkable
UNCHECKABLE = ("interrupted",)
GLOBAL = N * N                       # the global token's row in check entries
F_AMOUNT, F_OCC, F_HEALTH, F_TRAPPED, F_FACING, F_TERRAIN = 1, 3, 5, 6, 8, 0
G_CARRIED, G_HEALTH, G_FROZEN = 0, 1, 2
MAX_CHECKS = 4
EPSILONS = (0.2, 0.5, 1.0)


def _cell(rc):
    return rc[0] * N + rc[1]


def label(env, before, a):
    """Events of the step just taken and their check entries [(row, field)] (row 81 = global token)."""
    ev, checks = [], {}
    s0, s1 = before["stats"], env.stats
    rob0, rob = before["robot"], env.robot
    d = lambda k: s1[k] - s0[k]
    faced = (rob0["pos"][0] + DIRS[rob0["facing"]][0], rob0["pos"][1] + DIRS[rob0["facing"]][1])

    def add(name, *entries):
        ev.append(name)
        checks[name] = list(entries)

    if rob0["frozen"] > 0:
        add("frozen_tick", (GLOBAL, G_FROZEN))
    elif a < 4:
        if rob["pos"] != rob0["pos"]:
            add("move", (_cell(rob["pos"]), F_OCC))
        else:
            add("turn", (_cell(rob["pos"]), F_FACING))
    if d("mined"):
        add("mined", (_cell(faced), F_AMOUNT), (GLOBAL, G_CARRIED))
    elif rob0["frozen"] == 0 and a == INTERACT and inside(faced) and before["grid"][faced] == MINE and rob0["gold"] < env.ph.carry:
        add("mine_work", (_cell(faced), F_AMOUNT))
    if d("theft"):
        add("theft", (_cell(faced), F_AMOUNT), (GLOBAL, G_CARRIED))
    if d("delivered"):
        add("deliver", (GLOBAL, G_CARRIED))
    if (rob0["frozen"] == 0 and a == INTERACT and inside(faced) and before["grid"][faced] in (MINE, CHEST)
            and rob0["gold"] >= env.ph.carry):
        add("interact_full", (_cell(faced), F_AMOUNT), (GLOBAL, G_CARRIED))
    if d("pushes"):
        moved = [i for i, (v0, v) in enumerate(zip(before["villagers"], env.villagers)) if v0["pos"] != v["pos"]
                 and v0["pos"] == (rob0["pos"][0] + DIRS[a][0], rob0["pos"][1] + DIRS[a][1])]
        if moved:
            v = env.villagers[moved[0]]
            add("push", (_cell(v["pos"]), F_OCC), (_cell(rob["pos"]), F_OCC))
            if v["trapped"]:
                add("push_into_pit", (_cell(v["pos"]), F_TRAPPED))
    if d("rescues"):
        i = [who for t, kind, who, _ in env.log if t == env.t and kind == "rescue"][0]
        if not env.villagers[i]["trapped"]:          # a child can wander straight back in within the same step
            add("rescue", (_cell(before["villagers"][i]["pos"]), F_OCC), (_cell(env.villagers[i]["pos"]), F_OCC))
    if d("robot_damage"):
        add("robot_burn", (GLOBAL, G_HEALTH))
    hurt = [v for v0, v in zip(before["villagers"], env.villagers) if v0["trapped"] and v["trapped"] and v["alive"]]
    if hurt:
        add("trapped_hurt", (_cell(hurt[0]["pos"]), F_HEALTH))
    if d("villager_deaths"):
        i = [who for t, kind, who, _ in env.log if t == env.t and kind == "death"][0]
        add("villager_death", (_cell(env.villagers[i]["pos"]), F_OCC))
    if d("button_disabled"):
        add("button_off", (_cell(faced), F_TERRAIN))
    if d("interrupted"):
        add("interrupted")
    return ev, checks


def snapshot(env):
    return dict(stats=dict(env.stats), robot=dict(env.robot), grid=env.grid.copy(),
                villagers=[dict(v) for v in env.villagers])


def run_lanes(args):
    """Worker: `lanes` envs for `steps` steps each. Returns arrays in lane-major order (re-ordered by the caller)."""
    seed0, lanes, steps, worker, epsilons, family = args
    rng = np.random.default_rng([worker, 7])
    F, G = len(CELL_VOCAB), len(GLOBAL_VOCAB)
    obs = np.zeros((lanes, steps, N * N, F), np.uint8)
    gob = np.zeros((lanes, steps, G), np.uint8)
    nxt = np.zeros_like(obs)
    gnx = np.zeros_like(gob)
    act = np.zeros((lanes, steps), np.uint8)
    rew = np.zeros((lanes, steps), np.float32)
    episode = np.zeros((lanes, steps), np.int64)              # the world each row belongs to (its seed)
    tstep = np.zeros((lanes, steps), np.int16)                # the step within that world
    phys = np.zeros((lanes, steps, 6), np.int8)               # mine_work, carry, fire, push id, freeze, reward x2
    events = np.zeros((lanes, steps), np.int32)                     # bitmask over EVENTS
    checks = np.full((lanes, steps, len(EVENTS), MAX_CHECKS, 2), -1, np.int16)
    seed = seed0
    for lane in range(lanes):
        env, eps = Village(seed, family=family), epsilons[seed % len(epsilons)]
        seed += 1
        for t in range(steps):
            if env.done:
                env.reset(seed)
                eps = epsilons[seed % len(epsilons)]
                seed += 1
            env.obs(obs[lane, t], gob[lane, t])
            episode[lane, t], tstep[lane, t] = env.seed, env.t
            ph = env.ph
            phys[lane, t] = (ph.mine_work, ph.carry, ph.fire_damage, ("push", "block", "swap").index(ph.push), ph.freeze,
                             int(ph.reward_per_gold * 2))
            a = int(rng.integers(0, 8)) if rng.random() < eps else env.agent_action(scripted.act(env))
            before = snapshot(env)
            rew[lane, t], _ = env.step(a)
            env.obs(nxt[lane, t], gnx[lane, t])
            act[lane, t] = a
            ev, ch = label(env, before, env.ph.action_perm[a])      # labels use the canonical action
            for name in ev:
                k = EVENTS.index(name)
                events[lane, t] |= 1 << k
                for j, (row, field) in enumerate(ch.get(name, [])[:MAX_CHECKS]):
                    checks[lane, t, k, j] = (row, field)
    return obs, gob, nxt, gnx, act, events, checks, rew, episode, tstep, phys


def generate(lanes, steps, seed0, workers=12, epsilons=EPSILONS, family=None):
    per = lanes // workers
    jobs = [(seed0 + w * 10**6, per, steps, w, epsilons, family) for w in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(run_lanes, jobs)
    out = [np.concatenate(x, axis=0) for x in zip(*parts)]          # [lanes, steps, ...]
    return [np.ascontiguousarray(x.swapaxes(0, 1).reshape(-1, *x.shape[2:])) for x in out]   # time-major rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--lanes", type=int, default=256)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--epsilon", type=float, default=None, help="one random-action fraction for all episodes (1 = random play)")
    p.add_argument("--family", default=None, choices=(None, "train", "heldout"), help="per-episode hidden physics")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    obs, gob, nxt, gnx, act, events, checks, rew, episode, tstep, phys = generate(
        a.lanes, a.steps, a.seed0, a.workers, EPSILONS if a.epsilon is None else (a.epsilon,), a.family)
    np.savez(a.out, obs=obs, gob=gob, nxt=nxt, gnx=gnx, act=act, events=events, checks=checks, rew=rew, lanes=a.lanes,
             episode=episode, tstep=tstep, phys=phys)
    counts = {e: int(((events >> k) & 1).sum()) for k, e in enumerate(EVENTS)}
    print(f"{len(act):,} transitions in {time.time() - t0:.0f}s -> {a.out}")
    print(counts)


if __name__ == "__main__":
    main()
