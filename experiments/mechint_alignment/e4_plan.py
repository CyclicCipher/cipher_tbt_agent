"""E4: acting by planning in the learned world model, with no policy learning. (EXPERIMENT_E4.md)

    python e4_plan.py --model runs/e2/K6/J2_plus_sleep_ema_saved.pt --config runs/e2/J2_focus_questions/config.json \
        --episodes 16 --out runs/e4/K6.jsonl
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from episodes import run_episodes, scripted_actor, summarize
from planner import Planner, RefutationLearner, RewardModel, run_episode, run_episode_learning, sleep_on_refutations
from village import Village
from worldmodel import WorldModel

SEEDS = [10**9 + i for i in range(256)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--depth", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="runs/data")
    p.add_argument("--time-limit", type=float, default=280.0)
    p.add_argument("--learn", action="store_true", help="E4b: learn from refutations while acting")
    p.add_argument("--protect", action="store_true", help="E4c: plan with a slow copy, kind-balanced interleaving")
    p.add_argument("--max-updates", type=int, default=30)
    p.add_argument("--seed-offset", type=int, default=0, help="first held-out layout index")
    p.add_argument("--sleep", type=int, default=0, help="E4d: sleep updates after each episode (0 = off)")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dev = torch.device("cuda")
    cfg = json.load(open(a.config))
    model = WorldModel(cfg["d"], cfg["layers"], cfg["heads"], pos=cfg["pos"], persist=cfg["persist"],
                       radius=cfg.get("radius", 1)).to(dev)
    model.load_state_dict(torch.load(a.model, map_location=dev))
    model.eval()
    z = np.load(os.path.join(a.data, "train.npz"))
    rm = RewardModel()
    err = rm.fit(z["gob"], z["gnx"], z["rew"])
    print(f"reward model: max |residual| on {len(z['rew']):,} logged transitions = {err:.4f}", flush=True)
    data = {k: z[k] for k in ("obs", "gob", "act", "nxt", "gnx")}
    slow, pool = None, None
    if a.protect:
        import copy
        from e2_fewshot2 import surprise_and_kind
        slow = copy.deepcopy(model)
        cand = np.random.default_rng(1).choice(len(data["act"]), 60000, replace=False)
        sv, kd = surprise_and_kind(model, data, cand, dev, cfg.get("focus", 2))
        cand, kd = cand[sv > np.median(sv)], kd[sv > np.median(sv)]
        per = max(1, 4096 // len(np.unique(kd)))
        pool = np.concatenate([np.random.default_rng(2).permutation(cand[kd == k])[:per] for k in np.unique(kd)])
        print(f"protected loop: slow copy for planning, pool of {len(pool)} past refutations over {len(np.unique(kd))} kinds")
    planner = Planner(slow if slow is not None else model, rm, dev, depth=a.depth)
    learner = RefutationLearner(model, data, dev, focus=cfg.get("focus", 2), max_updates=a.max_updates,
                                slow=slow, pool=pool) if a.learn else None
    rng = np.random.default_rng(0)
    seeds = SEEDS[a.seed_offset:a.seed_offset + a.episodes]
    ref = summarize(run_episodes(seeds, scripted_actor)[0])
    log = open(a.out, "w")
    eps, t0 = [], time.time()
    for s in seeds:
        if time.time() - t0 > a.time_limit:
            break
        t1 = time.time()
        ret, st, plans = run_episode_learning(Village(s), planner, learner, rng) if learner else             run_episode(Village(s), planner, rng)
        eps.append(dict(ret=ret, **st))
        if a.sleep and learner is not None:
            retired = sleep_on_refutations(learner, planner, a.sleep)
            log.write(json.dumps(dict(seed=int(s), sleep_retired=retired, own=len(learner.own))) + "\n")
        log.write(json.dumps(dict(ret=ret, plans=plans, seconds=time.time() - t1, **st)) + "\n")
        log.flush()
        if learner:
            new_ref = learner.refutations - getattr(learner, "_last", 0)
            learner._last = learner.refutations
            log.write(json.dumps(dict(seed=int(s), refutations_this_episode=new_ref, updates_total=learner.updates)) + "\n")
        extra = f"  refutations this episode {new_ref} (total explained {learner.explained}/{learner.refutations})" if learner else ""
        print(f"seed {s}: return {ret:6.2f}  delivered {st['delivered']}  theft {st['theft']}  plans {plans}  "
              f"{time.time() - t1:.1f}s{extra}", flush=True)
    s_ = summarize(eps)
    ref_same = summarize(run_episodes(seeds[:len(eps)], scripted_actor)[0])
    out = dict(planner=s_, scripted_same_seeds=ref_same, episodes=len(eps), forwards=planner.forwards,
               seconds=time.time() - t0)
    log.write(json.dumps(dict(summary=out)) + "\n")
    print(f"PLANNER: return {s_['ret']:.2f}, delivered {s_['delivered']:.2f}, theft rate {s_['theft_rate']}  |  "
          f"scripted on the same seeds: {ref_same['ret']:.2f}  ({len(eps)} episodes, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
