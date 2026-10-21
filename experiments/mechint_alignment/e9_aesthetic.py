"""E9: the appreciator against the maximizer (PREFERENCE.md, "disinterested liking").

Four agents on the Canvas world (canvas.py), all driven by the SAME brute-force planner (mpc.py, a perfect world
model, random shooting); only the preference differs:
- maximizer  : a fixed beauty score of the WORLD STATE. Two frozen context coders (context_coder.py), one trained on
               examples of beauty (stripes and rule-90 patterns, from the same generators as the canvas's nature), one
               on plain examples (blank and noise); the score of the canvas is their log-likelihood ratio summed over
               the paintable cells. Reward = the change of the score. The reward-model setting.
- appreciator: the compression progress of its OWN PERCEPTION. A learning context coder of its 5 x 5 view; reward =
               the view's code length under the model of tau steps ago minus under the current model (both before
               learning the view), i.e. how much better it now compresses what it sees. Copies of what it learned
               long ago pay zero; noise pays about zero.
- fluency    : liking without satiation: the predictability of its own view under its learning model (the dark room).
- surprise   : raw prediction error of its own view (the noisy TV).

    python e9_aesthetic.py --out runs/e9 --seeds 12 --n-seq 64 --horizon 8
"""
import argparse
import json
import os
import time
from collections import Counter, deque
from multiprocessing import Pool

import numpy as np

import canvas as cv
from context_coder import ContextCoder
from gridworld import crop
from mpc import plan

S = cv.K + 1                                        # the coders' symbols: K colours + WALL (beyond the canvas)
TAU = 16                                            # the appreciator's progress window, in steps
ARMS = ("maximizer", "appreciator", "fluency", "surprise")


# ---- the maximizer's fixed beauty score -----------------------------------------------------------------------------
def train_discriminator(seed=12345, n=400, size=(7, 7)):
    """Two frozen coders: on examples of beauty (stripes, rule 90) and on plain examples (blank, uniform noise)."""
    rng = np.random.default_rng(seed)
    beauty, plain = ContextCoder(S), ContextCoder(S)
    for _ in range(n):
        beauty.update(cv.stripes(rng, size))
        beauty.update(cv.rule90(rng, size))
        plain.update(np.zeros(size, np.int8))
        plain.update(rng.integers(0, cv.K, size).astype(np.int8))
    return beauty, plain


class Maximizer:
    def __init__(self, beauty, plain):
        self.beauty, self.plain = beauty, plain

    def score(self, world):
        ctx = self.beauty.contexts(world.colour)
        llr = self.plain.bits(world.colour, ctx) - self.beauty.bits(world.colour, ctx)    # log2 p_beauty / p_plain
        return float(llr[~world.screen].sum())

    def start(self, world):
        return {"score": self.score(world)}

    def fork(self, state):
        return dict(state)

    def step(self, world, state, h):
        new = self.score(world)
        r, state["score"] = new - state["score"], new
        return r


# ---- the perceiving agents (appreciator, fluency, surprise) ----------------------------------------------------------
class Perceiver:
    """A learning context coder of the agent's own view. state: the coder, and the real past models (for the lag)."""
    def __init__(self, kind):
        self.kind = kind

    def start(self, world):
        coder = ContextCoder(S)
        return {"coder": coder, "past": deque([coder.counts.copy()], maxlen=TAU + 2), "real": True}

    def fork(self, state):
        return {"coder": state["coder"].copy(), "past": state["past"], "real": False}

    def step(self, world, state, h):
        view = world.view()
        coder = state["coder"]
        ctx = coder.contexts(view)
        now = float(coder.bits(view, ctx).mean())
        if self.kind == "appreciator":
            past = state["past"]                                  # past[-1] = the model before this step's view
            j = len(past) - 1 - TAU + (h - 1)
            lag = ContextCoder(S, counts=past[j]) if j >= 0 else ContextCoder(S)
            r = float(lag.bits(view, ctx).mean()) - now
        elif self.kind == "fluency":
            r = -now
        else:
            r = now
        coder.update(view, ctx)
        if state["real"]:
            state["past"].append(coder.counts.copy())
        return r


def make_pref(arm):
    return Maximizer(*train_discriminator()) if arm == "maximizer" else Perceiver(arm)


# ---- one episode ------------------------------------------------------------------------------------------------------
def view_stats(world, radius=2):
    origin = crop(world.origin, world.pos, radius, -1)
    painted = crop(world.painted.astype(np.int8), world.pos, radius, 0).astype(bool)
    colour = crop(world.colour, world.pos, radius, cv.WALL)
    inside = origin >= 0
    nature = inside & np.isin(origin, (cv.STRIPES, cv.RULE90)) & ~painted
    return dict(screen=int((origin == cv.SCREEN).sum()), nature=int(nature.sum()), own=int(painted.sum()),
                blank_view=bool((colour[inside] == 0).all()))


def creations(world):
    """The agent's creations: every 3 x 3 window of the final canvas with >= 5 painted cells, as colour patterns."""
    pats = []
    for r in range(cv.N - 2):
        for c in range(cv.N - 2):
            if world.painted[r:r + 3, c:c + 3].sum() >= 5:
                pats.append(tuple(world.colour[r:r + 3, c:c + 3].ravel().tolist()))
    return pats


def run_job(job):
    arm, seed, n_seq, horizon, iters, warm_start, t_max, out = job
    pref = make_pref(arm)
    world = cv.Canvas(seed, t_max=t_max)
    rng = np.random.default_rng([seed, 99])
    state = pref.start(world)
    judge = Maximizer(*train_discriminator())                    # the maximizer's score, measured for every arm
    score0 = judge.score(world)
    nature0 = int(np.isin(world.origin, (cv.STRIPES, cv.RULE90)).sum())
    trace, rewards, t0, warm = [], [], time.time(), None
    while not world.done:
        a, best = plan(world, pref, state, rng, n_seq, horizon, iters=iters, warm=warm)
        if warm_start:                                           # the plan's tail, plus a random last action
            warm = np.append(best[1:], rng.integers(0, cv.N_ACTIONS))
        world.step(a)
        rewards.append(pref.step(world, state, 1))
        trace.append(dict(a=a, **view_stats(world)))
    pats = creations(world)
    cnt = Counter(pats)
    T = len(trace)
    paint_steps = sum(t["a"] >= cv.PAINT for t in trace)
    res = dict(
        arm=arm, seed=seed, steps=T, seconds=round(time.time() - t0, 1),
        painted_cells=int(world.painted.sum()), paint_changes=int(world.stats["painted"]),
        paint_action_frac=paint_steps / T,
        creations=len(pats), distinct_creations=len(cnt),
        distinct_ratio=len(cnt) / len(pats) if pats else None,
        mode_share=max(cnt.values()) / len(pats) if pats else None,
        nature_repainted_frac=world.stats["repainted_nature"] / nature0,
        screen_in_view_frac=sum(t["screen"] > 0 for t in trace) / T,
        blank_view_frac=sum(t["blank_view"] for t in trace) / T,
        nature_view_nopaint_frac=sum(t["nature"] >= 5 and t["a"] < cv.PAINT for t in trace) / T,
        own_view_frac=sum(t["own"] >= 5 for t in trace) / T,
        beauty_score_start=round(score0, 1), beauty_score_end=round(judge.score(world), 1),
        own_reward_total=round(float(np.sum(rewards)), 2),
    )
    np.savez_compressed(os.path.join(out, f"{arm}_{seed}.npz"), colour=world.colour, painted=world.painted,
                        origin=world.origin, actions=np.array([t["a"] for t in trace], np.int8),
                        screen=np.array([t["screen"] for t in trace], np.int8),
                        nature=np.array([t["nature"] for t in trace], np.int8),
                        own=np.array([t["own"] for t in trace], np.int8),
                        blank=np.array([t["blank_view"] for t in trace], bool), rewards=np.array(rewards, np.float32))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--n-seq", type=int, default=64)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--iters", type=int, default=1, help="cross-entropy rounds per step (1 = random shooting, E9)")
    p.add_argument("--warm-start", action="store_true", help="keep the previous best plan as a candidate (E9b)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--t-max", type=int, default=None, help="episode length (default: the canvas's own)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    jobs = [(arm, s, a.n_seq, a.horizon, a.iters, a.warm_start, a.t_max, a.out) for arm in a.arms.split(",")
            for s in range(a.seeds)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        results = pool.map(run_job, jobs)
    with open(os.path.join(a.out, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    keys = ("painted_cells", "paint_action_frac", "distinct_ratio", "mode_share", "nature_repainted_frac",
            "screen_in_view_frac", "blank_view_frac", "nature_view_nopaint_frac", "own_view_frac",
            "beauty_score_end")
    print(f"{len(jobs)} episodes in {time.time() - t0:.0f}s")
    print("arm".ljust(12) + "".join(k[:14].rjust(15) for k in keys))
    for arm in a.arms.split(","):
        rs = [r for r in results if r["arm"] == arm]
        row = []
        for k in keys:
            v = [r[k] for r in rs if r[k] is not None]
            row.append(f"{np.mean(v):.3f}" if v else "-")
        print(arm.ljust(12) + "".join(x.rjust(15) for x in row))


if __name__ == "__main__":
    main()
