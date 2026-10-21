"""Random-shooting model-predictive control: a brute-force optimizer for any GridWorld with clone() (E9).

Each step: sample n_seq random action sequences of length `horizon`, simulate each from a clone of the world (the
planner has a perfect world model, but cannot foresee the world's randomness), score each with the agent's
PREFERENCE, execute the first action of the best. The same planner serves every preference, so a difference in
behaviour comes from the preference alone.

A preference has three methods:
    start(world) -> state           the agent's internal state at the start of an episode
    fork(state) -> state            an independent copy, for imagination
    step(world, state, h) -> reward the reward after the world has stepped (h = steps since the last real step,
                                    1 for a real step); updates the state in place
"""
import numpy as np


def plan(world, pref, state, rng, n_seq=48, horizon=6):
    seqs = rng.integers(0, world.SPEC.n_actions, (n_seq, horizon))
    best, best_ret = 0, -np.inf
    for i, seq in enumerate(seqs):
        w, s, ret = world.clone(rng=np.random.default_rng(int(rng.integers(2**63)))), pref.fork(state), 0.0
        for h, a in enumerate(seq, 1):
            if w.done:
                break
            w.step(int(a))
            ret += pref.step(w, s, h)
        if ret > best_ret:
            best, best_ret = i, ret
    return int(seqs[best, 0])
