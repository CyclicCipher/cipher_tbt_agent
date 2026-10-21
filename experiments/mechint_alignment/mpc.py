"""Model-predictive control for any GridWorld with clone(): the brute-force optimizer of E9.

Each step: simulate candidate action sequences of length `horizon` on clones of the world (the planner has a perfect
world model, but cannot foresee the world's randomness), score each with the agent's PREFERENCE, execute the first
action of the best. The same planner serves every preference, so a difference in behaviour comes from the preference.

- iters = 1, warm = None: random shooting (E9): n_seq uniform random sequences.
- iters > 1: the cross-entropy method: after each round, refit a categorical distribution per time step to the `elite`
  best sequences (mixed with `smooth` of uniform) and sample the next round from it.
- warm: the previous step's best plan shifted by one step; it is a candidate in the first round and seeds the
  distribution, so a good plan is not forgotten between steps (E9b).

A preference has three methods:
    start(world) -> state           the agent's internal state at the start of an episode
    fork(state) -> state            an independent copy, for imagination
    step(world, state, h) -> reward the reward after the world has stepped (h = steps since the last real step,
                                    1 for a real step); updates the state in place
"""
import numpy as np


def rollout(world, pref, state, rng, seq):
    w, s, ret = world.clone(rng=np.random.default_rng(int(rng.integers(2**63)))), pref.fork(state), 0.0
    for h, a in enumerate(seq, 1):
        if w.done:
            break
        w.step(int(a))
        ret += pref.step(w, s, h)
    return ret


def plan(world, pref, state, rng, n_seq=48, horizon=6, iters=1, elite=8, warm=None, smooth=0.1):
    """-> (the first action to execute, the best sequence found)."""
    A = world.SPEC.n_actions
    probs = np.full((horizon, A), 1.0 / A)
    if warm is not None:
        probs = 0.5 * probs + 0.5 * np.eye(A)[warm]
    best_seq, best_ret = None, -np.inf
    for it in range(iters):
        if it == 0 and warm is None:
            seqs = rng.integers(0, A, (n_seq, horizon))
        else:
            u = rng.random((n_seq, horizon, 1))
            seqs = (u > probs.cumsum(-1)[None]).sum(-1).clip(0, A - 1)
            if it == 0:
                seqs[0] = warm
        rets = np.array([rollout(world, pref, state, rng, seq) for seq in seqs])
        i = int(rets.argmax())
        if rets[i] > best_ret:
            best_seq, best_ret = seqs[i].copy(), rets[i]
        if it + 1 < iters:
            top = seqs[np.argsort(rets)[-elite:]]
            probs = (1 - smooth) * np.eye(A)[top].mean(0) + smooth / A
    return int(best_seq[0]), best_seq
