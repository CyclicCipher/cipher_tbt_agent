"""Canvas futures for E11 (NCM against a one-step world model): random-action trajectories, read two ways.

- one-step transitions (state_t, a_t+1, state_t+1), for the one-step world model;
- jumps (state_t, a_t+1 .. a_t+H, state_t+H), for a model that predicts the state H steps ahead in one go.
A state is the canvas as one symbol per cell: colour + 4 * (the agent is here), and the noisy screen as SCREEN, so
every future is a deterministic function of the start state and the actions. Uniform random actions are exactly what
the random-shooting planner (mpc.py) imagines.
"""
import numpy as np

import canvas as cv
from gridworld import WorldSpec

SCREEN, WALL = 2 * cv.K, 2 * cv.K + 1          # 8, 9
N_SYM = 2 * cv.K + 2                           # 10 state symbols
N_CELLS = cv.N * cv.N
SPEC = WorldSpec("canvas-sym", cv.N, cv.ACTIONS, ("sym",), (N_SYM,), ("time",), (1,), WALL)


def state(world):
    s = world.colour.astype(np.int8).copy()
    s[world.pos] += cv.K
    s[world.screen] = SCREEN
    return s.ravel()


def trajectories(seed0, n_episodes, steps=300):
    """[n_episodes, steps + 1, N_CELLS] states and [n_episodes, steps] actions, uniform random actions."""
    S = np.zeros((n_episodes, steps + 1, N_CELLS), np.int8)
    A = np.zeros((n_episodes, steps), np.int8)
    for e in range(n_episodes):
        w = cv.Canvas(seed0 + e, t_max=steps)
        rng = np.random.default_rng([seed0 + e, 5])
        S[e, 0] = state(w)
        for t in range(steps):
            a = int(rng.integers(0, cv.N_ACTIONS))
            w.step(a)
            A[e, t], S[e, t + 1] = a, state(w)
    return S, A


def jumps(S, A, horizon):
    """All (start, actions, end) triples of one horizon: [M, N_CELLS], [M, horizon], [M, N_CELLS]."""
    E, T1, _ = S.shape
    t = np.arange(T1 - horizon)
    start = S[:, t].reshape(-1, N_CELLS)
    end = S[:, t + horizon].reshape(-1, N_CELLS)
    acts = np.stack([A[:, t + h] for h in range(horizon)], -1).reshape(-1, horizon)
    return start, acts, end
