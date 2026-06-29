"""The epistemic drive = LEARNING PROGRESS, not raw prediction error (reference_animal_exploration). The decisive case
is the NOISY TV: a source of irreducible prediction error (here a cell with random transitions). RAW-error curiosity
is drawn to it forever (high error = high reward -- Schmidhuber's noisy-TV trap); learning-progress recognises there is
nothing to learn (persistent error, no reduction) and IGNORES it, spending its exploration on the task instead. The
'progress' value gates the novelty frontier off such confirmed-noise regions -- and is identical to count-novelty on
learnable structure, so it costs nothing there."""

from __future__ import annotations

import os
import random
import statistics
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.agent import Agent  # noqa: E402


class NoisyTVWorld:
    """A sparse-reward grid with a NOISY-TV cell: at the TV the transition is RANDOM (irreducible prediction error), a
    perpetual lure for error-curiosity. It sits off the path to the goal, so dwelling there is a pure detour."""

    MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def __init__(self, N=6, goal=(5, 5), tv=(0, 5), seed=0):
        self.N, self.goal, self.tv = N, goal, tv
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self.pos

    def step(self, a):
        dx, dy = self.rng.choice(self.MOVES) if self.pos == self.tv else self.MOVES[a]   # the noise: random at the TV
        x, y = self.pos
        self.pos = (min(max(x + dx, 0), self.N - 1), min(max(y + dy, 0), self.N - 1))
        return self.pos, (1 if self.pos == self.goal else 0)


def _run(epistemic, steps=4000, seed=0):
    env = NoisyTVWorld(seed=seed)
    ag = Agent(n_actions=4, seed=seed, epistemic=epistemic)
    s, delta, comps, tv = env.reset(), 0, 0, 0
    for _ in range(steps):
        a = ag.step(s, delta)
        s, delta = env.step(a)
        if s == env.tv:
            tv += 1
        if delta > 0:
            ag.step(s, delta); ag.new_episode(); s = env.reset(); delta = 0; comps += 1
    return comps, tv


def test_learning_progress_ignores_the_noisy_tv_that_raw_error_chases():
    """Across seeds: the raw-error agent is DRAWN to the noisy TV far more than the learning-progress agent, which
    recognises the irreducible error and avoids it -- while BOTH still solve the sparse-reward task."""
    seeds = (0, 1, 2)
    err = [_run("error", seed=s) for s in seeds]
    prog = [_run("progress", seed=s) for s in seeds]
    err_tv = statistics.mean(tv for _c, tv in err)
    prog_tv = statistics.mean(tv for _c, tv in prog)
    assert err_tv > 5 * prog_tv, f"raw-error not trapped by the noisy TV: error visits {err_tv:.0f} vs progress {prog_tv:.0f}"
    assert all(c > 100 for c, _ in prog), f"learning-progress agent did not solve the task: {[c for c, _ in prog]}"
    assert all(c > 100 for c, _ in err), f"error agent did not solve the task: {[c for c, _ in err]}"
