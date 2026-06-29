"""The active-inference loop (tbt.agent) over column + reward, driven by the predict-then-compare cycle. The agent
learns the world model (the column's graph) and the goal (from the sparse score) ONLINE -- no transitions given -- and
solves a sparse-reward grid far better than random. Also checks the predictive state fires (HTM/reafference)."""

from __future__ import annotations

import os
import random
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.agent import Agent  # noqa: E402
from tbt.reward import GridWorld  # noqa: E402


def _run(agent, env, steps):
    s, delta, completions = env.reset(), 0, 0
    for _ in range(steps):
        a = agent.step(s, delta) if agent is not None else random.randrange(4)
        s, delta = env.step(a)
        if delta > 0:
            if agent is not None:
                agent.step(s, delta)                         # let it learn the rewarding arrival, then re-localise
                agent.new_episode()
            completions += 1
            s, delta = env.reset(), 0
    return completions


def test_agent_solves_sparse_reward_grid_online():
    """A 5x5 sparse-reward grid, everything learned online (the column's graph + the goal from the score). The
    predict-then-compare loop reaches the goal many times -- far more than a random walk."""
    env = GridWorld(N=5, goal=(4, 4))
    completions = _run(Agent(n_actions=4, seed=0), env, steps=2000)
    rnd = sum(_run(None, GridWorld(N=5, goal=(4, 4)), steps=2000) for _ in range(3)) / 3
    assert completions > 5 * max(rnd, 1), f"online agent {completions} vs random {rnd}"


def test_predictive_state_fires_and_settles():
    """The HTM predict-then-compare: early on the agent is often surprised (model unlearned); once the dynamics are
    learned, a correctly-predicted move leaves it un-surprised. The predictive state is real, not vestigial."""
    env = GridWorld(N=4, goal=(3, 3))
    agent = Agent(n_actions=4, seed=1)
    s, delta = env.reset(), 0
    early = late = 0
    for t in range(800):
        a = agent.step(s, delta)
        s, delta = env.step(a)
        if delta > 0:
            agent.step(s, delta); agent.new_episode(); s, delta = env.reset(), 0
            continue
        if t < 100:
            early += agent.surprised
        elif t >= 700:
            late += agent.surprised
    assert early > late, f"prediction did not improve: early surprises {early}, late {late}"
