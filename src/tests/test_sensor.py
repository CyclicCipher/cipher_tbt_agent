"""The sensor (tbt.sensor): raw frames -> the column's input. The scene state is translation-invariant (so states
recur), the change stream reports what moved, and END TO END the SAME agent solves a sparse-reward scene read FROM
FRAMES (segment -> track -> state -> loop) -- the offline gate before the live game."""

from __future__ import annotations

import os
import random
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.agent import Agent  # noqa: E402
from tbt.sensor import Sensor, config_state, salient_cells  # noqa: E402

MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
N = 12


def _clamp(v):
    return min(max(v, 0), N - 1)


class FrameScene:
    """A rendered-frame sparse-reward scene: a 1-cell mover (colour 7) + a 2x2 landmark (colour 5, the config anchor).
    Reaching the border ring scores -- a goal directed exploration reliably reaches. Emits raw grids (frames)."""

    def __init__(self, pos=(6, 6)):
        self.pos = pos

    def render(self):
        g = [[0] * N for _ in range(N)]
        for dx in (0, 1):
            for dy in (0, 1):
                g[2 + dy][2 + dx] = 5                         # 2x2 landmark (the largest object -> the anchor)
        g[self.pos[1]][self.pos[0]] = 7                      # the (1-cell) mover
        return g

    def step(self, a):
        dx, dy = MOVES[a]
        self.pos = (_clamp(self.pos[0] + dx), _clamp(self.pos[1] + dy))
        scored = 1 if (self.pos[0] in (0, N - 1) or self.pos[1] in (0, N - 1)) else 0
        return self.render(), scored


def test_config_state_is_translation_invariant():
    """The same RELATIVE arrangement shifted anywhere on the board is ONE state -- so states recur for the SR/operator."""
    a = {0: ((3.0, 3.0), 1), 1: ((6.0, 6.0), 4)}
    b = {0: ((8.0, 8.0), 1), 1: ((11.0, 11.0), 4)}           # the same arrangement, shifted by +5
    assert config_state(a) == config_state(b)
    c = {0: ((3.0, 5.0), 1), 1: ((6.0, 6.0), 4)}             # a different arrangement
    assert config_state(a) != config_state(c)


def test_salient_cells_reports_what_changed():
    prev = [[0, 0], [0, 7]]
    cur = [[0, 7], [0, 0]]
    assert salient_cells(prev, cur) == {(1, 0), (1, 1)}


def test_sensor_tracks_a_mover_and_reports_change():
    s = Sensor()
    env = FrameScene(pos=(6, 6))
    st0, ch0 = s.read(env.render())
    assert ch0 == set()                                      # no previous frame -> no change yet
    frame, _ = env.step(3)                                   # mover steps right
    st1, ch1 = s.read(frame)
    assert st1 != st0                                        # the state changed (the mover moved)
    assert ch1                                               # the change stream reports the moved cells


def _run(agent, sensor, steps):
    env = FrameScene()
    s, _ = (sensor.read(env.render()) if sensor is not None else (None, None))
    delta, completions = 0, 0
    for _ in range(steps):
        a = agent.step(s, delta) if agent is not None else random.randrange(4)
        frame, delta = env.step(a)
        if agent is not None:
            s, _ = sensor.read(frame)
        if delta > 0:
            if agent is not None:
                agent.step(s, delta)                         # learn the rewarding arrival
                agent.new_episode(); sensor.reset()
            completions += 1
            env.pos = (6, 6)
            if agent is not None:
                s, _ = sensor.read(env.render())
            delta = 0
    return completions


def test_agent_solves_a_scene_read_from_frames():
    """END TO END: the agent solves a sparse-reward scene perceived entirely FROM FRAMES (sensor: segment -> track ->
    translation-invariant state -> the predict-then-compare loop). Far more completions than a random walk."""
    completions = _run(Agent(n_actions=4, seed=0), Sensor(), steps=4000)
    rnd = sum(_run(None, None, steps=4000) for _ in range(3)) / 3
    assert completions > 5 * max(rnd, 1), f"sensor+agent {completions} vs random {rnd}"
