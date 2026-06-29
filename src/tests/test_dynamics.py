"""The KIND-general operator (step 5b): an action does not only MOVE things. L5 learns a position-invariant DELTA in
whatever feature dimension an action changes -- a POSE delta (movement) AND a CONTENT transition (a colour flip / any
in-place change) -- so the SAME column models recolouring the way it models movement, generalizes it across position,
combines it with movement, and lets CONTEXT (the edge exceptions) override the over-general rule. End to end the agent
solves a colour-change scene read entirely from frames -- the effect kind the public tests showed (no movement)."""

from __future__ import annotations

import os
import random
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.agent import Agent                       # noqa: E402
from tbt.l5_displacement import L5_Displacement   # noqa: E402
from tbt.sensor import Sensor, config_state       # noqa: E402

FLIP = 0
N = 12


def _scene(pose, colour, anchor=(2, 2)):
    """A recolouring object (size 1) + a fixed 2x2 anchor (size 4). `colour` is the object's content."""
    return config_state({0: (pose, 1), 1: (anchor, 4)}, {0: colour, 1: None})


def test_recolour_generalizes_across_position():
    """A colour change learned at ONE position is predicted at a position never visited -- the in-place change
    generalizes over location exactly as movement does (the dorsal 'what changed', same operator)."""
    op = L5_Displacement()
    op.observe(_scene((6, 6), "red"), FLIP, _scene((6, 6), "blue"))   # learn red->blue (pose unchanged)
    novel = _scene((9, 9), "red")                                     # same change, a NEW location
    assert novel not in op.edges
    assert op.predict(novel, FLIP) == _scene((9, 9), "blue")          # recoloured in place, never observed here


def test_context_overrides_the_over_general_recolour():
    """The position-invariant recolour is the RULE; an observed no-change is the EXCEPTION (a context where the action
    does nothing) and the edge OVERRIDES the rule -- the 'conditioned on context' half, via the same exception path
    that holds walls/doors."""
    op = L5_Displacement()
    op.observe(_scene((6, 6), "red"), FLIP, _scene((6, 6), "blue"))   # the rule: red -> blue
    off = _scene((4, 4), "red")
    op.observe(off, FLIP, off)                                        # here (a different context) FLIP does nothing
    assert op.predict(off, FLIP) == off                              # the exception wins -> no recolour
    assert op.predict(_scene((9, 9), "red"), FLIP) == _scene((9, 9), "blue")   # elsewhere the rule still fires


def test_movement_and_colour_change_together():
    """One action can BOTH move and recolour; the operator learns and applies both deltas at an unvisited state."""
    op = L5_Displacement()
    op.observe(_scene((6, 6), "red"), FLIP, _scene((7, 6), "blue"))   # moves +1 in x AND recolours
    assert op.predict(_scene((3, 3), "red"), FLIP) == _scene((4, 3), "blue")


class ColorScene:
    """A frame-rendered STATE-CHANGE scene (no movement): a 1-cell object at a FIXED cell whose COLOUR steps along a
    palette under action 0 (+1, clamped) / action 1 (-1); actions 2,3 are no-ops. A fixed 2x2 anchor gives the
    translation-invariant frame. Reaching the LAST colour scores -- learnable only by modelling the colour operator."""

    PALETTE = [2, 3, 4, 6, 7, 8]                                     # avoid 0 (background) and 5 (the anchor)

    def __init__(self):
        self.idx = 0

    def render(self):
        g = [[0] * N for _ in range(N)]
        for dx in (0, 1):
            for dy in (0, 1):
                g[1 + dy][1 + dx] = 5                                # the 2x2 anchor
        g[6][6] = self.PALETTE[self.idx]                            # the colour-changing object (fixed cell)
        return g

    def step(self, a):
        if a == 0:
            self.idx = min(self.idx + 1, len(self.PALETTE) - 1)
        elif a == 1:
            self.idx = max(self.idx - 1, 0)
        scored = 1 if self.idx == len(self.PALETTE) - 1 else 0
        return self.render(), scored


def _run(agent, sensor, steps):
    env = ColorScene()
    s, _ = (sensor.read(env.render()) if sensor is not None else (None, None))
    delta, completions = 0, 0
    for _ in range(steps):
        a = agent.step(s, delta) if agent is not None else random.randrange(4)
        frame, delta = env.step(a)
        if agent is not None:
            s, _ = sensor.read(frame)
        if delta > 0:
            if agent is not None:
                agent.step(s, delta); agent.new_episode(); sensor.reset()
            completions += 1
            env.idx = 0
            if agent is not None:
                s, _ = sensor.read(env.render())
            delta = 0
    return completions


def test_agent_solves_a_colour_change_scene_from_frames():
    """END TO END: the SAME agent solves a sparse-reward scene whose only dynamics are an in-place COLOUR change,
    perceived from frames -- far more completions than a random walk. The effect kind beyond movement."""
    completions = _run(Agent(n_actions=4, seed=0), Sensor(), steps=4000)
    rnd = sum(_run(None, None, steps=4000) for _ in range(3)) / 3
    assert completions > 5 * max(rnd, 1), f"colour-change agent {completions} vs random {rnd}"
