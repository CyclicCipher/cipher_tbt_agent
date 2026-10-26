"""Tests of the preference graph's first instance (prefgraph.py). Run: python -m pytest tests/test_prefgraph.py """
import numpy as np

from worlds import canvas as cv
from agents.prefgraph import N_CTX, Appreciation, Graph, Preservation


def _nature_cell(world):
    return tuple(int(x) for x in np.argwhere(np.isin(world.origin, (cv.STRIPES, cv.RULE90)))[0])


def test_destroying_appreciated_structure_costs_and_unvalued_structure_does_not():
    world = cv.Canvas(100)
    pres = Preservation()
    st = pres.start(world)
    cell = _nature_cell(world)
    world.colour[cell] = (world.colour[cell] + 1) % cv.K
    assert pres.step(world, dict(st, value=np.zeros(N_CTX)), 1, True, {"felt": np.zeros(N_CTX)}) == 0.0
    st2 = pres.start(world)
    world.colour[cell] = (world.colour[cell] + 1) % cv.K
    assert pres.step(world, dict(st2, value=np.ones(N_CTX)), 1, True, {"felt": np.zeros(N_CTX)}) < 0.0


def test_the_content_edge_writes_what_was_appreciated():
    world = cv.Canvas(100)
    app, pres = Appreciation(), Preservation()
    sa, sp = app.start(world), pres.start(world)
    total = np.zeros(N_CTX)
    for _ in range(2):                                   # the first view teaches; the second is felt as progress
        msg = {}
        app.step(world, sa, 1, True, msg)
        pres.step(world, sp, 1, True, msg)
        total += msg["felt"]
    assert np.allclose(sp["value"], total) and (sp["value"] > 0).any()


def test_a_veto_dominates_any_trade():
    class Fixed:
        def __init__(self, name, sig):
            self.name, self.sig = name, sig

        def start(self, world):
            return {}

        def fork(self, st):
            return {}

        def step(self, world, st, h, real, msg):
            return self.sig
    world = cv.Canvas(0)
    g = Graph([Fixed("joy", 100.0), Fixed("keep", -1.0)], {"joy": 1.0}, veto=("keep",), tol=0.5)
    assert g.step(world, g.start(world), 1) < 0
    g = Graph([Fixed("joy", 100.0), Fixed("keep", -0.4)], {"joy": 1.0}, veto=("keep",), tol=0.5)
    assert g.step(world, g.start(world), 1) == 100.0


def test_the_noisy_screen_causes_no_preservation_loss():
    world = cv.Canvas(103)
    app, pres = Appreciation(), Preservation()
    sa, sp = app.start(world), pres.start(world)
    rng = np.random.default_rng(0)
    for _ in range(60):                                   # walking and waiting only: nothing the agent destroys
        world.step(int(rng.integers(0, 5)))
        msg = {}
        app.step(world, sa, 1, True, msg)
        assert pres.step(world, sp, 1, True, msg) == 0.0
