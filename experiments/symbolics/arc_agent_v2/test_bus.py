"""Tests of the bus's guarantees for efference copies and receptive fields (bus.py). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/test_bus.py -q
"""
import numpy as np

from arc_agent_v2.brain import Brain
from arc_agent_v2.config import WALK, BrainConfig, Region
from arc_agent_v2.walk_world import WalkWorld

TWO = BrainConfig(WALK.sdr, ("motor", "retina"),
                  (Region("assoc", 3, "learned", ("motor", "retina"), moves=("motor",)),
                   Region("vis", 2, "learned", ("retina",), moves=())), efference=("motor",))


def trained(cfg, seed=0, steps=1000):
    world = WalkWorld("E1 torus", seed=seed)
    acts = np.random.default_rng(seed).integers(0, world.A, steps).tolist()
    brain = Brain(cfg)
    brain.episode(world.stream(0, acts))
    brain.sleep()
    return world, brain


def test_columns_sharing_a_field_stay_in_step():
    world, brain = trained(TWO)
    acts = np.random.default_rng(9).integers(0, world.A, 200).tolist()
    brain.begin()
    for sensor, x in world.stream(0, acts):
        brain.step(x, sensor, learn=False)
        places = [c.l6.place for c in brain.regions["assoc"]]
        assert places[0] is not None and all(p == places[0] for p in places)
    assert brain.bus.faults == 0


def test_a_column_never_moves_on_a_source_outside_its_field():
    world, brain = trained(TWO)
    vis = brain.regions["vis"]
    assert all(c is not v for c, _ in brain.bus.routes["motor"] for v in vis)
    brain.begin()
    starts = [c.l6.place for c in vis]
    for sensor, x in world.stream(0, [0, 2, 1, 3, 0]):
        brain.step(x, sensor, learn=False)
    assert [c.l6.place for c in vis] == [c.l6.start for c in vis] == starts


def test_a_missed_message_is_caught_not_drifted():
    world, brain = trained(WALK)
    col = brain.columns()[0]
    brain.begin()
    brain.step(world.labels[0], "retina", learn=False)
    col.step(world.action[0], "motor", True, col.stamp + 5)   # stepped outside the bus
    brain.step(world.action[2], "motor", learn=False)
    assert brain.bus.faults == 1 and col.l6.place is None


def test_efference_copies_are_never_content():
    world, brain = trained(WALK)
    assert brain.predict("motor") is None                      # nothing predicts the agent's own action
    contents = brain.hippocampus.contents()
    assert contents and all(s == "retina" for _, s, _, _ in contents)
