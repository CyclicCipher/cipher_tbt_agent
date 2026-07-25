"""test_feature_at_location_live.py — L4 ⊗ the thalamic register, DRIVEN BY THE LIVE AGENT (`Agent._sense_frame`).

`test_feature_at_location.py` exercises the L4↔L6a primitive in isolation. This one pins that the agent actually DRIVES it
while playing: the sensorimotor scan places L6a at each location, asks L4 what it expects there, senses what is actually
there, and binds (feature ⊗ location) into the thalamus. It replaced `_static_cells`, which re-segmented the whole frame
each step into a hand-built dict maintained outside any column.

The property that matters is the one the dict could never give: a change the agent did NOT cause becomes a measured
PERCEPTUAL PREDICTION ERROR. That is the signal the key→door association needs, and it is why the scan covers every
location L4 has BOUND rather than only the cells currently occupied — a door opening is a cell going EMPTY, which a scan of
occupied cells cannot see by construction.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent          # noqa: E402
from tbt.perceive import segment     # noqa: E402

SELF, WALL, DOOR = 2, 1, 5


def _room(door: bool = True):
    """A small walled room with the agent inside and a DOOR that can open (become background)."""
    g = np.zeros((6, 8), dtype=int)
    g[0, :] = WALL; g[5, :] = WALL; g[:, 0] = WALL; g[:, 7] = WALL
    if door:
        g[2, 4] = DOOR
    g[1, 1] = SELF
    return g.tolist()


def _agent():
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a._extent = [(0, 7), (0, 5)]
    return a


def test_the_scan_learns_the_room_and_stops_being_surprised():
    """L4 learns feature-at-location from the agent's own scan: the first sweeps have nothing to predict from, and once the
    room is learned the agent is no longer surprised by it. A static world it understands costs nothing."""
    a = _agent()
    surprises = [a._sense_frame(segment(_room()), SELF)[1] for _ in range(4)]
    assert surprises[-1] == 0.0, f"a learned static room must stop surprising, got {surprises}"
    assert a._bound, "the scan must record the locations it has bound"


def test_a_change_the_agent_did_not_cause_is_a_prediction_error():
    """THE POINT. The door opens somewhere the agent never touched. Under the old hand-built dict that was a silently
    different entry; now L4 expected a door there and got background, so it is a measured surprise — which then ABSORBS as
    L4 relearns, so it reports the change rather than complaining forever."""
    a = _agent()
    for _ in range(4):
        a._sense_frame(segment(_room(door=True)), SELF)
    assert a._sense_frame(segment(_room(door=True)), SELF)[1] == 0.0, "the closed room must be fully predicted first"

    opened = [a._sense_frame(segment(_room(door=False)), SELF)[1] for _ in range(6)]
    assert opened[0] > 0.0, "the door opening must register as a perceptual prediction error"
    assert opened[-1] == 0.0, f"and must absorb once L4 relearns, got {opened}"


def test_an_emptied_cell_is_still_scanned():
    """Why the scan covers BOUND locations, not just occupied ones: the door cell leaves the frame's objects entirely when it
    opens, so a scan of what is currently there could never notice. It stays in `_bound` and is still checked."""
    a = _agent()
    a._sense_frame(segment(_room(door=True)), SELF)
    assert (4, 2) in a._bound, "the door's cell must be bound while it is there"
    surface, _ = a._sense_frame(segment(_room(door=False)), SELF)
    assert (4, 2) in a._bound, "and must remain scanned once it has emptied"
    assert (4, 2) not in surface, "an emptied cell is not part of the occupied surface"


def test_the_surface_serves_the_planner_and_excludes_movers():
    """The scan still produces the `{cell: feature}` surface the world model presses into — the static, occupied part. Movers
    are perceived and learned by L4 like everything else, but they are tracked as objects in their own right."""
    a = _agent()
    a._movers.add(WALL)                                  # pretend the wall colour is a tracked mover
    surface, _ = a._sense_frame(segment(_room()), SELF)
    assert all(f != WALL for f in surface.values()), "a tracked mover must not appear in the static surface"
    assert (4, 2) in surface and surface[(4, 2)] == DOOR, "a non-mover static feature must be in the surface"
