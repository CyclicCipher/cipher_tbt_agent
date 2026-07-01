"""The column learns its location frame ONLINE via the TD successor representation (no batch eigendecomposition): feed
transitions one at a time, refresh() (eigh-free), and the per-action operator + content readout still predict the next
state correctly. This is step 2 wired into the column."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.column import CorticalColumn  # noqa: E402


def test_column_learns_a_ring_online_without_eigendecomposition():
    """A ring of K states, action 0 = step to the next, fed transition-by-transition. After the eigh-free refresh, the
    column predicts each state's successor under action 0 — the online SR place codes + L5 operator work end to end."""
    K = 6
    col = CorticalColumn(n_entities=K, seed=0)
    for _ in range(50):                                      # learn online: transitions arrive one at a time
        for i in range(K):
            col.observe(i, 0, (i + 1) % K)
    col.refresh()                                           # online consolidation -- NO eigendecomposition
    preds = [col.predict(i, 0) for i in range(K)]
    assert preds == [(i + 1) % K for i in range(K)], preds


def test_sr_reachability_and_value_read_from_the_online_sr():
    """The column reads VALUE + REACHABILITY natively from the online SR (no graph BFS): on a chain 0->1->2->3 with a
    reward only at the absorbing goal 3, every state can REACH the reward and its value rises toward it; a never-seen /
    disconnected state is unreachable and valued 0. This is the deep-planning read the dead-zone / GSG (M1) use."""
    col = CorticalColumn(n_entities=8, seed=0)
    for _ in range(200):
        for i in (0, 1, 2):
            col.observe(i, 0, i + 1)
        col.observe(3, 0, 3)                                    # the goal absorbs (occupies itself)
    R = {3: 1.0}
    assert all(col.reachable(s, R) for s in (0, 1, 2, 3))       # the reward is reachable from the whole chain
    assert col.value(0, R) < col.value(1, R) < col.value(2, R) < col.value(3, R)   # value rises toward the goal
    assert not col.reachable(9, R) and col.value(9, R) == 0.0   # a never-seen state -> unreachable, value 0


def test_l6_is_read_as_the_location_substrate():
    """C1 (COLUMN_AUDIT): the column READS L6 as the location -- `locate(state)` returns the SR-eigenframe place code,
    which ENCODES TOPOLOGY (adjacent states' locations more similar than the antipode) and lives in the binding space
    L4/L5 will use. `None` for a state the L6 frame has not seen. A correctness (mechanism) test, not a score."""
    col = CorticalColumn(n_entities=8, seed=0)
    for _ in range(80):                                      # learn a ring -> the L6 frame has place codes
        for i in range(6):
            col.observe(i, 0, (i + 1) % 6)
    l0, l1, l3 = col.locate(0), col.locate(1), col.locate(3)
    assert l0 is not None and l0.shape[0] == col.d_mem       # the location lives in the binding space (d_mem)
    assert float(l0 @ l1) > float(l0 @ l3)                   # adjacent more similar than the antipode -> topology encoded
    assert col.locate(9) is None                             # a state unknown to the L6 frame -> no location


def test_sense_at_is_l4_over_l6_predict_then_compare():
    """C2 (COLUMN_AUDIT): the TBT cycle step -- L4 predicts the feature at the L6 location, compares to the sensed
    feature, learns by binding it. A FRESH location predicts nothing (not surprised, just learns); re-sensing the SAME
    feature there is PREDICTED (not surprised); a DIFFERENT feature is SURPRISING (the predict-then-compare fires)."""
    col = CorticalColumn(n_entities=16, seed=0)
    for _ in range(80):                                      # an L6 frame over locations 0..5
        for i in range(6):
            col.observe(i, 0, (i + 1) % 6)
    fa, fb = col.L4.encode(("red",)), col.L4.encode(("blue",))
    assert col.sense_at(0, fa) is False                     # nothing bound at 0 yet -> not surprised; learns fa@0
    assert col.sense_at(0, fa) is False                     # re-sense fa@0 -> PREDICTED, not surprised
    assert col.sense_at(0, fb) is True                      # a DIFFERENT feature at 0 -> SURPRISED (the learning signal)


def test_l23_recognition_wired_to_feature_at_location():
    """C4 (COLUMN_AUDIT): L2/3 RECOGNITION wired into the feature-at-location cycle -- `sense_object` recognises the
    sensed object (pose-invariant identity via L2/3) and binds THAT identity at the L6 location, so the map is over
    RECOGNISED objects, not raw patches. Re-sensing the SAME object at a location is recognised (not surprised); a
    DIFFERENT object there is a boundary (surprised). 'The object settled by recognition.'"""
    col = CorticalColumn(n_entities=64, seed=0)
    for _ in range(80):
        for i in range(6):
            col.observe(i, 0, (i + 1) % 6)
    L = [(0, 0), (1, 0), (0, 1)]                             # an L-tromino
    bar = [(0, 0), (1, 0), (2, 0)]                           # a bar -- a structurally different object
    col.sense_object(L, 0)                                  # learn the L-object + bind its identity at location 0
    name, surprised = col.sense_object(L, 0)                # re-sense the SAME object -> recognised
    assert surprised is False, (name, col.feature_at(0))
    _n2, surprised2 = col.sense_object(bar, 0)              # a DIFFERENT object at 0 -> boundary
    assert surprised2 is True


def test_object_state_tracks_the_dynamic_scene():
    """C4 (COLUMN_AUDIT): L2/3's OBJECT STATE -- the compact summary of the DYNAMIC scene (features CHANGED at their
    locations, from sense_at's surprise). Learning the initial scene sets NO state; a CHANGE (a feature replaced at a
    known location -- a key collected) advances the object-state; distinct board-states differ. Not config_state:
    layer-derived, metric, and only the dynamic part."""
    col = CorticalColumn(n_entities=16, seed=0)
    for _ in range(80):
        for i in range(6):
            col.observe(i, 0, (i + 1) % 6)
    key, blk, empty = col.L4.encode(("key",)), col.L4.encode(("block",)), col.L4.encode(("floor",))
    for _ in range(4):                                       # learn the initial scene: key@0, block@3 (antipodes)
        col.sense_at(0, key)
        col.sense_at(3, blk)
    assert col.object_state() == frozenset()                # learning the scene -> no DYNAMIC state yet
    col.sense_at(0, empty)                                   # the key at 0 is COLLECTED (a known feature changes)
    assert col.object_state() == frozenset({(0, empty)})    # the object-state records the change at 0
    col.sense_at(3, empty)                                   # the block leaves 3 -> another change
    assert (3, empty) in col.object_state() and (0, empty) in col.object_state()   # a distinct board-state
    col.reset_object_state()
    assert col.object_state() == frozenset()                # a level boundary resets the dynamic state


def test_the_cycle_recognizes_a_multi_location_object():
    """C2 (COLUMN_AUDIT): the L4-over-L6 cycle over MOVEMENT builds a multi-location OBJECT and RECOGNISES it -- after
    learning distinct features at separated locations, re-sensing each is PREDICTED (not surprised); a wrong feature
    SURPRISES (a boundary). Works because the location code is DG-SPARSIFIED (near-orthogonal across locations); the raw
    diffuse SR place code would degenerate to a global bag."""
    col = CorticalColumn(n_entities=16, seed=0)
    for _ in range(80):                                      # the L6 frame over a 6-location ring
        for i in range(6):
            col.observe(i, 0, (i + 1) % 6)
    feats = {0: col.L4.encode(("A",)), 2: col.L4.encode(("B",)), 4: col.L4.encode(("C",))}
    for _ in range(4):                                       # move over the object: sense each location
        for loc, f in feats.items():
            col.sense_at(loc, f)
    for loc, f in feats.items():
        assert col.sense_at(loc, f) is False, (loc, col.feature_at(loc), f)   # recognised: predicted, not surprised
    assert col.sense_at(0, col.L4.encode(("X",))) is True    # a wrong feature at a known location -> surprise (boundary)


def test_feature_at_location_map_binds_and_reads_back():
    """M5/L7-A: the column maintains an online allocentric MAP -- bind a SENSED feature at a LOCATION (L4 feature ⊗
    L6 place code) across a sensorimotor sequence, then READ it back (predict_feature). An object seen at a place is
    REMEMBERED there, distinct from another place -- the feature-at-location substrate the §3 mechanic library needs."""
    col = CorticalColumn(n_entities=16, seed=0)
    for _ in range(80):                                      # learn a ring so L6 has distinct place codes
        for i in range(6):
            col.observe(i, 0, (i + 1) % 6)
    fa, fb = col.L4.encode(("red",)), col.L4.encode(("blue",))   # two distinct features
    col.bind_at(0, fa)                                       # red at location 0, blue at the antipode 3
    col.bind_at(3, fb)
    assert col.feature_at(0) == fa                           # the map remembers red at 0
    assert col.feature_at(3) == fb                           # and blue at 3
    assert col.feature_at(9) is None                         # a location unknown to the L6 frame -> None


def test_path_integration_is_discrete_graph_tracking():
    """Path integration = PREDICT the next node by the learned edge (no observation needed -- partial observability),
    CORRECT by snapping to a sighting. Discrete graph tracking, exact and online -- no matrix operator over codes."""
    K = 6
    col = CorticalColumn(n_entities=K, seed=0)
    for _ in range(5):
        for i in range(K):
            col.observe(i, 0, (i + 1) % K)
    col.loc_reset(0)
    assert col.loc_move(0) == 1                              # dead-reckon by the learned edge
    assert col.loc_move(0) == 2
    assert col.loc_sense(5) == 5                             # snap to a sighting (correction)
    assert col.loc_move(0) == 0                              # 5 -> 0 on the ring, from the corrected node
    assert col.loc_where() == 0


def test_column_learns_a_line_online():
    """A non-cyclic line (open boundary) learned online — the SR handles the open topology with no metric switch."""
    K = 7
    col = CorticalColumn(n_entities=K, seed=1)
    for _ in range(60):
        for i in range(K - 1):
            col.observe(i, 0, i + 1)
    col.refresh()
    preds = [col.predict(i, 0) for i in range(K - 1)]
    assert preds == [i + 1 for i in range(K - 1)], preds


def test_column_owns_continuous_path_integration_and_disambiguates_animation():
    """P1: the COLUMN (not the sensor) path-integrates the metric location belief. Fed the residual candidates a sensor
    would detect, it LEARNS the per-action translation (L5), tracks the controllable mover, rejects an animation
    distractor via the efference prediction, and coarsens to a recurring state node once controllable."""
    col = CorticalColumn(n_entities=64, seed=0)
    col.track_reset()
    assert col.track(None, [], (2.0, 2.0), cold=(2.0, 2.0)) == (2.0, 2.0)     # cold start: foveate the object
    assert col.track(0, [(4.0, 2.0)], (4.0, 2.0)) == (4.0, 2.0)               # action 0 -> the mover stepped +2x
    assert col.L5.controllable(), "the first real move learns a non-trivial translation -> controllable"
    # a SECOND action-0 with a distractor (autonomous animation far away): the efference (fovea + learned +2x) picks
    # the action-consistent residual, NOT the distractor.
    assert col.track(0, [(6.0, 2.0), (18.0, 18.0)], (18.0, 18.0)) == (6.0, 2.0)
    assert abs(col.L5.move(0)[0] - 2.0) < 0.5 and abs(col.L5.move(0)[1]) < 0.5
    assert col.track_pos() == (6.0, 2.0)
    assert col.track_state(pos_bin=4) == (1, 0)                               # (6//4, 2//4) -- the coarse recurring node


def test_track_gate_stays_off_for_a_non_controllable_scene():
    """P1 (the relocated gate): a scene whose change is NOT action-driven (in-place animation, a constant residual)
    learns a ~zero translation -> NOT controllable -> the state position stays the constant gate-off value, preserving
    the recurring local view a state-change game depends on. Replaces the sensor-internal gate test."""
    col = CorticalColumn(n_entities=64, seed=0)
    col.track_reset()
    for t in range(12):                                                      # a 2x2 block toggling colour IN PLACE
        col.track(t % 4, [(10.5, 10.5)], (10.5, 10.5))
    assert not col.L5.controllable(), f"gate wrongly ON: deltas {col.L5.move_delta}"
    assert col.track_state(pos_bin=4) == (0, 0), "non-controllable scene must keep the constant gate-off position"


# ── SR shortest-path navigation (navigate_to) + grid-cell VECTOR navigation (vector_action) ──────────────────
def test_navigate_to_takes_the_sr_shortest_path_to_a_goal():
    """M1/P3: navigate_to picks the action whose OUTCOME has the highest SR occupancy M[next, goal] (~ γ^distance),
    so it steps along the SHORTEST path to a known goal -- read directly from the SR (no sweep)."""
    col = CorticalColumn(n_entities=6, seed=0)
    ring = 6
    for _ in range(200):                                    # learn the ring: action 0 = +1, action 1 = -1
        for i in range(ring):
            col.observe(i, 0, (i + 1) % ring)
            col.observe(i, 1, (i - 1) % ring)
    R = {2: 1.0}                                            # reward at state 2
    assert col.navigate_to(0, R, [0, 1]) == 0               # 0->1->2 (dist 2) beats 0->5->4->3->2 (dist 4)
    assert col.navigate_to(4, R, [0, 1]) == 1               # 4->3->2 (dist 2) beats the long way
    assert col.navigate_to(0, {99: 1.0}, [0, 1]) is None    # unreachable reward -> None (explore takes over)


def test_vector_action_steers_along_the_goal_vector():
    """V1 (VECTOR_NAV): the ATTRACTIVE field -- vector_action picks the action whose L5 displacement `move_delta` best
    aligns with the goal vector `goal − here`, steering straight toward the goal (grid-cell vector navigation)."""
    col = CorticalColumn(n_entities=16, seed=0)
    for a, d in {0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)}.items():     # the 4 moves' displacements (as P1 learns them)
        col.L5.observe_move(a, d)
    assert col.vector_action((0, 0), (5, 0), [0, 1, 2, 3]) == 0             # goal to the +x -> move right
    assert col.vector_action((0, 0), (0, 5), [0, 1, 2, 3]) == 3             # goal to the +y -> move down
    assert col.vector_action((5, 0), (0, 0), [0, 1, 2, 3]) == 1             # goal to the -x -> move left
    assert col.vector_action((3, 3), (3, 3), [0, 1, 2, 3]) is None          # at the goal -> no move
