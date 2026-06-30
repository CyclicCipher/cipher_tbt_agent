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
