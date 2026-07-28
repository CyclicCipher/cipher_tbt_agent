"""hippocampus/map.py — the allocentric world-STATE: the EC/place core of the hippocampus (DESIGN §2, slice 1).

WHAT. A forkable snapshot of the world the rollout simulates *in*: the AGENT's self-location (place cells), OBJECTS at
world poses (object-vector cells), and the BOUNDARY (the frame's extent / boundary cells). It is the STATE — "the world
right now, including me" — that binds recognition (WHAT things are) and dynamics (HOW they change) into something a rollout
can copy, branch, and run forward (DESIGN §0). Without it the rollout has nothing coherent to simulate in; that is the block
this removes.

WHY IT IS DATA, NOT A COLUMN. A column is STATEFUL and shared — you cannot cheaply fork it to try a hypothetical without
disturbing the live percept. The complementary-learning-systems split (Rolls; `reference_hippocampus`) is exactly this: the
slow, shared, learned TRANSFORM lives in the column's `operator` (path integration) + object dynamics; the fast, episodic,
FORKABLE state lives here, as plain poses in dicts. So `snapshot` is a cheap copy (a pose is an immutable tuple), and a
rollout forks the state while REUSING the one learned model — it re-derives neither path integration nor value. This is the
`project_representation_shortcut_lesson` applied: make the STATE general (a forkable world), and the rollout falls out.

REUSE, NOT REINVENT (DESIGN §5). The `body` operator (`operator.MotionOperator`, ego=True) is the column's OWN — path
integration in the WORLD frame is the same group action as in the object frame, so the map borrows the learned operator by
reference and applies it; it never learns motion itself. A pose is `(position, R)` (operator.py): an n-vector + an n×n
rotation matrix, so n=2 (ARC) and n=3 (3-D) are one code path.

SCOPE (this slice). State + snapshot + agent path-integration + place/remove + boundary + loop-closure anchor. Moving
OBJECTS under the learned (state-conditioned) dynamics is the ROLLOUT's job (replay.py, next slice), which forks a map and
drives the column's forward model over it. The SDR place read-out (grid code of the self-location, for DG/CA3) is added
where DG/CA3 need it. The ego→allo transform for a MOVING sensor stays deferred — ARC's board is world-anchored, so the two
source columns (nav pose, scene objects) are already in one world frame (DESIGN §4; `reference_hippocampus`).
"""

from __future__ import annotations


class WorldMap:
    """The allocentric world-state: `agent` (self-location pose), `objects` ({id: pose}), `bounds` (per-axis extent).

    Forkable by `snapshot` (poses are immutable, so copying the dicts forks the whole state); the learned `body` operator
    is SHARED by reference, never copied — the fast-forkable-state / slow-shared-model split. `move_agent` is functional
    (returns a NEW map) so a rollout branches without disturbing its parent; `place`/`remove`/`anchor` mutate the fork you
    hold, which is what an in-progress simulation does to its own copy."""

    def __init__(self, agent, objects=None, bounds=None, body=None, static=None, kinds=None) -> None:
        self.agent = agent                                   # (position, R) — self-location (place cells); may be None pre-localize
        self.objects = dict(objects) if objects else {}      # object_id -> (position, R) — objects at world poses
        self.bounds = tuple(bounds) if bounds else None      # per-axis (lo, hi) extent (boundary cells); None = unbounded
        self.body = body                                     # the SHARED, learned body operator (ego); referenced, not owned
        self.kinds = dict(kinds) if kinds else {}            # object_id -> the FEATURE that object is an instance of. The id
        #   says WHICH one (so the right body moves); the kind says WHAT it is (so its dynamics are learned once for the type
        #   and not afresh per instance). They were the same thing only while every object had a unique colour.
        self.static = dict(static) if static else {}         # {cell: feature} — what is PERCEIVED at each untracked cell. Raw
        #                                                       sensory occupancy, NOT a verdict: whether pressing into a feature
        #                                                       goes anywhere is the forward model's prediction, never stored here.

    def snapshot(self) -> "WorldMap":
        """A forked copy for a hypothetical branch: the poses (immutable tuples) and the object dict are copied, the learned
        `body` operator + `static` occupancy are shared. Cheap — O(objects), which is why the state is data, not a re-run column."""
        return WorldMap(self.agent, self.objects, self.bounds, self.body, self.static, self.kinds)

    def kind_of(self, occupant):
        """What KIND a thing pressed into is. A tracked object maps to its feature; a static feature, the "edge" and None are
        already kinds and pass through. This is what the forward model must key its learned dynamics on — pressing a crate
        teaches you about CRATES, not about that particular crate."""
        return self.kinds.get(occupant, occupant)

    def place(self, obj_id, pose) -> None:
        """Put/replace an object at a world pose (mutates this fork)."""
        self.objects[obj_id] = pose

    def remove(self, obj_id) -> None:
        """Remove an object from this fork (e.g. a cleared block) — a no-op if absent."""
        self.objects.pop(obj_id, None)

    def move_agent(self, action) -> "WorldMap":
        """Path-integrate the agent by `action` in the WORLD frame, reusing the shared learned body operator — returns a NEW
        map (functional, so a rollout tree branches cleanly). This is FREE motion, the efference copy: it knows nothing about
        obstacles, because whether a press goes anywhere is a PREDICTION the forward model makes from what is felt, not a fact
        the map stores. An unlearned action is the identity (the operator's own correct prior: predict staying put)."""
        m = self.snapshot()
        if self.body is not None:
            m.agent = self.body.apply(self.agent, action)
        return m

    def occupant(self, cell):
        """What is at `cell`: a TRACKED object's id if one is there, else the PERCEIVED static feature, else the "edge" beyond
        the board, else None (in-bounds and empty). The forward model asks "what am I pressing into", and the boundary is a
        real answer to that — an object pressed against the edge is backed by the edge, a perceived backdrop like any other,
        so the occasion-setting gate treats it as one instead of confusing "edge behind" with "open space behind"."""
        for oid, pose in self.objects.items():
            if tuple(round(c) for c in pose[0]) == cell:
                return oid
        if not self.in_bounds(cell):
            return "edge"
        return self.static.get(cell)

    def anchor(self, position) -> None:
        """LOOP CLOSURE: reset the agent's POSITION to a sensed landmark coordinate, keeping its orientation — re-seeing a
        landmark is what bounds path-integration drift (`reference_hippocampus`). Orientation-only re-anchoring and full
        pose fixes come with the moving-sensor slice; ARC re-anchors position from the world-anchored board."""
        if self.agent is None:
            self.agent = (tuple(float(c) for c in position), None)
            return
        _, R = self.agent
        self.agent = (tuple(float(c) for c in position), R)

    def in_bounds(self, position) -> bool:
        """Is a position within the world's extent (the boundary as a container)? True if no bounds are set. Used by the
        rollout to keep hypotheticals on the board."""
        if self.bounds is None:
            return True
        return all(lo <= x <= hi for x, (lo, hi) in zip(position, self.bounds))

    def key(self):
        """A hashable signature of the state for VISITED-PRUNING in a rollout search — two states with the same agent + object
        CELLS are the same search node, so a graph search over world-states stays O(states) and never re-expands, avoiding the
        2^K flat-action blow-up. Orientation is included too (it matters for an object that turns).

        Position collapses to the CELL, not to three decimals, and that granularity is load-bearing rather than cosmetic. A
        model still converging predicts fractional deltas — a wall whose blocking correction has reached −0.94 leaves the body
        0.06 of a cell inside it — and at three decimals every one of those near-identical states is a distinct node, so
        pruning stops working and the search explodes. The board is cells; every other predicate here already rounds to one;
        the search's notion of "the same state" has to agree with them. Nothing about the model's PREDICTIONS is rounded — only
        which nodes count as the same node."""
        def _sig(pose):
            if pose is None:
                return None
            p, R = pose
            return (tuple(round(c) for c in p),
                    tuple(round(x, 3) for row in R for x in row) if R is not None else None)
        return (_sig(self.agent), tuple(sorted((oid, _sig(p)) for oid, p in self.objects.items())))
