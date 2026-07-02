"""L6_NONABELIAN Stage 1c -- a NON-ABELIAN test environment (the prerequisite for testing the redesign / the cross-layer
unification). A heading-carrying agent (pose = x, y, theta) with BODY-FRAME actions FORWARD / TURN_L / TURN_R = the SE(2)
group. FORWARD moves in the CURRENT heading, so its effect DEPENDS on theta: the actions do NOT commute, and the abelian
`move_delta` (ONE translation per action) CANNOT represent FORWARD. This is the concrete inconsistency the refactor fixes:
L6 tracks POSITION but must track the POSE (the group element)."""

from __future__ import annotations

import math
import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


class OrientationWorld:
    """A heading-carrying agent on the plane -- the smallest SE(2) (non-abelian) dynamics. Pose = (x, y, theta); actions:
    FORWARD (step in the current heading), TURN_L / TURN_R (rotate by +/- 90 deg, so headings stay axis-aligned/clean)."""

    STEP = 1.0
    DTHETA = math.pi / 2

    def __init__(self):
        self.x, self.y, self.theta = 0.0, 0.0, 0.0

    def pose(self):
        return (round(self.x, 6), round(self.y, 6), round(self.theta % (2 * math.pi), 6))

    def pos(self):
        return (round(self.x, 6), round(self.y, 6))

    def step(self, action):
        if action == "FORWARD":
            self.x += self.STEP * math.cos(self.theta)
            self.y += self.STEP * math.sin(self.theta)
        elif action == "TURN_L":
            self.theta += self.DTHETA
        elif action == "TURN_R":
            self.theta -= self.DTHETA
        return self.pose()


def _drive(actions):
    w = OrientationWorld()
    for a in actions:
        w.step(a)
    return w


class _St:
    def __init__(self, name):
        self.name = name


class OrientationGame:
    """L6_NONABELIAN S1e step 1 -- a NON-ABELIAN nav FRAME (duck-typed like NavGame, drivable by TbtPolicy). An ORIENTED
    mover (an asymmetric L, so heading is visible) with BODY-FRAME actions FORWARD / TURN_L / TURN_R (SE(2)); reach the
    interior goal -> level up. FORWARD depends on the heading, so navigation requires COMPOSING turns + forwards
    (non-abelian) -- the env the pose machinery (`track_pose`/`pose_state`) targets and the abelian `move_delta` cannot."""

    N = 24
    STEP = 2
    GOAL = (12, 12)
    SHAPE = [(0, 0), (1, 0), (0, 1)]                                # an ASYMMETRIC L-tromino -> its 4 rotations are DISTINCT,
    #                                                                so a turn is VISIBLE and heading is READABLE from the shape
    HEADINGS = [(1, 0), (0, 1), (-1, 0), (0, -1)]                   # h = 0..3 -> E/N/W/S (theta = h*90 matches the shape rotation)

    def __init__(self, levels=8):
        self.n_levels = levels
        self.levels_completed = 0
        self.state = _St("NOT_PLAYED")
        self.actions_taken = 0
        self.mx, self.my, self.h = 2, 2, 0

    @staticmethod
    def _rotate(shape, h):
        out = []
        for dx, dy in shape:
            for _ in range(h % 4):
                dx, dy = -dy, dx                                    # 90 deg CCW about the anchor
            out.append((dx, dy))
        return out

    @property
    def frame(self):
        g = [[0] * self.N for _ in range(self.N)]
        for dx, dy in self._rotate(self.SHAPE, self.h):            # heading is VISIBLE -> the mover is rendered ROTATED
            x, y = self.mx + dx, self.my + dy
            if 0 <= x < self.N and 0 <= y < self.N:
                g[y][x] = 7
        return [g]

    @property
    def available(self):
        return ["FORWARD", "TURN_L", "TURN_R"]

    def step(self, name, data=None):
        if name == "RESET":
            self.state = _St("NOT_FINISHED")
            self.mx, self.my, self.h = 2, 2, 0
            return self
        if self.state.name in ("WIN", "NOT_PLAYED"):
            return self
        self.actions_taken += 1
        if name == "FORWARD":
            dx, dy = self.HEADINGS[self.h]
            self.mx = min(max(self.mx + dx * self.STEP, 0), self.N - 2)
            self.my = min(max(self.my + dy * self.STEP, 0), self.N - 2)
        elif name == "TURN_L":
            self.h = (self.h + 1) % 4
        elif name == "TURN_R":
            self.h = (self.h - 1) % 4
        if abs(self.mx - self.GOAL[0]) <= 1 and abs(self.my - self.GOAL[1]) <= 1:
            self.levels_completed += 1
            if self.levels_completed >= self.n_levels:
                self.state = _St("WIN")
            else:
                self.mx, self.my, self.h = 2, 2, 0
        return self


def test_orientation_game_is_a_valid_non_abelian_frame():
    """Step 1: OrientationGame is a duck-typed FRAME (frame/available/step/levels) whose dynamics are non-abelian --
    FORWARD's effect depends on heading (turn-then-forward != forward), and it is SOLVABLE (turn to face the goal, advance)."""
    g = OrientationGame()
    g.step("RESET")
    assert len(g.frame) == 1 and len(g.frame[0]) == 24 and set(g.available) == {"FORWARD", "TURN_L", "TURN_R"}
    assert sum(v == 7 for row in g.frame[0] for v in row) == 3               # the 3-cell (L-tromino) mover is rendered
    # a TURN rotates the ASYMMETRIC mover -> the frame CHANGES: heading is VISIBLE in the shape (route-1 perception)
    t = OrientationGame(); t.step("RESET"); before = t.frame; t.step("TURN_L")
    assert t.frame != before                                                 # turn IS visible (the asymmetric shape rotates)
    # non-abelian: from the start, FORWARD (heading E) vs TURN_L-then-FORWARD (heading N) go different ways
    a = OrientationGame(); a.step("RESET"); a.step("FORWARD")
    b = OrientationGame(); b.step("RESET"); b.step("TURN_L"); b.step("FORWARD")
    assert (a.mx, a.my) != (b.mx, b.my) and a.mx > 2 and b.my > 2
    # SOLVABLE: face E, advance to x=12; face N, advance to y=12 -> goal
    g = OrientationGame(levels=1); g.step("RESET")
    for _ in range(5):
        g.step("FORWARD")                                                    # E: x 2->12
    g.step("TURN_L")
    for _ in range(5):
        g.step("FORWARD")                                                    # N: y 2->12
    assert g.levels_completed == 1


def test_env_is_non_abelian_forward_and_turn_do_not_commute():
    """The env is genuinely NON-ABELIAN: FORWARD then TURN lands in a different place than TURN then FORWARD (because
    FORWARD's direction depends on the heading TURN changes). This is the order-dependence the abelian grid cannot hold."""
    fwd_then_turn = _drive(["FORWARD", "TURN_L"]).pos()
    turn_then_fwd = _drive(["TURN_L", "FORWARD"]).pos()
    assert fwd_then_turn != turn_then_fwd                                    # order matters -> SE(2) is non-abelian
    assert fwd_then_turn == (1.0, 0.0) and turn_then_fwd == (0.0, 1.0)       # concretely: +x vs +y


def test_abelian_move_delta_CANNOT_represent_forward():
    """The concrete INCONSISTENCY the refactor fixes: over POSITION-only, the SAME action FORWARD has FOUR different
    displacements (one per heading), so `move_delta` (which learns ONE Δ per action) is ill-defined -- the position-only
    dynamics are NON-DETERMINISTIC. The state must be the full POSE for FORWARD to be a well-defined operator."""
    deltas = set()
    for turns in range(4):                                                  # start at each of the 4 headings
        w = OrientationWorld()
        for _ in range(turns):
            w.step("TURN_L")
        x0, y0 = w.pos()
        w.step("FORWARD")
        x1, y1 = w.pos()
        deltas.add((round(x1 - x0, 6), round(y1 - y0, 6)))
    assert len(deltas) == 4                                                  # 4 distinct FORWARD displacements -> no single move_delta
    # ... whereas over the full POSE, FORWARD IS deterministic (pose -> pose'): a well-defined operator, once L6 tracks pose
    seen = {}
    for turns in range(4):
        w = OrientationWorld()
        for _ in range(turns):
            w.step("TURN_L")
        before = w.pose()
        after = w.step("FORWARD")
        seen[before] = after
    assert len(seen) == 4 and all(a != b for b, a in seen.items())          # deterministic pose->pose (distinct per heading)


def test_S1e_operator_pose_path_integration_makes_forward_deterministic():
    """L6_NONABELIAN S1e (the ENGINE): the column path-integrates a POSE by COMPOSING learned body-frame operators
    (`track_pose`: P ← P·G), so the belief tracks HEADING, matches OrientationWorld, and makes FORWARD DETERMINISTIC over
    the pose state -- which the additive POSITION cannot. The operator DRIVING a non-abelian state. (Live wiring into the
    agent awaits HEADING PERCEPTION -- the agent perceives position, not orientation.)"""
    import numpy as np

    from tbt.column import CorticalColumn
    from tbt.operator import Operator

    def se2(x, y, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

    # LEARN the body-frame operator per action: G = pose_before^-1 · pose_after -- CONSTANT per action (so learnable).
    # (Use the RAW pose attributes, not pose() which rounds to 6 decimals.)
    Gs = {}
    for a in ("FORWARD", "TURN_L", "TURN_R"):
        incs = []
        for turns in range(4):
            w = OrientationWorld()
            for _ in range(turns):
                w.step("TURN_L")
            before = se2(w.x, w.y, w.theta)
            w.step(a)
            incs.append(np.linalg.inv(before) @ se2(w.x, w.y, w.theta))
        assert all(np.allclose(incs[0], g, atol=1e-9) for g in incs)         # the body-frame op is CONSTANT -> learnable
        Gs[a] = Operator(incs[0])

    # path-integrate a sequence through the column -> matches OrientationWorld's pose
    col = CorticalColumn(n_entities=8, seed=0)
    col.track_pose_reset()
    world = OrientationWorld()
    pose = (0.0, 0.0, 0.0)
    for a in ["FORWARD", "TURN_L", "FORWARD", "FORWARD", "TURN_R", "FORWARD", "TURN_L", "FORWARD"]:
        pose = col.track_pose(Gs[a])
        world.step(a)
    assert np.allclose(pose[:2], (world.x, world.y), atol=1e-6)              # position matches the env
    assert abs(((pose[2] - world.theta + np.pi) % (2 * np.pi)) - np.pi) < 1e-6   # heading matches (dead-reckoned from turns)

    # FORWARD is DETERMINISTIC over the POSE state (heading is in the key) -- 4 headings -> 4 distinct outcomes
    fwd = {}
    for turns in range(4):
        col.track_pose_reset()
        for _ in range(turns):
            col.track_pose(Gs["TURN_L"])
        before = col.pose_state()
        col.track_pose(Gs["FORWARD"])
        fwd[before] = col.pose_state()
    assert len(fwd) == 4 and len(set(fwd.values())) == 4                     # distinct, deterministic (vs 4-valued over position)

    # non-abelian in the BELIEF: FORWARD∘TURN ≠ TURN∘FORWARD
    col.track_pose_reset(); col.track_pose(Gs["FORWARD"]); col.track_pose(Gs["TURN_L"])
    p1 = col.pose_state()
    col.track_pose_reset(); col.track_pose(Gs["TURN_L"]); col.track_pose(Gs["FORWARD"])
    p2 = col.pose_state()
    assert p1 != p2


def test_S1e_step4_pose_aware_vector_nav_navigates_the_non_abelian_env():
    """S1e step 4 (FIX vector nav): the POSE-AWARE achiever composes TURN + FORWARD to reach a goal in the non-abelian env
    -- ALIGN-THEN-ADVANCE -- which the abelian `vector_action` (fixed per-action displacement) cannot. Given the pose belief
    + the learned SE(2) operators, it descends Φ = distance + λ·heading-error to the goal, and it USES turns."""
    import numpy as np

    from tbt.column import CorticalColumn
    from tbt.operator import Operator

    def se2(x, y, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

    Gs = {}                                                                    # learn the body-frame SE(2) operator per action
    for idx, name in [(0, "FORWARD"), (1, "TURN_L"), (2, "TURN_R")]:
        w = OrientationWorld()
        before = se2(w.x, w.y, w.theta)
        w.step(name)
        Gs[idx] = Operator(np.linalg.inv(before) @ se2(w.x, w.y, w.theta))

    col = CorticalColumn(n_entities=8, seed=0)
    col.pose_ops = Gs
    col.track_pose_reset()                                                     # _pose = identity (0, 0, heading 0/east)
    goal, used_turn = (4.0, 4.0), False
    for _ in range(60):
        a = col._pose_vector_action(goal, [0, 1, 2])
        if a is None:
            break
        used_turn = used_turn or a in (1, 2)
        col.track_pose(Gs[a])                                                  # apply the chosen operator (dead-reckon the pose)
    x, y = float(col._pose[0, 2]), float(col._pose[1, 2])
    assert (x - 4.0) ** 2 + (y - 4.0) ** 2 <= 2.0                              # navigated to the goal (align-then-advance)
    assert used_turn                                                          # it TURNED -- non-abelian, not just forward


def test_S1e_step4a_pose_ops_learned_online_then_the_achiever_navigates():
    """S1e step 4a: the per-action SE(2) operators are LEARNED ONLINE (`learn_pose_op` from pose transitions), recover the
    true body-frame increments, and the pose-aware achiever then navigates OrientationWorld to the goal with the LEARNED
    operators (not hand-given ones)."""
    import numpy as np

    from tbt.column import CorticalColumn
    from tbt.operator import Operator

    def se2(x, y, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

    col = CorticalColumn(n_entities=8, seed=0)
    # LEARN the operators online: for each action, feed several (pose_before, pose_after) transitions from the env
    for idx, name in [(0, "FORWARD"), (1, "TURN_L"), (2, "TURN_R")]:
        for turns in range(4):                                                  # from each heading -> the SAME body-frame increment
            w = OrientationWorld()
            for _ in range(turns):
                w.step("TURN_L")
            before = se2(w.x, w.y, w.theta)
            w.step(name)
            col.learn_pose_op(idx, before, se2(w.x, w.y, w.theta))
    # the learned FORWARD op is the body-frame translate(+1); TURN_L is a +90 rotation
    assert np.allclose(col.pose_ops[0].M, se2(1.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(col.pose_ops[1].M[:2, :2], se2(0, 0, np.pi / 2)[:2, :2], atol=1e-6)

    # the achiever navigates with the LEARNED operators
    col.track_pose_reset()
    used_turn = False
    for _ in range(60):
        a = col._pose_vector_action((4.0, 4.0), [0, 1, 2])
        if a is None:
            break
        used_turn = used_turn or a in (1, 2)
        col.track_pose(col.pose_ops[a])
    assert (col._pose[0, 2] - 4.0) ** 2 + (col._pose[1, 2] - 4.0) ** 2 <= 2.0 and used_turn


def test_S1e_step4_solves_orientation_game_end_to_end():
    """S1e step 4 (the LIVE SOLVE): the REAL agent solves the non-abelian OrientationGame end to end -- ROUTE-1 perception
    (orientation from the mover's shape via L2/3) → online pose-operator learning (`learn_pose_op`) → the pose-aware achiever
    (`_pose_vector_action`, align-then-advance) → the goal in RAW metric coords derived from the completing action's operator.
    This is the whole S1e stack (perception + operators + achiever) closing the loop on a genuinely non-abelian env that the
    abelian machinery cannot represent. The abelian games stay green (no regression -- test_path_integration NavGame 8/8)."""
    from arc_sdk import TbtPolicy
    game = OrientationGame(8)
    policy = TbtPolicy(seed=0, local=True, integrate=True)
    frame = game
    for _ in range(1500):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
    assert game.levels_completed == 8, game.levels_completed          # SOLVED the non-abelian env end to end
    gr = policy.agent._goal_raw
    assert gr is not None and abs(gr[0] - 12.0) <= 1.0 and abs(gr[1] - 12.0) <= 1.0   # the RAW goal, derived via the completing operator


def test_S2_discover_relations_from_the_agents_own_learned_operators():
    """L6_NONABELIAN Stage 2 (from LEARNED operators, not a hand-built group): after the agent plays OrientationGame and
    learns its SE(2) `pose_ops` online, `col.discover_relations` LOOP-CLOSES the free monoid on them into the Cayley graph.
    It discovers, purely from the learned operators, that a TURN round-trip is a RELATION (TURN_L∘TURN_R = e, a length-2 loop
    to identity), and that the group is NON-ABELIAN (FORWARD does not commute with a TURN). The quotient a planner searches."""
    from arc_sdk import TbtPolicy
    game = OrientationGame(8)
    policy = TbtPolicy(seed=0, local=True, integrate=True)
    frame = game
    for _ in range(400):                                                    # play enough to learn the three pose operators
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
    col = policy.agent.col
    assert set(col.pose_ops) == {0, 1, 2}                                   # FORWARD / TURN_L / TURN_R learned
    _elements, relations = col.discover_relations()
    assert any(eqw == () and len(w) == 2 for w, eqw in relations)           # a TURN round-trip closes to identity (a discovered relation)
    turn_l, fwd = col.pose_ops[1], col.pose_ops[0]
    assert not turn_l.commutes_with(fwd)                                    # NON-ABELIAN: FORWARD∘TURN ≠ TURN∘FORWARD (the quotient needs search)


def test_pose_achiever_honors_the_cost_field_V2_one_vector_nav():
    """(b) vector-nav unification: the POSE achiever is a FULL vector-nav citizen on the ONE cost currency (V2), NOT a
    stripped parallel variant. With the forward cell WALLED (cost = IMPASSABLE) it EXCLUDES that step, where without the wall
    it would advance -- the same cost-field repulsion the abelian `vector_action` has. (The geodesic detour when fully stuck
    stays the SR `navigate_to`, not a search -- no Dijkstra.)"""
    import numpy as np

    from tbt.column import CorticalColumn, IMPASSABLE
    from tbt.operator import Operator

    def se2(x, y, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

    col = CorticalColumn(n_entities=8, seed=0)
    col.pose_ops = {0: Operator(se2(1, 0, 0)), 1: Operator(se2(0, 0, np.pi / 2)), 2: Operator(se2(0, 0, -np.pi / 2))}
    col.track_pose_reset()                                                # at (0,0), facing east
    goal = (4.0, 0.0)
    assert col._pose_vector_action(goal, [0, 1, 2]) == 0                  # normally: FORWARD, straight to the east goal
    col.learn_cost((1, 0), IMPASSABLE)                                    # WALL the forward cell (FORWARD lands at (1,0))
    assert col._pose_vector_action(goal, [0, 1, 2]) != 0                  # V2: FORWARD excluded -- the cost field repels the pose achiever too


def test_S1e_step2_pose_path_engages_on_the_non_abelian_env():
    """S1e step 2 (live wiring): driving the REAL agent on OrientationGame, the POSE path ENGAGES -- the non-abelian GATE
    trips (`heading_dependent`: FORWARD's direction is inconsistent) and the agent's state node becomes a POSE (3-tuple:
    x, y, heading), not just position. NavGame stays on `track_state` (no abelian regression -- test_path_integration).
    (SOLVING OrientationGame is step 4; this validates the wiring + gate.)"""
    from arc_sdk import TbtPolicy
    game = OrientationGame(8)
    policy = TbtPolicy(seed=0, local=True, integrate=True)
    frame = game
    for _ in range(400):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
    col = policy.agent.col
    assert col.L5.heading_dependent()                                       # the non-abelian gate tripped (FORWARD inconsistent)
    assert policy.agent._prev is not None and len(policy.agent._prev[0]) == 3   # the agent's state node is a POSE (x, y, heading)


def test_S1e_heading_PERCEIVED_from_movement_direction_drives_the_pose():
    """L6_NONABELIAN S1e (heading perception): the agent PERCEIVES its heading from HOW IT MOVES -- a forward move's
    position-delta direction IS the heading (no shape-orientation needed, reusing the position observation). This is the
    observable signal that feeds the pose belief and makes the 4 headings DISTINGUISHABLE in the state -- so FORWARD becomes
    deterministic. (Limitation: a turn produces no movement, so heading is stale until the next forward.)"""
    import numpy as np

    from tbt.column import CorticalColumn
    col = CorticalColumn(n_entities=8, seed=0)

    states = set()
    for turns in range(4):                                                      # each of the 4 headings
        w = OrientationWorld()
        for _ in range(turns):
            w.step("TURN_L")
        x0, y0 = w.x, w.y
        w.step("FORWARD")                                                       # a forward move reveals the heading
        perceived = col.track_heading((w.x - x0, w.y - y0))
        assert abs(((perceived - w.theta + np.pi) % (2 * np.pi)) - np.pi) < 1e-6   # movement direction == true heading
        col.sense_pose(x0, y0, perceived)                                       # correct the pose belief from perception
        states.add(col.pose_state())
    assert len(states) == 4                                                     # perception distinguishes the 4 headings -> FORWARD deterministic

    # the honest limitation: a TURN (no movement) leaves the heading stale
    h = col.track_heading((2.0, 0.0))                                           # a real move sets heading to 0
    assert col.track_heading((0.0, 0.0)) == h                                   # a zero-delta (turn) leaves it unchanged
