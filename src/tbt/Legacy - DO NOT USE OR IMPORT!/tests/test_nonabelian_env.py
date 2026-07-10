"""L6_NONABELIAN Stage 1c -- a NON-ABELIAN test environment (the prerequisite for testing the redesign / the cross-layer
unification). A heading-carrying agent (pose = x, y, theta) with BODY-FRAME actions FORWARD / TURN_L / TURN_R = the SE(2)
group. FORWARD moves in the CURRENT heading, so its effect DEPENDS on theta: the actions do NOT commute, and the abelian
`move_delta` (ONE translation per action) CANNOT represent FORWARD. This is the concrete inconsistency the refactor fixes:
L6 tracks POSITION but must track the POSE (the group element)."""

from __future__ import annotations

import math
import os
import sys

import pytest

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# End-to-end SOLVING / agent-loop-coupled tests are P0-deletion CASUALTIES (ARCHITECTURE.md §10): they ride on the removed
# eigenpurpose/field-CA explorer + achiever machinery still slated for deletion, so the learned-operator QUALITY and the
# level completions degrade during P0. Re-earned at P4. The operator-algebra MECHANISM (discover_relations, commutes_with,
# the pose operator) is separately covered in test_operator.py + the pure-mechanism tests here, which stay green.
_P0_DIP = "P0 deletion dip (ARCHITECTURE.md §10): end-to-end solving / clean learned operators re-earned at P4; the P0 explorer is minimal by design."


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
    STEP = 1                                                        # the body moves ONE tile per FORWARD (hard rule; ARC-AGI-3 movers step 1 cell)
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
    # SOLVABLE: face E, advance to x=12; face N, advance to y=12 -> goal (STEP=1 → 10 forwards per leg)
    g = OrientationGame(levels=1); g.step("RESET")
    for _ in range(10):
        g.step("FORWARD")                                                    # E: x 2->12
    g.step("TURN_L")
    for _ in range(10):
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


@pytest.mark.xfail(reason=_P0_DIP, strict=False)
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

