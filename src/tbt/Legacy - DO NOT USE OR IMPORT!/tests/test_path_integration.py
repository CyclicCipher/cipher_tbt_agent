"""Path integration (step 7c): the egocentric view recurs but has NO allocentric position, so distinct board
locations ALIAS and the agent gets stuck (the live wall after 7b). Tracking the controllable object's path-integrated
position (efference + correct) separates the aliased views and lets novelty drive COVERAGE + navigation. Gated offline
by a NAVIGATION game whose goal is FAR (pure local sensing can't navigate there): path-integration solves it, pure
egocentric does not. A non-controllable (state-change) scene keeps the gate OFF, so its recurring local view survives."""

from __future__ import annotations

import os
import sys

import pytest

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_sdk import TbtPolicy  # noqa: E402

# End-to-end SOLVING is a P0-deletion CASUALTY, not a regression: NavGame 8/8 rode on the field-CA epistemic bonus + the
# eigenpurpose explorer (both removed by design, ARCHITECTURE.md §8/§10) and on achiever machinery still slated for
# deletion. The capability is RE-EARNED at P4 (the one epistemic-term explorer + SR-geodesic achiever). The P0 explorer is
# intentionally minimal; the MECHANISM tests (operator algebra, SR, recognition, the one value's Bellman) stay green.
_P0_DIP = "P0 deletion dip (ARCHITECTURE.md §10): end-to-end solving re-earned at P4; the P0 explorer is minimal by design."
_HIPPO = ("Re-seated on the HIPPOCAMPUS module (HIPPOCAMPUS.md H1–H3): the SE(2) pose_ops / pose_operator / sense_pose "
          "API this asserts is retired; abelian+non-abelian path integration becomes the HD-ring + gain-field transform.")


class _St:
    def __init__(self, name):
        self.name = name


class NavGame:
    """Duck-typed like the raw frame. A 2x2 mover on a 24x24 board must reach a goal in the OPEN INTERIOR, marked by a
    DISTINCT-COLOUR cell (a salience cue -- not invisible background), far from any edge. Discovery is therefore not a
    blind random-contact search: the cue lets a hypothesis form ('that colour is the goal') and be tested by navigating
    to it -- but the goal marker enters the EGOCENTRIC window only when the mover is near it, so COVERAGE is still needed
    to bring it into view, and allocentric POSITION is still what re-finds it across levels. Moves are 2 cells. Reaching
    the goal completes the level; advances in place; last -> WIN. (A goal at an edge/corner would be solvable by local
    sensing of the out-of-bounds edge alone -- the interior placement is what forces real coverage.)"""

    N = 24
    STEP = 1                                                  # the body moves ONE tile per action (hard rule; ARC-AGI-3 movers step 1 cell)
    GOAL = (12, 12)
    GOAL_COLOR = 3                                            # the goal cell is PERCEPTUALLY DISTINCT (a salience cue), not background
    MOVES = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}

    def __init__(self, levels=8):
        self.n_levels = levels
        self.levels_completed = 0
        self.state = _St("NOT_PLAYED")
        self.actions_taken = 0
        self.mx, self.my = 0, 0

    @property
    def frame(self):
        g = [[0] * self.N for _ in range(self.N)]
        gx, gy = self.GOAL
        g[gy][gx] = self.GOAL_COLOR                          # paint the goal marker first...
        for dx in (0, 1):
            for dy in (0, 1):
                g[self.my + dy][self.mx + dx] = 7            # ...then the 2x2 mover on top (it occludes the marker on arrival)
        return [g]

    @property
    def available(self):
        return ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    def step(self, name, data=None):
        if name == "RESET":
            self.state = _St("NOT_FINISHED"); self.mx, self.my = 0, 0
            return self
        if self.state.name in ("WIN", "NOT_PLAYED"):
            return self
        self.actions_taken += 1
        dx, dy = self.MOVES.get(name, (0, 0))
        self.mx = min(max(self.mx + dx * self.STEP, 0), self.N - 2)
        self.my = min(max(self.my + dy * self.STEP, 0), self.N - 2)
        if abs(self.mx - self.GOAL[0]) <= 1 and abs(self.my - self.GOAL[1]) <= 1:   # reached the interior goal
            self.levels_completed += 1
            if self.levels_completed >= self.n_levels:
                self.state = _St("WIN")
            else:
                self.mx, self.my = 0, 0                       # next level, in place
        return self


def _drive(policy, game, budget):
    frame = game
    for _ in range(budget):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
    return game.levels_completed


@pytest.mark.xfail(reason=_P0_DIP, strict=False)
def test_path_integration_navigates_where_pure_egocentric_stalls():
    """With path integration the agent covers the board and reaches the far goal repeatedly; pure egocentric sensing
    (no position) aliases the uniform interior and reaches it far less often."""
    nav = _drive(TbtPolicy(seed=0, local=True, integrate=True), NavGame(8), budget=3000)
    ego = _drive(TbtPolicy(seed=0, local=True, integrate=False), NavGame(8), budget=3000)
    assert nav >= 6, f"path-integration agent only completed {nav}/8 levels"
    assert nav >= 3 * max(ego, 1) and ego <= 2, f"path-integration {nav} not far above pure-egocentric {ego}"


    # The non-controllable-scene gate is now tested at the COLUMN level (where L5+L6 own path integration): see
    # test_column_online.py::test_track_gate_stays_off_for_a_non_controllable_scene (the P1 unification).


@pytest.mark.xfail(reason=_P0_DIP, strict=False)
def test_achiever_beelines_a_known_goal_after_learning_it():
    """V4 (VECTOR_NAV): once a level is solved the agent REMEMBERS the goal position, and the cost-aware ACHIEVER BEELINES
    there on later levels -- cross-level transfer to near-ORACLE cost (12 two-cell moves from the origin to the (12,12)
    goal), the RHAE efficiency lever, vs the swept value's wandering. The GSG's `reward` generator made live; integrate mode."""
    game = NavGame(8)
    policy = TbtPolicy(seed=0, local=True, integrate=True)
    frame, per_level, last = game, [], 0
    for _ in range(2000):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        before = game.levels_completed
        frame = game.step(name, coords)
        if game.levels_completed > before:
            per_level.append(game.actions_taken - last); last = game.actions_taken
    assert game.levels_completed == 8, per_level
    assert max(per_level[3:]) <= 16, per_level                     # steady-state at/near ORACLE (12) once the goal is learned
    assert per_level[0] >= 2 * max(per_level[3:]), per_level       # the first (discovery) level costs far more than the transfer levels


@pytest.mark.xfail(reason=_P0_DIP, strict=False)
def test_gsg_unification_reward_goal_is_live_and_drives_the_achiever():
    """Phase II (GSG unification, VECTOR_NAV_PLAN): the reward goal is no longer INERT -- it competes in the ONE
    basal-ganglia competition (`propose_goals` + the reward generator) and, when it wins, `self.goal` DISPATCHES the shared
    achiever (`col.achieve`), retiring the separate V4 exploit branch. Driving NavGame, the reward goal is SELECTED on the
    exploit/transfer levels (`self.goal.kind == 'reward'`) and the run still solves 8/8 at oracle -- the achiever executed by
    the competition winner, not a hard-coded branch. (The abelian direction-based achiever; OrientationGame covers SE(2).)"""
    game = NavGame(8)
    policy = TbtPolicy(seed=0, local=True, integrate=True)
    frame, kinds = game, set()
    for _ in range(2000):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        ag = getattr(policy, "agent", None)                            # created lazily on the first real move (after RESET)
        if ag is not None and ag.goal is not None:
            kinds.add(ag.goal.kind)
        frame = game.step(name, coords)
    assert game.levels_completed == 8
    assert "reward" in kinds                                        # the reward goal was SELECTED by the competition (live, not computed-and-discarded)
    assert policy.agent._goal_pos is not None                       # ... carrying the remembered completing target the achiever navigates to


@pytest.mark.xfail(reason=_HIPPO, strict=False)
def test_path_integrate_unifies_abelian_and_nonabelian_by_the_operator():
    """P1 slice 2a: the ONE location belief is path-integrated by the learned `operator(action)` — `location ←
    operator(action)·location` (§2 L6). Abelian (a translation) and non-abelian (an SE(2) rotation+translation) go
    through the SAME `path_integrate`, the translation being the commuting special case; `sense_pose` snap-corrects. The
    operator has NO `heading_dependent` game-type gate — a learned `pose_ops` entry (not a game assumption) selects the
    SE(2) form."""
    import numpy as np
    from tbt.column import CorticalColumn
    from tbt.l5_displacement import pose_operator

    # ABELIAN: per-action translations -> operator() returns the translation (the commuting special case)
    col = CorticalColumn(n_entities=8, seed=0)
    col.L5.observe_move(0, (2.0, 0.0))                         # action 0 = +2x
    col.L5.observe_move(1, (0.0, 3.0))                         # action 1 = +3y
    assert np.allclose(col.operator(0).M, col.L5.operator(0).M)   # no pose op learned -> the translation
    col.track_pose_reset()
    col.path_integrate(0); col.path_integrate(0)
    x, y, th = col.path_integrate(1)                           # (0,0) -> +2x +2x +3y = (4, 3), heading 0
    assert abs(x - 4.0) < 1e-6 and abs(y - 3.0) < 1e-6 and abs(th) < 1e-6
    col.sense_pose(10.0, 7.0, 0.0)                             # CORRECT: snap to an observed pose
    xx, yy, _ = col.path_integrate(0)                         # from the corrected (10,7): +2x -> (12, 7)
    assert abs(xx - 12.0) < 1e-6 and abs(yy - 7.0) < 1e-6

    # NON-ABELIAN: a learned SE(2) op -> operator() returns the pose op; path_integrate composes in the BODY frame
    col2 = CorticalColumn(n_entities=8, seed=0)
    col2.pose_ops = {0: pose_operator(0.0, (1.0, 0.0)),        # FORWARD: +1 in body-x
                     1: pose_operator(np.pi / 2, (0.0, 0.0))}  # TURN_L: +90 deg in place
    assert np.allclose(col2.operator(1).M, col2.pose_ops[1].M)   # the LEARNED pose op is used (not gated on the game)
    col2.track_pose_reset()
    col2.path_integrate(0)                                     # forward -> (1, 0), heading 0
    col2.path_integrate(1)                                     # turn L -> heading 90, still at (1, 0)
    x2, y2, th2 = col2.path_integrate(0)                       # forward in the NEW heading -> +1 in y -> (1, 1)
    assert abs(x2 - 1.0) < 1e-6 and abs(y2 - 1.0) < 1e-6 and abs(th2 - np.pi / 2) < 1e-6


def test_perceive_learns_translating_operators_on_navgame():
    """P1 (MECHANISM, live): driving the agent on NavGame, `perceive` recognizes the mover each step and LEARNS per-action
    operators that TRANSLATE the location (path integration is live on the real game). NB NavGame's mover is a
    rotationally SYMMETRIC 2×2 block, so its heading is ambiguous — the operator is the SDR circulant-shift
    (`col.action_ops`), and the stabilizer-quotient (`L23.best`) keeps the shift a clean translation."""
    game = NavGame(8)
    policy = TbtPolicy(seed=0, local=True, integrate=True)
    frame = game
    for _ in range(300):                                           # enough to move in every direction + learn the operators
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
    col = policy.agent.col
    moved = [a for a in col.L5.eff                               # the actions whose learned efference (L5) displaces the location
             if (col.L5.eff[a][0] ** 2 + col.L5.eff[a][1] ** 2) ** 0.5 > 0.5]
    assert moved, {a: col.L5.eff.get(a) for a in policy.agent.actions}   # perceive learned real (non-trivial) translating displacements
    assert col.controllable()                                      # -> the mover is controllable (its location is informative)
