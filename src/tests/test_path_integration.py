"""Path integration (step 7c): the egocentric view recurs but has NO allocentric position, so distinct board
locations ALIAS and the agent gets stuck (the live wall after 7b). Tracking the controllable object's path-integrated
position (efference + correct) separates the aliased views and lets novelty drive COVERAGE + navigation. Gated offline
by a NAVIGATION game whose goal is FAR (pure local sensing can't navigate there): path-integration solves it, pure
egocentric does not. A non-controllable (state-change) scene keeps the gate OFF, so its recurring local view survives."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_sdk import TbtPolicy  # noqa: E402


class _St:
    def __init__(self, name):
        self.name = name


class NavGame:
    """Duck-typed like the raw frame. A 2x2 mover on a 24x24 board must reach a goal in the OPEN INTERIOR (no marker,
    far from any edge) -- so the egocentric window is uniform there and ALIASES with everywhere else; only allocentric
    POSITION can navigate to it. Moves are 2 cells. Reaching the goal completes the level; advances in place; last ->
    WIN. (A goal at an edge/corner would be solvable by local sensing alone -- the window sees the out-of-bounds edge.)"""

    N = 24
    STEP = 2
    GOAL = (12, 12)
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
        for dx in (0, 1):
            for dy in (0, 1):
                g[self.my + dy][self.mx + dx] = 7            # the 2x2 mover (the only object -- the goal is invisible)
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


def test_path_integration_navigates_where_pure_egocentric_stalls():
    """With path integration the agent covers the board and reaches the far goal repeatedly; pure egocentric sensing
    (no position) aliases the uniform interior and reaches it far less often."""
    nav = _drive(TbtPolicy(seed=0, local=True, integrate=True), NavGame(8), budget=3000)
    ego = _drive(TbtPolicy(seed=0, local=True, integrate=False), NavGame(8), budget=3000)
    assert nav >= 6, f"path-integration agent only completed {nav}/8 levels"
    assert nav >= 3 * max(ego, 1) and ego <= 2, f"path-integration {nav} not far above pure-egocentric {ego}"


    # The non-controllable-scene gate is now tested at the COLUMN level (where L5+L6 own path integration): see
    # test_column_online.py::test_track_gate_stays_off_for_a_non_controllable_scene (the P1 unification).


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


def test_online_operator_learns_from_the_live_stream_no_regression():
    """L6_NONABELIAN Stage 1 (live wiring, the PARALLEL learner): driving the REAL agent on NavGame, the per-action
    operators learned ONLINE in the column converge to FAITHFUL operators (spectral radius 1, low grid-code prediction
    error) -- validating online operator learning on the agent's OWN exploration stream (COVERAGE in practice, the reframed
    linchpin) -- while the run still solves 8/8 (the learner is PARALLEL to the additive `move`; zero behaviour change)."""
    import numpy as np
    import torch
    game = NavGame(8)
    policy = TbtPolicy(seed=0, local=True, integrate=True)
    frame = game
    for _ in range(1000):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
    assert game.levels_completed == 8                              # NO REGRESSION: the parallel learner didn't disturb the achiever
    col = policy.agent.col
    assert set(col.action_ops) == {0, 1, 2, 3}                     # one operator learned per nav action, online
    for a in col.action_ops:
        op = col.action_operator(a)
        assert abs(op.spectral_radius() - 1.0) < 1e-6             # the constraint held throughout (orthogonal)
        dx, dy = col.L5.move(a)
        errs = []
        for p in [(3.0, 5.0), (10.0, 10.0), (18.0, 7.0), (2.0, 20.0)]:
            b = col.L6.code_at(torch.tensor(p)).numpy()
            tgt = col.L6.code_at(torch.tensor((p[0] + dx, p[1] + dy))).numpy()
            errs.append(float(np.linalg.norm(op.apply(b) - tgt) / 3.0))
        assert np.mean(errs) < 0.15                               # learned online from the REAL stream -> predicts the next code (coverage OK)
