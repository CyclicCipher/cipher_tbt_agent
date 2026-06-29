"""The click action wired up (step 7a): the coordinate action (ACTION6) is a real, learnable action -- the policy adds
one CLICK-slot per tracked object (resolved to that object's centroid), and the agent LEARNS which click matters (it is
not told). Gated offline by a mock click-game: clicking the TARGET button completes the level; movement and clicks
elsewhere are no-ops. The agent discovers the right click and transfers it across levels, far better than random."""

from __future__ import annotations

import os
import random
import statistics
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_sdk import TbtPolicy  # noqa: E402


class _St:
    def __init__(self, name):
        self.name = name


class ClickGame:
    """Duck-typed like the raw frame (.state/.frame/.levels_completed/.available). Three static buttons; clicking the
    TARGET (the LARGEST object, so it is click-slot 0) with ACTION6 completes the level. Movement actions and clicks on
    the decoys are no-ops. Advances in place; the last completion -> WIN. NOT_PLAYED until RESET."""

    N = 20

    def __init__(self, levels=20):
        self.n_levels = levels
        self.levels_completed = 0
        self.state = _St("NOT_PLAYED")
        self.actions_taken = 0
        self.target = {(x, y) for x in range(10, 14) for y in range(10, 14)}    # 4x4 (size 16) -> slot 0
        self.decoy1 = {(x, y) for x in range(2, 4) for y in range(2, 4)}        # 2x2
        self.decoy2 = {(16, 4)}                                                 # 1x1

    @property
    def frame(self):
        g = [[0] * self.N for _ in range(self.N)]
        for (x, y) in self.target:
            g[y][x] = 7
        for (x, y) in self.decoy1:
            g[y][x] = 3
        for (x, y) in self.decoy2:
            g[y][x] = 4
        return [g]

    @property
    def available(self):
        return ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"]

    def step(self, name, data=None):
        if name == "RESET":
            self.state = _St("NOT_FINISHED")
            return self
        if self.state.name in ("WIN", "NOT_PLAYED"):
            return self
        self.actions_taken += 1
        if name == "ACTION6" and data is not None and tuple(data) in self.target:  # clicked the target button
            self.levels_completed += 1
            if self.levels_completed >= self.n_levels:
                self.state = _St("WIN")
        return self


def _drive(policy, game, budget):
    frame = game
    marks, prev = [], 0
    for _ in range(budget):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
        if game.levels_completed > len(marks):
            marks.append(game.actions_taken - prev)
            prev = game.actions_taken
    return game.levels_completed, marks


def _drive_random(game, budget):
    game.step("RESET")
    acts = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    centroid = (12, 12)                                     # the target's centroid (a fair random clicker knows the loci)
    decoys = [(2, 2), (16, 4)]
    for _ in range(budget):
        if game.state.name == "WIN":
            break
        if random.random() < 0.5:
            game.step(random.choice(acts))
        else:
            game.step("ACTION6", random.choice([centroid] + decoys))
    return game.levels_completed, game.actions_taken


def test_agent_learns_which_click_completes_and_transfers():
    """The agent, given click-slots it must learn to use, discovers the click that completes the level and -- the model
    persisting -- repeats it near-optimally on every later level, far below random."""
    levels = 20
    completed, marks = _drive(TbtPolicy(seed=0), ClickGame(levels), budget=400)
    assert completed == levels, f"only completed {completed}/{levels} levels"

    late = statistics.median(marks[-10:])                  # after the right click is learned
    rnd = [_drive_random(ClickGame(levels), budget=400) for _ in range(5)]
    rnd_per_level = (sum(u for _c, u in rnd) / len(rnd)) / levels
    assert late <= 2, f"did not converge: late median {late} actions/level (optimum 1)"
    assert late < 0.5 * rnd_per_level, f"agent {late}/lvl not far below random {rnd_per_level:.1f}/lvl"


def test_movement_only_game_unchanged_by_click_wiring():
    """A game with no coordinate action gets NO click-slots -- the action space is exactly the simple actions (the
    movement path is untouched by the click wiring)."""
    policy = TbtPolicy(seed=0)
    game = ClickGame(levels=1)
    game.available_no_click = ["ACTION1", "ACTION2", "ACTION3"]
    # a frame exposing only simple actions
    class _F:
        state = _St("NOT_FINISHED")
        frame = game.frame
        levels_completed = 0
        available = ["ACTION1", "ACTION2", "ACTION3"]
    policy.choose_action([], _F())
    assert policy.click is None and policy.n_clicks == 0
    assert policy.simple == ["ACTION1", "ACTION2", "ACTION3"]
