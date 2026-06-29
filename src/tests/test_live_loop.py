"""The LIVE loop wiring (step 6): the SAME Sensor + Agent, wrapped by `arc_sdk.TbtPolicy`, drives a game through the
EXACT `(frames, latest_frame) -> (action_name, coords)` / `is_done` contract the hosted API and the SDK use. Gated
OFFLINE (no arcengine, no API key) by a duck-typed multi-LEVEL mock game: the policy resets the unstarted game, reads
each frame to a state, learns from the score (levels_completed), persists the model across levels, and drives all
levels to WIN -- far fewer actions than random. This is the bridge to the live `arc_run.play_remote`."""

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


class MockLiveGame:
    """An offline stand-in for the live ARC env, duck-typed like the RAW frame `TbtPolicy` consumes: `.state` (has
    `.name`), `.frame` (a stack of grids), `.levels_completed`, `.available` (action names). A multi-LEVEL reach-the-
    border movement game rendered as FRAMES: a 1-cell mover (colour 7) + a fixed 2x2 anchor (colour 5); reaching the
    border completes the level (levels_completed++) and advances in place, the LAST completion -> WIN. NOT_PLAYED until
    RESET. The frame object IS the game (its properties recompute each read)."""

    ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    MOVES = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}
    N = 12

    def __init__(self, levels=3):
        self.n_levels = levels
        self.levels_completed = 0
        self.state = _St("NOT_PLAYED")
        self.pos = (6, 6)
        self.actions_taken = 0

    @property
    def frame(self):
        g = [[0] * self.N for _ in range(self.N)]
        for dx in (0, 1):
            for dy in (0, 1):
                g[2 + dy][2 + dx] = 5                        # the 2x2 anchor
        g[self.pos[1]][self.pos[0]] = 7                      # the mover
        return [g]                                           # a stack of one grid (single-grid game)

    @property
    def available(self):
        return list(self.ACTIONS)

    def step(self, name, data=None):
        if name == "RESET":
            self.state = _St("NOT_FINISHED")
            self.pos = (6, 6)
            return self
        if self.state.name in ("WIN", "NOT_PLAYED"):
            return self
        self.actions_taken += 1
        dx, dy = self.MOVES.get(name, (0, 0))
        x, y = self.pos
        self.pos = (min(max(x + dx, 0), self.N - 1), min(max(y + dy, 0), self.N - 1))
        if self.pos[0] in (0, self.N - 1) or self.pos[1] in (0, self.N - 1):
            self.levels_completed += 1
            if self.levels_completed >= self.n_levels:
                self.state = _St("WIN")
            else:
                self.pos = (6, 6)                            # the next level, in place
        return self


def _drive_policy(policy, game, budget):
    """Mirror `arc_run.play_remote`'s loop, but arcengine-free: feed the raw frame to the policy, step the game by the
    returned action NAME until WIN or the budget runs out. Returns (levels_completed, total_actions, per_level_actions)."""
    frame = game                                            # NOT_PLAYED; the first choose_action returns RESET
    marks, prev = [], 0
    for _ in range(budget):
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = game.step(name, coords)
        if game.levels_completed > len(marks):              # a level just completed -> record its action cost
            marks.append(game.actions_taken - prev)
            prev = game.actions_taken
    return game.levels_completed, game.actions_taken, marks


def _drive_random(game, budget):
    game.step("RESET")
    for _ in range(budget):
        if game.state.name == "WIN":
            break
        game.step(random.choice(MockLiveGame.ACTIONS))
    return game.levels_completed, game.actions_taken


def test_agent_drives_the_live_contract_and_transfers_across_levels():
    """Through the live policy contract the agent completes ALL levels (reaches WIN with no stall) AND -- because the
    world model + the GOAL transfer across levels -- converges to NEAR-ORACLE per-level cost, far below random. The
    offline proof that the Sensor+Agent continuous loop is correctly wired to the live runner."""
    levels = 60
    completed, used, marks = _drive_policy(TbtPolicy(seed=0), MockLiveGame(levels), budget=6000)
    assert completed == levels, f"only completed {completed}/{levels} levels (a stall)"

    early = statistics.mean(marks[:20])                     # the exploration phase
    late = marks[40:]                                       # after the model + GOAL are learned
    late_median, late_mean = statistics.median(late), statistics.mean(late)
    rnd = [_drive_random(MockLiveGame(levels), budget=6000) for _ in range(5)]
    rnd_per_level = (sum(u for _c, u in rnd) / len(rnd)) / levels
    assert late_median <= 6, f"did not converge to oracle: late median {late_median} actions/level (oracle ~5)"
    assert late_mean < early, f"no transfer: late {late_mean:.1f} not below early {early:.1f}"
    assert late_mean < 0.5 * rnd_per_level, f"agent {late_mean:.1f}/lvl not far below random {rnd_per_level:.1f}/lvl"


def test_policy_resets_an_unstarted_game():
    """Lifecycle: a NOT_PLAYED game's first action is RESET; a WIN reports done -- the contract's two endpoints."""
    policy = TbtPolicy(seed=0)
    game = MockLiveGame(levels=1)
    assert policy.choose_action([], game) == ("RESET", None)
    game.state = _St("WIN")
    assert policy.is_done([], game) is True
