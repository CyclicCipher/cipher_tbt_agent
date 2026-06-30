"""Offline benchmark: drive the revived replica games (`src/tasks/games`) through the in-process harness with a
policy -- the SAME `(frames, latest_frame) -> (name, coords)` / `is_done` contract as `arc_run` / `arc_sdk`, so the
identical agent runs here (replicas), on the hosted public games (`arc_run`), and in the Kaggle sandbox.

This is the "homework" suite the current one-model agent is measured on (revived from git history, deleted in the
reset 5b11220): the NAVIGATION / recurring-state games (LockPath, MultiKey, Sokoban) exercise the SR-sweeping planner
grain; the DYNAMICS games (Toggle, Tetris, CollectAll) exercise the forward-model grain. One config, no per-game
flags -- the arbitration picks the grain per state.

    python src/arc_offline.py all 2500      # the whole suite
    python src/arc_offline.py sokoban 2500  # one game
"""

from __future__ import annotations

from tasks import games as _games
from tasks.core import GameAction
from tasks.harness import Environment

GAMES = {g.lower(): getattr(_games, g)
         for g in ("Sokoban", "LockPath", "MultiKey", "Toggle", "Tetris", "CollectAll")}


class _GameFrame:
    """Adapt the harness `FrameData` to the policy frame contract: it needs `.available` as action NAMES (our games'
    `available_actions` are `tasks.core.GameAction`, not the arcengine ids `arc_sdk._action_names` expects). `.level`
    carries the levels-completed score the policy reads for the reward."""

    def __init__(self, fd):
        self.state = fd.state                                   # our GameState (has .name)
        self.frame = fd.frame                                  # list of grids
        self.levels_completed = fd.score                       # completed-level count -> the policy's reward signal
        self.level = fd.level
        self.available = [a.name for a in fd.available_actions]


def play(game_cls, policy, budget: int = 2500):
    """Drive one game through the harness with `policy` until WIN or `budget` actions. Returns
    `(levels_completed, level_count, actions_used, final_state_name)`."""
    game = game_cls()
    env = Environment(game)
    frame = _GameFrame(env.reset())
    n = 0
    while n < budget and frame.state.name != "WIN":
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = _GameFrame(env.step(GameAction[name], tuple(coords) if coords is not None else None))
        n += 1
    return frame.levels_completed, game.level_count, n, frame.state.name


if __name__ == "__main__":
    import sys
    import time

    from arc_sdk import TbtPolicy

    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    games = list(GAMES.values()) if which == "all" else [GAMES[which]]
    for game_cls in games:
        t0 = time.perf_counter()
        lv, nlev, act, st = play(game_cls, TbtPolicy(seed=0, local=False), budget)
        print(f"  {game_cls.__name__:10s}: {lv}/{nlev} levels  actions={act}  {st}  ({time.perf_counter() - t0:.0f}s)")
