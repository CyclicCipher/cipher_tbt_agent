"""Offline benchmark: drive the revived replica games (`src/tasks/games`) through the in-process harness with a
policy -- the SAME `(frames, latest_frame) -> (name, coords)` / `is_done` contract as `arc_run` / `arc_sdk`, so the
identical agent runs here (replicas), on the hosted public games (`arc_run`), and in the Kaggle sandbox.

This is the "homework" suite the current one-model agent is measured on (revived from git history, deleted in the
reset 5b11220): the NAVIGATION / recurring-state games (LockPath, MultiKey, Sokoban) exercise the SR-sweeping planner
grain; the DYNAMICS games (Toggle, Tetris, CollectAll) exercise the forward-model grain. One config, no per-game
flags -- the arbitration picks the grain per state.

The BFS `oracle` (a teacher that knows the true dynamics) gives the optimal action count per level, so we report
EFFICIENCY relative to it -- the RHAE proxy `(oracle / agent)^2` per level (0 if unsolved) -- the only direct data on
the model's learning + action efficiency we get this run (the hosted API withholds the human baseline).

    python src/arc_offline.py all 2500      # the whole suite, with oracle efficiency
    python src/arc_offline.py sokoban 2500  # one game
"""

from __future__ import annotations

from tasks import games as _games
from tasks.core import GameAction
from tasks.harness import Environment
from tasks.oracle import solve_level

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
    `(levels_completed, level_count, actions_used, final_state_name, marks)` where `marks[i]` = the cumulative action
    count when level `i` was completed (so level `i`'s cost = `marks[i] - marks[i-1]`)."""
    game = game_cls()
    env = Environment(game)
    frame = _GameFrame(env.reset())
    n, marks = 0, []
    while n < budget and frame.state.name != "WIN":
        if policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        frame = _GameFrame(env.step(GameAction[name], tuple(coords) if coords is not None else None))
        n += 1
        if frame.levels_completed > len(marks):                # a level just completed -> record its cumulative cost
            marks.append(n)
    return frame.levels_completed, game.level_count, n, frame.state.name, marks


def oracle_counts(game_cls):
    """The BFS oracle's optimal action count for each level (None if it bailed -- unsolvable / coordinate-action /
    over the combinatorial-state budget). Computed on a fresh game (each level is a self-contained puzzle)."""
    g = game_cls()
    out = []
    for i in range(g.level_count):
        g.load_level(i)
        path = solve_level(g)
        out.append(len(path) if path is not None else None)
    return out


def efficiency(marks, oracle):
    """Per-level RHAE proxy `(oracle / agent_cost)^2` (capped at 1; 0 for an unsolved level), and its mean. The
    agent's cost for level `i` is `marks[i] - marks[i-1]`."""
    per, prev = [], 0
    for i, opt in enumerate(oracle):
        if i < len(marks) and opt:
            cost = marks[i] - prev
            prev = marks[i]
            per.append(min((opt / cost) ** 2, 1.0) if cost else 0.0)
        else:
            per.append(0.0)
    return per, (sum(per) / len(per) if per else 0.0)


if __name__ == "__main__":
    import sys
    import time

    from arc_sdk import TbtPolicy

    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    games = list(GAMES.values()) if which == "all" else [GAMES[which]]
    scores = []
    for game_cls in games:
        oracle = oracle_counts(game_cls)
        t0 = time.perf_counter()
        lv, nlev, act, st, marks = play(game_cls, TbtPolicy(seed=0, local=False), budget)
        per, rhae = efficiency(marks, oracle)
        scores.append(rhae)
        costs = [marks[i] - (marks[i - 1] if i else 0) if i < len(marks) else None for i in range(nlev)]
        detail = "  ".join(f"L{i}:{c if c is not None else '-'}/{oracle[i]}" for i, c in enumerate(costs))
        print(f"  {game_cls.__name__:10s} {lv}/{nlev} {st:13s} RHAE~{rhae:.2f}  ({time.perf_counter()-t0:.0f}s)"
              f"   agent/oracle  {detail}")
    if len(scores) > 1:
        print(f"  {'MEAN':10s} {'':17s} RHAE~{sum(scores)/len(scores):.2f}")
