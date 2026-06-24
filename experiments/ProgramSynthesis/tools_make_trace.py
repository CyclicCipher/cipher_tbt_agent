"""Generate a winning oracle trace through LockPath for visualization.

Uses a BFS oracle (it reads game internals to plan an optimal solution) purely to
produce a clean, watchable replay — this is a demo/regression tool, NOT the agent
under study. Writes a compact JSON trace to stdout.
"""

from __future__ import annotations

import json
from collections import deque
from typing import List, Optional

from arc_agi_3 import Environment, GameAction, GameState
from arc_agi_3.games import LockPath

DIRECTIONS = [GameAction.ACTION1, GameAction.ACTION2,
              GameAction.ACTION3, GameAction.ACTION4]


def _capture(g):
    return (g.agent, frozenset(g.blocks), frozenset(g.keys), g.has_key, g._dead)


def _restore(g, d):
    g.agent, g.blocks, g.keys, g.has_key, g._dead = d[0], set(d[1]), set(d[2]), d[3], d[4]


def solve_level(game) -> Optional[List[GameAction]]:
    level = game._level
    start = _capture(game)
    seen, queue, sol = {start}, deque([(start, [])]), None
    while queue:
        state, path = queue.popleft()
        _restore(game, state)
        if game.level_complete():
            sol = path
            break
        for a in DIRECTIONS:
            _restore(game, state)
            game.apply(a, None)
            if game._dead:
                continue
            nxt = _capture(game)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [a]))
    game.load_level(level)
    return sol


def crop(frame):
    g = frame.grid
    h, w = frame_dims(frame)
    return [row[:w] for row in g[:h]]


def frame_dims(frame):
    # board occupies the top-left region; recover it from the game extent
    return env.game.height, env.game.width


def record(frame, action):
    return {
        "action": action,
        "state": frame.state.value,
        "score": frame.score,
        "level": frame.level,
        "actions": frame.action_counter,
        "w": env.game.width,
        "h": env.game.height,
        "grid": crop(frame),
    }


env = Environment(LockPath())
frame = env.reset()
steps = [record(frame, "START")]

for _ in range(env.game.level_count):
    path = solve_level(env.game)
    assert path is not None
    for a in path:
        frame = env.step(a)
        steps.append(record(frame, a.name))
    if frame.state == GameState.WIN:
        break

print(json.dumps({"game_id": "lp01", "steps": steps}))
