"""Human-play harness + full step/frame logger for the replica games (LockPath, MultiKey, Sokoban, ...).

Purpose: capture a REAL human trace to ground the agent design (the oracle/human/agent gap analysis). You play the
game move by move; every step records the frame you saw and the action you chose to a JSONL trace, so the decisions
can be reconstructed and compared against the oracle and the agent.

Faithful to the "ignorant at the start" condition: cells are shown as DISTINCT SYMBOLS BY COLOUR, with NO semantic
labels (no "this is a key/door") -- you discover the mechanics exactly as the agent must, from the colour grid + the
score signal alone. The legend maps colour-number -> symbol only.

    python src/play.py lockpath                 # play; trace -> human_traces/lockpath_<ts>.jsonl
    python src/play.py sokoban --out my.jsonl    # choose the trace path
    echo "d d s s ..." | python src/play.py lockpath   # or pipe a move sequence (also replays)

Keys (only those whose action the current frame offers are active):
    w/a/s/d = up/left/down/right (ACTION1/3/2/4)   e = interact (ACTION5)   c = click (ACTION6, then enter: x y)
    r = reset level   u = undo (ACTION7)   q = quit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from tasks import games as _games
from tasks.core import GameAction
from tasks.harness import Environment

GAMES = {g.lower(): getattr(_games, g)
         for g in ("Sokoban", "LockPath", "MultiKey", "Toggle", "Tetris", "CollectAll")}

KEY_TO_ACTION = {"w": "ACTION1", "s": "ACTION2", "a": "ACTION3", "d": "ACTION4",
                 "e": "ACTION5", "c": "ACTION6", "u": "ACTION7", "r": "RESET"}
ACTION_TO_KEY = {v: k for k, v in KEY_TO_ACTION.items()}
# 16 visually distinct symbols by colour index (0 = background). NO semantics -- the human infers, like the agent.
PALETTE = [".", "#", "@", "O", "*", "=", "%", "+", "!", "o", "x", "T", "H", "S", "$", "&"]


def _bbox(grid):
    """Bounding box of the non-background (non-0) cells, with a 1-cell margin (so the board, not the 64x64 canvas)."""
    ys = [y for y, row in enumerate(grid) for v in row if v]
    xs = [x for row in grid for x, v in enumerate(row) if v]
    if not xs:
        return 0, 0, len(grid[0]), len(grid)
    return max(min(xs) - 1, 0), max(min(ys) - 1, 0), min(max(xs) + 2, len(grid[0])), min(max(ys) + 2, len(grid))


def render(grid):
    """ASCII view of the primary grid (cropped to the board), plus a colour->symbol legend of the colours present."""
    x0, y0, x1, y1 = _bbox(grid)
    present = sorted({v for row in grid for v in row})
    lines = ["".join(PALETTE[grid[y][x]] if grid[y][x] < len(PALETTE) else "?" for x in range(x0, x1))
             for y in range(y0, y1)]
    legend = "  ".join(f"{PALETTE[c]}={c}" for c in present if c < len(PALETTE))
    return "\n".join(lines) + f"\n  colours: {legend}"


def _read_key():
    line = sys.stdin.readline()
    if not line:                                            # EOF (piped input exhausted) -> quit
        return "q", None
    parts = line.split()
    if not parts:
        return None, None
    k = parts[0].strip().lower()
    coords = None
    if k == "c":                                            # click: "c x y" inline, else prompt
        if len(parts) >= 3:
            coords = (int(parts[1]), int(parts[2]))
        else:
            xy = sys.stdin.readline().split()
            coords = (int(xy[0]), int(xy[1])) if len(xy) >= 2 else (0, 0)
    return k, coords


def play(game_name, out_path):
    game = GAMES[game_name]()
    env = Environment(game)
    fd = env.reset()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    trace = open(out_path, "w", encoding="utf-8")
    trace.write(json.dumps({"game": game_name, "levels": game.level_count, "started": time.time()}) + "\n")
    n = 0
    print(f"\n=== {game_name} ({game.level_count} levels) -> trace {out_path} ===")
    while True:
        avail = [a.name for a in fd.available_actions]
        keys = " ".join(f"{ACTION_TO_KEY.get(a, '?')}:{a}" for a in avail if a in ACTION_TO_KEY)
        print(f"\nlevel {fd.level}  score {fd.score}  actions {fd.action_counter}  state {fd.state.name}")
        print(render(fd.grid))
        if fd.state.name == "WIN":
            print(">>> WIN (all levels complete)")
            trace.write(json.dumps({"step": n, "event": "WIN", "actions": fd.action_counter}) + "\n")
            break
        print(f"[{keys}  q:quit]  > ", end="", flush=True)
        k, coords = _read_key()
        if k in (None,):
            continue
        if k == "q":
            print("quit")
            break
        name = KEY_TO_ACTION.get(k)
        if name is None or name not in avail:
            print(f"  (key '{k}' -> {name or '?'} not available here; available: {avail})")
            continue
        grid_before = [row[:] for row in fd.grid]
        fd = env.step(GameAction[name], coords)
        n += 1
        trace.write(json.dumps({"step": n, "level": fd.level, "score": fd.score, "state": fd.state.name,
                                "key": k, "action": name, "coords": coords, "grid": grid_before}) + "\n")
        if fd.state.name == "GAME_OVER":
            print(">>> GAME_OVER (press r to reset the level)")
    trace.write(json.dumps({"step": n, "event": "end", "ended": time.time()}) + "\n")
    trace.close()
    print(f"\ntrace written: {out_path}  ({n} steps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Human-play harness + step/frame logger for the replica games.")
    ap.add_argument("game", choices=sorted(GAMES), help="which game to play")
    ap.add_argument("--out", default=None, help="trace JSONL path (default: human_traces/<game>_<ts>.jsonl)")
    args = ap.parse_args()
    out = args.out or os.path.join("human_traces", f"{args.game}_{int(time.time())}.jsonl")
    play(args.game, out)
