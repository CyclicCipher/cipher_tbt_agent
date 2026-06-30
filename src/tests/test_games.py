"""The revived replica games (`src/tasks/games`, restored from the pre-reset git history) drive the CURRENT one-model
agent through the harness -- the offline "homework" suite. A light regression guard: the agent must still WIN the two
FAST games it solves (Toggle, CollectAll), so a change that breaks the games, the `_GameFrame` adapter, or the agent's
capability on them is caught here. The full baseline (incl. the partially-solved navigation games) is run via
`src/arc_offline.py` (kept out of the suite -- LockPath/Tetris are slow)."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_offline import GAMES, play  # noqa: E402
from arc_sdk import TbtPolicy  # noqa: E402


def test_agent_wins_toggle():
    """Toggle (in-place state-change, the forward-model grain) -- the agent reaches WIN, all levels."""
    lv, nlev, _act, st = play(GAMES["toggle"], TbtPolicy(seed=0, local=False), budget=1500)
    assert st == "WIN" and lv == nlev, f"Toggle {lv}/{nlev} {st}"


def test_agent_wins_collectall():
    """CollectAll (gather all targets) -- the agent reaches WIN, all levels, through the revived harness."""
    lv, nlev, _act, st = play(GAMES["collectall"], TbtPolicy(seed=0, local=False), budget=1500)
    assert st == "WIN" and lv == nlev, f"CollectAll {lv}/{nlev} {st}"
