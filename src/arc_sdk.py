"""Bridge THE agent to the real ARC-AGI-3-Agents SDK — the portable `choose_action`/`is_done` harness.

This is the conduit from the live benchmark (arcprize/ARC-AGI-3-Agents + `arcengine`) to our agent. The SDK
drives the game (its `main()` loops `choose_action` → `take_action`); an agent only implements two methods:

    is_done(frames, latest_frame) -> bool
    choose_action(frames, latest_frame) -> GameAction      # ACTION6 carries (x,y) via action.set_data

The translation is split so it is TESTABLE WITHOUT the SDK or an API key:
  * `TbtPolicy` — SDK-free. Wraps our `tbt.agent.Agent` (perception + planner), consumes a DUCK-TYPED frame
    (anything with `.state`, `.frame`, `.levels_completed`/`.level`), and returns `(action_name, coords)`. It is
    driven identically by the real SDK and by our offline replica `Environment` (whose `FrameData` is modelled on
    the real one), so the same code path is exercised in the test suite — no parallel harness.
  * `make_arc_agent(agent_factory)` — lazily imports the real SDK base + `arcengine.GameAction` and returns a
    class subclassing the SDK `Agent`, delegating to a `TbtPolicy` and mapping the action NAME back to the SDK
    enum (+ `set_data` for coordinates). Call it from a tiny stub inside the ARC-AGI-3-Agents repo.

The agent is INJECTED, so this file is task-agnostic (the thin-shell discipline): it owns only the SDK lifecycle
(NOT_PLAYED/GAME_OVER → RESET, stop on WIN, name↔enum), never a game rule. ACTION6 grounding (what a click DOES)
and online world-model learning belong to the injected agent, learned from the public games — not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tasks import GameState as _OurGameState     # our perception speaks this enum; the SDK's is mapped onto it

Coords = Optional[Tuple[int, int]]


# ── frame translation (duck-typed: works for the real arcengine FrameData AND our replica FrameData) ────────
@dataclass
class _Obs:
    """The minimal observation our `Perception.read` consumes — `.state` (our GameState), `.level`, `.grid`."""
    state: object
    level: int
    grid: list


def _state_name(state) -> str:
    """The lifecycle name, whether `state` is an arcengine GameState, our GameState, or a bare string."""
    return getattr(state, "name", None) or str(state)


def _our_state(state):
    """Map any GameState-like onto OUR GameState (same member names: NOT_PLAYED/NOT_FINISHED/WIN/GAME_OVER)."""
    try:
        return _OurGameState[_state_name(state)]
    except KeyError:
        return _OurGameState.NOT_FINISHED


def _primary_grid(frame_field) -> list:
    """The primary grid as list[list[int]]. `frame` is 1..N grids; take the last (single-grid games return one).
    Distinguishes a 3-D stack-of-grids from a single 2-D grid by depth, and coerces numpy ints to plain ints."""
    arr = frame_field
    if len(arr) and hasattr(arr[0], "__len__") and len(arr[0]) and hasattr(arr[0][0], "__len__"):
        arr = arr[-1]                                       # a list/stack of grids → the primary (last) one
    return [[int(v) for v in row] for row in arr]


def _to_obs(latest_frame) -> _Obs:
    level = getattr(latest_frame, "levels_completed", None)     # the real SDK's name…
    if level is None:
        level = getattr(latest_frame, "level", 0)               # …our replica's name (and a safe default)
    return _Obs(state=_our_state(latest_frame.state), level=level, grid=_primary_grid(latest_frame.frame))


# ── the SDK-free policy (the part the test drives, no arcengine needed) ─────────────────────────────────────
class TbtPolicy:
    """Adapt `tbt.agent.Agent` to the SDK's `choose_action`/`is_done` contract, returning `(action_name, coords)`.

    Lifecycle handled here: a NOT_PLAYED game must be RESET to start; a WIN ends play. GAME_OVER is delegated to
    the agent itself (its `choose_action` fires `planner.on_death()` and emits the reset action), so the planner's
    death handling is exercised — not bypassed."""

    def __init__(self, agent):
        self.agent = agent
        self.agent.reset()

    def is_done(self, frames, latest_frame) -> bool:
        return _state_name(latest_frame.state) == "WIN"

    def choose_action(self, frames, latest_frame) -> Tuple[str, Coords]:
        if _state_name(latest_frame.state) == "NOT_PLAYED":
            return "RESET", None                            # the game has not started; first action must be RESET
        action, coords = self.agent.choose_action(_to_obs(latest_frame))
        return action.name, coords                          # name → mapped back to the SDK enum by the caller


# ── the live SDK agent (lazy import; only usable inside the ARC-AGI-3-Agents repo) ──────────────────────────
def make_arc_agent(agent_factory, max_actions: int = 100_000):
    """Return a class subclassing the real SDK `Agent`, delegating to a `TbtPolicy(agent_factory())`. `agent_factory`
    is a 0-arg callable building a fresh `tbt.agent.Agent` per game (a LEARNING agent, for real games).

    Going live needs Python >=3.12 + `pip install "arc-agi>=0.9.1"` (the ARC-AGI-3 toolkit — it provides BOTH
    `arcengine` and `arc_agi.EnvironmentWrapper`; NB this supersedes the old "arc-agi = static ARC-1/2 only" note).
    Steps inside a clone of arcprize/ARC-AGI-3-Agents:
      1. make our `src/` importable there (PYTHONPATH=…/predictive-coding-agent/src, or `pip install -e`);
      2. add `agents/templates/tbt_agent.py`:
             from arc_sdk import make_arc_agent
             from my_builder import build_agent
             TbtArcAgent = make_arc_agent(build_agent)
      3. import it in `agents/__init__.py` so `Agent.__subclasses__()` discovers it (→ AVAILABLE_AGENTS keyed by
         the lowercased class name);
      4. `export ARC_API_KEY=…` then `uv run main.py --agent=tbtarcagent --game=<game_id>`.
    """
    try:
        from agents.agent import Agent as _SdkAgent         # the SDK base (the ARC-AGI-3-Agents repo)
        from arcengine import GameAction as _SdkGameAction  # the live action enum
    except ImportError as exc:                              # pragma: no cover - exercised only on the live host
        raise ImportError(
            "make_arc_agent must run inside the ARC-AGI-3-Agents repo (needs `agents` + `arcengine`). "
            "Clone arcprize/ARC-AGI-3-Agents and place a stub there that calls make_arc_agent."
        ) from exc

    class TbtArcAgent(_SdkAgent):
        MAX_ACTIONS = max_actions                           # play to a win, not the SDK's tiny default

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.policy = TbtPolicy(agent_factory())

        def is_done(self, frames, latest_frame) -> bool:
            return self.policy.is_done(frames, latest_frame)

        def choose_action(self, frames, latest_frame):
            name, coords = self.policy.choose_action(frames, latest_frame)
            action = getattr(_SdkGameAction, name)
            if coords is not None:                          # ACTION6: the coordinate (click) action
                action.set_data({"x": int(coords[0]), "y": int(coords[1])})
            return action

    return TbtArcAgent
