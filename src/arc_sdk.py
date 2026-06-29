"""Bridge THE agent to the real ARC-AGI-3-Agents SDK — the portable `choose_action`/`is_done` harness.

This is the conduit from the live benchmark (arcprize/ARC-AGI-3-Agents + `arcengine`) to our agent. The SDK
drives the game (its `main()` loops `choose_action` → `take_action`); an agent only implements two methods:

    is_done(frames, latest_frame) -> bool
    choose_action(frames, latest_frame) -> GameAction      # ACTION6 carries (x,y) via action.set_data

The translation is split so it is TESTABLE WITHOUT the SDK or an API key:
  * `TbtPolicy` — SDK-free. Wraps the `Sensor` + `tbt.agent.Agent` into ONE CONTINUOUS online loop, consumes a
    DUCK-TYPED frame (anything with `.state`, `.frame`, `.levels_completed`/`.level`, and the available actions),
    and returns `(action_name, coords)`. Each call it reads the frame to a state, learns from the score change
    (levels_completed), and the agent's predict-then-compare step picks the next action — the model PERSISTS across
    levels (a level boundary only resets the per-episode link + the sensor tracker, never the model). It is driven
    identically by the real SDK and by an offline mock game, so the same code path is exercised in the test suite.
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


def _action_names(latest_frame) -> List[str]:
    """The available action NAMES, duck-typed: a frame may expose `.available` (names, our mock/_LiveFrame) or
    `.available_actions` (arcengine action IDs, the real SDK — mapped via `GameAction.from_id`)."""
    avail = getattr(latest_frame, "available", None)
    if avail is not None:
        return [n if isinstance(n, str) else n.name for n in avail]
    ids = getattr(latest_frame, "available_actions", None) or []
    from arcengine import GameAction                         # lazy: only the real-SDK path needs arcengine
    return [GameAction.from_id(a).name for a in ids]


# ── the SDK-free policy (the part the test drives, no arcengine needed) ─────────────────────────────────────
class TbtPolicy:
    """Adapt the `Sensor` + `tbt.agent.Agent` to the SDK's `choose_action`/`is_done` contract, as ONE CONTINUOUS
    online loop, returning `(action_name, coords)`.

    Each `choose_action` observes the RESULT of the previous action (the frame), reads it to a translation-invariant
    state, learns from the score change (levels_completed — the only ARC-AGI-3 reward), and the agent's
    predict-then-compare step picks the next action; the world model + value PERSIST across levels. A level boundary
    is a discontinuity: reset the sensor's tracker + the agent's per-episode link (NOT the model). Lifecycle: a
    NOT_PLAYED/GAME_OVER game is RESET (GAME_OVER self-heals the level and the run continues); a WIN ends play.

    Action set: fixed from the first playable frame to the non-coordinate game actions (the agent plans over a stable
    index space). The coordinate / click action (ACTION6) is a PURPOSEFUL-attention target — step 7 (the saccade /
    GSG); until then it is excluded, and if a game offers only coordinate actions a centroid placeholder is used."""

    def __init__(self, build_agent=None, seed: int = 0):
        from tbt.sensor import Sensor                        # lazy: keep arc_sdk importable without torch
        self.sensor = Sensor()
        self._build = build_agent
        self.seed = seed
        self.agent = None
        self.names: List[str] = []                           # action index -> name (fixed at the first playable frame)
        self.coords: set = set()                             # names that need (x, y)
        self.prev_level = 0

    def is_done(self, frames, latest_frame) -> bool:
        return _state_name(latest_frame.state) == "WIN"

    def _init_actions(self, latest_frame) -> None:
        from tbt.agent import Agent                          # lazy
        avail = _action_names(latest_frame)
        coord = {n for n in avail if n in ("ACTION6", "COMPLEX")}   # the click / coordinate actions
        names = [n for n in avail if n != "RESET" and n not in coord] or [n for n in avail if n != "RESET"] or avail
        self.names, self.coords = names, coord
        build = self._build or (lambda n: Agent(n_actions=n, seed=self.seed))
        self.agent = build(len(self.names))

    def _target(self, obs, name) -> Coords:
        """The (x, y) for a coordinate action — a placeholder until the step-7 attention policy: the centroid of the
        largest tracked object (a salient locus), else None."""
        if name not in self.coords:
            return None
        objs = self.sensor.objects()
        if not objs:
            return (0, 0)
        (px, py), _size = max(objs.values(), key=lambda v: (v[1], v[0]))
        return (int(round(px)), int(round(py)))

    def choose_action(self, frames, latest_frame) -> Tuple[str, Coords]:
        st = _state_name(latest_frame.state)
        if st in ("NOT_PLAYED", "GAME_OVER"):               # not started / died -> RESET (GAME_OVER self-heals)
            return "RESET", None
        obs = _to_obs(latest_frame)
        if self.agent is None:
            self._init_actions(latest_frame)
        score_delta = max(obs.level - self.prev_level, 0)   # ARC's only reward: a level completion
        if score_delta > 0:                                 # a level boundary: a perceptual + linkage discontinuity
            self.agent.complete(score_delta)                # the completing transition -> GOAL; ends the episode
            self.sensor.reset()
            self.prev_level = obs.level
        state, _change = self.sensor.read(obs.grid)
        a = self.agent.step(state, 0.0)                     # learn + choose (predict-then-compare); reward via complete()
        name = self.names[a] if a < len(self.names) else self.names[0]
        return name, self._target(obs, name)


# ── the live SDK agent (lazy import; only usable inside the ARC-AGI-3-Agents repo) ──────────────────────────
def make_arc_agent(build_agent=None, max_actions: int = 100_000):
    """Return a class subclassing the real SDK `Agent`, delegating to a `TbtPolicy(build_agent)`. `build_agent` is an
    optional `(n_actions) -> tbt.agent.Agent` factory (the action count is known only once the game's frame is seen);
    the default builds a fresh learning `tbt.agent.Agent` per game.

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
            self.policy = TbtPolicy(build_agent)

        def is_done(self, frames, latest_frame) -> bool:
            return self.policy.is_done(frames, latest_frame)

        def choose_action(self, frames, latest_frame):
            name, coords = self.policy.choose_action(frames, latest_frame)
            action = getattr(_SdkGameAction, name)
            if coords is not None:                          # ACTION6: the coordinate (click) action
                action.set_data({"x": int(coords[0]), "y": int(coords[1])})
            return action

    return TbtArcAgent
