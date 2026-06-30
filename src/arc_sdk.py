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

    Action set: fixed from the first playable frame to the simple game actions PLUS, if the game offers a coordinate
    action (ACTION6 — the click), one CLICK-SLOT per tracked object (a stable size-ordered index). A click-slot is
    resolved to that object's centroid as (x, y), so the agent plans over a stable index space and LEARNS each click's
    effect — which click matters EMERGES (self-free), generalizing to any coordinate action, not just ACTION6."""

    def __init__(self, build_agent=None, seed: int = 0, local: bool = True, integrate: bool = True):
        from tbt.sensor import Sensor                        # lazy: keep arc_sdk importable without torch
        self.sensor = Sensor(local=local, integrate=integrate)   # egocentric + path-integration -- recurrence + navigation
        # The generative forward model (FM1-4) is ALWAYS fed the frame -- it is ONE model with the tabular loop, not a
        # per-game mode: the column learns the field dynamics every step, and its value only FILLS IN where the tabular
        # value is INDIFFERENT (the arbitration in agent._choose), so it drives a dynamics game's novel states yet
        # never disturbs a converged tabular policy. OBSTACLES are handled by the forward model NATIVELY (a blocked move
        # is predicted as no-change -> the planner makes no progress there); the old recognition-based `barriers`
        # faculty is gone (step 1 of folding everything into the one forward model).
        self._build = build_agent
        self.seed = seed
        self.agent = None
        self.simple: List[str] = []                          # the non-coordinate game actions (index 0..len-1)
        self.click: Optional[str] = None                     # the coordinate / click action name (ACTION6), if offered
        self.n_clicks = 0                                    # number of click-slots (one per object, capped)
        self.prev_level = 0
        self._last_a = None                                  # the last agent action index (the efference for path integration)

    def is_done(self, frames, latest_frame) -> bool:
        return _state_name(latest_frame.state) == "WIN"

    def _init_actions(self, latest_frame) -> None:
        """Fix the action space from the first playable frame: the simple game actions, then one CLICK-slot per tracked
        object if a coordinate action is offered (so the click is a real, learnable action — which one matters emerges)."""
        from tbt.agent import Agent                          # lazy
        avail = _action_names(latest_frame)
        coord = [n for n in avail if n in ("ACTION6", "COMPLEX")]
        self.simple = [n for n in avail if n != "RESET" and n not in coord]
        self.click = coord[0] if coord else None
        self.n_clicks = min(len(self.sensor.objects()), 12) if self.click else 0   # one click-slot per object (capped)
        if not self.simple and not self.n_clicks:                                  # degenerate: nothing else playable
            self.simple = [n for n in avail if n != "RESET"] or avail
        build = self._build or (lambda n: Agent(n_actions=n, seed=self.seed))
        self.agent = build(len(self.simple) + self.n_clicks)

    def _ordered_objects(self):
        """Tracked objects' (pose, size) ordered by (size desc, pose) — a STABLE index so 'click-slot k' means the same
        object across frames (static buttons keep their slot)."""
        return sorted(self.sensor.objects().values(), key=lambda v: (-v[1], v[0]))

    def _resolve(self, a) -> Tuple[str, Coords]:
        """Map an action index to (name, coords): a simple action, or a CLICK at the slot-th object's centroid."""
        if a < len(self.simple):
            return self.simple[a], None
        slot = a - len(self.simple)
        objs = self._ordered_objects()
        if slot < len(objs):
            (px, py), _size = objs[slot]
            return self.click, (int(round(px)), int(round(py)))
        return (self.simple[0] if self.simple else self.click), None   # no object for this slot -> a harmless fallback

    def choose_action(self, frames, latest_frame) -> Tuple[str, Coords]:
        st = _state_name(latest_frame.state)
        if st in ("NOT_PLAYED", "GAME_OVER"):               # not started / died -> RESET (GAME_OVER self-heals)
            return "RESET", None
        obs = _to_obs(latest_frame)
        score_delta = max(obs.level - self.prev_level, 0)   # ARC's only reward: a level completion
        if score_delta > 0:                                 # a level boundary: a perceptual + linkage discontinuity
            if self.agent is not None:
                self.agent.complete(score_delta)            # the completing transition -> GOAL; ends the episode
            self.sensor.reset()
            self.prev_level = obs.level
            self._last_a = None
        if self.agent is None:                              # first playable frame: build the agent + wire the L4 encode
            self.sensor.field.perceive(obs.grid)            # peek to size the click-slots (objects())
            self._init_actions(latest_frame)
            self.sensor.encode = self.agent.col.L4.encode   # the sensor emits FEATURE-at-location via the column's L4
            self.sensor.field.reset()                       # undo the peek -- the real read starts the tracker clean
        state, _change = self.sensor.read(obs.grid, action=self._last_a)   # efference = last action (path integration)
        a = self.agent.step(state, 0.0, frame=obs.grid)     # the FRAME -> the generative forward model (FM1-4); obstacles
        #                                                     handled natively (a blocked move -> no-change); reward via complete()
        self._last_a = a
        return self._resolve(a)


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
