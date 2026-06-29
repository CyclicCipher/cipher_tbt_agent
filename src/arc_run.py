"""Lean runner: drive a policy against the LIVE ARC-AGI-3 hosted API via the `arc_agi` toolkit.

The development (and submission-shaped) counterpart to `arc_sdk.py`. Where `arc_sdk` targets the official
ARC-AGI-3-Agents framework (which drags the whole LLM-agent stack), this drives `arc_agi.Arcade` directly: it is
light, and it is the SAME shape a Kaggle submission notebook takes (the no-internet sandbox runs the games locally
through the toolkit). A `policy` here uses the exact `(frames, latest_frame) -> (action_name, coords)` /
`is_done(...)` contract as `arc_sdk.TbtPolicy`, so the SAME agent runs offline (the replica), here (hosted public
games), and in the sandbox (local games), unchanged.

Requirements (see the project_arc_agi3_live_env memory): Python >=3.12, `pip install "arc-agi>=0.9.1"`, the
ARC_API_KEY (from `api_key.env`, `.env`, or the environment), and on a TLS-intercepting host (e.g. Norton)
`pip install pip-system-certs`. All third-party imports are lazy so this file imports cleanly without them.

    python src/arc_run.py ls20 40         # play ls20 for up to 40 actions with the TBT agent (the continuous loop)
    python src/arc_run.py ls20 40 random  # the random baseline (pipeline validation)
"""

from __future__ import annotations

import os
import random

from arc_sdk import _primary_grid              # the canonical frame -> primary-grid extractor (duck-typed)


def load_api_key(path: str = "api_key.env") -> str:
    """Load ARC_API_KEY from `path`, then a standard `.env`, then the environment. Never logs the value."""
    from dotenv import load_dotenv
    load_dotenv(path)
    load_dotenv(".env")
    key = os.getenv("ARC_API_KEY", "")
    if not key:
        raise SystemExit(f"ARC_API_KEY not set (looked in {path}, .env, and the environment)")
    return key


def _state_name(state) -> str:
    return getattr(state, "name", None) or str(state)


class RandomPolicy:
    """A pipeline-validation baseline: RESET when not playing, else a random AVAILABLE action (random x,y for a
    coordinate action). Same contract as `arc_sdk.TbtPolicy`, so the runner is policy-agnostic — swap in TbtPolicy
    (wrapping the real agent) once it is ready."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def is_done(self, frames, latest_frame) -> bool:
        return _state_name(latest_frame.state) == "WIN"

    def choose_action(self, frames, latest_frame):
        from arcengine import GameAction
        if _state_name(latest_frame.state) in ("NOT_PLAYED", "GAME_OVER"):
            return "RESET", None
        avail = [GameAction.from_id(a) for a in (latest_frame.available_actions or [])]
        playable = [a for a in avail if a.name != "RESET"] or avail or [GameAction.RESET]
        a = self.rng.choice(playable)
        coords = (self.rng.randint(0, 63), self.rng.randint(0, 63)) if a.is_complex() else None
        return a.name, coords


def play_remote(policy, game_id: str, max_actions: int = 80, tags=None,
                base_url: str | None = None, verbose: bool = True):
    """Open a scorecard, make `game_id` (ONLINE), drive it with `policy` until WIN or `max_actions`, close and
    return the scorecard. The action comes back BY NAME and is mapped to the live `arcengine.GameAction` (with
    `set_data({"x","y"})` for a coordinate action) — the same name-boundary `arc_sdk` crosses for the framework."""
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    key = load_api_key()
    kw = {"arc_base_url": base_url} if base_url else {}
    arc = Arcade(arc_api_key=key, operation_mode=OperationMode.ONLINE, **kw)
    card_id = arc.open_scorecard(tags=tags or ["cipher-tbt", "dev"])
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        raise SystemExit(f"could not make game {game_id} (check the game_id and the API key)")

    frame = env.observation_space                              # the post-reset first frame
    if verbose and frame is not None:
        shape = frame.frame[0].shape if frame.frame else None
        print(f"game {game_id} | state={_state_name(frame.state)} levels={frame.levels_completed} "
              f"grids={len(frame.frame)} shape={shape} | actions={[a.name for a in env.action_space]}")

    actions = 0
    for _ in range(max_actions):
        if frame is None or policy.is_done([], frame):
            break
        name, coords = policy.choose_action([], frame)
        data = {"x": coords[0], "y": coords[1]} if coords is not None else None
        frame = env.step(getattr(GameAction, name), data=data)
        actions += 1

    result = arc.close_scorecard(card_id)
    if verbose:
        final = _state_name(frame.state) if frame is not None else "None (step returned None)"
        levels = frame.levels_completed if frame is not None else "?"
        print(f"done: actions={actions} final_state={final} levels_completed={levels} card_id={card_id}")
    return result


def _our_state(state):
    """Map an arcengine GameState (or anything with a .name) onto OUR GameState (same member names)."""
    from tasks import GameState as OurGS
    try:
        return OurGS[_state_name(state)]
    except KeyError:
        return OurGS.NOT_FINISHED


class _LiveFrame:
    """Present an arc_agi `FrameDataRaw` as the frame the agent expects: `.grid` / `.state` (our GameState) /
    `.level` / `.score` / `.action_counter` / `.is_win()`. score = level = levels_completed (a completion both
    raises the score and advances the level). A None raw (a step error) is reported terminal so the loop stops."""

    def __init__(self, raw, action_counter):
        from tasks import GameState as OurGS
        self.action_counter = action_counter
        if raw is None:
            self.grid, self.level, self.score, self.state = [[0]], 0, 0, OurGS.GAME_OVER
            self.available, self.coord_actions = [], []
        else:
            from arcengine import GameAction
            self.grid = _primary_grid(raw.frame)
            self.score = self.level = int(raw.levels_completed)
            self.state = _our_state(raw.state)
            avail = [GameAction.from_id(a) for a in (raw.available_actions or [])]
            self.available = [a.name for a in avail]                            # action keys
            self.coord_actions = [a.name for a in avail if a.is_complex()]      # the click (ACTION6) -- needs (x, y)

    def is_win(self):
        from tasks import GameState as OurGS
        return self.state == OurGS.WIN


class RemoteEnv:
    """Drive an arc_agi ONLINE game through a reset/step `Environment` protocol: it maps our `GameAction` -> arcengine
    (+ (x,y) for a coordinate action) and `FrameDataRaw` -> a `_LiveFrame` the agent reads. (The lean path the TBT
    agent actually runs on is `play_remote` + `arc_sdk.TbtPolicy`, the choose_action/is_done contract; this env-protocol
    form is kept for an env-style driver.) One game per instance."""

    def __init__(self, arc, game_id, card_id):
        self.arc, self.game_id, self.card_id = arc, game_id, card_id
        self.env = None
        self._actions = 0

    def reset(self):
        from arcengine import GameAction
        self.env = self.arc.make(self.game_id, scorecard_id=self.card_id)
        if self.env is None:
            raise SystemExit(f"could not make game {self.game_id}")
        self._actions = 0
        raw = self.env.step(GameAction.RESET)         # `make` leaves the game "available but not started"; RESET begins it
        return _LiveFrame(raw, self._actions)

    def step(self, action, coords=None):
        from arcengine import GameAction
        name = action if isinstance(action, str) else action.name   # accept a name key (Player) or a GameAction (old)
        data = {"x": coords[0], "y": coords[1]} if coords is not None else None
        raw = self.env.step(getattr(GameAction, name), data=data)
        self._actions += 1
        return _LiveFrame(raw, self._actions)


class _PlayEnv(RemoteEnv):
    """An agent's live env. On top of RemoteEnv (which RESETs to start the game), it SELF-HEALS a GAME_OVER by sending
    RESET to reload the level so the run continues, and hides RESET from `available` so the agent only ever plans real
    game actions; the click (ACTION6) is exposed via `coord_actions` (the agent supplies the (x, y))."""

    @staticmethod
    def _playable(frame):
        # plan real game actions only -- not RESET (the lifecycle command). The click (ACTION6) is exposed via
        # coord_actions; the agent picks a target and supplies its cell as (x, y).
        frame.available = [n for n in frame.available if n != "RESET"]
        frame.coord_actions = [n for n in frame.coord_actions if n in frame.available]
        return frame

    def reset(self):
        return self._playable(super().reset())

    def step(self, action, coords=None):
        from arcengine import GameAction
        from tasks import GameState
        frame = super().step(action, coords)
        if frame.state == GameState.GAME_OVER:                # reload the level and keep going
            frame = _LiveFrame(self.env.step(GameAction.RESET), self._actions)
        return self._playable(frame)


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    which = sys.argv[3] if len(sys.argv) > 3 else "agent"
    if which == "random":
        policy = RandomPolicy(seed=0)                          # the pipeline-validation baseline
    else:
        from arc_sdk import TbtPolicy                          # the TBT agent: Sensor + Agent, the continuous online loop
        policy = TbtPolicy(seed=0)
    play_remote(policy, game, max_actions=n)
