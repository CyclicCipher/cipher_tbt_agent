"""agent.py — Generic game-agnostic agent episode loop.

Architecture
============
This file contains the general agent loop.  It knows nothing about any specific
game, modality, or goal domain.  It orchestrates the interaction between:

  - A modality/environment  (provides observations, accepts actions)
  - A state builder         (converts raw observations to agent state dict)
  - A planning engine       (DecisionEngine or AIFEngine; selects actions)
  - A world model           (game-specific tracker; holds derived state)

The caller (game adapter file) provides all game-specific logic via the
``build_state_fn``, ``world_model``, and ``engine`` arguments.  This file
never imports or references any game module.

Adapter contract
================
To plug in a new game:

  1. ``GameModality``   — any object with::

         .step(action: str) -> (obs: str, score: float, done: bool, info: dict)
         .get_events()     -> List[dict]   # optional; [] if absent
         .reset()          -> str          # initial observation

     For keyboard+mouse games, ``action`` may be a structured Action object
     (see AIF_ROADMAP.md Phase R7).  The modality translates it to OS events.

  2. ``GameWorldModel`` — any object with::

         .reset()          -> None         # clear at episode start
         .update(obs, info, prev_state, action) -> None  # after each step

  3. ``build_state_fn``  — ``(obs: str, world: GameWorldModel) -> dict``

     Converts raw observation text + world model state into the agent's state
     dict.  This is the ONLY game-specific logic that the episode loop calls.
     All other game knowledge is inside ``world_model.update()``.

  4. ``engine``  — ``DecisionEngine | AIFEngine``

     No changes required.  Both engines have the same ``decide()`` /
     ``feedback()`` interface.

Design principle
================
**THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST BE GENERAL.**

This file enforces that principle structurally: there is no game-specific code
here, not even a comment mentioning a specific game.

See AIF_ROADMAP.md for the full Phase R implementation plan.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Summary of one completed episode.

    Attributes
    ----------
    score       Final score at episode end (higher is better).
    steps       Number of steps taken.
    done        True if the environment signalled episode completion.
    success     Caller-defined success criterion met?  (optional; None = unknown)
    duration_s  Wall-clock time for the episode in seconds.
    trace       Per-step records: list of dicts with keys
                'step', 'location', 'action', 'reason', 'score', 'done'.
    """
    score:      float
    steps:      int
    done:       bool
    success:    Optional[bool]    = None
    duration_s: float             = 0.0
    trace:      List[dict]        = field(default_factory=list)

    def summary(self) -> str:
        status = ('DONE' if self.done else 'TIMEOUT')
        suc    = (f'  success={self.success}' if self.success is not None else '')
        return (
            f'score={self.score:.1f}  steps={self.steps}  '
            f'{status}{suc}  t={self.duration_s:.1f}s'
        )


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------

def run_episode(
    env:             Any,
    build_state_fn:  Callable[[str, Any], dict],
    engine:          Any,
    world_model:     Any,
    rng:             Optional[random.Random]    = None,
    max_steps:       int                        = 200,
    success_fn:      Optional[Callable[[dict], bool]] = None,
    on_step:         Optional[Callable[[int, dict, str, str, float], None]] = None,
    verbose:         bool                       = False,
) -> EpisodeResult:
    """Run one episode of the agent in the environment.

    This is the core agent-environment loop.  It is entirely game-agnostic.
    All game-specific knowledge is encapsulated in the arguments.

    Parameters
    ----------
    env
        Environment / modality.  Must implement:
          .step(action) -> (obs, score, done, info)
          .get_events() -> List[dict]   (optional)
          .reset()      -> str          (initial observation)
        If ``env`` has a ``.connect()`` method, it is called instead of
        ``.reset()`` (for modalities that require a connection step).

    build_state_fn
        ``(obs: str, world: GameWorldModel) -> dict``
        Converts raw observation + world model into the agent's state dict.
        This is the ONLY game-specific function called by run_episode.

    engine
        Planning engine with ``decide(state, rng) -> (action, reason)`` and
        ``feedback(prev_state, action, new_state, events) -> None`` methods.
        Accepts both ``DecisionEngine`` and ``AIFEngine``.

    world_model
        Game-specific world state tracker.  Must implement:
          .reset() -> None
          .update(obs, info, prev_state, action) -> None  (optional)

    rng
        Random number generator.  If None, creates one with default seed.

    max_steps
        Maximum steps before forced episode termination.

    success_fn
        Optional ``(state: dict) -> bool``.  Called on each new_state to
        determine if the episode was successful.  None → success is unknown.

    on_step
        Optional callback ``(step, state, action, reason, score) -> None``
        called after each step.  Use for real-time visualisation or logging.

    verbose
        If True, print step-by-step trace to stdout.

    Returns
    -------
    EpisodeResult
        Summary of the completed episode.
    """
    if rng is None:
        rng = random.Random()

    # --- Reset -----------------------------------------------------------
    t_start = time.perf_counter()

    # Clear engine sub-goals so previous episode's stack doesn't bleed over.
    if hasattr(engine, 'goal_stack') and engine.goal_stack is not None:
        engine.goal_stack.clear()

    # Reset world model.
    if hasattr(world_model, 'reset'):
        world_model.reset()

    # Get initial observation.
    if hasattr(env, 'connect'):
        obs = env.connect()
    elif hasattr(env, 'reset'):
        obs_raw = env.reset()
        obs = obs_raw[0] if isinstance(obs_raw, tuple) else obs_raw
    else:
        raise AttributeError(
            f'env {type(env).__name__} must have .connect() or .reset()'
        )

    score: float = 0.0
    done:  bool  = False
    trace: List[dict] = []

    # Build initial state.
    state = build_state_fn(obs, world_model)

    # --- Main loop -------------------------------------------------------
    for step in range(1, max_steps + 1):

        # Choose action.
        action, reason = engine.decide(state, rng)

        if verbose:
            loc = state.get('location', '?')
            print(f'  step {step:>3}: [{loc}] {action!r}  ({reason})')

        # Execute action in environment.
        result     = env.step(action)
        obs_new    = result[0]
        score_new  = float(result[1]) if len(result) > 1 else score
        done       = bool(result[2])  if len(result) > 2 else False
        info       = result[3]        if len(result) > 3 else {}

        # Get events (inventory changes, etc.) — optional.
        events: List[dict] = []
        if hasattr(env, 'get_events'):
            events = env.get_events() or []

        # Update world model.
        if hasattr(world_model, 'update'):
            world_model.update(obs_new, info, state, action)

        # Build new state.
        new_state = build_state_fn(obs_new, world_model)

        # Record engine feedback.
        engine.feedback(state, action, new_state, events)

        # Per-step callback.
        if on_step is not None:
            on_step(step, new_state, action, reason, score_new)

        # Trace record.
        trace.append({
            'step':     step,
            'location': state.get('location', ''),
            'action':   action,
            'reason':   reason,
            'score':    score_new,
            'done':     done,
        })

        score = score_new
        state = new_state

        if done:
            break

    # --- Build result ----------------------------------------------------
    success: Optional[bool] = None
    if success_fn is not None:
        success = success_fn(state)

    duration = time.perf_counter() - t_start

    return EpisodeResult(
        score      = score,
        steps      = len(trace),
        done       = done,
        success    = success,
        duration_s = duration,
        trace      = trace,
    )


# ---------------------------------------------------------------------------
# run_episodes  (convenience wrapper for multi-episode evaluation)
# ---------------------------------------------------------------------------

def run_episodes(
    n:               int,
    env_factory:     Callable[[], Any],
    build_state_fn:  Callable[[str, Any], dict],
    engine:          Any,
    world_model:     Any,
    rng:             Optional[random.Random]    = None,
    max_steps:       int                        = 200,
    success_fn:      Optional[Callable[[dict], bool]] = None,
    verbose:         bool                       = False,
    episode_sep:     str                        = '',
) -> List[EpisodeResult]:
    """Run ``n`` episodes and return all results.

    Parameters
    ----------
    n              Number of episodes to run.
    env_factory    Callable that returns a fresh environment each episode.
                   If the env is stateful and must be reset, pass a factory.
                   If the same env can be reused, pass ``lambda: env``.
    (remaining parameters identical to run_episode)

    Returns
    -------
    List[EpisodeResult]
        One result per episode.  Use ``results[-1].score`` for final score,
        ``sum(r.success for r in results if r.success)`` for win count.
    """
    if rng is None:
        rng = random.Random()

    results: List[EpisodeResult] = []
    for ep in range(1, n + 1):
        if verbose and episode_sep:
            print(episode_sep)
            print(f'Episode {ep}/{n}')
        env = env_factory()
        result = run_episode(
            env            = env,
            build_state_fn = build_state_fn,
            engine         = engine,
            world_model    = world_model,
            rng            = rng,
            max_steps      = max_steps,
            success_fn     = success_fn,
            verbose        = verbose,
        )
        results.append(result)
        if verbose:
            print(f'  → {result.summary()}')

    return results


# ---------------------------------------------------------------------------
# AgentConfig  (convenience dataclass for parameterising the run)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration bundle for a run_episode / run_episodes call.

    Keeps all parameters in one place so adapter files can construct
    an AgentConfig once and pass it to multiple run functions.

    Example
    -------
    ::

        cfg = AgentConfig(max_steps=300, verbose=True)
        result = run_episode(env, build_state, engine, world, **cfg.episode_kwargs())
    """
    max_steps:   int  = 200
    verbose:     bool = False
    rng_seed:    Optional[int] = None

    def make_rng(self) -> random.Random:
        return random.Random(self.rng_seed)

    def episode_kwargs(self) -> dict:
        return {
            'rng':       self.make_rng(),
            'max_steps': self.max_steps,
            'verbose':   self.verbose,
        }
