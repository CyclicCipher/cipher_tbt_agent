"""textworld_adapter.py — Domain adapter for TextWorld / NanoTextEnv / MicroTextWorld.

All TextWorld-specific knowledge (key names, skip lists, navigation detection,
featurization) lives here.  The engine core imports nothing from this file;
the caller imports TEXTWORLD_DOMAIN and passes it to engine constructors.

Usage
-----
    from textworld_adapter import TEXTWORLD_DOMAIN
    from generative_model import GenerativeModel, TransitionModel
    from agent import run_episode

    model = GenerativeModel()
    model.transition = TransitionModel(context_fn=TEXTWORLD_DOMAIN.context_of)

    result = run_episode(
        env, build_state_fn, engine, world,
        domain=TEXTWORLD_DOMAIN,
        verbose=True,
    )

    # For VariationalBelief:
    belief.update_from_obs(state, skip_keys=TEXTWORLD_DOMAIN.skip_keys_belief)

    # For discover_goals:
    goals = discover_goals(
        history, drives,
        skip_keys=TEXTWORLD_DOMAIN.skip_keys_discovery,
    )
"""
from __future__ import annotations

from domain_protocol import DomainConfig


# ---------------------------------------------------------------------------
# TEXTWORLD_DOMAIN
# ---------------------------------------------------------------------------

#: Keys in the TextWorld / NanoTextEnv / MicroTextWorld state dict that are
#: too large or too structured to encode usefully in a categorical belief.
TEXTWORLD_SKIP_KEYS_BELIEF: list = [
    'admissible',       # list of ~5-20 admissible commands (changes every step)
    'description',      # long room description text
    'text',             # full observation text (alias / raw form)
    'info',             # info dict from TextWorld API
    'raw_obs',          # raw observation string before parsing
]

#: Additional keys to skip in discover_goals() (superset of belief skip list).
TEXTWORLD_SKIP_KEYS_DISCOVERY: list = TEXTWORLD_SKIP_KEYS_BELIEF + [
    'unvisited',          # set of unvisited exits (changes frequently)
    'unexplored_exits',   # alias
    'unlock_cmds',        # list of locked-door commands
    'quest_item',         # TextWorld quest item name (task-specific)
    'quest_dest',         # TextWorld quest destination (task-specific)
]

#: TextWorld-style navigation actions: "go <dir>" or bare cardinal directions.
_NAV_DIRECTIONS = frozenset({'north', 'south', 'east', 'west', 'up', 'down'})


def _tw_context(state: dict) -> str:
    """Extract the current room name from a TextWorld state dict."""
    return state.get('location', '')


def _tw_is_nav(action: str) -> bool:
    """Return True if action is a TextWorld navigation command."""
    return action.startswith('go ') or action in _NAV_DIRECTIONS


TEXTWORLD_DOMAIN: DomainConfig = DomainConfig(
    context_fn           = _tw_context,
    skip_keys_belief     = TEXTWORLD_SKIP_KEYS_BELIEF,
    skip_keys_discovery  = TEXTWORLD_SKIP_KEYS_DISCOVERY,
    context_label        = 'location',
)
"""Domain config for TextWorld / NanoTextEnv / MicroTextWorld.

Use this instead of hard-coding 'location' anywhere in the engine core.
Pass ``TEXTWORLD_DOMAIN.context_of`` as ``context_fn`` to TransitionModel.
Pass ``TEXTWORLD_DOMAIN.skip_keys_belief`` to ``VariationalBelief.update_from_obs()``.
Pass ``TEXTWORLD_DOMAIN.skip_keys_discovery`` to ``discover_goals()``.
Pass ``domain=TEXTWORLD_DOMAIN`` to ``run_episode()`` for context-aware traces.
"""


# ---------------------------------------------------------------------------
# TiTS stub (Trials in Tainted Space)
# ---------------------------------------------------------------------------

def _tits_context(state: dict) -> str:
    """Extract the current scene/location from a TiTS state dict."""
    return state.get('scene', state.get('location', ''))


TITS_DOMAIN: DomainConfig = DomainConfig(
    context_fn    = _tits_context,
    # TiTS state dicts have fewer large structured keys; adjust as needed.
    skip_keys_belief     = ['buttons', 'raw_text', 'raw_obs'],
    skip_keys_discovery  = ['buttons', 'raw_text', 'raw_obs'],
    context_label = 'scene',
)
"""Domain config for Trials in Tainted Space (TiTS).

Stub — extend skip_keys as the TiTS state dict evolves.
"""
