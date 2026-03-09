"""domain_protocol.py — Domain adapter for the Active Inference engine.

The engine core (TransitionModel, VariationalBelief, DecisionEngine, etc.) is
fully generic.  All domain-specific knowledge lives here: which state key holds
the "context" (room, scene, position…), which state keys to skip during belief
updates, how to featurize a state for the AffordanceModel, etc.

Usage
-----
    from domain_protocol import DomainConfig, NULL_DOMAIN

    # Minimal config (no context key, no skip keys):
    cfg = NULL_DOMAIN

    # TextWorld config (in textworld_adapter.py):
    cfg = DomainConfig(
        context_fn           = lambda s: s.get('location', ''),
        skip_keys_belief     = ['admissible', 'description', 'text',
                                 'info', 'raw_obs'],
        skip_keys_discovery  = ['admissible', 'description', 'text',
                                 'raw_obs', 'info', 'unvisited',
                                 'unexplored_exits', 'unlock_cmds',
                                 'quest_item', 'quest_dest'],
        context_label        = 'location',
    )

    # Pass to TransitionModel:
    model = TransitionModel(context_fn=cfg.context_of)

    # Pass to run_episode:
    run_episode(env, build_state_fn, engine, world, domain=cfg)
"""
from __future__ import annotations

from typing import Callable, List, Optional


class DomainConfig:
    """Adapter bundle: all domain-specific callables and key names.

    Pass one DomainConfig instance to engine constructors; they call its
    methods without inspecting domain semantics.  The default config is
    compatible with any state dict (empty context, no skips).

    Parameters
    ----------
    context_fn
        ``(state: dict) -> str`` — extracts the "context" identifier used
        by TransitionModel to group (context, action) → transition pairs.
        In navigation games this is typically ``state.get('location', '')``.
        In a visual novel it might be ``state.get('scene', '')``.
        Default: always returns '' (all transitions share one context).
    skip_keys_belief
        State dict keys to ignore in ``VariationalBelief.update_from_obs()``.
        High-cardinality or structured keys (lists of admissible commands,
        raw text) should be listed here to avoid polluting the belief state.
        Default: [] — no keys skipped (caller must supply domain list).
    skip_keys_discovery
        State dict keys to ignore in ``discover_goals()``.
        Same rationale as skip_keys_belief; often a superset.
        Default: [] — no keys skipped.
    context_label
        Human-readable label used in trace records and verbose output.
        Set to 'location' for navigation games, 'scene' for visual novels,
        'context' for generic agents.
        Default: 'context'.
    """

    def __init__(
        self,
        context_fn:           Optional[Callable[[dict], str]] = None,
        skip_keys_belief:     Optional[List[str]]             = None,
        skip_keys_discovery:  Optional[List[str]]             = None,
        context_label:        str                             = 'context',
    ) -> None:
        self._context_fn         = context_fn or (lambda s: '')
        self.skip_keys_belief    = list(skip_keys_belief    or [])
        self.skip_keys_discovery = list(skip_keys_discovery or [])
        self.context_label       = context_label

    def context_of(self, state: dict) -> str:
        """Return the context identifier for this state (e.g. current room)."""
        try:
            return str(self._context_fn(state))
        except Exception:
            return ''

    def __repr__(self) -> str:
        return (
            f'DomainConfig(label={self.context_label!r}, '
            f'skip_belief={len(self.skip_keys_belief)}, '
            f'skip_discovery={len(self.skip_keys_discovery)})'
        )


# ---------------------------------------------------------------------------
# Singleton: null config — safe default for any domain
# ---------------------------------------------------------------------------

NULL_DOMAIN: DomainConfig = DomainConfig()
