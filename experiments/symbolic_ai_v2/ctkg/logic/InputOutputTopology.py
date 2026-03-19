"""Agent topology: edge-type registry for multi-stream observations.

Every token in an observation sequence carries an integer edge type that tells
the MorphismGraph which structural stream the token belongs to.  The three
canonical streams are:

  extero  (0) — exteroceptive / world-state tokens  (SEE_*, PROP_*, EXIT_*)
  intero  (1) — interoceptive / self-state tokens    (AT_*, HUNGER_*, HOLD_*)
  action  (2) — action tokens fed by the AgentLoop between observations

This module provides:
  EtypeRegistry  — maps string names to stable integer codes
  Topology       — wraps a registry; used by environments and the AgentLoop
  agent_topology — factory returning the standard 3-stream topology

The design follows the agent_topology() convention from the old SpectralPredictor
architecture (experiments/symbolic_ai_v2/environments/textworld.py line 68).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# EtypeRegistry
# ---------------------------------------------------------------------------

@dataclass
class EtypeRegistry:
    """Maps edge-type names to stable integer codes.

    Codes are assigned in registration order and never change.  The first
    agent_topology() call registers extero=0, intero=1, action=2.

    Parameters
    ----------
    _map:
        Internal name→code dict.  Populated by code().

    Examples
    --------
    >>> reg = EtypeRegistry()
    >>> reg.code('extero')
    0
    >>> reg.code('intero')
    1
    >>> reg.code('extero')   # idempotent
    0
    """

    _map: dict[str, int] = field(default_factory=dict)

    def code(self, name: str) -> int:
        """Return the integer code for `name`, registering it if new."""
        if name not in self._map:
            self._map[name] = len(self._map)
        return self._map[name]

    def name(self, code: int) -> str:
        """Return the name for integer `code` (raises KeyError if unknown)."""
        for n, c in self._map.items():
            if c == code:
                return n
        raise KeyError(f"Unknown edge-type code {code!r}")

    def all_types(self) -> list[tuple[str, int]]:
        """Return all (name, code) pairs sorted by code."""
        return sorted(self._map.items(), key=lambda kv: kv[1])


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@dataclass
class Topology:
    """Container for an EtypeRegistry.

    Environments expose their topology via ``env.topology`` so that the
    AgentLoop can inspect edge types without knowing the concrete environment.
    """

    registry: EtypeRegistry


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def agent_topology() -> Topology:
    """Return the standard 3-stream agent topology.

    Edge-type assignments (stable across sessions):
      extero → 0
      intero → 1
      action → 2

    Returns
    -------
    Topology with those three streams registered.
    """
    reg = EtypeRegistry()
    reg.code('extero')   # 0
    reg.code('intero')   # 1
    reg.code('action')   # 2
    return Topology(registry=reg)
