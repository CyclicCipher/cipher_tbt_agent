"""Phase-2 prior-minimal world-model agent (the replica test).

Goal: solve the ARC-AGI-3 replica using only the bare floor of domain-general priors
({sensory interface, metric, compression}) + the `volume/` MDL machinery, discovering everything the
Phase-1 `agent/wm/` had seeded (agency, objectness, contact, the rule-type vocabulary). See
docs/phase2/VOLUME_CONCEPTS.md.
"""

from .agent import EFEAgent, ForwardPlanAgent, HierarchicalPlanAgent, VolumeAgent
from .perceive import changes, color_displacements, discover_blockers, discover_dynamics
from .world_model import DiscoveredWorldModel

__all__ = [
    "VolumeAgent", "ForwardPlanAgent", "HierarchicalPlanAgent", "EFEAgent", "DiscoveredWorldModel",
    "changes", "color_displacements", "discover_dynamics", "discover_blockers",
]
