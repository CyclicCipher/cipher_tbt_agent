"""TBT — the thalamo-cortical architecture package (see THALAMO_CORTICAL_ARCHITECTURE.md).

The reusable machinery: the Column (a TEM module — L6 grid, L5 displacement, L4 content, L23 object),
the RewardModel (domain-agnostic critic/planner), and — as they are built — the thalamus, basal ganglia,
eigenoptions, and the agentic wrapper.

Exports are LAZY (PEP 562): importing a pure-stdlib component (RewardModel) does not pull in torch; the
torch-backed components load only when first referenced.
"""

import importlib

# public name -> submodule that defines it
_EXPORTS = {
    "CorticalColumn": ".column",
    "OnlineSR": ".l6_sr",                          # L6 — the online successor-representation location code (the ONE L6 substrate)
    "L5_Displacement": ".l5_displacement",
    "L4_FeatureLocation": ".l4_feature_location",
    "L23_Object": ".l23_object",
    "RewardModel": ".reward",                     # pure stdlib — no torch
    "Thalamus": ".thalamus",                      # inter-column routing / conjunction (torch)
    "BasalGanglia": ".basal_ganglia",             # the gate selector / emergent allocator (pure stdlib)
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
