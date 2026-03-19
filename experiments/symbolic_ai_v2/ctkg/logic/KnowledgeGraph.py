"""
KnowledgeGraph — the CTKG object.

The single authoritative store for all knowledge the system has acquired.
Every node, edge, morphism, and typed relation lives here.
All other modules read from and write to this object — nothing is stored
anywhere else.

Contents (to be implemented):
- KnowledgeGraph class: add/query/remove nodes, edges, morphisms
- Typed edge kinds: IMPLIES, FITTED_LAW, PRIM_OP, LATENT, PARADIGM_SHIFT, etc.
- Object and morphism identity: opaque integer IDs, never string dispatch
- Persistence: serialise/deserialise the full graph
- Query API: source_morphisms, target_morphisms, morphism_by_id, path_find
"""
from __future__ import annotations

from experiments.symbolic_ai_v2.ctkg.logic.Library import *
