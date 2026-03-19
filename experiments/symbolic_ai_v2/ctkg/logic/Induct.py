"""
Induct — inductive learning grounded in the CTKG.

Given a stream of observations, discovers the underlying law or structure
and stores it as morphisms in the KnowledgeGraph. The same code handles
physics observation streams (Einstein test) and natural language corpora.

Iron Law: all operator identity is by morphism ID (opaque int). No string
dispatch on operator names anywhere in this module.

Bitter Lesson: structure is discovered from data via graph traversal, not
assumed via hardcoded hypothesis classes.

Contents (to be implemented):
- seed_primitives(kg): register primitive operations as PRIM_OP morphisms
- induce_law(kg, observations) -> MorphismId: discover functional form + fit
  parameters. Uses abductive backward reasoning over PRIM_OP morphisms —
  NOT beam search enumeration.
- learn_from_stream(kg, stream) -> TheoryId: apply induce_law to each
  observation set in a stream; store results in a theory compartment.
"""
from __future__ import annotations

from experiments.symbolic_ai_v2.ctkg.logic.KnowledgeGraph import *
