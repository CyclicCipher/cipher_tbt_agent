"""
Deduct — deductive inference grounded in the CTKG.

Given a query and the current KnowledgeGraph, produces a prediction or
conclusion by following morphism edges. The same code handles logical
deduction (A→B, B→C ⊢ A→C), physics prediction (given law morphisms,
predict output), and natural language tasks (given grammar morphisms,
parse or generate).

All reasoning traverses the KnowledgeGraph. No in-memory shadow graphs.
No re-parsing of rule strings at query time.

Contents (to be implemented):
- predict(kg, query) -> Answer: primary inference entry point.
  Dispatches by query type (continuous prediction, symbolic deduction,
  sequence completion) by reading typed morphisms from kg — never by
  string-matching query content.
- forward_chain(kg, premises) -> set[NodeId]: follow IMPLIES edges to
  fixed point.
- evaluate_law(kg, morphism_id, inputs) -> float: apply a FITTED_LAW
  morphism to continuous inputs.
"""
from __future__ import annotations

from experiments.symbolic_ai_v2.ctkg.logic.KnowledgeGraph import *
