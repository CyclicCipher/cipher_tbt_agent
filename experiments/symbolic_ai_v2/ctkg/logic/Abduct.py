"""
Abduct — abductive reasoning grounded in the CTKG.

Given an observation that the current KnowledgeGraph cannot explain,
hypothesises the simplest extension to the graph that would explain it.
The same code handles physics anomalies (Michelson-Morley null result)
and language anomalies (novel grammatical construction).

Abduction is the engine of hypothesis generation. Induct confirms and
quantifies. Deduct predicts. Consolidation discovers structure across
hypotheses after the fact.

Contents (to be implemented):
- hypothesise(kg, anomaly) -> Hypothesis: propose a new morphism, latent
  node, or theory that explains the anomaly. Scored by MDL (simplest
  explanation wins).
- retract(kg, morphism_id): remove a morphism when a better explanation
  supersedes it. Checks preservation: prior correct predictions must not
  be invalidated.
- paradigm_shift(kg, anomaly_set) -> NewTheory: when no single-morphism
  extension explains the anomalies, create a new concept node and wire
  existing morphisms to it via PROJECTION/INCLUSION edges.
"""
from __future__ import annotations

from experiments.symbolic_ai_v2.ctkg.logic.KnowledgeGraph import *
