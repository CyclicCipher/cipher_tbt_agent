"""
AgenticLoop — universal engine for all tests and environments.

Every interaction between the CTKG and the outside world goes through here,
with zero exceptions. The Einstein test, the math benchmark, the corpus
benchmark, and every game environment all use this same loop.

No environment-specific logic lives here. The loop is defined by:
  observe → surprise? → abduct/induct → deduct → act → repeat

Contents (to be implemented):
- AgenticLoop class: the main loop.
  - observe(input: Observation): parse input via InputOutputTopology,
    write to KnowledgeGraph, check surprise.
  - act() -> Output: call Deduct to produce the next output, format via
    InputOutputTopology.
  - learn(): if surprise exceeds threshold, call Induct or Abduct.
  - run(environment): full loop until environment signals done.
- Observation and Output types are defined in InputOutputTopology.
- All information that enters or exits the model passes through here.
"""
from __future__ import annotations

from experiments.symbolic_ai_v2.ctkg.logic.KnowledgeGraph import *
from experiments.symbolic_ai_v2.ctkg.logic.InputOutputTopology import *
from experiments.symbolic_ai_v2.ctkg.logic.Deduct import *
from experiments.symbolic_ai_v2.ctkg.logic.Induct import *
from experiments.symbolic_ai_v2.ctkg.logic.Abduct import *
