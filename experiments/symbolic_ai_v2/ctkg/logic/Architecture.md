# CTKG Architecture

## The Goal

A single symbolic AGI that learns, reasons, and acts in any environment —
physics problems, mathematics, natural language, video games — using one
unified architecture. No environment-specific code. No hardcoded domain
knowledge.

## The Seven Files

```
ctkg/logic/
├── Library.py            — data types and math primitives
├── KnowledgeGraph.py     — the CTKG object (single source of truth)
├── Induct.py             — discover laws from observations
├── Deduct.py             — predict and reason from known laws
├── Abduct.py             — hypothesise explanations for anomalies
├── InputOutputTopology.py — all input/output formats for all environments
├── AgenticLoop.py        — the universal observe→learn→act engine
└── Consolidation.py      — find structure across existing knowledge
```

## The Three Operations

**Induction** (Induct.py): Given observations, discover what law produced them.
Stores the result as a morphism in KnowledgeGraph. Used for the Einstein test
(discover Newtonian mechanics from position/velocity/acceleration observations)
and for natural language (discover grammar rules from a corpus).

**Deduction** (Deduct.py): Given a query and the KnowledgeGraph, follow morphism
edges to produce an answer. Used for physics prediction, logical proof, and
language generation. The only way information exits the model.

**Abduction** (Abduct.py): Given an anomaly (an observation that contradicts the
current KnowledgeGraph), hypothesise the simplest graph extension that explains
it. Used for the Michelson-Morley null result (abduct: light speed is constant),
Mercury precession (abduct: Newtonian gravity needs a correction), and novel
language constructions.

**Consolidation** (Consolidation.py): Periodically scan the KnowledgeGraph for
structure that was implicit but not yet explicit: isomorphisms between theories,
adjunctions between operations, universal constructions. Writes new morphisms
that accelerate future induction, deduction, and abduction.

## Architectural Laws

**Iron Law**: No dispatch on string names of operators or concepts.
All identity is by morphism ID (opaque integer). A system that works with named
operators `{mul, add, sub}` must work identically with anonymous operators
`{⊕, ⊗, ⊖}`. If it does not, it has hardcoded domain knowledge.

**Single Source of Truth**: The KnowledgeGraph is the only place knowledge lives.
No Python dicts, no in-memory shadow graphs, no module-level caches that
duplicate graph content. If it is not in KnowledgeGraph, it does not exist.

**Universality**: The same Induct, Deduct, Abduct, and AgenticLoop code runs
on the Einstein test, the math benchmark, the corpus benchmark, and every
future environment. If you find yourself writing environment-specific logic
in any of these files, the abstraction is wrong.

**No Parallel Systems**: Every algorithm flows *through* the KnowledgeGraph —
reads from it, writes to it, traverses its edges. An algorithm that operates
*alongside* the graph (takes observations as Python objects, produces Python
objects, never touches the graph) is not part of this architecture.

## The Loop

```
for each timestep:
    observation = environment.step(action)
    kg.observe(observation)                      # InputOutputTopology parses
    surprise = Deduct.predict(kg) vs observation # how unexpected?
    if surprise > threshold:
        hypothesis = Abduct.hypothesise(kg, observation)
        Induct.confirm(kg, hypothesis)           # fit + store morphism
    action = Deduct.act(kg)                      # next output
```

## What Does Not Belong Here

- Environment simulators (textworld, Danganronpa, etc.) — these live outside ctkg/
- Corpus generators — these live in corpus/
- Neural networks — not part of this architecture
- Any file larger than ~300 lines — a sign that one concept has become two
