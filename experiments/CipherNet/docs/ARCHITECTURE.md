# CipherNet Architecture v4 — Dynamic Columnar Growth

## Core Principles

1. **The unit of creation is always a COLUMN, never a bare node.**
   Every node in the neocortex belongs to a column. Every column has
   the standard layer structure (L4, L2/3, L5, L6). Every column
   connects through the thalamus. No orphan nodes.

2. **Columns for THINGS. Edges for RELATIONSHIPS.**
   New digit '3' → new column. New operator '+' → new column.
   Successor relationship between '2' and '3' → edge between
   existing columns. The displacement rule for addition → the
   addition column's internal model changes, no new column.

3. **Numbers are positions on a graph-native number line.**
   0-9 (or 0-20) as columns connected by successor edges.
   Addition is co-terminating activation waves walking along
   successor edges. No Python arithmetic. The graph IS the calculator.

4. **Large numbers are sequential digit processing.**
   307 is '3', '0', '7' processed one digit at a time by the PFC.
   Same mechanism that chains 4+5+1+9. Each digit within ANS range.

5. **Intelligence is navigation in reference frames (TBT).**
   Arithmetic = navigating the number line. Object recognition =
   navigating the object's surface. Planning = navigating task space.
   Same column mechanism for all.

## The Update Rule

Every node in the graph updates by the same rule each timestep:

```
1. GATE: signal from incoming GATE edges controls retention.
   gate > 0 → accept new input. gate = 0 → hold state.

2. WRITE: sum weighted activations from incoming TEMPORAL edges.

3. RECURRENCE: blend old state with new input.
   activation = retain * old + (1 - retain) * new_input
   Self-loop temporal edges provide additional persistence.

4. INHIBITION: negative SPATIAL edges reduce activation.
```

## Subgraphs (Brain Regions)

### Innate priors (loaded from JSON at initialization)

| Subgraph | Nodes | Role |
|----------|-------|------|
| ANS | 8 | Magnitude comparison (Weber's law) |
| PFC | 21 | 3 WM stripes + inhibitor + monitor + sequencer |
| Basal ganglia | 14 | Go/NoGo gating (D1/D2 MSNs, GPe, STN, GPi, dopamine) |
| Thalamus | 9 | Relay + reticular (routing + competition) |

### Learned structure (created dynamically from experience)

Columns, edges, and number line structure created as the system
encounters new stimuli and discovers relationships.

## Column Factory

When the system encounters something new, it creates a column:

```
create_column(name) produces:
  1. Column subgraph with 4 nodes:
     - L4 (input): receives from thalamus
     - L2/3 (processing): recurrent, learns representations
     - L5 (output): broadcasts to other columns
     - L6 (feedback): prediction error
  2. Standard internal wiring (L4→L23→L5→L6→L4)
  3. A thalamic relay node (automatically added to thalamus subgraph)
  4. Relay wired to column L4 and from column L5
  5. Column registered in the graph
```

Every column automatically gets a thalamic relay — its connection
to the rest of the system. No column exists without a relay.

## Number Line (Graph-Native)

Numbers 0-N as columns connected by successor edges. Addition is
computed by co-terminating activation waves:

```
Addition (a + b):
  Reference wave:    starts at position 0, walks toward b
  Computation wave:  starts at position a, walks in lockstep
  When reference arrives at b → computation position IS the answer

Two layers per position (ref and comp) prevent interference.
Done detector nodes fire when the reference wave arrives.
```

The number line grows dynamically. When the system first encounters
the digit '5', it creates a column for '5' and connects it between
'4' and '6' via successor edges (ordering discovered through ANS).

## Signal Flow for Multi-Step Computation

Example: `4 + 5 + 1 + 9 = 19`

```
Tokens arrive one at a time
         │
         ▼
  ┌── Thalamus ──┐
  │  (routing)    │
  └──┬────────┬───┘
     │        │
     ▼        ▼
  ┌──────┐  ┌──────────────────┐
  │ PFC  │  │  Number Line     │
  │ WM0  │──│  (ref + comp     │
  │ WM1  │──│   wave layers)   │
  │ WM2  │  └──────────────────┘
  └──┬───┘
     │
     ▼
  ┌──────────────┐
  │ Basal Ganglia│
  │ (Go/NoGo)    │
  └──────────────┘

Step 1: Token '4' → PFC loads 4 into WM0 (BG gates stripe 0)
Step 2: Token '+' → PFC sets goal to "addition" in WM2
Step 3: Token '5' → PFC loads 5 into WM1 (BG gates stripe 1)
Step 4: Both operands ready → number line computes 4+5=9
        BG gates result 9 into WM0. WM1 cleared.
Step 5: Token '+' → goal unchanged
Step 6: Token '1' → PFC loads 1 into WM1
Step 7: Compute 9+1=10 → WM0=10, WM1 cleared
Step 8: Token '+' → goal unchanged
Step 9: Token '9' → WM1=9
Step 10: Compute 10+9=19 → WM0=19
Step 11: Token '=' → output WM0 = 19
```

All sequencing happens through graph dynamics. The PFC sequencer
learns the pattern (from examples) of when to trigger computation
and when to gate results.

## Dynamic Graph Growth

### When does new structure get created?

1. **Novel stimulus**: character never seen before → create column
2. **Discovered relationship**: ANS determines ordering → create
   successor edge between existing columns
3. **Discovered operation**: pattern of displacements consistent
   across examples → column's L2/3 learns the displacement model
4. **Number line growth**: new digit encountered → new position
   column + successor edges + ref/comp/done layers

### What triggers creation?

- **PFC error monitor**: high prediction error = surprise = novel input
- **ANS comparison**: ordering information triggers successor edges
- **Repetition**: seeing the same pattern multiple times triggers
  consolidation into permanent structure

### What is NEVER created dynamically?

- Bare nodes outside columns (everything is columnar)
- Subcortical structures (ANS, BG, thalamus are innate priors)
- The update rule itself (same for all nodes, forever)

## Edge Types

| Type | Direction | Purpose |
|------|-----------|---------|
| SPATIAL (0) | Undirected | Metric structure, lateral inhibition |
| TEMPORAL (1) | Directed | Transitions, causality, successor |
| BINDING (2) | Directed | Stimulus → column (grounding) |
| GATE (3) | Directed | Controls retention vs update |

## Tokenization

ALL input is tokenized at the level of individual characters.
The number 307 is three tokens: '3', '0', '7'. No exceptions.

## File Structure

```
experiments/CipherNet/
  docs/
    ARCHITECTURE.md      ← this document
    PLAN.md              — high-level vision
    RULES.md             — non-negotiable constraints
    LESSONS.md           — lessons from experimentation
    TBT_RESEARCH.md      — Thousand Brains Theory research
    PFC_RESEARCH.md      — PFC biology research
    PFC_PLAN.md          — PFC design
    MULTI_STEP_PLAN.md   — multi-step computation plan
    MANIFOLD_REDESIGN.md — from lookup tables to displacement
  priors/
    config.json          — which priors to load, how they connect
    ans.json             — Approximate Number System
    pfc.json             — Prefrontal Cortex
    basal_ganglia.json   — Go/NoGo gating circuit
    thalamus.json        — Relay + reticular
  src/
    graph.py             — core graph + step() update rule
    displacement.py      — isometry group displacement learning
    number_line.py       — graph-native number line + wave arithmetic
    prior_loader.py      — loads JSON priors into graph
  tests/
    test_pfc.py          — PFC capability tests (29 tests)
```
