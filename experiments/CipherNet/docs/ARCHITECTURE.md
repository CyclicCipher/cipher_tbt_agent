# CipherNet Architecture v6 — Predictive Coding + Mamba Accumulation + Dendritic Computation

## North Star

An AI that can play Danganronpa: Trigger Happy Havoc using only vision,
audio, keyboard, and mouse. CipherNet is the brain.

## Core Principles

1. **The unit of creation is always a COLUMN, never a bare node.**
   Every neocortical node belongs to a column with standard layer
   structure. Every column connects through the thalamus.

2. **Columns for THINGS. Edges for RELATIONSHIPS.**
   New digit '3' -> new column. Successor between '2' and '3' -> edge.

3. **Intelligence is navigation in reference frames (TBT).**
   Arithmetic = navigating the number line. Recognition = navigating
   object surfaces. Planning = navigating task space. Same mechanism.

4. **Predictive coding is the learning rule.**
   Bottom-up errors + top-down predictions. Learning minimizes
   prediction error. Credit assignment is automatic via backward
   error propagation (equivalent to backprop at equilibrium).

5. **Mamba-style accumulation is the update rule.**
   new_act = decay * old + input (signal ADDS to state).
   NOT convex blend (retain * old + (1-retain) * input) which kills signal.

6. **Dendritic computation gives non-linear conjunctions.**
   AND/OR/XOR within single nodes via dendritic segments, avoiding
   combinatorial explosion of dedicated conjunction nodes.

7. **No Python orchestration of brain regions.**
   All behavior emerges from graph.step(). If the graph can't do it,
   the graph structure is wrong.

## The Cortical Column (Predictive Coding Microcircuit)

Based on Bastos et al. (2012). Each column = one level of hierarchy.

```
  Top-down predictions from higher area
         |
         v
  +----- L1 -----+  (apical dendrites, backward predictions)
  |               |
  |  L2/3  -------->  FORWARD: prediction ERRORS up
  |  (superficial |    Gamma (30-100 Hz, fast transients)
  |   pyramidal)  |
  |       ^       |
  |       |       |
  |  L4  -+       |  ERROR LAYER: error = input - prediction
  |  (spiny       |  Receives feedforward from lower area
  |   stellate)   |  Receives prediction from own L6
  |       ^       |
  |       |       |
  |  L5  -------->  BACKWARD: PREDICTIONS down
  |  (deep        |    Beta (13-30 Hz, slow/smooth)
  |   pyramidal)  |    Skip L4! (predictions != sensory)
  |       |       |
  |  L6  -+       |  PREDICTION GENERATOR for own L4
  +---------------+
```

### Message types

| Direction | Carries | Origin | Target in next area |
|-----------|---------|--------|---------------------|
| Feedforward | Prediction ERRORS | L2/3 | L4 of higher area |
| Feedback | PREDICTIONS | L5 | L2/3 of lower area (skip L4) |

## The Update Rule (Mamba-Style Accumulation)

The foundation. Every node, every timestep:

```
1. GATE: signal from GATE edges controls decay.
   gate=1.0 -> decay=0 (flush state). gate=0 -> decay=retain.

2. DENDRITIC COMPUTATION on incoming temporal edges:
   - Group by segment (dendritic branch)
   - Within segment: multiplicative (AND — all must be active)
   - Across segments: additive (OR — any branch can fire)
   - Split into SENSORY vs PREDICTION (feedback role)

3. PREDICTION ERROR: error = sensory - prediction (clamped to [-1,1])

4. MAMBA ACCUMULATION: new_act = decay * old_act + sensory
   Input ADDS to decayed state (not blended). Signal persists.

5. INHIBITION: subtract from negative spatial edges.
6. TANH COMPRESSION: smooth saturation (not hard clamp).
```

### Why accumulation, not blending

Old rule: `new = 0.7 * old + 0.3 * input` -> signal per hop: 30%
After 5 hops with w=0.3: 0.3^5 * 0.3 = 0.00007 (dead)

Mamba rule: `new = 0.7 * old + input` -> signal per hop: 100%
After 5 hops with w=0.3: still 0.3^5 = 0.002 (13x better)

The decay only affects OLD state; new input enters at full strength.

## Dendritic Computation

Based on Bhatt et al. (2015) "Synaptic clustering within dendrites"
and Fu et al. (2012) motor learning clustering studies.

### Biology

Real neurons have multiple dendrite branches:
- **Within a branch**: inputs are multiplicative (AND). An NMDA spike
  fires only when enough synapses on the same branch are co-active.
- **Across branches**: branches sum at the soma (OR). Any branch
  can fire the neuron.

This gives Boolean logic without dedicated gates:
- **AND**: edges on SAME segment, all must be active
- **OR**: edges on DIFFERENT segments, any fires the node
- **XOR**: two segments with crossed inhibitory edges

### Implementation: Edge Segments

Each edge has a `segment` integer. Default: unique per edge (additive).
Learning merges segments when co-activation is reliable.

```python
# Within segment: geometric mean (AND)
# If ANY input is 0, segment output = 0
product = 1.0
for signal in segment_signals:
    product *= max(0, signal)
segment_output = product ** (1/len(segment_signals))

# Across segments: sum (OR)
total_sensory = sum(segment_outputs)
```

### Cluster Formation via Co-Activation Tracking

The brain forms dendritic clusters through REPEATED co-activation,
not single events. NMDA spikes + local biochemical spread (Ras, RhoA
GTPases, ~5-20 um) cause synapses that repeatedly fire together to
migrate onto the same dendritic branch.

**Our implementation**: Each edge pair to the same target node
maintains a co-activation counter. When both sources are active during
a learn() call, the counter increments. When the counter exceeds a
threshold (e.g., 10 co-activations), the edges merge into the same
segment.

```
For each target node with error:
  For each pair of active incoming edges (A, B):
    co_count[A, B] += 1
    if co_count[A, B] >= merge_threshold:
      B.segment = A.segment  (merge onto same dendrite)
```

**Key property**: edges that ALWAYS co-fire (like char:3 and char:+
in every addition involving 3) cluster first. Edges that SOMETIMES
co-fire (like char:3 and char:4, only in "3+4") cluster later or not
at all. The co-activation count naturally discovers which inputs are
truly conjunctive vs coincidentally concurrent.

This is NOT hardcoded. The system discovers its own AND-gates from
the statistics of its experience.

## Column Voting (TBT Consensus)

Based on Numenta's Cortical Messaging Protocol (2024).

Columns share HYPOTHESES (not raw features) through lateral
connections. The settle() process IS the voting protocol: columns
iteratively adjust activations based on neighbors' signals until
consensus (minimum total prediction error).

- TRN inhibition (negative SPATIAL edges): competition/disagreement
- Positive backward edges (L5 -> L2/3): agreement/prediction
- Workspace binding (Broca): hierarchical structure consensus

## Workspace Conjunction (Broca's Area)

The computation path for arithmetic:

```
Input columns -> relay -> temporal cortex buffer -> Broca workspace
                                                         |
                                                         v
                                                   Output cortex
```

Broca's workspace holds COMBINATIONS. When ws0="3" and ws1="+4",
the combined activation pattern is a unique sparse distributed
representation that drives the correct output. The workspace IS the
conjunctive representation — no explosion of combination nodes.

Broca's Merge operation: takes two workspace elements, creates a
hierarchical binding {A, B}. The order of Merges determines operator
precedence (learned via reward, not hardcoded).

## Brain Oscillations

| Band | Hz | Role | Layers | Decay |
|------|----|------|--------|-------|
| Gamma | 30-100 | Feedforward errors | L4, L2/3 | 0.3 (fast) |
| Beta | 13-30 | Feedback predictions | L5, L6 | 0.7 (slow) |
| Theta | 4-8 | WM maintenance | PFC | 0.95 (persistent) |

## Context-Dependent Gating (Mamba Selection)

Context changes effective connectivity. BG gate = Mamba's delta.
Context is LEARNED: the BG discovers through reward that certain
token patterns require different gating. Synaptogenesis creates GATE
edges when a gating pattern reduces prediction error.

## Credit Assignment

### Forward: settle (20 steps)
Activations propagate forward with clamped I/O. Each step, prediction
errors are computed locally. Prospective configuration finds the
internal state consistent with both input and desired output.

### Backward: propagate_errors_backward (3 passes)
After settle, sweep error from output clamp back through all outgoing
edges. Each node receives credit proportional to its contribution to
downstream errors. Errors clipped to [-1,1] to prevent cycle explosion.

### Learn: one call
delta_w = lr * target.error * source.activation

At equilibrium (settle converged), this is equivalent to full
backpropagation through the graph (Whittington & Bogacz 2017).

## Subgraphs (Brain Regions)

### Innate priors (subcortical + architectural)

| Subgraph | Nodes | Role |
|----------|-------|------|
| ANS | 8 | Magnitude comparison (subcortical) |
| PFC | 21 | 3 WM stripes + inhibitor + monitor + sequencer |
| Basal ganglia | 22 | Go/NoGo gating, 5 stripes |
| Thalamus | 13 | Relay + reticular + VA for Broca |
| Output cortex | 50 | 49 tokens + inhibitor (winner-take-all) |
| Broca's area | 18 | BA44/BA45 + 4 workspace slots |
| Temporal cortex | 16 | Phonological buffer + lexical |

### Learned (neocortical, from experience)

Input columns, number line, inter-column associations, dendritic
segment structure — all discovered dynamically.

## Edge Types

| Type | Value | Direction | Purpose |
|------|-------|-----------|---------|
| SPATIAL | 0 | Undirected | Metric structure, lateral inhibition |
| TEMPORAL | 1 | Directed | Signal, predictions, transitions |
| BINDING | 2 | Directed | Stimulus -> column grounding |
| GATE | 3 | Directed | Context-dependent routing |

Each edge also has a `segment` field for dendritic branch assignment.

## Tokenization

ALL input is character-level. "307" = '3', '0', '7'. No exceptions.

## File Structure

```
experiments/CipherNet/
  docs/
    ARCHITECTURE.md      <- this document
    RULES.md             -- non-negotiable constraints
    NORTH_STAR_PLAN.md   -- Danganronpa goal
    BROCAS_AREA_DESIGN.md -- Merge operation
    LESSONS.md           -- lessons learned
    TBT_RESEARCH.md      -- Thousand Brains Theory
    PFC_RESEARCH.md      -- PFC biology
    PFC_PLAN.md          -- PFC design
  priors/
    config.json          -- loading + inter-prior connections
    ans.json             -- Approximate Number System
    pfc.json             -- Prefrontal Cortex
    basal_ganglia.json   -- Go/NoGo gating
    thalamus.json        -- Relay + reticular
    output_cortex.json   -- Token output
    broca.json           -- Broca's area
    temporal_cortex.json -- Phonological buffer
  src/
    graph.py             -- core: step() + settle() + learn() + backward
    brain.py             -- Brain wrapper
    prior_loader.py      -- JSON prior loader
    token_io.py          -- character-level I/O
    train.py             -- training teacher
    visualize.py         -- 3D plotly visualization
```
