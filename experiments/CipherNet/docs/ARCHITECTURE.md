# CipherNet Architecture v5 — Predictive Coding on Cortical Columns

## North Star

An AI that can play Danganronpa: Trigger Happy Havoc using only vision,
audio, keyboard, and mouse. CipherNet is the brain.

## Core Principles

1. **The unit of creation is always a COLUMN, never a bare node.**
   Every neocortical node belongs to a column. Every column has the
   standard layer structure. Every column connects through the thalamus.

2. **Columns for THINGS. Edges for RELATIONSHIPS.**
   New digit '3' -> new column. Successor between '2' and '3' -> edge.

3. **Intelligence is navigation in reference frames (TBT).**
   Arithmetic = navigating the number line. Recognition = navigating
   object surfaces. Planning = navigating task space. Same mechanism.

4. **Predictive coding is the learning rule.**
   The brain is simultaneously a recognition model (bottom-up errors)
   and a generative model (top-down predictions). Learning minimizes
   prediction error at every node. Credit assignment is automatic.

5. **No Python orchestration of brain regions.**
   Code must NEVER say "use the thalamus" or "route to Broca's area."
   All behavior emerges from graph.step() propagating activations and
   prediction errors through edges.

## The Cortical Column (Predictive Coding Microcircuit)

Based on Bastos et al. (2012) "Canonical Microcircuits for Predictive
Coding." Each column implements one level of the predictive hierarchy.

```
  Top-down predictions from higher area
         |
         v
  +----- L1 -----+  (apical dendrites, receives backward predictions)
  |               |
  |  L2/3  -------->  FORWARD: sends prediction ERRORS up
  |  (superficial |    to L4 of next higher area
  |   pyramidal)  |    Gamma frequency (fast transients)
  |       ^       |
  |       |       |
  |  L4  -+       |  ERROR LAYER: computes error = input - prediction
  |  (spiny       |  Receives feedforward from lower area
  |   stellate)   |  Receives prediction from own L6
  |       ^       |
  |       |       |
  |  L5  -------->  BACKWARD: sends PREDICTIONS down
  |  (deep        |    to L1/L2/3 of next lower area (skip L4!)
  |   pyramidal)  |    Beta frequency (slow, smooth)
  |       |       |
  |  L6  -+       |  PREDICTION GENERATOR: produces top-down prediction
  |  (feedback)   |  Feeds prediction to own L4 for error computation
  +---------------+
```

### Message types

| Direction | Carries | Origin | Target in next area |
|-----------|---------|--------|---------------------|
| Feedforward (up) | Prediction ERRORS | L2/3 superficial pyramidal | L4 of higher area |
| Feedback (down) | PREDICTIONS | L5 deep pyramidal | L2/3 of lower area (SKIP L4) |

### Why skip L4 on feedback?

L4 is the error-computing layer. It compares bottom-up input against
top-down prediction. Top-down predictions arrive at L2/3 (where they
modulate the error signal) and L1 (apical dendrites), NOT at L4. If
predictions went to L4, they'd be indistinguishable from sensory input.

## Brain Oscillations

Neural oscillations are not an implementation detail — they separate
the forward (error) and backward (prediction) message streams.

| Frequency | Band | Role | Column layers |
|-----------|------|------|---------------|
| 30-100 Hz | Gamma | Feedforward prediction errors | L2/3 (superficial) |
| 13-30 Hz | Beta | Feedback predictions | L5/L6 (deep) |
| 4-8 Hz | Theta | Episodic sequencing, WM maintenance | Hippocampus, PFC |
| 1-4 Hz | Delta | Sleep consolidation | Global |

In graph.step(), oscillations emerge from the different time constants:
- Superficial layers (L2/3): fast update, low self-loop = gamma
- Deep layers (L5/L6): slow update, high self-loop = beta
- PFC WM stripes: very high self-loop (0.95) = theta
- Columns communicate at their natural frequency; the thalamic relay
  acts as a bandpass filter (only passes signals in the right band).

### Implementation

Each node has a `frequency` property (set by its layer):
- L4, L2/3: frequency = 'gamma' (fast response, low persistence)
- L5, L6: frequency = 'beta' (slow response, high persistence)
- PFC nodes: frequency = 'theta' (very persistent)

In step(), the `default_decay` varies by frequency band:
- Gamma nodes: decay = 0.3 (fast, responsive)
- Beta nodes: decay = 0.7 (slow, persistent)
- Theta nodes: decay = 0.95 (very persistent, WM-like)

This creates the spectral asymmetry predicted by Bastos et al.:
fast gamma errors propagate quickly, slow beta predictions persist.

## Context-Dependent Gating (Mamba Selection)

Inspired by Mamba's selective state space model: CONTEXT changes
which edges are effectively active, so the same digit input produces
different internal states depending on the surrounding context.

- Mamba's delta (step size) = BG gate signal (how much new input to accept)
- Mamba's B (input selection) = which edges write to state
- Mamba's C (output selection) = which edges read from state

Context is LEARNED, not hardcoded. The BG discovers through reward
that certain token patterns (like '+' preceding a digit) require
different gating patterns than others. Synaptogenesis creates GATE
edges when a gating pattern reduces prediction error. We never tell
the system "operators are gate sources" — it discovers this.

## Subgraphs (Brain Regions)

### Innate priors (subcortical + architectural, loaded from JSON)

| Subgraph | Nodes | Role |
|----------|-------|------|
| ANS | 8 | Magnitude comparison (Weber's law, subcortical) |
| PFC | 21 | 3 WM stripes + inhibitor + monitor + sequencer |
| Basal ganglia | 22 | Go/NoGo gating, 5 stripes (3 PFC + Broca + temporal cortex) |
| Thalamus | 13 | Relay + reticular (routing + competition, VA for Broca) |
| Output cortex | 50 | 49 output tokens + inhibitor (winner-take-all) |
| Broca's area | 18 | BA44 Merge + BA45 selection + 4 workspace slots |
| Temporal cortex | 16 | Phonological buffer (4 slots) + lexical access |

### Learned structure (neocortical, created from experience)

Input columns, number line structure, inter-column associations —
all created dynamically as the system encounters stimuli and learns.

## The Update Rule (Predictive Coding step())

Every node, every timestep, same rule:

```
1. GATE: signal from incoming GATE edges controls retention.
   gate > 0 -> accept new input. gate = 0 -> hold state.

2. SPLIT incoming temporal edges into:
   - SENSORY: from non-feedback nodes (bottom-up / lateral)
   - PREDICTION: from feedback nodes (role='feedback', top-down)

3. PREDICTION ERROR: error = sensory - prediction
   This is stored on the node for learn() to use.

4. UPDATE: activation tracks sensory input, modulated by gate.
   activation = retain * old + (1 - retain) * sensory

5. INHIBITION: negative spatial edges reduce activation.
```

For clamped nodes (during settle/training):
```
   error = desired - computed  (teaching signal)
   activation = desired        (override)
```

## Learning (Error-Driven Weight Updates)

```
delta_w = learning_rate * target.error * source.activation
```

- Target has positive error (surprised): strengthen edge from active source
- Target has negative error (over-predicted): weaken edge
- Target has zero error (correct): NO weight change (prevents forgetting)

Synaptogenesis: when a node has HIGH error and a potential source is
strongly active, create an edge. The node "needs" more input.

Online learning: weights adjust at EVERY step during settle(), not
just at the end. Error propagates one hop per step; over N steps,
credit assignment reaches N hops deep.

## Settle (Prospective Configuration)

Before committing to weight changes, find the internal state that's
CONSISTENT with both input and desired output:

```
1. Clamp input columns (sensory evidence)
2. Clamp desired output (goal / prediction)
3. Run step() repeatedly with clamps held
4. Internal nodes adjust to minimize total prediction error
5. Learn from this settled state
```

This is prospective configuration: "what SHOULD the internal state be?"
It finds the answer even if the weights aren't yet configured for it.

## Active Inference

Goals are predictions. When you predict "the answer is 7" (clamp out:7)
and the current state doesn't produce 7, prediction error drives:
- **Perception**: adjust internal beliefs (weight updates)
- **Action**: adjust the world (motor output to make reality match prediction)

The same prediction error machinery handles both learning and behavior.

## Edge Types

| Type | Value | Direction | Purpose |
|------|-------|-----------|---------|
| SPATIAL | 0 | Undirected | Metric structure, lateral inhibition |
| TEMPORAL | 1 | Directed | Feedforward signal, predictions, transitions |
| BINDING | 2 | Directed | Stimulus -> column (grounding) |
| GATE | 3 | Directed | Context-dependent routing (Mamba selection) |

## Tokenization

ALL input is character-level. The number 307 is three tokens: '3', '0', '7'.
The system must learn what groups of characters mean.

## File Structure

```
experiments/CipherNet/
  docs/
    ARCHITECTURE.md      <- this document
    RULES.md             -- non-negotiable constraints
    NORTH_STAR_PLAN.md   -- Danganronpa goal
    BROCAS_AREA_DESIGN.md -- Merge operation design
    LESSONS.md           -- lessons from experimentation
    TBT_RESEARCH.md      -- Thousand Brains Theory
    PFC_RESEARCH.md      -- PFC biology
    PFC_PLAN.md          -- PFC design
    PLAN.md              -- variational manifold learning
    MANIFOLD_REDESIGN.md -- from lookup to displacement
    MULTI_STEP_PLAN.md   -- graph-driven multi-step
  priors/
    config.json          -- prior loading + inter-prior connections
    ans.json             -- Approximate Number System (8 nodes)
    pfc.json             -- Prefrontal Cortex (21 nodes)
    basal_ganglia.json   -- Go/NoGo gating (22 nodes, 5 stripes)
    thalamus.json        -- Relay + reticular (13 nodes, incl. VA)
    output_cortex.json   -- Token output (50 nodes)
    broca.json           -- Broca's area (18 nodes)
    temporal_cortex.json -- Wernicke's analog (16 nodes)
  src/
    graph.py             -- core graph: step() + settle() + learn()
    brain.py             -- Brain class (thin wrapper)
    prior_loader.py      -- loads JSON priors into graph
    token_io.py          -- character-level I/O
    train.py             -- training teacher (predictive coding)
    visualize.py         -- 3D plotly visualization
  visualizations/        -- generated HTML visualizations
```
