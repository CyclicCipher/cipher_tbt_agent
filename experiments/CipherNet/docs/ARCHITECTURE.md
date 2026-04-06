# CipherNet Architecture v7 — Cortical Columns with Dendritic Computation, Oscillatory Timing, and Subcortical Physics

## North Star

An AI that can play Danganronpa: Trigger Happy Havoc using only vision,
audio, keyboard, and mouse. CipherNet is the brain.

## Core Principles

1. **Cortical columns are the unit of neocortical computation.**
   Every neocortical node belongs to a column with L4/L23/L5/L6.
   Columns connect through the thalamus.

2. **Dendritic segments give non-linear conjunction.**
   AND within a segment, OR across segments. Segments are learned
   through conflict-driven merging. Parallel edges allow the same
   input to participate in multiple AND-gates.

3. **Predictive coding is the learning rule.**
   Bottom-up errors (gamma, fast) + top-down predictions (beta, slow).
   Learning minimizes local prediction error at every node.
   PC inference mode: dmu/dt = +error + downstream_error.

4. **Oscillatory timing separates computation into phases.**
   Gamma (every step): feedforward sweep. Beta (every 3): predictions
   update. Theta (every 8): WM/sequence advance. Alpha (every 10):
   thalamic relay cycle. Not cosmetic — functionally necessary.

5. **Subcortical structures have their own physics.**
   Thalamus, BG, and neuromodulatory systems follow specialized
   update rules reflecting their unique biology. Cortical columns
   follow the standard cortical step. The graph.step() function is
   a physics engine dispatching to the right rule per structure type.

6. **No Python orchestration.**
   No code says "now use Broca" or "route to PFC." All behavior
   emerges from the physics engine running every step. Different
   structures have different physics, but all run automatically.

## The Cortical Column

Based on Bastos et al. (2012) predictive coding microcircuit.

```
  L2/3 (gamma) ──── FORWARD: prediction ERRORS up to next L4
        ^
        |
  L4   (gamma) ──── ERROR = sensory - prediction from L6
        ^
        |
  L5   (beta)  ──── BACKWARD: PREDICTIONS down to next L2/3
        |
  L6   (beta)  ──── PREDICTION for own L4
```

### Interneurons within the column

- **PV (parvalbumin)**: Fast perisomatic inhibition. Fires at the
  END of each gamma cycle, resetting L23 activations. Creates
  discrete computation slots — prevents accumulation/saturation.
- **SST (somatostatin)**: Dendritic branch-specific inhibition.
  Suppresses specific segments while leaving others active.
  Local structural attention within the column.
- **VIP (vasoactive intestinal peptide)**: Inhibits SST
  (disinhibition). Releases branches from SST suppression.
  The intra-column attention spotlight.

## Dendritic Computation

Based on Bhatt et al. (2015) synaptic clustering.

### Segments

Each edge has a `segment` ID. Edges on the same segment compute
multiplicatively (AND). Different segments sum additively (OR).
Default: each edge in its own segment (purely additive).

### Conflict-driven segment merging

Tracks two signals per edge:
- **solo_conflict**: source active AND target has negative error
  (this edge causes false positives when firing alone)
- **paired_success**: both sources active AND target has positive
  error (this pair is useful together)

When solo_conflict AND paired_success both exceed threshold:
merge the pair into one segment. The AND-gate prevents the solo
false positives while preserving the paired successes.

### Parallel edges

One presynaptic neuron can form multiple synapses on different
dendritic branches. When an input participates in multiple
AND-gates, parallel edges are created on different segments.

### Segment state (dendritic calcium)

Each segment maintains:
- **calcium**: accumulates on NMDA spike (segment fires), decays
  at beta rate. Tracks the branch's recent history.
- **threshold**: base * (1 + calcium). Recently-active segments
  require stronger input (metaplasticity/habituation).

### Results

| Level | Task | Result |
|-------|------|--------|
| 1 | (A AND B) OR (C AND D) | 100% by epoch 20 |
| 2 | (A AND B AND C) OR (D AND E) OR (F AND G AND H) | 100% by epoch 30 |
| 3 | (A AND B) OR (A AND C) OR (D AND E) — overlapping | 100% by epoch 30 |
| 4 | 16 inputs, 5 overlapping segments | 85% stable |

## Oscillatory Timing

Each graph.step() = 1 gamma cycle (~30ms).

| Band | Period | What updates | Role |
|------|--------|-------------|------|
| Gamma | Every step | L4, L23, PV reset, dendritic segments | Feedforward sweep, one info slot |
| Beta | Every 3 steps | L5, L6, segment calcium, BG decision | Predictions, branch history |
| Theta | Every 8 steps | PFC WM, temporal buffer, dopamine | Sequence position, WM gating |
| Alpha | Every 10 steps | Thalamic relay, TRN competition | Column selection, attention |

### PV gamma reset

At the end of each gamma cycle, PV interneurons reset L23
activations. This solves the accumulation/saturation problem:
only the current cycle's input survives. No need for complex
lateral inhibition to suppress stale tokens.

### Theta-gamma coupling

The temporal cortex buffer has 4 slots = 4 gamma phases within
one theta cycle. Each theta cycle advances the buffer: slot 0
gets the current token, slots 1-3 hold previous tokens.
Sequential items at different gamma phases within theta.

### Beta as prediction persistence

L5/L6 accumulate gamma-rate errors over a beta period, then
produce smoothed predictions. High beta = system is "locked on"
(prediction matches input). Beta breakdown = surprise/update.

## Subcortical Physics

### Thalamus

The thalamus is NOT a cortical column. It has specialized physics:

- **Relay gating**: GPi controls whether the relay is open (tonic
  mode, faithful transmission) or closed (no signal passes).
- **Burst mode**: When a relay has been hyperpolarized (closed)
  for several alpha cycles and then released, it fires a BURST
  that amplifies weak signals. State-dependent gain control.
- **TRN competition**: The reticular nucleus creates winner-take-all
  between relays at alpha rate. Which columns get driven changes
  every alpha cycle based on competition.
- **Alpha generation**: The thalamo-cortical loop
  (relay -> cortex L4 -> cortex L6 -> TRN -> relay) cycles at
  alpha frequency, creating the fundamental attentional rhythm.

### Basal Ganglia

Specialized action selection circuit:

- **Go/NoGo competition** at beta rate: D1 (Go) inhibits GPi,
  disinhibiting the thalamic relay. D2 (NoGo) excites GPi via
  GPe/STN, keeping the relay closed.
- **Dopamine modulation**: Reward prediction error from VTA/SNc.
  Positive RPE: strengthens D1, weakens D2 (facilitate actions
  that led to reward). Negative RPE: opposite. Operates at
  theta rate (slower than individual decisions).
- **Selection**: Only ONE action (gate) opens at a time via
  lateral inhibition. The BG is the argmax over candidate actions.

### Neuromodulatory Systems

Volume transmission (broadcast, not synaptic):

- **Acetylcholine** (from basal forebrain): THE cortical attention
  signal. Operates at theta rate. Effects:
  - Enhances thalamocortical input (more signal through relay)
  - Lowers dendritic NMDA thresholds (AND-gates fire more easily)
  - Suppresses intracortical spread (less lateral noise)
  - Net: sharpens cortical representation
  - Driven by: PFC error monitor (novelty), task demands

- **Dopamine** (from VTA/SNc): Reward prediction error. Modulates
  BG Go/NoGo balance. Indirect effect on cortical dendrites via
  the BG -> thalamus -> cortex loop. Also: directly modulates
  cortical plasticity rules (what gets strengthened vs weakened).

- **Norepinephrine** (from locus coeruleus): Arousal/salience.
  Global gain knob. High NE = everything more responsive.

## The Update Rule (Physics Engine)

graph.step() dispatches to the right physics per structure type:

```
Every step (gamma):
  cortical_gamma_step:  L4/L23 update, PV reset, segment computation

Every 3 steps (beta):
  cortical_beta_step:   L5/L6 update (predictions)
  bg_step:              Go/NoGo competition
  segment_calcium:      calcium decay, threshold update

Every 8 steps (theta):
  pfc_theta_step:       WM update (if BG gate open)
  buffer_shift:         temporal cortex buffer advance
  neuromod_broadcast:   ACh/NE gain applied to all cortical segments

Every 10 steps (alpha):
  thalamic_alpha_step:  relay cycle, TRN competition
```

### Cortical gamma step

For each L4/L23 node:
1. Split incoming temporal into sensory vs prediction (by source role)
2. Dendritic computation: AND within segments, OR across
3. Prediction error = sensory - prediction (clamped [-1, 1])
4. PC inference: new_act = old + rate * (+error + downstream_error)
   OR feed mode: new_act = decay * old + sensory
5. Inhibition from negative edges
6. PV reset: clear L23 at end of cycle (prevent accumulation)

### Cortical beta step

For each L5/L6 node:
1. Accumulate errors from L23 over the beta period
2. Produce smoothed prediction
3. Send prediction backward to lower area L23 (skip L4)

### Learning

Local predictive coding:
  delta_w = lr * target.error * source.activation

Only cross-subgraph edges learn (structural edges protected).
8-bit quantized weights: excitatory [0, 1.0], inhibitory [-1.0, 0].
256 levels per sign (~0.004 per quantum).

Segment merging: conflict-driven, based on solo_conflict and
paired_success statistics accumulated over many examples.

## Three Timescales of Flexibility

| Timescale | Mechanism | What changes | Speed |
|-----------|-----------|-------------|-------|
| Fastest (gamma) | PV gating | Whether a column outputs this cycle | ~30ms |
| Medium (seconds) | Segment calcium | How easily each AND-gate fires | ~100ms-1s |
| Slow (minutes) | ACh/NE modulation | Global cortical sensitivity | minutes |
| Structural (hours) | Segment merging, weight learning | Which inputs are conjunctive | hours |

This replaces the O(n^2) per-step flexibility of transformer attention
with O(1) per-step structural routing + dynamic gain modulation.

## Subgraphs

### Innate (subcortical + architectural)

| Subgraph | Nodes | Role | Physics |
|----------|-------|------|---------|
| ANS | 8 | Magnitude comparison | Standard |
| PFC | 21 | 3 WM stripes + monitor + sequencer | Theta-rate |
| Basal ganglia | 22 | Go/NoGo, 5 stripes | Beta-rate, selection |
| Thalamus | 13 | Relay + TRN + VA | Alpha-rate, mode switching |
| Output cortex | 50 | 49 tokens + inhibitor | Gamma-rate, WTA |
| Broca | 18 | BA44/BA45 + workspace | Gamma/beta |
| Temporal cortex | 16 | Buffer + lexical | Theta-coupled |

### Learned (neocortical)

Input columns, inter-column edges, dendritic segment structure,
parallel edges — all discovered from experience.

## Edge Types

| Type | Value | Direction | Purpose |
|------|-------|-----------|---------|
| SPATIAL | 0 | Undirected | Metric structure, lateral inhibition |
| TEMPORAL | 1 | Directed | Signal, predictions, inhibition (negative weight) |
| BINDING | 2 | Directed | Stimulus -> column grounding |
| GATE | 3 | Directed | BG -> thalamic relay control |

Each edge has: weight (8-bit quantized), segment ID, metadata.

## Column Voting (TBT Consensus)

Based on Hawkins et al. (2017) and Numenta's CMP (2024).

Columns share HYPOTHESES through a feedback loop:
```
Column A:L23 -> output obj1 -> Column A:L4 (reinforcement)
                     |
Column B:L23 -> output obj1 -> Column B:L4 (reinforcement)
```

When both A and B drive obj1, obj1 feeds back to both columns'
L4. This feedback is MODULATORY — it only boosts columns that
already have feedforward sensory support. The PV gamma reset
prevents inactive columns from being activated by feedback alone.

The settle() process IS the voting protocol. Over multiple gamma
cycles, the feedback loop amplifies the consensus hypothesis and
suppresses alternatives through output competition (inhibitor).

### Hawkins formal mechanism (Equation 5)

A L2/3 cell fires if it has BOTH:
- Feedforward support (sensory match from L4)
- Lateral support >= threshold (other columns agree)

The lateral support comes through basal dendritic segments — an
AND-gate requiring BOTH sensory input AND lateral agreement.

### 4-column disambiguation test

The correct dendritic segments ARE discovered ({A,B} for obj1,
{A,C} for obj2, {B,D} for obj3). But cross-contamination from
the feedback loop creates spurious segments alongside the correct
ones. Segment merging needs to be restricted to feedforward-driven
co-activation only (not feedback-driven). Work in progress.

## Tokenization

ALL input is character-level. "307" = '3', '0', '7'.

## File Structure

```
experiments/CipherNet/
  docs/
    ARCHITECTURE.md      <- this document
    RULES.md             -- constraints
    NORTH_STAR_PLAN.md   -- Danganronpa goal
    BROCAS_AREA_DESIGN.md
    LESSONS.md
    TBT_RESEARCH.md
    PFC_RESEARCH.md
    PFC_PLAN.md
  priors/
    config.json, ans.json, pfc.json, basal_ganglia.json,
    thalamus.json, output_cortex.json, broca.json,
    temporal_cortex.json
  src/
    graph.py             -- physics engine: step() + learn()
    brain.py             -- Brain wrapper
    prior_loader.py      -- JSON loader
    token_io.py          -- character I/O
    train.py             -- training teacher
    visualize.py         -- 3D visualization
```
