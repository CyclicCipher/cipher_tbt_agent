# CipherNet Sequence Modeling — Planning Document

## Goal

Extend CipherNet from a spatial classifier (MNIST, 88.6% @ 100 examples/class) into a
sequence learner capable of next-token prediction, eventually building toward a TBT-LLM:
a language model grounded in Thousand Brains Theory rather than backpropagation.

The path: **spatial TBT** (done) → **temporal TBT** (HTM temporal memory) →
**abstract reference frames** (language axes) → **TBT-LLM**.

---

## Theoretical Foundations

### What HTM Temporal Memory adds (the missing piece)

Current CipherNet has:
- Reference frames (spatial, per column) ✓
- SDR unions per location key ✓
- Multiple minicolumns (multiple object hypotheses per column) ✓
- WTA + multi-fixation evidence accumulation ✓

What it lacks for sequences:
- **Cells within minicolumns** with distal dendrite connections to recent context
- **Bursting** as prediction failure / learning signal
- **Temporal location keys** (what position in time/structure am I at?)
- **Reference frame alignment for language** (what anchors the frame?)

### The HTM → TBT connection

HTM sequence learning = temporal displacement (what follows what in time).
TBT object learning = spatial displacement (what's at this retinal position).

These are the **same algorithm in different reference frame spaces**.
The key open question for language: what ARE the axes of the reference frame?

---

## Open Research Questions

These must be answered (or committed to a working hypothesis) before full
architecture design:

### Q1 — What are the basis vectors of language reference frames?

Candidates (not mutually exclusive):
- **Linear position**: token index, or position relative to sentence start
- **Syntactic displacement**: subject→verb, head→modifier, clause depth
- **Semantic axes**: word2vec-like dimensions, but learned Hebbian not via backprop
- **Discourse axes**: topic continuity, entity tracking, tense/aspect state
- **Temporal axes**: past/present/future, event boundaries

Neuroscience evidence needed: See research below.

### Q2 — How does the brain represent time?

Competing hypotheses:
- **Chain**: one cell fires, triggers the next — purely sequential
- **Map**: a positional space of past/future states, navigable via displacement

Evidence: hippocampal time cells, theta sequences, replay/preplay, successor
representation. See research below.

### Q3 — What is the granularity of the "fixation" for language?

For vision: one fixation = one patch of retinal image.
For language: one "fixation" could be:
- One character
- One token (subword)
- One word
- One phrase (one eye fixation in reading ≈ one word/phrase unit)

The biological reading system actually uses saccades spanning ~7-9 characters
with parafoveal preview. This is closer to token-level than character-level.

### Q4 — Supervised signal: where does it come from?

For MNIST: OutputCortex learns Hebbian associations (IT winner → label).
For language: no labels exist in the same sense.
Options:
- Prediction error (bursting signal) as implicit self-supervised signal
- Contrastive: does the context predict the next token or not?
- This maps to HTM's unsupervised temporal memory learning

---

## Architecture Plan (working draft)

### Phase S1 — Temporal Memory in isolation

**Task**: Next-symbol prediction on a simple deterministic sequence
(e.g., counting: 0 1 2 3 4 5 6 7 8 9 0 1 2 ...; or repeating patterns).

**What to build**:
```
TemporalMiniColumn
  cells: list[Cell]   # K cells per minicolumn (K=10 initially)
  distal_segments: list[DistalSegment]  # per cell, per recent context

DistalSegment
  synapses: dict[CellId, float]  # cell → permanence value
  ACTIVATION_THRESHOLD = 10      # synapses needed to depolarize

TemporalMacroColumn
  minicolumns: list[TemporalMiniColumn]
  recent_active: deque[frozenset[CellId]]  # last N steps, N=5
  burst_count: int
  predict_count: int
```

**Learning rule** (identical to HTM temporal memory):
1. Feedforward input activates columns (spatial pooler / direct SDR)
2. For each active column:
   - If any cell is depolarized (distal segment above threshold): activate only those cells
   - Else: burst — activate all cells, increment burst_count
3. Learning: for cells that just became active, reinforce the distal segment
   that was most activated by the previous step's cells

**Metric**: anomaly score = burst_count / active_columns per timestep.
Should drop toward 0 as the model learns the sequence.

### Phase S2 — Temporal location keys

Replace the spatial `RetinotopicFrame` with a `TemporalFrame` variant that
produces location keys capturing recent context.

Options:
- **Position index**: key = (t mod window_size,) — absolute temporal position
- **Context hash**: key = hash(frozenset of recent active cell IDs) — context fingerprint
- **Displacement**: key = displacement from last event boundary

Start with position index (simplest). Upgrade to context hash if position
keys don't generalize.

### Phase S3 — Character/token sequence prediction

Scale up Phase S1 to real text:
- Input: one character or BPE token at a time
- Encode each token as a sparse SDR (spatial pooler over token embedding)
- Feed SDR into temporal memory
- Predict next token from depolarized cells
- Train on a small corpus (WikiText-2 already available in the repo)

Key question: does a flat (single-layer) temporal memory generalize to language,
or does it need a hierarchy?

### Phase S4 — Hierarchical temporal memory

Multiple temporal memory layers, each operating on a different timescale:
- Layer 1: character/token level (fast, fine)
- Layer 2: word/phrase level (temporal pooling over Layer 1 bursts)
- Layer 3: sentence/clause level

Layer N+1 activates when Layer N reaches a stable state (low burst rate).
This is the biological "temporal receptive window" hierarchy.

### Phase S5 — Reference frame integration

Introduce abstract reference frame axes for language, informed by neuroscience
findings. Candidate: syntactic displacement (head→dependent relations).

This phase requires the most design work and is contingent on Q1 and Q2 answers.

---

## Neuroscience Research Notes

*(To be filled in as research is completed)*

### Brain language representations (Q1)

**Finding: Language uses the same grid-cell/place-cell machinery as spatial navigation,
operating in a geometric semantic space.**

**Grid cells in conceptual space (Constantinescu et al. 2016, Science)**
The entorhinal cortex and vmPFC encode abstract concepts using 6-fold periodic
grid-like representations — the same mechanism as spatial navigation. The brain
literally navigates semantic space using coordinate systems. Concepts have geometry;
relationships are encoded as navigational trajectories (displacements in that space).

**Semantic space is high-dimensional, with broad domain structure**
Huth et al. 2016 (Nature) mapped a continuous semantic atlas across the cortex
using natural speech. The space is HIGH-DIMENSIONAL (hundreds+ effective dimensions)
organised into broad perceptual/motor/social/temporal domains — not six scalar axes.
"Vision," "action," "space," etc. are domains, each of which is itself a
multi-dimensional subspace (vision alone spans color, shape, motion, depth, texture;
action spans force, direction, limb, speed, ...). Related concepts occupy nearby
cortical territory; the gradients are smooth and continuous, not discrete labels.
The grid-cell-like coding navigates this space with arbitrary axes — not anatomically
pre-labelled, just as grid cells don't care which direction is "north."

**Hierarchical temporal receptive windows (Hasson et al.)**
The brain processes language in nested timescales:
- Primary auditory cortex: ~500ms windows (phonemes)
- Superior temporal sulcus: phrase/sentence level (~12s)
- Temporo-parietal junction: discourse/narrative (~36s+)
Crucially, higher layers don't just integrate more tokens — they respond to
*structural boundaries*, not just temporal extent.

**Cortical entrainment to structure (Ding et al. 2016)**
MEG shows simultaneous tracking at word, phrase, and sentence levels via
concurrent neural oscillations at different frequencies — independent of
prosodic cues or word predictability. Syntactic structure is explicitly
represented, not inferred from statistics.

**Broca's area = Merge operation**
BA44 is the neural locus of the Merge operation (the fundamental
structure-building operation in generative linguistics). It shows hierarchical
activation where each successive word reduces activation when words merge into
a phrase. This is a symbolic grouping operation, not linear sequence processing.

**Shared reference frames: language and spatial navigation**
There is direct evidence (Brunec et al.) that cortical networks for reference-frame
processing are shared between language and spatial navigation. The same
egocentric/allocentric transformation machinery applies to both.

**Implications for architecture:**

The brain's language reference frame is NOT linear position. It is a multi-scale
geometric space where:
- **Sub-word**: motor/articulatory feature space (phonological)
- **Word**: semantic space with 6 axes (vision, action, space, time, social, emotion)
- **Phrase**: compositional tree space (Merge operation, Broca's area)
- **Sentence/discourse**: extended propositional/narrative space

The displacement operations in this reference frame correspond to:
- Semantic similarity (move along an axis in meaning space)
- Syntactic dependency (head → dependent relation)
- Discourse coherence (topic continuity, entity tracking)

**This is not next-token prediction. The brain predicts structure, not tokens.**
The N400 is a semantic prediction error; there is a distinct signature for
syntactic prediction errors. Prediction operates over structured representations.

Key papers:
- Constantinescu et al. 2016, Science — grid cells in conceptual space
- Hasson et al. — temporal receptive windows
- Ding et al. 2016 — cortical entrainment to linguistic structure
- Fedorenko et al. — composition as core driver of language network
- Brunec et al. — shared reference frames: language and navigation

### Brain time representation (Q2)

**Finding: The brain uses a map of time, not a chain. The same grid-cell/place-cell
machinery that encodes spatial position encodes temporal position.**

**Time cells (Eichenbaum et al.)**
Hippocampal neurons fire at specific temporal delays — each cell tuned to a
different elapsed time since an event, covering a window of up to tens of seconds.
This is NOT a chain of activations (A fires, triggers B, triggers C). It is a
population code: the full ensemble of time-cell firing rates at any moment encodes
a temporal coordinate, analogous to a place code. The brain's answer to "where are
we in time?" is a point in a high-dimensional firing-rate space.

**Lateral vs. medial entorhinal cortex: time and space as parallel dimensions**
- Lateral EC → temporal coding (time cells, context drift)
- Medial EC → spatial coding (grid cells, spatial context)
Both feed into hippocampus, which binds them. The brain treats time as a dimension
analogous to space — this is architecturally explicit, not metaphorical.

**Successor representation (Dayan 1993; Stachenfeld et al. 2017)**
The hippocampus doesn't encode "where am I now?" — it encodes a discounted
probability distribution over future states: "from here, what states am I likely
to visit, weighted by temporal proximity?"
This is map-based: you hold the entire near-future trajectory simultaneously,
not step-by-step. SR experiments show that V1 neurons fire for successor
locations — the brain literally represents where it will be next.

**Theta sequences: prospective compression**
During active navigation, place cells fire in forward order within a single 8Hz
theta cycle (theta phase precession). This compresses a future trajectory into
~125ms. This is prospective coding: the brain is simulating the near future at
high speed within the theta cycle. It cannot be purely sequential — it requires
a map to compress onto.

**Hippocampal replay and preplay**
During rest and sleep, hippocampus replays trajectories both forward AND backward.
More strikingly: preplay — firing sequences for novel paths never previously
experienced. A purely sequential (chain) system cannot preplay novel sequences.
Preplay requires a generative map model of the state space.

**Temporal Context Model (Howard & Kahana)**
Memory retrieval is explained by a continuously drifting context vector — a point
slowly moving through a high-dimensional space. Recent memories are nearby;
older memories are further away. Retrieval is cued by jumping to a similar context
point, not by playing back a sequence. This is explicit map-based navigation
through a temporal context space.

**Event segmentation: multi-scale map structure (Zacks et al.)**
The brain segments experience into hierarchical events with boundaries at multiple
scales (seconds → minutes → hours). This matches multi-scale grid-cell organisation
(coarse → fine). The brain constructs a multi-resolution temporal map, like a
geospatial map at different zoom levels.

**Positional encoding, not chaining, in working memory**
When remembering a list, the brain does NOT store word₁→word₂→word₃ chains.
It stores (word_identity ⊗ temporal_position) — tensor products of content and
position. Retrieval reconstructs the sequence from these coordinates. This is
exactly what transformers do with positional embeddings. The chain is an output
format, not the storage format.

**Implications for architecture:**

1. **Time is a dimension, not a direction.** Represent temporal context as a
   coordinate in a space, not as a state in a chain. Use displacement operations
   to navigate the space ("3 tokens ago" = a displacement vector).

2. **Separate identity from position.** Store (token_identity, temporal_position)
   as a factorised pair. Current CipherNet does this for space (SDR = identity,
   location_key = position). The same factorization applies to time.

3. **Multi-scale temporal context.** One temporal frame for character-level
   context (fast), another for phrase-level context (slow). The slow frame only
   updates at structural boundaries (clause, sentence). This matches event
   segmentation.

4. **Successor representation for prediction.** Rather than predicting only the
   next token, maintain a distribution over near-future tokens weighted by
   temporal distance. This enables planning over multiple steps, not just
   next-step prediction.

Key papers:
- Eichenbaum et al. — time cells in hippocampus
- Stachenfeld et al. 2017 — hippocampal successor representation
- Dayan 1993 — successor representation
- Howard & Kahana — temporal context model
- Zacks et al. — event segmentation theory
- Drieu & Zugaro — theta sequences / preplay

### Cortical column architecture (Q — multi-column coordination)

**Finding: The macrocolumn, not the minicolumn, is the functional unit. Coordination
across columns is distributed — no central binder.**

**Minicolumn (~80–100 neurons, 20–100 μm):**
A narrow feature detector tuned to one aspect of the input (one orientation in V1,
one direction of whisker deflection in S1). It is NOT a complete unit — it sees
a sliver of the input space. Mountcastle's "universal computation" claim has not held
up: minicolumn function is entirely domain-specific.

**Macrocolumn / hypercolumn (~300–500 μm, many minicolumns):**
Achieves completeness for ONE location in sensory space. The V1 hypercolumn covers
ALL orientations + both eyes + multiple spatial frequencies for one retinal point.
The barrel column covers all aspects of one whisker's stimulation. The macrocolumn
integrates via lateral inhibition (competing hypotheses suppress each other) and
vertical integration (layer 4 → 2/3 → 5). Result: a winner-takes-most ensemble
that reaches local consensus. This is exactly CipherNet's WTA per macrocolumn.

**Multi-column coordination (binding problem):**
Three mechanisms — NO central coordinator:
1. Long-range horizontal connections (layer 2/3, up to 8mm): link columns with
   similar feature preferences. Enables contour integration. Feature-based grouping,
   not identity-based binding.
2. Top-down feedback gating (corticothalamic + corticocortical): higher areas
   selectively amplify or suppress which lower-area columns are active. Dynamic
   context setting.
3. Thalamic Reticular Nucleus (TRN) as attentional searchlight: GABAergic shell
   around thalamus. Cortex layers 5+6 drive TRN; TRN selectively inhibits/disinhibits
   thalamic relay cells — controls which column populations update together. Closest
   analogue to a global coordinator, but bottom-up, not centrally planned.
Temporal synchrony (gamma binding) is largely disproven. Binding is via *enhanced
firing rates* in co-selected columns, not oscillatory synchrony.

**Implication for the high-dimensional semantic space problem:**
No single column holds all hundreds of semantic axes. Different macrocolumns
specialize in different subspaces — exactly as V1 columns specialize in orientation
and S1 columns in touch modalities. Coordination across the full semantic space
uses long-range connections (semantic similarity → feature grouping) and thalamic
gating (attention selects which subspaces update). Transformers' multi-head attention
accidentally rediscovered this: attention heads ARE the learned version of TRN-gating
+ long-range horizontal connections combined. The key gap: TBT grounds columns in
reference frames (location). Transformers don't.

**Implication for TBT's completeness claim:**
Strongest for low-dimensional sensorimotor tasks (one whisker, one finger).
Weakest for high-dimensional abstract domains (language, multi-modal concepts).
Completeness in language requires coordination across many columns, not one.
This does NOT invalidate TBT — it means the multi-column coordination mechanism
(Phase S5) is as important as the single-column model.

Key sources:
- Mountcastle 1997 — cortical column as modular unit
- Rockland & Lund 1983 — long-range horizontal connections V1
- Haeusler & Maass 2007 — computational properties of cortical column models
- Purushothaman et al. 2012 — thalamic reticular nucleus as searchlight
- Milner 1974; Shadlen & Movshon 1999 — synchrony critique
- Bhatt et al. 2007 — lateral connections and contour integration
- Hawkins et al. 2017 (Frontiers Neural Circuits) — TBT columns paper
- Rao & Ballard 1999 — canonical microcircuits for predictive coding

---

## Design Constraints

- **No backpropagation** through the temporal memory. Learning is local Hebbian only.
- **Online learning**: one token at a time, one pass over the data. No epochs.
- **Sparse representations**: SDRs throughout. No dense float vectors in the core path.
- **Interpretable**: every stored pattern should be human-readable
  (which cells fired at which context keys).
- **No external embeddings**: don't inject word2vec or any pretrained representation.
  The system should discover structure from raw token sequences.

---

## Relationship to CipherNet Vision System

The temporal memory is an extension of the existing architecture, not a replacement.
The shared foundation:

```
MacroColumn
  minicolumns: list[MiniColumn]   ← spatial: stores SDRs per location
  ↓ extend to:
  minicolumns: list[TemporalMiniColumn]  ← temporal: stores context per cell
```

The `SensorModality` abstraction already exists. A `TokenModality` would encode
token IDs into SDRs the same way `VisualModality` encodes image patches — the
column machinery is modality-agnostic.

Reference frames already support `TemporalFrame` (1D time axis). This is the
starting point for temporal location keys.

---

## Known Unknowns

1. Whether single-layer temporal memory generalizes to language, or whether
   hierarchical temporal pooling is necessary from the start.

2. What the right granularity is for the "fixation" unit in language.

3. Whether bursting alone (unsupervised prediction error) is sufficient as a
   training signal for language, or whether some contrastive signal is needed.

4. How to implement reference frame alignment for language
   (the equivalent of centroid-relative fixations for text).

5. Whether the abstract reference frame for language should be learned
   (emergent from data) or constructed (from linguistic theory like dependency
   grammar axes).

6. What mechanism replaces the TRN searchlight / top-down gating for
   multi-column coordination in a TBT-LLM. Attention is the transformer's answer;
   the Hebbian-only constraint requires an alternative that doesn't use backprop.

7. How many macrocolumns are needed for language, and how to partition the
   semantic subspace across them. (Biology answer: emerges from data, not designed.)

---

## Status

- [ ] S1 — Temporal memory prototype (next-symbol prediction, simple sequences)
- [ ] S2 — Temporal location keys
- [ ] S3 — Character/token sequence on WikiText-2
- [ ] S4 — Hierarchical temporal memory
- [ ] S5 — Reference frame integration for language
- [x] Neuroscience Q1 answered (language reference frame axes) — high-dimensional
      geometric semantic space; broad domain subspaces; grid-cell coding navigates it;
      axes are emergent, not pre-labelled.
- [x] Neuroscience Q2 answered (brain time representation) — map-based, not chain;
      time cells as population code; successor representation; theta sequences as
      prospective compression; temporal context model.
- [x] Neuroscience Q3 (column architecture) — macrocolumn is the complete unit
      (not minicolumn); coordination is distributed via long-range horizontal
      connections + TRN gating + enhanced firing rates; no central binder.
