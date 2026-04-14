# OaK Architecture Options

*Written to survive context compaction. Records the reasoning behind design decisions,
not just the decisions themselves.*

---

## The Core Problem

ARC-AGI requires a model to do four distinguishable things in sequence:

1. **Perceive** — encode the example grids (input/output pairs) into some internal representation
2. **Induce the rule** — extract the transformation that maps inputs to outputs across all examples
3. **Apply the rule** — reason about how the transformation applies to the novel test input
4. **Generate the output** — produce the test output grid

These are not the same operation. The critical architectural question is whether they are:
- **Implicit** — all happening inside a single forward pass, entangled in the hidden state
- **Explicit** — separated into distinct modules or phases with identifiable intermediate representations

This question recurs at every scale of the problem, and the answer has implications for
every other design choice.

---

## The Tensions

### Tension 1: Bidirectional vs Causal

**Why bidirectionality helps for ARC:**
- The output grid has no inherent left-to-right ordering. Predicting cell (0,0) before
  cell (2,3) is arbitrary. Autoregressive errors on early cells corrupt later predictions.
- The rule is a global property of the entire example set — it cannot be known from the
  first few tokens. A causal model reading example 1 can't yet know what example 2 will
  constrain.
- Bidirectional models (BiMamba, transformers) can predict all output cells simultaneously,
  conditioned on the full example context.

**Why causality is required for actions:**
- A model generating keyboard/mouse events, audio, or text cannot condition on future
  outputs it hasn't yet decided on. Causality is not a limitation here — it is correct.
- A purely bidirectional model applied to action generation would need to see the entire
  future action sequence before generating any action, which is not possible at inference time.

**The unresolvable conflict:** One model cannot be simultaneously causal everywhere and
bidirectional everywhere. The resolution is to choose the right mode per component.

### Tension 2: Implicit vs Explicit Rule Representation

**Implicit (in-context learning):** The rule is never named or represented explicitly. After
seeing the examples, the hidden state h_T contains an entangled summary that includes the
rule as well as everything else. The model generates the output by continuing from this state.

- Pros: No need to define a rule representation space. End-to-end trainable. Works for GPT,
  works for Mamba.
- Cons: The rule is uninterpretable, not reusable across episodes, and can't be verified.
  The model has no explicit mechanism to "check" its inferred rule against all examples before
  applying it.

**Explicit (rule as a latent or program):** The rule is extracted into a separable
representation — a continuous vector, a discrete code, a natural language description, or a
program — and the output is generated conditioned on this representation.

- Pros: Interpretable. Can sanity-check the rule. Can reuse the rule for multiple test inputs.
  The symbolic AI codebase (experiments/symbolic_ai/) is exactly this approach.
- Cons: Requires defining what "a rule" looks like as a representation. Harder to train
  end-to-end. The space of ARC rules does not have an obvious finite basis.

**Current position:** The neural approaches in OaK use implicit rule representation.
The symbolic approach uses explicit rules (CTKG process primitives). These are not in
conflict — the long-term vision is for the neural model to eventually ground into the
symbolic layer.

### Tension 3: Spatial vs Temporal Data

The claim "transformers for spatial, Mamba for temporal" is approximately useful but
technically wrong. The real axes are:

| | Global receptive field | Scales to long sequences | Causality |
|---|---|---|---|
| CNN | No (local only) | O(n) | None |
| Transformer | Yes | O(n²) | Optional |
| Causal Mamba | Yes (causal) | O(n) | Required |
| BiMamba | Yes (full) | O(n) | None |

The practical consequence for OaK:
- **ARC grids** are spatial and output-non-causal → BiMamba is correct
- **Audio** is very long and causal → Causal Mamba is uniquely efficient (44100 Hz = 220K
  samples per 5 seconds; a transformer over raw audio is not viable at 4GB VRAM)
- **Keyboard/mouse events** are sparse and causal → Causal Mamba
- **Game screenshots** are spatial and can be processed bidirectionally per frame → BiMamba
  per frame, causal Mamba across frames

No single directionality is correct for all modalities. The architecture must handle both.

### Tension 4: Efficiency vs Expressiveness

At 4GB VRAM, any architecture must be efficient. This rules out:
- Quadratic attention over long sequences (no full-sequence transformers on 900-token grids
  unless d_model is tiny)
- Very deep networks

BiMamba is 2× the compute of causal Mamba (two passes) but still O(n), still fits.

---

## Architectural Options

### Option A: Monolithic Causal Sequence Model (Current OaK baseline)

The entire sequence — examples, test input, QUERY, test output — is fed to a causal Mamba.
The model generates the test output autoregressively, left-to-right.

```
[SEP][in1][SEP][out1]...[SEP][test_in][QUERY] → autoregressive decode → [out_cell_0][out_cell_1]...
```

**Pros:** Simple. All existing OaK code. Works for Phase 1.

**Cons:**
- Autoregressive output is wrong for 2D grids (no left-to-right ordering, errors compound).
- The causal state at QUERY must compress the entire rule into a single vector — implicit
  and fragile.
- Doesn't extend to non-causal modalities without modification.

**Current status:** Implemented. Phase 1 result: 75% exact match (teacher-forced), 98.6%
per-cell. Real autoregressive performance would be lower.

---

### Option B: Causal Encoder + Diffusion Decoder (Hybrid)

Two-phase processing:
1. Causal Mamba encodes the example sequence → conditioning context c (the hidden state at QUERY)
2. Small BiMamba diffusion head takes c + masked test output → iteratively unmasks all cells

```
[SEP][in1][SEP][out1]...[QUERY]
              ↓ causal Mamba
              c (context vector)
              ↓ BiMamba diffusion head (conditioned on c)
[mask][mask]...[mask] → iterative refinement → [out_cell_0][out_cell_1]...
```

**Pros:**
- Keeps causal Mamba for context encoding (efficient, familiar).
- Non-autoregressive output (correct for grids).
- Clean separation: encoder handles "what is the rule?", decoder handles "what is the output?".
- Extends naturally: swap the diffusion head for a causal head for action generation.

**Cons:**
- The causal encoder cannot use the test input to guide its attention during rule induction.
  (The model reads examples without knowing what the test input looks like.)
- Two separate training objectives (CE for encoder, MDLM for decoder).
- Modular but more complex to implement.

**Assessment:** This is the "safer" path. Well-founded in the vision-language model literature
(e.g., LLaVA: causal encoder + decoder). For ARC, the inability to see the test input during
rule induction is a real limitation — it can't focus on the aspects of the rule that matter
for this specific test case.

---

### Option C: Full BiMamba + Masked Diffusion (Full Pivot)

The entire sequence is processed by a bidirectional Mamba. All positions see all other
positions. Training uses masked diffusion: mask the test output (and occasionally example
outputs), train the model to reconstruct them.

```
Bidirectional processing of:
[SEP][in1][SEP][out1][SEP][in2][SEP][out2]...[SEP][test_in][QUERY][MASK][MASK]...[MASK]
                        ↓ BiMamba (full bidirectional)
                        ↓ iterative unmasking
[SEP][in1][SEP][out1]...[test_in][QUERY][out_cell_0][out_cell_1]...
```

**Pros:**
- Output cells can attend to all examples AND the test input simultaneously. The rule
  is never "compressed into a state" — it lives in the bidirectional context.
- Non-autoregressive output.
- Simpler training: single MDLM objective for the whole sequence.
- The test input can guide what aspects of the rule are attended to.
- DiffuMamba paper demonstrates this works: 8.2× faster inference than transformer MDLMs.

**Cons:**
- **Bidirectionality is wrong for actions.** When this model is extended to keyboard/mouse
  output, every action token would attend to future action tokens — which are not available
  at inference time. The architecture as designed cannot generate sequential actions.
- BiMamba is 2× the compute of causal Mamba.
- No explicit separation of "rule induction" from "rule application" — still implicit.

**Current decision:** Implement this for Phase 1 (ARC grids only). The action generation
problem does not arise until Phase 3 (interactive environment). When it does, address it
separately.

---

### Option D: NPR-Style Parallel Reasoning (Rejected for now)

The model generates a "plan" (explicit rule description in tokens) and then executes all
output cells in parallel branches, one per cell. Each branch is isolated from the others
via attention masking.

```
[examples] → [plan: rotate 90°] → parallel branches:
  branch_0: predict cell (0,0) given plan + test_input
  branch_1: predict cell (0,1) given plan + test_input
  ...all simultaneously
```

**Pros:** Explicit rule representation (the plan tokens). Correct parallel output.
Validated by NPR at 4.6× speedup.

**Cons:**
- NPR's attention masking tricks are transformer-specific. Cannot be directly applied to
  Mamba, which has no attention matrix to mask.
- Requires the model to generate natural language rule descriptions, which requires language
  capability we haven't built.
- The plan tokens are supervision signals for rule induction — where do training labels for
  the plan come from? (NPR uses RL; for ARC this would be sparse reward only.)

**Assessment:** The right long-term architecture for interpretable rule induction, but
requires language capability and transformer attention or equivalent. Deferred.

---

### Option E: Two-Stage with Explicit Rule Vector (JEPA-style)

Stage 1: Encode all example pairs → single rule vector z (a learned latent)
Stage 2: Apply z + test input → output grid

```
(in1, out1), (in2, out2) → encoder → z
z + test_input → decoder → test_output
```

**Pros:** z is the rule, explicitly. Can be inspected via nearest-neighbour lookup.
Clean separation of induction (stage 1) and application (stage 2).
Very fast inference (encode once, decode once).

**Cons:**
- How to train z without rule labels? Contrastive (episodes with same rule should have
  similar z) or reconstruction (z must enable reconstruction of all example outputs from
  their inputs). Both work but add complexity.
- z is a single vector — may not have enough capacity for complex compositional rules.
- Requires defining the encoder architecture separately.

**Assessment:** Good long-term direction, especially for interpretability and the CTKG
connection (z could be grounded to a symbolic rule). Not the current priority.

---

## Research Findings: Multimodal and VLA Architectures

*Added after surveying: Chameleon, Show-o, Unified-IO 2, 4M, Flamingo, LLaVA (multimodal);
RT-2, π0, OpenVLA, Octo, RoboMamba (VLA). These findings resolve the tensions above.*

---

### Finding 1: The Bidirectional/Causal Tension Is Already Solved in VLA Literature

Vision-Language-Action models face exactly OaK's Phase 3 problem: bidirectional observation
encoding + causal action generation. The VLA consensus (RT-2, OpenVLA, π0, RoboMamba) is:

**Bidirectional backbone → modality-specific output head.**

The backbone processes all observations bidirectionally (full context). The output head is
swapped per task:
- For grid output: bidirectional diffusion head (predict all cells in parallel)
- For action output: causal head or flow-matching head (predict the next action chunk)

The backbone never needs to be causal. Causality is a property of the *output head*, not the
*encoder*. This is the key insight that dissolves Tension 1.

**RoboMamba** (arXiv:2406.04339): Causal Mamba backbone + small causal action head. Achieves
3× faster inference than transformer VLAs. Demonstrates that Mamba-based VLA is viable at
small scale. Relevant: if causal Mamba suffices for Robotics (simpler reasoning), BiMamba
is the correct choice for ARC (complex spatial reasoning).

---

### Finding 2: Show-o Pattern — Per-Modality Attention Masking

**Show-o** (ByteDance, 2024): A single model handles text (causal) and image generation
(discrete diffusion) by applying *different attention masks per token type* within the same
forward pass:
- Text tokens: causal mask (upper-triangular)
- Image tokens: full bidirectional mask

This is directly applicable to OaK's Phase 3 architecture: text/action tokens use causal
masking; grid tokens use bidirectional masking. A single forward pass handles both output
types without separate models or two-phase processing.

**Implementation implication for BiMamba:** Mamba does not have an explicit attention matrix
to mask. The Show-o trick requires a transformer's explicit masking. For Mamba-based OaK,
the equivalent is: causal Mamba for action sequences, BiMamba for grid sequences, with a
shared embedding space and a router that dispatches each output to the appropriate head.
This is Option B (Encoder + Head per modality) generalized.

---

### Finding 3: π0 — Flow Matching for Action Chunks (Not Autoregressive)

**π0** (Physical Intelligence, arXiv:2410.24164): Actions are generated as a *chunk* of K
future timesteps using flow matching — not autoregressively, one step at a time.

Flow matching is a continuous-space analog of discrete diffusion:
- Training: add noise to the action chunk, train a denoiser to recover it
- Inference: start from noise, iteratively denoise to produce the full K-step action chunk

This sidesteps the autoregressive bottleneck for action generation. **The key insight:** the
human hand does not plan one muscle command at a time — it plans a movement trajectory
(the "action chunk"). Chunked flow matching is more biologically plausible and more efficient.

For OaK's Phase 3 keyboard/mouse control: predict a *sequence of N input events* as a
unit using flow matching or discrete diffusion, rather than predicting one keypress at a time.
This makes action generation non-autoregressive at the chunk level, even though the chunk
itself is temporally ordered.

---

### Finding 4: Unified Tokenization (Chameleon)

**Chameleon** (Meta, arXiv:2405.09818): All modalities — text, images, grids — are tokenized
into a shared discrete vocabulary. A single model with a single token embedding table processes
all modalities without separate visual encoders or cross-attention bridges.

**Why this matters for OaK:** The current architecture already does this (color tokens 0-9,
SEP, QUERY, PAD all in a small unified vocabulary). This is the correct long-term direction.
When adding screenshots, audio spectrograms, or keypress events, the goal is to tokenize
them into the same vocabulary space rather than building separate encoder branches.

**VQVAE residual quantization** is the standard method for tokenizing continuous observations
(images, audio) into discrete codes that live in the same space as text/event tokens.

---

### Finding 5: Interleaved-MRoPE for Mixed-Modality Position Encoding

**Interleaved-MRoPE** (arXiv:2510.23095, from Qwen2-VL): The standard for mixed-modality
position encoding. Position IDs are assigned to tokens based on their modality type:
- Text tokens: 1D sequential position
- Image tokens: 2D (row, col) position with shared temporal index

OaK's 2D-PoPE (already implemented) is the Mamba equivalent of 2D-RoPE. The multi-modal
extension follows the same factored-cumsum pattern — when introducing game screenshots,
each frame gets 2D positions; the sequence of frames gets a causal temporal index.

---

### Finding 6: Multimodal Architecture Patterns

| Architecture | Backbone | Visual Encoding | Output | Key takeaway |
|---|---|---|---|---|
| Flamingo | Causal LM | Frozen ViT + Perceiver resampler | Causal text | Perceiver resampler compresses long visual sequences → O(1) latent per image |
| LLaVA | Causal LM | Frozen CLIP + linear projection | Causal text | Simple and effective; linear projection works |
| Unified-IO 2 | T5 encoder-decoder | Patch tokens | Token prediction | All modalities as token sequences |
| 4M | MAE backbone | Masked patches | Parallel reconstruction | Masking any subset of modalities, predicts the rest |
| Show-o | GPT-like | VQ-encoded image patches | Text=causal, image=diffusion | Per-modality attention mask within single forward pass |
| Chameleon | GPT-like | VQ-encoded to shared vocab | All: next-token | Fully unified; simplest architecture |

**The gradient of complexity:** Flamingo (frozen vision encoder, frozen LM, small adapter) →
LLaVA (frozen encoder, full LM) → Chameleon (fully joint, no frozen components). OaK should
start at the LLaVA end (minimal coupling) and move toward Chameleon (unified vocab) as the
system matures.

---

## The Multimodal Future: Resolved Design

Given the research above, the long-term architecture is:

**Bidirectional BiMamba backbone + modality-specific output heads.**

```
Observations (grids, screenshots, audio spectrogram, events)
    ↓ modality-specific tokenizer (VQ for continuous; direct for discrete)
    ↓ shared embedding table
    ↓ BiMamba backbone (bidirectional, full context)
    ↓ task router
   /        \
Grid head   Action head
(diffusion)  (flow matching on K-step chunk)
```

This resolves all four tensions:
1. **Bidirectional vs Causal**: backbone = bidirectional, heads = per-task
2. **Implicit vs Explicit rule**: BiMamba backbone = implicit; symbolic layer (Option E/D) = explicit, connected later
3. **Spatial vs Temporal**: 2D-PoPE for spatial tokens, 1D-PoPE for temporal tokens; same backbone
4. **Efficiency**: BiMamba is O(n) like causal Mamba; VQ tokenization compresses continuous observations

The **Perceiver IO pattern** remains the aspirational long-term architecture for extreme-length
sequences (raw audio at 44.1kHz), but is not needed until Phase 4+.

---

## Current Decision Summary

| Phase | Architecture | Objective | Output | Status |
|---|---|---|---|---|
| Phase 1 (ARC rules) | Full BiMamba | Masked diffusion (MDLM) | Iterative unmasking | **To implement** |
| Phase 2 (objects) | Full BiMamba | Same | Same | Pending Phase 1 |
| Phase 3 (interactive) | BiMamba backbone + causal/flow-matching action head | MDLM + flow matching | Grid: diffusion; Actions: K-step chunk | Design resolved |
| Multimodal | BiMamba backbone + VQ tokenization + modality-specific heads | Same | Per-modality | Long-term |

---

## Open Questions

1. **Phase 3 action head design:** Flow matching (continuous, like π0) or discrete diffusion
   (same MDLM objective, action tokens are discrete keypresses)? For keyboard/mouse, actions
   are already discrete events — discrete diffusion over an action chunk is simpler and
   consistent with the MDLM backbone. Chunk size K: probably 4–16 timesteps.

2. **When does explicit rule representation become necessary?**
   Implicit (in-context) rule induction may saturate at some ARC difficulty. The failure
   mode is: model learns to pattern-match on seen rule types rather than induce novel rules.
   Explicit z (Option E) or scratchpad (Option D) may become necessary for ARC-AGI-3.

3. **Is MDPO or PAPO the right fine-tuning strategy after supervised pretraining?**
   Both address the train/inference gap. MDPO (RL for MDLMs via reward shaping) is more
   natural for the masked diffusion setup. PAPO (on-policy RL with binary outcome reward)
   is more general. Both use exact grid match as the reward signal.

4. **Where does the symbolic layer (CTKG) connect?**
   The neural model learns implicit rules; the symbolic layer expresses explicit rules.
   One hypothesis: the neural model's rule latent (Option E) or plan tokens (Option D)
   eventually ground to CTKG process primitives. This is the long-term bridge between
   the two codebases.

5. **When to move from causal to BiMamba backbone?**
   The current OaKMixer is causal (using standard ssd_trapz). The full BiMamba pivot
   (two passes, forward + backward, combined by addition) requires replacing OaKMixer
   with a BiOaKMixer. This is the next implementation step after Phase 1 validation.
