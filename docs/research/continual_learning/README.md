# Continual Learning Research for Compositional Curriculum

Created: 2026-02-16

## Context

Our compositional arithmetic curriculum demonstrates that staged skill-building
solves the generalization problem (100% train + chance test → 95%+ test via
curriculum). However, Stage 6 (compose DOT counting + TEN counting → two-digit
numbers) reveals catastrophic forgetting: gradient updates for the new stage
actively destroy DOT-counting circuits learned in Stages 2/4.

The S2/S4 oscillation pattern (bouncing between 10-30% and 100% accuracy across
epochs) shows this is not a replay buffer sizing issue — it's a fundamental
stability-plasticity conflict. The model needs DOT-counting weights to be
simultaneously stable (for replay) and plastic (for the new composition).

**Our problem is not standard continual learning.** Standard CL assumes
independent tasks. Our problem is compositional CL — Stage 6 must USE the
counting circuits, not merely avoid destroying them. This makes hard-freezing
approaches counterproductive.

## Key Finding: What Doesn't Work for Compositional CL

| Method | Why It Fails for Composition |
|--------|------------------------------|
| PackNet | Frozen disjoint subnetworks prevent reuse |
| SupSup | Random base weights, independent masks per task |
| Progressive Nets | Separate columns, 12x model size for 12 stages |
| ANML/OML | Learned sparsity creates independent circuits |

## What Should Work (Ranked by Priority)

### 1. Differential Learning Rates by Layer (FREE)

Lower LR for early layers (counting circuits), higher LR for later layers
(composition/output). Standard in transfer learning (fast.ai discriminative
fine-tuning). Provides implicit protection where it matters most.

- Implementation: ~5 lines, zero overhead
- Rationale: counting skill lives in early/middle layers; output mapping in later layers

### 2. DER++ (Dark Experience Replay++)

Store model's **logits** (not just inputs) when each stage completes. During
replay, add MSE loss between current logits and stored logits. This preserves
the *internal behavior* (how the model counts), not just accuracy.

Standard CE replay can maintain accuracy through re-memorization (different
circuits); logit matching prevents representational drift.

- Paper: Buzzega et al. 2020 — "Dark Experience for General Continual Learning"
- Storage: ~50-100 bytes per sample (answer-position logits only)
- Key advantage: preserves circuit behavior, not just task accuracy

### 3. EWC (Elastic Weight Consolidation)

After each stage, compute diagonal Fisher Information Matrix (FIM) measuring
weight importance. During subsequent stages, add quadratic penalty pulling
important weights back toward their post-convergence values.

L_total = L_new_task + (λ/2) Σᵢ Fᵢ(θᵢ - θ*ᵢ)²

- Paper: Kirkpatrick et al. 2017 — arXiv:1612.00796
- Overhead: ~2MB/stage (FIM diagonal + old params, stored not in model)
- Key advantage: soft constraints allow compositional reuse
- Key limitation: after 12 stages, nearly all weights become "important"

### 4. La-MAML Per-Parameter Learning Rates

Maintain per-parameter LR. Each step, compute gradients on both current and
replay batches. Where gradients align (positive dot product) → increase LR
(parameter helps both tasks = composition!). Where they conflict → decrease LR
(protect).

Most composition-aware approach: automatically distinguishes "Stage 6 builds on
DOT-counting" (aligned → learn fast) from "Stage 6 destroys DOT-counting"
(conflicting → protect).

- Paper: Gupta et al. 2020 — "La-MAML: Look-ahead Meta Learning"
- Overhead: ~2x compute per step, 2MB extra memory

### 5. Google's Nested Learning (CMS)

NeurIPS 2025 theoretical framework (Behrouz et al.). Different model components
update at different timescales. Fast-updating components handle adaptation,
slow-updating components consolidate stable knowledge.

- Paper: arXiv:2512.24695
- Blog: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
- For us: the multi-timescale idea IS differential LR by layer (#1 above)
- Full HOPE architecture requires 1.3B+ scale, not viable at 500K params

## Supporting Literature

### Compositional Learning Validation

**Zhao et al. (NeurIPS 2024)** — "Can Models Learn Skill Composition from Examples?"
Fine-tuning on compositions of k=2-3 skills teaches a META-SKILL for composition
that generalizes to k=4-5. LLaMA-2-13B composition success: 4% → 37%.
This validates our entire curriculum hypothesis.
- Paper: arXiv:2409.19808

**Lee et al. (2025)** — Compositional Curricula in In-Context Learning
Curriculum-trained transformers develop internal representations of intermediate
computation values. Confirms interleaved counting approach.
- Paper: arXiv:2506.13253

**ICLR 2025 Workshop WARNING**: Modular networks only compose when task structure
is EXPLICITLY provided. Our explicit counting structure (interleaved tokens) may
be crucial — implicit structure fails.

### Two-System Compositional CL

**Shan et al. (NeurIPS 2025)** — "What and How" Two-System Approach
Separates task learning into:
- "What" system: Bayesian inference of compositional task structure
- "How" system: Low-rank RNN with modular components, W = Σₖ αₖ uₖvₖᵀ

Key insight: recurrence weights decomposed into composable low-rank modules,
different tasks activate different subsets. Could adapt to Mamba3.
- Paper: arXiv:2510.20709

### Skills-in-Context (SKiC)

**Chen et al. (EMNLP 2024)** — LLMs achieve compositional generalization when
prompts contain skill definitions + composition examples + problem.
- Paper: https://aclanthology.org/2024.findings-emnlp.812/

## Our Experimental Evidence

### Stage 6 Failure Pattern (seed sensitivity)

Three runs with different seeds, same code:
1. **seed=42 (original)**: Passed Stage 6 quickly (~20-30 epochs)
2. **seed=X (run 2)**: Failed at 77% test after 100 epochs
3. **seed=Y (run 3)**: Failed at 48% test after 100 epochs

### S2/S4 Oscillation (the smoking gun)

During Stage 6 training, previous-stage DOT-counting accuracy oscillates:
```
ep  1   S2=0.30  S4=0.81    ← DOT counting broken from start
ep 10   S2=0.20  S4=0.80    ← still broken
ep 40   S2=1.00  S4=0.69    ← S2 recovers, S4 drops
ep 50   S2=1.00  S4=0.22    ← S4 crash
ep 60   S2=0.10  S4=1.00    ← S2 crash, anti-correlated!
ep 70+  S2=1.00  S4=1.00    ← finally stable, but damage done
```

The anti-correlation between S2 and S4 suggests the model is toggling between
two different DOT-counting circuits (1-digit vs 2-digit output format) rather
than maintaining a unified counting skill.

### Per-Token Accuracy = Product

Exact match ≈ Π(per-token accuracies). For [1.00|0.77]: exact = 0.77.
Implication for Stage 11 (3-digit output): need ~98.3% per-token for 95% exact.

## Implementation Status

- [x] Stratified replay (equal samples per stage) — committed
- [x] Epoch-level replay resampling — committed
- [x] Separate --data_seed from --seed — committed
- [ ] Differential LR by layer
- [ ] DER++ logit snapshots
- [ ] EWC quadratic penalty
- [ ] La-MAML per-parameter LR
- [ ] Per-token accuracy threshold (instead of exact-match)

## File Index

- `README.md` — This file (overview + recommendations)
- `references.md` — Full bibliography with URLs
