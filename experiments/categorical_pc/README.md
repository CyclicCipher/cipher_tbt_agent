# Categorical Predictive Coding

## Motivation

Exploring whether **category theory can serve as an inductive bias** for neural networks, rather than just a formalism for analyzing existing architectures.

Current categorical deep learning research (Natural Graph Networks, etc.) uses category theory to *describe* neural networks post-hoc. This experiment tests whether *enforcing* categorical laws during learning improves performance.

## Status: Exploratory

We're in the "we don't know what we're doing" stage. This is NOT:
- A replacement for the main predictive coding implementation
- Abandoning the knowledge graph idea
- A commitment to any particular approach

This IS:
- Testing if CT constraints help or hurt learning
- Building empirical evidence before theoretical speculation
- Keeping options open (pure neural network vs. CTKG vs. hybrid)

## Categorical Constraints Being Tested

### 1. Compositional Predictions (Phase 1 - Current)

**Constraint:** Long-range predictions must factor through intermediate layers.

For layers i > k > j:
```
W_i→j = W_i→k ∘ W_k→j
```

**Hypothesis:** Enforcing composition will:
- Force hierarchical representations
- Improve compositional generalization
- Prevent inconsistent multi-scale predictions

**Implementation:** Added as soft constraint (regularization term in loss)

### 2. Functorial Cross-Network Mappings (Phase 2 - Planned)

**Constraint:** Mappings between subnetworks (e.g., Vision→Association) must preserve compositional structure.

**Hypothesis:** Ensures different modalities align compositionally.

### 3. Universal Properties for Architecture (Phase 3 - Planned)

**Constraint:** Use categorical constructions (products, coproducts) to determine subnetwork organization.

**Hypothesis:** Provides principled alternative to ad-hoc architectural choices.

## Experiments

### Experiment 1: MNIST Digit Recognition

**Baseline:** Standard modular PC network (from `tests/train_grounded_math.py`)
**Categorical:** Same architecture + compositional prediction constraint
**Metric:** Sample efficiency (accuracy vs. training examples)

**Prediction:** If CT helps, categorical version should learn faster with fewer examples.

### Experiment 2: Compositional Generalization

**Task:** Learn digits 0-7, test on 8-9 (or learn parts, test on novel combinations)
**Metric:** Zero-shot accuracy on held-out test set

**Prediction:** If composition enforcement helps, should better generalize to unseen combinations.

## Running Experiments

### Quick Start

```bash
# Compare vanilla vs categorical networks on digit recognition
cd experiments/categorical_pc
python compare_networks.py
```

This will:
1. Create two networks with identical architectures
2. Train both on **synthetic digit recognition** (300 samples)
3. Compare test accuracy and composition error
4. Generate comparison plots

**Note:** We're using synthetically generated digit images (via `GroundedMathCurriculum`), not actual MNIST dataset files. The curriculum programmatically renders digits 0-9 with variations in style, size, and position - similar to MNIST but without requiring dataset downloads.

### Expected Runtime
- ~5-10 minutes on CPU
- ~2-3 minutes on GPU

### Output
- Console: Epoch-by-epoch accuracy comparison
- Plot: `comparison_results.png` showing accuracy curves
- Interpretation: Whether categorical constraints help/hurt

## Results

*To be filled in after running experiments*

### Hypothesis Testing

**If categorical ≈ vanilla accuracy:**
- Compositional constraint is neutral (neither helps nor hurts)
- May indicate constraint is already implicitly satisfied by vanilla PC

**If categorical > vanilla accuracy:**
- Compositional enforcement improves learning!
- Evidence that CT constraints provide useful inductive bias
- Worth exploring additional categorical constraints (functoriality, etc.)

**If categorical < vanilla accuracy:**
- Constraint is harmful (over-regularization)
- May need different formulation or weaker λ
- Suggests composition isn't the right inductive bias for this task

## Open Questions

1. **Does composition enforcement help?** Or does it just slow learning by constraining the optimization landscape?
2. **How much regularization?** What's the right λ for the composition loss term?
3. **Hard vs. soft constraints?** Should composition be enforced exactly, or just encouraged?
4. **Which constraints matter?** Composition? Functoriality? Universal properties? All of them? None?

## Related Work

- Natural Graph Networks (NeurIPS 2020): Equivariant GNN kernels as natural transformations
- Categorical Deep Learning: Algebraic frameworks for formalizing architectures
- Categories for AI (cats.for.ai): Resources on compositional AI

## Next Steps

1. Implement basic compositional constraint
2. Run baseline vs. categorical comparison on MNIST
3. Analyze: sample efficiency, generalization, learned representations
4. Decide: pursue further, abandon, or modify approach
