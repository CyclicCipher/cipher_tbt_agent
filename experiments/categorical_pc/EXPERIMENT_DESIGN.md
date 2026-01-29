# Experiment Design: Category Theory as Inductive Bias

## Research Question

**Can category theory provide useful inductive biases for neural networks, beyond being a formalism for analyzing existing architectures?**

## Background

Current categorical deep learning research (Natural Graph Networks, etc.) uses category theory to **describe** neural networks post-hoc. This experiment tests whether **enforcing** categorical laws during learning improves performance.

## Hypothesis

Enforcing compositional structure (W_i→j = W_k→j ∘ W_i→k) will improve:
1. **Sample efficiency:** Learn faster with fewer examples
2. **Compositional generalization:** Better transfer to unseen combinations
3. **Interpretability:** Learned representations have clear hierarchical structure

## Experimental Setup

### Architecture

Both networks use identical structure:
- **Position 0:** Vision (256→128→64) + Motor (10→32)
- **Position 1:** Association (96→128→64→10)
- **Total:** ~150K parameters

### Key Difference

**Vanilla Network:**
- Standard predictive coding
- No architectural constraints beyond PC dynamics

**Categorical Network:**
- Same PC dynamics
- Additional constraint: Compositional predictions
- Enforced via regularization: λ * composition_error
- λ = 0.1 (tunable hyperparameter)

### Compositional Constraint

For layers with 3+ layers (e.g., association subnet):
```
state_2 = tanh(W_1→2 @ state_1)
pred_2_from_0 = tanh(W_1→2 @ tanh(W_0→1 @ state_0))

Constraint: Minimize ||pred_2_from_0 - state_2||²
```

This enforces that layer 2's state is **consistent with** the composition of bottom-up predictions through layer 1.

### Dataset

MNIST digit recognition:
- **Training:** 300 samples (30 per digit)
- **Test:** 100 samples (10 per digit)
- **Input:** 100×100 grayscale images, retinal preprocessing
- **Output:** 10-class classification

### Training

- **Epochs:** 5
- **Learning rate:** 0.01 (weights), 0.05 (inference)
- **Optimizer:** StableProspectiveLearning
- **Inference iterations:** 30 per forward pass

### Metrics

1. **Test Accuracy:** Primary metric for performance
2. **Sample Efficiency:** Accuracy curve over epochs
3. **Composition Error:** How much predictions violate composition (categorical only)

## Possible Outcomes & Interpretation

### Outcome 1: Categorical ≈ Vanilla (±2%)

**Interpretation:** Compositional constraint is neutral.

**Possible explanations:**
- Vanilla PC already learns compositional structure implicitly
- Constraint is satisfied "for free" by gradient descent
- Task doesn't require compositional reasoning

**Next steps:**
- Test on compositional generalization task (learn parts, test on combinations)
- Increase λ to see if stronger constraint changes anything
- Analyze learned representations: are they compositional even without constraint?

### Outcome 2: Categorical > Vanilla (+5% or more)

**Interpretation:** Compositional enforcement improves learning! 🎉

**Possible explanations:**
- Constraint acts as useful inductive bias
- Prevents non-compositional solutions that overfit
- Forces hierarchical representations that generalize better

**Next steps:**
- Test additional categorical constraints (functoriality, universal properties)
- Scale up to harder tasks (compositional reasoning, multi-step problems)
- Analyze: What aspects of composition are most important?
- Consider applying to CTKG: Do graphs benefit from composition enforcement?

### Outcome 3: Categorical < Vanilla (-5% or more)

**Interpretation:** Constraint is harmful (over-regularization).

**Possible explanations:**
- Composition is too restrictive for this task
- λ too high (over-regularizes)
- Wrong formulation of compositional constraint
- Real-world doesn't obey categorical laws strictly

**Next steps:**
- Tune λ: Try 0.01, 0.05, 0.2 to find sweet spot
- Try different formulation: Hard constraint vs. soft regularization
- Test on different task: Maybe vision doesn't need composition, but reasoning does?
- Consider: Maybe composition isn't the right CT constraint (try functoriality instead?)

## Limitations

1. **Small dataset:** 300 training samples may not show sample efficiency gains
2. **Simple task:** Digit recognition may not require compositional reasoning
3. **Single constraint:** Only testing composition, not full categorical structure
4. **Specific architecture:** Results may not generalize to other network types

## Future Experiments

If compositional constraint shows promise:

### Experiment 2: Compositional Generalization
- **Train:** Digits 0-7
- **Test:** Digits 8-9 (novel combinations of learned features)
- **Metric:** Zero-shot accuracy on held-out digits

### Experiment 3: Functorial Mappings
- **Constraint:** Vision→Association mapping must preserve structure
- **Test:** Does enforcing functoriality improve vision-language alignment?

### Experiment 4: Universal Properties
- **Constraint:** Use products/coproducts to determine architecture
- **Test:** Do categorical constructions lead to better architectures?

## Connection to CTKG

**Why test on neural network first?**
- Easier to implement and debug
- Faster iteration (no need to build graph infrastructure)
- Results inform whether CT constraints are worth pursuing for CTKG

**If categorical constraints help neural networks:**
- Strong evidence they'd help CTKG too
- Can apply same principles to graph learning
- Validates "CT as inductive bias" approach

**If categorical constraints don't help neural networks:**
- May still work for CTKG (graphs are more naturally categorical)
- Or may indicate we need different approach entirely
- Saves time by testing core hypothesis early

## Success Criteria

**Minimum success:** Learn what works and what doesn't
- Even negative results are valuable (tells us what not to do)
- Better to fail fast on simple experiment than build CTKG on wrong assumptions

**Strong success:** Categorical > Vanilla by 5%+
- Evidence that CT constraints improve learning
- Justifies exploring full categorical architecture
- Informs CTKG design with validated principles

**Transformative success:** Categorical enables compositional generalization
- Can learn from parts and generalize to wholes
- Opens path to sample-efficient, composable AI
- Validates category theory as foundation for AGI architecture
