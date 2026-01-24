# Predictive Coding Theory from Research Literature

**Created:** 2026-01-24
**Purpose:** Document the actual theory from research literature, not Gemini's confabulations

## Sources

### Primary Papers
- [Rao & Ballard (1999): Predictive coding in the visual cortex](https://www.nature.com/articles/nn0199_79) - Original predictive coding paper
- [Whittington & Bogacz (2017): Approximation of Error Backpropagation with Hebbian Plasticity](https://direct.mit.edu/neco/article/29/5/1229/8261) - Local Hebbian learning
- [Millidge et al. (2022): Predictive Coding Networks Tutorial](https://arxiv.org/abs/2107.12979) - Comprehensive review
- [Song et al. (2024): Inferring neural activity before plasticity](https://www.nature.com/articles/s41593-023-01514-1) - Prospective configuration
- [Tutorial: Predictive Coding Networks (July 2024)](https://arxiv.org/abs/2407.04117) - Implementation guide
- [Introduction to PC Networks (2024)](https://arxiv.org/abs/2506.06332) - PyTorch tutorial

### Implementations
- [alec-tschantz/predcoding](https://github.com/alec-tschantz/predcoding) - Whittington & Bogacz 2017 Python implementation
- [bjornvz/PRECO](https://github.com/bjornvz/PRECO) - PyTorch PC networks library

## Core Theory

### Two Types of Neurons

Predictive coding networks have **two types of neurons per layer**:

1. **Value neurons (r)**: Represent the layer's current state/activity
2. **Error neurons (e)**: Represent prediction errors

**Key finding from literature:** "Value neurons project to both the error neurons of the layer below and error neurons at the current layer. Error neurons receive inhibitory top-down inputs from value neurons of the layer above and excitatory inputs from value neurons at the same layer."

### Energy Function

The network minimizes an energy function based on prediction errors:

```
E = Σ_layers (1/2) * ||e_i||²
```

Where each error neuron computes:
```
e_i = r_i - prediction_i
```

**From literature:** "Predictive coding minimizes a local energy function for each layer" via "gradient descent on the variational Free Energy."

### Two-Phase Learning (Prospective Configuration)

**Phase 1: Inference** (minimize energy w.r.t. neuron states)
- Update value neurons r_i to minimize prediction errors
- Iterate until convergence (equilibrium)
- States change but weights stay fixed

**Phase 2: Learning** (minimize energy w.r.t. weights)
- After inference converges, update weights
- Use converged states and errors
- Weights change to consolidate the activity pattern

**From literature:** "In prospective configuration, before synaptic weights are modified, neural activity changes across the network so that output neurons better predict the target output; only then are the weights modified to consolidate this change in neural activity."

### Weight Update Rules

**Key finding:** "The weight update rule obeys Hebbian plasticity since it is simply a multiplication of the prediction error of the layer below and the value neurons at the current layer."

More specifically: "The update rules for weight matrices are precisely Hebbian, since they are outer products between the prediction errors and the value neurons of the layer above."

**General form:**
```
ΔW_ij = -η * ∂E/∂W_ij
```

This gradient descent on energy leads to:
```
ΔW = η * (error neurons) ⊗ (value neurons)
```

**Critical question to resolve:** Is it:
- `ΔW = η * e_below ⊗ r_current`  (error below, value current)
- `ΔW = η * e_current ⊗ r_above`  (error current, value above)

The exact formulation depends on the connection direction.

### Inference Dynamics

During inference, value neurons are updated via gradient descent on energy:

```
dr_i/dt = -∂E/∂r_i
```

This involves:
1. Bottom-up errors pushing states up
2. Top-down predictions pulling states down
3. Iteration until equilibrium

**From literature:** "Perception in the network corresponds to estimating latent vectors by updating neural responses through network dynamics to minimize prediction errors via gradient descent."

## Our Current Implementation vs Theory

### What We Have
- Value neurons: `layer.state` (correct ✓)
- Errors: `layer.error` (correct ✓)
- Two-phase: inference then learning (correct ✓)

### What May Be Wrong

1. **Error computation sign?**
   - We use: `error = state - prediction_from_above`
   - Should it be: `error = prediction_from_above - state`?

2. **Weight update sign?**
   - We use: `W += lr * error * input` (additive)
   - Should it be: `W -= lr * error * input` (subtractive)?

3. **Which states for weight update?**
   - We use: raw states from adjacent layers
   - Correct according to Hebbian rule ✓

4. **Learning rate magnitude?**
   - We use: 0.01
   - Literature often uses: 0.001-0.1 depending on formulation
   - May need adjustment based on sign conventions

## Unknowns Requiring Further Investigation

1. **Exact sign convention for error:**
   - `e = r - prediction` OR `e = prediction - r`?

2. **Exact sign convention for weight updates:**
   - `W += η * e * r` OR `W -= η * e * r`?

3. **Prediction computation:**
   - Currently: `prediction = W.T @ state`
   - Is this correct or should it be `W @ state`?

4. **Error neuron location:**
   - Are errors computed AT each layer (state - prediction from above)?
   - Or BETWEEN layers (state above - prediction by layer below)?

## Next Steps

1. **Find accessible source code** showing actual implementation
2. **Resolve sign conventions** from working implementation
3. **Test with known-good benchmark** (e.g., MNIST from tutorials)
4. **Fix our implementation** based on verified equations

## Hypothesis Based on Current Evidence

Based on weight explosion at Layer 0 (+2000% apical, +380% basal):

**Hypothesis:** The weight update sign is wrong. Energy minimization via gradient descent requires:
```
W -= η * ∂E/∂W
```

If we're using `+=` instead of `-=`, we're doing **gradient ascent** (maximizing energy), causing divergence.

**Test:** Flip the sign in `neuron.py:131-132` from `+=` to `-=` and rerun diagnostics.
