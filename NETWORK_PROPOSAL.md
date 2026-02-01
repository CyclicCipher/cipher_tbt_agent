# 7-Layer Predictive Coding Network Proposal for MNIST

## Design Philosophy

Based on research into VERSES AI, standard predictive coding implementations (Whittington & Bogacz, Millidge et al.), and recent scaling solutions, this proposal uses **proven standard approaches** rather than custom designs.

## Network Architecture

### Layer Structure
```
Input Layer (784 nodes) → L1 (256) → L2 (256) → L3 (256) → L4 (256) → L5 (256) → L6 (128) → Output (10)
```

**Total depth:** 7 layers (excluding input)

### Node Organization (Standard PC Architecture)

Each layer l contains:
- **Value nodes** (v_l): Hold predictions/representations
- **Error nodes** (e_l): Hold prediction errors

**Key principle:** Predictions flow forward (bottom-up), errors flow backward (top-down).

### Activation Functions
- **Hidden layers:** ReLU or tanh (start with ReLU - better gradient flow)
- **Output layer:** Softmax for classification

## Algorithm: Two-Phase Process

### Phase 1: Inference (Iterative Error Minimization)

For T_inference iterations (~35 steps = 5×L for 7 layers):
1. Compute predictions from layer l to l+1
2. Compute prediction errors at each layer
3. Update value nodes based on errors
4. Iterate until convergence

### Phase 2: Learning (Weight Updates)

After inference converges:
1. Update weights based on stabilized errors
2. Use local Hebbian-like updates (no backprop needed)

## Addressing the Scaling Problem

### Problem
PC networks degrade beyond 5-7 layers due to:
- Exponentially imbalanced errors between layers
- Vanishing error signals in deep networks
- Poor energy propagation

### Solution: μPC Parameterization

Based on recent research (μPC: Scaling Predictive Coding to 100+ Layer Networks):

1. **Residual connections** with proper scaling: scale by 1/√L
2. **Precision-weighted optimization** of latent variables
3. **Proper initialization** (depth-aware)

For our 7-layer network:
- Residual scaling factor: 1/√7 ≈ 0.378
- Use skip connections from layer l to layer l+2

### What NOT to do (from MISTAKES.md)
- ❌ Exponential precision scaling (10x per layer)
- ❌ Output clamping during training
- ❌ Custom two-compartment neuron designs
- ❌ CNN-PC hybrids without proven architecture

## Hyperparameters (Starting Point)

Based on literature review:

```python
# Network
n_layers = 7
hidden_dims = [256, 256, 256, 256, 256, 128, 10]
activation = 'relu'

# Inference
T_inference = 35  # 5 × L
inference_lr = 0.1  # Learning rate for value node updates

# Learning
weight_lr = 0.001  # Weight update learning rate
batch_size = 64

# Regularization
use_residual = True
residual_scale = 1 / sqrt(7)  # μPC scaling
```

## MNIST Training Protocol

### Dataset
- Training: 60,000 images
- Test: 10,000 images
- Input: 784 (28×28 flattened, normalized to [0,1])

### Training Procedure
1. **No pretraining** - train end-to-end
2. For each batch:
   - Run inference for T_inference steps
   - Update weights once after convergence
3. Evaluate on test set every epoch

### Success Criteria
- Achieve >95% test accuracy (comparable to backprop baseline)
- Monitor error signal magnitude across layers (should not vanish)
- Track inference convergence speed

## Implementation Approach

### Option 1: Use Existing Library (Recommended)
Use proven implementation from:
- [Bogacz-Group/PredictiveCoding](https://github.com/Bogacz-Group/PredictiveCoding)
- [infer-actively/pypc](https://github.com/infer-actively/pypc)
- [bjornvz/PRECO](https://github.com/bjornvz/PRECO)

**Rationale:** Don't reinvent the wheel. Past attempts at custom implementations failed repeatedly.

### Option 2: Minimal Custom Implementation
If building from scratch:
1. Start with simplest possible version (no residuals)
2. Add μPC scaling only if needed for stability
3. Extensively test each component
4. **Document every deviation from standard approaches in MISTAKES.md**

## Diagnostic Monitoring

Track during training:
1. **Per-layer error magnitudes** - detect vanishing errors
2. **Inference convergence** - number of steps to stabilize
3. **Weight gradient norms** - detect exploding/vanishing gradients
4. **Test accuracy** - compare to backprop baseline
5. **Energy landscape** - monitor free energy reduction

## Expected Challenges

### Challenge 1: Slow Inference
**Problem:** 35 inference iterations per batch is slow
**Mitigation:**
- Use GPU acceleration
- Consider parallel inference across batch
- Monitor if fewer iterations suffice (try 20 first)

### Challenge 2: Hyperparameter Sensitivity
**Problem:** PC networks can be sensitive to learning rates
**Mitigation:**
- Start with published values
- Use learning rate schedulers
- Grid search if needed

### Challenge 3: Numerical Stability
**Problem:** Deep networks may have stability issues
**Mitigation:**
- Proper initialization (Xavier/He)
- Gradient clipping if needed
- Monitor for NaN/Inf values

## Timeline & Milestones

### Milestone 1: Library Integration (1-2 days)
- Install and test existing PC library
- Run their MNIST example
- Verify results match published performance

### Milestone 2: 7-Layer Architecture (2-3 days)
- Modify to 7 layers
- Add μPC residual connections
- Verify stable training

### Milestone 3: Performance Tuning (2-3 days)
- Optimize hyperparameters
- Achieve >95% accuracy
- Document findings

### Milestone 4: Analysis & Documentation (1-2 days)
- Analyze error propagation
- Compare to backprop baseline
- Write up results
- Update MISTAKES.md with lessons learned

## References

### Key Papers
1. **Whittington & Bogacz (2017)** - "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network"
2. **Millidge et al. (2022)** - "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?"
3. **μPC Paper (2025)** - "μPC: Scaling Predictive Coding to 100+ Layer Networks"
4. **Salvatori et al. (2026)** - "A survey on neuro-mimetic deep learning via predictive coding"

### Code Repositories
- [Bogacz-Group/PredictiveCoding](https://github.com/Bogacz-Group/PredictiveCoding)
- [infer-actively/pypc](https://github.com/infer-actively/pypc)
- [BerenMillidge/PredictiveCodingBackprop](https://github.com/BerenMillidge/PredictiveCodingBackprop)

### VERSES AI
- [Benchmarking Predictive Coding Networks Made Simple](https://www.verses.ai/research-blog/benchmarking-predictive-coding-networks-made-simple)
- Karl Friston's work on active inference and free energy principle

## Risk Assessment

### High Risk Items
1. ❌ Building custom implementation from scratch (failed repeatedly before)
2. ❌ Using non-standard neuron designs
3. ❌ Output clamping strategies

### Low Risk Items
1. ✅ Using proven libraries
2. ✅ Following published architectures
3. ✅ Standard PC algorithm with μPC scaling

## Decision: Recommended Approach

**Use the Bogacz-Group/PredictiveCoding library** with the following configuration:
- 7 layers as specified above
- Their supervised learning PC variant
- Add μPC residual scaling if stability issues arise
- Extensively document results and compare to their 3-layer MNIST baseline

This minimizes risk while allowing us to learn from a working implementation before attempting any custom modifications.

## Next Steps

1. Review this proposal with team
2. Install and test chosen library
3. Run baseline MNIST experiment (their default architecture)
4. Modify to 7 layers
5. Document everything in MISTAKES.md as we go
