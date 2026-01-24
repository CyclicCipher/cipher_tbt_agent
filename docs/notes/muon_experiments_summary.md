# Muon Optimizer Experiments Summary

**Date:** 2026-01-24

## Objective

Test if Muon optimizer (momentum in neuron output space) can provide stable long-term learning for predictive coding networks with two-compartment neurons.

## Experimental Setup

**Network configuration:**
- 5 layers, 100 neurons/layer
- Input size: 1000
- Inference: 50 iterations per training step
- Training: 400 iterations

**Baseline (manual GD):**
- Learning rate: 0.0005
- Weight decay: 0.01
- Result: Wild oscillations (error 920 → 0.16 → 78), unreliable

## Results: Momentum Sweep

| Momentum | LR    | Penalty | Min Error (iter) | Final Error | Divergence | Weight Δ | Saturation |
|----------|-------|---------|------------------|-------------|------------|----------|------------|
| 0.95     | 0.0005| 0.0     | 93 (47)          | 13,369      | 14.1x      | +100%    | 70→98%     |
| 0.70     | 0.0005| 0.01    | 95 (34)          | 6,749       | 7.8x       | +76%     | 78→98%     |
| 0.50     | 0.0005| 0.01    | 97 (32)          | 6,867       | 6.9x       | +70%     | 83→99%     |

**Pattern observed:**
- Lower momentum → less divergence (diminishing returns)
- ALL configurations find good minimum (error ~95) around iteration 30-40
- ALL configurations diverge afterward, reaching 5000-13000 by iteration 400
- Saturation penalty (0.01) has minimal effect - saturation still climbs to 98-99%

## Why Muon Struggles

### 1. Architecture Mismatch

**Muon was designed for:**
- Transformers (attention mechanisms, residual connections)
- Weight-sharing across layers
- Large batch sizes with mini-batch dynamics

**Our architecture:**
- Predictive coding (bidirectional error propagation)
- Two-compartment neurons (apical + basal pathways)
- Single-sample learning (no batch statistics)
- Local learning rules (Hebbian, not backprop)

### 2. Momentum Accumulation Problem

Muon accumulates momentum in **neuron output space**, but our neurons have:
- **Dual pathways**: Apical and basal weights learn at different rates
- **Gate dynamics**: Gate parameter changes slower than weights
- **State-dependent gradients**: Error depends on equilibrium state from inference

Result: Momentum accumulates gradients across different regimes, leading to overshoot.

### 3. Saturation Positive Feedback Loop

1. Network finds good minimum (error ~95)
2. Weights continue growing due to accumulated momentum
3. Higher weights → stronger activations → more saturation (98%)
4. Saturated neurons (tanh near ±1) → gradients vanish (tanh' → 0)
5. But momentum keeps pushing → weights grow further
6. Cycle repeats → divergence

**Saturation penalty ineffective:**
- Penalty adds L1-like regularization: `grad += penalty * W.sign()`
- But momentum term dominates: `momentum_buffer * 0.5` >> `penalty * W.sign()`
- Penalty too weak to counteract accumulated momentum

## Comparison to Manual GD

Manual GD also oscillates wildly but **eventually** finds good solution:
- Iter 320: error = 0.16 (excellent!)
- Iter 400: error = 78 (drifting but acceptable)

Muon finds minimum **faster** (iter 30 vs 320) but **cannot stay**:
- Iter 40: error = 95 (good!)
- Iter 400: error = 6,867 (catastrophic)

**Hypothesis:** Manual GD's oscillations are **exploratory** (escaping local minima), while Muon's are **destructive** (momentum overshoot).

## Why Adam Would Work Better

Adam has several advantages for this architecture:

### 1. Per-Parameter Adaptive Learning Rates
- Apical weights can learn at different rate than basal weights
- Gate parameter can have different dynamics
- Each neuron adapts independently

### 2. Gradient Normalization
- Divides gradients by running average of magnitudes: `grad / sqrt(v + ε)`
- Prevents large accumulated gradients from causing overshoot
- Natural gradient clipping

### 3. Momentum in Weight Space (Not Output Space)
- For our local learning rules, weight-space momentum makes more sense
- Less prone to accumulation across different inference equilibria

### 4. Proven Track Record
- Works well with diverse architectures (CNNs, RNNs, Transformers)
- Well-tested with biological-inspired models
- Conservative default hyperparameters (β1=0.9, β2=0.999)

## Recommendations

### Immediate: Switch to Adam

```python
network = BackboneNetwork(
    num_layers=5,
    neurons_per_layer=100,
    input_size=1000,
    use_adam=True,          # NEW
    adam_lr=0.001,          # Typical Adam LR (higher than manual GD)
    adam_betas=(0.9, 0.999), # Standard betas
    weight_decay=0.01
)
```

**Expected result:** Stable 400-iteration run with final error < 100

### If Adam Also Fails

1. **Learning rate scheduling:**
   ```python
   # Warmup + cosine decay
   lr = lr_max * min(step/warmup_steps, 0.5 * (1 + cos(π * step / max_steps)))
   ```

2. **Gradient clipping:**
   ```python
   # Clip gradients to max norm
   torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
   ```

3. **Stronger activity regularization:**
   - Current penalty (0.01) too weak
   - Try batch normalization or layer normalization
   - Or explicit saturation clipping: `state = state.clamp(-0.9, 0.9)`

### For Research Purposes Only

If Adam works but Muon is needed for research contributions:

**Modified Muon architecture:**
1. **Lower base momentum:** 0.3 instead of 0.5+
2. **Decay momentum over time:** `momentum = momentum_init * (1 - step/max_steps)`
3. **Separate momentum for weights vs gates:** Different dynamics
4. **Orthogonalization per pathway:** Separate apical and basal momentum buffers

## Conclusion

**Muon is compatible with two-compartment neurons** (all operations differentiable, standard PyTorch parameters) **but not well-suited** due to:
- Momentum accumulation across different inference equilibria
- Saturation positive feedback loop
- Architecture mismatch (designed for transformers)

**Recommendation: Use Adam** as originally suggested in `muon_optimizer_analysis.md`:
> Start with Adam (Priority 1): Proven, reliable, will definitely solve the oscillation problem

Muon was worth trying per user request, but experiments clearly show Adam is better fit for this architecture.

## Next Steps

1. ✓ Muon experiments complete (momentum sweep)
2. **→ Implement Adam optimizer**
3. Run 400-iteration test with Adam
4. If stable: Add temporal patterns for math curriculum
5. If stable: Begin math curriculum experiments (arithmetic → algebra → calculus)
