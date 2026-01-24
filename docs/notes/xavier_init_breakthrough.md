# Breakthrough: Xavier Initialization Solves Vanishing Activations

**Date:** 2026-01-24
**Session:** claude/learn-project-docs-TnvRZ

## Problem Diagnosis

After implementing correct inference dynamics, the network still showed:
- 30% error reduction, then plateau
- Higher layers frozen (~0% weight change)
- Only layer 0 learning

## Root Cause Found: Vanishing Activations

**Diagnostic Results** (with 0.01 weight initialization):
```
Layer states after inference:
  Layer 0: mean_abs=0.2461, max_abs=0.7588
  Layer 1: mean_abs=0.0224, max_abs=0.0626   (10x smaller)
  Layer 2: mean_abs=0.0022, max_abs=0.0068   (100x smaller!)
  Layer 3: mean_abs=0.0002, max_abs=0.0006   (1000x smaller!!)

Value errors for weight updates:
  Layer 0: value_error=8.82   (learning happens)
  Layer 1: value_error=0.07   (almost no learning signal)
  Layer 2: value_error=0.00   (no learning)
  Layer 3: value_error=0.00   (no learning)
```

**Why this happens:**
- Weights initialized with scale 0.01 (neuron.py lines 53, 58)
- Each layer computes: x_l = tanh(W_basal @ x_{l-1})
- With small weights and tanh activation, activations shrink exponentially
- Layer 2 receives input of magnitude 0.02, produces 0.002
- Layer 3 receives 0.002, produces 0.0002
- No error signal → no learning

## The Fix: Xavier Initialization

Changed weight initialization from:
```python
self.W_basal = nn.Parameter(
    torch.randn(num_neurons, basal_size, dtype=dtype) * 0.01
)
```

To:
```python
basal_scale = (1.0 / basal_size) ** 0.5  # Xavier: sqrt(1 / fan_in)
self.W_basal = nn.Parameter(
    torch.randn(num_neurons, basal_size, dtype=dtype) * basal_scale
)
```

For network with 100 neurons/layer:
- Old scale: 0.01
- Xavier scale: 1/sqrt(100) = 0.1 (10x larger!)

## Results After Xavier Init

**Layer activations:**
```
Layer states (with Xavier init):
  Layer 0: mean_abs=0.4679, max_abs=0.9688
  Layer 1: mean_abs=0.4047, max_abs=0.8686   ✓ Similar magnitude!
  Layer 2: mean_abs=0.3333, max_abs=0.7892   ✓ Similar magnitude!
  Layer 3: mean_abs=0.2499, max_abs=0.7792   ✓ Similar magnitude!

Value errors for weight updates:
  Layer 0: value_error=18.95   ✓ Strong signal
  Layer 1: value_error=18.61   ✓ Strong signal
  Layer 2: value_error=10.20   ✓ Strong signal
  Layer 3: value_error=9.38    ✓ Strong signal
```

All layers now have substantial activations and error signals!

## Performance Improvements

| Metric | Before (0.01 init) | After (Xavier init) |
|--------|-------------------|---------------------|
| Error reduction | 30% | 78% (with lr=0.0005) |
| Reconstruction error | -40% | -89% |
| Layer 0 weight change | +2.9% | +8.9% |
| Layer 1 weight change | -0.5% | ~0% |
| Layer 2-3 weight change | ~0% | ~0% |

**Note:** With lr=0.001, all layers show non-zero learning (+0.3% to +35%), but network becomes unstable.

## Learning Rate Sensitivity

Xavier init creates larger activations → larger gradients → need smaller learning rate

| Learning Rate | Result |
|--------------|--------|
| 0.005 | Diverges (+62,000% error!) |
| 0.001 | Unstable (oscillates after iter 30) |
| 0.0005 | Stable, 78% error reduction |

Song et al. (2022) use lr=0.001 for weights, suggesting 0.0005-0.001 is the right range.

## Remaining Issues

1. **Higher layers still barely learning** (with stable lr=0.0005)
   - Despite having strong error signals (9-19), weight changes are tiny
   - Might need: longer training, layer-specific learning rates, or different update rule

2. **Why temperature didn't help**
   - Temperature test showed noise HURTS performance (29.9% → 16.1%)
   - The "local minimum" wasn't actually a local minimum
   - It was vanishing activations preventing higher layers from activating at all!

## Key Insights

1. **Proper weight initialization is critical** for deep predictive coding networks
2. **The "local minimum" plateau was actually vanishing gradients** due to poor initialization
3. **Temperature/noise is NOT the solution** - fixing the root cause (initialization) is
4. **Inference dynamics were correct** - the bug was in initialization, not dynamics

## Files Modified

1. `src/network/neuron.py` - Xavier initialization for W_apical and W_basal
2. `src/network/backbone.py` - Added feedforward initialization before inference
3. `tests/test_network_diagnostics.py` - Reduced learning rate to 0.0005

## Next Steps

1. Investigate why layers 1-3 still barely learn despite strong error signals
2. Try layer-specific learning rates (higher for deeper layers?)
3. Run longer training (100-200 iterations instead of 50)
4. Check if weight update rule is correct for higher layers
5. Compare with Song et al. implementation more carefully

## References

- Xavier initialization: Glorot & Bengio (2010) "Understanding the difficulty of training deep feedforward neural networks"
- For tanh activation: Var(W) = 1/fan_in gives E[activation] ≈ E[input]
