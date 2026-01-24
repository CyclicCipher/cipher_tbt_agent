# Optimizer Stability Findings and Solutions

**Date:** 2026-01-24

## Executive Summary

After extensive experiments with Muon and Adam optimizers, we've identified the root cause of learning instability: **saturation feedback loop**, not optimizer choice. Adam with lr=0.0001 is nearly stable (1.1x divergence), requiring only stronger saturation prevention to achieve full stability.

## Complete Experimental Results

### Muon Optimizer Experiments

| Momentum | LR    | Final Error | Divergence | Weight Δ | Saturation |
|----------|-------|-------------|------------|----------|------------|
| 0.95     | 0.0005| 13,369      | 14.1x      | +100%    | 70→98%     |
| 0.70     | 0.0005| 6,749       | 7.8x       | +76%     | 78→98%     |
| 0.50     | 0.0005| 6,867       | 6.9x       | +70%     | 83→99%     |

**Pattern:** All find minimum ~95 at iter 30-40, then diverge due to momentum overshoot.

**Conclusion:** Muon's momentum in neuron space conflicts with local Hebbian learning rules.

### Adam Optimizer Experiments

| LR     | Final Error | Divergence | Weight Δ | Saturation | Min Error (iter) |
|--------|-------------|------------|----------|------------|------------------|
| 0.001  | 4,139       | 4.5x       | +41%     | 91→97%     | 440 (10)         |
| 0.0001 | 1,125       | **1.1x**   | +14%     | 72→99%     | 400 (94)         |

**Pattern:** Both reach good minima but error climbs back due to saturation feedback loop.

**Conclusion:** Adam with lr=0.0001 is nearly stable - only needs saturation fix.

### Baseline (Manual GD)

- LR: 0.0005, weight decay: 0.01
- Result: 920 → 0.16 (iter 320) → 78 (drift)
- Wild oscillations but eventually finds solution
- Unreliable, unpredictable

## Root Cause: Saturation Feedback Loop

### The Mechanism

1. **Learning works initially**
   - Error drops: 1002 → 400 (60% reduction!)
   - Network finds good predictive weights

2. **Saturation begins**
   - Strong weights → high activations
   - tanh(x) approaches ±1 for large x
   - Saturation rate: 72% → 99%

3. **Feedback loop engages**
   - Hebbian rule: `ΔW = lr * error * activation`
   - High activations → large weight updates
   - Larger weights → even higher activations
   - Positive feedback: saturation begets more saturation

4. **Gradient vanishing**
   - Saturated neurons: tanh'(x) → 0
   - Gradients vanish for updating those neurons
   - But they still contribute high activations to Hebbian updates
   - System stuck in high-activation state

5. **Error climbs back**
   - Over-saturated network loses representational capacity
   - Can't adjust to minimize prediction errors
   - Error rises: 400 → 1,125

### Why Current Saturation Penalty Fails

Current implementation:
```python
if saturation_rate > 0.1:
    layer.neurons.W_basal.grad += saturation_penalty * W_basal.sign()
    # saturation_penalty = 0.01
```

**Problem:** Penalty adds L1-like regularization (±0.01 per weight), but:
- Hebbian updates are much larger: `lr * error * activation`
- With activation ≈ 1.0, error ≈ 20, lr = 0.0001: update ≈ 0.002
- Penalty 0.01 vs update 0.002 seems ok, but...
- Penalty only applies when saturation > 10%
- By then, 99% of neurons saturated - too late!

**Root issue:** Reactive instead of proactive. Needs to prevent saturation, not penalize after it happens.

## Solution: Activity Clipping

### Approach 1: Hard Clipping (Simplest)

Clip activations before they saturate:

```python
def _inference_step(self):
    # ... compute new_state ...

    # Prevent saturation: clip to safe range
    new_state = new_state.clamp(-0.85, 0.85)

    layer.state.copy_(new_state)
```

**Pros:**
- Simple, guaranteed to prevent saturation
- No hyperparameters to tune
- Stops feedback loop immediately

**Cons:**
- Non-differentiable at boundaries (minor issue for inference)
- Artificial constraint on representations

**Expected result:** Stable learning, error stays near minimum

### Approach 2: Soft Saturation Penalty (Proactive)

Penalize activations as they approach saturation:

```python
def _add_activity_regularization(self, activations):
    for i, act in enumerate(activations):
        # Penalty grows quadratically near saturation
        # penalty = λ * (|x| - threshold)^2 for |x| > threshold
        threshold = 0.7
        over_threshold = (act.abs() - threshold).clamp(min=0)
        penalty = 0.1 * (over_threshold ** 2).mean()

        # Add to gradients
        penalty_grad = 0.1 * 2 * over_threshold * act.sign()
        if layer.neurons.W_basal.grad is not None:
            # Distribute penalty across weights
            layer.neurons.W_basal.grad += penalty_grad.unsqueeze(1) * 0.01
```

**Pros:**
- Differentiable, smooth
- Acts before saturation is severe
- Allows high activations when needed

**Cons:**
- More complex
- Requires tuning (threshold, penalty strength)

### Approach 3: Batch Normalization

Normalize layer activations:

```python
class PredictiveCodingLayer(nn.Module):
    def __init__(self, ...):
        # ...
        self.batch_norm = nn.BatchNorm1d(num_neurons, affine=False)

    def normalize_state(self):
        # Normalize to mean=0, std=1
        self.state.copy_(self.batch_norm(self.state.unsqueeze(0)).squeeze(0))
```

**Pros:**
- Standard technique, well-understood
- Automatically maintains healthy activation distribution
- Can use running statistics for inference

**Cons:**
- Batch norm designed for batched data (we use single samples)
- May conflict with predictive coding dynamics
- Adds complexity

## Recommendation

### Immediate: Hard Clipping

Start with hard clipping (Approach 1) for immediate stability:

```python
# In backbone.py, _inference_step()
new_state = new_state.clamp(-0.85, 0.85)
```

**Why this first:**
- Simplest to implement (1 line)
- Guaranteed to work
- Gets us to stable 400-iteration runs quickly
- Can iterate from stable baseline

**Expected result:** Error stays near minimum (~400-500), no divergence

### If Clipping Works

1. **Test on math curriculum** - Can we learn sequences?
2. **Add temporal patterns** - Simple recurrence for sequential learning
3. **Optimize if needed** - Try softer approaches (Approach 2) for more expressiveness

### If Clipping Doesn't Fully Work

Try combining approaches:
```python
# Soft clip during inference
new_state = 0.95 * torch.tanh(new_state / 0.95)  # Smoother than hard clip

# Plus proactive penalty during weight updates
penalty_grad = 0.1 * (act.abs() - 0.7).clamp(min=0) * act.sign()
```

## Why This Will Work

### Evidence from Low-LR Adam Test

With lr=0.0001:
- Error dropped 60%: 1002 → 400 ✓
- Learning worked until saturation hit
- Divergence only 1.1x (nearly stable)
- Weight change only +14% (moderate)

**Interpretation:** The learning algorithm is sound. Optimizer is good. Only issue is saturation.

### Theoretical Justification

Predictive coding networks **should** have distributed representations:
- Sparse codes (few neurons active per input)
- No single neuron should saturate for all inputs
- tanh activation: designed to squash values to [-1, 1] but should center around 0

Current behavior (99% saturation) is pathological, not fundamental to the architecture.

### Comparison to Biology

Cortical neurons:
- Fire at 1-10 Hz baseline, up to 100 Hz max
- Dynamic range: 10-100x
- NOT saturated at ceiling constantly

Our neurons (99% saturated):
- Equivalent to all neurons firing at max rate
- No dynamic range left
- Can't encode differences

Clipping restores healthy dynamics.

## Implementation Priority

1. **Implement hard clipping** (today)
   - Modify `_inference_step()` in `backbone.py`
   - Add `new_state = new_state.clamp(-0.85, 0.85)`

2. **Run 400-iteration test with clipping** (today)
   - Use Adam lr=0.0001 (proven nearly-stable base)
   - Expect: error ~400-500 throughout, no divergence

3. **If stable: Add temporal patterns** (next)
   - Simple recurrence (state → state connections)
   - Test on sequence prediction

4. **Begin math curriculum** (goal)
   - Arithmetic (2+3=5)
   - Pattern recognition (Fibonacci)
   - Algebraic rules (commutativity, distributivity)
   - Simple calculus (derivatives)

## Next Steps

**Immediate:**
```bash
# Edit backbone.py to add clipping
# Run: python tests/test_adam_with_clipping.py
# If stable (error stays <1000): Celebrate! Move to temporal patterns
```

**After stability proven:**
1. Document clipping threshold choice (why 0.85?)
2. Add temporal recurrence for sequential learning
3. Design math curriculum experiments
4. Test catastrophic forgetting prevention

## Appendix: Why Not Earlier?

**Why didn't we try clipping immediately?**

We needed to systematically rule out optimizer issues first:
- Muon: Does momentum in neuron space help? → No, makes it worse
- Adam standard LR: Does adaptive LR help? → Helps, but not enough
- Adam low LR: Does lower LR prevent overshoot? → Yes! Nearly stable

Only by trying Adam at low LR (1.1x divergence) could we confirm the optimizer is good and isolate the saturation issue.

**Lesson:** Sometimes the solution is simple (clipping), but you need experiments to know it's the right solution.

## Confidence Level

**High confidence** (90%) that hard clipping will achieve stable 400-iteration runs:
- Adam lr=0.0001 already nearly stable (1.1x divergence)
- Saturation clearly identified as remaining issue
- Clipping directly prevents saturation feedback loop
- Simple intervention, low risk

**Medium confidence** (60%) that clipping alone is sufficient long-term:
- May need softer approaches for complex tasks
- Math curriculum may reveal other issues
- But clipping gives us stable foundation to iterate from

**Low confidence** (20%) that we'll need fundamentally different architecture:
- Learning works! (60% error reduction achieved)
- Just need to prevent pathological saturation
- Current architecture design is sound
