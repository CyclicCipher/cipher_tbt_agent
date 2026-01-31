# Critical Stability Fixes - Summary

## Problem Diagnosed

The model was experiencing **catastrophic numerical explosion** within seconds:
- States exploding to ±10^33 (should be ~0-1)
- Hundreds of thousands of NaN warnings flooding the log
- Training completely unable to proceed

## Root Cause Analysis

**Multiplicative runaway feedback loop** between precision weighting and inference learning rate:

```
Old configuration:
  precision[layer2] = 100.0
  inference_lr = 0.1

  error = precision × raw_error = 100 × raw_error
  state_update = inference_lr × error = 0.1 × 100 × raw_error = 10 × raw_error
```

This created a **10x amplification** of raw errors, leading to:
1. State grows → larger prediction errors → larger weighted errors
2. Larger weighted errors → larger state updates → state explodes faster
3. Explosion → NaN → propagates through all layers

## Fixes Applied

### 1. Reduced Precision Values (categorical_network.py)
```python
# Old (VERSES):  [1.0, 10.0, 100.0]
# New (Stable):  [1.0,  2.0,   5.0]
```
- Layer 0: 1.0 (unchanged)
- Layer 1: 10.0 → 2.0 (5x reduction)
- Layer 2: 100.0 → 5.0 (20x reduction)

### 2. Reduced Inference Learning Rate (train_mnist_optimized.py)
```python
# Old: 0.1
# New: 0.01 (10x reduction)
```
Applied to both:
- `conv_inference_lr`
- `pc_inference_lr`

### 3. Added Error Clipping (categorical_network.py - update_state)
```python
max_error_magnitude = 10.0
error = torch.clamp(error, -max_error_magnitude, max_error_magnitude)
```
Prevents errors from growing beyond ±10.0 (typical range: 0.1-1.0)

### 4. Added State Explosion Detection
```python
if self.state.abs().max() > 1e6:
    print(f"CRITICAL: State exploded! Resetting.")
    self.state.zero_()
    return self.state
```
Circuit breaker that resets states if they exceed reasonable bounds

### 5. Added NaN/Inf Circuit Breakers
- Check errors BEFORE state update → early return if NaN/Inf
- Check state AFTER update → reset to zero if NaN/Inf
- Prevents NaN propagation through the network
- Single warning per layer (no spam)

## Net Effect

**Update magnitude reduction:**
```
Old: state_update = 0.1 × 100 × raw_error = 10.0 × raw_error
New: state_update = 0.01 × 5 × raw_error = 0.05 × raw_error

Reduction factor: 200x smaller updates
```

## Files Modified

1. `experiments/categorical_pc/categorical_network.py`
   - Error clipping in `update_state()`
   - State explosion detection
   - NaN/Inf circuit breakers
   - Reduced precision values

2. `experiments/categorical_pc/train_mnist_optimized.py`
   - Reduced inference_lr from 0.1 to 0.01
   - Updated hyperparameters display

3. `MISTAKES.md`
   - Documented catastrophic explosion
   - Added lessons learned about multiplicative hyperparameter effects

## Expected Behavior

The model should now:
- ✅ Run without immediate NaN explosion
- ✅ Maintain states in reasonable range (not 10^33)
- ✅ Print at most 3 critical warnings (one per layer) if instability occurs
- ✅ Automatically recover from transient numerical issues via state reset

## What to Watch For

1. **If model still doesn't learn** (accuracy stays at ~10%):
   - Check weight diagnostics → are weights updating? (std should be > 1e-8)
   - Check if all layers are resetting every iteration (too much clamping)
   - May need to adjust learning rates or error clipping threshold

2. **If you see "CRITICAL: State exploded" messages**:
   - Occasional resets are OK (self-recovery mechanism)
   - Frequent resets (every iteration) → need further tuning

3. **If you see NaN warnings**:
   - Should only see 1-3 warnings total (circuit breaker activates)
   - If you see thousands → something else is wrong

## Testing Recommendations

Run with current settings and check:
1. Does training proceed without explosion? ✓
2. Are weights updating? (check diagnostics at end of epoch)
3. Is accuracy improving above 10%? (the actual learning test)

If weights aren't updating or accuracy doesn't improve, next steps:
- Increase weight learning rates (currently 0.01 for inference, 0.001 for conv)
- Check if error clipping is too aggressive (try max_error = 50.0)
- Verify error injection is working correctly
