# Performance Issues Diagnosis

## The Problem

Training is **extremely slow** and consuming excessive RAM, forcing program termination.

## Root Causes

### 1. **Excessive Iterations Per Sample** (CRITICAL)

Current configuration (train_mnist.py:235-237):
```python
num_conv_iterations=20,
num_inference_iterations=20,
```

**Impact:**
- Each sample requires 20 conv iterations × 3 PC conv layers = **60 conv updates**
- Plus 20 inference iterations × 3 PC inference layers = **60 inference updates**
- **Total: 120+ forward/backward passes per sample**

For 6000 samples per epoch:
- **720,000 operations per epoch minimum**
- At 1 second per sample = **100 minutes per epoch** = 8+ hours for 5 epochs!

### 2. **Precision Computation Overhead**

Every iteration calls `compute_error()` which:
```python
current_var = raw_error.var(dim=(0, 2, 3), keepdim=False).mean()  # Expensive!
```

This computes variance across batch and spatial dimensions **on every iteration**:
- 20 iterations × 3 conv layers = **60 variance computations per sample**
- For 100×100 input → 50×50 → 25×25 → 13×13 feature maps
- Each variance computation processes thousands of values

### 3. **Memory Accumulation**

Potential issues:
- State tensors for each layer accumulating in memory
- Gradient history not being cleared between samples
- Curriculum manager caching predictions

## Quick Fixes

### **IMMEDIATE: Reduce Iterations**

Edit `train_mnist.py` line 235-237:

```python
# BEFORE (SLOW):
num_conv_iterations=20,
num_inference_iterations=20,

# AFTER (FASTER):
num_conv_iterations=3,
num_inference_iterations=5,
```

**Expected speedup:** 4-6x faster

### **MEDIUM: Reduce Precision Computation Frequency**

Only compute precision every N iterations instead of every iteration:

```python
# In categorical_network.py compute_error():
if self.iteration_count % 5 == 0:  # Only every 5 iterations
    current_var = raw_error.var(dim=(0, 2, 3), keepdim=False).mean()
    self.error_variance.mul_(0.9).add_(current_var * 0.1)
    stable_var = torch.clamp(self.error_variance, min=1e-4, max=1e2)
    self.precision.copy_(1.0 / stable_var)
```

**Expected speedup:** 20-30% faster

### **LONG-TERM: Batch Processing**

Process multiple samples in parallel instead of one at a time:
- Current: batch_size=1 (sequential)
- Better: batch_size=32 (parallel on GPU)

## Run the Profiler

Before making changes, run:

```bash
cd experiments/categorical_pc
python profile_training.py
```

This will show:
- Exact time per iteration
- Memory usage per step
- Estimated epoch time
- Bottleneck identification

## Expected Reasonable Performance

With optimized settings (3/5 iterations):
- **Per sample:** 1-5 seconds
- **Per epoch (6000 samples):** 10-30 minutes
- **Total training (5 epochs):** 1-2.5 hours

Current settings (20/20 iterations) would take **8+ hours** for 5 epochs!

## Recommended Test Configuration

Start with minimal settings to verify everything works:

```python
# Fast testing config
num_conv_iterations=1,
num_inference_iterations=3,
train_size = 100  # Instead of 6000
```

Once it works, gradually increase:
1. Test with 100 samples, 1/3 iterations → should complete in ~1-2 minutes
2. Test with 1000 samples, 3/5 iterations → should complete in ~10-20 minutes
3. Full training: 6000 samples, 5/10 iterations → should complete in ~1-2 hours

## Why Iterations Matter

Each iteration does:
1. Compute prediction from neighboring layers
2. Compute error (with variance calculation!)
3. Update state via gradient descent
4. Repeat for all layers

With 20 iterations × 3 layers = **60 operations** that could be done in **9 operations** (3 iterations × 3 layers).

## Next Steps

1. **Run profiler:** `python profile_training.py`
2. **Reduce iterations:** Change to 3/5 in train_mnist.py
3. **Test with small dataset:** Use 100 samples first
4. **Monitor memory:** Watch RAM/GPU usage
5. **Gradually scale up:** Only increase iterations if accuracy demands it
