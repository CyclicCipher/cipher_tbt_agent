# Training Diagnostics - Why Test Accuracy Stayed at 14%

## Observed Behavior

```
Epoch | Train Acc | Test Acc | Gap
------|-----------|----------|-------
  1   |  37.67%   |  14.90%  | 22.77%
  2   |  55.50%   |  14.90%  | 40.60%
  3   |  60.08%   |  14.60%  | 45.48%
  4   |  66.43%   |  14.70%  | 51.73%
  5   |  66.48%   |  13.90%  | 52.58%
```

## Key Observations

1. **Train accuracy increases steadily** (37% → 66%)
2. **Test accuracy stays flat** (~14%, barely better than random 10%)
3. **Gap grows massively** (22% → 52%)

This is **severe overfitting**.

## Root Cause Analysis

### Hypothesis 1: Wrong Learning Algorithm ✓ LIKELY

**Problem:** Using backprop on a PC network
- PC inference iterations (10-20 steps) converge to equilibrium
- But we use `.data` to prevent gradient tracking
- Backprop only sees final equilibrium state
- Gradients are weak and don't capture the inference dynamics

**Evidence:**
- Train loss decreases slowly (2.25 → 1.88 over 5 epochs)
- Test loss actually INCREASES (2.28 → 2.87)
- This suggests learned weights are specific to training samples

**Fix:** Implement proper PC learning rules (local Hebbian updates)

### Hypothesis 2: Network Memorizing Training Set ✓ LIKELY

**Problem:** 6000 training samples, millions of parameters
- Conv layers: ~200K params
- PC layers: ~500K params (4-bit quantized)
- Ratio: 12 samples per parameter (underconstrained)

**Evidence:**
- Train accuracy keeps improving
- Test accuracy doesn't budge
- Classic overfitting signature

**Fix:**
- Regularization (dropout, weight decay)
- Data augmentation (random crops, rotations)
- More training data

### Hypothesis 3: Inference Not Converging ⚠️ POSSIBLE

**Problem:** Only 10 PC iterations might not be enough
- Network hasn't reached equilibrium
- Different convergence on train vs test

**Diagnostic:**
```python
# Track prediction error across iterations
errors = []
for iteration in range(50):  # Test longer
    error = compute_error()
    errors.append(error.mean().item())

# Plot convergence
plt.plot(errors)
# Should asymptote to near-zero
# If still decreasing at iteration 10, need more iterations
```

**Fix:** Increase num_iterations or use adaptive stopping

### Hypothesis 4: Conv Features Not Meaningful ⚠️ POSSIBLE

**Problem:** Conv preprocessor might extract poor features
- Random initialization
- Only trained via backprop through PC layers
- Weak gradients might not train conv properly

**Diagnostic:**
```python
# Visualize conv layer activations
with torch.no_grad():
    features = model.conv_preprocess(test_image)

# Check:
# - Are features diverse? (not all zeros/saturated)
# - Do similar images give similar features?
# - Can you see structure in feature maps?
```

**Fix:**
- Pre-train conv layers on reconstruction task
- Use larger learning rate for conv layers
- Initialize with pre-trained weights

### Hypothesis 5: Test Set Different Distribution ✗ UNLIKELY

**Problem:** MNIST train/test from different source

**Evidence against:**
- MNIST is well-curated
- Other models work fine on it
- 14% is only slightly better than random

**Verdict:** Not the issue

## Recommended Diagnostic Steps

### 1. Check Inference Convergence
```python
def diagnose_inference(model, image, num_iterations=50):
    errors = []
    for i in range(num_iterations):
        # Run one inference step
        error = model.step_inference(image)
        errors.append(error.mean().item())

    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Prediction Error')
    plt.title('Inference Convergence')
    plt.show()

    # Should see exponential decay to near-zero
    converged = errors[-1] < 0.01
    return converged
```

### 2. Check Weight Gradients (Current Backprop Approach)
```python
def diagnose_gradients(model, image, label):
    model.zero_grad()
    output = model(image)
    loss = F.cross_entropy(output, label)
    loss.backward()

    # Check gradient magnitudes
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: {grad_norm:.6f}")

            if grad_norm < 1e-6:
                print(f"  ⚠ WARNING: Vanishing gradient!")
            if grad_norm > 100:
                print(f"  ⚠ WARNING: Exploding gradient!")
```

### 3. Check Feature Quality
```python
def diagnose_features(model, train_loader, test_loader):
    # Extract features for train and test
    train_features = []
    train_labels = []

    with torch.no_grad():
        for data, target in train_loader:
            feat = model.conv_preprocess(data[0])
            train_features.append(feat.cpu())
            train_labels.append(target.item())

    # Same for test
    test_features = []
    test_labels = []
    # ...

    # Compute within-class vs between-class distances
    # Good features: within < between
    within_dist = compute_within_class_distance(train_features, train_labels)
    between_dist = compute_between_class_distance(train_features, train_labels)

    print(f"Within-class distance: {within_dist:.4f}")
    print(f"Between-class distance: {between_dist:.4f}")
    print(f"Ratio: {between_dist / within_dist:.4f}")

    # Should be > 1.5 for good features
```

### 4. Compare to Baseline
```python
# Train a simple CNN with backprop (no PC)
# On same data, same architecture (minus PC layers)
# This tells us if problem is:
# - PC specific (PC baseline worse than CNN)
# - Architecture specific (both fail)
# - Data specific (both succeed)
```

## Expected Findings

**If we implement proper PC learning:**
- Train accuracy: Still high (~60-80%)
- Test accuracy: Should improve to 70-90%
- Gap: Should narrow to <20%

**If problem is architecture:**
- Both PC and CNN baselines fail
- Need architectural changes (bigger conv layers, etc.)

**If problem is data:**
- Would need more samples or augmentation

## Next Steps

1. **Implement PC learning** (highest priority)
2. **Run diagnostics** to identify bottleneck
3. **Compare to CNN baseline** to isolate issue
4. **Iterate** on findings

## Success Criteria

- Test accuracy > 90% (your requirement)
- Generalization gap < 15%
- Learning is stable (monotonic improvement)
