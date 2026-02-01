# CRITICAL BUG FIX - Optimizer Parameter Separation

## What Was Wrong

Your training failure (stuck at 8.35% accuracy) was caused by a **catastrophic bug** in how the weight optimizer was created.

### The Bug (pc_trainer.py:55-57)

```python
# WRONG - includes value nodes!
self.optimizer_p = optimizer_p_fn(
    self.model.parameters(),  # Returns ALL parameters
    lr=weight_lr
)
```

### Why This Destroyed Learning

`model.parameters()` returns:
1. Linear layer weights/biases ✓ (should be optimized by optimizer_p)
2. **PCLayer `_x` value nodes** ✗ (should ONLY be optimized by optimizer_x)

This meant value nodes received **conflicting updates from two optimizers**:

**During inference (t=0 to 34):**
- `optimizer_x.step()` updates `_x` to minimize prediction errors ✓

**During learning (t=34):**
- `optimizer_p.step()` ALSO updates `_x` ✗ ← CONTRADICTS INFERENCE

The value nodes got pulled in opposite directions, the energy landscape became incoherent, and **no learning could occur**.

### The Training Process (How PC Should Work)

**For each batch of 64 images:**

1. **Inference Phase** (35 iterations):
   - Show batch to network
   - Initialize value nodes `x[layer]` from forward pass
   - For t = 0 to 34:
     - Compute: free_energy = cross_entropy_loss + sum(prediction_errors)
     - Backward through computational graph
     - `optimizer_x.step()` ← Update ONLY value nodes
     - Weights stay frozen
   - Value nodes converge to minimize free energy for this batch

2. **Learning Phase** (1 update):
   - `optimizer_p.step()` ← Update ONLY weights (Linear params)
   - Value nodes stay frozen
   - Weights learn to make predictions that reduce future errors

**Key principle:** Value nodes and weights are optimized **in separate phases**, never simultaneously.

**What our bug did:** Optimized them simultaneously, causing chaos.

## The Fix

### Added Method to PCNetwork (pc_layer.py)

```python
def get_network_parameters(self):
    """Get only network parameters (weights/biases), excluding value nodes."""
    # Get all value nodes to exclude
    value_node_set = set(self.get_value_nodes())

    # Yield only parameters that are NOT value nodes
    for param in self.parameters():
        if not any(param is x for x in value_node_set):
            yield param
```

### Updated Trainer (pc_trainer.py)

```python
# CORRECT - excludes value nodes
self.optimizer_p = optimizer_p_fn(
    self.model.get_network_parameters(),  # Only Linear weights/biases
    lr=weight_lr
)
```

### Fixed Test (tests/test_pc_basic.py)

- Moved to `tests/` directory (as requested)
- Fixed gradient flow test to compare network parameters only
- Does forward pass first to initialize all parameters before comparison

## Why Standard Batching Works

You asked: "Shouldn't we train on one example until high accuracy before moving to the next?"

**No.** That's not how batch gradient descent works:

**Standard training:**
- Show batch of 64 examples
- Compute average gradient
- Take small step in weight space
- Show next batch (different examples)
- Repeat for many epochs

**Why this works:**
- Each example contributes a gradient direction
- Averaged gradients point toward minima that work for MANY examples
- Small learning rate prevents overfitting to single examples
- Seeing diverse examples prevents memorization

**If we trained on single examples to convergence:**
- Network would memorize each example
- No generalization to unseen data
- Catastrophic forgetting (learning example 2 destroys knowledge of example 1)
- This is NOT how neural networks learn

**MLPs can achieve ~97-98% on MNIST** with proper training. They're not ideal (CNNs are better), but they work for this simple task.

## Expected Results After Fix

With the optimizer bug fixed, you should see:

**Epoch 1:**
- Train accuracy: 70-80%
- Test accuracy: 75-85%

**Epoch 5:**
- Train accuracy: 95-97%
- Test accuracy: 94-96%

**Epoch 10:**
- Train accuracy: 97-98%
- Test accuracy: 95-97%

Loss should steadily decrease, not stay flat.

## Next Steps

1. **Rerun training:**
   ```bash
   python train_mnist_pc.py
   ```

2. **It should now learn properly** - you'll see accuracy increasing each epoch

3. **If it still doesn't work:** Check diagnostics plots for:
   - Inference convergence (should decrease)
   - Layer energies (should be stable)
   - Report back with new plots

## About MLPs vs CNNs

You're right that we're using a plain MLP (Multi-Layer Perceptron). Here's why:

**MLPs for MNIST:**
- ✓ Can achieve ~97-98% accuracy
- ✓ Simple architecture
- ✓ What Bogacz papers use for PC research
- ✗ Many parameters (200k+)
- ✗ No translation invariance
- ✗ Treats each pixel position independently

**CNNs are better because:**
- Shared weights (fewer parameters)
- Translation invariance (recognizes digits anywhere)
- Spatial structure (local features)
- ~99%+ accuracy on MNIST

**For our project:**
- We're doing **retinal preprocessing** (center-surround, movement detection)
- This extracts features BEFORE the network
- So an MLP operating on preprocessed features might be sufficient
- Later we can add convolutions if needed

**Transformers:**
- Designed for sequences (text, time series)
- Use attention mechanisms
- Overkill for MNIST
- Not biologically plausible (our goal)

## Root Cause Analysis

**What happened:**
1. I implemented PC algorithm from literature
2. Bogacz code has `get_model_parameters()` that filters value nodes
3. I skimmed over this, didn't understand why it was needed
4. Used `model.parameters()` directly (seemed simpler)
5. Created optimizer that conflated two parameter types
6. Learning failed completely

**Lesson learned:**
- Every detail in reference implementations has a reason
- "Simplifying" without understanding causes bugs
- Parameter filtering is CRITICAL in PC networks
- Test on small example before full training

**Now documented in MISTAKES.md as mistake #7.**
