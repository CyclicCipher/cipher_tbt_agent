# Predictive Coding Learning - Proper Implementation

## The Problem

We were using backpropagation (`.backward()`) which is:
- **Global credit assignment** - requires full computation graph
- **Not biologically plausible** - brain doesn't store activations for backprop
- **Incompatible with PC inference** - our iterative inference doesn't integrate well

## The Solution: Local Learning Rules

Predictive coding uses **local Hebbian-like rules** where each layer updates its own weights based on:
- Pre-synaptic activity (input)
- Post-synaptic error (prediction error)

### Phase 1: Inference (Correct - No Changes Needed)

```python
for iteration in range(num_iterations):
    # Compute predictions
    prediction = tanh(W_feedforward @ input)

    # Compute error
    error = state - prediction

    # Update state (minimize free energy)
    state.data -= lr_inference * error.data  # No gradient tracking
```

### Phase 2: Learning (NEEDS FIXING)

**Current (WRONG):**
```python
loss.backward()  # Backprop
optimizer.step()
```

**Correct:**
```python
def update_weights_pc(layer, input_below, error_this_layer, state_above=None):
    """
    Local predictive coding weight update.

    Hebbian rule: Δw ∝ pre × post_error
    """
    lr = 0.001

    # Feedforward weights: learn to predict current layer from input
    # Δw_ff ∝ input_below × error_this_layer
    delta_w_ff = lr * torch.outer(error_this_layer, input_below)
    layer.W_feedforward.data += delta_w_ff

    # Lateral weights: learn to predict from neighbors
    if layer.W_lateral is not None:
        delta_w_lat = lr * torch.outer(error_this_layer, layer.state)
        layer.W_lateral.data += delta_w_lat

    # Feedback weights: learn top-down predictions
    if layer.W_feedback is not None and state_above is not None:
        delta_w_fb = lr * torch.outer(error_this_layer, state_above)
        layer.W_feedback.data += delta_w_fb
```

### For Supervised Learning

For classification, we need a "clamping" mechanism for the output layer:

```python
def train_with_pc(model, image, label, num_iterations=20):
    """Train one sample using predictive coding."""

    # 1. Run inference to convergence
    model.forward(image, num_iterations=num_iterations)

    # 2. Clamp output layer to target
    # Create target representation (e.g., one-hot)
    target_state = torch.zeros(num_classes)
    target_state[label] = 1.0

    # 3. Run inference again with clamped output
    # This propagates error signal down the network
    for iteration in range(num_iterations):
        # Top layer: clamp to target
        error_top = model.layer2.state - target_state
        model.layer2.state.data = target_state  # Clamp

        # Middle layers: normal inference
        # (errors propagate down via feedback connections)
        # ... (same as before)

    # 4. Update weights using local rules
    update_weights_pc(
        model.layer0,
        input_below=image,
        error_this_layer=error_0,
        state_above=model.layer1.state
    )
    update_weights_pc(
        model.layer1,
        input_below=model.layer0.state,
        error_this_layer=error_1,
        state_above=model.layer2.state
    )
    update_weights_pc(
        model.layer2,
        input_below=model.layer1.state,
        error_this_layer=error_top,
        state_above=None
    )
```

## Key Principles

1. **No backprop** - no `.backward()`, no optimizer
2. **Local updates** - each layer uses only its own activity and neighbors
3. **Two-phase learning:**
   - Phase 1: Free-running inference (unsupervised)
   - Phase 2: Clamped inference (supervised signal)
4. **Hebbian** - weights strengthen when pre and post co-activate

## Advantages

- **Biologically plausible** - matches cortical learning
- **Memory efficient** - no computation graph storage
- **Parallelizable** - layers can update independently
- **Online learning** - one sample at a time

## Implementation Priority

1. Fix CanonicalMicrocircuit to track errors during inference
2. Implement local weight update functions
3. Remove backprop from training loop
4. Test on MNIST
5. Compare to backprop baseline

## References

- Whittington & Bogacz (2017) - "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity"
- Millidge et al. (2022) - "Predictive Coding: a Theoretical and Experimental Review"
