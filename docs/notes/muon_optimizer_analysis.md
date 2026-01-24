# Muon Optimizer for Predictive Coding Networks

**Date:** 2026-01-24

## Can Muon Work with Our Two-Compartment Neurons?

### What is Muon?

Muon is a new optimizer designed by the Mamba team (2024) that combines:
- **Momentum in neuron space** (like Adam in weight space)
- **Orthogonalization** (prevents gradient interference)
- **Designed for transformers** but generalizes to any architecture

Key insight: Momentum should be in the space that matters (output activations, not weights)

### Our Neuron Architecture

**Standard perceptron:**
```python
output = W @ input
```

**Our two-compartment neuron:**
```python
apical_activity = tanh(W_apical @ apical_input)
basal_activity = tanh(W_basal @ basal_input)
output = gate * apical_activity + (1 - gate) * basal_activity
```

**Learnable parameters:**
- `W_apical` (num_neurons, apical_size)
- `W_basal` (num_neurons, basal_size)
- `gate` (num_neurons,)

### Compatibility Analysis

**✓ MUON SHOULD WORK!**

Reasons:
1. **Differentiable:** All operations are differentiable
2. **PyTorch parameters:** Uses `nn.Parameter`, compatible with all optimizers
3. **Standard shapes:** Just matrices and vectors, nothing exotic
4. **No special constraints:** Gate is clamped, but that's post-update

**Potential issues:**
1. **Gate parameter has different scale** than weights
   - Solution: Use parameter groups with different learning rates
   ```python
   optimizer = Muon([
       {'params': [W_apical, W_basal], 'lr': 0.01},
       {'params': [gate], 'lr': 0.001}  # Smaller LR for gate
   ])
   ```

2. **Two-pathway structure might benefit from custom logic**
   - But not required - Muon treats all parameters equally

### How to Integrate Muon

**Option 1: Direct replacement (simplest)**
```python
import torch
from muon import Muon  # Hypothetical import

# In backbone.py:
def __init__(self, ...):
    # ... create layers ...
    self.optimizer = Muon(self.parameters(), lr=0.01)

def update_weights(self):
    # Compute gradients using current manual method
    for layer in self.layers:
        # ... compute errors ...
        error_col = layer_error.unsqueeze(1)

        # Set gradients manually (since we're not using autograd)
        layer.neurons.W_basal.grad = -(error_col @ input_from_below.unsqueeze(0))
        layer.neurons.W_apical.grad = -(error_col @ input_from_above.unsqueeze(0))
        # Note: Negative because we're maximizing (error reduction)

    # Muon handles momentum, orthogonalization, updates
    self.optimizer.step()
    self.optimizer.zero_grad()
```

**Option 2: Hybrid approach**
```python
# Use Muon for weights, manual for gate
self.weight_optimizer = Muon([W_apical, W_basal], lr=0.01)
# Gate updated manually with custom logic (gate movement based on accuracy)
```

### Comparison: Muon vs Adam vs Custom Optimizer

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Muon** | - Momentum in neuron space<br>- Orthogonalization<br>- State-of-the-art for transformers | - New, less tested<br>- Designed for different architecture | Large-scale, when gradient interference is an issue |
| **Adam** | - Well-tested<br>- Adaptive per-param LR<br>- Momentum + RMSprop | - Momentum in weight space<br>- Can be suboptimal | General-purpose, safe default |
| **Custom** | - Tailored to predictive coding<br>- Can incorporate bio constraints<br>- Full control | - Time to develop<br>- Need to test extensively<br>- Might reinvent wheel | If standard optimizers fail, or for research |

### Recommendation for Predictive Coding

**Start with Adam** (Priority 1):
- Proven, reliable
- Will definitely solve the oscillation problem
- Easy to implement

**Then try Muon** (Priority 2):
- If Adam works but could be better
- Especially if scaling to larger networks
- May handle two-pathway structure better

**Custom optimizer** (Priority 3):
- Only if both Adam and Muon have issues
- Or for research contributions

### Custom Optimizer Ideas for Predictive Coding

If we do build custom:

**1. Separate dynamics for apical vs basal:**
```python
# Apical weights: slower learning (stable predictions)
W_apical += lr_apical * gradient_apical

# Basal weights: faster learning (adapt to data)
W_basal += lr_basal * gradient_basal
```

**2. Error-modulated learning rate:**
```python
# Learn faster when error is high
adaptive_lr = lr * (1 + error.abs().mean())
```

**3. Biological constraints:**
```python
# Dale's law: weights can't change sign (excitatory vs inhibitory)
W_excitatory = W_excitatory.clamp(min=0)
W_inhibitory = W_inhibitory.clamp(max=0)

# Homeostatic plasticity: maintain target firing rate
activity_target = 0.2  # 20% neurons active
activity_current = state.abs().mean()
lr_adjusted = lr * (activity_target / activity_current)
```

**4. Inference-aware updates:**
```python
# Only update weights that contributed to current inference
# (sparse credit assignment)
active_mask = (state.abs() > threshold).float()
gradient = gradient * active_mask
```

## Conclusion

**Muon is compatible with our neurons!** The two-compartment structure doesn't prevent its use - it's still just differentiable PyTorch parameters.

**Recommended path:**
1. Implement Adam first (1 hour)
2. Test on 400-iteration runs (see if oscillations disappear)
3. If satisfied, done! If not, try Muon (1-2 hours)
4. Custom optimizer only if needed for research contributions

The key insight: **Our neurons are more complex than perceptrons, but optimizers work at the parameter level, not the neuron level.**
