# Temporal Pattern Learning Status

## Current Implementation: NO Temporal Patterns

**From neuron.py (lines 1-5):**
```python
"""
Two-compartment neuron for predictive coding.

Implements a simplified version of the two-compartment neuron architecture
described in the planning document. Temporal convolution deferred to Phase 3.
```

**Current neurons are MEMORYLESS:**
- No temporal convolution kernels
- No frame buffer
- No recurrent connections
- Each forward pass is independent

**What we have:**
- Spatial processing (within current input)
- Hierarchical processing (across layers)
- ❌ NO temporal processing (across time steps)

## What Was Planned (Phase 3)

**From ARCHITECTURE.md:**
- Temporal convolution kernels in compartments
- Frame buffer for temporal context
- Process sequences of inputs over time

**For math curriculum, we need temporal processing because:**
- Math problems are sequences: "2 + 3 = ?" requires processing tokens in order
- Solutions require multi-step reasoning
- Need to remember intermediate results

## Options for Adding Temporal Patterns

### Option 1: Simple Recurrent Connections (Easiest)
```python
# Add hidden state that persists across time
self.hidden_state = torch.zeros(num_neurons)

# Update each time step:
new_state = f(current_input, self.hidden_state)
self.hidden_state = new_state
```

### Option 2: LSTM/GRU-style Gates
```python
# Forget gate: what to keep from previous time
forget = sigmoid(W_forget @ [input, hidden])
# Input gate: what to add from current input
input_gate = sigmoid(W_input @ [input, hidden])
# Update hidden state
hidden = forget * hidden + input_gate * tanh(W @ input)
```

### Option 3: Temporal Convolution (Most Powerful)
```python
# Convolve over time window
# [t-2, t-1, t] → kernel → prediction at t+1
output = conv1d(input_sequence, temporal_kernel)
```

### Option 4: Transformer-style Attention (State-of-the-art)
```python
# Self-attention over sequence
# Each position attends to all previous positions
attention_weights = softmax(Q @ K.T / sqrt(d))
output = attention_weights @ V
```

## Recommendation for Math Curriculum

**Start with Option 1 (Simple Recurrence):**
- Minimal code change
- Test if temporal patterns help
- Can upgrade later if needed

**Implementation:**
```python
class RecurrentBackbone(BackboneNetwork):
    def __init__(self, ...):
        super().__init__(...)
        # Add hidden state buffer for each layer
        for layer in self.layers:
            layer.register_buffer('hidden', torch.zeros(layer.num_neurons))

    def forward(self, sensory_input, num_iterations=50):
        # Incorporate previous hidden state
        for i, layer in enumerate(self.layers):
            # Mix current state with previous hidden
            current = layer.get_state()
            previous = layer.hidden

            # Temporal integration (simple weighted average)
            layer.state = 0.7 * current + 0.3 * previous
            layer.hidden = layer.state.clone()

        # Run normal inference
        super().forward(sensory_input, num_iterations)
```

**For math specifically:**
- Process one token at a time: "2" → "+" → "3" → "=" → "?"
- Hidden state carries context (we've seen "2 +")
- Final prediction uses full sequence context

## Timeline

**Immediate (this session):**
1. Implement Muon optimizer
2. Add saturation penalty
3. Test 400 iterations

**Next session:**
1. Add simple recurrence (Option 1)
2. Test on sequence task (predict next number: 1,2,3,?,5)
3. If works: proceed to math curriculum

**Phase 3 (future):**
1. Full temporal convolution (as planned)
2. Attention mechanisms
3. Long-range dependencies

The good news: Simple recurrence is easy to add and should help with math!
