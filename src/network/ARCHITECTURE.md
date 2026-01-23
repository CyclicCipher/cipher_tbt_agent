# Network Architecture - Phase 2: Minimal Viable Network

## Overview

Phase 2 implements a minimal predictive coding backbone to verify the core learning loop before adding complexity. This is a stripped-down version of the full architecture described in the planning document.

## Minimal Viable Network (MVP) Specifications

**What's Included:**
- 5-layer rectangular backbone
- Two-compartment neurons (apical/basal)
- Simple activation functions (no temporal convolution yet)
- Prospective learning solver
- Prediction error propagation

**What's Deferred:**
- Temporal convolution kernels (Phase 3)
- Sparse overlay connections (Phase 5)
- Hippocampal integration (Phase 5)
- Audio and peripheral vision inputs (Phase 3)
- Multi-layer depth (MVP uses 5, full uses 10-12)

## Network Structure

```
Layer 0 (Input): Foveal vision (320x320x3 flattened)
         ↕
Layer 1: 1500 neurons
         ↕
Layer 2: 1500 neurons
         ↕
Layer 3: 1500 neurons
         ↕
Layer 4: 1500 neurons
         ↕
Layer 5 (Top): 1500 neurons
```

Each ↕ represents bidirectional connections:
- Bottom-up: Prediction errors flow upward
- Top-down: Predictions flow downward

## Two-Compartment Neuron (Simplified)

Each neuron has two compartments:

**Apical Compartment:**
- Receives top-down predictions from layer above
- Input: `apical_input = W_apical @ layer_above_state`
- Activation: `apical_activity = tanh(apical_input)`

**Basal Compartment:**
- Receives bottom-up signals from layer below
- Input: `basal_input = W_basal @ layer_below_state`
- Activation: `basal_activity = tanh(basal_input)`

**Somatic Integration:**
- Combines apical and basal: `state = gate * apical_activity + (1 - gate) * basal_activity`
- Gate parameter (learnable): Controls relative influence
- Default gate: 0.5 (equal weighting)

**Prediction Error:**
- Each neuron compares its prediction against incoming signal
- Error: `error = signal - prediction`
- Squared error for energy minimization: `energy = 0.5 * error^2`

## Prospective Learning

**Key Idea:** Solve for equilibrium states that minimize prediction error across all layers simultaneously.

**Algorithm:**
1. Forward pass: Compute predictions layer-by-layer
2. Compute prediction errors at each layer
3. Update neuron states to minimize errors (iterative)
4. Update weights based on converged states (local rules)

**Weight Update Rules:**

Apical weights (top-down):
```
ΔW_apical = lr * error * layer_above_state
```

Basal weights (bottom-up):
```
ΔW_basal = lr * error * layer_below_state
```

Gate parameter:
```
Δgate = lr * error * (apical_activity - basal_activity)
```

**Prospective Solve:**
For MVP, use simple iterative refinement:
```python
for iteration in range(num_inference_steps):
    # Update each layer's state based on prediction errors
    for layer in layers:
        apical_pred = layer_above.predict_down()
        basal_signal = layer_below.signal_up()
        layer.state = integrate(apical_pred, basal_signal, gate)
```

## Block Tridiagonal Structure

The weight matrices form a block tridiagonal pattern:

```
        L0   L1   L2   L3   L4   L5
    L0 [  ][ W ]
    L1 [ W ][  ][ W ]
    L2     [ W ][  ][ W ]
    L3         [ W ][  ][ W ]
    L4             [ W ][  ][ W ]
    L5                 [ W ][  ]
```

Each block W represents connections between adjacent layers.
This structure enables efficient prospective solving.

## Data Flow (Prediction Cycle)

**Step 1: Sensory Input**
- Foveal image → Layer 0
- Flatten 320x320x3 → 307200 dimensions
- Normalize to [0, 1]

**Step 2: Forward Pass (Bottom-Up)**
- Each layer receives signals from below
- Neurons update basal compartments
- Errors propagate upward

**Step 3: Backward Pass (Top-Down)**
- Top layer generates predictions
- Predictions flow downward through apical compartments
- Each layer predicts the layer below

**Step 4: Equilibrium**
- Iterate Steps 2-3 until convergence
- Convergence criterion: max(errors) < threshold

**Step 5: Weight Update**
- Once equilibrium reached, update all weights
- Use local learning rules (no backprop)
- Weights change to minimize future prediction errors

## Implementation Modules

```
src/network/
├── __init__.py           # Package exports
├── neuron.py             # TwoCompartmentNeuron class
├── layer.py              # PredictiveCodingLayer class
├── backbone.py           # BackboneNetwork class
├── solver.py             # ProspectiveSolver class
└── ARCHITECTURE.md       # This file
```

## Testing Strategy

**Unit Tests:**
- Neuron: Test compartment integration
- Layer: Test prediction error computation
- Solver: Test convergence on simple patterns

**Integration Tests:**
- Full network: Verify prediction error decreases over training
- Image prediction: Train on static images, verify reconstruction

**Performance Targets (MVP):**
- Inference: <100ms per frame (5 layers, 1500 neurons each)
- Memory: <2GB for network weights (FP16)
- Convergence: <10 iterations for equilibrium

## Deferred Features

These are explicitly deferred to later phases:

**Phase 3:**
- Temporal convolution kernels in compartments
- Frame buffer for temporal context
- Audio and peripheral vision integration

**Phase 5:**
- Sparse overlay connections
- Hippocampal memory integration
- Skip connections and hub neurons

**Phase 6:**
- Text pretraining pipeline
- Multimodal integration

## Success Criteria for Phase 2

MVP is successful if:
1. Network initializes without errors
2. Forward and backward passes execute
3. Prospective solver converges
4. Prediction error decreases over training iterations
5. Network can predict next frame given previous frame (simple test)

Once these criteria are met, Phase 2 is complete and we can proceed to Phase 3.
