# Categorical Predictive Coding Network - Notes

## Current Implementation

This folder contains a **minimal, focused implementation** of predictive coding networks for MNIST classification.

### Files (ONLY THESE 4 FILES)

1. **categorical_network.py** - All network logic
   - PCConvLayer: PC convolutional layers with local learning
   - PCConvVisionPreprocessor: 3-layer vision hierarchy (V1 simple → V1 complex → V2/V4)
   - CanonicalPCLayer: Dense PC layers with dendrite structure
   - CanonicalMicrocircuit: 3-layer canonical microcircuit

2. **train_mnist.py** - MNIST training script
   - Full PC architecture (conv + inference layers)
   - Active curriculum learning
   - Direct error injection (no teaching signal)

3. **diagnostics.py** - Lightweight diagnostics (no full training runs)
   - Weight statistics
   - State dynamics tracking
   - Single-sample testing

4. **NOTES.md** - This file

### Architecture

```
Input (100×100×3)
  ↓
PC Conv Layer 0: 3→64 channels (V1 simple, precision=1.0)
  ↓
PC Conv Layer 1: 64→128 channels (V1 complex, precision=10.0)
  ↓
PC Conv Layer 2: 128→256 channels (V2/V4, precision=100.0)
  ↓ (flatten to 1024)
PC Inference Layer 0: 1024→512 (superficial, has lateral)
  ↓
PC Inference Layer 1: 512→1024 (middle)
  ↓
PC Inference Layer 2: 1024→10 (output)
```

### Key Features

**Precision Weighting** (VERSES Option A)
- Prevents error signal decay in deep networks
- Layer 0: 1.0, Layer 1: 10.0, Layer 2: 100.0

**Bi-directional Predictions** (VERSES Option B)
- Each layer predicts both up and down
- Creates richer error signals for learning

**Local Learning Only**
- NO backpropagation
- Hebbian-style weight updates: Δw = lr * (error ⊗ input)
- All learning is local to each layer

**Direct Error Injection**
- Instead of teaching signal blending
- Inject "surprise" error at output layer
- Let it propagate backward through bi-directional predictions

### Training

```bash
cd experiments/categorical_pc
python train_mnist.py
```

Hyperparameters:
- PC inference layer LR: 0.01
- PC conv layer LR: 0.001 (10x smaller)
- Training samples: 6000 (subset of MNIST)
- Epochs: 5
- Active curriculum: learning_progress strategy

### Diagnostics

```bash
python diagnostics.py
```

Quick checks without full training:
- Network architecture summary
- Weight statistics
- Test with random input

### Design Decisions

**Why 4 files only?**
- Eliminates iteration bloat
- Forces clear separation of concerns
- Network logic vs experiment vs diagnostics vs notes
- Makes it impossible to accumulate cruft

**Why no backprop?**
- Testing pure predictive coding
- All learning must be local
- Biologically plausible

**Why precision weighting?**
- Deep PC networks suffer from error decay
- Higher precision in deeper layers compensates
- Based on VERSES AI research (2025)

**Why active curriculum?**
- Focus on learnable samples
- Avoid wasting time on mastered or noise samples
- Learning progress tracking

### Current Status

Last commit: Full PC conv hierarchy with VERSES optimizations
- All conv layers are PC layers with state and local learning
- Precision-weighted errors
- Bi-directional predictions
- Direct error injection

### Dependencies

From main codebase:
- `src/active_inference.py` - Active curriculum manager
- `src/network/conv_layer.py` - DEPRECATED (merged into categorical_network.py)

### Future Work

- [ ] Test if precision weighting actually helps (compare with/without)
- [ ] Test if bi-directional predictions help (compare with/without)
- [ ] Compare pure PC vs hybrid PC+backprop
- [ ] Scale to full MNIST (60k samples)
- [ ] Try other datasets (CIFAR-10)

### References

- VERSES AI research on scaling PC networks (2025)
- Canonical microcircuit structure (Douglas & Martin)
- Predictive coding theory (Rao & Ballard, Friston)
