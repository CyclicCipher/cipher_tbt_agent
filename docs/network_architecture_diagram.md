# Network Architecture Diagram

## Current Structure

```
═══════════════════════════════════════════════════════════════════
POSITION 0: Parallel Sensory & Motor Processing
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────┐  ┌─────────────────────────────┐
│  VISION SUBNET              │  │  MOTOR SUBNET               │
│  Position: 0                │  │  Position: 0                │
├─────────────────────────────┤  ├─────────────────────────────┤
│                             │  │                             │
│  input_buffer: 30000 dims   │  │  input_buffer: 10 dims      │
│  (100×100×3 flattened)      │  │  (motor commands)           │
│         ↓                   │  │         ↓                   │
│  ┌───────────────┐          │  │  ┌───────────────┐          │
│  │ Layer 0: 256  │          │  │  │ Layer 0: 10   │          │
│  │ (predicts     │          │  │  │ (predicts     │          │
│  │  input)       │          │  │  │  input)       │          │
│  └───────┬───────┘          │  │  └───────┬───────┘          │
│          ↓                  │  │          ↓                  │
│  ┌───────────────┐          │  │  ┌───────────────┐          │
│  │ Layer 1: 128  │          │  │  │ Layer 1: 32   │          │
│  │ (predicts     │          │  │  │ (motor        │          │
│  │  layer 0)     │          │  │  │  primitives)  │          │
│  └───────┬───────┘          │  │  └───────────────┘          │
│          ↓                  │  │                             │
│  ┌───────────────┐          │  │  Output (to pos 1): 32 dims │
│  │ Layer 2: 64   │          │  │  ← layer 1 state            │
│  │ (top layer)   │          │  │                             │
│  └───────────────┘          │  └─────────────────────────────┘
│                             │
│  Output (to pos 1): 64 dims │
│  ← layer 2 state            │
└─────────────────────────────┘

                    ↓ Concatenate outputs ↓

         Vision (64) + Motor (32) = 96 dims

═══════════════════════════════════════════════════════════════════
POSITION 1: Association / Integration
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│  ASSOCIATION SUBNET                                             │
│  Position: 1                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  input_buffer: 96 dims (vision 64 + motor 32)                  │
│         ↓                                                       │
│  ┌───────────────┐                                              │
│  │ Layer 0: 128  │                                              │
│  │ (integrates   │                                              │
│  │  vision+motor)│                                              │
│  └───────┬───────┘                                              │
│          ↓                                                      │
│  ┌───────────────┐                                              │
│  │ Layer 1: 64   │                                              │
│  │               │                                              │
│  └───────┬───────┘                                              │
│          ↓                                                      │
│  ┌───────────────┐                                              │
│  │ Layer 2: 10   │  ← TOP LAYER (represents digit classes)     │
│  │ (digit        │                                              │
│  │  categories)  │                                              │
│  └───────────────┘                                              │
│                                                                 │
│  Output: 10 dims (digit prediction)                            │
└─────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
CROSS-POSITION PREDICTIONS (Added in recent commit)
═══════════════════════════════════════════════════════════════════

Association Layer 2 (10 dims)
       │
       │ W_pred_assoc_to_vision: 256×10 matrix
       ├──────────────────────────────────────→ Vision Layer 0 (256 dims)
       │
       │ W_pred_assoc_to_motor: 10×10 matrix
       └──────────────────────────────────────→ Motor Layer 0 (10 dims)

These predictions are SKIPPED when motor is clamped (training).
These predictions are ACTIVE when motor is free (inference).
```

## Information Flow During Training

### Forward Pass (with motor clamped to target)

```
1. INPUT PHASE
   ┌──────────────────┐
   │ Visual Input     │ → vision.input_buffer (30000 dims)
   │ (digit image)    │
   └──────────────────┘

   ┌──────────────────┐
   │ Target One-Hot   │ → motor.input_buffer (10 dims) [CLAMPED]
   │ e.g., [0,0,0,0,  │
   │       0,1,0,0,   │
   │       0,0]       │
   └──────────────────┘

2. INFERENCE ITERATIONS (×30)

   Position 0:
   ┌─────────────────────────────────────────────────┐
   │ Vision: Processes visual input                  │
   │   • Layer 0 predicts input_buffer               │
   │   • Layer 1 predicts layer 0                    │
   │   • Layer 2 predicts layer 1                    │
   │   • States settle to minimize errors            │
   └─────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────┐
   │ Motor: Tries to predict clamped input           │
   │   • Layer 0 predicts input_buffer (TARGET)      │
   │   • Layer 1 predicts layer 0                    │
   │   • CONFLICT: Layer 0 caught between:           │
   │     - Bottom-up: what input_buffer says         │
   │     - Top-down: what layer 1 predicts           │
   │   • Cross-position feedback DISABLED (clamped)  │
   └─────────────────────────────────────────────────┘

   Position 1:
   ┌─────────────────────────────────────────────────┐
   │ Association: Integrates vision + motor          │
   │   • Receives: vision layer 2 (64) +             │
   │               motor layer 1 (32) = 96 dims      │
   │   • Layer 0 predicts concatenated input         │
   │   • Layer 1 predicts layer 0                    │
   │   • Layer 2 (top) has no prediction from above  │
   └─────────────────────────────────────────────────┘

   After each iteration:
   └─→ Re-clamp motor.input_buffer to target

3. WEIGHT UPDATE

   For each layer in each subnet:
   ┌─────────────────────────────────────────────────┐
   │ error = state - prediction_from_above           │
   │ ΔW_basal = lr × error × input_from_below        │
   │ ΔW_apical = lr × error × input_from_above       │
   └─────────────────────────────────────────────────┘

   Cross-position weights also update:
   ┌─────────────────────────────────────────────────┐
   │ prediction = W_cross @ association.layer2.state │
   │ error = motor.layer0.state - prediction         │
   │ ΔW_cross = lr × error × association.layer2.state│
   └─────────────────────────────────────────────────┘
```

## The Problem

```
MOTOR SUBNET CONFUSION:

What we're reading as "output":
   motor.layers[0].state  (10 dims) ← This is what we check

What motor layers are actually doing:
   Layer 0 (10 neurons): PREDICTS the 10-dim input_buffer
   Layer 1 (32 neurons): PREDICTS layer 0 state

During training with clamping:
   input_buffer = [0,0,0,0,0,1,0,0,0,0]  (target: digit 5)

   Layer 0 computes:
   └─→ bottom_up = tanh(W_basal @ input_buffer)
       │ ↓
       │ W_basal is random 10×10 matrix
       │ ↓
       └─→ bottom_up ≈ random transformation of target

   Layer 1 computes:
   └─→ top_down feedback to layer 0
       │ ↓
       │ Layer 1 has random 32-dim state
       │ ↓
       └─→ pushes layer 0 toward random prediction

   Layer 0 state settles to COMPROMISE:
   ├─ What bottom_up (from clamped input) says
   └─ What top_down (from random layer 1) says

   Result: Motor "predicts" digit 7, not 5
```

## Expected vs. Actual

```
WHAT WE WANT:
   motor.input_buffer = target
                 ↓
   motor.layers[0].state = target  ← Output we read

WHAT WE GET:
   motor.input_buffer = target (clamped correctly)
                 ↓
   motor.layers[0].state = compromise(
       tanh(W_basal @ target),  ← Bottom-up from target
       layer1_prediction         ← Top-down from random layer 1
   )

   layers[0].state ≠ target
```

## Possible Solutions

### Option 1: Clamp layer 0 state directly (not just input_buffer)
```python
# After inference step, force layer 0 to match target
if clamped:
    motor.layers[0].state.copy_(target)
```
**Problem**: Breaks predictive coding - layer 0 should predict input, not equal it

### Option 2: Remove layer 1 from motor subnet (single layer)
```python
motor_subnet = SubNetwork(
    layer_sizes=[10],  # Only one layer
    input_size=10,
)
```
**Problem**: No motor primitives, less expressiveness

### Option 3: Flip motor architecture (output at top, not bottom)
```python
# Currently: input(10) → layer0(10) → layer1(32)
# Proposed: input(32) → layer0(32) → layer1(10)
#           motor primitives         digit output
```
**Problem**: Requires architectural changes, may break active inference

### Option 4: Use different reading (read from input_buffer, not layer 0)
```python
# Currently read: motor.layers[0].state
# Proposed read: motor.input_buffer
```
**Problem**: Input buffer is what we SET, not what network computes

---

## Questions

1. **What is the intended role of motor layer 0 vs. layer 1?**
   - Layer 0: Digit outputs (10 dims)?
   - Layer 1: Motor primitives (32 dims)?

2. **Where should we read the motor "output"?**
   - layers[0].state? (current, but doesn't match input)
   - layers[1].state? (32 dims, not 10)
   - input_buffer? (we set this, not computed)

3. **Should motor predict its input, or equal its input?**
   - Predict: Standard predictive coding (current)
   - Equal: Supervised learning (what we need?)

4. **Is the cross-position feedback the actual issue?**
   - Or is it the intra-subnet feedback (layer 1 → layer 0)?
