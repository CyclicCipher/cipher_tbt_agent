# Abstract Reasoning Architecture: Concrete Specification

## Executive Summary

**Size:** ~10M parameters (same scale as vision system)
**Structure:** 18 interconnected subnets organized categorically
**Memory:** ~40 MB total (19MB params + 20MB activations + 8MB state)
**Type:** NOT a single layer - a SYSTEM of subnets

## Position in Full Agent

```
Position 0: VISION + MOTOR
├─ Vision: Conv → (Ventral × Dorsal) → 768 dims
└─ Motor: Keyboard + Mouse + Gaze → actions

Position 1: ASSOCIATION
├─ Input: 768 dims (from vision)
├─ Processing: Multimodal integration
└─ Output: 512 dims (features)

Position 2: ABSTRACT REASONING ← THIS DOCUMENT
├─ Input: 1536 dims (association + language + memory)
├─ Processing: Compositional inference
└─ Output: Actions, memory updates, attention

Lateral: WORKING MEMORY
├─ Persistent state: 2048 dims
└─ Bidirectional connections to reasoning
```

## Detailed Architecture

### Input Sources (1536 dims total)

```
Association Cortex → [512 dims] ─┐
                                  │
Language Encoder   → [512 dims] ─┼─→ Combined Input (1536 dims)
                                  │
Working Memory     → [512 dims] ─┘
```

**What each provides:**
- **Association:** Vision-derived features (objects, spatial relationships)
- **Language:** Dialogue semantics (what characters said, text on screen)
- **Memory:** Retrieved relevant facts (past evidence, conclusions)

### Evidence Processing (Product: Testimony × Physical × Timeline)

```
Language Input (512) → Testimony Processor → [512 dims]
                              ↓
Association (512)    → Physical Processor  → [512 dims]  → PRODUCT → Evidence (1536 dims)
                              ↓
Memory (512)         → Timeline Processor  → [512 dims]
```

**Parameters:** ~750K
**Categorical structure:** Product (need ALL components jointly)

### Core Reasoning Loop (Adjunction: Abstraction ⊣ Concretization)

```
Evidence (1536 dims)
      ↓
  Abstraction ←─────────────┐ (Recurrent: iterates for multi-step inference)
      ↓                     │
  Theory (256 dims)         │
      ↓                     │
  Hypotheses ───────────────┘ (Coproduct: one active theory)
      ↓
  Concretization
      ↓
  Predictions (1536 dims)
      ↓
  Compare to Evidence ─→ ERROR SIGNAL ─→ Update Theory
```

**Parameters:**
- Abstraction: ~3M (recurrent, multilayer 1536→768→512→256)
- Hypotheses: ~400K (6 alternatives, each 256→256)
- Concretization: ~3M (multilayer 256→512→768→1536)

**Categorical structure:** Exponentials forming adjunction (predictive coding for reasoning!)

### Working Memory (State Monad)

```
Theory (256)
    ↓
Memory Write (256 → 2048) ─→ Working Memory (2048 dims, persistent)
                                      ↓
                             Memory Read (2048 → 512) ─→ Back to Evidence
```

**Parameters:** ~2M
**Categorical structure:** State monad (threads state through computation)
**Persistence:** State survives across timesteps (accumulates evidence)

### Action Selection (Exponential: Theory → Action)

```
Theory (256 dims)
    ├─→ Present Evidence (256 → 1536) ─→ Which evidence to show
    ├─→ Advance Dialogue (256 → 1)    ─→ Continue or wait
    └─→ Attention Control (256 → 256) ─→ Where to look next
```

**Parameters:** ~500K
**Categorical structure:** Exponentials (function spaces Theory → Actions)

## Full System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ INPUTS (1536 dims)                                               │
│   • Association: 512 dims                                        │
│   • Language: 512 dims                                           │
│   • Memory Read: 512 dims                                        │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│ EVIDENCE PROCESSING (Product: ~750K params)                      │
│   Testimony × Physical × Timeline → 1536 dims                    │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│ ABSTRACTION (Recurrent Exponential: ~3M params)                  │
│   Evidence (1536) → Theory (256)                                 │
│   ↺ Iterative refinement (composes with self)                    │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│ HYPOTHESES (Coproduct: ~400K params)                             │
│   H₁ + H₂ + H₃ + H₄ + H₅ + H₆  (one active at a time)           │
│   Each: 256 → 256 dims                                           │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
            ┌────────────┴───────────┐
            ↓                        ↓
┌──────────────────────┐  ┌──────────────────────┐
│ CONCRETIZATION       │  │ ACTION SELECTION     │
│ (Exponential: ~3M)   │  │ (Exponentials: ~500K)│
│                      │  │                      │
│ Theory → Predictions │  │ Theory → Actions     │
│ (256 → 1536)         │  │ • Present evidence   │
│                      │  │ • Advance dialogue   │
│ Adjunction with      │  │ • Attention control  │
│ Abstraction!         │  │                      │
└──────────────────────┘  └──────────────────────┘
            ↓                        ↓
┌──────────────────────┐  ┌──────────────────────┐
│ ERROR SIGNAL         │  │ OUTPUTS              │
│ Predictions vs       │  │ • Motor commands     │
│ Evidence             │  │ • Memory updates     │
│                      │  │ • Gaze shifts        │
└──────────────────────┘  └──────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────────────┐
│ WORKING MEMORY (State Monad: ~2M params)                         │
│   2048 dims persistent state                                     │
│   Read: 2048 → 512 | Write: 256 → 2048                           │
│   ↺ Recurrent: State persists across timesteps                   │
└──────────────────────────────────────────────────────────────────┘
```

## Parameter Breakdown

| Component | Parameters | Description |
|-----------|------------|-------------|
| Evidence processing | ~750K | Product: Testimony × Physical × Timeline |
| Abstraction | ~3M | Recurrent exponential: Evidence → Theory |
| Hypothesis tracking | ~400K | Coproduct: 6 alternative theories |
| Concretization | ~3M | Exponential: Theory → Predictions |
| Working memory | ~2M | State monad with read/write |
| Action selection | ~500K | Exponentials: Theory → Actions |
| **TOTAL** | **~9.7M** | **Full reasoning system** |

## Comparison to Other Systems

| System | Parameters | Memory | Structure |
|--------|------------|--------|-----------|
| Vision | ~10M | ~20 MB | Conv layers, dual-stream |
| Reasoning | ~10M | ~40 MB | 18 subnets, categorical |
| Motor | ~1M | ~2 MB | Simple mapping |
| **Full Agent** | **~21M** | **~62 MB** | **Integrated system** |

## Key Architectural Principles

### 1. It's NOT One Subnet

**Traditional approach:** Single "reasoning layer" between association and motor

**Categorical approach:** 18 interconnected subnets organized by mathematical structure

### 2. Structure is Prescribed, Not Arbitrary

Category theory REQUIRES:
- **Product** for evidence (need all pieces jointly)
- **Coproduct** for hypotheses (alternatives, one active)
- **Exponentials** for inference (function spaces)
- **Adjunction** for abstraction/concretization (bidirectional)
- **State monad** for memory (persistent state)
- **Recurrence** for composition (iterative inference)

### 3. Size is Constrained by Task

**Input dimensionality:**
- Must receive from association (~512), language (~512), memory (~512)
- Total: ~1536 dims

**Output dimensionality:**
- Must produce actions, memory updates, attention shifts
- Total: ~1536-2048 dims

**Internal dimensionality:**
- Theory space: 256 dims (abstract hypothesis)
- Evidence space: 1536 dims (concrete observations)
- Hypothesis space: 256 dims (selected theory)

**Result:** ~10M parameters emerge from these constraints

### 4. Deeply Recurrent, Not Feedforward

Unlike vision (mostly feedforward with limited feedback):
- **Abstraction iterates:** Evidence → Theory₁ → Theory₂ → ... → Final
- **Memory persists:** State accumulates across timesteps
- **Predictions loop back:** Theory → Predictions → Error → Update Theory

This is **essential** for reasoning, not optional.

## Connection to Brain Regions

### Inputs From:
- **Temporal/Parietal Association Cortex:** Object features, spatial relationships
- **Wernicke's Area (Language):** Semantic understanding of dialogue
- **Hippocampus (Memory):** Retrieved relevant facts and episodes

### Outputs To:
- **Premotor Cortex:** Action planning (keyboard, mouse)
- **Frontal Eye Fields:** Gaze control (where to look)
- **Hippocampus:** Memory consolidation (store new conclusions)
- **Self (PFC):** Iterative refinement (recurrent loops)

### Biological Correspondence:
- **Dorsolateral PFC:** Working memory maintenance
- **Ventrolateral PFC:** Hypothesis selection
- **Medial PFC:** Error monitoring (predictions vs. evidence)
- **Anterior PFC:** Abstraction (concrete → abstract)

## Implementation Roadmap

### Phase 1: Evidence Processing (Simplest)
- Build testimony, physical, timeline processors
- Test product structure (concatenation)
- Verify dimensions match

### Phase 2: Abstraction/Concretization (Core)
- Implement multilayer abstraction (1536 → 256)
- Implement multilayer concretization (256 → 1536)
- Test adjunction (compose both directions)
- Add recurrence to abstraction

### Phase 3: Hypothesis Tracking (Novel)
- Create 6 hypothesis subnets
- Implement coproduct (selection mechanism)
- Test mutual exclusivity

### Phase 4: Working Memory (State)
- Implement persistent state buffer
- Create read/write mechanisms
- Test state persistence across timesteps

### Phase 5: Integration
- Connect all components
- Test full reasoning loop
- Validate categorical constraints with validator tool

### Phase 6: Training
- Unsupervised pre-training on evidence → prediction
- Supervised fine-tuning on Danganronpa cases
- Test compositional generalization

## Critical Differences from LLMs

| Aspect | LLM (Transformer) | Categorical Reasoning |
|--------|-------------------|----------------------|
| Structure | Flat, uniform layers | Hierarchical, specialized subnets |
| Evidence | Implicit in attention | Explicit product structure |
| Hypotheses | Distributed in weights | Explicit coproduct (one active) |
| Inference | Single forward pass | Iterative recurrent refinement |
| Memory | Context window | Persistent state monad |
| Abstraction | Learned implicitly | Explicit functor (adjunction) |
| Compositionality | Limited | Enforced by categorical laws |

## Validation Status

✓ Architecture is **categorically valid** (verified by validator tool)
✓ Dimensions are **consistent** (inputs/outputs match)
✓ Structure is **prescribed by category theory** (not arbitrary)
✓ Size is **tractable** (~10M params, ~40 MB memory)
✓ Connections are **biologically plausible** (matches PFC organization)

## Next Steps

1. **Verify vision encoder works** (current blocker for digit recognition)
2. **Build evidence processing subnets** (simplest component)
3. **Test abstraction/concretization adjunction** (core mechanism)
4. **Add working memory state** (persistence across time)
5. **Integrate with full agent** (vision → reasoning → motor)

---

**Key Insight:** Abstract reasoning is NOT a single layer. It's a SYSTEM of 18 subnets with ~10M parameters, organized by categorical laws. The structure is mathematically necessary for compositional inference.
