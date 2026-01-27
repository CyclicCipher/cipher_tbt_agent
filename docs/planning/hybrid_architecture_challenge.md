# Hybrid Architecture Challenge: Columnar and Non-Columnar Cortex

## The Problem

Neuroscience evidence shows the cortex is **not uniformly columnar**:

### Columnar Regions (Macrocolumn Abstraction Applies)
- **Primary Visual (V1):** Retinotopic columns
- **Primary Auditory (A1):** Tonotopic columns
- **Primary Somatosensory (S1):** Somatotopic columns
- **Some Association Areas:** Columnar organization persists

### Non-Columnar Regions (Macrocolumn Abstraction Does NOT Apply)
- **Prefrontal Cortex (PFC):** Abstract reasoning, planning, cognitive control
- **Olfactory Cortex (Piriform):** Distributed, association-like processing
- **Orbitofrontal Cortex (OFC):** Valuation, decision-making
- **Some Limbic Structures:** Emotion, memory consolidation

**Implication:** Any architecture using macrocolumn abstractions must be **hybrid** - columnar for sensory processing, non-columnar for abstract/executive functions.

---

## Why Prefrontal Cortex Matters

**PFC is critical for intelligence:**
- **Working memory:** Holding information in mind
- **Cognitive control:** Task switching, inhibition
- **Abstract reasoning:** Planning, analogies, counterfactuals
- **Meta-cognition:** Thinking about thinking

**Damage to PFC:**
- Phineas Gage: Frontal lobe damage → personality change, impulsivity
- Frontal lobe syndrome: Impaired planning, poor decision-making
- PFC lesions: Working memory deficits, perseveration

**PFC is ~30% of human neocortex** - cannot be ignored.

---

## Architectural Options for Hybrid Networks

### Option 1: Heterogeneous Units (Different Abstractions per Region)

**Structure:**
```
Sensory Cortex:
├─ V1: Macrocolumn units (sparse, columnar)
├─ A1: Macrocolumn units (tonotopic)
└─ S1: Macrocolumn units (somatotopic)

Association Cortex:
├─ Parietal: Hybrid (some columnar structure)
└─ Temporal: Hybrid (object representations)

Executive Cortex:
├─ PFC: Dendritic compute units OR standard PC layers
└─ OFC: Dendritic compute units OR standard PC layers
```

**Communication:**
- Within region: Dense/sparse based on structure
- Between regions: Sparse, functorial mappings

**Advantage:** Matches biology (different regions have different structure)

**Disadvantage:** Complex to implement (multiple abstraction types)

### Option 2: Dendritic Units Everywhere (Universal Abstraction)

**Structure:**
```
All Cortex:
└─ Dendritic compute units

Configuration varies:
- Sensory: Few branches, topographic arrangement
- Association: Moderate branches, clustered
- Executive: Many branches, fully distributed
```

**Advantage:** Single abstraction type (simpler)

**Disadvantage:** May not efficiently capture columnar structure where it exists

### Option 3: Layered Hybrid (Columnar Sensory → Non-Columnar Executive)

**Structure:**
```
Layer 1 (Sensory): Macrocolumns
    ↓ (sparse projection)
Layer 2 (Association): Transition (hybrid)
    ↓ (functorial mapping)
Layer 3 (Executive): Dendritic units or standard PC layers
```

**Information flow:**
- Sensory → compressed via columnar sparse codes
- Association → integrates across modalities
- Executive → abstract, compositional reasoning

**Advantage:** Clean separation of concerns

**Disadvantage:** May not capture recurrent loops (PFC → sensory)

### Option 4: Graph-Based (No Fixed Hierarchy)

**Structure:**
```
Cortical regions as graph nodes:
- Node type determines internal structure (macrocolumn vs. dendritic vs. standard)
- Edges = functorial mappings (sparse, structure-preserving)

Example:
V1 (macrocolumn) ←→ V4 (macrocolumn) ←→ IT (dendritic) ←→ PFC (dendritic)
  ↓                                                           ↓
Thalamus (relay) ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←┘
```

**Advantage:** Flexible, biologically plausible (cortex is graph-like)

**Disadvantage:** Complex connectivity, hard to parallelize

---

## How Non-Columnar Regions Might Work

### Prefrontal Cortex: Hub Networks and Compositional Abstraction

**Computational role:** Manipulate abstract representations

**Hypothesis:** PFC uses compositional structure without spatial topology

**Possible implementation:**

```python
class PrefrontalUnit(nn.Module):
    """
    Non-spatial, compositional computation unit.
    Operates on abstract objects and morphisms.
    """
    def __init__(self, num_objects, embedding_dim):
        # Objects: Abstract concepts (not spatial)
        self.objects = nn.Parameter(torch.randn(num_objects, embedding_dim))

        # Morphisms: Relations between concepts
        self.morphisms = nn.ParameterDict({
            "cause": ...,
            "contains": ...,
            "precedes": ...,
            # Abstract relations, not spatial
        })

        # Working memory: Which objects are currently active
        self.working_memory = torch.zeros(num_objects)

    def compose(self, obj1, relation, obj2):
        """Apply morphism: obj1 -[relation]-> obj2"""
        morphism = self.morphisms[relation]
        return morphism(self.objects[obj1], self.objects[obj2])

    def reason(self, query):
        """Compositional reasoning using morphisms"""
        # Build chain of morphisms to answer query
        # Example: "If A causes B and B causes C, does A cause C?"
        # Compose: cause(A,B) ∘ cause(B,C) = cause(A,C)
        pass
```

**Key differences from sensory cortex:**
- **No spatial topology:** Objects aren't arranged retinotopically
- **Explicit morphisms:** Relations are first-class (not just weights)
- **Compositional:** Chains of reasoning via morphism composition
- **Working memory:** Actively maintains objects (not just feed-forward)

### Olfactory Cortex: Distributed Codes Without Topology

**Computational role:** Recognize chemical features, associate with memories/emotions

**Hypothesis:** Uses association-like processing (distributed, non-topographic)

**Possible implementation:**

```python
class OlfactoryUnit(nn.Module):
    """
    Distributed feature detection without spatial organization.
    """
    def __init__(self, num_features, embedding_dim):
        # Features: Chemical properties (not positions)
        self.features = nn.Parameter(torch.randn(num_features, embedding_dim))

        # No spatial arrangement - fully distributed
        # Connections to limbic system (emotion, memory)
        self.limbic_connections = ...

    def detect(self, odor_input):
        """Detect chemical features (distributed pattern)"""
        # NOT: spatial convolution
        # BUT: feature matching across full input
        return self.features @ odor_input
```

**Key differences:**
- **No topology:** Doesn't preserve spatial structure (because odors don't have one)
- **Associative:** Similar to higher-order association cortex
- **Limbic integration:** Direct connections to emotion/memory (unique to olfaction)

---

## Proposed Hybrid Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│ SENSORY CORTEX (Columnar - Macrocolumn Abstraction)        │
├─────────────────────────────────────────────────────────────┤
│ V1: Retinotopic macrocolumns (sparse codes)                 │
│ A1: Tonotopic macrocolumns (frequency)                      │
│ S1: Somatotopic macrocolumns (body map)                     │
└─────────────────┬───────────────────────────────────────────┘
                  │ (sparse projection, ~3 connections/macrocolumn)
┌─────────────────▼───────────────────────────────────────────┐
│ ASSOCIATION CORTEX (Hybrid - Transition Zone)              │
├─────────────────────────────────────────────────────────────┤
│ Parietal: Sparse macrocolumns (spatial attention)           │
│ Temporal: Dendritic units (object recognition)              │
│ Occipital: Macrocolumns → dendritic transition              │
└─────────────────┬───────────────────────────────────────────┘
                  │ (functorial mapping - structure-preserving)
┌─────────────────▼───────────────────────────────────────────┐
│ EXECUTIVE CORTEX (Non-Columnar - Compositional)            │
├─────────────────────────────────────────────────────────────┤
│ PFC: Compositional units (abstract reasoning)               │
│   - Objects: Concepts (not spatial)                         │
│   - Morphisms: Causal relations, logical operations         │
│   - Working memory: Active maintenance                      │
│ OFC: Value units (reward, decision-making)                  │
└─────────────────┬───────────────────────────────────────────┘
                  │ (recurrent feedback to all levels)
┌─────────────────▼───────────────────────────────────────────┐
│ THALAMUS (Communication Hub)                                │
├─────────────────────────────────────────────────────────────┤
│ Relays between cortical regions                             │
│ Implements message-passing interface                        │
│ Sparse, gated connections                                   │
└─────────────────────────────────────────────────────────────┘
```

### Communication Between Regions

**Functorial mappings** preserve structure:

```python
# Sensory (macrocolumn) → Association (hybrid)
F_sensory_to_assoc: MacrocolumnCode → DendriticFeatures
# Preserves: sparseness, topology

# Association (hybrid) → Executive (compositional)
F_assoc_to_exec: DendriticFeatures → AbstractObjects
# Preserves: compositional structure

# Executive (compositional) → Sensory (macrocolumn) [top-down]
F_exec_to_sensory: AbstractObjects → MacrocolumnCode
# Provides: predictions, attention signals
```

**Key property:** Morphisms compose
```
F_sensory_to_assoc ∘ F_assoc_to_exec = F_sensory_to_exec
```

This ensures end-to-end compositionality.

---

## Implementation Strategy

### Phase 1: Prove Concepts Separately

**Experiment 1 (experiments/macrocolumn/):**
- Implement macrocolumn units
- Test on simple sensory task (MNIST)
- Verify sparse coding, parameter budget

**Experiment 2 (experiments/dendritic_compute/):**
- Implement dendritic units
- Test on compositional task (simple reasoning)
- Verify compositional structure emerges

### Phase 2: Build Hybrid

Once both work independently:
1. Create sensory layer (macrocolumns)
2. Create association layer (transition/hybrid)
3. Create executive layer (dendritic or compositional units)
4. Connect with functorial mappings
5. Test end-to-end on complex task

### Phase 3: Add Temporal Dynamics

Add oscillations, synchrony, temporal binding:
- Necessary for deployment (real-time decision-making)
- Enables dynamic composition
- May improve learning (temporal credit assignment)

---

## Open Questions

1. **What is the right abstraction for association cortex?**
   - Pure macrocolumns? Pure dendritic? Hybrid?
   - How does it transition from sensory to executive?

2. **How do we implement functorial mappings efficiently?**
   - Structure preservation is critical
   - But must be computable on GPU

3. **Can PFC use the same prospective learning as sensory cortex?**
   - Or does abstract reasoning require different learning rules?
   - Is there a "prospective configuration" for logical reasoning?

4. **How much of executive function can be pre-trained?**
   - Sensory: yes (static images)
   - Motor: yes (supervised targets)
   - Abstract reasoning: unclear (may require interactive deployment)

5. **What is the minimal PFC implementation for Danganronpa?**
   - What executive functions are actually needed?
   - Working memory? Planning? Theory of mind?
   - Can we start with simpler executive functions?

---

## Next Steps

1. ✅ Create experiment folders (done)
2. Implement macrocolumn prototype (experiments/macrocolumn/)
3. Implement dendritic prototype (experiments/dendritic_compute/)
4. Design functorial mapping interface
5. Plan hybrid integration strategy

**Guiding principle:** Keep main codebase stable. All experiments isolated. Only merge when proven.
