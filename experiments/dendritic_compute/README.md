# Dendritic Computation Units Experiments

## Overview

This folder contains experimental implementations using dendritic branches as primary computational units, based on evidence that dendrites perform non-linear integration in compartments.

## Motivation

**Biological observation:** Neurons are not simple integrators. Dendrites compute locally:
- Each branch performs local nonlinear computation
- Soma composes branch outputs
- This creates compositional structure: f(x) = g(h₁(x), h₂(x), h₃(x))

**Parameter budget:** ~1000 dendrites per neuron × 5 params per dendrite = 5K params/neuron
- Matches macrocolumn budget
- More flexible than macrocolumn abstraction (no columnar constraint)

## Compositional Structure

**Dendritic branches = morphisms:**
```
Branch 1: h₁: Input → Feature₁
Branch 2: h₂: Input → Feature₂
Branch 3: h₃: Input → Feature₃

Soma: g: (Feature₁, Feature₂, Feature₃) → Output

Composition: f = g ∘ (h₁, h₂, h₃)
```

**Category theory interpretation:**
- Objects: Feature spaces (inputs, branch outputs, soma output)
- Morphisms: Branch computations (h₁, h₂, h₃)
- Composition operator: Soma (g)
- Product category: Branches compute in parallel (categorical product)

## Advantages Over Macrocolumns

1. **No columnar constraint:** Works for all cortical regions (prefrontal, olfactory, etc.)
2. **Biological precedent:** Dendrites demonstrably compute
3. **Flexible granularity:** Can model different neuron types with different branch counts
4. **Dynamic composition:** Branches can gate each other (context-dependent computation)

## Disadvantages

1. **Less studied:** Dendritic computation rules less understood than columnar structure
2. **Temporal dynamics:** May require modeling dendritic voltage dynamics (slower)
3. **Unclear learning rules:** How do branch weights update? Local vs. global?

## Implementation Considerations

**Key questions:**
- How many branches per neuron? (biological: ~10-100 major dendrites)
- How do branches partition input? (spatial? feature-based?)
- How does soma combine branches? (linear? nonlinear gating?)
- What are the learning rules? (Hebbian? Prospective?)

## Implementation Status

⚠️ **Experimental** - Do not modify main codebase. All experiments isolated here.

## References

- Dendritic computation: Major & Tank (2004) "Persistent neural activity"
- Nonlinear integration: Poirazi et al. (2003) "Pyramidal neuron as two-layer network"
