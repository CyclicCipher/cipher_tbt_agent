# The Canonical Microcircuit: What It Actually Is

## TL;DR
- **NOT** minicolumns or macrocolumns (those are V1-specific)
- **IS** the laminar (layer) structure and connectivity pattern
- **YES** PFC has it, but with variations (agranular vs granular)

---

## What is the Canonical Microcircuit?

From [Exploring Architectural Biases of the Cortical Microcircuit (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11142214/):

The canonical microcircuit (CMC) is:
- An interconnected **column of neurons**
- With **specific connections between excitatory and inhibitory neurons across layers**
- **NOT** about horizontal organization (minicolumns)
- **IS** about vertical organization (6 cortical layers)

### The 6-Layer Structure

```
Layer I:    Apical dendrites, feedback inputs
Layer II/III: Superficial pyramidal (intracortical connections)
Layer IV:   Granular (sensory input) - ABSENT in PFC!
Layer V:    Deep pyramidal (subcortical output)
Layer VI:   Corticothalamic (feedback to thalamus)
```

**Key insight:** The canonical microcircuit is about **laminar connectivity**, not columnar organization.

---

## PFC vs V1: Same Canonical Circuit, Different Variants

### Visual Cortex (V1): "Granular"

- **Prominent Layer IV** (receives thalamic input)
- **Macrocolumns** (orientation, ocular dominance)
- **Minicolumns** (~100 neurons, repeating units)
- Strong horizontal organization

From [Wikipedia: Cerebral Cortex](https://en.wikipedia.org/wiki/Cerebral_cortex):
> "The striate primary visual cortex (area V1) in gyrencephalic primates has the most recognizable cortical architecture."

### Prefrontal Cortex: "Agranular"

- **No Layer IV** (or rudimentary)
- **NO macrocolumns**
- **NO minicolumns** (in the V1 sense)
- Less horizontal organization

From [Towards a "canonical" agranular cortical microcircuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC4294159/):
> "Cortical areas that lack layer IV are called agranular, while those with only a rudimentary layer IV are called dysgranular."

**Critical point:** PFC lacks the columnar structure of V1, but **still has the laminar (layered) structure**.

---

## The Canonical Pattern Across All Cortex

From [Canonical microcircuits for predictive coding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3777738/):

**Universal features (all cortical areas):**

1. **Bottom-up (feedforward) connections:**
   - Layer IV → Layer II/III (in granular areas)
   - Layer II/III → Layer II/III (across areas)

2. **Top-down (feedback) connections:**
   - Layer VI → Layer I (feedback from higher areas)
   - Target apical dendrites of pyramidal cells

3. **Lateral (horizontal) connections:**
   - Within Layer II/III
   - Within Layer V

4. **Output pathways:**
   - Layer V → Subcortical structures
   - Layer VI → Thalamus

**Regional variations:**
- Granular (V1): Has Layer IV for sensory input
- Agranular (PFC): Lacks Layer IV, receives processed input from other cortex

---

## What This Means: Inductive Biases from Laminar Structure

From [Exploring Architectural Biases (2024)](https://direct.mit.edu/neco/article/37/9/1551/131940/Exploring-the-Architectural-Biases-of-the-Cortical):

**The laminar connectivity pattern creates inductive biases:**

1. **Feedback modulation** (Layer VI → Layer I targets apical dendrites)
   - Enables top-down predictions
   - Natural substrate for predictive coding

2. **Error signaling** (Layer II/III pyramidal cells)
   - Compare feedforward input with feedback predictions
   - Send error signals to higher areas

3. **Functional modularization** (2024 finding)
   - Presence of feedback correlates with differentiation of cortical populations
   - Provides **natural inductive bias to differentiate expected vs unexpected inputs**

**This is an architectural inductive bias!** The laminar pattern creates predictive coding dynamics.

---

## Answering Your Question: Is PFC Made of Canonical Microcircuits?

**YES**, but with a critical difference:

### V1 Canonical Microcircuit:
```
Layer I:     Feedback (top-down predictions)
Layer II/III: Error computation
Layer IV:    Sensory input ← PRESENT
Layer V:     Output to superior colliculus
Layer VI:    Feedback to LGN
```
Plus horizontal organization (minicolumns, macrocolumns)

### PFC Canonical Microcircuit:
```
Layer I:     Feedback (top-down predictions)
Layer II/III: Error computation + integration
[Layer IV:   ABSENT - no direct sensory input]
Layer V:     Output to subcortical (motor, basal ganglia)
Layer VI:    Feedback to thalamus (mediodorsal nucleus)
```
No horizontal columnar organization

**Same vertical (laminar) pattern. Different horizontal organization.**

---

## Implications for Architecture

### What We SHOULD Copy from Canonical Microcircuit:

1. **Layered structure** (even if simplified to 2-3 layers in our implementation)
   - Superficial layers: Error computation, lateral connections
   - Deep layers: Output to downstream areas

2. **Feedback connections** (Layer VI-like → Layer I-like)
   - Top-down predictions
   - Modulation of lower areas

3. **Laminar differentiation** (from 2024 DLPFC study)
   - Superficial: Manipulation/updating
   - Deep: Output/motor preparation

### What We Should NOT Blindly Copy:

1. **Minicolumns** - V1-specific, not in PFC
2. **Macrocolumns** - V1-specific, not in PFC
3. **Layer IV** - Sensory areas only, not PFC

### What This Tells Us About Subnetwork Design:

**For vision subnet:**
- Conv layers approximate minicolumns (local feature detectors with weight sharing)
- Multiple layers approximate laminar hierarchy
- Could benefit from explicit feedback connections

**For association/reasoning subnets:**
- **Don't need minicolumns** (PFC doesn't have them)
- **Do need layered structure** (2-3 layers minimum)
- **Do need feedback** (top-down modulation)
- **Do need layer differentiation** (manipulation vs output)

---

## The 2024 Finding: Architectural Bias from Feedback

From [Exploring Architectural Biases (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11142214/):

> "The presence of feedback connections correlates with the functional modularization of cortical populations in different layers, and provides the microcircuit with a **natural inductive bias to differentiate expected and unexpected inputs** at initialization."

**This is huge:** The laminar structure with feedback creates an **inductive bias for predictive coding**.

Not because we engineer it. Because the **architecture naturally implements it**.

**Implication:** If we want predictive coding to work well, we should mirror the laminar feedback structure.

---

## Practical Architecture Recommendations

Based on canonical microcircuit research:

### Minimum Laminar Structure per Subnet:

```python
class CanonicalSubnet:
    def __init__(self):
        # Superficial layers (II/III analog)
        self.superficial = PCLayer(
            role="error_computation",
            lateral_connections=True,  # Within-area integration
            receives_feedback=True     # From higher areas
        )

        # Deep layers (V/VI analog)
        self.deep = PCLayer(
            role="output",
            projects_to_subcortical=True,  # Motor, attention, etc.
            sends_feedback=True             # To lower areas
        )
```

### Vision Subnet (Granular):
- **Add Layer IV analog:** Direct input reception
- **Add horizontal structure:** Conv layers (approximate minicolumns)
- **Add feedback:** From association back to vision

### Association/Reasoning (Agranular):
- **No Layer IV:** Receives processed input from vision
- **No columnar structure:** Just laminar
- **Strong feedback:** To vision and motor

### Working Memory (PFC-like):
- **Agranular structure:** No direct sensory input
- **Superficial dominance:** Manipulation happens in superficial layers (2024 finding)
- **Deep output:** To motor planning

---

## Summary: The Canonical Microcircuit Across All Cortex

| Feature | V1 (Granular) | PFC (Agranular) | Our Implementation |
|---------|---------------|-----------------|-------------------|
| Layer I | ✓ | ✓ | Implicit in feedback |
| Layer II/III | ✓ | ✓ | Superficial layer |
| Layer IV | ✓ | ✗ | Vision only |
| Layer V | ✓ | ✓ | Deep layer (output) |
| Layer VI | ✓ | ✓ | Feedback pathway |
| Minicolumns | ✓ | ✗ | Conv (vision only) |
| Macrocolumns | ✓ | ✗ | No |
| Feedback | ✓ | ✓ | **Yes - critical!** |

**Key takeaway:** The canonical microcircuit is about **laminar structure + feedback**, NOT columnar organization. PFC has it. We should implement it.

---

## Sources

- [Exploring Architectural Biases of Cortical Microcircuit (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11142214/)
- [Laminar organization in macaque cortex (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10996711/)
- [Canonical microcircuits for predictive coding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3777738/)
- [Towards a "canonical" agranular cortical microcircuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC4294159/)
- [Cerebral cortex structure](https://en.wikipedia.org/wiki/Cerebral_cortex)
