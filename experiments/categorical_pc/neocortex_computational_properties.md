# Neocortical Computational Properties: What We Actually Know

## Research Question
What inductive biases (if any) do different neocortical regions have? Which are relevant for a computer-using agent?

## Key Finding: Canonical Circuits
**Important:** The neocortex uses **similar architecture everywhere** (canonical circuits), suggesting most specialization comes from **connectivity patterns** and **temporal dynamics**, not fundamentally different computation.

Source: [Revealing the Computational Meaning of Neocortical Interarea Signals](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.00074/full)

This suggests we may NOT need radically different subnet architectures - differences may be in connections and timescales.

---

## Regional Specialization: What Differs?

### 1. Dorsolateral Prefrontal Cortex (DLPFC)

**Function:** Working memory, cognitive control, manipulation of information

**Computational Properties (2024 research):**

- **Layer-specific processing** ([Dynamic layer-specific processing in PFC](https://www.nature.com/articles/s42003-024-06780-8)):
  - Superficial layers: Information **manipulation** (updating, transforming)
  - Deep layers: Motor output (sending signals to premotor areas)

- **Domain specialization** ([Adjacent DLPFC regions](https://www.biorxiv.org/content/10.64898/2025.12.22.695973v1.full.pdf)):
  - Different DLPFC subregions specialized for different domains (social vs. cognitive)
  - Suggests multiple parallel processing streams

- **Temporal processing** ([DLPFC temporal anomalies](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2024.1494227/full)):
  - Maintains temporal order information
  - Top-down modulation of attention

**Inductive Bias:** Likely **none** - appears to be general-purpose working memory with layer specialization. Flexibility is the feature.

**For computer-using agent:** Yes, need working memory subnet with similar properties.

---

### 2. Ventromedial Prefrontal Cortex (vmPFC)

**Function:** Value-based decision making, social cognition, emotion regulation

**Computational Properties:**

- **Structural assumptions** ([Abstract state-based inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC6673813/)):
  - Exploits **higher-order structure** in decision problems
  - NOT just simple reinforcement learning
  - Infers interdependencies between stimuli/actions/rewards

- **Contextual biases** ([Reduced decision bias after vmPFC damage](https://pmc.ncbi.nlm.nih.gov/articles/PMC8064028/)):
  - vmPFC imposes contextual priors on decisions
  - Lesion patients more "rational" but context-insensitive
  - Suggests vmPFC adds **learned structural priors**

**Inductive Bias:** **Yes** - assumes decisions have compositional structure (actions relate to each other, not independent).

**For computer-using agent:** Unclear. Do we need value-based decisions for Danganronpa? Possibly for strategy, but might not be critical initially.

---

### 3. Hierarchical Temporal Processing

**Key Discovery:** Different cortical areas operate at **different temporal scales**.

From [Theory of Multiregional Neocortex](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-110920-035434):

- **Early sensory areas:** Fast timescales (milliseconds)
  - Specialize on rapid sensory processing

- **Higher transmodal areas:** Slow timescales (seconds to minutes)
  - Perform temporal integration
  - Abstract, sustained representations

From [Signatures of hierarchical temporal processing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012355):

- Visual hierarchy has ~10 levels
- Each level integrates over longer timescales
- Not arbitrary - emerges from recurrent connectivity

**Inductive Bias:** **Yes** - hierarchical temporal integration. Higher areas assume inputs are temporally extended patterns, not instantaneous snapshots.

**For computer-using agent:** Critical. Need temporal hierarchy (fast: pixels, slow: game state).

---

### 4. Specialized Feedback Circuits

From [PFC reaches back to shape other regions](https://news.mit.edu/2025/prefrontal-cortex-reaches-into-brain-to-shape-how-other-regions-function-1219):

- **Orbitofrontal cortex (OFC):** Sends arousal signals to visual cortex
- **Anterior cingulate (ACA):** Sends motion signals to motor cortex
- Feedback is **target-specific** (not global modulation)

**Inductive Bias:** Task-dependent modulation - different PFC regions shape different target areas for specific purposes.

**For computer-using agent:** Suggests we need specialized feedback for attention control (where to look) and motor planning (what to do).

---

## Summary: What Inductive Biases Actually Exist?

| Region | Inductive Bias | Evidence Strength | Relevance for Agent |
|--------|---------------|-------------------|---------------------|
| DLPFC | **None** (general WM) | Strong | High - need working memory |
| vmPFC | **Compositional structure** in decisions | Moderate | Medium - maybe for strategy |
| Temporal hierarchy | **Timescale separation** | Very Strong | High - critical for temporal tasks |
| Feedback circuits | **Task-specific modulation** | Strong | High - for attention/motor control |
| Canonical circuits | **Same architecture everywhere** | Very Strong | Critical insight! |

---

## Key Insight: Minimal Differentiation

**The neocortex does NOT have radically different computational units in different regions.**

What varies:
1. **Connectivity** (who connects to whom)
2. **Timescales** (fast vs. slow processing)
3. **Feedback targets** (what to modulate)

What's the same:
1. **Canonical microcircuit** (similar architecture everywhere)
2. **Predictive coding dynamics** (error-driven learning)
3. **Layered structure** (superficial vs. deep)

---

## Implications for Architecture

### What This Suggests We NEED:

1. **Working memory subnet** (DLPFC-like)
   - No special inductive bias
   - Layer differentiation (manipulation vs. output)
   - Recurrent dynamics for persistence

2. **Temporal hierarchy** (all regions)
   - Multiple timescales (fast sensory → slow abstract)
   - Integration across time
   - **This is an inductive bias we MUST have**

3. **Feedback control** (PFC → sensory/motor)
   - Attention modulation (to vision)
   - Motor preparation (to motor system)
   - Task-dependent, not fixed

### What This Suggests We DON'T NEED:

1. **18 different subnet types** - way over-engineered
2. **Specialized deduction engine** - no evidence for this
3. **Separate evidence/hypothesis/concretization subnets** - not biologically grounded

---

## Proposed Minimal Architecture (Brain-Inspired)

Based on actual neuroscience:

```
Position 0: SENSORY/MOTOR (fast timescales)
├─ Vision: Conv PC network (ms timescales)
├─ Language: Text encoding (ms timescales)
└─ Motor: Action output (ms timescales)

Position 1: ASSOCIATION (medium timescales, 100ms-1s)
├─ Multimodal integration
├─ Temporal integration over short windows
└─ Feed to higher areas

Position 2: ABSTRACT (slow timescales, 1s-10s)
├─ Working memory (persistent state, no special bias)
├─ Temporal integration over long windows
└─ Feedback to lower areas (attention, motor prep)

LATERAL: Feedback control
├─ Attention signals → Vision
├─ Motor preparation → Motor
└─ Task modulation → Association
```

**Total subnets needed: ~5-7**, NOT 18.

**Key inductive biases:**
1. Temporal hierarchy (different timescales)
2. Working memory persistence
3. Feedback modulation
4. (Maybe) compositional structure for decisions

**No special bias needed for:** "Reasoning" is general-purpose working memory + temporal integration, not specialized deduction.

---

## Critical Questions for Design

1. **How many timescales?**
   - Vision hierarchy: ~10 levels
   - Do we need all 10, or can we compress to 3-4?

2. **Working memory capacity?**
   - Human: ~4 chunks (Cowan's limit)
   - How much do we need for Danganronpa?

3. **Feedback specificity?**
   - Do we need separate attention/motor feedback?
   - Or can one subnet handle both?

4. **Domain specialization?**
   - Should we have social reasoning subnet?
   - Or is general WM enough?

---

## What We Should Do Next

1. **Build temporal hierarchy** (vision with multiple timescales)
2. **Add working memory** (recurrent state, no special structure)
3. **Add feedback control** (attention to vision, prep to motor)
4. **Test on simple tasks** before adding complexity

**Do NOT:**
- Build 18 specialized subnets
- Add categorical structure for everything
- Over-engineer before we understand requirements

**The brain teaches us:** Start with canonical circuits + connectivity patterns + temporal dynamics. Add specialization only when proven necessary.

---

## Sources

- [Theory of Multiregional Neocortex (Annual Reviews)](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-110920-035434)
- [Dynamic layer-specific PFC processing (Nature)](https://www.nature.com/articles/s42003-024-06780-8)
- [Adjacent DLPFC regions (bioRxiv)](https://www.biorxiv.org/content/10.64898/2025.12.22.695973v1.full.pdf)
- [DLPFC temporal processing (Frontiers)](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2024.1494227/full)
- [vmPFC abstract inference (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6673813/)
- [Reduced bias after vmPFC damage (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8064028/)
- [PFC feedback circuits (MIT News)](https://news.mit.edu/2025/prefrontal-cortex-reaches-into-brain-to-shape-how-other-regions-function-1219)
- [Hierarchical temporal processing (PLOS)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012355)
- [Canonical circuits (Frontiers)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.00074/full)
- [Neocortex self-organization (PNAS)](https://www.pnas.org/doi/10.1073/pnas.2011724117)
