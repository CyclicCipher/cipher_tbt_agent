# Brain Biology Reference — CipherNet

A reference document for biological facts that inform CipherNet's
architecture. Sourced from neuroscience research. Updated as we learn.

## The Pyramidal Neuron (Two-Compartment)

Based on Larkum et al. (1999, 2020, 2024), Spruston (2008).

### Anatomy

```
                APICAL TUFT (in Layer 1)
                branches extensively
                receives: higher cortical feedback, higher-order thalamus
                |
                APICAL TRUNK (spans multiple layers)
                contains Ca2+ spike initiation zone at bifurcation
                electrotonically separated from basal
                |
            ┌───┴───┐
            │  SOMA  │──→ axon → output (Na+ spikes)
            └───┬───┘
                |
            BASAL DENDRITES (3-6 primary branches, ~200-360um)
            receives: feedforward from L4, lateral same-layer
            NMDA spike clustering → AND-gates
```

### Basal Dendrites
- 3.5-5.6 primary branches per neuron (varies by region)
- Extend 190-360um from soma
- 590-19,220 total spines (varies enormously by region)
- Human temporal cortex neurons much larger than primary visual
- NMDA spikes: 40-50mV amplitude, 50-100ms duration
- Cannot generate calcium spikes (only NMDA-mediated)
- Each thin branch (~20-50um) = independent computational subunit
- Proximal: classical STDP plasticity
- Distal: requires NMDA spike + BDNF for plasticity (higher threshold)

### Apical Dendrites
- Single trunk extending to layer 1, bifurcating into tuft
- Trunk length: 200-255um (soma to tuft)
- Tuft radius ~300-380um
- Ca2+ spike initiation zone near trunk-tuft bifurcation
- Electrotonically separated from basal by Ih (HCN) channels
- Different plasticity rules: anti-Hebbian, context association
- Different branches associate with distinct context signals

### BAC Firing (Backpropagation-Activated Calcium spike)
Discovered by Larkum, Zhu & Sakmann (1999, Nature).
1. Somatic Na+ spike backpropagates up apical trunk
2. If it coincides (~3ms window) with subthreshold apical input...
3. Triggers full Ca2+ action potential in apical dendrite
4. Ca2+ spike propagates back to soma → high-frequency BURST
5. Neither the bAP alone nor the apical input alone triggers it

Burst = "I see it AND it's expected" (basal AND apical coincidence).
Single spike = "I see it but it's not predicted."

### Apical Amplification (Phillips & Larkum 2024)
- Apical does NOT subtract prediction from sensory
- Apical AMPLIFIES the sensory signal when context matches
- Feedforward information is PRESERVED, not modified
- Asymmetric: apical depends on basal, but NOT vice versa
- Three modes:
  - Amplification (wake): basal+apical → burst → conscious perception
  - Isolation (NREM sleep): soma ignores apical → local consolidation
  - Drive (REM sleep): apical drives without sensory → dreaming

### Sources
- Larkum et al. 1999, Nature (BAC firing)
- Gidon et al. 2020, Science (human dendrites solve XOR)
- Suzuki & Larkum 2020, Cell (anesthesia decouples apical)
- Phillips & Larkum 2024, PMC (apical amplification)
- Benavides-Piccione et al. 2024, Cerebral Cortex (human morphology)
- Spruston 2008, Nature Reviews Neuroscience (dendritic structure)

---

## Cortical Layers and Their Functions

Based on Bastos et al. (2012), Hawkins et al. (2017).

| Layer | Cell types | Function | Connections |
|-------|-----------|----------|-------------|
| L1 | Few neurons, mostly axons | Receives apical tuft input | Higher cortical feedback, higher-order thalamus |
| L2/3 | Superficial pyramidal | Prediction errors (feedforward) | → L4 of higher area, lateral within layer |
| L4 | Spiny stellate | Receives feedforward input | ← L2/3 of lower area, ← first-order thalamus |
| L5 | Deep pyramidal (thick-tufted) | Predictions (feedback), subcortical output | → L2/3 of lower area (skip L4!), → higher-order thalamus, → BG, → brainstem |
| L6 | Deep pyramidal (thin) | Feedback to own L4, modulates thalamus | → own L4, → first-order thalamus |
| L6b | Subplate remnant | Brain state control | → L1, responsive to orexin (wake signal) |

### Key wiring rules
- Feedforward: L2/3 → next area's L4 (through first-order thalamus or direct)
- Feedback: L5 → lower area's L1 and L2/3 (SKIP L4)
- Lateral: L2/3 → L2/3 of same-layer columns (long-range horizontal)
- Transthalamic: L5 → higher-order thalamus → distant cortical area

### Frequency bands
- Gamma (30-100 Hz): feedforward errors, L2/3, fast
- Beta (13-30 Hz): feedback predictions, L5/L6, slow
- Theta (4-8 Hz): WM maintenance, hippocampal-PFC
- Alpha (8-13 Hz): thalamo-cortical idle/suppression

### Sources
- Bastos et al. 2012, Neuron (canonical microcircuits for PC)
- Hawkins et al. 2017, Frontiers (columns learn structure of world)

---

## Cortical Interneurons

Based on cortical microcircuit research (2024).

| Type | Location | Target | Function |
|------|----------|--------|----------|
| PV (parvalbumin) | L3-L5 (intermediate) | Perisomatic (soma, proximal) | Fast output gating, gamma rhythm generation |
| SST (somatostatin) | L5-L6 (deep) | Dendritic (specific branches) | Branch-specific inhibition, attentional suppression |
| VIP (vasoactive intestinal peptide) | L2/3 (superficial) | Inhibits SST (disinhibition) | Attention spotlight, releases branches from suppression |

### PV-SST-VIP interaction
- VIP fires → SST suppressed → dendrites disinhibited → amplification enabled
- This is the intra-column attention mechanism
- Acetylcholine drives VIP cells → cholinergic attention

### Sources
- Cortical networks with multiple interneuron types (PLOS Comp Bio 2024)

---

## Thalamus

Based on transthalamic pathway research (2024).

### Two types of thalamic nuclei

| | First-Order | Higher-Order |
|---|---|---|
| Input from | Periphery (retina, cochlea) | Cortical L5 |
| Projects to | Primary sensory cortex L4 | Other cortical areas |
| Examples | LGN (vision), MGN (auditory), VPL (somatosensory) | Mediodorsal (MD) → PFC, Pulvinar → parietal |
| Function | Relay sensory input | Bridge between cortical areas |
| Driver synapses | From periphery (Class 1) | From cortical L5 (Class 1) |
| Modulatory control | Less | More (BG gating, cortical feedback) |
| TRN inhibition | Present | More prominent |

### Thalamic modes
- Tonic: faithful relay (awake, attending)
- Burst: amplifies weak signals after hyperpolarization (transitions, alerting)

### Mediodorsal (MD) nucleus — PFC partner
- Receives driver input from cortical L5 (various areas)
- Projects to PFC (dlPFC, vmPFC, OFC)
- BG gates the MD relay (controls what enters WM)
- MD→PFC sustains working memory activity
- Two projection types:
  - D2-expressing: amplifies PFC signals when input is sparse
  - GRIK4-expressing: suppresses PFC noise when input is dense

### Thalamic Reticular Nucleus (TRN)
- Surrounds the thalamus like a shell
- ALL thalamocortical axons pass through TRN
- Inhibitory: creates competition between thalamic relays
- Different sectors for different modalities (visual, auditory, etc.)
- Controls which relay channels are open vs suppressed

### The transthalamic pathway
Two parallel routes for cortico-cortical communication:
1. Direct: L2/3 of area A → L4 of area B (cortico-cortical)
2. Transthalamic: L5 of area A → higher-order thalamus → L4 of area B
Both routes are present in PARALLEL for most cortical connections.

### Sources
- Transthalamic Pathways for Cortical Function (J Neurosci 2024)
- A transthalamic pathway crucial for perception (Nature Comm 2024)
- Thalamic projections sustain PFC activity during WM (Nature Neurosci)
- Thalamic circuits for PFC signal/noise control (Nature 2021)
- MD thalamus regulates task uncertainty (Nature Comm 2025)

---

## Basal Ganglia

Based on PBWM model and BG literature.

### Circuit
```
Cortex (PFC, sensory) → Striatum → GPi/SNr → Thalamus → Cortex
                              ↑
                          Dopamine (VTA/SNc)
```

### Pathways
- Direct (Go): D1 MSNs → inhibit GPi → disinhibit thalamus → gate OPENS
- Indirect (NoGo): D2 MSNs → GPe → STN → excite GPi → gate stays CLOSED
- Hyperdirect: Cortex → STN → GPi (fast broad inhibition, then selective release)

### Dopamine
- Source: VTA (ventral tegmental area), SNc (substantia nigra pars compacta)
- Signal: reward prediction error (RPE)
- Positive RPE: strengthens D1 (Go), weakens D2 (NoGo)
- Negative RPE: opposite
- Operates at theta timescale (slower than individual decisions)

### Striatal organization
- Stripes/patches: parallel BG channels for different cortical loops
- Each stripe gates a different thalamic relay → different PFC region
- Multiple stripes can be open simultaneously (parallel gating)

### Default state
- GPi tonically active → thalamus inhibited → gates CLOSED
- WM holds by DEFAULT (high retention)
- Go signal must ACTIVELY fire to open a gate (update WM)
- This prevents random overwriting of WM contents

### Sources
- O'Reilly & Frank, PBWM model
- Numenta 2024, CMP/BG interactions

---

## Dendritic Plasticity

Based on Bhatt et al. (2009, 2015), Gerstner et al. (2018).

### Synaptic clustering on basal dendrites
- Correlated inputs cluster onto the same dendritic branch
- NMDA spike = local AND-gate requiring ~10-50 co-active synapses
- Clustering driven by: Ras, RhoA GTPases spreading ~5-20um
- Repeated co-activation → synapses migrate to same branch
- Cluster formation timescale: minutes to hours

### Plasticity compartments (basal dendrites)
- Proximal: classical STDP (pre-before-post → LTP within ~25ms)
- Distal: requires NMDA spike + BDNF (higher threshold)
  - BDNF is the essential gating molecule
  - Timing window: ~150ms (broader than STDP's 20-100ms)
  - Only potentiation, never depression
  - ~320% increase when conditions met

### Eligibility traces (three-factor learning)
- Factor 1: presynaptic activity (glutamate release)
- Factor 2: postsynaptic state (voltage/calcium elevation)
- Factor 3: neuromodulatory signal (dopamine, norepinephrine, ACh)
- Pre+post co-fire → set eligibility trace (molecular flag)
- Trace does NOT immediately change weight
- When factor 3 arrives within trace lifetime → weight changes

### Eligibility trace timescales
- Striatum: ~1 second (dopamine-dependent)
- Cortex: 5-10 seconds (norepinephrine→LTP, serotonin→LTD)
- Hippocampus: ~2 seconds to minutes
- Synaptic tags: 30 minutes to 2 hours (consolidation)

### Calcium time constants (nested credit assignment windows)
- Na+ spikes: ~2ms (action selection)
- NMDA spikes: 50-100ms (dendritic integration)
- Ca2+ spikes (apical): ~50-100ms plateau (coincidence detection)
- Biochemical signals: 1-2 seconds (eligibility trace)
- Synaptic tags: 30 min - 2 hours (consolidation)

### Sources
- Bhatt et al. 2009, J Neurosci (plasticity compartments)
- Bhatt et al. 2015, PMC (synaptic clustering review)
- Gerstner et al. 2018, Frontiers (three-factor learning)
- Sacramento et al. 2018, NeurIPS (dendritic microcircuits ≈ backprop)

---

## Hemispheric Lateralization

Based on Gotts et al. (2021, Neuron).

### Why two hemispheres?
- **Parallel dual processing**: language (left) and spatial (right) simultaneously
- **Wiring cost reduction**: specialization avoids duplicate cross-hemisphere connections
- **Two processing styles**: left=sequential/analytical (System 2), right=holistic/contextual (System 1)
- **Resilience**: partial redundancy — if one side damaged, other compensates

### Corpus callosum (interhemispheric connection)
- Anterior: INHIBITORY during competition (keep hemispheres independent)
- Posterior: FACILITATORY during cooperation (share information)
- Intentional BOTTLENECK — prevents constant cross-talk noise
- Can integrate OR isolate depending on context

### 2024 finding: different conduction velocities
- Left hemisphere: faster conduction → precise temporal analysis (speech)
- Right hemisphere: slower conduction → broader integration window (melody)

### For CipherNet
Not needed yet. Relevant when:
- Vision + language run simultaneously
- Analytical + holistic processing on same input
- Damage resilience required

### Sources
- Gotts et al. 2021, Neuron (hemispheric specialization and cognition)
- Two distinct forms of functional lateralization (PNAS 2013)
- Hemispheric specialization tuned by conduction velocities (bioRxiv 2024)
- Corpus callosum: excitation or inhibition? (Neuropsychology Review)

---

## Neuromodulatory Systems

### Acetylcholine (from basal forebrain / nucleus basalis)
- THE cortical attention signal
- Effects:
  - Enhances thalamocortical input (more signal through relay)
  - Lowers dendritic NMDA thresholds (AND-gates fire more easily)
  - Suppresses intracortical lateral spread (less noise)
  - Drives VIP interneurons → disinhibits dendrites
- Net effect: sharpens cortical representation (more signal, less noise)
- Driven by: novelty, task demands, PFC error monitor

### Dopamine (from VTA / substantia nigra)
- Reward prediction error
- Positive: strengthens Go pathway, facilitates WM gating
- Negative: strengthens NoGo, blocks WM updates
- Also directly modulates cortical plasticity rules
- Two receptor types in cortex: D1 (excitatory), D2 (inhibitory)

### Norepinephrine (from locus coeruleus)
- Arousal / salience / urgency
- Global gain knob for cortical responsiveness
- Modulates eligibility trace conversion (factor 3 for LTP)
- High NE = high arousal = everything more responsive

### Serotonin (from raphe nuclei)
- Mood / valence / patience
- Modulates eligibility trace conversion (factor 3 for LTD)
- Slower timescale than dopamine

---

## Dopamine and Credit Assignment

Based on Yagishita et al. (2014), Gerstner et al. (2018), Fisher et al. (2025).

### Reward Prediction Error (RPE)
- Dopamine neurons in VTA/SNc encode RPE = actual - expected reward
- Positive RPE (better than expected): phasic dopamine burst
- Negative RPE (worse than expected): dopamine dip below baseline
- Zero RPE (as expected): no change
- 2025 finding: ALSO encodes action prediction error (APE) — reinforces repeated actions independent of reward (lateral SNc → tail of striatum)

### The Silent Eligibility Trace in Striatum
Discovered by Yagishita et al. (2014, Science). The mechanism:
1. Pre + post fire together → SILENT trace set at synapse
   (no immediate plasticity change — hence "silent")
2. Molecular basis: transient CP-AMPA receptor insertion in spine
   (appears within 0-4s, detectable by inward rectification)
3. Trace window: ~2 SECONDS (strict — dopamine at -2s, 0s, or +4s has no effect)
4. When dopamine arrives at +2s:
   - D1 receptor activation → adenylate cyclase signaling
   - Redirects calcium from L-type VSCCs to CP-AMPA receptors
   - Converts t-LTD into t-LTP (unmasking)
5. Without dopamine: pre-post pairing → t-LTD (depression)
   With dopamine at +2s: same pairing → t-LTP (potentiation)

### D1 vs D2 MSN Differences
- D1 MSNs (Go pathway): show eligibility trace + dopamine conversion
  - Pre-post → t-LTD (baseline)
  - Pre-post + dopamine at +2s → t-LTP
  - Supralinear calcium rises in dendritic spines
- D2 MSNs (NoGo pathway): NO eligibility trace in this study
  - Pre-post → no significant change
  - Separate dopamine circuitry (D2 receptor, different signaling)
  - May use different plasticity rules (punishment-driven)

### Solving temporal credit assignment
The 2-second eligibility window bridges the delay between:
- Action (gating decision in BG) at time t
- Reward (dopamine signal) at time t + 2s
Without this trace, the brain couldn't associate actions with
delayed rewards. The trace "tags" which synapses were active
BEFORE the reward, allowing retroactive reinforcement.

### Two types of dopaminergic teaching signals (2025)
- RPE (reward prediction error): reinforces reward-driven actions
  - VTA and medial SNc neurons
  - Projects to ventral striatum (nucleus accumbens)
- APE (action prediction error): reinforces repeated actions
  - Lateral SNc neurons
  - Projects to tail of striatum
  - Independent of reward — drives habit formation

### For CipherNet
- Eligibility traces on cortex→D1 edges (~20 gamma cycles = ~2s)
- Dopamine fires when training provides reward signal
- D1 edges with active traces get potentiated (Go strengthened)
- D1 edges without traces are NOT affected (temporal specificity)
- D2 pathway may need separate plasticity rule

### Sources
- Yagishita et al. 2014, Science (silent eligibility trace in striatum)
- Gerstner et al. 2018, Frontiers (three-factor learning rules)
- Fisher et al. 2025, Nature (RPE + APE dual teaching signals)
- Dopamine, Updated: RPE and Beyond (Current Opinion Neurobiol 2021)
- Silent eligibility trace in mouse striatum (PMC 2019)
- Striatal dopamine signals errors across domains (Science Advances 2024)

### Sources (neuromodulatory systems)
- Adrenergic modulation of L5 dendritic excitability (Cell Reports 2018)
- Cholinergic control of neocortical output neurons (Neuron 2018)
- Neuromodulation of STDP (Neuron 2019)

---

## Predictive Coding in Dendrites

Based on Mikulasch et al. (2023), Sacramento et al. (2018).

### Where is the error?
- Classical PC: separate error neurons (problematic — who computes the error?)
- Dendritic PC: errors computed LOCALLY in dendritic compartments
- Basal dendrite: compares feedforward input with prediction
- Apical dendrite: compares top-down feedback with expected context
- No dedicated error neurons needed

### Sacramento et al. (2018) — dendritic microcircuits ≈ backprop
- Apical dendrites compute prediction errors continuously
- Errors from mismatch between:
  - Lateral interneuron prediction (what top-down SHOULD be)
  - Actual top-down feedback (what it IS)
- Local dendritic error + local Hebbian rule ≈ backprop gradient
- Simultaneous representation + error computation (no separate phases)

### Sources
- Mikulasch et al. 2023, Trends Neurosci (dendritic hierarchical PC)
- Sacramento et al. 2018, NeurIPS (dendritic microcircuits ≈ backprop)

---

## Thousand Brains Theory (Column Voting)

Based on Hawkins et al. (2017), Numenta (2024).

### Column communication
- L2/3 lateral connections carry HYPOTHESES (not raw features)
- Each column votes on what object it thinks it's sensing
- Voting uses sparse activity patterns in L2/3 output layer
- Mutual reinforcement: cells representing same object positively bias each other

### Formal mechanism (Hawkins 2017, Equation 5)
Cell i fires IF:
- Feedforward support (sensory match from L4)
- AND lateral support >= threshold (enough other columns agree)

### Cortical Messaging Protocol (Numenta 2024)
Messages contain:
- Location (x,y,z body-centric)
- Morphological features (3x3 orientation)
- Non-morphological features (color, texture)
- Confidence [0,1]
- Sender ID
- NOT the raw sensory features (abstract only)

### Voting accounts for spatial relationships
- Not a "bag of features" operation
- Depends on relative arrangement of features in the world
- Each column communicates its hypothesis + where on the object

### Sources
- Hawkins et al. 2017, Frontiers (columns learn world structure)
- Numenta 2024, arXiv (Thousand Brains Project, CMP)

---

## Synaptic Weight Precision

Based on Bhatt et al. (2024), neuromorphic hardware research.

- ~4.1 to 4.59 bits per synapse (Shannon entropy)
- ~24 distinguishable synaptic sizes (dendritic spine head volume)
- Determined by AMPA receptor count in postsynaptic membrane
- Range: 2-178 AMPA receptors per synapse (Purkinje cells)
- Advantages of increasing precision vanish rapidly after first few bits
- 4-bit resolution sufficient for STDP in neuromorphic hardware

### Our implementation: 8-bit (256 levels)
- Excitatory: [0, 1.0] in 255 steps (~0.004 per quantum)
- Inhibitory: [-1.0, 0] in 255 steps
- Changes smaller than one quantum have no effect

### Sources
- Synaptic Information Storage Capacity (Neural Computation 2024)
- Is a 4-bit synaptic weight resolution enough? (Frontiers 2012)

---

## Larkum Lab Key Papers

1. **Larkum et al. 1999, Nature**: BAC firing discovered
2. **Gidon et al. 2020, Science**: Human dendrites solve XOR
3. **Suzuki & Larkum 2020, Cell**: Anesthesia decouples apical
4. **Doron et al. 2020, Science**: Perirhinal → L1 controls learning
5. **Takahashi et al. 2020, Nature Neurosci**: Dendritic currents gate perception
6. **Aru et al. 2020, Trends Cog Sci**: Cellular mechanisms of consciousness
7. **Larkum 2022, Neuroscience**: "Are Dendrites Conceptually Useful?" (yes)
8. **Zolnik et al. 2024, Neuron**: L6b controls brain state via apical
9. **Storm, Larkum et al. 2024, Neuron**: Integrative view on consciousness theories
