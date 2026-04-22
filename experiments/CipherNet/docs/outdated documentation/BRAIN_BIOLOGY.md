# Brain Biology Reference — CipherNet

A reference document for biological facts that inform CipherNet's
architecture. Sourced from neuroscience research. Updated as we learn.

## Approximate Number System (ANS)

Based on Nieder (2016, 2024), Piazza et al. (2004), Izard et al. (2009).

### What is it?
INNATE, evolutionarily ancient cognitive system for representing
approximate numerical magnitudes. Present at BIRTH. Shared with
monkeys, birds, fish. NOT learned from experience — the architecture
is genetic, though precision refines with experience.

### Neural implementation
- **Location:** bilateral intraparietal sulcus (IPS), parietal cortex.
  Right-lateralized in infants (before language). Also PFC neurons
  (task-dependent numerosity coding).
- **Number neurons:** each neuron has a PREFERRED NUMEROSITY and
  a Gaussian tuning curve on a LOGARITHMIC scale.
  - Neuron tuned to 4: fires strongly for 4, weakly for 3 and 5
  - Tuning WIDTH proportional to preferred numerosity (Weber's law)
  - Neuron tuned to 4: width ~1. Tuned to 20: width ~5.
- **Population code (labeled-line):** different neurons tuned to
  different numerosities. The most-active neuron IS the perceived
  number. As a population, they tile the number line.
- **Logarithmic compression:** more cortical space for small numbers.
  1-4: many neurons, high precision (subitizing).
  >4: fewer neurons, lower precision (estimation).

### Tuning curve equation
```
response(n) = A * exp(-(log(n) - log(preferred))^2 / (2 * sigma^2))
```
Gaussian on log scale. Width sigma proportional to preferred numerosity.

### Weber fraction (precision by age)
| Age | Weber fraction | Discriminable ratio |
|-----|---------------|-------------------|
| Newborn | ~0.33 | 1:3 |
| 6 months | ~0.50 | 1:2 |
| 9 months | ~0.67 | 2:3 |
| Preschool | ~0.75 | 3:4 |
| Adult | ~0.88 | 7:8 |

Architecture innate, precision improves with experience.

### Two systems (or one?)
- **Subitizing (1-4):** precise, parallel, fast. More neurons per
  numerosity → higher precision. Labeled-line code.
- **Estimation (>4):** approximate, ratio-dependent. Fewer neurons
  → logarithmic compression. Population code.
- May be a single system with naturally higher resolution for
  small numbers (more cortical space devoted to small numerosities).

### What is innate vs learned
- **INNATE:** parietal number neurons, approximate numerosity
  discrimination, right-parietal specialization (before language),
  abstract cross-modal number (visual + auditory matched at birth)
- **LEARNED:** symbol-to-number mapping (digit "3" = quantity three),
  exact arithmetic, the number line as an ordered spatial structure,
  Weber fraction refinement

### Parietal vs prefrontal number neurons
- **IPS (parietal):** stimulus-driven, pure numerosity coding.
  Tuning curves stable across tasks.
- **PFC:** task-dependent. Change tuning based on what the monkey
  is doing. Arithmetic rule neurons are HERE, not in parietal.
  PFC number neurons are more flexible but less reliable.

### For CipherNet
Our current ANS prior (8 nodes: comparator circuit) is wrong.
The biological ANS is:
- A set of cortical columns, each tuned to a preferred numerosity
- Gaussian tuning on log scale → Weber's law emerges
- Population code across columns → number representation
- Comparison via lateral inhibition between number columns
- Pre-wired approximately, refined by experience

### Sources
- Nieder 2016, Nature Rev Neurosci (the neuronal code for number)
- Nieder 2024, Physiol Rev (the calculating brain)
- Piazza et al. 2004, Neuron (tuning curves in human IPS)
- Izard et al. 2009, PNAS (newborns perceive abstract numbers)
- Decarli et al. 2023, Dev Science (12-month ANS predicts 4-year math)
- Nature Human Behaviour 2023 (distinct coding small vs large numbers)
- Nature Comm 2023 (learning-induced number neuron reorganization)

---

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

### BG gating learning (PBWM model, O'Reilly & Frank 2006)

How the BG learns WHICH gate to open WHEN — via reinforcement learning.

**The algorithm (synaptic tagging + dopamine reinforcement):**
1. When a Go neuron fires (gate opens), its active synapses get TAGGED
   (eligibility trace). Weights don't change yet.
2. Later, dopamine signal arrives (reward prediction error):
   - Positive RPE: D1/Go tagged synapses STRENGTHEN, D2/NoGo WEAKEN
   - Negative RPE: D2/NoGo tagged synapses STRENGTHEN, D1/Go WEAKEN
3. Only tagged synapses update (solves temporal credit assignment)
4. Update rule: delta_w = learning_rate * dopamine * eligibility_trace

**Stripe specialization (emergent, not hardcoded):**
- All stripes receive the SAME dopamine signal
- But each stripe's synapses are tagged by DIFFERENT inputs
- Over many trials: each stripe converges on the gating policy that
  works for its input domain
- Specialization emerges from input-driven tagging, not explicit assignment

**Cold start (exploration → exploitation):**
- High tonic dopamine → low selectivity → multiple gates open (exploration)
- As learning progresses, selectivity increases (exploitation)
- This is explore/exploit at the neural level

**Three gating functions (Chatham & Badre 2015):**
- Input gating: WHEN to update WM (rostral/associative striatum)
- Output gating: WHICH WM content to deploy (caudate)
- Maintenance: WHEN to clear obsolete info (ventral striatum)
- Each learns independently through dopamine

**PFC biasing (not controlling):**
- PFC sends maintained representations → striatum
- Provides BIAS (context), not INSTRUCTION
- The actual Go/NoGo decision is made by the striatum
- BG discovers optimal gating policy relative to PFC context

### Sources
- O'Reilly & Frank 2006, Neural Computation (PBWM model)
- Hazy, Frank & O'Reilly 2007 (executive without homunculus)
- Chatham & Badre 2015 (multiple gates on working memory)
- Frank & Badre 2012 (corticostriatal output gating)

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

### Cross-example contamination prevention
The brain prevents one episode's traces from contaminating the next:
- **Dopamine at reward = episode boundary.** Phasic dopamine "closes"
  the current eligibility window. Sustained action representation
  terminates. New episode begins clean. (Rubin et al. 2021)
- **Theta phase reset on novelty.** D1 receptors in ventral hippocampus
  trigger persistent theta that reorganizes vHPC activity. Existing
  vHPC-mPFC synaptic strength WEAKENED for new encoding capacity.
  (Guise & Bhatt, Nature Communications 2021)
- **Event boundary signal.** At event boundaries: hippocampus spikes,
  temporal context resets, items across boundary recalled as more
  distant. No single "new episode" neuron — convergence of dopamine
  novelty tagging + theta reset + pattern separation + SWR replay +
  PFC context updating. (Zheng et al., Nature Communications 2022)
- **BG sustained activation.** Only ONE action representation sustained
  at a time (winner-take-all). When dopamine arrives, only the
  still-active synapses get reinforced. Mechanically prevents
  cross-action contamination. (Rubin et al. 2021)
- **Hippocampal pattern separation.** Dentate gyrus orthogonalizes
  similar inputs into distinct representations. 5x more neurons
  than input → sparse coding. Prevents catastrophic interference.

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

### Phase-dependent dendritic computation

NMDA spikes are sensitive to the TIMING of inputs within the theta oscillation cycle.

**Coincidence detection timing windows:**
- Single dendritic branch: ~5-10ms for NMDA spike generation
- ~20 synchronized inputs within <6ms needed for basal dendrite NMDA spike
- Distal dendrites: broader windows (~8-20ms), supralinear response to clustered inputs
- Proximal dendrites: tight windows (~2-5ms), linear summation

**Phase-dependent plasticity (Huerta & Lisman 1995, Pavlides et al. 1988):**
- Stimulation at theta PEAK: LTP (potentiation)
- Stimulation at theta TROUGH: LTD (depression)
- 4-pulse burst at 100Hz at theta peak induces robust LTP
- Same burst at theta trough induces depotentiation

**Constructive/destructive interference in dendrites:**
- Two inputs arriving on the same dendritic branch at the SAME theta phase: constructive interference (amplified response)
- Two inputs at OPPOSITE theta phases: destructive interference (suppressed response)
- This is voltage-based: overlapping depolarization waveforms sum constructively or destructively
- Concordance factor: (1 + cos(phase_difference)) / 2 (ranges 0-1)

**HCN channel resonance:**
- Pyramidal neurons express HCN channels creating intrinsic theta-frequency resonance
- Distal dendrites: higher HCN conductance = higher resonance impedance at theta
- Creates a "phase filter": same input produces stronger response at the resonant phase for that compartment
- Proximal inputs drive spikes at depolarizing theta peaks
- Distal inputs drive spikes at hyperpolarizing theta troughs

**Sources:**
- Springer et al. 2008, JNP (NMDA plateau potentials)
- Huerta & Lisman 1995, J Neurosci (theta phase determines LTP vs LTD)
- Pavlides et al. 1988, Nature (phase-dependent plasticity)
- Gidon et al. 2020, Science (human dendritic computation)

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

### Acetylcholine (from nucleus basalis of Meynert / basal forebrain)
THE cortical attention signal. Wired and PHASIC (not slow diffuse
volume transmission — Sarter et al. 2009 settled this debate).

**Source:** nucleus basalis of Meynert (NBM). Topographic: rostral
NBM → broad cortical layers, caudal NBM → deep layers.

**Transmission:** point-to-point synaptic + en passant release.
Millisecond-scale phasic signals detected by modern biosensors.

**Effects on cortical processing:**
- **Thalamocortical input: ENHANCED.** Nicotinic receptors on
  thalamic axon terminals in L4 lower threshold and increase
  sensory-evoked responses. Direct bottom-up boost.
- **Intracortical lateral: SUPPRESSED.** Presynaptic muscarinic M2
  receptors reduce efficacy of lateral/recurrent connections.
  Dampens internal recurrence and lateral spread.
- **Dendritic NMDA thresholds: LOWERED.** ACh facilitates NMDA
  spikes, making dendritic AND-gates fire more easily.
- **VIP→SST disinhibitory cascade:** ACh depolarizes VIP (nicotinic)
  → VIP inhibits SST → SST releases pyramidal dendrites →
  pyramidal cells amplified. This IS the gain mechanism.
- **Apical-basal coupling: ENHANCED.** ACh facilitates BAC firing
  by lowering apical calcium spike threshold. Top-down context
  more easily amplifies bottom-up sensory.
- **Net effect: SHARPEN.** Boost direct sensory, suppress lateral
  noise, disinhibit dendrites. Signal-to-noise ratio improves.

**What drives ACh release:**
- PFC → basal forebrain: top-down attention commands
- Amygdala → basal forebrain: emotional salience
- Hippocampus → basal forebrain: novelty/context
- Hypothalamus/orexin → basal forebrain: arousal state
- Striatum → basal forebrain: reward-related gating

**Time course:**
- Phasic: 200-500ms onset, lasts seconds. Cue detection.
- Tonic: minutes. General cognitive demand, working memory.

**In predictive coding (eLife 2024):**
ACh = PRECISION WEIGHTING of prediction errors. ACh modulates
mismatch negativity (error) without affecting repetition suppression.
Most neurons suppressed, few encoding precise errors enhanced.
This IS the Friston/Feldman precision framework implemented.

**When blocked (scopolamine):**
- Impaired memory encoding, verbal recall, recognition
- Signal-to-noise degradation in sensory processing
- Planning/WM relatively spared (those use other neuromodulators)

**For CipherNet:**
ACh during training prevents merge contamination by:
1. Enhancing thalamocortical path (direct feature input boosted)
2. Suppressing intracortical lateral paths (indirect activation blocked)
3. Only directly-fed features stay strongly active
4. Merge sees correct feature pairs, not indirect contamination

### Sources (ACh)
- Sarter et al. 2009, Nature Rev Neurosci (wired, not volume)
- Forebrain cholinergic signalling (Nature Rev Neurosci 2023)
- Nicotinic control of thalamocortical transmission (Nature Neurosci)
- VIP-SST disinhibitory circuit (Neuron 2024)
- ACh modulates precision of prediction error (eLife 2024)
- Zolnik/Larkum 2024, Neuron (L6b NMDA spikes via L1)
- Scopolamine meta-analysis (PMC 2025)

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

## Broca's Area (BA44 / BA45)

NOT a special structure — standard cortical columns with specific connectivity.

### BA44 (pars opercularis)
- **Anterior BA44**: Merge operation — hierarchical structure building.
  Major hub for syntactic Merge (Minimalist Program).
  Standard cortical columns that learned to process hierarchy.
- **Posterior BA44**: premotor/motor sequencing — selects actions
  based on contextual signals from other cortical areas.
- Evolved from an action-related region into a bipartite system:
  posterior = action, anterior = syntax. (PLOS Biology 2023)

### BA45 (pars triangularis)
- Controlled semantic retrieval and top-down lexical selection.
- Standard cortical columns specialized for meaning, not structure.

### Connectivity (what makes Broca special)
- **Arcuate fasciculus** (dorsal stream): Broca ↔ temporal cortex.
  Carries phonological/articulatory information.
- **Extreme capsule** (ventral stream): Broca ↔ temporal cortex.
  Carries semantic/comprehension information.
- **Anterior putamen** (BG): direct cortical-striatal projection.
  BA44 → putamen for procedural gating of Merge.
- **Ventral anterior thalamus** (VA): higher-order thalamic relay.
  BA44 ↔ VA for the Broca cortico-BG-thalamocortical loop.
- **BA45 → caudate head**: separate BG channel for semantic retrieval.

### Sources
- Language and action in Broca's area (Brain & Cognition 2021)
- Broca's striatal and thalamic connections (Frontiers Neuroanatomy 2013)
- Morphological evolution of language areas (PLOS Biology 2023)
- Revisiting Broca's role (Frontiers Language Sciences 2025)

---

## Temporal Cortex (STG / Wernicke's Area)

NOT a special structure — standard cortical columns in auditory hierarchy.

### Processing hierarchy
Primary auditory cortex → belt → parabelt → STG → association areas.
Each level = cortical columns with increasingly complex representations.

### Two output streams from STG
- **Dorsal** (via arcuate fasciculus + superior longitudinal fascicle):
  STG → Broca. Phonological processing, sublexical repetition,
  sound-to-articulation mapping.
- **Ventral** (via extreme capsule + middle longitudinal fascicle):
  STG → ventrolateral PFC. Semantic processing, comprehension,
  sound-to-meaning mapping.

### Wernicke's area (posterior STG)
- Speech perception, phonological representation
- Electrical stimulation → paraphasic errors (phonological deficit)
- Increasing left lateralization through development
- NOT a distinct cortical type — standard columns, specific connectivity

### Thalamic connections
- **Pulvinar** → temporal lobe: four distinct tracts
  Bridge dorsal and ventral streams (Brain 2024)
- **Medial geniculate nucleus (MGN)** → primary auditory cortex:
  first-order thalamic relay for auditory input

### Phonological buffer
NOT a fixed 4-slot hardware buffer. Emerges from:
- Cortical column self-loops (recurrent persistence)
- Theta-gamma coupling (sequential items at different phases)
- The "buffer" IS the pattern of active columns with their
  self-loop persistence, not dedicated buffer nodes.

### Sources
- From Sound to Meaning: Navigating Wernicke's Area (Cureus 2024)
- Beyond ventral and dorsal streams (Brain 2024)
- Dual stream model (PNAS 2008)
- Encoding of speech sounds in STG (Neuron 2019)

---

## Why 6 Layers (The PC Signal Fidelity Hypothesis)

### The hypothesis
Cortical columns have 6 layers because this is the maximum depth
at which predictive coding maintains useful learning signal fidelity.

### Supporting evidence
- PC error = input - f(W * prediction). Each layer adds nonlinear
  approximation error through f(). After 6-7 layers, accumulated
  error exceeds useful signal.
- Backprop uses EXACT transpose weights → no accumulation →
  can go arbitrarily deep. PC uses local dendritic approximation.
- PC is competitive with backprop up to ~6-7 layers, then loses.
  (Sacramento et al. 2018 showed PC ≈ backprop on shallow networks)
- The brain scales by WIDTH (more columns), not depth (more layers).
  Neocortex grows in surface area, not thickness.

### The thalamic depth reset
When signal goes column A → thalamus → column B:
- Column B's L4 receives a fresh input (depth resets to 0)
- Column B runs its own 6-layer PC cycle
- Total processing depth = unlimited (many hops)
- Each hop = only 6 layers (within signal fidelity range)
- The thalamus prevents deep-network signal degradation

### Implications for data efficiency
- A single cortical column is a tiny network (~600 neurons)
- Tiny networks don't overfit → need very little training data
- Built-in spatial structure (reference frames) prevents memorization
- The scaling law INVERTS: more columns = more capability
  without proportionally more data per column

### Status: hypothesis, not proven
Good empirical grounding but needs formal verification.
The ~6-7 layer PC fidelity limit is an empirical observation
from computational studies, not a proven theorem.

---

## Theta-Gamma Phase Coding (Position Encoding in Working Memory)

The brain encodes the POSITION of items in a working memory sequence using the PHASE of neural oscillations. This is the primary mechanism for serial order.

### Theta-gamma coupling

- Each theta cycle (~125-250ms, 4-8 Hz) is divided into multiple gamma windows (~30-100 Hz)
- Each gamma burst within a theta cycle carries ONE working memory item
- The theta PHASE at which an item's gamma burst occurs encodes that item's POSITION in the sequence
- Position 1 = gamma burst at theta phase 0deg, Position 2 = gamma burst at 90deg, etc.
- Cross-frequency theta-gamma coupling increases during sequence learning
- Delta and theta power increase monotonically with serial position

**Sources:** Heusser et al. 2016 (PMC5039104), Leszcznski et al. 2015, Axmacher et al. 2010 (PNAS)

### Phase precession

- As the brain processes sequential items, neurons fire at progressively EARLIER phases of theta
- This compresses entire sequences into single theta cycles (5-10x temporal compression)
- Both firing RATE and spike PHASE encode position information
- Phase coding is more precise than rate coding for fine position discrimination
- Phase precession enables spike-timing-dependent plasticity for sequence binding

**Sources:** Qasim et al. 2021 (PMC8195854), Stangl et al. 2024 (Nature Human Behaviour)

### Theta sequences (compressed trajectory encoding)

- During a single theta oscillation, neurons representing a trajectory fire in compressed sequential order
- Encodes both WHAT (which items) and WHERE (in what order)
- Theta sequences are modulated by goals and intentions (planning role)
- Hippocampal theta sequences segment experience into discrete events

**Sources:** Drieu & Bhatt 2016 (PMC5049882), Patel et al. 2015 (PMC4428659)

### Content-Position separation (the PoPE principle)

- CONTENT (identity of item) is encoded by firing RATE (magnitude)
- POSITION (where in sequence) is encoded by spike TIMING relative to theta (phase)
- These two dimensions are ORTHOGONAL: you can vary position without changing content and vice versa
- This is exactly what PoPE (Polar Positional Embedding) implements computationally: content = magnitude, position = phase in polar coordinates
- Domain-general: the same phase coding works for spatial position, digit position, word position

---

## Grid Cells and Abstract Position Encoding

The entorhinal grid cell system provides a domain-general coordinate system for organizing any ordered information, not just physical space.

### Grid cells for conceptual spaces

- Constantinescu et al. 2016 (Science): humans navigating abstract 2D feature spaces show hexagonal grid-like signals in entorhinal cortex IDENTICAL to spatial navigation patterns
- Grid cells provide coordinate systems for auditory tone sequences, visual feature spaces, and conceptual navigation
- Subjects with stronger hexagonal modulation perform better at abstract tasks
- The code is stable across sessions (hours and weeks apart)

### The spatial scaffolding hypothesis

- The brain repurposes spatial coding mechanisms (grid/place cells) for organizing non-spatial sequential and conceptual information
- Place cells encode individual concepts; grid cells organize these conceptually
- The spatial network can be activated from purely internal processes (voluntary recall, imagery, planning)
- A unified neural representation model (PNAS 2024) produces place cells, grid cells, and concept cells using identical computational principles

### Time cells

- Hippocampal and entorhinal neurons fire at specific TIMES within a sequence interval (temporal "place fields")
- Parallel to place cells but in the time domain
- Provide position-in-sequence encoding through temporal tuning curves
- The brain integrates both time and space for sequence position

### Toroidal topology

- Gardner et al. 2022 (Nature): grid cell joint activity resides on a 2D torus
- Positions on torus correspond to positions in environment
- Toroidal structure is maintained across environments and sleep
- Naturally handles periodic/cyclic patterns (like modular arithmetic)
- Population dynamics on the torus are stable and context-invariant

### Number-space mapping (SNARC effect)

- Small numbers associated with left space, large numbers with right
- Anterior IPS codes magnitude (how large); posterior IPS codes spatial position (where)
- Ordinal position in WM correlates with space: beginning=left, end=right
- Grid cell network used for both physical navigation and mental number line

**Sources:** Constantinescu et al. 2016 (Science), Gardner et al. 2022 (Nature), Doeller et al. 2010 (Nature)

---

## Competitive Queuing and Motor Planning Buffers

The brain converts parallel planning into serial execution via competitive queuing. This is how multi-digit/multi-word output is produced in the correct left-to-right order.

### Competitive queuing model

- All planned output elements are simultaneously active in a PARALLEL PLANNING layer
- Each element has a different activation level encoding its serial position
- The MOST active element is selected first (winner-take-all)
- After selection, the winner SELF-INHIBITS (activation drops)
- The next most active element then wins
- Repeat until sequence is complete
- Fidelity of competitive queuing correlates with behavioral performance

### Motor planning buffers

- **Phonological output buffer (left posterior inferior frontal sulcus):** holds assembled sound units BEFORE motor articulation. Capacity ~2 seconds of speech (~4-6 items)
- **Sequential structure buffer (bilateral preSMA):** stores the ordered structure/FRAME for the upcoming sequence. Separate from content buffer.
- **SMA proper:** encodes linear sequences and controls motor output. Connected to primary motor cortex and spinal cord.
- **Pre-SMA (rostral):** higher-level planning. Blood flow remains high during imagined movement even when motor cortex is inactive.

### Broca's area subregions for sequence planning

- **BA44 (posterior):** organizes individual motor sequence components BEFORE execution
- **BA45 (anterior):** encodes lexical/hierarchical planning
- Planning is complete before execution begins (hierarchical: first plan structure, then fill content)

### BG chunking for serial execution

- Striatal fast-spiking interneurons fire at beginning and END of learned routines, quiet during execution
- Creates behavioral "chunks" where individual movements are bound into single units
- Dorsolateral striatum controls learned action programs retrieved as complete units
- Dopamine gates sequence initiation via direct/indirect pathways

### Planning-to-execution transition

- Distinct brain activity patterns for planning vs execution phases
- A midbrain-thalamus-cortex circuit generates a "GO" signal that switches from planning to execution mode
- The readiness potential (Bereitschaftspotential): slow electrical buildup 1500ms before movement, from SMA and premotor cortex
- Early component (~1500-400ms): bilateral, reflects preplanning
- Late component (~400-0ms): primary motor cortex, reflects immediate execution preparation

### Coarticulation (lookahead planning)

- Forward coarticulation: articulation of a speech segment is affected by UPCOMING segments
- Demonstrates that the speaker plans several phonemes ahead before starting to speak
- Motor planning involves syllable-level planning (above phoneme level)
- The brain resolves all dependencies before serial execution begins

**Sources:** Bullock 2004, Bohland et al. 2010 (Frontiers HN), Fujii & Graybiel 2003 (PMC4523429), Soldado-Magraner et al. 2024

---

## Position Selection for Serial Output

Based on empirical fMRI, single-neuron, and lesion studies.

### The hierarchy (NOT motor cortex)

Position selection happens UPSTREAM of motor cortex in a distributed system:

1. **PFC encodes serial order**: each element gets a rank (activation strength).
   Neurons respond differently to the same item at position 1 vs 2 vs 3.
   Relative ranking, not absolute position. The next element = highest rank.
2. **Pre-SMA selects "what comes next"**: 71% of neurons are order-sensitive.
   Implements the position-pointer function. Lexical/item SELECTION.
3. **SMA-proper (posterior)**: linear sequence encoding and motor control.
4. **PPC maintains position information**: encodes both current AND future
   targets in parallel (not just the immediate next). Spatial position
   as proxy for temporal order.
5. **BG gates WHEN to release** (temporal gating, NOT element selection).
   Direct pathway = Go (release), indirect = NoGo (hold).
   Both fire at sequence initiation/termination, not during.
6. **Cerebellum provides timing context**: when to expect next element.
7. **Motor cortex is a PASSIVE output stage**: receives pre-selected,
   rank-weighted input. Activity = linear combination of constituents,
   strongest weight on the current element. Does NOT re-select.

### Output gating vs element selection (critical distinction)

- **Element selection**: "which information to prepare next" (PFC, pre-SMA)
- **Output gating**: "when this information can influence behavior" (BG, striatum)
- These are DIFFERENT functions, not the same

### Competitive queuing implementation

- Planning layer (PFC/parietal): all elements active in parallel, graded activation
- Competitive choice layer (striatum/premotor): winner-take-all via lateral inhibition
- Suppression: winner inhibits its own planning-layer representation
- Output: winning node sends selected element to motor cortex

### Sources
- Baldauf et al. 2008, PMC (PPC encodes both goals in double-reach sequences)
- Averbeck et al. 2003 (SMA/pre-SMA order-sensitive neurons)
- Averbeck et al. 2006 (prefrontal cortex serial order encoding)
- Chatham et al. 2014, PMC (corticostriatal output gating from WM)
- Cisek & Kalaska 2005 (motor cortex competitive selection)

---

## TBT Reference Frames and Efference Copies

Based on Hawkins et al. 2018, Lewis et al. 2019, Hawkins & Ahmad 2016.

### The location signal (L6 grid cells)

Each cortical column maintains a LOCATION representation in Layer 6 using grid
cell-like neurons. This encodes "where in the reference frame am I" — for
spatial objects, this is position on the object's surface; for sequences,
this is position-in-sequence.

- L6 cells have grid cell-like properties (multi-scale, path-integrable)
- Multiple grid cell modules at different scales provide unambiguous position
- The location code is ALLOCENTRIC (object-centered, not viewer-centered)
- For sequences: position within the sequence IS the location

### Efference copy as displacement vector

When the motor system acts (produces an output), L5 sends a DISPLACEMENT
signal through the thalamus to L6 of sensory columns. This is NOT the
content of the output — it's the POSITION CHANGE.

- Pathway: Motor cortex L5 → higher-order thalamus → sensory cortex L6
- The displacement operates in reference frame coordinates
- For sequences: displacement = "advance one position" (not "I produced digit X")
- L6 updates: new_location = old_location + displacement (path integration)

### The prediction cycle

1. L6 grid cells encode current position in reference frame
2. L6 → L4: position predicts what sensory input to expect
3. Motor action occurs (output produced)
4. Efference copy (displacement) arrives at L6 via thalamus
5. L6 updates position via path integration
6. Updated L6 → L4: predicts next position's expected input
7. New sensory input arrives at L4, compared against prediction
8. Error = input - prediction (standard PC)

### HTM cell selection (position via context)

In HTM, all cells in a mini-column receive the same feedforward input.
WHICH cells activate depends on context (previous input, sequence position):
- Predicted cells: only they activate (sparse, contextual)
- No predicted cells: all activate (ambiguous/learning state)
- Different active cells = different position in the sequence
- Basal dendrites recognize preceding patterns (sub-threshold depolarization)

### Displacement cells (L5)

L5 cells compute the displacement vector from:
- Current location (from L6)
- Motor command (from motor cortex)
displacement = f(motor_command, current_location_context)

They send this to thalamus (for other columns) and to L6 (for local update).

### Multi-scale grid modules for sequences

- Module 1: position within single digit (0-9 scale)
- Module 2: position within a number (tens, hundreds)
- Module 3: position within an expression (operand, operator, result)
- The combination creates a unique code for any position

### Extension to abstract domains

Grid-like codes emerge in non-spatial domains (Constantinescu et al. 2016):
- Perceptual spaces, conceptual/semantic spaces, social hierarchies
- Mathematical reasoning uses the same spatial navigation circuitry
- Reading/producing language = sensorimotor navigation through sentence space

**Sources:**
- Hawkins et al. 2018, Frontiers Neural Circuits (grid cell framework)
- Lewis et al. 2019, Frontiers Neural Circuits (locations in neocortex)
- Hawkins & Ahmad 2016, Frontiers Neural Circuits (thousands of synapses)
- Nature Neuroscience 2025 (grid cells track movement despite reference frame switches)

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
