# Cortical Layers & Sublaminae — Full Primary-Source Research

**What this is.** The complete, verbatim output of a six-thread parallel primary-source review (gathered 2026-07-09) into
the neocortical layers and sublaminae that the typical Thousand Brains Theory explanation smooths over — undertaken to
inform a from-scratch TBT cortical-column implementation. Each thread did 12–31 web searches + primary-source fetches and
returned a dense, cited writeup with uncertainty flags. **A condensed synthesis + the whole-column build plan lives in
`src/tbt/column.py` (PART A / PART B); THIS file is the full, unabridged research so nothing is lost.**

Threads: (1) Layers 2/3 · (2) Layer 4 · (3) Layer 5 · (4) Layer 6 · (5) Layer 1 + interneurons + real inter-laminar wiring
+ thalamic core/matrix · (6) TBT's own layer mapping + what it omits.

---
---

# THREAD 1 — Neocortical Layers 2 and 3

**Scope note / honesty flag up front.** The single most important fact for a modeller is that "how distinct L2 and L3 are"
is *species-dependent and partly unresolved*. In **rodents** the modern consensus is that L2/3 pyramidal cells form a
**continuum** with depth, not two discrete classes; in **primates** L2 vs L3 are architectonically distinct and L3 is
further sublaminated (3A/3B/3C). Much of the classic "L2/3 = one thing" literature is rodent-derived, and the TBT/Numenta
account inherits that simplification. I have tried to separate what is established, what is species-specific, and what is
genuinely uncertain.

## 1. Why L2 and L3 are lumped as "L2/3" — history, anatomy, rodent vs primate

**The proximate reason is architectonic, and it is a rodent fact.** "In rodents, there is no clear architectonic boundary
between layer 2 (L2) and L3 of the cortex, and these layers are therefore often referred [to] as L2/3" ([Luo et al. 2017,
*Front. Neuroanat.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5742574/)). Nissl-stained rodent supragranular cortex simply
does not show a crisp L2/L3 line the way primate cortex does, so the combined label was adopted for practical convenience
and then propagated into the physiology and modelling literature.

**In rodents the lumping is now defensible on a deeper ground: L2/3 is a gradient, not two types.** A systematic study of
mouse V1 found that "L2/3 pyramidal cells do not display discrete subtypes… instead, their multiple functional and
structural properties systematically correlate with their depth, forming a continuum rather than discrete subtypes"
([Weiler et al. 2023, *Cerebral Cortex*; preprint 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC10068292/)). Concretely:
the **apical (not basal) dendritic tree** varies with pial depth; **lower L2/3 cells receive more L4 input** while **upper
L2/3 cells receive proportionally more intralaminar input**; and **deeper cells are more visually responsive and more
contralateral-eye-driven** — all covarying smoothly, "but this variability does not indicate clusters." So "L2/3" in rodent
is a reasonable label for one continuously-varying population.

**In primates the lumping hides real structure, and the field knows it.** Harris & Shepherd's canonical review explicitly
warns that "although the supragranular layers are often studied as a single entity, hodological distinctions between
sublayers are an important aspect of inter-areal connectivity," and note "increasing distinctions between L2 and L3"
([Harris & Shepherd 2015, *Nat. Neurosci.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/)). In **alert macaque V1**,
single-unit recordings found striking physiological separations between L2 and L3: "almost all layer 2 cells generated
small spikes (81% ≤ 0.6 mV; median 0.5 mV) while the reverse was true for layer 3 (79% > 0.6 mV; median 1.2 mV)"; layer-3
classical receptive fields were much smaller (median 10.3 vs 23.5 min of arc); layer 2 was more spontaneously active; and —
most dramatically — "in layer 3, 41.5% of the cells were direction selective (DI > 0.5), while in layer 2 there were no
cells with DI > 0.4" ([Economides et al. 2008, *PMC2479568*](https://pmc.ncbi.nlm.nih.gov/articles/PMC2479568/)). Those
authors explicitly complain that "data from layers 2, 3A and 3B are routinely combined, so that differences among them are
obscured," and that "the anatomical distinctions are not reflected in the physiological literature."

**Human = the extreme case.** Human supragranular cortex is disproportionately expanded and *diversified*: single-cell
transcriptomics finds "five [glutamatergic] t-types in human supragranular MTG … versus three t-types each in mouse," with
"pronounced gradients as a function of cortical depth," and "the deep portion of layer 3 contained highly distinctive cell
types" ([Berg et al. 2021, *Nature*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8494638/)).

**Bottom line for Q1:** the "L2/3" abbreviation is (i) originally an architectonic convenience born of rodent
cytoarchitecture, (ii) genuinely justifiable in rodent because the population is a depth-continuum, but (iii) an
oversimplification in primate/human, where L2 and L3 differ physiologically and L3 is sublaminated.

## 2. What layer 2 does vs layer 3 — cell types, inputs, outputs, computational role

Both L2 and L3 principal cells are **excitatory pyramidal neurons of the intratelencephalic (IT) class** — "thin-tufted"
apical dendrites, axons confined to cortex and striatum, callosal projections defining the class ([Harris & Shepherd
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/)). They are *not* PT (pyramidal-tract, L5) or CT (corticothalamic,
L6) neurons. The differences between L2 and L3 are gradations within the IT family plus differences in input source and
projection distance.

**Distinct inputs (the clearest L2-vs-L3 divergence):**
- **L3** receives "core-type [driver] thalamocortical input on their basal dendrites … matrix-type and higher-order
  cortical input on their apical dendrites," plus "many inputs from local L4 ITs." **L2** receives "matrix-type input from
  POm" and "little core-type input" ([Harris & Shepherd 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/)). So L3 is
  more tightly coupled to the feedforward sensory (L4/core-thalamic) stream; L2 is biased toward matrix-thalamic and
  cortical feedback.
- **Firing regime differs:** "L2/3 ITs fire sparsely in vivo … with L2 ITs exhibiting sparser firing than L3 ITs"
  ([Harris & Shepherd 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/)).

**The L4 → L3 → L5 spine of the microcircuit:** "Layer 4 spiny neurones make a dense, topographically precise projection to
layer 3 (and to upper layer 5)"; then "layer 3 pyramidal cell axons ramify densely in layers 3 and 2 and send a descending
axon to layer 5" ([Thomson & Lamy 2007, *Front. Neurosci.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/)). Notably,
"the outputs of layer 2 pyramids remain to be studied in detail" — i.e., L3 is the well-characterised feedforward relay,
and L2's output is comparatively under-studied.

**Outputs — two functionally different jobs:**
1. **Within-column descending output to L5.** "The L2/3 → 5A/B pathway appears to be a particularly prominent and
   consistent feature of cortical circuits across areas and species," and L3 ITs preferentially target PT (L5) neurons over
   other IT neurons (in motor cortex) ([Harris & Shepherd 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/)). This is
   the main route by which the superficial processing result reaches the cortical *output* stage (L5 → subcortex).
2. **Long-range corticocortical feedforward to higher areas.** In the hierarchy framework, "feedforward projections
   originate primarily from the supragranular layers (layers 2/3) and target layer 4" of the higher area, whereas feedback
   comes from infragranular layers ([Felleman & Van Essen 1991; Markov et al. 2014, summarised
   in](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2017.00071/full)). **Which of L2/L3
   supplies this is sublaminar (see §3): deep L3 is the feedforward source; L2 and upper L3 lean feedback.**

**Computational division of labour (best current reading):**
- **L3** = the principal **feedforward integrator/output**: reads the L4 sensory pattern, sharpens it (smaller RFs, sharper
  orientation/direction tuning in primate V1), and exports it both down to L5 and forward to the next area's L4.
- **L2** = a more **associative/feedback-biased, sparser** compartment, dominated by matrix-thalamic and top-down cortical
  input onto apical dendrites, with L2 marginal cells forming an atypical subpopulation (§3). In rodent barrel/temporal
  cortex, L2 is described as "a primary target of ipsilateral feedback-type cortical projections" ([discussed around Luo et
  al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5742574/)).

## 3. Sublaminae of L2 and L3

**Yes — especially in primates.** The superficial cortex is not two flat sheets.

**Primate V1 is the textbook case.** "The superficial region of V1 is divided into … layer 1, and three cell-dense regions
… designated layers 2, 3A and 3B. Layers 2 and 3A receive little thalamic input, and they do not receive direct inputs from
layer 4C, in contrast to layer 3B that receives a massive 4C input" ([context via macaque V1
studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC2479568/); [Webvision, *Primary Visual
Cortex*](https://www.ncbi.nlm.nih.gov/books/NBK11524/)). Layer 3 is further split "into three sublayers, 3A, 3B, and 3C,
based on differences in cell size and density in chimpanzees, macaques, and humans," with 3A "small- and medium-sized cells
evenly distributed." Callaway's local-circuit anatomy resolves macaque V1 into "layers 2/3A, 3B, 4A, and 4B," and shows
exquisite sublaminar targeting — "spiny stellate neurons in layer 4Cβ specifically target layers 4A and 3B but do not
branch in the intervening layer 4B" ([Callaway 1998, *Annu. Rev. Neurosci.* / Yabuta &
Callaway](https://www.cns.nyu.edu/~tony/vns/readings/callaway-1998.pdf)).

**The functionally decisive sublaminar split is feedforward vs feedback within L3.** Barone et al. 2000 (primate):
**"feedforward projections tend to arise from L5 and deep L3, adjacent to L4, and project to L4, while feedback projections
arise from L6 and upper L3 [and L2], and project to L1/L2, upper L3, and L6"** (summarised in [multiple hierarchy
reviews](https://pmc.ncbi.nlm.nih.gov/articles/PMC11952746/)). This gives a clean, load-bearing rule:
- **Deep L3** (bordering L4) = **feedforward output** to higher-area L4.
- **Upper L3 + L2** = **feedback-biased**, projecting to L1/L2 of lower areas.

**Callosal vs ipsilateral is another sublaminar/subset axis.** "A different subset of layer 3 pyramidal neurons in each
region projects to the homotopic region of the contralateral hemisphere," distinct from the ipsilateral-projecting subset
([callosal/IP projection studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977376/)); in V1 the callosal cells cluster at
the vertical-meridian representation.

**Deep L3 harbours a distinctive long-range cell.** In human, "the deep portion of layer 3 contained highly distinctive
cell types, two of which express a neurofilament protein (NEFH/SMI-32) that labels long-range projection neurons in
primates and are selectively depleted in Alzheimer's disease"; the authors argue "increased cellular diversity in deep L3
may enhance efficiency of feedforward signal processing connecting distant regions of the expanded primate neocortex"
([Berg et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8494638/)). One transcriptomic type (FREM3) "varies
continuously in all phenotypes across layers 2 and 3" — i.e., part of the L2/L3 axis is a gradient even where discrete
deep-L3 types also exist.

**L2 itself has substructure.** In rodent temporal cortex an "L2 marginal neuron" (L2MN) population sits at the L1/L2 border
with "oblique apical dendrites or … no obvious apical dendrites," "basal dendrites invad[ing] L1 extensively," and distinct
intrinsics ("higher firing rate, larger sag ratio, higher input resistance," less-hyperpolarised rest) versus ordinary
L2/3 regular-spiking cells ([Luo et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5742574/)). So "L2 proper" is not
homogeneous either.

**Matrix vs core thalamic input maps onto the sublaminae:** matrix (e.g. POm) terminates superficially (L1/upper L2, apical
tufts), core drivers terminate on L4 and deep-L3 basal dendrites ([Harris & Shepherd
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/)) — consistent with deep L3 = feedforward/driver-coupled, upper
L2/3 = matrix/feedback-coupled.

## 4. Inputs to L2/3 and outputs from L2/3 (consolidated)

**Inputs:**
- **From L4 (main feedforward driver):** dense, topographically precise L4→L3/upper-L5 projection ([Thomson & Lamy
  2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/)); lower L2/3 gets more L4 than upper L2/3 ([Weiler et al.
  2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10068292/)).
- **From L1 / apical feedback:** higher-order cortical feedback and matrix-thalamic axons synapse on L2/3 apical tufts in L1
  ([Harris & Shepherd 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/)).
- **From thalamus:** matrix (POm-type) preferentially to **L2** and superficial apical dendrites; core drivers to **deep
  L3** basal dendrites (in some areas/species) ([Harris & Shepherd 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/);
  [Thomson & Lamy 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/)).
- **From other columns (horizontal):** long-range intralaminar L2/3→L2/3 connections; upper L2/3 is enriched for
  intralaminar input ([Weiler et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10068292/)).

**Outputs:**
- **To L5A/B (within column):** the prominent, conserved L2/3→L5 pathway; L3 IT → L5 PT preferentially ([Harris & Shepherd
  2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/); [Thomson & Lamy 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/)).
- **Feedforward corticocortical to higher-area L4:** from **deep L3** (Barone rule).
- **Feedback corticocortical to lower-area L1/L2:** from **L2 + upper L3** ([Barone et al. 2000, via
  review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11952746/)).
- **Callosal:** a distinct L3 subset to homotopic contralateral cortex ([callosal/IP
  studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977376/)).
- **Local horizontal / cross-columnar:** dense L2/3↔L2/3 lateral axons (the anatomical substrate TBT calls "voting").

Note the important asymmetry the modeller should keep: **L2/3 is the corticocortical output; L5 is the subcortical output.**
L2/3's "output" role is to (a) drive L5 and (b) send feedforward to the next cortical area — these are two different
populations, not one undifferentiated "output."

## 5. Inhibitory interneurons of L2/3 and their computational roles

The canonical MGE/CGE division ([Kepecs & Fishell 2014, *Nature*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4349583/);
[interneuron reviews](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.929469/full)):

- **PV⁺ fast-spiking basket cells** — perisomatic targeting of pyramidal somata/proximal dendrites; "exert strong
  perisomatic inhibition … regulate synchronous and oscillatory activity" (gamma); implement **gain control and precise
  spike timing / feedforward inhibition** ([Thomson & Lamy 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/); [monkey
  PFC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2693619/)).
- **PV⁺ chandelier (axo-axonic) cells** — target the **axon initial segment**; powerful control of pyramidal output/veto
  ([Thomson & Lamy 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/)).
- **SST⁺ Martinotti cells** — "fine axons that ramify densely within, and in all layers superficial to the layer of
  origin," ascending to target **distal apical dendrites/tufts in L1**; frequency-**facilitating** recruitment; implement
  **dendritic inhibition, lateral/surround suppression, and control of top-down input to the apical tuft** ([Thomson & Lamy
  2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/); [SST role](https://www.nature.com/articles/ncomms13664)).
- **VIP⁺ (CGE) interneurons** — "the largest fraction (~60% of VIP⁺ cells in L2/3) does not directly contact excitatory
  cells; instead it primarily targets inhibitory SST⁺ cells, forming a **disinhibitory circuit**" ([VIP
  disinhibition](https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2022.811484/full)). VIP
  cells are driven by long-range top-down/neuromodulatory input (e.g. motor cortex during whisking), so they are the
  **context/attention gate** that releases pyramidal apical dendrites by suppressing SST ([Harris & Shepherd
  2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/)).
- **Neurogliaform (NGF) cells** — dense local axon, **volume transmission** of GABA producing **slow GABA_A + GABA_B**
  inhibition; "inhibit layer 2/3 pyramids," couple via gap junctions; provide diffuse, slow, blanket inhibition ([Thomson &
  Lamy 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/); [monkey PFC: NGFCs are the "initial-adapting" class, ~22
  Hz](https://pmc.ncbi.nlm.nih.gov/articles/PMC2693619/)).
- **Primate-specific: double-bouquet cells** — "found within layers II and III … long descending bundles of axon
  collaterals that are columnar … target pyramidal cells within a very narrow space," a "specialization of **minicolumn
  inhibition** within the primate order," and **absent in rodents** ([DeFelipe / del Río; comparative
  review](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/neuro.05.003.2010/full)).

**Predictive-coding functional assignment (a competing view of L2/3 the modeller should note):** superficial pyramids are
widely modelled as **prediction-error units**; two-compartment L2/3 pyramids with "feedforward input targeting the soma and
top-down feedback reaching the distal apical dendrite," combined with **PV (divisive/perisomatic), SST (subtractive/
dendritic), and VIP (disinhibitory)** inhibition, compute sign-specific and uncertainty-weighted prediction errors
([Mikulasch/uncertainty-modulated PE, *eLife* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12140629/); [predictive-coding
V1 L2/3 model 2025](https://www.biorxiv.org/content/10.1101/2025.11.01.686040v1)). This is a *different* computational story
for the same layer than "object pooling."

Primate caveat: monkey L2/3 shows **8 morphological / 3 physiological interneuron types** and lacks the stuttering/bursting
and delayed-spike NGF phenotypes seen in rodent — "direct translation of classification schemes … might be inappropriate,"
and the GABAergic fraction is larger in primate ([Zaitsev/Krimer/Lewis, monkey
PFC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2693619/)).

## 6. What the Thousand Brains / Numenta account smooths over

**What the TBT model actually says.** In the foundational Numenta paper, L4 is the **input layer** (feedforward sensory +
location), and **L2/3 is the "output layer"** whose "set of active cells … represents objects"; each output cell "pools
over multiple feature/location representations in the input layer," the object representation "remains active over multiple
movements" (temporal pooling), and "the modulatory input to cells in the output layer comes from other output cells
representing the same object … within the column as well as from neighboring columns via long-range lateral connections"
(voting). Location is supplied by **L6a** ([Hawkins, Ahmad & Cui 2017, *Front. Neural
Circuits*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5661005/)). Crucially, **they never distinguish L2 from L3** — they
"consistently refer to them together as 'L2/3'." The later Thousand Brains Project engineering paper is explicit that "we
do not need to strictly adhere to all biological details" and object models are "explicit graphs in 3D Cartesian space"
([TBP 2024](https://arxiv.org/html/2412.18354v1)). To their credit, the 2017 paper flags the relevant caveat itself: "cells
we describe as residing in separate layers may actually intermingle."

**Given the anatomy above, here is what that single-"output-layer" abstraction omits or flattens:**

1. **It erases the L2 vs L3 division of labour.** Real L2 and L3 differ in thalamic input (matrix/POm vs core), firing
   sparsity (L2 sparser), receptive-field/tuning properties (primate V1), and feedback vs feedforward bias ([Harris &
   Shepherd 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/); [Gur/Snodderly
   V1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2479568/)). TBT's "object pool" is one population; biology has (at least)
   two.

2. **It conflates two anatomically separate outputs.** TBT calls L2/3 "the major output of the column." But (a) the
   *within-column* output is L2/3→**L5**, and (b) the *feedforward corticocortical* output specifically comes from **deep
   L3**, while (c) **L2 + upper L3** are feedback-biased ([Barone et al.
   2000](https://pmc.ncbi.nlm.nih.gov/articles/PMC11952746/)). The model's "stable object → higher region" export and its
   "lateral voting" are, in tissue, different cell populations at different sublaminar depths — not one homogeneous sheet.

3. **It ignores the L3 sublaminae (3A/3B/3C, 4B) and their thalamic wiring.** In primate V1, 3B gets massive 4C drive while
   2/3A get almost none; deep L3 vs upper L3 differ in feedforward/feedback role and in harbouring distinctive SMI-32
   long-range cells ([Berg et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8494638/); [Callaway
   1998](https://www.cns.nyu.edu/~tony/vns/readings/callaway-1998.pdf)). A single "output layer" cannot represent this
   feedforward-driver-vs-associative gradient.

4. **It is implicitly rodent-flavoured and misses primate-specific structure.** The "one canonical column" gloss is most
   valid in rodent (where L2/3 is a depth-continuum) and least valid in primate/human, which add clearer L2/L3 separation,
   L3 sublamination, primate-only **double-bouquet minicolumn inhibition**, and human **supragranular expansion/
   diversification** ([Luo et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5742574/); [DeFelipe comparative
   review](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/neuro.05.003.2010/full); [Berg et al.
   2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8494638/)).

5. **It reduces a rich inhibitory microcircuit to abstract "modulatory" connections.** The real L2/3 computation is *shaped*
   by PV (gain/timing), SST (dendritic/surround suppression), VIP (top-down disinhibitory gating), NGF (slow blanket
   GABA_B), and — in primate — double-bouquet columnar inhibition. TBT's temporal-pooling/voting model folds all of this
   into excitatory pooling plus a generic modulatory bias, discarding the specific dendritic and disinhibitory computations
   that these cells perform ([Thomson & Lamy 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/); [VIP
   disinhibition](https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2022.811484/full);
   [monkey PFC interneurons](https://pmc.ncbi.nlm.nih.gov/articles/PMC2693619/)).

6. **It commits to one functional story where the field has (at least) two.** TBT: L2/3 = stable-object / temporal-pooling /
   voting. A large parallel literature: L2/3 = **prediction-error / uncertainty computation** via compartmentalised pyramids
   and sign-specific PV/SST/VIP inhibition ([Bastos et al.; uncertainty-modulated PE
   2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12140629/)). These are not obviously the same computation, and the TBT
   account does not engage the error-coding role that the apical-dendrite/feedback wiring of L2 (specifically) seems built
   for.

**Fair caveats on the critique.** (a) The Numenta authors *explicitly* frame the mapping as loose and acknowledge layers
"may intermingle," so some of this "smoothing over" is deliberate abstraction, not error. (b) In **rodent**, treating L2/3
as one depth-continuum is genuinely defensible ([Weiler et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10068292/));
the abstraction fails hardest in primate/human. (c) TBT's "voting via long-range L2/3 horizontal connections" *does*
correspond to a real, prominent anatomical feature — the critique is that TBT assigns *only* that role to L2/3 and to
*both* sublayers indiscriminately.

**Practical implication for your column model.** If you want biological fidelity beyond rodent: split the "output layer"
into at least (i) a **deep-L3 feedforward-export** population (driven by L4, projecting to the next area's L4 and down to
L5) and (ii) an **L2/upper-L3 feedback-associative** population (matrix-thalamic + top-down apical input, projecting back to
lower areas' L1/L2), with **lateral voting** carried by horizontal L2/3 axons across both — and treat the PV/SST/VIP/NGF
quartet as the operators (gain, dendritic gating, disinhibitory context, slow normalisation) rather than a single
"modulatory" term.

**Uncertainty flags:** (1) Whether L2/L3 are "discrete vs continuous" is unresolved and species-dependent — rodent data
favour a continuum, primate data favour discreteness with 3A/3B/3C sublamination; treat any universal claim skeptically.
(2) The Barone-2000 "deep-L3 feedforward / upper-L3+L2 feedback" rule is a primate generalisation; mouse shows weak-to-no
supragranular laminar segregation of FF vs FB. (3) L2 pyramidal *outputs* are genuinely under-characterised ("remain to be
studied in detail," Thomson & Lamy), so the L2-specific claims are the least certain in this thread.

---
---

# THREAD 2 — Neocortical Layer 4

*Note on method: several primary PDFs (Harris & Shepherd 2015; Douglas & Martin 2004) would not parse as text, so a few
claims attributed to them are **paraphrases via secondary summaries** and are flagged as such — treat those specific
wordings as approximate, not verbatim quotes.*

## 1. The SUBLAMINAE of Layer 4

### Primate V1 (macaque) — the most elaborated L4 in the brain

Primate striate cortex (V1) has the thickest, most subdivided L4 of any cortical area, and it is the canonical evidence
that "L4" is not a monolith. The classic Hässler/Brodmann-derived scheme (elaborated by **Lund** and **Callaway**) divides
it into **4A, 4B, 4Cα, and 4Cβ**, each with distinct thalamic input, cell populations, and projection targets.

**4Cα (magnocellular-recipient) vs 4Cβ (parvocellular-recipient).** The magnocellular (M) division of the LGN terminates in
**4Cα**; the parvocellular (P) division terminates in **4Cβ**. This is the foundational Lund/Callaway result: "M and P
pathways converge on V1, where they segregate their inputs into layers 4Cα and 4Cβ, respectively" ([Yabuta & Callaway, *J
Neurosci* 1998, PMC6792868](https://pmc.ncbi.nlm.nih.gov/articles/PMC6792868/); [Callaway, *Annu Rev Neurosci*
1998](https://pages.ucsd.edu/~msereno/201/readings/01.05-LocalCortCirc.pdf)).

**Projection targets differ sharply between the two 4C sublaminae** — the spiny stellate cells in each sublayer send their
ascending axons to *different* upper-layer targets ([Yabuta & Callaway 1998,
PMC6792868](https://pmc.ncbi.nlm.nih.gov/articles/PMC6792868/)):
- **4Cα** spiny stellates project preferentially up to **layer 4B** and to the CO **blobs** in L2/3. "More than half of all
  layer 2/3 synaptic boutons from these cells are located within 50 μm of the center of a blob."
- **4Cβ** spiny stellates project through 4B (without branching there) to **layers 4A and 3B** (upper L2/3), without a
  strong blob/interblob preference.
- Quantitatively: **"4Cα cells provide ~5× more synapses than 4Cβ cells to layer 4B, whereas 4Cβ cells provide ~5× more
  synapses than 4Cα cells to layer 2/3."** So the M-stream is biased toward 4B (→ motion/MT), the P-stream toward
  superficial layers (→ form/colour).

**4B** is functionally the *output* sublamina for the magnocellular stream. It contains large pyramidal cells (including
giant **Meynert cells** near the 4B/5 or 5/6 border) and projects out of V1 to **area MT/V5** (motion) and to V2 thick
stripes ([Local circuits of 4B neurons, *J Neurosci* 2017](https://www.jneurosci.org/content/37/2/422); [Four projection
streams V1→V2, PMC2909028](https://pmc.ncbi.nlm.nih.gov/articles/PMC2909028/)). 4B is myelin-dense (the **stria of
Gennari**, which gives "striate" cortex its name). Note: 4B behaves more like a superficial output layer than a
thalamorecipient input layer — a first crack in the "L4 = input" identity.

**4A** is a thin sublamina receiving the **koniocellular (K)** pathway plus some P input; the K projection to 4A (and to the
CO blobs) carries mostly **S-cone / blue-yellow** signals ([Hendry & Reid, *Annu Rev Neurosci*
2000](https://www.cns.nyu.edu/~tony/vns/readings/hendry-reid-2000.pdf)). 4A has a distinctive honeycomb CO pattern. So V1
has **three** parallel thalamic input streams (M→4Cα, P→4Cβ, K→4A + blobs), not one.

**Take-home:** in the one area where L4 has been dissected most finely, it is *four* sublaminae carrying *three* segregated
thalamic streams with *divergent* output targets — the opposite of a single homogeneous "input layer."

### Barrel cortex L4 (rodent S1) — a different kind of sublamination

Rodent whisker/barrel cortex organizes L4 not by depth-sublaminae but by **tangential somatotopic modules**: each large
facial whisker maps 1:1 onto a discrete cytoarchitectonic **"barrel"** in L4 ([Woolsey & Van der Loos
1970](https://pubmed.ncbi.nlm.nih.gov/27086973/); [Petersen, *Neuron*
2007](https://www.cell.com/fulltext/S0896-6273(07)00715-5)). A barrel is an oval of cell-dense **walls** surrounding a
cell-sparse **hollow** (neuropil + thalamocortical afferents + L4 dendrites); adjacent barrels are separated by **septa**.
Lemniscal thalamic (VPM) input drives the barrel hollows/columns; paralemniscal/POm input is associated with septa — i.e.,
two thalamic streams again, spatially interdigitated ([Feldmeyer review, *Front Neuroanat*
2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)). L4 neurons' dendrites and
axons are **confined to their home barrel**, giving a hard columnar quantization of the sensory sheet ([Lübke/Feldmeyer, *J
Neurosci* 2000](https://www.jneurosci.org/content/20/14/5300)).

## 2. The CELL TYPES of Layer 4

L4 excitatory neurons fall on a morphological continuum defined by **how much apical dendrite survives**:

- **Spiny stellate cells** — no (or vestigial) apical dendrite; a radially symmetric, star-shaped, spiny dendritic tree
  **confined to L4**. This is the "specialist" L4 excitatory cell, prominent in primary sensory cortex (V1 4C, S1 barrels)
  ([Callaway 1998](https://pages.ucsd.edu/~msereno/201/readings/01.05-LocalCortCirc.pdf); [Douglas & Martin, *Annu Rev
  Neurosci* 2004](https://www.cns.nyu.edu/~tony/vns/readings/douglas-martin-2004.pdf)). Importantly, spiny stellates are
  **developmentally retracted pyramids**: L4 cells are "all pyramidal in origin," and the apical dendrite is later pruned
  under the influence of sensory/retinal input and local GABAergic interneurons ([Callaway; developmental sculpting, *J
  Neurosci* 2011](https://www.jneurosci.org/content/31/20/7456)).
- **Star pyramids** — an intermediate: a truncated apical dendrite that reaches partway into L2/3. In barrel cortex they
  receive weak input from other layers that spiny stellates do not ([Feldmeyer
  2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)).
- **L4 pyramidal cells** — a full apical dendrite reaching L1. These dominate in some areas and are the *only* L4-type in
  motor cortex (see §4). (Note: a distinct "L4 pyramid" type has not been cleanly separated from star pyramids in all
  studies — [Feldmeyer 2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full).)

**Proportions.** In rat barrel cortex, of synaptically coupled L4 spiny neurons, **~80% were spiny stellate, ~20% star
pyramidal** ([Feldmeyer et al., summarized
2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)). *Flag:* the exact split
is species/area-dependent.

**Intrinsic properties.** L4 spiny stellates and star pyramids are **both regular-spiking (RS)** with essentially
*comparable passive membrane properties* — morphology does not map to a distinct firing class. RS L4 cells fire regular
trains (rat barrel: first ISI ≈ 22.9 ms, second ≈ 50.4 ms) ([Feldmeyer et al. 1999,
PMC2290091](https://pmc.ncbi.nlm.nih.gov/articles/PMC2290091/)). This matters for a column model: **the excitatory L4
population is electrophysiologically fairly uniform** even though morphologically graded.

**Local vs projecting — a key architectural fact.** L4 excitatory cells are essentially **local interneurons of the
excitatory network**: their axons stay within the column (mainly to L2/3 and within L4), and they **lack long-range
corticocortical and subcortical projections**, unlike L2/3 and L5 output cells ([barrel connectivity
review](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)). L4 stellates are
classed as an **IT (intratelencephalic) subclass with local-only axons**. **Exception / flag:** a recently described
corticostriatal L4 pyramidal type in auditory cortex (CS-L4) does send a long-range projection and receives direct thalamic
input, so "L4 never projects out" is not absolute ([Chen et al., *J Neurosci* 2022,
PMC8883864](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883864/)).

L4 also contains **inhibitory interneurons** (notably fast-spiking PV cells) that mediate powerful thalamocortical
**feedforward inhibition**; a small number of FS cells can gate spiny-stellate output ([*J Neurosci* 2006, barrel
FFI](https://www.jneurosci.org/content/26/4/1219)).

## 3. THALAMORECIPIENT ROLE — and the quantitative paradox

L4 is the principal target of **"core"/first-order (driver) thalamic** afferents (LGN→V1, VPM→S1, MGv→A1) — the classic FF
sensory input to cortex ([Douglas & Martin 2004](https://www.cns.nyu.edu/~tony/vns/readings/douglas-martin-2004.pdf);
[Harris & Shepherd 2015](https://www.nature.com/articles/nn.3917)). But the **fraction of L4 synapses that is thalamic is
strikingly small** — the "thalamic paradox," which any faithful column model must confront:

| Area / prep | Thalamic fraction of L4 excitatory synapses | Source |
|---|---|---|
| Cat V1 (spiny stellate) | **~6%** (Ahmed 1994); **<10%** (da Costa & Martin 2011) | [Ahmed et al. 1994](https://pmc.ncbi.nlm.nih.gov/articles/PMC6623786/); [da Costa & Martin, *J Neurosci* 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC6623786/) |
| Rat barrel S1 | **~10–20%** | [Feldmeyer 2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full) |
| Mouse S1 (EM) | **17.2%** of asymmetric synapses in L4 | [Bopp et al. 2017, PMC6596845](https://pmc.ncbi.nlm.nih.gov/articles/PMC6596845/) |
| Mouse M1 (EM) | **12.1%** (≈ half the *absolute* count of S1) | [Bopp et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC6596845/) |
| General cortex (Bruno & Sakmann) | **~15%** of synapses onto cortical neurons | [Bruno & Sakmann, *Science* 2006](https://www.science.org/doi/10.1126/science.1124593) |

**Where the rest comes from** (cat V1 spiny stellate, Ahmed et al. 1994 breakdown): **~45% from L6 pyramidal cells**, **~28%
from other spiny stellate cells**, **~6% from thalamus**, remainder from other sources ([Ahmed et al. 1994, via da Costa &
Martin 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC6623786/)). So the **single largest excitatory input to L4 is feedback
from L6, not the thalamus**, and recurrent L4→L4 excitation is the second largest. *Flag:* the 45/28/6 split is the
widely-cited Ahmed figure; later EM work (da Costa & Martin) revised thalamic to "<10%".

**How so few synapses drive cortex:** thalamocortical synapses are individually weak, but the convergent inputs are
**numerous and synchronous**, so thalamus can drive L4 (and even L5/6) *without* intracortical amplification ([Bruno &
Sakmann 2006](https://www.science.org/doi/10.1126/science.1124593)). The intracortical (L6 + recurrent L4) inputs act as
**amplification/gain** on the thin but synchronous thalamic drive ([Feldmeyer
2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)).

**The L4 → L2/3 feedforward projection** is the canonical first intracortical step. It is strong, columnar, and **largely
unidirectional** (L4 drives L2/3 far more than L2/3 drives L4). In rat barrel cortex, L4→L2/3 connectivity is ~**10–15%**,
uEPSPs ~0.6–1.0 mV, over 200–400 μm axonal reach onto basal dendrites ([Feldmeyer
2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)). Intra-L4 recurrence is
even denser (**25–36% L4→L4 connectivity**, uEPSP ~1.6 mV — the highest in neocortex) ([Feldmeyer
2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)).

## 4. GRANULAR vs AGRANULAR cortex — the biggest problem for a uniform column theory

**The fact.** A cytoarchitectonic L4 ("granular" layer, named for its densely packed small granule/stellate cells) is
prominent in **primary sensory** cortex, thins in association cortex, and is **classically absent** in **agranular** cortex
— **primary motor cortex (M1)** and much of prefrontal/cingulate/limbic cortex. Motor cortex is the textbook "agranular"
area ([Harris & Shepherd 2015](https://www.nature.com/articles/nn.3917); [Yamawaki et al., *eLife* 2014,
PMC4290446](https://pmc.ncbi.nlm.nih.gov/articles/PMC4290446/)).

**What "replaces" it — the modern revision.** Two lines of work argue L4 is not truly *absent* in M1, only
cytoarchitecturally invisible:

- **Yamawaki, Harris & Shepherd (2014)** identified a thin band of excitatory neurons at the **L3/5A border** of mouse M1
  with **prototypical L4 input–output circuitry**: they receive thalamic input and send a **largely unidirectional
  excitatory projection to L2/3** — "L4 in M1 has been 'lost' only at the level of cytoarchitecture but not of cellular
  connectivity." But these M1 "L4" cells are **all pyramidal — no star pyramids or spiny stellates** — and the thalamic
  drive is weaker and more diffuse than in S1 ([Yamawaki et al. 2014,
  PMC4290446](https://pmc.ncbi.nlm.nih.gov/articles/PMC4290446/)).
- **Bopp et al. (2017, EM)** confirmed M1 *does* receive thalamic synapses in its L4-equivalent, but only **12.1%** of
  asymmetric synapses (vs 17.2% in S1) and **~half the absolute number** — quantitatively a diminished L4 ([Bopp et al.
  2017, PMC6596845](https://pmc.ncbi.nlm.nih.gov/articles/PMC6596845/)).

**Where thalamic drive goes in agranular cortex.** With a weak/absent granular L4, driver thalamic input in agranular areas
terminates more heavily in **L3 and L5** (and L1 via matrix/higher-order thalamus). More generally, **Constantinople & Bruno
(2013)** showed — even in *granular* S1 — that thalamus activates **L5/6 directly and in parallel with L4**, at the same
latency, so **L4 is not an obligatory first relay**: "thalamus activates two separate, independent strata of cortex in
parallel," and "L4 is not an obligatory distribution hub" ([Constantinople & Bruno, *Science*
2013](https://www.science.org/doi/10.1126/science.1236425)).

**Implication for a uniform column theory.** Harris & Shepherd's resolution is **serial homology + "themes and
variations"**: the same basic laminar excitatory scheme (an input-recipient stage → superficial IT → deep PT/CT stages)
recurs everywhere, but with **area- and species-specific variation** — L4 can expand (V1 4C), split (V1 sublaminae), or
shrink to near-invisibility (M1) while its *circuit role* persists ([Harris & Shepherd
2015](https://www.nature.com/articles/nn.3917); [Douglas & Martin
2004](https://www.cns.nyu.edu/~tony/vns/readings/douglas-martin-2004.pdf)). *Flag: the Harris & Shepherd wording here is
paraphrased from a secondary summary.* The honest reading for a modeler: a **single uniform "L4-input" primitive is a
defensible abstraction of the input-recipient circuit stage**, but the biology says that stage is (a) not always granular,
(b) not always the sole/first thalamic target, and (c) morphologically different where it does exist.

## 5. INPUTS and OUTPUTS of L4

**Inputs to L4:**
- **Core/first-order thalamus (driver, FF sensory):** the defining input, but only ~6–20% of synapses (§3). Terminates on
  proximal dendrites of spiny stellates ([da Costa & Martin 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC6623786/); [Bruno
  & Sakmann 2006](https://www.science.org/doi/10.1126/science.1124593)).
- **L6 corticothalamic/corticocortical pyramids (feedback):** the **largest** excitatory input (~45% in cat V1),
  modulatory, onto basal/proximal dendrites — a within-column gain/context signal ([Ahmed et al. 1994 / da Costa & Martin
  2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC6623786/)).
- **Recurrent L4→L4:** dense (~25–36% connectivity), the amplifier ([Feldmeyer
  2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)).
- **L2/3 feedback:** relatively **weak** — the L4↔L2/3 loop is asymmetric, dominated by the ascending direction ([Yamawaki
  et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4290446/); [Feldmeyer
  2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)).
- **Local inhibition:** strong thalamus-driven feedforward inhibition via FS/PV interneurons ([*J Neurosci*
  2006](https://www.jneurosci.org/content/26/4/1219)).

**Outputs of L4:**
- **Primary: L4 → L2/3** (columnar, strong, unidirectional) — the canonical FF step ([Feldmeyer
  2012](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2012.00024/full)).
- **Secondary: L4 → L5/L6** (and within-L4). In V1 the 4C stellates specifically target sublaminae (4Cα→4B/blobs;
  4Cβ→4A/3B) ([Yabuta & Callaway 1998](https://pmc.ncbi.nlm.nih.gov/articles/PMC6792868/)). General flow: **L4 → L2/3 →
  L5/6 → out of cortex**.
- **Essentially no long-range output** (local IT cells; rare corticostriatal exception noted in §2)
  ([PMC8883864](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883864/)).

The **canonical microcircuit** (Douglas & Martin): **L4 → L2/3 → L5 → L6**, with L6 feeding back to L4 — L4 is the entry
node of this loop ([Douglas & Martin 2004](https://www.cns.nyu.edu/~tony/vns/readings/douglas-martin-2004.pdf)).

## 6. CRUCIAL — What the Thousand Brains Theory / Numenta account SMOOTHS OVER

The Numenta/Hawkins mapping treats L4 as **the single "input/feature layer"**: L4 receives driver thalamic sensory input,
binds it to a **location/orientation signal from L6a** (grid-cell-derived), and projects to **L2/3 as the object-output
layer**; grid cells make bidirectional connections with L4 so the column *predicts* its next L4 input ([Hawkins et al.,
*Front Neural Circuits* 2017, PMC5661005](https://pmc.ncbi.nlm.nih.gov/articles/PMC5661005/); [Hawkins et al. 2019
Framework, PMC6336927](https://pmc.ncbi.nlm.nih.gov/articles/PMC6336927/); [Lewis et al. 2019, "Locations in the
Neocortex"](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2019.00022/full)). The account gets
the **feedforward-thalamic-recipient, feature-at-location, L4→L2/3, L6-modulates-L4** skeleton right. What it **omits or
smooths over**:

1. **The sublaminae.** TBT's L4 is unitary. Real primate L4 is **4A/4B/4Cα/4Cβ carrying three segregated thalamic streams
   (M/P/K) with divergent output targets** (§1). The verified fetch of the 2017 paper: **"no discussion of L4 sublaminae
   differentiation, spiny stellate cells specifically, or agranular cortex"** ([Hawkins et al. 2017,
   PMC5661005](https://pmc.ncbi.nlm.nih.gov/articles/PMC5661005/)). A model treating L4 as one feature vector cannot express
   the M/P/K parallel-channel factorization that V1 anatomy demands.

2. **Spiny stellate vs star pyramid vs pyramid.** TBT doesn't specify a cell type; implicitly it assumes the granular
   spiny-stellate "feature-detector." It skips that (a) L4 excitatory cells are **local, non-projecting**, and (b) the
   morphological type is **area-variable** and, in M1, **entirely pyramidal**.

3. **Agranular cortex has no granular L4.** TBT posits **the same column with an L4 input layer everywhere**. Motor and
   prefrontal cortex are **agranular** (§4). TBT's "uniform column" is closest to *primary sensory* cortex and glosses the
   granular↔agranular gradient; the honest defense is Harris & Shepherd's **serial homology** (the *stage* is conserved
   even where the *granular layer* is not), but TBT does not engage this.

4. **The thalamic input is a small minority of L4 synapses (~6–20%), and L6 feedback is the majority (~45%).** TBT frames L4
   as "receiving sensory input" as if thalamus dominates; anatomically L4 is dominated by **L6 feedback + recurrent L4**
   with thalamus as a **thin, synchronous driver** (§3). TBT's use of **L6a as the *location* input to L4** is actually
   *consistent* with the anatomy that L6→L4 is the largest projection — a point in TBT's favor — but it recasts a
   numerically dominant, physiologically modulatory feedback pathway purely as "location."

5. **L4 is not the obligatory first/only cortical relay.** TBT's strict FF chain (thalamus → L4 → L2/3) is challenged by
   **Constantinople & Bruno (2013)**: thalamus drives **L5/6 in parallel with L4** ([Science
   2013](https://www.science.org/doi/10.1126/science.1236425)).

6. **L4 also outputs to L5/6 and (in V1) to specific sublaminae**, not only "up to L2/3" (§5).

**Bottom line for the implementation:** TBT's "L4 = feature-at-location input layer, bound to L6 location, feeding L2/3
object output" is a **reasonable computational abstraction of the primary-sensory input stage**, and its L6→L4 location
wiring even matches the dominant anatomical projection. But it **erases** (a) V1's M/P/K sublaminar parallelism, (b) the
spiny-stellate/star-pyramid/pyramid distinction and L4's local non-projecting nature, (c) the granular-vs-agranular reality
that ~half the cortex has no granular L4, and (d) the parallel (not strictly serial) thalamic activation of deep layers.

**Key uncertainty flags:** (1) Harris & Shepherd 2015 and Douglas & Martin 2004 quotes are paraphrases via secondary
summaries. (2) The 45%/28%/6% cat-V1 synapse breakdown is the classic Ahmed et al. 1994 figure; later EM (da Costa & Martin
2011) revised thalamic to "<10%". (3) Spiny-stellate share of L4 excitatory cells is ~80% in the barrel data. (4) The
Numenta "L6a → L4 location, ~45%" phrasing blends the paper with the Ahmed L6 figure — the *qualitative* claim is in the
paper; treat any exact percentage as unverified against the Numenta text.

---
---

# THREAD 3 — Neocortical Layer 5

The single most important synthesis point is in §7.

## 1. Sublaminae and cell classes of L5: L5A/L5B and the IT vs PT split

**The two-axis taxonomy.** Modern cortical taxonomy classifies L5 excitatory neurons primarily by their long-range
projection target, cross-cut by laminar sublayer. The canonical framework (Harris & Shepherd 2015) recognizes three
excitatory *projection classes* neocortex-wide — **IT (intratelencephalic), PT (pyramidal tract), and CT (corticothalamic)**
— defined by axonal target ([Harris & Shepherd 2015, *Nat Neurosci*](https://www.nature.com/articles/nn.3917)). L5 is
dominated by the IT and PT classes (CT is mostly an L6 phenomenon).

**IT vs PT — the core L5 dichotomy** (all from [Frontiers review "Neocortical layer 5 subclasses,"
2022](https://www.frontiersin.org/journals/synaptic-neuroscience/articles/10.3389/fnsyn.2022.1006773/full), corroborated by
[Harris & Shepherd 2015](https://www.nature.com/articles/nn.3917) and [Ramaswamy & Markram
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4481152/)):

| Property | **IT (intratelencephalic)** | **PT (pyramidal tract)** — a.k.a. ET (extratelencephalic) |
|---|---|---|
| Common name | slender-tufted (L5st) | **thick-tufted** (L5tt / TTL5) |
| Laminar position | small somata in **upper L5 (L5A)** | large somata in **lower L5 (L5B)** |
| Apical dendrite | thin, poorly-branched distal tuft, few obliques | **thick apical trunk, elaborate tuft arborizing in L1** |
| Intrinsic firing | **regular-spiking (RS)**; high input resistance, pronounced spike-frequency adaptation | **intrinsically bursting (IB)**; high-frequency AP bursts, little sAHP |
| Soma size | smaller | largest pyramidal cells in the cortex |

Direct quotes: *"IT neurons have relatively small cell bodies in the upper half of L5 (L5a), whereas ET neurons with large
cell bodies are found primarily in lower L5 (L5b)"*; PT/ET show a *"tendency for their APs to appear in high-frequency
bursts,"* whereas IT show *"pronounced adaptation of the AP frequency"* ([Frontiers
2022](https://www.frontiersin.org/journals/synaptic-neuroscience/articles/10.3389/fnsyn.2022.1006773/full)). The
thick-tufted cell *"terminate[s] with a crown-like thick tuft of dendrites in layer 1"* ([Ramaswamy & Markram
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4481152/)).

**Caveat / flag:** The L5A↔IT and L5B↔PT mapping is a strong tendency, **not** a strict rule. A finer molecular parcellation
(e.g. NP "near-projecting") exists beyond the IT/PT binary ([Kim et al. 2015,
*Neuron*](https://www.sciencedirect.com/science/article/pii/S0896627315009812), which resolves **three** L5 types).

## 2. Projection targets — and which class is the true "motor output"

**IT neurons → telencephalon-only, bilateral.** IT axons stay within the telencephalon: ipsi- and contralateral cortex (via
corpus callosum) and **bilateral striatum**. They form the *"backbone of communication within and between cortical areas and
hemispheres"* ([Harris & Shepherd 2015](https://www.nature.com/articles/nn.3917)). IT is the corticostriatal/corticocortical
class ([Shepherd 2013, *Nat Rev Neurosci*](https://www.nature.com/articles/nrn3469)).

**PT neurons → subcortical, the descending output.** PT/thick-tufted axons leave the telencephalon and *"project to
subcortical areas, including the ipsilateral striatum, higher-order thalamic nuclei, superior colliculus (SC), and pons"*
([Frontiers 2022](https://www.frontiersin.org/journals/synaptic-neuroscience/articles/10.3389/fnsyn.2022.1006773/full)), and
more fully to **thalamus, superior colliculus/tectum, pons, brainstem, and spinal cord** ([Frontiers "Corticothalamic
Pathways From Layer 5," 2021](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2021.730211/full)).
Critically, **a single PT axon branches to several of these targets at once** (see §6).

**Which is the "true motor output"?** **PT (thick-tufted, L5B).** PT neurons *"serv[e] as the principal output pathway
funneling information flow to subcortical structures"* ([Ramaswamy & Markram
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4481152/); [Frontiers
2021](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2021.730211/full)). Corticospinal and
corticobulbar (true motor) axons are PT-class. **IT is NOT a direct motor output** — its striatal projection influences
action selection through the basal-ganglia loop, but it never leaves the telencephalon. So when TBT says "L5 is the motor
output," the biological referent is specifically the **PT/thick-tufted subclass of L5B**, not L5 as a whole.

## 3. The intracolumnar flow: is there an L4 → L2/3 → IT → PT hierarchy?

**Yes, broadly — with an asymmetric IT→PT step at the end.**

- **Interlaminar backbone:** thalamic input enters **L4 → L2/3 → L5/L6**; L2/3 also projects forward to the next area, and
  L6 closes the loop back to L4 ([Bastos et al. 2012, canonical microcircuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/)).
- **The IT→PT asymmetry inside L5:** IT and PT interconnect **unidirectionally**. *"IT neurons connect to ET neurons, but ET
  neurons do not connect back to IT neurons, indicating unidirectional information flow between the two neuron types"*
  ([Frontiers 2022](https://www.frontiersin.org/journals/synaptic-neuroscience/articles/10.3389/fnsyn.2022.1006773/full)).
  Independently: *"IT cells form excitatory synapses onto PT cells as well as other IT cells, [while] PT cells preferentially
  connect to other PT cells"* ([Harris & Shepherd 2015](https://www.nature.com/articles/nn.3917)).

So the effective serial chain is **L4 → L2/3 → L5-IT → L5-PT → (subcortical/thalamic broadcast)**: IT integrates local
cortical computation and "gates" it into PT, which broadcasts the result. **Caveat / flag:** it is a *dominant direction of
flow*, not a clean pure feedforward pipeline (translaminar recurrence exists; both L5 types receive direct thalamic input).

## 4. Larkum's apical dendrite / BAC firing — the candidate mechanism for the "apical tiebreak" in HTM

**Two spatially-separated integration zones (a two-compartment cell).** The thick-tufted L5 pyramid has **two** action-
potential initiation sites: the axo-somatic Na⁺ site (driven by **basal/perisomatic feedforward** input) and a **distal
apical Ca²⁺ spike zone** near the top of the apical trunk, at the L1 tuft (driven by **feedback**) ([Larkum 2013, *Trends
Neurosci*](https://www.sciencedirect.com/science/article/abs/pii/S0166223612002032); original [Larkum, Zhu & Sakmann 1999,
*Nature*](https://www.researchgate.net/publication/13106497_A_new_cellular_mechanism_for_coupling_inputs_arriving_at_different_cortical_layers)).

**BAC firing = the coincidence detector.** When a somatic spike back-propagates and **coincides** with distal apical input,
it triggers a **B**ackpropagation-**A**ctivated **C**a²⁺ spike, producing a somatic **burst** ([Ramaswamy & Markram
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4481152/)). Functional reading: *"when apical and basal synaptic input
co-occur, the neuron responds with rapid bursting activity that exceeds firing rates achievable under basal input alone"* — a
within-cell **AND** of feedforward × feedback ([PLoS Comput Biol
2015](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1004090)).

**Timescale.** The coincidence window is ~**5–30 ms**, set largely by NMDA-receptor kinetics. NMDA spikes in the fine tuft
branches are the dominant mechanism translating distal input into somatic firing ([Larkum et al. 2009,
*Science*](https://www.science.org/doi/10.1126/science.1171958)).

**What each compartment reads.** **Basal/perisomatic = feedforward** ("what is here now"): thalamic + L4/L2-3 drive.
**Apical tuft in L1 = feedback/context**: L1 horizontal fibers carry long-range corticocortical + thalamocortical
associational input; the apical Ca²⁺ spike *"control[s] the synaptic efficacy of cortico-cortical inputs"* ([Ramaswamy &
Markram 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4481152/)). Larkum frames this as the cellular substrate for
**context-sensitive selective amplification** ([Larkum
2013](https://www.sciencedirect.com/science/article/abs/pii/S0166223612002032)).

**Mapping to HTM's "apical tiebreak" — and where it *diverges*.** HTM's neuron uses this two-zone idea but assigns roles
slightly differently: in the columns paper, the **apical dendrite carries top-down/location feedback** that biases
recognition — *"Depolarized cells fire sooner than, and thereby inhibit, non-depolarized cells that recognize the same
feedforward patterns"* ([Hawkins, Ahmad & Cui
2017](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2017.00081/full)). **Flag the mismatch:**
HTM implements the tiebreak as *subthreshold depolarization + earlier firing + WTA* (a timing/rank operation), whereas
Larkum's mechanism is a genuine *nonlinear Ca²⁺ burst* (a rate/gain operation). Same computation (context-gated coincidence
detection), different currency. If you want biological fidelity, the apical segment should *multiplicatively amplify/gate*
the feedforward response, not merely add a depolarization offset.

## 5. Displacement cells (TBT): location and function — the uncertain one

**What they compute.** *"Grid cells determine a new location based on a current location and a displacement vector (i.e.,
movement). Displacement cells determine what displacement is required to reach a new location from a current location"* —
i.e., the **difference between two sets of active grid cells** ([Hawkins et al. 2019, "A Framework for Intelligence and
Cortical Function Based on Grid Cells"](https://www.frontiersin.org/articles/10.3389/fncir.2018.00121)). Functionally they
enable **object compositionality** (a displacement vector uniquely encodes "logo at a relative position on the cup") and
**behavior as displacement sequences**.

**Where are they hypothesized to be? — L5 thick-tufted, but this is an explicit hypothesis.** The frameworks paper proposes:
*"displacement cells are located in L5 (specifically L5 thick-tufted neurons)."* The rationale is the PT dual-projection
biology from §2/§6: thick-tufted L5 cells send branches **both** subcortically **and** to *"thalamic relay nuclei, which
then project to hierarchically higher cortical regions"*; the authors suggest L5 displacement cells *"alternately represent
movements (sent sub-cortically) and then represent compositional objects (sent to higher regions via thalamic relay
cells)"* ([Hawkins et al. 2019](https://www.frontiersin.org/articles/10.3389/fncir.2018.00121)).

**Flag (this is the load-bearing uncertainty):** The L5-thick-tufted placement is a **theoretical proposal, not an empirical
finding** — displacement cells are a *predicted* cell type, not yet observed. Note also the **tension inside TBT's own
corpus**: the *location/grid* apparatus is placed in **L6a**, the *sensory* layer in **L4**, the *object-output* layer in
**L2/3** ([Lewis et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6491744/)) — while *displacement* cells are pushed
down to **L5**. So the two mathematically-complementary computations (grid vs displacement) are hypothesized to live in
different layers, and the wiring between them is conjectural.

## 6. L5 as the thalamus DRIVER (the transthalamic pathway)

**Two functionally opposite corticothalamic pathways.** **L6 CT → first-order (and higher-order) thalamus = MODULATOR**;
**L5 PT → higher-order thalamus = DRIVER** (the *content*-carrying pathway). *"L5 inputs to HO nuclei are sparse and display
characteristic 'driver' properties, in contrast to L6 projections, which… provide modulatory input"* ([Frontiers
2021](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2021.730211/full)).

**The transthalamic (cortico-thalamo-cortical) loop.** Higher-order nuclei (pulvinar, POm) relay **L5-driver output of one
cortical area to a higher cortical area**: *"Cortical L5tt neurons in one source region send information to a secondary
cortical recipient region via HO thalamus"* — a **parallel route to direct corticocortical connections** ([Frontiers
2021](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2021.730211/full)).

**The efference-copy geometry — same axon, two destinations.** *"L5tt neurons send branching collaterals to subcortical
targets including the HO thalamus, brainstem, and spinal cord — even at the level of single L5 neurons innervating more than
one subcortical region"* ([Frontiers
2021](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2021.730211/full)). So one PT axon
simultaneously (a) commands a motor/subcortical target and (b) sends a **copy of that command** up to the next cortical
level via HO thalamus — a built-in **efference copy**. The synapses onto HO thalamus are **giant driver boutons**.

**Implementation relevance:** exactly the biology a TBT column needs if L5 both emits a motor command *and* forwards a
location/displacement update up the hierarchy — the "efference copy → predicted next sensory location" loop.

## 7. CRUCIAL — what the typical TBT/Numenta "L5 = motor output layer" account smooths over

1. **The IT vs PT split is collapsed into one "L5 motor layer."** Biologically, "L5" is *two* classes doing *different*
   jobs: **IT** (telencephalic — corticocortical + bilateral striatum, an *action-selection/association* role, **not** a
   direct motor output) and **PT** (the actual subcortical/spinal descending output). Calling all of L5 "the motor output"
   conflates the associative IT class with the true PT effector, and hides the **IT→PT** internal gate. For a faithful
   column you likely want *two* L5 populations.

2. **L5 as the higher-order-thalamus DRIVER / the transthalamic hierarchy is dropped.** The three-layer story treats
   hierarchy as direct L2/3→L4 wiring and L5 as a pure *exit*. It omits that PT/L5 is *also the driver that builds the next
   level of the cortical hierarchy through the thalamus*, and that a single L5 axon couples motor command + ascending copy.
   (The *full* frameworks paper does invoke this to justify putting displacement cells in L5 — so the omission is in the
   accessible/HTM-implementation account, not in Hawkins et al.'s deepest paper.)

3. **Apical dendritic computation (Larkum) is reduced to a scalar "predictive depolarization."** HTM keeps the *idea* of a
   second modulatory dendritic zone but implements it as subthreshold depolarization + earlier spiking + WTA. It smooths
   over the **nonlinear Ca²⁺-plateau/BAC burst** — the multiplicative gain and 5–30 ms NMDA coincidence window (§4).

4. **The location of displacement cells is presented as settled when it is a bare hypothesis** — and it sits awkwardly
   against TBT's own placement of grid cells in L6a.

5. **L5A/L5B sublamination and its input asymmetry vanish.** Upper-L5 (IT) and lower-L5 (PT) differ in intrinsic dynamics
   (RS vs IB), thalamic inputs, and inhibitory targeting (PV→PT, VIP→IT). The **intrinsic bursting** of PT cells is the
   output code of the BAC mechanism — flattening L5 to a rate unit throws away the "feedforward AND feedback agreed" signal.

**One-line takeaway:** "L5 = motor output" is true only of the **PT/thick-tufted/L5B** subclass, and even there it
undersells L5, because the *same* PT cell is simultaneously (i) the subcortical motor effector, (ii) the **driver** that
constructs the next cortical level via higher-order thalamus (efference copy), and (iii) — per TBT's own hypothesis — the
seat of **displacement/compositional** computation, all gated by a **Larkum apical×basal coincidence detector** that a
scalar "predictive depolarization" only approximates. A biologically honest TBT column should split L5 into IT (associative
integrator, reads L2/3) and PT (bursting driver/effector with a branching axon to both subcortex and HO-thalamus), and give
the PT cell a multiplicative apical gate.

**Uncertainty flags:** (a) L5A↔IT / L5B↔PT is a strong tendency, not a strict rule; (b) IT→PT is a dominant asymmetry, not
an exclusive wire; (c) displacement-cell placement in L5-thick-tufted is a TBT *hypothesis*, unconfirmed; (d) HTM's "apical
tiebreak" is computationally analogous to Larkum's BAC but implemented as timing+inhibition rather than a Ca²⁺ burst.

---
---

# THREAD 4 — Neocortical Layer 6

**Scope note / confidence:** Anatomy below is strongest for cat and rodent primary sensory cortex and weaker/species-
variable for primate.

## 0. One-paragraph orientation

Layer 6 is the deepest, oldest, and most heterogeneous cortical layer. It contains at least two large, non-overlapping
excitatory populations — **corticothalamic (CT)** pyramids that project *down and out* to the thalamus, and **corticocortical
(CC)** pyramids that project *sideways* within cortex — plus a distinct deep sublamina, **L6b**, a surviving remnant of the
embryonic **subplate**. The layer's most-cited computational role is **gain control**: L6 CT cells send a numerically large
but individually weak, facilitating, metabotropic feedback to both the thalamus and to L4. TBT reuse the *anatomy* of the
L6↔L4 loop but re-interpret its *function* as a grid-cell location signal.

## 1. Sublaminae: L6A vs L6B

**L6A** is the main body of layer 6 and contains the "classic" L6 pyramidal populations (CT and CC, §2). **L6B** is a thin
band at the very bottom of cortex, abutting the white matter, developmentally and molecularly distinct.

- **L6b is a persistent remnant of the subplate.** Marx & Feldmeyer found the subplate and L6b consist of "heterogeneous but
  comparable neuronal cell populations," each with ~5 distinct spine-bearing cell types ([Marx & Feldmeyer 2017, *Cerebral
  Cortex*](https://academic.oup.com/cercor/article/27/2/1011/3056173)). Recent lineage work concludes that **most early-born
  subplate neurons persist as adult L6b neurons** ([bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.11.20.689634v2.full)).
- **Distinct molecular markers separate L6b from L6a.** L6b is marked by *Ctgf*, *Cplx3*, *Nurr1/Nr4a2*, *Nxph3*; L6a CT
  neurons by *Ntsr1* ([Frontiers 2023, "Structure and function of neocortical layer
  6b"](https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2023.1257803/full)). L6b morphology
  is unusually diverse — inverted, horizontal, tangential, multipolar, upright.
- **L6b has its own projection logic and neuromodulation.** A *Drd1*-expressing subset of L6b CT neurons projects
  **exclusively to higher-order thalamus (POm)** ([Hoerder-Suabedissen et al. 2018, *Cereb.
  Cortex*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6018949/)). L6b is also **uniquely orexin-sensitive** (OX2 receptors),
  tying it to arousal/wakefulness ([Frontiers 2023](https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2023.1257803/full)).

**Uncertainty flag:** Thomson's and Briggs's classic reviews largely treat "layer 6" as a whole and do **not** foreground
L6b ([Thomson 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2885865/); [Briggs
2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2826182/)). The rich L6b-as-subplate-remnant picture is largely post-2015.

## 2. The two main projection classes: CT vs CC

| Feature | **CT (corticothalamic)** | **CC (corticocortical)** |
|---|---|---|
| Morphology | "Fairly short, **upright** pyramids" | "Atypical" — **inverted** pyramids, **bipolar**, horizontal cells |
| Local axon | Narrow arbour projecting **up toward L4/superficial** | **Long, horizontally oriented** axons in deep layers |
| Main target | **Thalamus** (+ nRT) and L4 | Other cortical areas, claustrum-like horizontal spread |
| Postsynaptic selectivity | CT cells "rarely innervated other pyramidal cells" | CC cells "rarely innervated interneurones" |

([Thomson 2010, *Front. Neuroanat.*](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2010.00013/full))

- **Proportions:** CT cells are "only some **30–50%** of the pyramidal cells in layer 6." **Species variation:** CT neurons
  are a *smaller* fraction of L6 in **primate** sensory cortex than in cat/rodent ([Briggs
  2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2826182/)).
- **Harris & Shepherd 2015** confirm CT cells are essentially **restricted to L6** ([Harris & Shepherd
  2015](https://www.nature.com/articles/nn.3917)).

**Key point for the model:** "L6 output" is not a single vector — one population talks to thalamus, the other laterally to
cortex, with opposite interneuron-vs-pyramid targeting.

## 3. CORE vs MATRIX thalamus, and DRIVER vs MODULATOR

Two separate dichotomies. Do not conflate them.

**(a) Core vs Matrix (Jones).** ([Jones 2001, *Trends Neurosci.*](https://pubmed.ncbi.nlm.nih.gov/11576674/))
- **Core** = *parvalbumin*-rich, dominant in primary relays; project **topographically to middle layers (L4, deep L3)**;
  carry **specific** content.
- **Matrix** = *calbindin*-rich, spread diffusely; project **broadly to superficial layers, especially L1**; **global/
  modulatory** (arousal, synchrony).
- A single relay neuron can have branched axons to both, so core/matrix may be ends of a continuum ([2024
  reconsideration](https://pmc.ncbi.nlm.nih.gov/articles/PMC11170670/)).

**(b) Driver vs Modulator (Sherman & Guillery; "Class 1" vs "Class 2").**
- **Drivers:** large boutons on **proximal** dendrites, **ionotropic**, **depressing**, all-or-none.
- **Modulators:** small boutons on **distal** dendrites, **metabotropic** (mGluR), **facilitating**, graded ([Sherman &
  Guillery 2024, *J. Neurosci.*](https://www.jneurosci.org/content/44/35/e0909242024)).

**(c) How L6 CT feedback fits:**
- **L6 CT → first-order ("core") thalamus is a MODULATOR** (small, slow, facilitating, mGluR, distal dendrites) ([Thomson
  2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2885865/)).
- **L5 PT → higher-order thalamus is a DRIVER** (large boutons, proximal) ([Sherman & Guillery
  2024](https://www.jneurosci.org/content/44/35/e0909242024)).
- So two descending cortical→thalamic systems: L6 = numerous, feedback, **modulatory**; L5 = sparse, feedforward,
  **driving**. L6 CT also uniquely innervates the **reticular nucleus (nRT)** — the substrate for attentional gating.

**Within L6 CT there are two subclasses:** **Upper L6 CT** → first-order thalamus + nRT; **Deep L6 CT** → higher-order
thalamus, not nRT; plus the L6b *Drd1* subset → POm only.

## 4. The L6 → L4 intracolumnar feedback projection

**(a) L6→L4 is numerically LARGE.** In cat V1, feedforward **thalamic (LGN) input is only ~5–10%** of L4's excitatory
synapses, while the single largest source is **layer 6 pyramidal cells** (~**45%**) ([Binzegger, Douglas & Martin 2004, *J.
Neurosci.* 24:8441](https://pubmed.ncbi.nlm.nih.gov/15456817/)).

**(b) But each L6→L4 synapse is WEAK and MODULATORY.** L6→L4 EPSPs ~**0.2 mV** (vs ~0.9 mV for L4→L4); L6→L4 synapses
**facilitate** and engage **group-I mGluR** — the modulator signature ([Thomson
2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2885865/); [Briggs 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2826182/)).
Briggs concludes these properties are "consistent with a **modulatory role**" — **gain control** and length-tuning.

**Reconciliation (the important nugget):** "many synapses" and "weak/modulatory" are **both true** — L6 blankets L4 with a
large number of individually feeble, facilitating, metabotropic contacts, ideal for setting **gain** rather than driving
spikes. Optogenetic work (Olsen/Bortone/Scanziani 2012) showed L6 CT activation predominantly **suppresses** the column —
gain-control/divisive-normalization. **Flag:** treat the exact **~45%** as the Binzegger-derived figure for **cat V1**.

**(c) The loop is reciprocal and topographically narrow.** L4 cells make large numbers of synapses back onto the *same* L6
cells — a tight bidirectional column-internal loop.

## 5. L6 as the "location / grid-cell" layer in TBT — does neuroscience support it?

**What TBT claims.** Numenta place **cortical grid cells (the location signal) in L6** and **displacement cells in L5**,
with L4 as sensory input; the L6→L4 projection lets the current *location* predict the *next feature* ([Hawkins et al. 2019,
"Framework"](https://pmc.ncbi.nlm.nih.gov/articles/PMC6336927/) — "We propose cortical grid cells are located in L6 and
displacement cells are in layer 5"; [Lewis et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6491744/) — location layer
= **L6a**).

**What the neuroscience actually supports:**
- **Grid cells are firmly established in the entorhinal cortex (EC)**, not the sensory neocortex. There is, to date, **no
  direct recording of hexagonal grid-cell firing in L6 of primary sensory cortex** — the biggest open empirical gap.
  **(Flag: strong extrapolation.)**
- **Anatomy is compatible, not confirmatory.** The narrow reciprocal L6a↔L4 loop, motor/L5→L6 input, path-integration-
  friendly recurrence do exist ([Lewis et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6491744/)). TBT is honest the
  L6a→L4 connection is weak ("they connect to distal dendritic segments of L4 cells") — matching the modulator physiology.
- **Path integration:** demonstrated in EC grid cells; in neocortex it is a TBT *proposal*.

**Bottom line:** L6-as-location-layer is a *plausible, anatomically-consistent hypothesis*, not an established fact.

## 6. Inputs and Outputs of L6 (consolidated)

**Inputs TO L6:** thalamus (direct topographic); the most prominent local drive from **L5**; strong **L4** (reciprocal);
within-L6 recurrence; top-down feedback + long-range CC; a "relatively strong" descending inhibitory input from **L4 basket
cells**; neuromodulation (L6b via orexin + nicotinic ACh) ([Thomson 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2885865/);
[Briggs 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2826182/)).

**Outputs FROM L6:** → **Thalamus** (CT: first-order + nRT for upper L6, higher-order for deep L6/L6b — modulatory); → **L4**
(numerous, weak, facilitating — gain control); → **L5** (a couple of documented *powerful* L6→L5 connections, >1 mV,
depressing — an exception to L6's weak local output); → **other cortex / claustrum** (CC); → **hippocampus** (entorhinal L6b).
Target selectivity: CT avoid other pyramids; CC avoid interneurons.

## 7. CRUCIAL — what the typical TBT/Numenta account SMOOTHS OVER

1. **CT vs CC is flattened.** L6 is *two* output systems with opposite jobs. Numenta pick a side: they attribute the
   ~45%-of-L4 figure to **"L6a corticocortical neurons,"** and argue **L6a is "the only known set of cells that meet [the]
   requirement"** for the reciprocal L4 loop ([Hawkins et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6336927/)) —
   leaning on **CC/L6a** while largely setting aside **CT→thalamus**. A faithful column should represent **both**.

2. **The thalamus is largely dropped.** L6's single most-cited biological role — **modulatory corticothalamic feedback +
   gain control + nRT attentional gating** — is not part of the grid-cell story. The **thalamic gain-control loop is the
   missing half of L6**.

3. **L6a vs L6b is collapsed.** TBT map location cells to **L6a** and ignore **L6b** — a genetically/developmentally distinct
   subplate remnant with its own higher-order-thalamus loop and arousal (orexin) control.

4. **"Modulatory / gain control" is re-labeled as "location→feature prediction."** The same weak, facilitating, mGluR L6→L4
   synapses that physiology calls **gain modulation** are re-interpreted as a **location signal**. Not contradictory, but the
   *content* (a path-integrated grid code) is an added hypothesis.

5. **Grid cells in L6 are assumed, not observed.**

6. **Species and area variation is smoothed** (CT fraction, L6 size, connection strengths differ across cat/rodent/primate).

**Net:** TBT's L6 is a *functional slice* — the L6a↔L4 reciprocal, path-integratable loop — extracted from a layer that
biologically is (i) a two-population cortical output hub (CT + CC), (ii) the cortex's modulatory controller of the thalamus
(core relay + nRT), and (iii) home to a distinct subplate-derived L6b. Adopting "L6 = location/grid" is defensible as a
model choice, but the attention/gain-control/thalamic-routing machinery lives in exactly the L6 sub-systems the standard TBT
account omits.

**Uncertainty flags:** (a) exact **~45%** L6→L4 is Binzegger-derived for **cat V1**. (b) CT-cell fraction (30–50%) and its
**primate** reduction are review-level, species-variable. (c) **Grid cells in L6** of sensory cortex are a TBT proposal, not
an observation. (d) L6 CT optogenetic gain-suppression is stated from background knowledge — verify primaries before citing.

---
---

# THREAD 5 — Layer 1, Interneurons, the Real Inter-Laminar Wiring, and Thalamic Core/Matrix

**Scope note.** The "canonical microcircuit" is a *useful abstraction*, not a settled fact. Its quantitative form is best-
characterized in cat/rodent primary sensory cortex and macaque visual cortex; connection strengths vary by area, species,
sub-lamina, and cell type.

## 1. LAYER 1 — contents and computational role

### What L1 physically contains

L1 (the molecular layer) is **almost devoid of excitatory somata** but is *not* empty — it is the cortex's top-down
integration zone and has "the highest density of excitatory synapses" of any layer ([Schuman et al. 2021, *Neocortical Layer
1*, PMC9012327](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012327/)). Its constituents:

- **Apical dendritic tufts of L2/3 and L5 pyramidal cells.** The defining feature; a tuft in L1 is "electrotonically
  segregated from the basal dendrites and the somato-axonal area," letting top-down input act on a separate compartment.
- **Long-range cortico-cortical FEEDBACK axons**, with **sublaminar organization** (upper L1a vs lower L1b, area-dependent).
- **Thalamic MATRIX / higher-order axons** (POm, pulvinar/LP, VM). ~3× more effective on L1 interneurons than on L2/3
  pyramids. Pulvinar→L1 in V1 carries "error/agreement" (visual-flow vs running-speed mismatch) signals — predictive-coding-
  relevant.
- **Resident GABAergic interneurons** at a density similar to other layers; four subtypes ([Schuman et al. 2019, *J
  Neurosci* 39:125](https://www.jneurosci.org/content/39/1/125)): **Neurogliaform (NGFC)** (NDNF⁺/NPY⁺, ~30% of L1 INs,
  **volume-transmit GABA → slow GABA_B**), **Canopy cells**, **α7 (nicotinic) cells** (→ L5a), **L1-VIP cells** (→ L5a/L6,
  disinhibitory). (Human L1 additionally has **"rosehip" cells** that inhibit L3 pyramidal tufts.)
- **Cajal-Retzius cells** — developmentally critical (reelin) but **transient**; essentially absent from adult L1
  computation.
- **Neuromodulatory afferents** — L1 is one of the most densely **cholinergic** layers — a **brain-state control** hub.

### L1's computational role — it is load-bearing, not "irrelevant"

L1 is the physical substrate for **coincidence detection between top-down and bottom-up signals**, which [Larkum
2013](https://www.sciencedirect.com/science/article/abs/pii/S0166223612002032) elevates to an **organizing principle of the
whole cortex**: the pyramidal neuron couples feedforward (basal/perisomatic) and feedback (L1 tuft) via **BAC firing**. Weak
tuft input alone doesn't fire the cell; coinciding (~5 ms) with a back-propagating spike, it triggers a Ca²⁺ plateau →
**high-frequency burst**. This is **apical amplification**: top-down context multiplicatively gates whether a bottom-up-
driven cell bursts. L1 also implements **gain/plasticity control by disinhibition** (e.g., fear learning: L1 INs → inhibit
L2/3 PV → disinhibit L2/3 pyramids).

**Verdict for an HTM/TBT column:** the HTM neuron model *already implicitly honors L1* — its "distal/apical dendrite =
prediction that depolarizes without firing" is precisely Larkum's tuft-coincidence idea ([Hawkins & Ahmad 2016, *Why Neurons
Have Thousands of Synapses*](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2016.00023/full)).
Calling L1 "irrelevant" is wrong: L1 is *where the feedback/prediction stream lands*. What HTM omits is the **L1 interneuron
machinery** (NGF GABA_B, disinhibition) that *controls* when apical input is allowed to amplify.

## 2. INHIBITORY INTERNEURONS — classes, layers, roles

GABAergic interneurons are ~20% of cortical neurons and fall into **three non-overlapping molecular groups covering ~100%**
([Rudy et al. 2011, PMC3556905](https://pmc.ncbi.nlm.nih.gov/articles/PMC3556905/); [Tremblay, Lee & Rudy 2016, *Neuron*,
PMC4980915](https://pmc.ncbi.nlm.nih.gov/articles/PMC4980915/)):

| Group | ~% of INs | Subtypes | Target on pyramidal cell | Role | Layers |
|---|---|---|---|---|---|
| **PV** | ~40% | Fast-spiking **basket** + **chandelier** | Soma/proximal (basket); **AIS** (chandelier) | **Perisomatic gain & spike-timing**; feedforward inhibition, gamma; chandelier vetoes output | Densest L4 & L5 |
| **SST** | ~30% | **Martinotti** (→ L1) + non-Martinotti | **Distal/apical dendrites** | **Dendritic inhibition** — controls apical Ca²⁺ spikes / top-down → gates *context/prediction* | Somata L5/L6; axons to L1 |
| **5HT3aR** (VIP + NGF) | ~30% | **VIP** (disinhibitory) + neurogliaform/L1 | VIP → *other interneurons*; NGF → tufts (GABA_B) | **Disinhibition** (VIP→SST); slow GABA_B | VIP enriched L2/3; NGF in L1 |

Key roles: **PV basket → perisomatic** = divisive/gain control + gamma timing; **PV chandelier → AIS** = output veto. **SST
Martinotti → apical tuft** = the biological knob on the **top-down/prediction channel**. **VIP → SST → pyramidal = the
canonical DISINHIBITORY motif** ([Pfeffer et al. 2013](https://pubmed.ncbi.nlm.nih.gov/23974708/); [Pi et al.
2013](https://pubmed.ncbi.nlm.nih.gov/24097352/)) — VIP (driven by long-range/neuromodulatory input) inhibits SST,
*releasing* pyramidal apical dendrites — a context/attention/reward-gated amplification switch.

**Why excitatory-only HTM omits them, and the cost.** HTM replaces all of this with **k-winners-take-all (kWTA)**. kWTA ≈
**PV-basket perisomatic inhibition only**. Lost: **SST dendritic inhibition** (no biological gate on the apical/prediction
compartment); **VIP disinhibition** (no mechanism for attention/reward to *transiently unlock* apical amplification); and the
distinct roles (perisomatic gain/timing vs dendritic context vs axo-axonic veto) collapse into one scalar competition. A
pure-excitatory + kWTA model is a defensible first cut, but it bakes in as a constant what biology makes an adaptively
controlled variable.

## 3. THE REAL INTER-LAMINAR WIRING — recurrent, dual-counterstream, not a stack

### Quantitative reality: cortex is dominated by RECURRENT/local excitation

**The thalamus supplies only ~5% of the excitatory synapses onto L4 spiny stellate cells**; the rest are cortical — L6
pyramids ~45%, other L4 stellates ~28% ([Binzegger, Douglas & Martin 2004, *J Neurosci* 24:8441,
PMC6729898](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729898/)). Across the whole circuit, **intralaminar self-innervation
(~34%) exceeds the classic feedforward interlaminar loop (~21%)**. Cortex **recurrently amplifies** a weak thalamic drive, it
does not simply relay it ([Douglas & Martin 2004](https://pubmed.ncbi.nlm.nih.gov/15217339/)).

### The documented directed excitatory connections

([Thomson & Lamy 2007, PMC2518047](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518047/); [Harris & Shepherd
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/); [Binzegger et al. 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729898/))
- **Thalamus (core) → L4** (and directly to L5B/L6): strong *driver* but numerically small share of L4 synapses.
- **L4 → L2/3**: strong, topographically precise feedforward. **L4 → L5A/B** also present. L4 "receives little excitatory
  input in return."
- **L2/3 → L5**: **strong and a defining, conserved feature** — in motor cortex the single strongest interlaminar connection.
- **L5 → L2/3**: **weak** — the flow is "almost unidirectional" L4→L3→L5.
- **L6 → L4 (feedback)**: L6 CT pyramids project up with small, **facilitating** EPSPs — modulatory gain control; L6 is the
  *dominant* excitatory source onto L4 by synapse count.
- **L5 → L6, L6 → L5**: present. **Intralaminar recurrence** everywhere (~34%).

### Output streams (the column's three "mouths")
- **L2/3 → cortico-cortical** (feedforward to next area's L4; callosal): **IT** neurons, ~80% of cortex.
- **L5B → subcortical**: **PT** neurons → brainstem, spinal cord, tectum, pons — the *motor/action output*; highest in-vivo
  firing rates.
- **L6 → thalamus**: **CT** neurons → modulatory feedback / gain control; mostly silent in vivo.

### THE TWO COUNTERSTREAMS — the crux

([Markov et al. 2014, PMC4255240](https://pmc.ncbi.nlm.nih.gov/articles/PMC4255240/); [Shipp
2007](https://pubmed.ncbi.nlm.nih.gov/17580069/); [Bastos et al. 2012, PMC3777738](https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/))
- **FEEDFORWARD stream**: originates in **supragranular** pyramids (L2/3), terminates in **L4** of the higher area;
  *driving*, higher weights. In predictive-coding terms it carries **prediction errors** upward.
- **FEEDBACK stream**: originates in **infragranular** pyramids (L5/L6, plus deep L2/3), **avoids L4**, terminates in **L1**
  (and L6) of the lower area; more *modulatory*, more numerous, longer-range. It carries **predictions** downward.
  Supragranular streams are point-to-point/topographic; infragranular streams are diffuse.

In [Bastos et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/)'s mapping onto predictive coding: **superficial
(L2/3) pyramids encode & broadcast prediction errors** upward; **deep (L5/6) pyramids encode conditional expectations
(predictions)** and send feedback to suppress errors below. Spectral signature: **superficial = more gamma (feedforward),
deep = more alpha/beta (feedback)**.

## 4. THALAMIC CORE vs MATRIX

([Jones 2001, *Trends Neurosci* 24:595](https://pubmed.ncbi.nlm.nih.gov/11576674/); [Harris & Shepherd
2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/); [Schuman et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012327/))

| | **CORE** | **MATRIX** |
|---|---|---|
| Marker | Parvalbumin⁺ | Calbindin⁺ |
| Target layer | **L4** (and L5B/L6), middle layers | **L1** and L5a; *skips L4* |
| Topography | Focal, topographic, area-specific | Diffuse, spreads across areal borders |
| Dendritic compartment | Basal / perisomatic (driving) | Apical tuft (modulatory/contextual) |
| Nuclei | First-order relays (LGN, VPM) | Higher-order / intralaminar (pulvinar, POm, VM) |
| Function | Specific sensory **drive**; feedforward | Global **synchronization**, arousal, context, top-down |

**Mapping to feedforward/feedback:** **core ≈ feedforward** (driver, → L4, focal); **matrix ≈ feedback/contextual**
(modulatory, → L1 tufts + L5a, diffuse). Matrix input to L1 lands on the *same tufts* that receive cortico-cortical
feedback — so thalamic matrix and cortical feedback are **convergent top-down channels** onto the apical compartment.

**Uncertainty flag (important):** the strict core/matrix dichotomy is contested. A 2024 reappraisal argues there are **no
clean criteria** separating matrix from core, and recommends **abandoning core/matrix for a graded classification** ([J
Neurosci 2024, PMC11170670](https://pmc.ncbi.nlm.nih.gov/articles/PMC11170670/)). Sherman & Guillery's orthogonal **driver vs
modulator** framework partially supersedes it. Treat core/matrix as a useful *first approximation* to feedforward-driving vs
feedback-modulatory, not a hard partition.

## 5. What the simplified "L4→L2/3→L5→L6 feedforward stack" SMOOTHS OVER

1. **The circuit is recurrent, not a relay.** Thalamus ~5% of L4's synapses; intralaminar self-connection (~34%) exceeds the
   feedforward interlaminar loop (~21%) ([Binzegger et al. 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729898/)).
2. **Deep layers are driven directly by thalamus, in parallel with L4 — the serial order is partly false.** Sensory
   information can **bypass upper layers entirely** ([Constantinople & Bruno 2013,
   Science](https://pubmed.ncbi.nlm.nih.gov/23812718/)).
3. **It ignores the descending/feedback counterstream** — predictions descending vs errors ascending, with distinct laminar
   origins, weights, topographies, and oscillatory bands (§3).
4. **It erases L1 and the apical/basal dichotomy** — the compartmentalization that lets a pyramidal cell *associate* feedback
   (L1 tuft) with feedforward (basal) via BAC firing ([Larkum
   2013](https://www.sciencedirect.com/science/article/abs/pii/S0166223612002032)). Ignoring L1 is self-inconsistent for HTM.
5. **It omits inhibition's structured control** — perisomatic (PV), axo-axonic (chandelier), dendritic (SST), disinhibition
   (VIP→SST). kWTA captures only the perisomatic competition.
6. **It collapses the three output streams into "L5→next layer"** — real L2/3-IT, L5B-PT, L6-CT are distinct classes with
   distinct targets; one (L6→thalamus) closes a **cortico-thalamo-cortical loop** the flat stack cannot represent.
7. **It flattens thalamus into "just the L4 input"** — omitting **core → L4 (driver)** vs **matrix → L1/L5a (modulator)** and
   transthalamic higher-order loops.

## Design implications (this thread's synthesis)

- **Keep the two counterstreams explicit.** Superficial output = *prediction error / feedforward to the next region's L4*;
  deep output = *prediction / feedback to lower-region L1*. Aligns with predictive coding ([Bastos et al.
  2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/)).
- **Treat L1 as the feedback landing pad + apical gate**, not a discardable layer.
- **If you add any inhibition beyond kWTA, add SST (dendritic) + VIP (disinhibition) first** — they gate the top-down/
  prediction channel and learning.
- **Model thalamus as two channels** (core-driver→L4, matrix-modulator→L1) and consider a cortico-thalamo-cortical loop, but
  hold the **core/matrix labels loosely**.
- **Don't assume strict serial L4→L2/3→L5→L6**: deep layers can be driven in parallel; recurrent local excitation dominates
  the synapse budget.

**Uncertainty flags:** two fetches were blocked (Shipp 2007 full text, one core/matrix HTML); those claims are corroborated
by cross-citing reviews rather than directly quoted.

---
---

# THREAD 6 — TBT's Own Layer Mapping, and What It Explicitly Simplifies

## Primary sources
- **HAC17** — Hawkins, Ahmad & Cui (2017), "A Theory of How Columns in the Neocortex Enable Learning the Structure of the
  World," *Front. Neural Circuits* 11:81 — [PMC5661005](https://pmc.ncbi.nlm.nih.gov/articles/PMC5661005/).
- **Framework19** — Hawkins, Lewis, Klukas, Purdy & Ahmad (2019), "A Framework for Intelligence and Cortical Function Based
  on Grid Cells," *Front. Neural Circuits* 12:121 — [PMC6336927](https://pmc.ncbi.nlm.nih.gov/articles/PMC6336927/).
- **Lewis19** — Lewis, Purdy, Ahmad & Hawkins (2019), "Locations in the Neocortex," *Front. Neural Circuits* 13:22 —
  [PMC6491744](https://pmc.ncbi.nlm.nih.gov/articles/PMC6491744/).
- **TBP24** — Thousand Brains Project (2024), arXiv:2412.18354 — [html](https://arxiv.org/html/2412.18354v1).
- **TBS25** — Clay, Leadholm et al. (2025), arXiv:2507.04494.
- **Mainstream comparison** — Harris & Shepherd (2015), *Nat. Neurosci.* 18:170 —
  [PMC4889215](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/); Douglas & Martin (2004).

## 1. Layer assignments in the sensorimotor-inference column

| Functional role | Numenta's layer label | Evidence they cite |
|---|---|---|
| **INPUT / feature (sensory) layer** | **Layer 4 (L4)** | "L4 is well understood to be the primary target of thalamocortical sensory inputs" (Lewis19) |
| **OUTPUT / object layer** | **Layer 2/3 (L2/3)** | "L2/3 cells project long distances within their layer and are also a major output of cortical columns" (HAC17) |
| **LOCATION layer** | **Layer 6a (L6a)** | ~45% of L4 synapses come from L6a; L6a→L4 contacts are weak and land on **distal/basal** dendrites (HAC17; Lewis19) |

The circuit logic: L4 cells receive the feedforward feature on **proximal** dendrites (driving) and the **location signal
from L6a on distal/basal** dendrites (modulatory) → the specific active cells in L4 encode *feature-at-a-location*; L2/3
pools these over movement into a **stable object representation**. L2/3 feeds back to L4 and to adjacent columns.

**Grid cells** are placed in **L6** specifically. Framework19 is explicit and hedged: *"Our prediction is they will be in
L6"* and notes the experimental evidence "is unfortunately mute on what cortical layers contain grid cells" — **a testable
prediction, not established fact**.

**The "possibly twice" claim.** HAC17 (verbatim): *"Anatomical evidence suggests that the sensorimotor inference model
described above exists at least once in each column (layers 4 and 2/3) and perhaps twice (layers 6a and 5)."* So the
**second instance** uses **L6a as input** and **L5 as output** — hedged: "Whether L6a and L5 can be interpreted as an
instance of the model is unclear."

## 2. L5, motor output, and the thalamus

**In the 2017/2019 theory papers these are modeled** — one of TBT's boldest anatomical claims.

- **L5 as the second output layer AND the motor/composition layer.** HAC17 identifies L5 **thick-tufted / pyramidal-tract**
  cells. Framework19 makes L5 the home of **displacement cells** and proposes they **time-multiplex two representations**:
  *"The L5 cells in question are displacement cells and they alternately represent movements (sent sub-cortically) and then
  represent compositional objects (sent to higher regions via thalamic relay cells)."* Framework19 frames this as resolving
  a puzzle ("It is difficult to understand how the same L5 cells can be both the motor output and the feedforward input to
  other regions") — **flagged by the authors as speculative**.

- **The thalamus** is (a) the **feedforward driver into L4/L6a**, and (b) crucially the **relay by which L5's output reaches
  hierarchically higher cortex** ("the same L5 cells send a branch of their axon to thalamic relay nuclei, which then project
  to hierarchically higher cortical regions," Framework19). TBT routes **cortico-cortical hierarchy through the higher-order
  thalamus (transthalamic path)**.

- **In Monty (the software), this changes completely.** Motor output is retained as a principle; **the thalamus is
  effectively omitted** (inter-module communication = the **Cortical Messaging Protocol (CMP)**, an engineering abstraction).
  Motor output is assigned to **no specific layer** (TBS25).

## 3. Displacement cells and the object-composition circuit

- **Layer placement: L5 (thick-tufted).** Framework19 Figure 7 caption: "we propose cortical grid cells are located in layer
  6 and displacement cells are in layer 5."
- **Commitment level: explicitly a new, uncertain prediction.** "The existence of displacement cells is a prediction
  introduced in this paper."
- **What they do:** encode the **relative pose between two objects/features** (pose-invariant) → the substrate for
  **compositional objects** and **behaviors** (movements as displacement sequences).
- **Caveat on Lewis19:** the "Locations" paper (with the concrete two-layer simulation) **does not implement displacement
  cells at all** — they are a **Framework19 construct**.

## 4. The "two networks / two circuits" idea

TBT contains **two distinct "two-ness" claims** that are easy to conflate:

**(a) Two *instances* of the *same* sensorimotor recognition circuit, stacked within one column** (HAC17): lower = **L4
(input) ↔ L2/3 (output)**, location from **L6a**; upper = **L6a (input) ↔ L5 (output)**, location again from L6a.

**(b) An object-*recognition* circuit vs an object-*composition/behavior* circuit** (Framework19): recognition = L4 + L2/3 +
L6a grid cells → *what object, at what pose*; composition/behavior = **L5 displacement cells** + trans-thalamic output → *how
objects are arranged and how they move*.

Note: hierarchy is folded into the **same repeating column** — every column is a full sensorimotor model of whole objects,
and hierarchy is achieved by **columns voting laterally** (L2/3 long-range) and by **L5→thalamus→higher-region** feedforward,
not by a dedicated non-columnar circuit. In TBP24/TBS25 this becomes **hierarchical LM stacking** + non-hierarchical
**voting**, both carried by the CMP with **no layer-specific pathway assignment**.

## 5. What TBT explicitly acknowledges simplifying or omitting

- **Number of cell layers modeled.** The 2017 *simulation* implements **only two layers of pyramidal neurons** (verbatim:
  "Our model contains two layers and one or more columns"). So: **6 anatomical layers → the theory engages ~4 (L4, L2/3, L6a,
  L5) → the running HAC17/Lewis19 simulation instantiates 2 pyramidal layers → Monty instantiates 0 explicit layers.**
- **Inhibitory interneurons — not modeled as cells.** Verbatim: "Our simulations do not model inhibitory neurons as
  individual cells. The functions of inhibitory neurons are encoded in the activation rules of the model." A "more detailed
  mapping to specific inhibitory neuron types is an area for future research."
- **Sublaminae.** Only **L6a** (vs L6b) and **L5 thick-tufted** are singled out; no L4, L5a/b, or L6a/b sublaminar
  decomposition. HAC17: "Cells we describe as residing in separate layers may actually intermingle in cortical tissue."
- **L2 vs L3.** **Collapsed** — a single "L2/3" output layer throughout.
- **Layer 1.** **Not modeled as a cell layer.** The HTM neuron *does* use apical segments for feedback (physiologically
  implicating L1), but L1 itself is never given cells or an explicit role.
- **Spikes / neural dynamics — dropped in the software.** TBP24/TBS25: "We do not need to simulate spikes"; object models are
  "explicit graphs in 3D Cartesian space"; learning is "the simplest possible form of Hebbian."
- **Agranular cortex.** TBT asserts the location machinery is universal across all regions; where L4 is not cytoarchitectonically
  visible, TBT's stance is that an **L4-equivalent input population still exists** — an assertion it leans on but does not
  resolve. Aligns with the minority "M1 has a cryptic L4" view ([Yamawaki et al. 2015,
  PMC4290446](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4290446/)) against the classical "M1 is agranular" position.
- **Other self-flagged unknowns (Framework19):** how grid/orientation/conjunctive cells jointly encode location; orientation
  representation "is unknown"; 3D extension in progress; the L5 time-multiplexing is speculative.

## 6. Comparison with the mainstream canonical microcircuit

**Where they AGREE:** L4 = principal thalamocortical *input*; L2/3 = principal cortico-cortical *output*; the L4→L2/3 motif;
L5 as a subcerebral/motor output with a thalamic-projecting branch (the PT/L5B class); L6 projects to thalamus.

**Where TBT DIVERGES:**
1. **The canonical serial cascade is L4 → L2/3 → L5 → L6.** Harris & Shepherd describe L5 PT as "the third and final stage"
   and L6 CT as an enigmatic feedback stage that is "remarkably silent," with "function largely a mystery." **TBT inverts the
   standing of L6:** rather than a quiet feedback endpoint, **L6a is an *active driver of L4*** carrying the location/grid
   signal. Mainstream anatomy treats L6→L4 as **modulatory/gain-control feedback**, not a computational *location code*.
   **This reinterpretation of the L6a↔L4 loop is TBT's single biggest anatomical departure.**
2. **Grid cells throughout neocortex** — mainstream localizes them to entorhinal cortex; TBT posits them in *every* column
   (in L6), a strong, unconfirmed claim.
3. **Displacement cells in L5** — a novel predicted cell type with no Harris & Shepherd counterpart.
4. **Hierarchy is columnar + trans-thalamic + lateral voting**, not a dedicated feedforward-hierarchy circuit.
5. **Organizing principle.** Harris & Shepherd argue the real invariants are **cell classes (IT/PT/CT) defined by projection
   ("hodology"), not lamination per se.** TBT is compatible (it keys on projection classes too) but frames its circuit in
   laminar terms, then in Monty discards lamination entirely.

## One-paragraph bottom line
TBT's sensorimotor column is a **two-layer input/output circuit — L4 (feature input) ⇄ L2/3 (stable object output) — steered
by a location signal from L6a**, where cortical **grid cells (predicted in L6)** perform path integration; a **possible
second copy runs L6a→L5**, and **L5 thick-tufted "displacement cells" (a new prediction)** handle object composition and
behavior, exporting motor commands (subcortical) and compositional-object signals (via **thalamic relay**) to higher regions.
The papers are explicit that this is heavy simplification: **only two pyramidal layers simulated**, **inhibition folded into
activation rules**, **L1 / sublaminae / L2-vs-L3 omitted**, "cells in separate layers may intermingle," grid/displacement
placement a **testable prediction not fact**; and **Monty drops the laminar mapping, spikes, grid cells, SDRs, and the
thalamus entirely**. Versus Douglas–Martin / Harris–Shepherd, TBT agrees on L4-input/L2-3-output/L5-subcerebral/
L6-corticothalamic identities but **diverges most sharply by turning the L6a→L4 feedback projection into an active grid-cell
location code and claiming grid cells exist in every column.**
