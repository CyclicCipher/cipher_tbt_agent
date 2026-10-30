# DSNN: a dendritic spiking neural network — design (2026-10-30)

**What this experiment is** (the user, 2026-10-30): "Why don't we start, then, with figuring out where everyone else
went wrong with spiking neural networks. We might find out a way to make a training algorithm that doesn't require
differentiability and can train SNNs if we investigate dendrites."
- **The goal:** "to perfect a model of the human neuron and how to run it efficiently."
- **The networks:** 2 wide and 2 deep, and 2 wide and 3 deep. My reading: 2 pyramidal neurons per layer and 2 or 3
  layers, so 4 or 6 neurons, with the inputs not counted (for the user to confirm, §8).
- **The user's two premises:**
  - brain organoids learn faster and are more efficient than ANNs;
  - dendrites do a lot of complex processing that perceptrons don't capture.

  §3 records what each rests on.

**Status:** nothing is built. This doc is the research step:
- where SNN work went wrong (§1);
- what human neurons do (§2);
- the premises, checked (§3);
- the neuron model and how to run it (§4);
- training without derivatives (§5);
- the first experiments (§6), each to be pre-registered in `LAB.md` before any code.

**Fact or inference:** every claim is a cited finding unless marked *(inference)*. Most dendritic data are from rodents.
Human data come from tissue removed in surgery, mostly layers 2/3 and 5. The model takes human values where they
exist and says so per parameter.

## 0. Terms

- **Spike:** the soma's all-or-none action potential, about 1 ms long.
- **Compartment:** a part of the neuron simulated as one voltage: the soma, one basal branch, the apical trunk, one
  tuft branch.
- **Basal dendrites:** the branches around the soma. Most feedforward input arrives here.
- **Apical tuft:** the far end of the apical dendrite, in layer 1. Feedback from higher areas and the thalamus arrives
  here.
- **Dendritic spike:** a self-amplifying event in a dendrite. Three kinds:
  - the **NMDA spike**, in thin branches, held as a plateau for tens to hundreds of ms;
  - the **Na⁺ dendritic spike**, about a millisecond;
  - the **Ca²⁺ spike**, in the apical "hot zone" near where the tuft begins, tens of ms.
- **Plateau potential:** a sustained depolarization of a branch after an NMDA or Ca²⁺ spike.
- **bAP** (backpropagating action potential): the somatic spike travelling back into the dendrites. (Not the
  algorithm: the name is old.)
- **Burst:** 2–4 spikes within about 10–20 ms, against an isolated spike.
- **dCaAP:** the dendritic calcium action potential of human layer 2/3 neurons (§2.3).
- **Eligibility trace:** a synapse-local memory that recent activity happened. It decays over a set time, and a later
  signal changes the synapse only while the trace is nonzero.
- **Third factor:** a signal besides pre- and postsynaptic activity that gates plasticity: a neuromodulator such as
  dopamine, or a dendritic plateau.
- **Point neuron:** a model whose whole state is one voltage. The usual one is the LIF (leaky integrate-and-fire):
  τ dV/dt = −V + R·I(t), with a spike and a reset at threshold.

## 1. Where SNN work went wrong

**What an SNN usually is:** a network of LIF point neurons. The synapses carry all the learning; the neuron is a leaky
sum and a threshold.

1. **The neuron stayed a perceptron.**
   - **The fact:** reproducing one rodent layer 5 pyramidal neuron's output spikes from its synaptic input took a
     temporal convolutional network of 5–8 layers. With its NMDA receptors removed, one hidden layer sufficed
     (Beniaguev, Segev & London 2021).
   - *(inference)* So N LIF units compute what N perceptrons with a leak compute, while N real pyramidal neurons
     compute something closer to N deep networks. The field took the spike from biology and left out the dendrite,
     which does most of the computing.
2. **The training was either borrowed or too weak.** Three routes:
   - **ANN-to-SNN conversion:** train an ANN, then map its activations to firing rates. The SNN becomes a slow
     approximation of the ANN, needing many time steps per input.
   - **Surrogate gradients through time** (Neftci, Mostafa & Zenke 2019):
     - **How:** replace the spike's zero derivative by a smooth stand-in, then run backprop through time.
     - **What it gives:** it works, near ANN accuracy on several benchmarks.
     - **The cost:** it is backprop again. It is non-local, it stores the whole activity history, and it treats the
       spike as a nuisance to smooth away.
   - **STDP with winner-take-all** (Diehl & Cook 2015: 95% on MNIST with 6,400 excitatory neurons): local and
     unsupervised, but with no credit assignment across layers. It hasn't scaled.

   *(inference)* The field took either backprop or biology's least informative rule. It skipped the rules that use
   the dendrite (§2.6).
3. **The plasticity model was pair STDP**: a weight change set by the timing of one presynaptic and one postsynaptic
   spike, within tens of ms. In cortex, plasticity depends on where the synapse sits and on dendritic events:
   - **Dendritic spikes alone give LTP,** with no somatic spike (Golding, Staff & Spruston 2002).
   - **STDP's sign and size change with location** on the dendrite (Froemke, Poo & Dan 2005).
   - **Distal synapses need cooperation:** dendritic depolarization from neighbouring inputs (Sjöström & Häusser
     2006). Rate, timing and cooperativity jointly decide the change (Sjöström, Turrigiano & Nelson 2001).
   - **One plateau can teach in one trial:** in the hippocampus's CA1 region, a single plateau potential creates a
     place field. It potentiates inputs active up to seconds before or after it, a window about 1,000 times wider
     than STDP's. This is BTSP, behavioral-timescale synaptic plasticity (Bittner et al. 2017).
4. **The tasks gave time no role.** SNNs were benchmarked mostly on static images converted to spike rates. *(inference)*
   On such tasks a spiking network can at best match the ANN.
5. **The hardware lottery** (Hooker 2020): GPUs reward dense, clocked matrix multiplies, and spikes are sparse events.
   Neuromorphic chips with multi-compartment neurons exist (Intel's Loihi: Davies et al. 2018), but most SNN work runs
   dense, time-stepped simulation on GPUs.
6. **The state that learning uses was left out:**
   - homeostasis;
   - short-term synaptic dynamics;
   - neuromodulation;
   - inhibition aimed at particular compartments;
   - branch excitability;
   - the timescales: a LIF has one membrane time constant (about 10–20 ms), while a neuron has 1-ms spikes, plateaus
     of about 100 ms, and plasticity windows of seconds.

   Each of these is a learning mechanism (§2), not a detail.

**What is not a mistake:** surrogate-gradient SNNs work. The critique is against this experiment's goal, efficient
training without derivatives, not against SNNs working at all.

## 2. What human neurons do

### 2.1 Shape and compartments
- **Two input zones** (Spruston 2008; Larkum 2013):
  - the basal dendrites near the soma receive mostly feedforward input;
  - the apical tuft in layer 1 receives feedback from higher areas and the thalamus.
- **Human dendrites are longer and more branched** than rodents' (layer 2/3 pyramidal neurons: Mohan et al. 2015).
- **Human dendrites are more isolated electrically** (layer 5: Beaulieu-Laroche et al. 2018). Tuft input changes the
  soma's output less, so the dendrite computes more on its own.
- **Synapses:** on the order of 10⁴ excitatory synapses per pyramidal neuron, mostly on spines. Human models from
  spines to spikes: Eyal et al. 2018.

### 2.2 The membrane and the synapses
- **Half the capacitance:** human cortical neurons' specific membrane capacitance is about 0.5 µF/cm², against the
  usual 0.9–1. Synaptic potentials reach the soma faster, and the neuron follows faster input (Eyal et al. 2016).
- **Fewer channels per volume:** across 10 mammals, layer 5 neurons keep their ion-channel conductance per volume
  roughly constant. Human neurons are the exception, with lower densities, which would lower the energy cost of
  signalling per volume (Beaulieu-Laroche et al. 2021).
- **Fast-recovering synapses:** human synapses recover from depression faster than rodents' and carry information at
  higher rates (Testa-Silva et al. 2014).
- **Strong single connections:** one human pyramidal spike can fire a postsynaptic interneuron and set off a chain of
  events (Molnár et al. 2008).

### 2.3 Dendritic nonlinearities: the neuron's subunits

| event | where | trigger | what it computes | source |
|---|---|---|---|---|
| NMDA spike and plateau | thin basal and tuft branches | tens of synapses active together on a short stretch of one branch | a sigmoid of the branch's summed input, held for tens to hundreds of ms. Clustered inputs sum supralinearly; inputs spread over branches sum linearly | Schiller et al. 2000; Polsky, Mel & Schiller 2004; Major, Larkum & Schiller 2013 |
| the two-layer neuron | the whole cell | — | each thin branch a sigmoid subunit and the soma their sum: the neuron's firing is predicted well as a two-layer network | Poirazi, Brannon & Mel 2003 |
| Ca²⁺ spike, BAC firing | layer 5 apical hot zone | a bAP and tuft input within a few ms | coincidence of feedforward (soma) and feedback (tuft) input gives a burst | Larkum, Zhu & Sakmann 1999 |
| dCaAP | human layer 2/3 apical dendrites | input just above threshold | non-monotonic: largest for threshold-level input, smaller for stronger input. One neuron classifies XOR | Gidon et al. 2020 |
| sequence selectivity | within one branch | inputs activated in order along the branch | inward and outward order give different responses: order and direction detection | Branco, Clark & Häusser 2010 |
| prediction by depolarization | basal and tuft segments | context input | an NMDA plateau depolarizes the soma without firing it, so the neuron fires sooner when the input it predicted arrives | Hawkins & Ahmad 2016 |

**Where the NMDA nonlinearity comes from:** the NMDA receptor conducts only when glutamate is bound *and* the membrane
is depolarized, which removes its Mg²⁺ block. The block's voltage dependence is
B(V) = 1 / (1 + [Mg²⁺]/3.57 mM · e^(−0.062·V/mV)) (Jahr & Stevens 1990). This one factor turns coincident input on a
branch into a regenerative event.

### 2.4 The output: events and bursts
- **Two signals on one axon** (Naud & Sprekeler 2018):
  - the **event rate** (isolated spikes and bursts, each counted once) follows the basal, feedforward input;
  - the **burst fraction** follows the apical, feedback input.
- **Separated at the target:** depressing synapses pass the event rate, and facilitating synapses pass bursts (same
  source).
- **Sparse and costly:**
  - most of the brain's signalling energy goes to reversing the ion flows of synaptic currents and action potentials
    (Attwell & Laughlin 2001);
  - the energy budget lets fewer than about 1% of cortical neurons be strongly active at once (Lennie 2003).

### 2.5 Inhibition aimed at compartments
- **PV basket cells target the soma:** gain and spike timing.
- **SST (Martinotti) cells target the apical dendrites:** they gate dendritic spikes and dendritic plasticity.
- **VIP cells target SST cells:** they release the dendrites from inhibition (Tremblay, Lee & Rudy 2016).
- **A human-specific type:** rosehip interneurons target pyramidal cells' apical dendrites (Boldog et al. 2018).

### 2.6 Plasticity: what a synapse can use
**The information local to a synapse:**
- its presynaptic input trace;
- the local dendritic voltage;
- the local calcium;
- whether its branch is in a plateau;
- bAPs, and bursts;
- neuromodulators.

| rule | what it uses | what it does | source |
|---|---|---|---|
| calcium thresholds | local Ca²⁺ only: above a high threshold LTP, between a low and the high threshold LTD | reproduces STDP, its dependence on rate, and its dependence on dendritic location | Graupner & Brunel 2012 |
| voltage-based STDP | the presynaptic trace × the low-passed local postsynaptic voltage | STDP plus rate effects; with homeostasis, the connectivity comes to reflect the coding | Clopath et al. 2010 |
| plateau-driven (BTSP) | eligibility (presynaptic activity within a window of seconds) × a plateau | one-shot learning; bidirectional, moving weights toward a level the plateau sets | Bittner et al. 2017; Milstein et al. 2021 |
| burst-dependent | the presynaptic event trace × (bursting − its running average) | apical feedback sets each neuron's sign of plasticity: credit assignment across the layers of a multilayer network | Payeur et al. 2021 |
| dendritic prediction of somatic spiking | the presynaptic trace × (somatic spikes − the rate the dendrite predicts) | the dendrite learns to predict the soma, so a teacher that drives the soma decides what the dendrite learns | Urbanczik & Senn 2014 |
| branch strength | repeated dendritic spikes in one branch (NMDA-receptor-dependent) | the branch's own excitability potentiates: the branch, not only the synapse, stores | Losonczy, Makara & Magee 2008 |
| three-factor | an eligibility trace × a neuromodulator | learning gated by reward or surprise, over seconds | Frémaux & Gerstner 2016 |
| structural | the making and removal of synapses; related inputs cluster on one branch | wiring as learning | Kastellakis & Poirazi 2019 |
| homeostatic scaling | the neuron's average rate against a set point | all its synapses scaled multiplicatively | Turrigiano et al. 1998 |
| stochastic release | each synapse's random release × a global reward | learning from noise the synapse already has: reward climbs with no derivative taken | Seung 2003 |

## 3. The two premises, checked

**"Organoids learn faster and are more efficient than ANNs":**
- **DishBrain** (Kagan et al. 2022):
  - **The setup:** about 800,000 cortical cells, human (from stem cells) or mouse, on a multi-electrode array, played a
    simplified Pong.
  - **The result:** performance improved within about 5 minutes.
  - **The feedback:** a predictable stimulus when the paddle hit, unpredictable noise when it missed. There was no
    reward signal, only predictability (the authors' framing: the free-energy principle).
- **Against deep RL** ([Khajehnejad et al. 2024](https://arxiv.org/pdf/2405.16946); journal version 2025): with about
  70 episodes each (one 20-minute session), the cultures beat DQN, A2C and PPO on rally measures.
  - **The caveat:** one simplified game, at a sample budget where deep RL is known to be weak.
  - **What it shows:** sample efficiency at small budgets, not general superiority.
- **Brainoware** (Cai et al. 2023): an organoid used as a reservoir, with a trained linear readout, about 78% on speaker
  identification. The organoid itself isn't trained by backprop; its own plasticity improved the score.
- **Energy:** I found no measured comparison of energy per unit of learning. The brain's roughly 20 W is the usual
  reference. Treat "more efficient" as a hypothesis this experiment measures (events per trial, §6).
- **What to translate:** DishBrain's learning signal is the predictability of the input. It is local, takes no
  derivative and needs no reward channel.

**"Dendrites do what perceptrons don't":**
- **Biology:** §2.3, and the 5–8 layers of Beniaguev et al. (§1).
- **In ANNs:**
  - **Dendritic ANNs:** matched or beat standard ANNs on image classification with far fewer trainable parameters
    ([Chavlis & Poirazi 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11419189/); the authors' claim, not
    replicated).
  - **One neuron with a binary dendritic tree** learns nontrivial tasks (Jones & Kording 2021).
- **Dendritic SNNs** exist, but are trained by surrogate gradients (e.g. Zheng et al. 2024): the dendrite was added, and
  the training kept.
- **The closest to our goal:** a local calcium-and-dopamine rule in one detailed striatal neuron, which solved the
  nonlinear feature-binding problem through dendritic plateaus ([eLife 2025, 97274](https://elifesciences.org/articles/97274)).

*(inference)* Dendrites with a training rule that takes no derivatives, in a network, is open.

## 4. The design: the neuron

### 4.1 What the model must do

| # | requirement | from | its test (the classic experiment, reproduced in the model) |
|---|---|---|---|
| N1 | branch subunits: a local threshold per thin branch, by the NMDA voltage dependence | §2.3 | clustered against dispersed input: supralinear against linear summation (Polsky et al. 2004) |
| N2 | two integration zones and their coincidence: basal with soma, and apical tuft. Both together give a burst | Larkum et al. 1999 | a bAP plus tuft input within a few ms gives a burst; either alone gives at most one spike |
| N3 | a non-monotonic compartment (the dCaAP) | Gidon et al. 2020 | XOR of two inputs in one neuron |
| N4 | order sensitivity within a branch | Branco et al. 2010 | inward and outward sequences give different responses |
| N5 | events and bursts as two outputs on one axon | Naud & Sprekeler 2018 | the event rate follows basal input; the burst fraction follows apical input |
| N6 | plateaus lasting tens to hundreds of ms | §2.3 | the response to an input depends on an input about 100 ms earlier |
| N7 | human values where known: capacitance about 0.5 µF/cm², a tuft more isolated from the soma | §2.1, §2.2 | — |
| N8 | sparse events, with rates held in a band by homeostasis | §2.4, §2.6 | — |

### 4.2 Running it efficiently: a ladder of models
- **The reference** (expensive, for fidelity only):
  - **What:** detailed multi-compartment Hodgkin–Huxley models of human neurons from ModelDB: Eyal et al. 2018's
    human layer 2/3 models, and Gidon et al. 2020's dCaAP model.
  - **How it runs:** in the NEURON simulator. *Not checked: whether NEURON installs in our Windows venv.*
- **The model we train** (cheap):
  - **Compartments:** a soma, a few basal branches, the apical trunk with its Ca²⁺ zone, and a few tuft branches.
  - **Dynamics:** each compartment is one voltage, with a leak and coupling conductances to its neighbours. The soma is
    an adaptive exponential integrate-and-fire unit (Brette & Gerstner 2005). NMDA input carries the Mg²⁺ block
    (§2.3). The Ca²⁺ spike and the dCaAP are threshold events of set shape.
  - **Precedents:** Pinsky & Rinzel 1994 (two compartments); Naud, Bathellier & Gerstner 2014 (a two-compartment model
    with BAC firing that predicts layer 5 spike times).
- **The fidelity measure** (Beniaguev et al.'s): drive the reference and the reduced model with the same random
  synaptic input, then score how well the reduced model predicts the reference's spike times and somatic voltage.
- **The cost measure:**
  - compartment updates and synaptic events per simulated second;
  - wall-clock time per simulated second, on CPU.
- **How to make it fast:**
  - **Jump, don't step:** between inputs, a linear compartment's voltage has an exact solution (Rotter & Diesmann
    1999), so the simulation can jump to the next event.
  - **Update only what moves:** compartments that receive input or are in an active event.
  - **Plateaus as timers:** a plateau can be an "on" state with a duration instead of integrated dynamics, if the
    fidelity measure allows it.
- *(inference)* At 4–6 neurons, cost doesn't limit the experiments themselves. It decides whether the model can grow,
  so it is measured from the start.

### 4.3 The network
- **Size:** 2 × 2 and 2 × 3 (my reading: see the header).
- **Wiring** (§2.1):
  - feedforward input reaches the next layer's basal branches;
  - feedback from the layer above reaches the apical tufts (for the top layer: the teacher or the environment);
  - inhibition is aimed at compartments, the soma or the dendrite.
- **Not designed:** whether inhibitory interneurons are cells (making the network wider than 2), or inhibitory input to
  the compartments.
- *(inference)* **Why this size suits the question:**
  - every compartment of every neuron can be inspected;
  - the parameter count is small enough for exact comparisons (true gradients, perturbation);
  - depth 3 adds a hidden layer whose credit must come through another learning layer, the case that separates
    credit-assignment rules from local tricks.

## 5. Training without differentiability

**The hypothesis** *(inference)*:
- **What the SNN field kept:** the perceptron's division of labour. The neuron sums, and a global error computed by the
  chain rule tells each weight what to do. The field swapped the activation for a spike, which the chain rule can't
  pass through.
- **The dendrite's division instead:**
  - each neuron compares two of its own compartments: feedforward on the basal side, feedback on the apical side;
  - the comparison's outcome is an event, a plateau or a burst;
  - each synapse learns from its local traces and that event.
- **Why there'd be nothing to differentiate:** the learning signal is itself a spike-like event.

**The candidates** (each local; from §2.6):
1. **Burst-dependent plasticity** (Payeur et al. 2021): shown to assign credit across layers. The closest to a
   replacement for backprop.
2. **Dendritic prediction** (Urbanczik & Senn 2014): the error is the soma's spiking minus the dendrite's prediction of
   it.
3. **Plateau-gated BTSP** (Bittner et al. 2017; Milstein et al. 2021): one-shot, over a window of seconds, with the
   plateau as the teacher.
4. **Calcium thresholds** (Graupner & Brunel 2012), as the one local variable the others act through. *(inference)*
   Plateaus, bursts and bAPs all raise calcium, so this may unify candidates 1–3.
5. **Three-factor learning:** eligibility traces and a global signal, reward or surprise. The eLife 2025 rule (calcium,
   dopamine and metaplasticity) solved nonlinear feature binding in one neuron.
6. **The predictability signal** (DishBrain): unpredictable input as the only negative signal.
7. **Stochastic release as the perturbation** (Seung 2003): the noise the synapse already has does the probing. It
   links to Dust 2's perturbation work.
8. **Structural plasticity:** moving synapses between branches (clustering).
9. **Stabilizers:** homeostasis and branch-strength plasticity.

**The problem all of them share** (the weight-transport problem): how does feedback reaching a hidden neuron's tuft get
the right sign for that neuron? Two known options:
- fixed random feedback (feedback alignment: Lillicrap et al. 2016);
- feedback weights learned to mirror the forward ones (Akrout et al. 2019).

The 2 × 3 network is where this shows.

**Ruled out** (the user, 2026-10-10, in Dust 2): equilibrium propagation and settling. It forces an architecture and
costs more than backprop.

**Controls:** each runs on the same dendritic network, so they test the rule, not a rival model.
- **Surrogate-gradient backprop through time:** the reference for what a local rule loses. It is differentiable, so
  it's for the user to allow (§8).
- **Weight or node perturbation** (Fiete & Seung 2006; Dust 2): the general baseline that takes no derivatives, cheap at
  this parameter count.
- **The dendrite removed:** the same tasks with point (LIF) neurons, and with a same-size perceptron network trained by
  backprop. This measures what the dendrite buys.

## 6. First experiments (each pre-registered in `LAB.md` before code)

- **E0, the neuron:**
  - the reduced model passes the tests of N1–N6;
  - its fidelity against the reference, if NEURON runs;
  - its cost, measured.
- **E1, one neuron learns:**
  - XOR in a single neuron (N3), and a task that needs branch clustering (N1);
  - learned by candidates 1–5;
  - the point-LIF control can't do XOR.
- **E2, the 2 × 2 and 2 × 3 networks:** tasks a same-size point-neuron network can't do, or does slowly:
  - **4-bit parity.** The standard threshold-unit construction uses 4 hidden units. One non-monotonic unit per odd
    count of active inputs would need 2. *To check exhaustively before use:* whether 2 × 2 monotone threshold units
    can't.
  - **Temporal order and interval tasks** (N4, N6).
  - **A closed-loop paddle game with DishBrain's predictability feedback:** the organoid premise, tested in our model.
- **Measures:**
  - trials to criterion (sample efficiency);
  - events and compartment updates per trial (an energy proxy);
  - wall-clock time per simulated second.
- **Runs:** CPU, each under 2 minutes; never re-run unchanged.

## 7. Rules this experiment inherits

- **The general-learner standard:** improve the learner only through the data, or through sound general architecture.
  Here the neuron's mechanisms are the architecture. Nothing is tuned for one task's outcome.
- **The working rules:** no meaning built in; pre-register before code; runs under 2 minutes; never re-run unchanged;
  verify the apparatus before trusting a number; one file = one concept.
- **From Dust 2** (the user): critical-state dynamics come from local thresholds and homeostasis, not top-k.

## 8. Open, for the user

1. **The reading of "2 wide":** 2 pyramidal neurons per layer, inputs and inhibition not counted?
2. **The fidelity target:** a detailed reference (NEURON with a published human model), or the classic experiments of
   N1–N6 alone?
3. **The control:** is the surrogate-gradient control allowed, as a reference only?

## Sources

**SNNs and their training**
- Neftci, Mostafa & Zenke 2019, *IEEE Signal Processing Magazine* 36:51 (surrogate gradients).
- Diehl & Cook 2015, *Front. Comput. Neurosci.* 9:99.
- Hooker 2020, "The Hardware Lottery", arXiv 2009.06489.
- Davies et al. 2018, *IEEE Micro* 38:82 (Loihi).
- Zheng et al. 2024, *Nat. Commun.* 15:277 (dendritic heterogeneity in SNNs).

**Single neurons and dendrites**
- Beniaguev, Segev & London 2021, *Neuron* 109:2727.
- Poirazi, Brannon & Mel 2003, *Neuron* 37:989.
- Polsky, Mel & Schiller 2004, *Nat. Neurosci.* 7:621.
- Schiller et al. 2000, *Nature* 404:285.
- Major, Larkum & Schiller 2013, *Annu. Rev. Neurosci.* 36:1.
- Larkum, Zhu & Sakmann 1999, *Nature* 398:338.
- Larkum 2013, *Trends Neurosci.* 36:141.
- Spruston 2008, *Nat. Rev. Neurosci.* 9:206.
- Branco, Clark & Häusser 2010, *Science* 329:1671.
- Hawkins & Ahmad 2016, *Front. Neural Circuits* 10:23.
- Jahr & Stevens 1990, *J. Neurosci.* 10:3178.
- Naud & Sprekeler 2018, *PNAS* 115:E6329.

**Human neurons**
- Gidon et al. 2020, *Science* 367:83.
- Eyal et al. 2016, *eLife* 5:e16553.
- Eyal et al. 2018, *Front. Cell. Neurosci.* 12:181.
- Beaulieu-Laroche et al. 2018, *Cell* 175:643.
- Beaulieu-Laroche et al. 2021, *Nature* 600:274.
- Mohan et al. 2015, *Cereb. Cortex* 25:4839.
- Testa-Silva et al. 2014, *PLoS Biol.* 12:e1002007.
- Molnár et al. 2008, *PLoS Biol.* 6:e222.
- Boldog et al. 2018, *Nat. Neurosci.* 21:1185.

**Inhibition and energy**
- Tremblay, Lee & Rudy 2016, *Neuron* 91:260.
- Attwell & Laughlin 2001, *J. Cereb. Blood Flow Metab.* 21:1133.
- Lennie 2003, *Curr. Biol.* 13:493.

**Plasticity**
- Golding, Staff & Spruston 2002, *Nature* 418:326.
- Froemke, Poo & Dan 2005, *Nature* 434:221.
- Sjöström & Häusser 2006, *Neuron* 51:227.
- Sjöström, Turrigiano & Nelson 2001, *Neuron* 32:1149.
- Bittner et al. 2017, *Science* 357:1033.
- Milstein et al. 2021, *eLife* 10:e73046.
- Payeur et al. 2021, *Nat. Neurosci.* 24:1010.
- Urbanczik & Senn 2014, *Neuron* 81:521.
- Graupner & Brunel 2012, *PNAS* 109:3991.
- Clopath et al. 2010, *Nat. Neurosci.* 13:344.
- Losonczy, Makara & Magee 2008, *Nature* 452:436.
- Frémaux & Gerstner 2016, *Front. Neural Circuits* 9:85.
- Kastellakis & Poirazi 2019 (synaptic clustering and memory formation).
- Turrigiano et al. 1998, *Nature* 391:892.
- Seung 2003, *Neuron* 40:1063.
- [eLife 2025, article 97274](https://elifesciences.org/articles/97274) (a local calcium- and reward-based rule
  enhancing dendritic nonlinearities).

**Credit assignment and perturbation**
- Fiete & Seung 2006, *Phys. Rev. Lett.* 97:048104.
- Lillicrap et al. 2016, *Nat. Commun.* 7:13276.
- Akrout et al. 2019, NeurIPS (learning without weight transport).

**Reduced models and simulation**
- Brette & Gerstner 2005, *J. Neurophysiol.* 94:3637.
- Pinsky & Rinzel 1994, *J. Comput. Neurosci.* 1:39.
- Naud, Bathellier & Gerstner 2014, *Front. Comput. Neurosci.* 8:90.
- Rotter & Diesmann 1999, *Biol. Cybern.* 81:381.

**Cultures, organoids and dendritic ANNs**
- Kagan et al. 2022, *Neuron* 110:3952.
- [Khajehnejad et al. 2024, arXiv 2405.16946](https://arxiv.org/pdf/2405.16946); journal version *Cyborg Bionic Syst.*
  2025.
- Cai et al. 2023, *Nat. Electron.* 6:1032.
- [Chavlis & Poirazi 2025, *Nat. Commun.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11419189/)
- Jones & Kording 2021, *Neural Comput.* 33:1554.
