# Research References — CTKG Architecture

Papers referenced in the architecture design, with summaries of key ideas
relevant to the implementation.

---

## Prospective Configuration / Credit Assignment

### Inferring Neural Activity Before Plasticity as a Foundation for Learning Beyond Backpropagation
- **Authors**: Yuhang Song, Beren Millidge, Tommaso Salvatori, Thomas Lukasiewicz, Zhenghua Xu, Rafal Bogacz
- **Published**: Nature Neuroscience 27, 348–358 (February 2024)
- **Link**: https://www.nature.com/articles/s41593-023-01514-1
- **Code**: https://github.com/YuhangSong/Prospective-Configuration

**Key ideas**:
- The brain solves credit assignment with **prospective configuration**, not backpropagation.
- Before synaptic weights change, neural activity across the network first settles to a **prospective state** — the activation pattern that would have occurred if the weights were already correct.
- Two phases: (1) inference/relaxation phase — activations settle given the target observation, (2) weight update phase — weights consolidate the settled activity.
- This is the reverse of backpropagation, where weight changes lead and activity changes follow.
- Prospective configuration is implicitly followed by energy-based networks including Hopfield networks and predictive coding networks.
- Avoids catastrophic interference: correcting one prediction doesn't damage other correct predictions, because side effects are anticipated and compensated during the inference phase.
- Outperforms backpropagation on online learning, continual learning, few-shot learning, and learning in changing environments.
- Validated against human motion capture, human reinforcement learning, and mouse conditioned reflexes.

**Relevance to CTKG**: The graph already has activation dynamics (spread) and Bayesian edge updates (observe_present/absent). The inference phase would be an iterative spread-and-correct loop that settles activations before the weight update. The weight update is already Bayesian. The missing piece is backward spread — running activation from target observations back toward sources to find the prospective configuration.

### Introduction to Predictive Coding Networks for Machine Learning
- **Authors**: Erik Stenlund
- **Published**: arXiv:2506.06332 (May 2025)
- **Link**: https://arxiv.org/abs/2506.06332

**Key ideas**:
- Practical introduction to predictive coding networks (PCNs) for ML practitioners.
- PCNs are characterized by two-timescale dynamics: fast inference, slow learning.
- Fast inference: iterative message passing to minimise prediction errors at each layer.
- Slow learning: weight updates after inference converges.
- Includes CIFAR-10 benchmark with PyTorch implementation.
- Covers foundational architecture, inference and learning update rules.

**Relevance to CTKG**: The two-timescale structure maps directly onto the CTKG's fast path (spread + learn each timestep) and slow path (consolidation). The inference step in PCNs corresponds to the prospective configuration's relaxation phase.

### Predictive Coding Light (PCL)
- **Authors**: (Nature Communications, 2025)
- **Link**: https://www.nature.com/articles/s41467-025-64234-z

**Key ideas**:
- Recurrent hierarchical spiking neural network for unsupervised representation learning.
- Unlike previous predictive coding, does NOT transmit prediction errors to higher processing stages.
- Instead, suppresses the most predictable spikes and transmits a compressed representation.
- Uses only biologically plausible spike-timing-based learning rules.

**Relevance to CTKG**: The "suppress predictable, transmit surprising" principle aligns with the CTKG's surprise-driven learning. Nodes whose activation matches prediction are not noteworthy; nodes with high surprise drive edge creation (abduction).

### Predictive Coding as a Neuromorphic Alternative to Backpropagation: A Critical Evaluation
- **Published**: Neural Computation 35(12), 1881 (2023)
- **Link**: https://direct.mit.edu/neco/article/35/12/1881/117833

**Key ideas**:
- Critical evaluation of predictive coding as backpropagation alternative.
- Identifies scalability bottleneck: performance degrades significantly with depth beyond 5-7 layers.
- Competitive on standard benchmarks with shallow architectures.
- Deep PCNs suffer from exponentially larger energy magnitudes in layers closer to output.

**Relevance to CTKG**: The CTKG is not layered — it's a graph. The depth problem may not apply, but the energy magnitude issue could manifest as activation magnitude issues in long chains. Worth monitoring.

### Inspires Effective Alternatives to Backpropagation
- **Published**: PMC (2025)
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11881729/

**Key ideas**:
- Community direction: move away from "PC = BP correspondence" and focus on the inference learning / prospective configuration case where PC is distinct from BP.
- Goal: find algorithms that don't approximate backprop but can still perform credit assignment well enough.
- The brain is an existence proof that such algorithms are possible.

---

## Active Inference / Free Energy Principle

### Generalised Free Energy and Active Inference
- **Authors**: Thomas Parr, Karl J. Friston
- **Published**: Biological Cybernetics 113, 495–513 (2019)
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/
- **PDF**: https://discovery.ucl.ac.uk/10082773/1/Parr-Friston2019_Article_GeneralisedFreeEnergyAndActive.pdf

**Key ideas**:
- Active inference: a corollary of the free energy principle.
- Action and perception are cast as maximising Bayesian model evidence under generative models.
- The brain uses an internal generative model to predict incoming sensory data.
- Improves fit either through perceptual inference (optimising beliefs) or through acting on the world.
- Common objective function (variational free energy) for both action and perception.
- Expected free energy (EFE) decomposes into:
  - **Epistemic value** (exploration): prefer observations that reduce uncertainty about the world.
  - **Pragmatic value** (exploitation): prefer observations consistent with prior preferences.
- Provides principled explanation for exploration-exploitation dilemma, novelty, salience.
- Beliefs about states and policies are continuously updated to minimise variational free energy.
- Posterior beliefs about policies are based upon expected free energy.

**Relevance to CTKG**: The EFE decomposition maps directly onto action selection. Pragmatic value = prefer nodes with high resting potential (homeostatic preferences). Epistemic value = prefer actions leading to high-entropy (uncertain) regions of the graph. The unified objective replaces the current "pick most activated action" with "pick action that minimises expected free energy."

### The Missing Reward: Active Inference in the Era of Experience
- **Published**: arXiv:2508.05619 (2025)
- **Link**: https://arxiv.org/html/2508.05619v1

**Key ideas**:
- Active inference replaces external reward signals with an intrinsic drive to minimise free energy.
- Agents naturally balance exploration and exploitation through a unified Bayesian objective.
- Proposes that AIF provides the missing foundation for autonomous AI agents that learn from experience without constant human reward engineering.
- Explores integrating LLMs as generative world models within the active inference framework.

**Relevance to CTKG**: Validates the approach of using homeostatic priors instead of external rewards. The CTKG's graph IS the generative model — spread IS the prediction — and minimising free energy IS selecting actions that reduce the divergence between predicted and preferred observations.

### Emergence of Goal-Directed Behaviors via Active Inference with Self-Prior
- **Published**: arXiv:2504.11075 (April 2025)
- **Link**: https://arxiv.org/html/2504.11075

**Key ideas**:
- How an agent can autonomously form motivation and generate behaviour in environments where no goals are given.
- Homeostatic reinforcement learning: agent generates behaviours by maintaining certain sensory channels to specified thresholds, without environmental goals.
- Under active inference, a similar mechanism emerges: by minimising free energy with respect to an internal setpoint (preferred prior), an agent generates behaviour.
- Controllability constraints appear explicitly: the agent must exert sufficient control to enable homeostasis.

**Relevance to CTKG**: Directly supports the design where homeostatic priors (ENERGY_sated, HEALTH_healthy, CONTAMINATION_clean) are the only hardcoded preferences, and all other goals (find keycard, fix generator, escape) are discovered as instrumentally necessary for maintaining homeostasis.

### Reframing the Expected Free Energy
- **Published**: arXiv:2402.14460 (February 2024)
- **Link**: https://arxiv.org/pdf/2402.14460

**Key ideas**:
- Formalises the problem of deriving different EFE formulations (risk plus ambiguity, information gain / pragmatic value) from a single root definition.
- In one setting, the agent cannot have arbitrary prior preferences over observations — only a limited class of preferences is compatible with the likelihood mapping of the generative model.
- Unification of different EFE decompositions.

**Relevance to CTKG**: The constraint that preferences must be compatible with the generative model is important. The CTKG's prior preferences (homeostatic nodes with high resting potential) must be reachable via the graph's transition structure. A preference for a state with no path to it is meaningless.

### Expected Free Energy-based Planning as Variational Inference
- **Published**: arXiv:2504.14898 (April 2025)
- **Link**: https://arxiv.org/abs/2504.14898

**Key ideas**:
- Minimising a variational free energy functional naturally yields policies that integrate goal-directed behaviour, information-seeking exploration, and bounded rationality.
- EFE-based planning arises from minimising variational free energy on a generative model augmented with preference and epistemic priors.
- Demonstrates that planning and inference are the same operation under different objectives.

**Relevance to CTKG**: Planning in the CTKG is multi-hop spread (forward simulation through the graph). This paper confirms that planning and inference should share the same mechanism — spread with different stopping criteria (one-hop for immediate prediction, multi-hop for planning).

### Free Energy Projective Simulation (FEPS)
- **Published**: PLOS One (2025)
- **Link**: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0331047

**Key ideas**:
- Integrates reinforcement learning and active inference into an interpretable agent.
- In biological agents, preferences could be genetic (homeostatic), socially learned, acquired, or externally given.
- Separates learning into two tasks: model the environment and attain a goal.
- During exploration, actions whose outcomes reduce prediction errors are favoured (epistemic drive).

**Relevance to CTKG**: The separation between modelling (learning edge weights) and goal attainment (action selection via EFE) maps onto the CTKG's architecture. The fast path models the environment. Active inference uses the model for action selection.

### Active Inference: The Free Energy Principle in Mind, Brain, and Behavior
- **Authors**: Thomas Parr, Giovanni Pezzulo, Karl J. Friston
- **Published**: MIT Press (Open Access monograph)
- **Link**: https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind

**Key ideas**:
- Comprehensive textbook treatment of active inference.
- Covers the mathematical foundations, generative models, belief updating, policy selection.
- Open access.

**Relevance to CTKG**: Reference text for implementation details when building the active inference action selection module.

---

## Markov Categories / Categorical Probability

### A Synthetic Approach to Markov Kernels (Fritz 2020)
- **Link**: https://arxiv.org/abs/1908.07021
- **Published**: Advances in Mathematics 370 (2020)

**Key ideas**:
- Markov categories: the central categorical framework for probability.
- Morphisms are stochastic maps. d-separation, sufficient statistics, conditional independence proved synthetically.
- FinStoch (stochastic matrices) as canonical example.

**Relevance to CTKG**: The CTKG's transition edges with Beta posteriors are stochastic maps. The per-source normalisation in spread() is the Markov category composition. d-separation could be used to determine conditional independence between graph regions.

### The d-Separation Criterion in Categorical Probability (Fritz & Klingler 2023)
- **Published**: JMLR 24(46) (2023)

**Key ideas**:
- d-separation theorem in fully abstract Markov category terms.
- Soundness for arbitrary generalised causal models.

### A Characterization of Entropy in Terms of Information Loss (Baez, Fritz & Leinster 2011)
- **Link**: https://arxiv.org/abs/1106.1791
- **Published**: Entropy 13 (2011)

**Key ideas**:
- Shannon entropy is the unique functorial information measure.
- Three properties (functorial, convex-linear, continuous) uniquely determine Shannon entropy.

**Relevance to CTKG**: edge_entropy() in graph.py computes Shannon entropy of outgoing distributions. This uniqueness result grounds the choice of entropy as the uncertainty measure.

### Causal Inference by String Diagram Surgery (Jacobs, Kissinger & Zanasi 2019)
- **Published**: MSCS (2019)

**Key ideas**:
- Interventions (do-calculus) as endofunctors performing surgery on string diagrams.
- Complete diagrammatic treatment of causal reasoning.

**Relevance to CTKG**: Future work on causal reasoning in the graph. When the agent needs to reason about "what would happen if I did X" vs "what happened when I observed X," the distinction is intervention vs conditioning — formalised as string diagram surgery.

---

## Cortical Excitation-Inhibition Balance

### Winner-Take-All Dynamics Through Excitatory and Inhibitory Plasticity
- **Published**: Frontiers in Computational Neuroscience (2014)
- **Link**: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2014.00068/full

**Key ideas**:
- WTA networks are recurrently connected excitatory and inhibitory neuron populations.
- Competition through shared inhibition: excitatory populations drive a common set of inhibitory neurons, which provide global negative feedback.
- Stability emerges through interaction of biologically plausible plasticity mechanisms on all synapses simultaneously.

**Relevance to CTKG**: The spread() normalisation of positive edges (competition among excitatory targets) mirrors WTA dynamics. The shared inhibition model supports the architectural choice where inhibitory edges set a threshold rather than competing with each other.

### E/I Balance and Decision Making in Cortical Circuits
- **Authors**: Various
- **Published**: Journal of Neuroscience 42(6):1035 (2022)
- **Link**: https://www.jneurosci.org/content/42/6/1035

**Key ideas**:
- Selective excitatory neuron populations accumulate evidence through ramping activity via recurrent NMDA excitation.
- Lateral inhibition via GABAergic interneurons produces winner-take-all competition and categorical choice.
- Both elevating AND lowering E/I ratio impairs decision-making (inverted-U dependence).
- Elevated E/I ratio → impulsive decisions. Lowered E/I ratio → weakened evidence integration.

**Relevance to CTKG**: The inverted-U finding suggests the spread normalisation must balance excitation and inhibition carefully. Too much normalisation (all competition, no threshold) → impulsive. Too little (no competition, raw weights) → no selection.

### Distinct Feedforward Inhibition by Basket and Bistratified Interneurons
- **Published**: PMC4633480 (2015)
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC4633480/

**Key ideas**:
- **Basket cells modulate threshold** (subtractive inhibition): target the soma/proximal dendrites of pyramidal cells.
- **Bistratified cells modulate gain** (divisive inhibition): target distal dendrites.
- Concomitant feedforward inhibition by both types synergistically extends the dynamic range.
- This is the biological basis for the asymmetry in spread(): positive edges are normalised (gain/competition), negative edges set a threshold (subtractive).

**Relevance to CTKG**: Directly grounds the architectural decision. Excitatory normalisation = gain control by bistratified-like competition. Raw inhibitory weights = threshold-setting by basket-like suppression. A target node fires when normalised excitation exceeds raw inhibition.

### Parvalbumin Interneurons in Cortical Sensory Processing
- **Published**: PMC5693245 (2017)
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5693245/

**Key ideas**:
- PV+ fast-spiking interneurons convey precisely timed feedforward inhibition that lags just behind feedforward excitatory input from the thalamus.
- This creates a narrow temporal window for excitation: the excitatory signal must arrive and drive the target neuron BEFORE the slightly delayed inhibition shuts it down.
- Inhibitory interneurons regulate gain, suppress noise, and coordinate spike timing.

**Relevance to CTKG**: The temporal window concept could inform future work on activation dynamics — currently spread is instantaneous, but biologically the inhibition arrives slightly after excitation, creating a temporal competition window.

---

## Sensory-Motor Shared Representations

### A Sensory–Motor Theory of the Neocortex (Active Predictive Coding)
- **Author**: Rajesh P. N. Rao
- **Published**: Nature Neuroscience 27, 1221–1235 (2024)
- **Link**: https://www.nature.com/articles/s41593-024-01673-9
- **PDF**: https://homes.cs.washington.edu/~rao/Rao-Nature-Neuro-2024.pdf

**Key ideas**:
- **Active Predictive Coding (APC)**: each cortical area estimates both latent sensory states AND actions. The same area computes a state-prediction function (world model) and a policy function (action selection). These feed outputs to each other, producing sequences of state-action predictions usable for tracking, planning, or internal simulation.
- The cortex is "surprisingly uniform across cortical areas" — the same computational principle runs everywhere, with sensory and motor as roles within a unified architecture, not separate systems.
- Feedback from higher areas modulates the dynamics of state and action networks in lower areas. This is hierarchical predictive coding extended to actions.
- Explains: object recognition via eye movements, perceptual stability during movement, compositional part-whole learning, complex action planning from simple actions, episodic memory formation, and abstract concept learning.

**Relevance to CTKG**: The CTKG should not have separate sensory and motor representations. The same node `5` is both "perceived five" and "produced five." The distinction is in the DIRECTION of activation flow and the edge structure (efference copy = transition edge from action to predicted consequence). APC provides the theoretical justification for bare digit tokens (no digit_ prefix) — the brain uses one representation, not two.

### Efference Copy and Corollary Discharge
- **Origin**: von Holst & Mittelstaedt (1950) — efference copy; Sperry (1950) — corollary discharge
- **Overview**: https://en.wikipedia.org/wiki/Efference_copy

**Key ideas**:
- When the motor system issues a command, a COPY of that command is sent to sensory areas. This predicts the sensory consequences of the action.
- Sensory signals from self-generated movement (reafference) are cancelled by the efference copy, leaving only external signals (exafference).
- The signals are "carried by the same neuronal channels" but the efference copy distinguishes them.
- Sensory attenuation: self-generated stimuli are perceived as less intense (you can't tickle yourself).
- Derangement of this mechanism is hypothesised to produce symptoms of schizophrenia (misattributing self-generated signals as external).

**Relevance to CTKG**: The transition edge from an action to the next observation IS the efference copy. When the agent emits `5` and then sees `FEEDBACK_correct`, the `5 → FEEDBACK_correct` transition edge records the predicted sensory consequence of the action. The system can distinguish "I said 5" from "I saw 5" not by using different nodes, but by the edge role (TRANSITION from action vs CO-OCCURRENCE from observation).

### Rule-Based Sensorimotor Transformation Across Cortical Areas
- **Published**: eLife (2024)
- **Link**: https://elifesciences.org/reviewed-preprints/92620

**Key ideas**:
- Preparatory activity in motor cortex depends on the CONTEXT in which an action will be carried out.
- The same motor command is modulated by context-dependent gating.
- Causal evidence shows these changes support "flexible gating of actions."

**Relevance to CTKG**: This is what sigma (dynamic state) does. The same node `5` has different effective weight depending on the context (which other nodes are co-active). Sigma IS the context-dependent gating described in this paper.

---

## PoPE: Polar Coordinate Positional Embeddings

### Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings
- **Authors**: Anand Gopalakrishnan, Robert Csordas, Jurgen Schmidhuber, Michael C. Mozer
- **Published**: arXiv 2509.10534 (2025)
- **Link**: https://arxiv.org/abs/2509.10534

**The RoPE entanglement problem**: RoPE attention score is
`a_ts = SUM mu_q mu_k cos((s-t)theta_c + phi_k - phi_q)`.
The `phi_k - phi_q` term entangles content (what) with position (where).
Phase depends on BOTH the token's content AND its position. Attention
cannot independently match "this content" AND "at this relative position."

**PoPE fix**: Transform q/k through softplus for non-negative magnitudes,
set phase = position only:
`a_ts = SUM softplus(q) * softplus(k) * cos((s-t)*theta_c + delta_c)`.
Magnitudes encode CONTENT. Phases encode POSITION ONLY. The attention
score factorises as content-match x position-match. This conjunction
property enables rules like "match this content at exactly this offset."

**Key results**:
- Indirect indexing (find char at relative offset): RoPE 11%, PoPE 95%
- Superior length extrapolation without fine-tuning
- Outperforms RoPE on music, genomics, language modeling across scales

**Relevance to CTKG**: Our co-occurrence edges confound content association
(3 co-occurs with 4 from counting) with positional proximity (3 is
adjacent to space in most observations). PoPE's decoupling principle
means we should store TWO values per edge:

1. **Content weight** (the "what"): association strength regardless of
   position. 3-4 has high content weight from counting. 3-space has low.

2. **Position pattern** (the "where"): at what relative positions do these
   tokens co-occur. 3-4 at distance +3 in counting. 3-space at distance +1.

Attention score = content_weight x position_match(observed_dist, typical_dist).

Solves discrimination: 3-space has high positional match (always adjacent)
but low content weight. 3-4 has high content weight AND correct positional
match when 3 is question digit and 4 is the answer.

**Open question**: Store mean and variance of observed distance per edge.
Position match = Gaussian exp(-(d - mu)^2 / (2*sigma^2)). Two extra floats
per edge gives decoupled what-where attention.

---

## Sheaf Theory for Context-Dependent Graph Learning

### Sheaf Theory: From Deep Geometry to Deep Learning
- **Authors**: Anton Ayzenberg, Thomas Gebhart, German Magai, Gavin Solomadin
- **Published**: arXiv:2502.15476 (February 2025)
- **Link**: https://arxiv.org/html/2502.15476v1

**Key ideas**:
- Major survey bridging classical sheaf theory and machine learning applications.
- A cellular sheaf assigns a vector space (stalk) to each node and edge, and linear restriction maps between them. This generalises GNNs: standard GNNs are sheaves with trivial (identity) restriction maps.
- Sheaf Laplacian: the spectral object measuring "how far the current state is from consistency." Sheaf diffusion = minimise the sheaf Laplacian energy = drive the system toward consistency.
- Local section: a collection of node states that satisfy consistency equations on a subgraph. Global section: consistency across the entire graph.
- ε-harmonic edges: edges where the consistency violation is below threshold ε. Creates a filtration of subgraphs — hierarchical reasoning from strict to approximate alignment.
- Most notions for cellular sheaves on regular cell complexes translate to sheaves on arbitrary posets.

**Relevance to CTKG**: The CTKG IS a cellular sheaf. Nodes carry activation (stalks = ℝ¹). Edges carry weights that transform activations (restriction maps = scalar multiplication). The sheaf Laplacian measures how inconsistent the current activation pattern is with the edge structure. Sheaf diffusion IS spread(). The ε-harmonic filtration could identify hierarchical structure in the graph (tight clusters vs loose associations).

### A Sheaf-Theoretic and Topological Perspective on Complex Network Modeling and Attention Mechanisms in Graph Neural Models
- **Authors**: Chuan-Shen Hu
- **Published**: arXiv:2601.21207 (January 2026)
- **Link**: https://arxiv.org/abs/2601.21207

**Key ideas**:
- **Attention weights themselves define a cellular sheaf** (Theorem 1): a GAT triple (graph, features, attention weights) directly induces a sheaf where restriction maps are scalar multiplication by attention weights.
- Harmonicity quantifies local alignment: an edge is harmonic when the transferred node signals agree. This is the sheaf consistency condition.
- Multiscale extension via ε-harmonic filtration: persistence barcodes encode scale-dependent behavior of harmonic substructures.

**Relevance to CTKG**: This paper proves that attention IS a sheaf. Our edge weights already define a cellular sheaf (scalar restriction maps). The harmonicity condition tells us when two connected nodes have consistent activations given the edge weight. Non-harmonic edges indicate prediction error — exactly our learn step's "compare" phase. This paper provides the mathematical proof that our architecture is already a sheaf neural network.

### Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding
- **Authors**: Thomas Gebhart, Jakob Hansen, Paul Schrater
- **Published**: AISTATS 2023, PMLR 206:9094-9116
- **Link**: https://proceedings.mlr.press/v206/gebhart23a.html

**Key ideas**:
- Knowledge graph embedding IS an approximate global section of a knowledge sheaf.
- Consistency constraints are induced by the KG's schema.
- Composite relation reasoning works through harmonic extension with respect to the sheaf Laplacian — no special training needed.
- The spectral theory of sheaf Laplacians measures local and global consistency of embeddings.

**Relevance to CTKG**: Our node activations are the "embedding." The question "what digit follows 4 + succ?" is a harmonic extension problem: given activations on some nodes (4, succ, ?), extend to a global section consistent with the sheaf structure. The answer (5) is the harmonic extension. This paper formalises what we're trying to do.

### Sheaf Attention Networks (SheafAN)
- **Authors**: Federico Barbero et al.
- **Published**: Cambridge MLMI Dissertation (2022)
- **Link**: https://www.mlmi.eng.cam.ac.uk/files/2021-2022_dissertations/attention-based-sheaf-neural-networks.pdf

**Key ideas**:
- Formulates attention mechanisms over cellular sheaves.
- Introduces Attentive Sheaf Diffusion (ASD) PDE.
- Shows that standard Graph Attention Networks (GATs) are the special case where the underlying sheaf is trivial.
- A non-trivial sheaf enables richer, context-dependent message passing.

**Relevance to CTKG**: GATs are trivial sheaves. Our system needs a non-trivial sheaf to handle context-dependent predictions. The non-trivial restriction maps are exactly the dynamic σ state from BDH — the context-dependent part of the edge weight.

### Sheaf4Rec: Sheaf Neural Networks for Graph-based Recommender Systems
- **Published**: ACM Transactions on Recommender Systems (2024)
- **Link**: https://arxiv.org/html/2304.09097v3

**Key ideas**:
- Nodes are represented using vector spaces rather than single static vectors.
- Vectors are "only actualised at inference time" — dynamic, context-dependent representations.
- Up to 11.29% improvement over static-vector approaches.

**Relevance to CTKG**: The "actualised at inference time" concept matches BDH's dynamic σ. The node's effective representation depends on what else is active — it's not a fixed embedding.

---

## Natural Transformations / Category Theory for Structure Discovery

### Category-Theoretical and Topos-Theoretical Frameworks in ML
- **Published**: Axioms 14(3):204 (March 2025)
- **Link**: https://www.mdpi.com/2075-1680/14/3/204

**Key ideas**:
- First survey covering topos-based machine learning (higher category theory).
- In certain ML methods, compositionality of functors plays a vital role.
- Higher-order category theory explores causality via sheaves and presheaves,
  capturing local-global relationships in datasets.
- Four perspectives: gradient-based, probability-based, invariance/equivalence-based,
  and topos-based learning.

**Relevance to CTKG**: The local-global relationship captured by sheaves is exactly
what the CTKG needs: local observations (edge weights in a subgraph) must be
consistent with global structure (the natural transformation law). Topos theory
may provide the right framework for formalising this.

### Fundamental Components of Deep Learning (Gavranović thesis)
- **Link**: https://www.brunogavranovic.com/assets/FundamentalComponentsOfDeepLearning.pdf
- **GitHub**: https://github.com/bgavran/Category_Theory_Machine_Learning

**Key ideas**:
- End-to-end categorical foundation for deep learning.
- Uses actegories and parametric weighted optics.
- Natural transformations formalise equivariant layers in neural networks.
- Framework is prescriptive (implementable), not just descriptive.

### What is Category Theory to Cognitive Science? (PMC 2022)
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9716143/

**Key ideas**:
- A natural transformation is a computational process for transforming
  representations.
- The commutativity condition (the "naturality square") means transformed
  representations must be comparable to original representations.
- Category theory, like cognitive science, is about (re-)representation
  and comparison of compositional structure via structure-preserving maps.

**Relevance to CTKG**: The naturality square is the key constraint for discovering
NTs in the graph: if F maps BOARD_3 to ANSWER_4, and G maps BOARD_7 to ANSWER_8,
then the NT η must satisfy η_BOARD_3 ∘ F = G ∘ η_BOARD_7. In graph terms: the
mapping from "problem digit" to "answer digit" must commute with the mapping
from "problem instance" to "answer instance". This commutativity is what makes
it a NATURAL transformation (not just any mapping).

---

## Brain-Inspired Graph Architectures

### The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain
- **Authors**: Adrian Kosowski et al. (Pathway)
- **Published**: arXiv:2509.26507 (September 2025)
- **Link**: https://arxiv.org/abs/2509.26507
- **Full HTML**: https://arxiv.org/html/2509.26507v1
- **Code**: https://github.com/pathwaycom/bdh

**Key ideas**:
- BDH is a scale-free biologically inspired network of locally-interacting neuron particles, designed as a "missing link" between tensor-based Transformers and distributed graph models of the brain.
- ~1B parameters, comparable to GPT-2, but only ~5% of neurons active at any time (sparse positive activations, emergent, not forced via L1 regularization).
- Two types of connections: **fixed parameters** (learned during training) and **dynamic state σ** (synaptic weights updated per token during inference via Hebbian learning).
- **Attention IS the Hebbian update.** When neuron i is active and neuron j is active, synaptic weight σ(i,j) increases. That updated weight affects the NEXT spread. Context-dependent prediction happens automatically — edge weights change DURING inference based on what's currently active.
- Two edge-reweighting kernels operate in parallel:
  - **Modus ponens (reasoning)**: If X(i), σ(i,j) → A(j). Weighted beliefs propagate through learned rules.
  - **Hebbian learning (memory)**: Y(i), X(j) → σ(i,j). Co-activation strengthens connections.
- Key-value state: each synapse (i,j) holds a scalar weight that evolves based on current activations, past states, and fixed parameters. O(n²) state variables across all synapses (1:1 ratio with parameters), vs O(n) state in RNNs.
- Linear attention: neurons attend to each other based on key-value states localised to each neuron, creating "attention as pairwise deformation of correlations." Complexity is O(n) per token, not O(n²).
- Graph topology creates attention patterns: learned topology develops high Newman modularity with heavy-tailed degree distribution. Some neuron pairs become highly interconnected, others remain sparse.
- Monosemantic representations: individual synapses consistently strengthen whenever BDH hears or reasons about a specific concept. Interpretability is an inherent feature.
- Binding problem solved through sparse positive representations: each concept activates a sparse subset of neurons, and synaptic weights track co-occurrence. The dot product of sparse query and key vectors naturally selects relevant past contexts.
- **Critical limitation**: Learning still relies on backpropagation through time. Training without BPTT shows significant performance degradation.

**Relevance to CTKG**:
- Our architecture is structurally almost identical to BDH: nodes = neurons, edges = synapses, activation = sparse firing, Hebbian learning = our learn step. The key difference is that BDH has BOTH fixed parameters AND dynamic per-inference state, while our CTKG has only one set of edge weights (alpha/beta) that serve both roles.
- **The critical missing piece**: BDH's dynamic state σ changes WITHIN a single inference pass based on what's currently active. This is how attention works — the prediction for "digit_5 after seeing [4, succ]" is different from "digit_5 after seeing [7, succ]" because the synaptic state has been transiently updated by the current context. Our CTKG lacks this transient state, which is why digit→FEEDBACK edges are context-free (every digit leads to mostly FEEDBACK_wrong regardless of the question).
- BDH uses BPTT for learning, which we replace with prospective configuration. This is potentially an advantage — our system doesn't need the biologically implausible backward pass.
- The sparse activation pattern (5%) matches our activation threshold. The monosemantic representations match our goal of interpretable node semantics.
- **Connection to sheaf theory**: BDH's dynamic state is context-dependent — the effective edge weight σ(i,j) depends on what's currently active. This is exactly a sheaf: the "section" of edge weights over a given context (open set of active nodes) depends on which context you're in. Different contexts produce different local sections. Sheaf consistency requires that overlapping contexts agree on their overlap.

---

### Analogical Reasoning as a Core AGI Capability (Springer 2025)
- **Link**: https://link.springer.com/article/10.1007/s43681-025-00785-7

**Key ideas**:
- Analogical reasoning as a core capability for AGI.
- Transfer of learning through discovery of common relational structures
  in separate association structures.
- Knowledge repositories facilitate extraction of relational patterns
  across domains — essential for analogical transfer.

**Relevance to CTKG**: The "common relational structure" IS the natural
transformation. Two domains (counting, succession) share a common structure
(successor relationship), and discovering that structure enables transfer.

### Seven Sketches in Compositionality (Fong & Spivak)
- **Link**: https://www.researchgate.net/publication/323771055_Seven_Sketches_in_Compositionality_An_Invitation_to_Applied_Category_Theory

**Key ideas**:
- Applied category theory textbook with concrete real-world examples.
- Compositionality as the organising principle.
- Pipeline: natural language → objects, morphisms, paths, types → vectors.

### Physical Computing: A Category Theoretic Perspective (arXiv)
- **Link**: https://arxiv.org/html/2210.00392

**Key ideas**:
- Natural transformations model relations between different abstract
  representations of the same physical object.
- Example: decimal, octal, and binary adders related via NTs.
- The categorical framework captures structured and compositional
  nature of computing processes.

**Relevance to CTKG**: Different "representations" (counting warmup, succession
training, test questions) of the same mathematical relationship (successor)
should be related by natural transformations. The NT is what allows the system
to recognise that "counting from 3 to 4" and "succ(3)=4" are the same thing.
