# The mathematics of beauty: innate, cross-species preferences as computable objectives

**Status:** research document started 2026-10-20 from the user's thesis and source collection. Nothing built. The evidence
notes mark which claims are solid, contested, or still to verify.

## The thesis (the user)

- **Alignment requires creating the preferences that are innate to humans,** the ones evolution created. Many human
  values and preferences are already observable in infants.
- **Some mathematical principles are shared between humans and animals,** and they underlie aesthetic preference.
- **The method:**
  1. separate out cultural influences by keeping only what is **common to humans and a diverse array of animal
     species**;
  2. then keep only what is **mathematical**: a principle that can score input data.
- "How do you quantify beauty, and thus program a preference for it?" Plato asked where beauty comes from, without
  neuroscience or linear algebra.

## The method in more detail, with its pitfalls

1. **Intersect across species.** A preference present in humans *and* in phylogenetically distant animals (insects,
   fish, birds, mammals) is unlikely to be cultural.
   - **Pitfall:** the overlap can be **convergent** (similar environments, similar solutions) rather than inherited.
     Both are fine for our purpose (not cultural), but they predict different things about which preferences are
     universal.
   - **Not a hard gate (the user, 2026-10-20: "I would caution against making the diverse animals filter too hard,
     in hindsight").** The filter is one line of evidence that a preference is not cultural, weighed with the others:
     - **infants** (before culture can act), and **cross-cultural uniformity,** especially in isolated groups
       (e.g. hunter-gatherers);
     - **absence in one species is weak evidence:** one study, one task, a sensory system and ecology unlike ours,
       and gaze is not preference (in the Hai//om group the two disagreed);
     - **an innate preference can be human-specific:** one that evolved after our split from the apes is still
       innate, not cultural. Language-related and face preferences are likely examples.
     - So each principle gets a **graded** "not cultural" weight, not a pass/fail.
2. **Keep the mathematical ones:** principles that define a function score(x) on raw input (an image, a sound, a
   grid, a sequence), not on cultural labels.
3. **Pitfalls to guard against:**
   - **Measurement mismatch:** animal "preference" is measured through mate choice or foraging; human preference
     through ratings. The same name can hide different quantities.
   - **Publication bias:** the fluctuating-asymmetry literature had a well-documented decline effect (A. R. Palmer
     1999, "Detecting publication bias in meta-analyses: a case study of fluctuating asymmetry and sexual selection").
     Strong early effects shrank.
   - **Innate is not the same as right.** An evolved preference can be exploited (supernormal stimuli, see below) or
     can conflict with other values. That is the meta-alignment question of `GOALS.md` "Scope of claims".

## The catalogue: principle → mathematics → a computable score → evidence

| # | principle | mathematics | a score on input x | evidence (humans / animals) | strength |
|---|---|---|---|---|---|
| 1 | **Symmetry** (bilateral, radial) | invariance under a group action: f(g·x) = f(x) for reflections or rotations g | 1 − ‖x − g·x‖ / ‖x‖, maximized over the group's elements and axes; or the description-length saving from coding the half | humans rate symmetric faces as more attractive (e.g. Rhodes and others). Barn swallow tails (Møller 1992); zebra finch plumage (Swaddle & Cuthill 1994); bees and flower symmetry (Møller 1995, a nectar link) | **mixed:** a human preference exists; the animal sexual-selection effects shrank with replication |
| 2 | **Scale invariance, 1/f spectra, fractal dimension** | amplitude spectrum ∝ 1/f^α (natural scenes: α ≈ 1, power α ≈ 2, Field 1987); box-counting fractal dimension D | the fitted spectral slope α; box-counting D | humans prefer mid-range fractal D ≈ 1.3–1.5 (Spehar, Clifford, Newell & Taylor 2003); efficient coding (Barlow 1961). Claims of stress reduction (Taylor 2006) and of animal neurons tuned to 1/f statistics | the human D preference is solid; **physiological and animal claims to verify** |
| 3 | **Averageness / prototypicality (koinophilia)** | distance to the centroid of a population distribution | −‖φ(x) − μ‖, or the log-likelihood under the population's distribution | composite faces are rated more attractive than most components (Langlois & Roggman 1990); koinophilia as a selection argument (Koeslag 1990) | human effect solid; **animal mate-choice evidence to verify** |
| 4 | **Peak shift / supernormal stimuli** | response grows along a discriminant direction v beyond the training examples: score(x + Δv) > score(x) | the projection onto a learned discriminant direction | pigeon generalization gradients (Hanson 1959); herring gull chicks peck most at a red rod with white bands (Tinbergen & Perdeck 1950; Tinbergen 1951); Ramachandran & Hirstein (1999): caricature as art's first law | **solid,** a classic learning phenomenon across species |
| 5 | **Contrast and edges** | gradient energy Σ‖∇I‖² at object boundaries | the edge density / gradient-energy statistic | the visual system is edge-driven (V1 receptive fields); Ramachandran & Hirstein's "contrast" law | the mechanism is solid; the preference claim is weaker |
| 6 | **Grouping and binding (Gestalt)** | the clustering / graph structure that best explains the input (fewest parts) | the description length of the best grouping; segmentation likelihood | Gestalt laws in humans; perceptual grouping in many animals | solid as perception; the "pleasure in discovering a grouping" is Ramachandran's conjecture |
| 7 | **Order against complexity** | Birkhoff (1933): M = O / C. Information aesthetics (Bense, Moles, 1960s): redundancy against innovation. Berlyne (1971): an inverted-U hedonic curve over "arousal potential" (the Wundt curve) | an entropy or compressibility ratio; the distance from an optimal intermediate complexity | the inverted U is widely replicated in humans; animal curiosity shows preference for intermediate novelty | the inverted U is solid; Birkhoff's specific formula is not validated |
| 8 | **Compression progress** (Schmidhuber 1997–2010) | interestingness = the *reduction* in the observer's code length as its model improves: d/dt of the compression achieved | **our learning progress / the epiplexity rate:** the slow minus the fast EMA of prequential loss, or the signed area | proposed as the common root of curiosity, beauty and humour; formal, but empirical tests are few | a **theory;** see "unification" below |
| 9 | **The golden ratio** | φ ≈ 1.618 | — | rigorous studies find weak, inconsistent preference (e.g. McManus 1980 and later) | **not supported** as an innate law |
| 10 | **Curvature** (added 2026-10-20) | contour curvature; the absence of sharp angles | the mean absolute curvature along contours; the fraction of corners sharper than some angle | humans prefer curved over sharp-angled objects (Bar & Neta 2006); curvilinear rooms rated more beautiful (Vartanian et al. 2013, PNAS); **humans and great apes share the preference** in the same forced-choice task, stronger in humans (Munar, Gómez-Puerto, Call & Nadal 2015, PLoS ONE) | **solid,** with the strongest cross-species evidence in the catalogue so far |
| 11 | **Mystery** (added 2026-10-20) | the promise of more information if the viewer moves further in: a partly hidden view, a path that bends out of sight | **expected information gain** of moving: the same quantity as our exploration signal (BALD, Bayesian R3) | one of the four predictors of landscape preference with coherence, legibility and complexity (Kaplan & Kaplan 1989, *The Experience of Nature*) | the human landscape result is established; **animal evidence not checked** |
| 12 | **Colour by ecological valence** (added 2026-10-20) | a colour's preference = the average liking of the objects that have it | a learned average over the observer's object values; not a function of the pixel alone | blue is the most widely liked hue; ecological valence theory: clear sky, clean water (Palmer & Schloss 2010, PNAS) | a **learned** preference computed over innate valences: an example of "a quantity over the learned model" |

**Cross-cultural and infant checks on symmetry (added 2026-10-20):**
- Namibian Hai//om hunter-gatherers and German town dwellers both fixate longer on symmetric patterns, at either
  complexity. **Orangutans do not** separate symmetric from asymmetric patterns. In the Hai//om group the gaze
  preference did not match their aesthetic ratings (Mühlenbeck, Liebal, Pritsch & Jacobsen 2016, *Frontiers in
  Psychology* 7:408). So the evidence that symmetry is not cultural comes from **humans across cultures and from
  infants,** not from apes. Under the graded filter (method, step 1) that is still strong evidence.
- British and Egyptian students agree on preferring reflectional and rotational symmetry; Egyptians preferred
  **simpler** patterns more than Britons did (Bode, Helmy & Bertamini 2017, *Psihologija* 50(3)).
- 4-month-olds show no symmetry preference but encode vertical symmetry faster; 12-month-olds prefer vertical
  symmetry (Bornstein, Ferdinandsen & Gross 1981, *Developmental Psychology* 17).

### Sources (the user's collection; my verification notes)

- **Ramachandran & Hirstein (1999),** "The Science of Art: A Neurological Theory of Aesthetic Experience", *Journal of
  Consciousness Studies* 6(6–7). It proposed **eight** laws of artistic experience. Ramachandran's *The Tell-Tale
  Brain* (2011) lists nine. *To verify:* the user's source says nine for the 1999 paper.
- **Birkhoff,** *Aesthetic Measure* (Harvard University Press, 1933). ✓
- **Bense** (*Aesthetica*, 1950s–60s); **Moles,** *Information Theory and Esthetic Perception* (1958 / 1966). ✓
- **Schmidhuber,** "Low-complexity art" (1997); "Formal theory of creativity, fun, and intrinsic motivation
  (1990–2010)" (IEEE TAMD 2010). ✓
- **Voland & Grammer (eds.),** *Evolutionary Aesthetics* (Springer, 2003). ✓
- **Tinbergen,** *The Study of Instinct* (1951); Tinbergen & Perdeck (1950) on gull chicks. ✓
- **Palmer, Schloss & Sammartino (2013),** "Visual aesthetics and human preference", *Annual Review of Psychology*
  64. ✓
- **To add:**
  - Berlyne (1971), *Aesthetics and Psychobiology* (the inverted U);
  - Reber, Schwarz & Winkielman (2004), "Processing fluency and aesthetic pleasure" (beauty as fluent processing);
  - Field (1987) and Barlow (1961) (natural-image statistics, efficient coding);
  - A. R. Palmer (1999) (publication bias in fluctuating asymmetry).
- **Checked 2026-10-20 (web):** Munar et al. 2015 (PLoS ONE, doi 10.1371/journal.pone.0141106); Mühlenbeck et al.
  2016 (Front. Psychol. 7:408); Bode, Helmy & Bertamini 2017 (Psihologija 50(3):383–402); Bornstein, Ferdinandsen &
  Gross 1981 (Dev. Psychol. 17:82–86); Lu & Steinhardt 2007 (Science 315:1106) and the disputing comment; Grünbaum
  2006 (Notices AMS 53:670–673).
- **From memory, not yet re-checked:** Bar & Neta 2006; Vartanian et al. 2013; Kaplan & Kaplan 1989; Palmer & Schloss
  2010; Balling & Falk 1982; Appleton 1975; Zeki et al. 2014; Escher's Alhambra visits (1922, 1936).

## A unification: most of these are quantities over the observer's own predictive model

Most of the principles are not separate rules but **different quantities computed from one generative model** of the
world:

| principle | the quantity over a predictive model p(x) |
|---|---|
| fluency, 1/f, efficient coding | the input is **cheap to encode:** high log p(x) under a model tuned to natural statistics |
| averageness | the input is near the **prior mean** μ: high prior probability |
| symmetry, grouping, order | a **short description:** the model finds structure (an invariance, a grouping) that compresses the input |
| peak shift | the **gradient of a learned discriminant:** the direction that separates "good" from "bad" examples, extrapolated |
| compression progress, interestingness | the **rate** at which the model's code length for the input falls: learning progress |
| inverted U, order against complexity | the balance point: too predictable means no progress left; too random means no progress possible. That is exactly where learning progress peaks |
| mystery | the **expected information gain** of moving to see more: our exploration quantity, applied to a scene |

Consequences:
- **Why preferences are shared across species:** a shared environment produces shared natural statistics, and so
  similar models and similar quantities.
- **Why culture changes them:** culture changes the *learned* model (the prior, the discriminants). The quantity
  computed over it stays the same.
- **This is directly measurable in our agents.** Learning progress and epiplexity (E6, E2, the TBT line), prior means
  and spreads (E8's hierarchical Bayes), and prequential code length are all already computed.
- **It suggests separate nodes in a preference graph:** "beautiful" (fluent, compressible, interesting), "useful" (task
  value), "fair / harmless" (the value dimensions of `WORLD_DESIGN.md` §8). Each is a different functional of the same
  world model.

## A worked case: why Islamic architecture appeals across cultures (2026-10-20)

The user's question: why is Islamic architecture so appealing to someone who did not grow up in an Islamic culture?

**It is close to a natural experiment for the method.** Aniconism (avoiding images of living beings in religious
buildings) removed most of the figurative and narrative content, which is the part of art that carries
culture-specific meaning. The craft went instead into geometry, light, water and colour, which is mostly the
culture-free part of the catalogue. The principles are also stacked: one building scores high on several at once.

| feature (examples) | principle | evidence it is not cultural |
|---|---|---|
| star patterns with 6-, 8-, 10-, 12-fold rotation; bilateral façades doubled by a reflecting pool (Taj Mahal) | 1, symmetry | cross-cultural gaze and ratings; early in infancy; **not orangutans** (see above) |
| girih: a few tile shapes, strict local rules, a pattern that does not visibly repeat | 7 and 8, order with complexity, compression progress | the inverted U is human-wide. Lu & Steinhardt (2007, *Science* 315:1106) argued the 1453 Darb-i Imam shrine (Isfahan) is a nearly perfect quasi-crystalline (Penrose-type) pattern; **a later *Science* comment disputes it** (periodic, with quasi-crystal-like discs). Either way, over the area a viewer sees, it looks non-repeating but is generated by a short rule |
| muqarnas vaults, patterns nested inside patterns, carving at room, metre and centimetre scale | 2, multi-scale / fractal structure | the human mid-range fractal-D preference (Spehar et al. 2003). **Not measured for these buildings:** a prediction to test |
| domes, horseshoe and pointed arches, arabesque scrolls | 10, curvature | humans and great apes |
| courtyard fountains, four-part gardens (chahar bagh), lattice screens (mashrabiya) and muqarnas breaking light into dapples, turquoise and cobalt tile | nature cues; 12, colour | children prefer savanna and water scenes (Balling & Falk 1982); blue = sky and clean water (Palmer & Schloss 2010); prospect-refuge (Appleton 1975): a courtyard is enclosed (refuge) and open to the sky (prospect) |
| arcades, successive courtyards, screens, partly hidden views | 11, mystery | the Kaplans' landscape-preference predictor |

**The compression-progress reading of girih.** A plain periodic tiling is "solved" in about a second: the viewer's model
predicts it fully and progress stops. Random noise is never solved: no progress is possible. A girih pattern is
generated by a short rule but does not visibly repeat, so the viewer keeps discovering structure for minutes. It sits
at the peak of learning progress, which is the inverted U's peak. Birkhoff's ratio M = O / C cannot express this
(high order and high complexity at once); the rate quantity can.

**What culture still contributes, for an outsider:**
- **Exposure and selection:** most outsiders have seen curated photographs of the best buildings. Western art absorbed
  the patterns: M. C. Escher's tessellations came from his visits to the Alhambra in 1922 and 1936.
- **Exposure cannot be the main story:** Egyptians preferred simpler patterns *more* than Britons (Bode et al. 2017),
  so growing up among elaborate ornament does not raise the preference for complexity. (The comparison is confounded,
  but it cuts against the exposure account.)
- **Individual differences and expertise (a hypothesis):** a viewer whose own model contains the group structure
  (reflections, rotations, the 17 wallpaper groups) can compress these patterns further, so they offer that viewer more
  progress. Mathematicians report beauty for equations, and it activates the same medial orbitofrontal region as
  visual and musical beauty (Zeki, Romaya, Benincasa & Atiyah 2014, *Frontiers in Human Neuroscience*).
- A fact check: the popular claim that the Alhambra contains **all 17 wallpaper groups** is contested. Müller (1944)
  found 11 or 12, later counts 13, Pérez-Gómez and Montesinos the rest; Grünbaum (2006, *Notices of the AMS*
  53:670) challenged those findings and questioned whether counting groups is meaningful at all.

**For experiment 1 (scorers on synthetic stimuli):** a girih generator needs no download. Three stimulus classes,
periodic tiling / girih-like quasi-periodic / random noise, give one sharp prediction: a learner's compression progress
over a viewing session is highest for girih, small for periodic (fast saturation), near zero for noise. The symmetry
and description-length scores alone should rank periodic ≥ girih; only the rate should put girih first.

## A second worked case: why LLM writing annoys readers (2026-10-20)

The user's recollection: a study showing that LLM text has a statistical distribution very different from human text.
Three candidates, checked on the web:

1. **Holtzman, Buys, Du, Forbes & Choi (2019/2020), "The Curious Case of Neural Text Degeneration"** (ICLR 2020,
   arXiv 1904.09751), Figure 2: the probability a model assigns to each successive token. Human text **fluctuates**:
   mostly likely tokens with irregular surprising ones. Maximization decoding (beam search) is **flat and high**, and
   falls into repetition.
2. **Reinhart et al. (2025), "Do LLMs write like humans? Variation in grammatical and rhetorical styles"** (PNAS,
   doi 10.1073/pnas.2422455122; arXiv 2410.16107). Parallel human and model texts from the same prompts, scored on
   Biber's grammatical features. Rates relative to human text (from the paper's Tables 3–4):
   - **base Llama 3 models are close to human** (present participial clauses 94–102%, nominalizations 88–91%,
     agentless passives 98–101%);
   - **instruction-tuned models are far off:** present participial clauses 224–527% (GPT-4o 527%), nominalizations
     145–214%, "that" clauses as subject 173–331%; GPT-4o uses agentless passives at about half the human rate;
   - **words:** GPT-4o uses "camaraderie" 162×, "tapestry" 155×, "intricate" 119× as often as humans;
   - larger models were **not** closer to human style than smaller ones; a classifier on the features tells human from
     each model at 93–98% accuracy pairwise.
3. **Kobak, González-Márquez, Horvát & Lause (2024), "Delving into ChatGPT usage in academic writing through excess
   vocabulary"** (arXiv 2406.07016): an excess-word analysis of ~14 million PubMed abstracts; the excess words after
   2023 are style words ("delves"), not content words; a lower bound of ≥ 10% of 2024 abstracts LLM-processed (later
   versions revise it upward).

**The catalogue's reading: two different defects, plus habituation.**
- **Too predictable per token (the inverted U's boring side).** Human text carries information at irregular moments;
  mode-seeking text is uniformly probable, so the reader's model has nothing to update on. Low compression progress
  per word read. (Holtzman.)
- **Peak shift on style (principle 4).** The base model reproduces the human distribution; instruction tuning moves it
  2–5× along a few directions (dense nominal, academic register; "evocative" vocabulary). That is the profile of
  optimizing against a learned judgment of "good writing": a **supernormal stimulus for the reward model,** a
  caricature for the human reader. It is the doc's alignment-angle prediction, observed. (Reinhart.)
  - **Caveat:** the paper compares base with instruction-tuned models; it does not separate supervised fine-tuning
    from preference optimization, or rule out the chat format and system prompts as causes.
  - **The averageness bound, in RLHF terms:** the KL penalty in InstructGPT-style training is to the supervised
    fine-tuned policy, which is already shifted; a bound to the base model (the human distribution) would be the
    averageness counterweight. A testable design question, not a claim.
- **Habituation across texts.** One output can look polished; after hundreds, the reader's model has learned the
  template, so each new output brings no progress, and the template words become a cue for "no effort spent". (The
  last part is a costly-signalling reading, social rather than perceptual: a hypothesis.)

### Follow-up: what the human–LLM gap says about human learning (2026-10-20)

The user: "We can judge by how different human and LLM text distributions are that human learning cannot be
'pretraining or RL but somehow way more efficient' … why? And does it have anything to do with human *preferences*,
or human *learning*?"

**More evidence (checked on the web):**
- **Kirk et al. (ICLR 2024),** "Understanding the effects of RLHF on LLM generalisation and diversity": RLHF
  generalises better than supervised fine-tuning but **significantly reduces output diversity.**
- **Zhang et al. (ICML 2026; arXiv 2510.01171),** "Verbalized Sampling": a measured cause of mode collapse is
  **typicality bias in human preference data.** Raters prefer familiar, fluent, predictable text (they cite mere
  exposure, processing fluency (Reber et al. 2004, our catalogue's source), availability, schema congruity). The model:
  - reward r(x, y) = r_true(x, y) + α log π_ref(y | x) + ε(x);
  - the KL-regularised optimum is then **π*(y | x) ∝ π_ref(y | x)^γ · exp(r_true / β), with γ = 1 + α/β > 1:** a
    lowered temperature on the base model, which sharpens it onto its mode;
  - estimated on HelpSteer pairs with equal correctness: α̂ = 0.57 ± 0.07 (Llama 3.1 405B base as π_ref), 0.65 ± 0.07
    (GLM 4.5 base).
- **Russell, Karpinska & Iyyer (ACL 2025):** people who often use ChatGPT for writing are accurate detectors (a
  majority of five such annotators misclassified 1 of 300 articles), mainly from "AI vocabulary". Detection is
  **learned through exposure,** not immediate. (One author interned at a commercial detector company.)

**The answer.**
1. **The output distribution is evidence about the objective, not the learning algorithm.** Base models already match
   human statistics (Reinhart), because pretraining's objective is to match them. The gap is made by post-training's
   objective.
2. **The measured cause is a human preference:** fluency / typicality, the averageness principle. Optimising it
   turns into a power of the base distribution, which sharpens onto its mode.
3. **Humans have the same preference, but their text does not collapse.** Four candidate reasons:
   - **the preference is self-limiting in time:** familiarity first raises liking, then repeated exposure brings
     boredom (an inverted U over exposures; the size of the decline is from memory, to check). A one-shot pairwise
     rating measures only the rising side. A frozen reward model then optimises that side forever;
   - **a writer's objective is the reader's information gain** (Grice's maxim of quantity; the Rational Speech Act
     model, Frank & Goodman 2012, *Science*: speaker utility = how much the listener learns − production cost). The
     mode carries the fewest bits, so it scores near zero;
   - **many drives compete, and each satiates.** No single direction is maximised, so none is peak-shifted;
   - **people sample, they do not maximise** (probability matching): a weaker candidate.
4. **So: preferences, but preferences defined over learning.** Boredom is the end of compression progress;
   informativeness is the listener's learning. The refined inference: human learning is not "RL against a frozen
   scalar reward, done more efficiently". **RL against a reward computed from the learner's and the audience's own
   model states** (non-stationary, satiating) is not ruled out.

**Corrections to a second opinion the user collected** (an AI-written analysis; the outline agrees with Holtzman and
Reinhart):
- It is unsourced; "15–22 words per sentence" has no source.
- It credits the flatness mainly to decoding (top-p, low temperature). Reinhart's base models matched humans, and the
  shift came with instruction tuning: the policy itself is sharpened (the γ above).
- **Its uniform-information-density point is backwards.** That hypothesis says speakers *even out* information per
  word (e.g. dropping an optional "that" when the next clause is predictable; Jaeger 2010, from memory), so a flat
  rate would satisfy it. The real defect: the rate is uniformly *low*, padding that does not track the content.
  - The sharper statement is **local typicality** (Meister, Pimentel, Wiher & Cotterell, "Locally typical sampling",
    TACL 2023, from memory): human words carry information close to the *expected* information (the conditional
    entropy) in their context.
  - The mode carries less than that. In high dimensions the most probable sequence is *not* a typical sample: the
    typical-set result of information theory.
- "Our brains evolved to notice the difference immediately": the detection evidence says the noticing is learned
  (Russell et al.).

**A remaining tension, for the preference graph:** an averaged (composite) face is attractive, yet the "average"
(mode) text is boring. A candidate resolution is the **function of the stimulus.**
- A face is judged as a **quality signal:** near the prototype means healthy and typical, which is good.
- A text is judged as a **channel:** the mode transmits nothing.
- So the same quantity, closeness to the prior, should feed different preference nodes with opposite signs.

## The preference graph: what would make it more than one objective (a sketch, not designed)

The user: "I do not expect a model whose objective function does not include provisions for beauty to prefer
beautiful things except by coincidence. Which is why I proposed somehow developing a preference graph as opposed to a
singular objective function … Though I have no idea how that would work."

- **A fixed weighted sum of nodes is still one objective.** The Verbalized Sampling algebra shows what happens: any
  fixed term in the reward is amplified by optimisation (γ > 1). A graph must have properties a scalar lacks.
- **Candidate properties:**
  1. **Satiating, state-dependent nodes.** Each node has a set point; reward = the reduction of the distance to it
     ("homeostatic RL", Keramati & Gutkin 2014, *eLife*, from memory). Overshooting a set point is penalised, so a
     node cannot be peak-shifted. Peak shift survives only where a node is monotone (the sweet tooth: evolution never
     needed a set point for sugar).
  2. **Nodes computed over the learner's own model state:** fluency = log p(x) under the model, compression progress
     = the fall in code length, informativeness = a listener's information gain. They are non-stationary: the same
     input pays less once it is learned. **There is no fixed argmax to collapse onto.**
  3. **Different time scales:** fluency is immediate, usefulness is per episode, boredom builds over exposures. A
     one-shot rating protocol measures only the fast nodes. That is the RLHF failure above.
  4. **Constraints as well as trade-offs:** some edges are vetoes (harm), not exchange rates. Beauty matters among
     options that are useful and harmless.
  5. **Selection by competition, not summation:** the drive with the highest current urgency wins (the basal-ganglia
     gate of the TBT line), and satiation hands control to the next drive. That gives variety without a noise term.
- **An old precedent:** Tinbergen's hierarchical model of instinct (*The Study of Instinct*, 1951) and Lorenz's
  "psycho-hydraulic" drive model: drive centres with their own motivational state and releasing stimuli; drive
  builds and is discharged; conflicts produce displacement activities. A preference graph with satiation, drawn
  before computers.
- **Values in the weights (the user's standing requirement):** each node a value head of the one network (as OneNet
  had per-action reward and Q heads), drive levels as network input, the selection as gating. The mechint question
  (G6) is then concrete: is each node a locatable circuit, and does it read the world model's own prediction
  circuits?
- **Where experiment 1 fits:** it finds which quantities (the candidate node functions) reproduce the published
  directions. It does not test that a model prefers anything: that needs the node in the objective, as the user says.
- **A decisive toy (proposed, not designed):** a speaker network emits short sequences to a listener model that
  learns. Arm A rewards the speaker by a frozen rater with a typicality bias (α log p_ref). Arm B rewards it by the
  listener's information gain. Prediction: A collapses onto the mode (the γ sharpening), B keeps a human-like spread
  of surprisal. CPU scale.

## The alignment angle

- **Peak shift is biology's reward hacking.** Optimizing a discriminant pushes inputs beyond the natural distribution:
  supernormal stimuli, caricatures, junk food, engagement-optimized media. A model trained to maximize a beauty score
  will produce supernormal stimuli: adversarial examples of the score.
- **Averageness is the counterweight.** It prefers what is probable under the prior. The product "discriminant score ×
  prior plausibility" (in log terms: score + log prior) has the same structure as RLHF's objective, reward minus a KL
  penalty to the reference model. Evolution may have found the same regularizer.
  - **A testable question:** does a preference built as "peak shift bounded by averageness" resist hacking better than
    a plain learned reward model?
- **Unverifiable tasks.** Humans learn taste far beyond the innate primitives, through culture: the learned model
  changes, the primitives stay. A model given the primitives plus a learned world model might learn "taste" the same
  way. Whether that transfers to unverifiable tasks (art, writing, judgment) is an open research question.
- **The scope reminder:** reproducing innate preferences is not the same as choosing the right values (`GOALS.md`,
  "Scope of claims").

## Research questions (`GOALS.md`, G8) and first experiments (not designed in detail)

1. **Scorers.** Implement scores for principles 1–8 on synthetic stimuli (grids, images, sequences). Check that each
   reproduces the published *direction* of its effect: symmetric > asymmetric; mid-range fractal D preferred; composite
   > individual; peak-shift gradients; an inverted U.
   - Validating against human or animal rating datasets needs downloads, which need the user's approval first.
2. **Does an innate preference make learning faster or smarter?** Give the Village robot (the world family of E8) an
   intrinsic reward from the model-based quantities: compression progress, fluency. Measure how fast it learns the
   world's rules against no intrinsic reward and raw surprise. This connects to E6's exploration question.
3. **A preference graph in Village.** Separate nodes for beautiful, useful, provision, harm. Does a structured
   preference generalize better to held-out dilemmas than a single learned reward model (the RLHF stand-in)? Is
   "bounded peak shift" (score + log prior) harder to hack?
4. **Clues for alignment.** Do the preference nodes become locatable circuits in the network's weights (G6), and are
   they shared with the world model's own prediction circuits, as the unification above predicts?
