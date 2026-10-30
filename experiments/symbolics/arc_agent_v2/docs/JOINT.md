# The joint experiment: arithmetic and MNIST in one brain (design, 2026-10-29)

**The user's terms** (2026-10-29):
- **The task:** "your next experiment will have to do both math and MNIST digit prediction with the same model. Mind
  you, it will be unlabeled, unsupervised learning."
- **The same model:** "the same brain instance with no domain flag anywhere, and regions differing only in their
  wiring."
  - **A region per sensor is fine:** the genome assigns regions of the neocortex to patches of sensory data.
- **Association regions:** "It is up to you to decide how to make associative or abstract regions that would combine
  multimodal information, and will probably require research."
- **The audit's replacements** (AUDIT.md §2.3) are made inside this design.

Nothing here is built. Part 12's pre-registration follows from §8.

**Settled in part 12's pre-registration** (`docs/LAB.md` part 12), where this note left them open:
- **The content model (§3.1):** per-bit counts with a prior of one average SDR, p_i = (c_i + w/d) / (N + 1), and an
  SDR weighed as one observation (log-likelihood × 1/w).
  - **Why the 1/w:** without it, a symbol's 40 bits would make every comparison 40 times too confident.
  - **What it replaces:** Laplace's rule, everywhere the brain weighs or predicts.
- **The Chinese-restaurant concentration (§3.3):** α = 1, the same strength as the content prior.
- **EM:** until no episode moves, at most 5 passes.
- **The k-cluster scores:** the k largest objects, with the brain's posterior restricted to them.
- **MAX_HYPS:** the start union holds every object, and is pruned only after the first input has weighed it.
- **The size:** 3,000 training and 1,000 test images per seed.

## 1. What the brain does (a short search)

1. **Convergence-divergence zones** ([Meyer & Damasio 2009, *Trends Neurosci.* 32:376](https://en.wikipedia.org/wiki/Convergence-divergence_zone);
   Damasio 1989).
   - **Convergence:** a zone receives from several sensory areas and projects back to them.
   - **What it holds:** binding codes, not the features themselves. It records which feature fragments were active
     together, and re-activates them in the sensory areas (divergence), for recall and recognition.
   - **The structure:** the zones form a hierarchy, multimodal at the top.
2. **Hub and spoke** ([Lambon Ralph et al. 2017, *Nat. Rev. Neurosci.* 18:42](https://springerlink.bibliotecabuap.elogim.com/10.1038/nrn.2016.150)).
   - **The hub:** concepts are a modality-independent hub in the anterior temporal lobes.
   - **The spokes:** the hub is joined to modality-specific areas.
   - **The evidence:** damage to the hub loses concepts in every modality at once (semantic dementia).
3. **Number becomes supramodal only high up** ([Nieder 2012, *PNAS*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3406836);
   [Nieder 2016, *Nat. Rev. Neurosci.*](https://homepages.uni-tuebingen.de/andreas.nieder/Nieder%20(2016)%20NRN.pdf)).
   - **The experiment:** monkeys judged the number of tones and the number of dots.
   - **Prefrontal cortex:** about 11% of the neurons coded the same number in both.
   - **Parietal VIP:** neurons coded one modality or the other, intermingled.
   - **So:** the abstract number forms where the modalities converge, not in either sensory stream.
4. **The Thousand Brains Project's Monty** ([arXiv 2507.04494](https://arxiv.org/html/2507.04494v1)).
   - **Sensor modules** send features at a pose, never an object's identity.
   - **Learning modules vote** by exchanging hypotheses (object, pose, evidence).
   - **Untested there:**
     - voting across modalities, "theoretically" possible ("we leave demonstrating this capability to future work");
     - hierarchy for compositional objects.
   - **2-D images** are read by moving a small patch over them. On Omniglot the maintainers report that it works but
     poorly, which they expect hierarchy to improve
     ([forum](https://thousandbrains.discourse.group/t/evaluating-montys-performance-on-the-omniglot-dataset/553)).
5. **Objects are learned from what follows in time** (Li & DiCarlo 2008, *Science* 321:1502).
   - **The finding:** IT neurons learn to treat as one object the views that follow each other in natural viewing,
     with no labels.
   - **The test:** swapping an object's identity across saccades changes their tolerance within hours.
6. **Same space, new environment** (global remapping: Leutgeb et al. 2005, *Science* 309:619; Fyhn et al. 2007,
   *Nature* 446:190).
   - **In a new environment,** hippocampal place cells take unrelated codes, while entorhinal grid cells keep their
     structure and only shift.
   - **So:** the location frame is shared, the environment is a separate variable, and the hippocampus conjoins the
     two.

**What this means for v2:**
- **A sensory region** learns features at locations in its own frame, one region per sensor (the user's genome
  point).
- **An object** is a hidden cause of the features at locations. It is learned from what occurs together within an
  episode (5), with the location frame shared across objects (6).
- **An association region** is a column like any other, whose inputs are other regions' outputs (1, 2).
  - Binding across modalities is co-occurrence in time, never a label.
  - The abstract (a number as such) forms where the streams converge (3).

## 2. The brain's shape

**Sensors** (the world transduces; the brain sees only SDRs):
- **`symbols`:** a random SDR per symbol, as now.
- **`retina`:** one 7 × 7 patch of an image, as an SDR.
  - **The transduction:** each pixel (on above 127, else off) contributes a fixed random sparse vector for its
    (position, on or off), and the top w bits of the sum are kept.
  - **What it gives:** similar patches give overlapping SDRs, and a blank patch always gives one fixed SDR.
  - **What it is:** a fixed random projection, the retina and optic nerve. It is not learned.
- **`gaze`:** an efference copy of the gaze's shift to the next patch.

**Regions** (`config`; wiring only):

| region | frame | feedforward | moves | for |
|---|---|---|---|---|
| `S` | learned | `symbols` | `symbols` | arithmetic, as now |
| `V` | innate:retinotopic-2d | `retina`, `gaze` | `gaze` | images: the 4 × 4 patch positions are given, as V1's retinotopy is (§0, rule 5; milestone B1) |
| `A` (part 13) | learned | `V`, `symbols` | both | the association region (§5) |

- **One brain, one code path.** S and V differ only in config:
  - which sensors they receive;
  - which of those move the place;
  - whether the frame is innate.

  The column, the hippocampus, consolidation and the thalamus don't know which region they serve.
- **The episode stream:** one stream, with arithmetic episodes and images interleaved at random.
  - **An image episode:** the patch at one fixed position (P1's start), then the other 15 in a random order, each
    after the gaze shift that brings it.
- **Why V's frame is innate, not learned from the gaze:**
  - biology gives V1's retinotopy (rule 5), and B1 is defined with it;
  - a frame learned from the gaze would be bounded (AUDIT §3, limit 2) and aliased (every blank patch looks alike).
    That is B2, a later step.

## 3. The general changes it needs

Each is justified by more than one world.

### 3.1 Content compared by overlap

- **Now:** a place's content is counts of exact inputs ({input: n}). An input never seen there has likelihood
  (0 + 1) / (N + K).
- **The change:**
  - **A cell's content** is per-bit counts: how often each of the d bits was active in what came next there.
  - **An input's likelihood** is the product over bits, each a Bernoulli with Laplace's rule.
  - **The prediction** is the w bits most probable under the mixture (the expected SDR), decoded by the world to the
    nearest symbol, as §6 A1 already says.
- **Why arithmetic should be unchanged:** two random SDRs share about 0.8 of 40 bits, so a symbol's likelihood at a
  place is high only where it was seen. This is a prediction, checked against the fingerprints' answers.
- **The worlds:**
  - MNIST: patches never repeat;
  - ARC: frames vary;
  - symbols: the identity limit.
- **Not changed yet:** P3 and P4 compare inputs by identity. Arithmetic and V's innate frame don't need more; a
  learned visual frame (B2) will.

### 3.2 The hippocampus conjoins what was sensed

- **Now:** the next place is DG(place, moving input). Sensed content doesn't enter it, so two images seen along the
  same gaze path share every place, and their episodes are mixed.
- **The change:** the next place is DG(place, moving input, the content sensed at the place since the last move).
  - **Why:** the situation is everything at that moment, and the situation is what DG separates (§1, 6).
- **Unchanged:**
  - arithmetic, whose symbols are both content and moving input already;
  - the walks' day 1, where a node's label is a function of the node.
- **The worlds:**
  - MNIST: each glance becomes its own episode;
  - relabelled walks: day 2's places become its own.

### 3.3 Objects as hidden causes (L2/3)

- **The idea:** what is at a location depends on which object is there.
  - The object is never seen. It is inferred from the content.
  - It is learned from what occurs together in one episode (§1, 5).
- **Learning, at sleep** (expectation maximization):
  - each replayed episode is assigned to the object that best predicts its content at its cells;
  - the cells are refit from the episodes assigned to them.
- **A new object is made on refutation:**
  - **When:** a fresh object (its prior: the region's average content) predicts an episode better than every known
    one. That is Bayes with a Chinese-restaurant prior.
  - **The rule:** "mint on refutation, never on incompleteness".
  - **Its identity:** each object gets a random SDR, which it publishes to the regions it feeds (§5).
- **Inference, awake:**
  - **The hypotheses:** the union's hypotheses become (object, phase, coordinate).
  - **The update:** each content input weighs them by Bayes, as now.
  - **The start:** an episode starts with every object, each weighted by how many episodes it explained.
- **What it predicts in each world:**
  - arithmetic: one object, since the cells are deterministic and no fresh object wins;
  - MNIST: objects, which are the clusters;
  - the relabelled walk: day 2 is a new environment on the same frame (§1, 6).
- **It replaces recency as an override** (AUDIT §2.3).
  - **Before a sleep,** the day's memory at the location is the content of one more object, "today's".
  - **On day 2,** the new labels refute the known object, so the weight moves to today's. That happens by evidence,
    not by rule.
  - **Part 10's best place goes:** each hypothesis reads the day's memory at its own place.
- **Not designed yet:**
  - the Chinese-restaurant concentration (one hyperparameter, to be set in the pre-registration);
  - how many EM passes;
  - whether objects can merge later.

### 3.4 One frame per receptive field

- **Now:** `brain.sleep` consolidates one frame for every column.
- **The change:** each region consolidates its own frame, from the hippocampus's replay of its own sensors' inputs.
- **An innate region skips frame learning:** its places and moves are given, and only its objects are learned.
- **The worlds:**
  - any brain with two sensory regions;
  - ARC, which has both space and the task.

## 4. The rest of AUDIT §2.3, and where each is exercised

| replaced | by | exercised by | part |
|---|---|---|---|
| recency as an override | today's memory as one more object (§3.3) | relabelled walks; MNIST | 12 |
| the thalamus's gate contexts (two inputs) | the thought's own state (its phase) and the input | precedence | 14 |
| the thalamus's operations (restart, re-feed, feed the name) | a subroutine call: suspend a state, pursue another, resume with its output | precedence; a second world, not chosen yet | 14 |
| the relation x ↦ k·x | a group map between shift classes over the whole group, chosen by MDL (the user's "analogical mapping as a native ability") | the reversal; a walk whose controls are mirrored in one context (new) | 15 |

## 5. The association region (built in part 13)

- **What it is:** a region like any other, the same column. Its feedforward sources include other regions, not only
  sensors.
- **What it receives (convergence):**
  - **from a region:** that region's output when its episode ends, the SDR of its most probable object;
  - **from a sensor wired to it:** the input, as usual.
- **What it reads:** a sequence. For example, an equation written with handwritten digits and heard operators arrives
  as [V: the object seen] [symbols: +] [V: object] [symbols: =] [V: object].
- **What it does:** what S does with symbols.
  - To it, V's objects are symbols, so A1's and A2's machinery applies unchanged.
  - Several objects for one digit (two styles of 7) are synonyms, which P3 and P4 merge by their futures.
- **What it returns (divergence):** its prediction of the next input. It goes back to the region the input comes from,
  as the prior over that region's objects at its next episode's start. The answer's image is then read expecting the
  sum.
- **Binding by co-occurrence only:** nothing pairs an image with a symbol. The only link is that both occur in one
  equation. A human learning written digits gets the same data: digits used in sums, with no one naming each image.
- **The abstract:** the number 7 as such is then the association region's coordinate, reached alike from the heard
  symbol and from the seen digit. That is §1's supramodal code (3).
- **Its test world (part 13):** equations whose digits are MNIST images (chosen by the world's label, which the brain
  never sees) and whose operators are symbols.
- **Its scores:**
  - the answer's object, mapped to its majority label;
  - with feedback, against without.

## 6. The data, against a human

- **MNIST:**
  - unlabeled training images, all ten digits, many writers;
  - each shown as its 16 patches at their positions, which is all of the image, as a human sees it.
- **Arithmetic:** as before: A1, A2 and the reversal at p = 13 and 97, and precedence.
- **The human comparison:**
  - **for arithmetic:** as before;
  - **for MNIST:** a person sorting unseen digits by shape, with no names. I know of no measured number for someone
    who has never seen digits. The pre-registration says so, rather than inventing one.

## 7. Measures and cost

- **MNIST** (§6 B1, labels only to score):
  - 100 clusters, each labelled by its training majority: ≥ 0.90;
  - 10 clusters, matched to the labels one-to-one: ≥ 0.60;
  - 1-nearest-neighbour on the objects' posteriors.
  - **The 10-cluster score needs a rule** when the brain's object count isn't 10. The pre-registration sets it, outside
    the brain.
- **Arithmetic:** part 6's and part 10's results reproduce, with the brain also learning images.
- **Non-interference:**
  - the arithmetic results with and without images in the stream;
  - the clusters with and without arithmetic.
- **Cost:** under 2 minutes, with the regression.
  - **The candidate size:** 5,000 training images and 1,000 test images.
  - **One resource bound bites:** MAX_HYPS = 64 caps a union that starts with every object. The pre-registration
    decides it.

## 8. The order

1. **Part 12, the joint experiment:** §3.1–3.4, the V region, the MNIST world and the joint stream.
2. **Part 13:** the association region and the multimodal equations (§5).
3. **Part 14:** the thalamus's replacements.
4. **Part 15:** the relation's general form.
5. **Then:**
   - B2, the learned visual frame (bounded frames, and overlap in P3 and P4);
   - A3, on the data a human gets.
