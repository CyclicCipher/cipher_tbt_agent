# Notes on "Neural Networks Are Already Clean" (Nick Cammarata, Goodfire, 2026-10-15)

**Source:** https://already-clean.goodfire.com/ (text version: `essay.md` on the same site), read in full on
2026-10-20.

**What it is:** an informal essay of about 65 minutes' reading, presented by the author as a *working hypothesis*.
- A companion technical paper (on language-model neurons, with Curt Tigges) is announced but not out.
- **No code or data has been released** (checked 2026-10-20).
- Everything below is in my own words. Numbers are as the essay reports them.

**The author:** Nick Cammarata. He co-authored *Zoom In*, *Curve Detectors* and *Multimodal Neurons*, was at OpenAI
from 2023, and is now at Goodfire. Thanks go to Curt Tigges (language-neuron work), Tarun Gogineni (editing), and
others including Chris Olah and Joshua Batson.

---

## 1. The thesis

1. **Most neurons are already interpretable.**
   - A network's own units (neurons, and circuits made of neurons) are legible building blocks.
   - One can study them directly ("normal science" on the network), rather than first "untangling" them into
     hypothesised features with sparse autoencoders (SAEs).
2. **The classic evidence for polysemanticity was misread.**
   - InceptionV1 neuron **4e:55** ("cat and car") is a **V-shape detector**. Its negative pole is an O shape.
   - Zoom In's "car hidden among dog neurons" superposition circuit was a **shared-pose** circuit: the car and the dogs
     are grouped because they share a pose, not stored together for lack of room. The author now calls his own 2019
     example an error.
3. **"Recursive generalization" replaces superposition as the main explanation of messy neurons.**
   - A neuron starts on a core pattern and sprouts offshoots along related axes, like branches of branches.
   - Example: an "iy" neuron → "diy" → "do it yourself", which no longer contains "iy".
   - It is the same learning principle as generalization, applied repeatedly (the author calls it "fractal").
   - It was earlier called "conjoined features" in *Multimodal Neurons*, renamed because "conjoined" suggests two
     things joined rather than a branching tree.
4. **Apparent polysemanticity has mostly ordinary, fixable causes.** The essay's five-way list, to be checked in this
   order:
   1. bad tools;
   2. bad explanations;
   3. bad learning (architecture, data, under-training);
   4. recursive generalization;
   5. superposition, *last*: its only symptom is "this doesn't make sense", while the others are testable.
5. **Highly speculative, in the author's own words:** interpretability of frontier models may be tractable much sooner
   than assumed.

## 2. Definitions the essay uses (and the definitional fight)

| term | the essay's meaning |
|---|---|
| **polysemantic neuron** | appears to have several unrelated jobs, or to be impossible to understand |
| **superposition (as used in the essay)** | neurons learning **unrelated** things together because those things rarely interfere: *non-local* packing |
| **superposition (Toy Models, 2022)** | representing **more features than dimensions**. The essay says this is a claim about feature counts, which cannot yet be defined or measured, and that it is satisfied even by monosemantic networks through **composition** (e.g. car-part neurons composing a car neuron) |
| **generalization** | grouping things by a shared pattern: *local* grouping |
| **recursive generalization** | generalization that branches repeatedly from a core |
| **pole** | each sign of a neuron's activation is a "pole". The positive pole admits inputs; the negative pole keeps them out, acting as a **gate**. Negative poles carry real, learned meaning (e.g. Santa lore lives on negative poles in Qwen) |
| **clean** | "parts built to be read and built upon by downstream readers", a "fractal of explanations" |
| **motif** | a recurring circuit pattern, e.g. the **summary motif** (commas storing a running summary), "say or just think" in the last layer, "say not" neurons, hedge writers |
| **feature** | called ambiguous. The author prefers specific, hard-to-vary words to the monosemantic/polysemantic frame |

**Key point for the debate:** the essay and the superposition literature use "superposition" for **different claims**.
- The essay attacks the **learning-principle** claim (arbitrary packing of unrelated features).
- It accepts **"more features than dimensions" via composition**, and calls it good news.

Much disagreement will be about which claim is meant. This bears on our goal of formal definitions (`GOALS.md`, G7).

## 3. Evidence, with the procedures as far as the essay states them

### 3a. Vision models

- **Models:**
  - InceptionV1 (2014, about 7M parameters; also retrained by the author, since the original checkpoints are lost);
  - **three ConvNeXts trained by the author on ImageNet: small 92k, medium 7.06M, large 28.6M parameters** (about 300×
    in size);
  - DINOv2 (a 10× larger model with about ten years of architectural change) still has generalized pose neurons.
- **InceptionV1's last layer, the fraction of neurons interpretable:**

  | condition | interpretable |
  |---|---|
  | 2023-era tools | about 5% (from the author's recollection) |
  | fixed tools | about 50% |
  | retrained with modern regularization | about 70% |
  | a ConvNeXt of the same size | about 99% |

  The essay's reading: architecture (residual connections, LayerNorm) and tools matter more than size.
- **Feature-visualization fixes:**
  - **Transformation robustness** (historical).
  - **Reward hacking** in deep layers: free optimization drifts to dog images.
  - **Path-anchored recipe:** take the neuron's top dataset examples, trace their paths, and reward the optimization for
    following those paths, often only for the first **64** steps.
  - Example: for car neuron 4c:447, two of its four readers show cars under path guidance, and two show a shared
    car–dog pose.
- **Crushing a layer, to see if pressure creates superposition:**
  - InceptionV1 layer **3b, 480 → 10 neurons**, with no bypass: **58.35% top-1, 81.14% top-5** ImageNet. The neurons
    stay interpretable but **coarser**: curves and loops, circles, eyes and faces, slanted or vertical edges, colours.
  - A **TopK SAE (640 features, 2 active per position)** on the crushed layer explained **82%** of variance. Splicing
    it in dropped top-1 **58.8 → 43.0%**.
  - The SAE found **no clean curve detectors**: median shape match 0.47–0.63, photo match ≤ 0.27. The same recipe on
    the original layer recovers clean curves (tuning r 0.96–1.00).
  - Medium ConvNeXt's **final layer 1,024 → 13** neurons, trained from scratch: the same broadening (neurons such as
    "Transit & small mammals", "Façades & dogs").
- **What dictionaries (SAEs) learn:**
  - No new building blocks were found.
  - **88%** of last-layer SAE activity is near-copies of single neurons.
  - Removing the dictionary directions that match no neuron leaves accuracy unchanged.
  - **A rotation search for the sparsest basis lands back on the neurons, in every block.**
  - What the SAE adds are loose groupings of neurons (e.g. feature 6717, "animals at the water's surface", built from
    neurons 733/170/920/482).
- **Weak activations:**
  - Medium ConvNeXt, all 18 blocks: removing the **weakest 75%** of activations moves accuracy **72 → 71%** (1 in 10
    answers changes, mostly near-ties).
  - Removing the **strongest 7%** drops it to **4%**.
  - InceptionV1's last layer: keeping the strongest quarter gives **69.47%** against 70.08%.
- **Memorization:**
  - Rare classes are consolidated into single neurons (one Triceratops neuron).
  - **Umbrella → bee:** a medium ConvNeXt trained to call one umbrella photo a bee changes its answer in **9 steps**.
    The changes are mostly in the last three blocks; transplanting those blocks makes the original model say "bee".
  - The essay's reading: memorization **attaches to existing routes**, not to isolated hidden features.
- **Adversarial examples go through normal semantic routes:** e.g. a triceratops skull read as a jaguar silences skull
  neurons and turns on jaguar inputs.
- **Watching neurons form** (checkpoints from epoch 0.7 to 12, InceptionV1 vs ConvNeXt on the same data):
  - neurons drift to **neighbouring** concepts, never to unrelated ones;
  - one apparent arbitrary mix was a zoom-setting error.
- **Residual stream:**
  - One huge coordinate (112) dominates the geometry. Removing it from the projection makes the maps readable.
  - Shrinking the stream 4× and 8× (down to 46 dims): accuracy **72 → 60 → 39%**; the maps stay similar, only messier.

### 3b. Language models

- **Qwen3-32B MLP neurons:**
  - **Preliminary audit:** about **90%** of neuron poles have one main job, about **7%** two related jobs, about
    **2–4%** not yet understood.
  - On strong activations only, about **97%** are non-arbitrary.
  - The audit used **1,000 randomly chosen "neuron papers"**, agent-written write-ups of a neuron.
  - The numbers depend on thresholds and on what counts as "one thing". The sampling and thresholds are **not
    specified** in the essay.
- **Earlier baselines:**
  - OpenAI's automated interpretability (2023, GPT-2 XL): about **1.7%** interpretable, by the most generous measure.
  - Towards Monosemanticity (2023): the median neuron scored **0/14** on its rubric.
  - Re-running 500 GPT-2 XL neuron papers: about **70%** "roughly one thing", about **5%** pristine.
  - The "polysemantic" neuron from *Towards Monosemanticity* is reread as a "**don't say a dot**" neuron: 8 of its 10
    strongest negative logit links begin with a dot.
- **Circuits built from neurons alone** (no transcoders):
  - **Medical diagnosis:** a **4-neuron hub**. Removing it makes the model ask about itching.
  - **Santa:** removing the **8** neurons with Santa on the *negative* pole drops P(Santa) to **0.059** (random
    removal: 0.997). Removing the 20 neurons *labelled* Christmas or Santa only drops it to 0.76.
  - **Lion King:** removing an 18-neuron step gives P(Scar) **0.24** (random: 0.91).
  - **Hedging:** removing 19 hedge writers takes P(hedge) **0.76 → 0.23**.
  - **"Say not":** removing 25 neurons gives P(Anna) **0.92** (random: 0.66).
  - **Kimi K3** (multi-trillion parameters), a wallet prompt answered "return": removing one hold-back neuron moves
    P(return) **0.83 → 0.98**.
  - Anthropic's cross-layer transcoder graphs predict the original model's next word only about half the time, and
    need error nodes.
- **Explanations that can lose** (the strongest evidence, in my view):
  - For **200 random neurons**, an agent read only each explanation and wrote **5 texts that should fire the neuron**
    and **5 near misses**.
  - Of the 183 neurons with positive-pole explanations, the texts fired the neuron strongly (≥ 30% of its corpus
    maximum) **76%** of the time, and the near misses **11%**.
  - The explanation won for **164**, tied for 15, and lost for 4.
  - **A packing test:** the agent invented the most distant concept it could. Only about **1%** of those texts fired
    the neuron strongly, and 179 of 183 neurons never fired.
- **"Blind maximum":** one agent writes an explanation; another, seeing only the explanation, writes text to maximize
  the neuron. The result often beats the strongest activation found in 10 billion tokens.
- **Double-negative SwiGLU:**
  - A negative gate × a negative value fires the neuron. For about **4%** of neurons this "echo" dominates the top of
    the activation list.
  - In the "enemy" neuron, 131 of 133 responses in the p75–p99 band are −×−; the strongest tail is +×+ (93/93 at
    p99.99).
  - Ablating the echo network-wide *raises* most benchmarks and lowers some reasoning ones.
- **Mixture of experts:** experts are more specialised than reported. The essay argues MoE is an "anti-superposition"
  architecture.
- **Attention:** a few thousand heads, against over a million neurons. Example: three layer-30 heads route pain → the
  preeclampsia neuron.
- **Memorization in language models:**
  - Removing a "Poe" neuron drops "dreary" from 87% to 50%.
  - Gating neurons above their ordinary-text p99.9 breaks recitation (e.g. of Psalm 23), while matched random
    interventions do not.

### 3c. Theory arguments

- **Circuit racing / gradient starvation:** related circuits capture the error first, so adding car features to an
  existing car area is faster than building a second one. This explains locality.
- **Toy models:** they price *interference* but not *reuse*.
  - The essay's correlated-input toy: 20 inputs in 4 firing groups, a 5-d bottleneck. Higher firing rates push it
    toward "PCA-like" solutions.
  - A hand-built packed code in InceptionV1 often failed to train, and when it worked the network relied on it less.
- **"Semantic gravity":** packing gets harder as circuits strengthen. Scale helps.
- **Confidence:** the author is about **85%** sure that nearly all structure comes from generative growth rather than
  arbitrary packing.
- **The essay's central risk claim:** the danger is deception "in plain sight that we can't read", not a missed hidden
  activation.
- **The alignment implication drawn:** if a system can be read, careful targeted changes (e.g. removing a toxic RL
  environment, editing a circuit) may replace blunt retraining.
- **A sketch of a solution:** a lab of automated scientists, guided by a "mechanistic constitution", proposing and
  testing explanations at many scales.

## 4. What a skeptic will say (fair-minded, to prepare for the debate)

1. **Informal and unreleased.** No code, data or companion paper yet. Many numbers are preliminary, threshold-dependent,
   or (for 2023 tools) from memory.
2. **Labelling relies on AI agents.** There is a risk of agreeable but unfalsifiable labels. The essay's answer is the
   "explanations that can lose" test, which is the key experiment to replicate.
3. **Selection effects.** The circuits shown are hand-picked (e.g. Kimi K3's twelve groups). The audit sample (1,000
   neuron papers) is random, but its labelling procedure is not specified.
4. **Definitions.** "Superposition = more features than dimensions" is not refuted. The essay narrows the term to
   "arbitrary packing of unrelated features". Both sides can be right about different claims.
5. **Neurons are clean, but is the computation?** Clean units do not automatically make *goal-directed* computation,
   such as deception, legible. That is the alignment question this folder exists to test.
6. **Vision first.** The strongest controlled evidence (crushing, SAE comparisons, rotation search, weak-activation
   ablation) is in vision models trained by the author; the language-model evidence is mostly case studies plus one
   audit.

## 5. What we can and cannot replicate here (4 GB GPU, runs ≤ 2 minutes on this machine)

Every claim below can be tested on **our own small models (1–10M parameters)**, trained in the test world of
`GOALS.md`:

| claim | cheap test |
|---|---|
| C1: most units have one main job per pole | label neurons by algorithm (the world's ground-truth variables are known) and count |
| C2: SAEs mostly rediscover neurons; a sparsest-rotation search lands on neurons | train an SAE, measure decoder–neuron alignment; run an orthogonal-rotation sparsity search from random starts |
| C3: a crushed layer gives coarser neurons, not packing | narrow vs wide versions of the same model |
| C4: weak activations matter little; strong ones matter a lot | percentile ablations |
| C5: neurons grow by recursive generalization, not arbitrary packing | track neurons across training checkpoints |
| C6: memorization attaches to existing routes | an umbrella→bee-style single-example edit; locate the change |
| C7: explanations can lose | write an explanation, then predict activations on new inputs and on near misses. In our world this can be checked **by algorithm** |
| C8: negative poles are meaningful gates | ablate by pole |
| C9: does superposition appear when theory says it must? | build a condition with many sparse, independent features and too few neurons (the Toy Models regime), and check whether the networks pack. This tests whether the essay's claim is general or regime-dependent |

**Not replicable here:**
- **ImageNet-scale ConvNeXts and InceptionV1** (training is a GPU-cluster job).
- **Qwen3-32B and Kimi K3 audits.**
- **Partial alternative:** an inference-only neuron audit on a small open language model (under 1B parameters) fits in
  4 GB. That would be a limited check of C1, C2, C4 and C7 on a real language model.
